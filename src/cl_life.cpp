#include "cl_life.h"
#include "kernel_source.h"

#include <vector>
#include <iostream>
#include <cstring>

#define CHECK_CL(err, msg) \
    if ((err) != CL_SUCCESS) { \
        std::cerr << msg << " (err = " << (err) << ")\n"; \
        return false; \
    }

bool CLLife::init(uint32_t w, uint32_t h,
    uint32_t numSpecies,
    const char* src)
{
    (void)numSpecies;

    cl_int err = CL_SUCCESS;

    cl_platform_id platform = nullptr;
    cl_device_id   device = nullptr;

    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    CHECK_CL(err, "clGetPlatformIDs failed");
    if (numPlatforms == 0) {
        std::cerr << "No OpenCL platforms found\n";
        return false;
    }

    std::vector<cl_platform_id> plats(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, plats.data(), nullptr);
    CHECK_CL(err, "clGetPlatformIDs(list) failed");

    for (cl_uint i = 0; i < numPlatforms && !device; ++i) {
        cl_uint numDevices = 0;
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices)
            != CL_SUCCESS || numDevices == 0)
            continue;

        std::vector<cl_device_id> devs(numDevices);
        clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, numDevices, devs.data(), nullptr);

        platform = plats[i];
        device = devs[0];
    }

    if (!device) {
        std::cerr << "No OpenCL GPU device found\n";
        return false;
    }

    cl_uint cu = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(cu), &cu, nullptr);
    if (err == CL_SUCCESS) {
        computeUnits = cu;
    }

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_CL(err, "clCreateContext failed");

    const cl_queue_properties qprops[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    queue = clCreateCommandQueueWithProperties(context, device, qprops, &err);
    CHECK_CL(err, "clCreateCommandQueueWithProperties failed");

    const size_t N = static_cast<size_t>(w) * static_cast<size_t>(h);
    const size_t bytes = N * sizeof(cl_uchar);

    bufA = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    CHECK_CL(err, "Failed to create bufA");

    bufB = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    CHECK_CL(err, "Failed to create bufB");

    const char* srcs[] = { src };
    size_t      lens[] = { std::strlen(src) };

    program = clCreateProgramWithSource(context, 1, srcs, lens, &err);
    CHECK_CL(err, "clCreateProgramWithSource failed");

    const char* buildOpts = "-cl-std=CL2.0";
    err = clBuildProgram(program, 1, &device, buildOpts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            logSize, log.data(), nullptr);
        std::cerr << "CL build error:\n" << log.data() << "\n";
        return false;
    }


    kAB = clCreateKernel(program, "life_step", &err);
    CHECK_CL(err, "Failed to create kernel kAB");

    kBA = clCreateKernel(program, "life_step", &err);
    CHECK_CL(err, "Failed to create kernel kBA");

    cl_int err2 = CL_SUCCESS;

    pipeProducer = clCreateKernel(program, "pipe_producer", &err2);
    if (!pipeProducer || err2 != CL_SUCCESS) {
        std::cerr << "Warning: pipe_producer kernel not available (err="
            << err2 << ")\n";
        pipeProducer = nullptr;
    }

    pipeConsumer = clCreateKernel(program, "pipe_consumer", &err2);
    if (!pipeConsumer || err2 != CL_SUCCESS) {
        std::cerr << "Warning: pipe_consumer kernel not available (err="
            << err2 << ")\n";
        pipeConsumer = nullptr;
    }

    if (pipeProducer && pipeConsumer) {
        cl_uint pipeCapacity = static_cast<cl_uint>(pipeWorkItems);

        statsPipe = clCreatePipe(context,
            CL_MEM_READ_WRITE,
            sizeof(cl_uint),
            pipeCapacity,
            nullptr,
            &err2);
        if (!statsPipe || err2 != CL_SUCCESS) {
            std::cerr << "Warning: failed to create statsPipe (err="
                << err2 << ")\n";
            statsPipe = nullptr;
        }

        statsBuffer = clCreateBuffer(context,
            CL_MEM_WRITE_ONLY,
            pipeCapacity * sizeof(cl_uint),
            nullptr,
            &err2);
        if (!statsBuffer || err2 != CL_SUCCESS) {
            std::cerr << "Warning: failed to create statsBuffer (err="
                << err2 << ")\n";
            statsBuffer = nullptr;
        }
    }

    flip = false;
    workItems = 0;
    localSize = 64;
    lastKernelMs = 0.0;
    lastLiveCells = 0;

    return true;
}

void CLLife::step(uint32_t w, uint32_t h,
    uint32_t numSpecies,
    std::vector<unsigned char>& host)
{
    const uint32_t S = numSpecies;
    const uint32_t W = w;
    const uint32_t H = h;
    const uint32_t N = W * H;

    size_t global = workItems ? workItems : static_cast<size_t>(N);

    static bool seeded = false;
    if (!seeded) {
        if (host.size() >= N) {
            cl_int errSeed = clEnqueueWriteBuffer(
                queue, bufA, CL_TRUE, 0,
                N * sizeof(cl_uchar),
                host.data(), 0, nullptr, nullptr);
            if (errSeed != CL_SUCCESS) {
                std::cerr << "Initial seed write failed (err="
                    << errSeed << ")\n";
            }
        }
        else {
            std::cerr << "Warning: host grid too small to seed device\n";
        }
        seeded = true;
    }

    cl_event evtKernel = nullptr;

    if (!flip) {
        clSetKernelArg(kAB, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kAB, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kAB, 2, sizeof(cl_uint), &W);
        clSetKernelArg(kAB, 3, sizeof(cl_uint), &H);
        clSetKernelArg(kAB, 4, sizeof(cl_uint), &S);

        clEnqueueNDRangeKernel(queue, kAB, 1, nullptr,
            &global, (localSize ? &localSize : nullptr),
            0, nullptr, &evtKernel);
    }
    else {
        clSetKernelArg(kBA, 0, sizeof(cl_mem), &bufB);
        clSetKernelArg(kBA, 1, sizeof(cl_mem), &bufA);
        clSetKernelArg(kBA, 2, sizeof(cl_uint), &W);
        clSetKernelArg(kBA, 3, sizeof(cl_uint), &H);
        clSetKernelArg(kBA, 4, sizeof(cl_uint), &S);

        clEnqueueNDRangeKernel(queue, kBA, 1, nullptr,
            &global, (localSize ? &localSize : nullptr),
            0, nullptr, &evtKernel);
    }
    
    clWaitForEvents(1, &evtKernel);
    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(evtKernel, CL_PROFILING_COMMAND_START,
        sizeof(t0), &t0, nullptr);
    clGetEventProfilingInfo(evtKernel, CL_PROFILING_COMMAND_END,
        sizeof(t1), &t1, nullptr);
    clReleaseEvent(evtKernel);

    lastKernelMs = static_cast<double>(t1 - t0) * 1e-6;

    if (pipeProducer && pipeConsumer && statsPipe && statsBuffer) {
        cl_int err = CL_SUCCESS;

        cl_mem curGrid = (!flip ? bufB : bufA);
        size_t pipeGlobal = pipeWorkItems;

        err = clSetKernelArg(pipeProducer, 0, sizeof(cl_mem), &curGrid);
        err |= clSetKernelArg(pipeProducer, 1, sizeof(cl_uint), &N);
        err |= clSetKernelArg(pipeProducer, 2, sizeof(cl_mem), &statsPipe);

        if (err == CL_SUCCESS) {
            cl_event evtProd = nullptr;
            err = clEnqueueNDRangeKernel(queue, pipeProducer, 1, nullptr,
                &pipeGlobal, nullptr,
                0, nullptr, &evtProd);
            if (err == CL_SUCCESS) {
                clWaitForEvents(1, &evtProd);
                clReleaseEvent(evtProd);
            }
            else {
                std::cerr << "pipeProducer enqueue failed (err=" << err << ")\n";
            }
        }
        else {
            std::cerr << "pipeProducer set args failed (err=" << err << ")\n";
        }

        if (err == CL_SUCCESS) {
            cl_uint numItems = static_cast<cl_uint>(pipeWorkItems);

            err = clSetKernelArg(pipeConsumer, 0, sizeof(cl_mem), &statsPipe);
            err |= clSetKernelArg(pipeConsumer, 1, sizeof(cl_mem), &statsBuffer);
            err |= clSetKernelArg(pipeConsumer, 2, sizeof(cl_uint), &numItems);

            if (err == CL_SUCCESS) {
                cl_event evtCons = nullptr;
                err = clEnqueueNDRangeKernel(queue, pipeConsumer, 1, nullptr,
                    &pipeGlobal, nullptr,
                    0, nullptr, &evtCons);
                if (err == CL_SUCCESS) {
                    clWaitForEvents(1, &evtCons);
                    clReleaseEvent(evtCons);

                    std::vector<cl_uint> partial(pipeWorkItems);
                    err = clEnqueueReadBuffer(queue, statsBuffer, CL_TRUE,
                        0, pipeWorkItems * sizeof(cl_uint),
                        partial.data(),
                        0, nullptr, nullptr);
                    if (err == CL_SUCCESS) {
                        uint32_t total = 0;
                        for (size_t i = 0; i < pipeWorkItems; ++i)
                            total += partial[i];
                        lastLiveCells = total;
                    }
                    else {
                        std::cerr << "Read statsBuffer failed (err=" << err << ")\n";
                    }
                }
                else {
                    std::cerr << "pipeConsumer enqueue failed (err=" << err << ")\n";
                }
            }
            else {
                std::cerr << "pipeConsumer set args failed (err=" << err << ")\n";
            }
        }
    }

    host.resize(N);

    if (!flip) {
        clEnqueueReadBuffer(queue, bufB, CL_TRUE, 0,
            N * sizeof(cl_uchar),
            host.data(),
            0, nullptr, nullptr);
    }
    else {
        clEnqueueReadBuffer(queue, bufA, CL_TRUE, 0,
            N * sizeof(cl_uchar),
            host.data(),
            0, nullptr, nullptr);
    }

    flip = !flip;
}

void CLLife::shutdown()
{
    if (statsBuffer)  clReleaseMemObject(statsBuffer);
    if (statsPipe)    clReleaseMemObject(statsPipe);
    if (pipeConsumer) clReleaseKernel(pipeConsumer);
    if (pipeProducer) clReleaseKernel(pipeProducer);

    if (kBA)      clReleaseKernel(kBA);
    if (kAB)      clReleaseKernel(kAB);
    if (program)  clReleaseProgram(program);
    if (bufB)     clReleaseMemObject(bufB);
    if (bufA)     clReleaseMemObject(bufA);
    if (queue)    clReleaseCommandQueue(queue);
    if (context)  clReleaseContext(context);

    statsBuffer = nullptr;
    statsPipe = nullptr;
    pipeConsumer = nullptr;
    pipeProducer = nullptr;
    kBA = nullptr;
    kAB = nullptr;
    program = nullptr;
    bufB = nullptr;
    bufA = nullptr;
    queue = nullptr;
    context = nullptr;
}
