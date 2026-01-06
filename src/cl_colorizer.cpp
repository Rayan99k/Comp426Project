// cl_colorizer.cpp
#include "cl_colorizer.h"
#include <cstring>
#include <iostream>

bool CLColorizer::init(uint32_t w, uint32_t h, const char* src)
{
    N = w * h;

    cl_int err = CL_SUCCESS;

    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "No OpenCL platforms found\n";
        return false;
    }

    std::vector<cl_platform_id> plats(numPlatforms);
    clGetPlatformIDs(numPlatforms, plats.data(), nullptr);

    device = nullptr;
    for (cl_uint i = 0; i < numPlatforms && !device; ++i) {
        cl_uint numDevices = 0;
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_CPU, 0, nullptr, &numDevices)
            != CL_SUCCESS || numDevices == 0)
            continue;

        std::vector<cl_device_id> devs(numDevices);
        clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_CPU, numDevices, devs.data(), nullptr);
        device = devs[0];
    }

    if (!device) {
        std::cerr << "No OpenCL CPU device found\n";
        return false;
    }

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!context || err != CL_SUCCESS) {
        std::cerr << "Failed to create CPU context\n";
        return false;
    }

    const cl_queue_properties qprops[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    queue = clCreateCommandQueueWithProperties(context, device, qprops, &err);
    if (!queue || err != CL_SUCCESS) {
        std::cerr << "Failed to create CPU queue\n";
        return false;
    }

    const char* srcs[] = { src };
    size_t lens[] = { std::strlen(src) };
    program = clCreateProgramWithSource(context, 1, srcs, lens, &err);
    if (!program || err != CL_SUCCESS) {
        std::cerr << "Failed to create CPU program\n";
        return false;
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            logSize, log.data(), nullptr);
        std::cerr << "CPU program build log:\n" << log.data() << "\n";
        return false;
    }

    kernel = clCreateKernel(program, "colorize_grid", &err);
    if (!kernel || err != CL_SUCCESS) {
        std::cerr << "Failed to create CPU kernel\n";
        return false;
    }

    bufGrid = clCreateBuffer(context, CL_MEM_READ_ONLY,
        N * sizeof(cl_uchar), nullptr, &err);
    bufImage = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        N * 4 * sizeof(cl_uchar), nullptr, &err);
    if (!bufGrid || !bufImage || err != CL_SUCCESS) {
        std::cerr << "Failed to create CPU buffers\n";
        return false;
    }

    return true;
}

void CLColorizer::colorize(const std::vector<unsigned char>& species,
    std::vector<unsigned char>& rgba)
{
    if (species.size() < N) return;

    cl_int err = CL_SUCCESS;

    err = clEnqueueWriteBuffer(queue, bufGrid, CL_TRUE,
        0, N * sizeof(cl_uchar),
        species.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "CPU colorizer: write buffer failed\n";
        return;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufGrid);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufImage);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &N);
    if (err != CL_SUCCESS) {
        std::cerr << "CPU colorizer: set args failed\n";
        return;
    }

    size_t global = N;
    cl_event evt = nullptr;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
        &global, nullptr, 0, nullptr, &evt);
    if (err != CL_SUCCESS) {
        std::cerr << "CPU colorizer: enqueue kernel failed\n";
        return;
    }

    clWaitForEvents(1, &evt);
    clReleaseEvent(evt);

    rgba.resize(N * 4);
    err = clEnqueueReadBuffer(queue, bufImage, CL_TRUE,
        0, N * 4 * sizeof(cl_uchar),
        rgba.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "CPU colorizer: read buffer failed\n";
        return;
    }
}

void CLColorizer::shutdown()
{
    if (bufImage) clReleaseMemObject(bufImage);
    if (bufGrid)  clReleaseMemObject(bufGrid);
    if (kernel)   clReleaseKernel(kernel);
    if (program)  clReleaseProgram(program);
    if (queue)    clReleaseCommandQueue(queue);
    if (context)  clReleaseContext(context);

    bufImage = bufGrid = nullptr;
    kernel = nullptr;
    program = nullptr;
    queue = nullptr;
    context = nullptr;
}
