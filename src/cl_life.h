#pragma once
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <vector>
#include <cstdint>

struct CLLife {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kAB = nullptr;
    cl_kernel kBA = nullptr;
    cl_mem bufA = nullptr;
    cl_mem bufB = nullptr;
    cl_kernel pipeProducer = nullptr;
    cl_kernel pipeConsumer = nullptr;
    cl_mem    statsPipe = nullptr;
    cl_mem    statsBuffer = nullptr;
    double lastKernelMs = 0.0;
    
    bool flip = false;

    size_t workItems = 0;
    size_t localSize = 0;
    size_t    pipeWorkItems = 64;
    uint32_t  lastLiveCells = 0;
    cl_uint computeUnits = 0;

    bool init(uint32_t w, uint32_t h, uint32_t numSpecies, const char* src);
    void step(uint32_t w, uint32_t h, uint32_t numSpecies, std::vector<unsigned char>& host);
    void set_work_items(size_t n) {
        workItems = n;
    }
    void set_local_size(size_t n) {
        localSize = n;
    }
    void shutdown();
};
