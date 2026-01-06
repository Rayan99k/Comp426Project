#pragma once
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <vector>
#include <cstdint>

struct CLColorizer {
    cl_context       context = nullptr;
    cl_device_id     device = nullptr;
    cl_command_queue queue = nullptr;
    cl_program       program = nullptr;
    cl_kernel        kernel = nullptr;
    cl_mem           bufGrid = nullptr;
    cl_mem           bufImage = nullptr;
    uint32_t         N = 0;

    bool init(uint32_t w, uint32_t h, const char* src);
    void colorize(const std::vector<unsigned char>& species,
        std::vector<unsigned char>& rgba);
    void shutdown();
};
