#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "cl_life.h"
#include "kernel_source.h"
#include "renderer.h"
#include "cl_colorizer.h"
#include "cpu_color_kernel.h"

static const uint32_t GRID_W = 1024;
static const uint32_t GRID_H = 768;
static const uint32_t GRID_N = GRID_W * GRID_H;

static uint32_t choose_species_count()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(10, 10);
    return dist(gen);
}

static void init_species_grid(std::vector<unsigned char>& grid,
    uint32_t numSpecies)
{
    grid.resize(GRID_N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> sp(1, (int)numSpecies);

    for (uint32_t i = 0; i < GRID_N; ++i) {
        grid[i] = static_cast<unsigned char>(sp(gen));
    }
}

static void glfw_error_callback(int error, const char* desc)
{
    std::cerr << "GLFW error " << error << ": " << desc << "\n";
}

int main()
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        1024, 768, "Multi-Species Game of Life (OpenCL Project)", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); //vsync

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    Renderer renderer;
    if (!renderer.init(GRID_W, GRID_H)) {
        std::cerr << "Renderer init failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    uint32_t numSpecies = choose_species_count();
    std::vector<unsigned char> speciesGrid;
    init_species_grid(speciesGrid, numSpecies);

    CLLife life;
    if (!life.init(GRID_W, GRID_H, numSpecies, LIFE_KERNEL_SRC)) {
        std::cerr << "Failed to init OpenCL (GPU life)\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    life.set_work_items(0);
    life.set_local_size(0);


    CLColorizer colorizer;
    if (!colorizer.init(GRID_W, GRID_H, COLOR_KERNEL_SRC)) {
        std::cerr << "Failed to init CPU OpenCL colorizer\n";
        life.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    std::vector<unsigned char> rgba;

    auto tLast = std::chrono::high_resolution_clock::now();
    double fpsAccum = 0.0;
    int    fpsFrames = 0;
    double fps = 0.0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        life.step(GRID_W, GRID_H, numSpecies, speciesGrid);

        colorizer.colorize(speciesGrid, rgba);

        renderer.updateTexture(GRID_W, GRID_H, rgba);
        renderer.draw();

        glfwSwapBuffers(window);

        auto tNow = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(tNow - tLast).count();
        tLast = tNow;

        fpsAccum += dt;
        fpsFrames += 1;
        if (fpsAccum >= 0.5) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.0;
            fpsFrames = 0;
        }

        size_t global = life.workItems ? life.workItems : (size_t)GRID_N;
        size_t local = life.localSize ? life.localSize : 0;

        char title[256];
        std::snprintf(title, sizeof(title),
            "GoL | FPS: %.1f | Species: %u | GPU CUs: %u | Global: %zu | Local: %zu | Kernel: %.3f ms",
            fps, numSpecies, life.computeUnits,
            global, local, life.lastKernelMs);
        glfwSetWindowTitle(window, title);
    }

    colorizer.shutdown();
    life.shutdown();
    renderer.shutdown();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}