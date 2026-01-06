#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <cstdint>
#include <string>

struct Renderer {
    GLFWwindow* window = nullptr;
    GLuint      prog = 0;
    GLuint      vao = 0;
    GLuint      tex = 0;

    bool init(uint32_t w, uint32_t h);
    void updateTexture(uint32_t w, uint32_t h, const std::vector<unsigned char>& rgba);
    void draw();
    void setTitle(const std::string& s);
    void shutdown();
};
