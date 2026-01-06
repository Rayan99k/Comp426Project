#pragma once

static const char* COLOR_KERNEL_SRC = R"CLC(
typedef unsigned char uchar;

inline uchar4 species_to_color(uchar s)
{
    if (s == 0) {
        return (uchar4)(0, 0, 0, 255);
    }

    switch (s) {
        case 1:  return (uchar4)(255,   0,   0, 255);
        case 2:  return (uchar4)(  0, 255,   0, 255);
        case 3:  return (uchar4)(  0,   0, 255, 255);
        case 4:  return (uchar4)(255, 255,   0, 255);
        case 5:  return (uchar4)(255,   0, 255, 255);
        case 6:  return (uchar4)(  0, 255, 255, 255);
        case 7:  return (uchar4)(255, 128,   0, 255);
        case 8:  return (uchar4)(128,   0, 255, 255);
        case 9:  return (uchar4)(  0, 128, 255, 255);
        case 10: return (uchar4)(255, 255, 255, 255);
        default:
            return (uchar4)(200, 200, 200, 255);
    }
}

__kernel void colorize_grid(__global const uchar* grid,
                            __global uchar4*      image,
                            const uint            N)
{
    uint gid = get_global_id(0);
    if (gid >= N) return;

    uchar s      = grid[gid];
    uchar4 color = species_to_color(s);

    image[gid] = color;
}
)CLC";
