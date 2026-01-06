# Game of Life — Heterogeneous OpenCL (CPU + GPU) Pipeline

A high-performance implementation of **Conway’s Game of Life** using **OpenCL**, designed to demonstrate **heterogeneous computing** across **CPU and GPU** while minimizing host/device transfers. The program uses **double buffering** and a simple **pipeline** so it can **render the current generation while computing the next**.

> Course-style project focused on parallelism, correctness, and performance measurement in a heterogeneous environment.

---

## Features

- **OpenCL kernel(s)** for Game of Life update rule
- **CPU + GPU workload split** (heterogeneous execution)
- **Double buffering (ping-pong buffers)** to avoid read/write hazards
- **Pipelined execution**
  - display `generation N`
  - compute `generation N+1` in parallel
- **Configurable grid size** and simulation parameters
- **Performance metrics**
  - kernel execution timing
  - throughput (iterations/sec), useful when FPS is capped by vsync

---

## How It Works

### Update Rule (Conway’s Game of Life)
Each cell is updated based on its 8 neighbors:

- Any live cell with **2 or 3** live neighbors survives
- Any dead cell with **exactly 3** live neighbors becomes alive
- All other live cells die in the next generation

### Double Buffering
Two device buffers are maintained:

- `bufA` — current generation (read-only for kernel)
- `bufB` — next generation (write-only for kernel)

Each step swaps them:

(bufA, bufB) = (bufB, bufA)

### Pipeline
The host program schedules work so rendering and computing overlap when possible:

- While the screen displays `bufA`, the compute queue generates `bufB`
- Synchronization ensures correctness before presenting the next frame

---

## Project Structure
````
.
└── bin/
|    └── Comp426Project.exe
|    └── Comp426Project.pdb
└── dependencies/
├── src/
|    ├── cl_colorizer.cpp
|    ├── cl_colorizer.h
|    ├── cl_life.cpp
|    ├── cl_life.h
|    ├── config.h
|    ├── cpu_color_kernel.h
|    ├── glad.c
|    ├── kernel_source.h
|    ├── main.cpp
|    ├── renderer.cpp
|    └── renderer.h
└── README.md
└── temp/

````
---

## Requirements

- OpenCL runtime (CPU and/or GPU):
  - NVIDIA / AMD GPU driver with OpenCL
  - or Intel CPU OpenCL runtime
- C/C++ compiler (MSVC, GCC, or Clang)
- (Optional) A windowing library (GLFW/SDL/etc.) if you render in a window

---

## Build & Run

### Option A — CMake (recommended)
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
````

Run:

```bash
./GameOfLife
```

### Option B — Manual / IDE

* Open the project in your IDE
* Ensure OpenCL headers/libs are configured:

  * `CL/cl.h`
  * link to OpenCL library (`OpenCL.lib` on Windows, `-lOpenCL` on Linux)

---

## Configuration

Common knobs (depending on your implementation):

* Grid width/height (e.g., `--w 1920 --h 1080`)
* Number of iterations or run-until-exit
* Device selection:

  * CPU only
  * GPU only
  * hybrid (CPU + GPU)

Example:

```bash
./GameOfLife --w 2048 --h 2048 --device gpu
```

---

## Performance Notes

* If your display is **vsync-capped**, FPS won’t reflect compute performance.
* Use kernel timing to compute throughput:

  * `iterations_per_second = 1000ms / kernel_time_ms`
* Keeping buffers on the device (avoiding frequent readbacks) is critical for performance.

---

## Common Troubleshooting

**No OpenCL platforms found**

* Install the correct GPU driver or CPU OpenCL runtime.

**Build errors: missing `CL/cl.h`**

* Install OpenCL headers (e.g., OpenCL SDK / ICD loader)
* Verify include paths in your build system.

**Kernel compile failure**

* Print the OpenCL build log when `clBuildProgram` fails.

---

## Acknowledgements

* Conway’s Game of Life
* OpenCL community examples/docs
