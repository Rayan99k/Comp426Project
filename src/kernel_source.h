#pragma once

static const char* LIFE_KERNEL_SRC = R"CLC(
typedef unsigned char U8;
typedef unsigned int  U32;

inline int IB(int x,int y,U32 w,U32 h) {
    return (x>=0 && y>=0 && (U32)x<w && (U32)y<h);
}

inline int CN(__global const U8* g,int x,int y,U32 w,U32 h,U8 s) {
    int c=0;
    for(int dy=-1;dy<=1;++dy){
        for(int dx=-1;dx<=1;++dx){
            if(dx==0 && dy==0) continue;
            int nx=x+dx, ny=y+dy;
            if(IB(nx,ny,w,h)){
                U32 id=(U32)ny*w+(U32)nx;
                if(g[id]==s) ++c;
            }
        }
    }
    return c;
}

__kernel void life_step(__global const U8* A, __global U8* B, const U32 W, const U32 H, const U32 NS) {
    U32 gid   = get_global_id(0);
    U32 gsize = get_global_size(0);

    U32 N = W * H;

    for (U32 id = gid; id < N; id += gsize) {
        U32 y = id / W;
        U32 x = id % W;

        U8 v   = A[id];
        U8 out = v;

        if (v != 0) {
            int n = CN(A, (int)x,(int)y, W,H, v);
            if (!(n == 2 || n == 3)) out = 0;
        } else {
            U8 pick = 0;
            for (U8 s = 1; s <= (U8)NS; ++s) {
                int n = CN(A, (int)x,(int)y, W,H, s);
                if (n == 3) { pick = s; break; }
            }
            out = pick;
        }

        B[id] = out;
    }
}
__kernel void pipe_producer(__global const uchar* grid,
                            const uint            N,
                            write_only pipe uint  outPipe)
{
    uint gid    = get_global_id(0);
    uint stride = get_global_size(0);

    uint count = 0;
    for (uint i = gid; i < N; i += stride) {
        if (grid[i] != (uchar)0)
            ++count;
    }

    write_pipe(outPipe, &count);
}

__kernel void pipe_consumer(read_only pipe uint  inPipe,
                            __global uint*       partial,
                            const uint           numItems)
{
    uint gid = get_global_id(0);
    if (gid >= numItems) return;

    uint v = 0;
    read_pipe(inPipe, &v);
    partial[gid] = v;
}
)CLC";
