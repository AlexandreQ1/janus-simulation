// gl_interop.cu — Janus Universe
// CUDA↔OpenGL interop : build the render vertex buffer DIRECTLY on the GPU into a
// mapped GL VBO, from the simulation's device buffers. Avoids the per-frame CPU
// round-trip (readback + rebuild + re-upload) that dominates at millions of particles.
//
// Include order matters on Windows: windows.h BEFORE cuda_gl_interop.h (which pulls
// <GL/gl.h> and conflicts with the SDL/GLEW headers).

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>

#include "gl_interop.cuh"

// Vertex layout = 6 floats : [x, y, z, value, origIdx, size].
//   value : mass (p.w) ; |v| if mode=1 ; aux[i] if mode=2. Signed by species.
//   size  : SPH smoothing length (signed by species) if softHalo, else ±1.
__global__ void build_vertices_kernel(
    float* __restrict__ out,
    const float4* __restrict__ pos,
    const float4* __restrict__ vel,
    const float*  __restrict__ smooth,
    const float*  __restrict__ aux,
    int n, int mode, int softHalo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 p = pos[i];

    float val = p.w;                         // mode 0 : masse (déjà signée)
    if (mode == 1 && vel) {                  // couleur-vitesse : |v| signé par espèce
        float4 v = vel[i];
        float s = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
        val = (p.w > 0.0f) ? s : -s;
    } else if (mode == 2 && aux) {           // couleur auxiliaire : temp/web/halo/densité
        float a = aux[i];
        val = (p.w > 0.0f) ? a : -a;
    }

    float sz = (softHalo && smooth) ? smooth[i] : 1.0f;
    sz = (p.w < 0.0f) ? -sz : sz;

    float* o = out + (size_t)i * 6;
    o[0] = p.x; o[1] = p.y; o[2] = p.z;
    o[3] = val; o[4] = (float)i; o[5] = sz;
}

bool interopRegister(void** outRes, unsigned int vbo)
{
    cudaGraphicsResource* res = nullptr;
    cudaError_t e = cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (e != cudaSuccess) {
        printf("[interop] register failed: %s\n", cudaGetErrorString(e));
        *outRes = nullptr;
        return false;
    }
    *outRes = res;
    return true;
}

void interopUnregister(void* res)
{
    if (res) cudaGraphicsUnregisterResource((cudaGraphicsResource*)res);
}

bool interopFillVertices(void* res, const void* d_pos, const void* d_vel,
                         const void* d_smooth, const void* d_aux,
                         int n, int mode, int softHalo)
{
    if (!res || n <= 0 || !d_pos) return false;
    cudaGraphicsResource* r = (cudaGraphicsResource*)res;
    if (cudaGraphicsMapResources(1, &r, 0) != cudaSuccess) return false;

    float* dptr = nullptr; size_t bytes = 0;
    if (cudaGraphicsResourceGetMappedPointer((void**)&dptr, &bytes, r) != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &r, 0);
        return false;
    }
    if (bytes < (size_t)n * 6 * sizeof(float)) {   // VBO too small -> bail (CPU path)
        cudaGraphicsUnmapResources(1, &r, 0);
        return false;
    }

    int block = 256, grid = (n + block - 1) / block;
    build_vertices_kernel<<<grid, block>>>(
        dptr, (const float4*)d_pos, (const float4*)d_vel,
        (const float*)d_smooth, (const float*)d_aux, n, mode, softHalo);

    cudaGraphicsUnmapResources(1, &r, 0);
    return true;
}

// Device float-buffer helpers (so the C++ app needs no CUDA headers here).
void* interopAllocFloats(int n)
{
    void* p = nullptr;
    if (n > 0 && cudaMalloc(&p, (size_t)n * sizeof(float)) != cudaSuccess) p = nullptr;
    return p;
}
void interopFree(void* p) { if (p) cudaFree(p); }
void interopUploadFloats(void* d, const float* host, int n)
{
    if (d && host && n > 0)
        cudaMemcpy(d, host, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
}
