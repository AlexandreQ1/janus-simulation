// volume_grid.cu — per-particle velocity computation (splatting, no voxels)
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "volume_grid.cuh"

static int    g_n = 0;
static float* d_speed = nullptr;     // [N] |v|
static float* d_acc   = nullptr;     // [2] somme |v|, compte
static std::vector<float> g_host;    // [N] velocities in RAM
static float  g_velRef = 1.0f;

__global__ void speedK(const float4* vel, int N, float* speed, float* acc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 v = vel[i];
    float s = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    speed[i] = s;
    atomicAdd(&acc[0], s);
    atomicAdd(&acc[1], 1.0f);
}

bool volumeInit(int N)
{
    if (N == g_n && d_speed) return true;
    volumeFree();
    g_n = N;
    if (cudaMalloc(&d_speed, (size_t)N * sizeof(float)) != cudaSuccess) { g_n = 0; return false; }
    if (cudaMalloc(&d_acc, 2 * sizeof(float)) != cudaSuccess) { volumeFree(); return false; }
    g_host.assign(N, 0.0f);
    return true;
}
void volumeFree()
{
    if (d_speed) cudaFree(d_speed);
    if (d_acc) cudaFree(d_acc);
    d_speed = d_acc = nullptr; g_n = 0; g_host.clear();
}

void volumeComputeSpeeds(const void* d_vel, int N)
{
    if (!d_speed || N <= 0 || !d_vel) return;
    if (N > g_n) N = g_n;
    float z[2] = { 0, 0 };
    cudaMemcpy(d_acc, z, 2 * sizeof(float), cudaMemcpyHostToDevice);
    int block = 256, grid = (N + block - 1) / block;
    speedK<<<grid, block>>>((const float4*)d_vel, N, d_speed, d_acc);
    float a[2]; cudaMemcpy(a, d_acc, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    g_velRef = (a[1] > 0.0f) ? (a[0] / a[1]) : 1.0f;
    cudaMemcpy(g_host.data(), d_speed, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost);
}

const float* volumeSpeedData() { return g_host.data(); }
float volumeVelRef() { return g_velRef; }
