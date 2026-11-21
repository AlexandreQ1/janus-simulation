// cuda_kernels.cuh — CUDA kernel launch wrappers
// Declarations for C++ code to call CUDA kernels compiled in .cu files

#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

// ── nbody kernel wrapper (from pair3.cu) ─────────────────────────────────────
// physicsModel: 0=Newton, 1=Janus, 2=Yukawa, 3=MOND, 4=Custom
void launch_nbody(
    const float4* p_start,
    float4* p_end,
    const float4* v_start,
    float4* v_end,
    float softening,
    float timestep,
    float G,
    unsigned int N,
    float sim_time,
    float sp_amplitude,
    float sp_pattern_speed,
    float sp_pitch_tan,
    unsigned int sp_arms,
    int physicsModel,
    dim3 grid,
    dim3 block,
    size_t sharedMemBytes);

// ── Spiral density-wave velocity kick (Lin-Shu) ──────────────────────────────
// Standalone kick so density waves work in Tree mode too (the tree kernel has
// no spiral term). Replicates exactly the spiral force in launch_nbody, applied
// to positive particles only. Call after tree_sr_kick, before the 2nd drift.
void launch_spiral_kick(
    const float4* pos, float4* vel,
    unsigned int N, float G, float timestep, float sim_time,
    float sp_amplitude, float sp_pattern_speed, float sp_pitch_tan,
    unsigned int sp_arms);

#endif
