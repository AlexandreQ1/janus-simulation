#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH
#include <cuda_runtime.h>

// PM helper kernels (standalone drift / Hubble drag for the PM DKD path)
void launch_drift_half(
    const float4* p_in, float4* p_out, const float4* vel,
    unsigned int N, float dt_half, float box_half,
    dim3 grid, dim3 block);

void launch_hubble_drag(
    const float4* pos, float4* vel,
    unsigned int N, float dt, float H,
    dim3 grid, dim3 block);

// Crash guard: sanitize NaN/Inf + cap runaway pos/vel so the tree never
// receives invalid coordinates (prevents freeze/driver-reset on over-collapse).
void launch_sanitize(
    float4* pos, float4* vel,
    unsigned int N, float posBound, float vMax,
    dim3 grid, dim3 block);

// Adaptive-dt helper: GPU reduction of max|accel| -> per-block maxima in blockMax
// (avoids the per-step full readback + CPU loop).
void launch_reduce_max_accel(
    const float4* acc, unsigned int N, float* blockMax, int nBlocks);

// Force subcycling: derive accel = (vNew - vOld)/dt after a full kick, then
// re-apply v += accel·dt on the cheap sub-steps (no tree rebuild/traversal).
void launch_accel_from_dv(
    const float4* vNew, const float4* vOld, float4* accel,
    unsigned int N, float dt, dim3 grid, dim3 block);
void launch_apply_accel_kick(
    float4* vel, const float4* accel, unsigned int N, float dt,
    dim3 grid, dim3 block);

// Anti global-bubble ("Jeans swindle"): remove, per species (sign of pos.w),
// the spherically-averaged radial acceleration about the origin — i.e. the
// smooth background infall of the finite sphere that otherwise collapses the
// whole negative cloud into one central blob. Local (non-radial) fluctuations
// that build the soap-bubble web are preserved. Operates in place on accel.
void launch_remove_radial_monopole(
    float4* accel, const float4* pos, unsigned int N, float frac,
    dim3 grid, dim3 block);

// Rebuild velocity from a base velocity + (corrected) accel: v = vBase + a·dt.
// Used after the monopole removal to re-apply the corrected kick (the tree
// kick had already applied the uncorrected accel in place). Preserves vel.w.
void launch_v_from_base_accel(
    float4* vel, const float4* vBase, const float4* accel,
    unsigned int N, float dt, dim3 grid, dim3 block);

// Old signature kept for compatibility during transition
void launch_nbody(
    const float4* p_start, float4* p_end,
    const float4* v_start, float4* v_end,
    float softening, float timestep, float G, unsigned int N,
    float sim_time, float sp_amplitude, float sp_pattern_speed,
    float sp_pitch_tan, unsigned int sp_arms, int physicsModel,
    float hubble, float box_half,
    dim3 grid, dim3 block, size_t sharedMemBytes);

// Cosmological leapfrog DKD kernel with periodic boundaries and Hubble drag
void launch_nbody_cosmo(
    const float4* p_start, float4* p_end,
    const float4* v_start, float4* v_end,
    float softening, float timestep, float G, unsigned int N,
    float a, float aDot, float omegaM, float omegaL, float rho_mean,
    int physicsModel, float box_half,
    dim3 grid, dim3 block, size_t sharedMemBytes);

// Force-only kernel (for adaptive timestep estimation)
void launch_nbody_force(
    const float4* p_start,
    float4* a_out,
    float softening, float G, unsigned int N,
    float a_inv3,
    int physicsModel, float box_half,
    dim3 grid, dim3 block, size_t sharedMemBytes);

#endif