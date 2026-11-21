// pair3.cu — N-body kernel with runtime-selectable physics model (CUDA)
// Shared memory tiling for coalesced access
// physicsModel: 0=Newton, 1=Janus, 2=Yukawa, 3=MOND, 4=Custom

#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// ── pair_force: pairwise interaction with model dispatch ─────────────────────
__device__ inline float4 pair_force(
    float4 pi, float4 pj, float4 ai, float softening, int physicsModel)
{
    float dx = pj.x - pi.x;
    float dy = pj.y - pi.y;
    float dz = pj.z - pi.z;

    float distSq   = dx*dx + dy*dy + dz*dz + softening;
    float invDist   = rsqrtf(distSq);
    float invDist3  = invDist * invDist * invDist;

    float s;
    switch (physicsModel)
    {
        case 2: { // Yukawa — Newton + short-range exponential correction
            float dist    = sqrtf(distSq);
            const float alpha  = 0.5f;
            const float lambda = 1000.0f;
            s = pj.w * invDist3 * (1.0f + alpha * expf(-dist / lambda));
            break;
        }
        case 3: { // MOND — interpolating function mu(a/a0)
            float dist = sqrtf(distSq);
            const float a0 = 1.2e-10f;
            float a_n  = fabsf(pj.w) * invDist3 * dist;
            float x    = a_n / a0;
            float mu   = x * rsqrtf(1.0f + x * x);
            s = pj.w * invDist3 * mu;
            break;
        }
        default: // 0=Newton, 1=Janus (sign_i at integration), 4=Custom
            s = pj.w * invDist3;
            break;
    }

    ai.x += dx * s;
    ai.y += dy * s;
    ai.z += dz * s;
    return ai;
}

// ── nbody: main N-body kernel ────────────────────────────────────────────────
__global__ void nbody(
    const float4* __restrict__ p_start,
    float4* __restrict__ p_end,
    const float4* __restrict__ v_start,
    float4* __restrict__ v_end,
    float softening,
    float timestep,
    float G,
    unsigned int N,
    float sim_time,
    float sp_amplitude,
    float sp_pattern_speed,
    float sp_pitch_tan,
    unsigned int sp_arms,
    int physicsModel)
{
    extern __shared__ float4 cached_stars[];

    unsigned int globalid    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int chunksize   = blockDim.x;
    unsigned int localid     = threadIdx.x;
    unsigned int threadcount = gridDim.x * blockDim.x;

    for (; globalid < N; globalid += threadcount)
    {
        float4 pos   = p_start[globalid];
        float4 vel   = v_start[globalid];
        float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        int numChunks = ((int)N + (int)chunksize - 1) / (int)chunksize;

        for (int chunk = 0; chunk < numChunks; chunk++)
        {
            int base = chunk * (int)chunksize;
            int idx  = base + (int)localid;

            cached_stars[localid] = (idx < (int)N)
                ? p_start[idx]
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            __syncthreads();

            for (int j = 0; j < (int)chunksize; j++)
                force = pair_force(pos, cached_stars[j], force, softening, physicsModel);

            __syncthreads();
        }

        // Janus eq. 4.4.10: sgn(m_i) only for Janus model
        float sign_i = (physicsModel == 1 && pos.w < 0.0f) ? -1.0f : 1.0f;

        // Lin-Shu density wave — tangential (azimuthal) perturbation, scaled to the
        // LOCAL gravity |force| so sp_amplitude is a FRACTION of gravity (G-independent).
        // (force still holds the accumulated gravity here; the common ×G below applies once.)
        if (pos.w > 0.0f && sp_amplitude > 0.0f)
        {
            float r_xy = sqrtf(pos.x * pos.x + pos.y * pos.y);
            if (r_xy > 1.0f)
            {
                float fmag  = sqrtf(force.x*force.x + force.y*force.y + force.z*force.z);
                float theta = atan2f(pos.y, pos.x);
                float phase = (float)sp_arms *
                    (theta + sp_pitch_tan * logf(r_xy) + sp_pattern_speed * sim_time);
                float Ft    = sp_amplitude * sinf(phase) * fmag;   // ε·|g_local|
                force.x += -sinf(theta) * Ft;
                force.y +=  cosf(theta) * Ft;
            }
        }

        // Leapfrog symplectique (drift-kick)
        pos.x += vel.x * timestep;
        pos.y += vel.y * timestep;
        pos.z += vel.z * timestep;

        vel.x += sign_i * force.x * G * timestep;
        vel.y += sign_i * force.y * G * timestep;
        vel.z += sign_i * force.z * G * timestep;

        p_end[globalid] = pos;
        v_end[globalid] = vel;
    }
}

// ── C++ wrapper ──────────────────────────────────────────────────────────────
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
    size_t sharedMemBytes)
{
    nbody<<<grid, block, sharedMemBytes>>>(
        p_start, p_end, v_start, v_end,
        softening, timestep, G, N,
        sim_time, sp_amplitude, sp_pattern_speed, sp_pitch_tan, sp_arms,
        physicsModel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("[CUDA] nbody kernel error: %s\n", cudaGetErrorString(err));
    // Implicit sync in cudaMemcpy (readPositions) — not needed here.
}

// ── Spiral density-wave velocity kick (standalone, for Tree mode) ────────────
// Tree mode has no per-particle gravity force here, so the local gravity is
// estimated by the centripetal acceleration v²/r (≈ |g| for near-circular orbits).
// The kick is then a FRACTION sp_amplitude of that — matching the N² kernel.
// Positive particles only. Velocities updated in place. (G unused: g already an accel.)
__global__ void spiral_kick(
    const float4* __restrict__ pos, float4* vel,
    unsigned int N, float G, float timestep, float sim_time,
    float sp_amplitude, float sp_pattern_speed, float sp_pitch_tan,
    unsigned int sp_arms)
{
    (void)G;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 p = pos[i];
    if (p.w <= 0.0f) return;                 // positive stars only
    float r_xy = sqrtf(p.x * p.x + p.y * p.y);
    if (r_xy <= 1.0f) return;
    float4 v = vel[i];
    float gmag = (v.x*v.x + v.y*v.y) / r_xy;  // ≈ |g_local| (centripetal v²/r)
    float theta = atan2f(p.y, p.x);
    float phase = (float)sp_arms *
        (theta + sp_pitch_tan * logf(r_xy) + sp_pattern_speed * sim_time);
    float Ft   = sp_amplitude * sinf(phase) * gmag;   // ε·|g_local|
    v.x += (-sinf(theta) * Ft) * timestep;
    v.y += ( cosf(theta) * Ft) * timestep;
    vel[i] = v;
}

void launch_spiral_kick(
    const float4* pos, float4* vel,
    unsigned int N, float G, float timestep, float sim_time,
    float sp_amplitude, float sp_pattern_speed, float sp_pitch_tan,
    unsigned int sp_arms)
{
    if (N == 0 || sp_amplitude <= 0.0f) return;
    dim3 block(256), grid((N + 255) / 256);
    spiral_kick<<<grid, block>>>(pos, vel, N, G, timestep, sim_time,
                                 sp_amplitude, sp_pattern_speed, sp_pitch_tan, sp_arms);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("[CUDA] spiral_kick error: %s\n", cudaGetErrorString(err));
}
