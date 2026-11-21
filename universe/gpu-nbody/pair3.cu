// pair3.cu — j-gravity-universe : cosmological N-body Janus + leapfrog DKD
// physicsModel: 0=Newton, 1=Janus, 2=Yukawa, 3=MOND, 4=Custom

#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// ------------------------------------------------------------------
//  Force helpers
// ------------------------------------------------------------------

__device__ inline float4 pair_force(
    float4 pi, float4 pj, float4 ai, float softening, int physicsModel, float box_half)
{
    float dx = pj.x - pi.x, dy = pj.y - pi.y, dz = pj.z - pi.z;
    // Minimum image convention (periodic boundary conditions)
    if (box_half > 0.0f) {
        if (dx >  box_half) dx -= 2.0f * box_half;
        else if (dx < -box_half) dx += 2.0f * box_half;
        if (dy >  box_half) dy -= 2.0f * box_half;
        else if (dy < -box_half) dy += 2.0f * box_half;
        if (dz >  box_half) dz -= 2.0f * box_half;
        else if (dz < -box_half) dz += 2.0f * box_half;
    }
    float distSq  = dx*dx + dy*dy + dz*dz + softening;
    float invDist  = rsqrtf(distSq);
    float invDist3 = invDist * invDist * invDist;
    float s;
    switch (physicsModel) {
        case 0: s = fabsf(pj.w) * invDist3; break;   // Newton : all masses attract
        case 2: { float d=sqrtf(distSq); s=fabsf(pj.w)*invDist3*(1.f+.5f*expf(-d/1000.f)); break; }
        case 3: { float d=sqrtf(distSq),a=fabsf(pj.w)*invDist3*d,x=a/1.2e-10f;
                  s=pj.w*invDist3*x*rsqrtf(1.f+x*x); break; }
        default: s = pj.w * invDist3; break;          // Janus (1) : signed mass
    }
    ai.x += dx*s; ai.y += dy*s; ai.z += dz*s;
    return ai;
}

// ------------------------------------------------------------------
//  Force-only kernel (for adaptive timestep estimation)
// ------------------------------------------------------------------

__global__ void nbody_force(
    const float4* __restrict__ p_start,
    float4* __restrict__       a_out,
    float softening, float G, unsigned int N,
    float a_inv3,
    int physicsModel,
    float box_half)
{
    extern __shared__ float4 cached_stars[];
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cs  = blockDim.x;
    unsigned int tc  = gridDim.x * blockDim.x;

    for (; gid < N; gid += tc)
    {
        float4 pos = p_start[gid];
        float4 acc = make_float4(0,0,0,0);
        int nc = ((int)N + (int)cs - 1) / (int)cs;
        for (int c = 0; c < nc; c++) {
            int idx = c * (int)cs + (int)threadIdx.x;
            cached_stars[threadIdx.x] = (idx < (int)N) ? p_start[idx] : make_float4(0,0,0,0);
            __syncthreads();
            for (int j = 0; j < (int)cs; j++)
                acc = pair_force(pos, cached_stars[j], acc, softening, physicsModel, box_half);
            __syncthreads();
        }
        acc.x *= a_inv3; acc.y *= a_inv3; acc.z *= a_inv3;
        float sign_i = (physicsModel == 1 && pos.w < 0.f) ? -1.f : 1.f;
        a_out[gid] = make_float4(sign_i * G * acc.x, sign_i * G * acc.y, sign_i * G * acc.z, 0.0f);
    }
}

// ------------------------------------------------------------------
//  Cosmological leapfrog DKD kernel
// ------------------------------------------------------------------

__global__ void nbody_cosmo(
    const float4* __restrict__ p_start,
    float4* __restrict__       p_end,
    const float4* __restrict__ v_start,
    float4* __restrict__       v_end,
    float softening, float timestep, float G, unsigned int N,
    float a, float aDot, float omegaM, float omegaL, float rho_mean,
    int physicsModel,
    float box_half)
{
    extern __shared__ float4 cached_stars[];
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cs  = blockDim.x;
    unsigned int tc  = gridDim.x * blockDim.x;

    float a_inv3 = 1.0f / (a * a * a);
    float H = aDot / a;
    float hubble_drag = 1.0f - 2.0f * H * timestep;

    for (; gid < N; gid += tc)
    {
        float4 pos = p_start[gid];
        float4 vel = v_start[gid];

        // ---- Drift (half-step) ----
        pos.x += vel.x * timestep * 0.5f;
        pos.y += vel.y * timestep * 0.5f;
        pos.z += vel.z * timestep * 0.5f;

        // ---- Force computation ----
        float4 force = make_float4(0,0,0,0);
        int nc = ((int)N + (int)cs - 1) / (int)cs;
        for (int c = 0; c < nc; c++) {
            int idx = c * (int)cs + (int)threadIdx.x;
            cached_stars[threadIdx.x] = (idx < (int)N) ? p_start[idx] : make_float4(0,0,0,0);
            __syncthreads();
            for (int j = 0; j < (int)cs; j++)
                force = pair_force(pos, cached_stars[j], force, softening, physicsModel, box_half);
            __syncthreads();
        }
        force.x *= a_inv3; force.y *= a_inv3; force.z *= a_inv3;

        // δρ correction: subtract mean-field background (Gauss law for uniform sphere).
        // acc_bg = -(4π/3) × ρ̄ × r  →  force -= acc_bg  ≡  force += (4π/3)×ρ̄×a⁻³×r
        // For symmetric Janus (rho_mean≈0) this is a no-op; non-zero only for asymmetric cases.
        {
            float bg = (4.0f * 3.14159265f / 3.0f) * rho_mean * a_inv3;
            force.x += bg * pos.x;
            force.y += bg * pos.y;
            force.z += bg * pos.z;
        }

        float sign_i = (physicsModel == 1 && pos.w < 0.f) ? -1.f : 1.f;

        // ---- Kick ----
        vel.x += sign_i * force.x * G * timestep;
        vel.y += sign_i * force.y * G * timestep;
        vel.z += sign_i * force.z * G * timestep;

        // Drag de Hubble appliqué uniquement sur la vitesse PECULIERE (v - H*r).
        // Drag sur vitesse totale (bug précédent) détruisait le flux de Hubble
        // en ~8 MY, figeant les particules. Formule correcte :
        //   v_new = (v - H*x)*drag + H*x
        // Pour v = H*x (flux pur, vitesse peculière = 0) : v_new = H*x → inchangé.
        if (fabsf(H) > 1e-12f) {
            float hub_x = H * pos.x;
            float hub_y = H * pos.y;
            float hub_z = H * pos.z;
            vel.x = (vel.x - hub_x) * hubble_drag + hub_x;
            vel.y = (vel.y - hub_y) * hubble_drag + hub_y;
            vel.z = (vel.z - hub_z) * hubble_drag + hub_z;
        }

        // ---- Drift (half-step) ----
        pos.x += vel.x * timestep * 0.5f;
        pos.y += vel.y * timestep * 0.5f;
        pos.z += vel.z * timestep * 0.5f;

        // ---- Periodic boundary: wrap into [-box_half, +box_half] ----
        if (box_half > 0.0f) {
            if (pos.x >  box_half) pos.x -= 2.0f * box_half;
            else if (pos.x < -box_half) pos.x += 2.0f * box_half;
            if (pos.y >  box_half) pos.y -= 2.0f * box_half;
            else if (pos.y < -box_half) pos.y += 2.0f * box_half;
            if (pos.z >  box_half) pos.z -= 2.0f * box_half;
            else if (pos.z < -box_half) pos.z += 2.0f * box_half;
        }

        p_end[gid] = pos;
        v_end[gid] = vel;
    }
}

// ------------------------------------------------------------------
//  Legacy kernel (kept for compatibility with galaxy mode)
// ------------------------------------------------------------------

__global__ void nbody(
    const float4* __restrict__ p_start,
    float4* __restrict__       p_end,
    const float4* __restrict__ v_start,
    float4* __restrict__       v_end,
    float softening, float timestep, float G, unsigned int N,
    float sim_time, float sp_amplitude, float sp_pattern_speed,
    float sp_pitch_tan, unsigned int sp_arms, int physicsModel,
    float hubble, float box_half)
{
    extern __shared__ float4 cached_stars[];
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int cs  = blockDim.x, lid = threadIdx.x;
    unsigned int tc  = gridDim.x*blockDim.x;

    for (; gid < N; gid += tc)
    {
        float4 pos = p_start[gid], vel = v_start[gid];
        float4 force = make_float4(0,0,0,0);
        int nc = ((int)N+(int)cs-1)/(int)cs;
        for (int c=0; c<nc; c++) {
            int idx = c*(int)cs+(int)lid;
            cached_stars[lid] = (idx<(int)N) ? p_start[idx] : make_float4(0,0,0,0);
            __syncthreads();
            for (int j=0; j<(int)cs; j++)
                force = pair_force(pos, cached_stars[j], force, softening, physicsModel, box_half);
            __syncthreads();
        }

        float sign_i = (physicsModel==1 && pos.w<0.f) ? -1.f : 1.f;

        vel.x += sign_i*force.x*G*timestep;
        vel.y += sign_i*force.y*G*timestep;
        vel.z += sign_i*force.z*G*timestep;

        if (hubble > 0.f) {
            float d = 1.f - hubble*timestep;
            if (d < 0.01f) d = 0.01f;
            vel.x *= d; vel.y *= d; vel.z *= d;
        }

        pos.x += vel.x*timestep;
        pos.y += vel.y*timestep;
        pos.z += vel.z*timestep;

        p_end[gid] = pos;
        v_end[gid] = vel;
    }
}

// ------------------------------------------------------------------
//  Host launch wrappers
// ------------------------------------------------------------------

void launch_nbody(
    const float4* p_start, float4* p_end,
    const float4* v_start, float4* v_end,
    float softening, float timestep, float G, unsigned int N,
    float sim_time, float sp_amplitude, float sp_pattern_speed,
    float sp_pitch_tan, unsigned int sp_arms, int physicsModel,
    float hubble, float box_half,
    dim3 grid, dim3 block, size_t sharedMemBytes)
{
    nbody<<<grid, block, sharedMemBytes>>>(
        p_start, p_end, v_start, v_end,
        softening, timestep, G, N,
        sim_time, sp_amplitude, sp_pattern_speed, sp_pitch_tan, sp_arms,
        physicsModel, hubble, box_half);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("[CUDA] nbody error: %s\n", cudaGetErrorString(err));
}

void launch_nbody_cosmo(
    const float4* p_start, float4* p_end,
    const float4* v_start, float4* v_end,
    float softening, float timestep, float G, unsigned int N,
    float a, float aDot, float omegaM, float omegaL, float rho_mean,
    int physicsModel, float box_half,
    dim3 grid, dim3 block, size_t sharedMemBytes)
{
    nbody_cosmo<<<grid, block, sharedMemBytes>>>(
        p_start, p_end, v_start, v_end,
        softening, timestep, G, N,
        a, aDot, omegaM, omegaL, rho_mean,
        physicsModel, box_half);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("[CUDA] nbody_cosmo error: %s\n", cudaGetErrorString(err));
}

void launch_nbody_force(
    const float4* p_start,
    float4* a_out,
    float softening, float G, unsigned int N,
    float a_inv3,
    int physicsModel, float box_half,
    dim3 grid, dim3 block, size_t sharedMemBytes)
{
    nbody_force<<<grid, block, sharedMemBytes>>>(
        p_start, a_out,
        softening, G, N,
        a_inv3,
        physicsModel, box_half);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("[CUDA] nbody_force error: %s\n", cudaGetErrorString(err));
}