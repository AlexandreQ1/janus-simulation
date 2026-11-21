#pragma once
#include <cuda_runtime.h>
#include <cufft.h>

struct PMContext {
    int    Ng;              // grid dimension (e.g., 256)
    float  box_size;        // box side length (= 2 × radius for periodic cube)
    float  softening;       // Plummer softening (Mpc)
    bool   periodic_bc;     // true = periodic cube (correct for Janus alveolar structure)
                            // false = open-boundary sphere (Hockney-Eastwood)
    float* d_density;       // real density [Ng³]

    // Open-boundary (H&E): padded (2Ng)³ work buffers
    cufftComplex* d_padded;
    cufftComplex* d_green_hat;
    cufftComplex* d_rho_scratch;  // rho_hat backup for H&E spectral gradient (3 passes)
    cufftHandle plan_r2c;   // R2C for 2Ng×2Ng×2Ng
    cufftHandle plan_c2r;   // C2R for 2Ng×2Ng×2Ng

    // Periodic: Ng³ work plans (reuse d_padded buffer — it is large enough)
    cufftHandle plan_r2c_per;   // R2C for Ng×Ng×Ng
    cufftHandle plan_c2r_per;   // C2R for Ng×Ng×Ng

    float4* d_force_field;   // combined force field: f_pos - f_neg [Ng³]
    float4* d_force_field2;  // temporary: second species force field [Ng³]
    float*  d_density2;      // second species density grid [Ng³]
    bool    initialized;

    // P³M short-range correction (Ewald split — GADGET2 style)
    // When sr_enabled: PM kernel multiplied by exp(-k²·r_split²/2) → long-range only.
    // SR kernel applies χ(r) = erfc(r/(√2·r_split)) + (r/r_split)·√(2/π)·exp(-r²/(2·r_split²)).
    // The two add up to the unsplit force exactly (no double-counting).
    // When !sr_enabled: PM behaves as pure PM (no smoothing, no SR pass).
    bool   sr_enabled;    // false → PM only (no Gaussian smoothing, SR pass skipped)
    float  sr_r_split;    // PM/SR split scale (Mpc) — typically ~1–2 cells
    float  sr_r_cut;      // short-range cutoff radius (Mpc) — typically 4.5·r_split
    int    sr_Nc;         // coarse grid cells per dimension
    int*   d_sr_count;    // [sr_Nc³] particles per coarse cell
    int*   d_sr_start;    // [sr_Nc³] prefix sum (cell start indices)
    int*   d_sr_slot;     // [sr_Nc³] atomic slot counter
    float4* d_sr_sorted;  // [N] particle positions sorted by cell
    int*   d_sr_orig;     // [N] original particle indices after sort
};

void pm_init(PMContext& ctx, int Ng, float box_size, float softening,
             bool periodic_bc = false, bool sr_enabled = true);

// P³M short-range direct Janus force (call after pm_force_step)
void pm_janus_sr_kick(PMContext& ctx, const float4* d_pos, float4* d_vel,
                      unsigned int N, float G, float a_inv3, float dt,
                      int physicsModel);

// Re-center all particles around the arithmetic mean position/velocity.
// Called each PM step to suppress the Janus CM drift.
void launch_recenter_velocities(float4* d_vel, unsigned int N);
void launch_recenter_positions(float4* d_pos, unsigned int N);
void pm_free(PMContext& ctx);

bool pm_force_step(
    PMContext&    ctx,
    const float4* d_pos,
    float4*       d_vel,
    unsigned int  N,
    float timestep,
    float G,
    float a_inv3,
    int   physicsModel);
