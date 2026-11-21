// pm_force.cu — Particle-Mesh gravitational force solver using cuFFT
// Zero-padded (open boundary) PM via Hockney-Eastwood algorithm.
// Correct open-boundary physics: particles interact only with each other,
// not with periodic images. The sphere expands freely into the zero region.
//
// Pipeline per step:
//   1. CIC density assignment to compact Ng³ grid (signed masses for Janus)
//   2. Embed Ng³ density into corner [0..Ng-1]³ of (2Ng)³ zero-padded array
//   3. Forward R2C FFT on (2Ng)³ → (2Ng)×(2Ng)×(Ng+1) complex
//   4. Hockney-Eastwood: φ̂ = G × Ĝ(k) × ρ̂(k)  (Ĝ pre-computed in pm_init)
//   5. Inverse C2R FFT back to (2Ng)³ reals → correct open-boundary potential
//   6. Read gradient from corner [0..Ng-1]³ of padded output
//   7. CIC force interpolation + kick

#include "pm_force.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// ------------------------------------------------------------------
//  Kernel 1: CIC density assignment
//  Each particle deposits signed mass to the 8 surrounding cells
//  using trilinear (Cloud-In-Cell) weights. Periodic boundaries.
// ------------------------------------------------------------------

__global__ void cic_deposit(
    const float4* __restrict__ pos,
    unsigned int  N,
    float*        density,
    float         box_half,
    float         inv_cell,
    int           Ng)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    float4 p = pos[gid];
    float mass = p.w;  // signed mass (positive or negative for Janus)

    // Map to grid coordinates [0, Ng)
    float gx = (p.x + box_half) * inv_cell;
    float gy = (p.y + box_half) * inv_cell;
    float gz = (p.z + box_half) * inv_cell;

    // Open boundary: skip particles outside the inner Ng³ grid (they're in the zero-padded region)
    if (gx < 0.0f || gx >= (float)Ng ||
        gy < 0.0f || gy >= (float)Ng ||
        gz < 0.0f || gz >= (float)Ng) return;

    // Integer cell (lower-left corner)
    int ix = (int)gx;
    int iy = (int)gy;
    int iz = (int)gz;

    // Fractional offset
    float dx = gx - (float)ix;
    float dy = gy - (float)iy;
    float dz = gz - (float)iz;

    // Deposit to 8 cells — skip (not clamp) neighbors outside grid (open boundary).
    // Clamping creates spurious mass concentration at cube faces; skipping avoids it.
#pragma unroll
    for (int a = 0; a < 2; a++) {
        int jx = ix + a; if (jx >= Ng) continue;
        float wx = (a == 0) ? (1.0f - dx) : dx;
#pragma unroll
        for (int b = 0; b < 2; b++) {
            int jy = iy + b; if (jy >= Ng) continue;
            float wy = (b == 0) ? (1.0f - dy) : dy;
#pragma unroll
            for (int c = 0; c < 2; c++) {
                int jz = iz + c; if (jz >= Ng) continue;
                float wz = (c == 0) ? (1.0f - dz) : dz;
                atomicAdd(&density[(jx * Ng + jy) * Ng + jz], mass * wx * wy * wz);
            }
        }
    }
}

// ------------------------------------------------------------------
//  Kernel 2: Embed compact Ng³ density into corner of padded array
//  Copies density[ix,iy,iz] → padded[(ix * Ng2 + iy) * stride + iz]
//  where stride = 2*Ng+2 (cuFFT R2C in-place padding for last dim)
// ------------------------------------------------------------------

__global__ void embed_density(
    const float* __restrict__ density,   // Ng³ compact
    float*       __restrict__ padded,    // (2Ng)×(2Ng)×(2Ng+2) reals, pre-zeroed
    int Ng)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= Ng * Ng * Ng) return;

    int iz  = gid % Ng;
    int tmp = gid / Ng;
    int iy  = tmp % Ng;
    int ix  = tmp / Ng;

    int Ng2    = 2 * Ng;
    int stride = Ng2 + 2;   // cuFFT C2R padding in last dim
    int dst    = (ix * Ng2 + iy) * stride + iz;
    padded[dst] = density[gid];
}

// ------------------------------------------------------------------
//  Kernel 3: Pre-compute real-space Hockney-Eastwood Green's function
//  G(r) = -1/r on the (2Ng)³ doubled grid, written into the padded
//  real layout (stride = 2Ng+2 in z).
//  Pre-divided by N = (2Ng)³ so that the unnormalized C2R output is
//  already the correct potential (no extra scaling needed).
//  G_grav is applied per-step in multiply_green_rho.
// ------------------------------------------------------------------

__global__ void compute_green_function(
    float* __restrict__ green_real,   // (2Ng)×(2Ng)×(2Ng+2), pre-zeroed
    float cell_size,
    float softening,                   // Plummer softening length (same as N² solver)
    int   Ng2)                         // Ng2 = 2*Ng
{
    int total = Ng2 * Ng2 * Ng2;
    int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    int iz  = gid % Ng2;
    int tmp = gid / Ng2;
    int iy  = tmp % Ng2;
    int ix  = tmp / Ng2;

    // Fold to nearest-image displacement in [-Ng, Ng]
    int dx = (ix > Ng2 / 2) ? ix - Ng2 : ix;
    int dy = (iy > Ng2 / 2) ? iy - Ng2 : iy;
    int dz = (iz > Ng2 / 2) ? iz - Ng2 : iz;

    float r2   = (float)(dx*dx + dy*dy + dz*dz) * cell_size * cell_size;
    float eps2 = softening * softening;
    // Plummer kernel: G(r) = -1/sqrt(r²+ε²)  — matches N² Plummer softening
    float denom = sqrtf(r2 + eps2);

    // Absorb C2R 1/(2Ng)³ normalization so no post-transform scaling is needed
    float norm = 1.0f / (float)(Ng2 * Ng2 * Ng2);

    int stride = Ng2 + 2;
    int dst    = (ix * Ng2 + iy) * stride + iz;
    green_real[dst] = -norm / denom;
}

// ------------------------------------------------------------------
//  Kernel 4: Hockney-Eastwood convolution (in-place)
//  rho_hat ← G_grav × G_hat × rho_hat   (complex multiply)
// ------------------------------------------------------------------

__global__ void multiply_green_rho(
    cufftComplex*       __restrict__ rho_hat,
    const cufftComplex* __restrict__ green_hat,
    float G_grav,
    int total,
    int Ng2)  // = 2*Ng, needed for MAS deconvolution
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    // MAS (Mass Assignment Scheme) deconvolution for CIC kernel.
    // CIC deposit introduces a smoothing window W_CIC(k) = sinc²(πk/Ng2)
    // in each dimension. Without correction, forces are ~20% weaker along
    // axis directions vs diagonals → lateral propagation artifact.
    // Standard cosmological codes (GADGET, PKDGRAV) all apply this correction.
    int Nk  = Ng2/2 + 1;
    int iz  = gid % Nk;
    int tmp = gid / Nk;
    int iy  = tmp % Ng2;
    int ix  = tmp / Ng2;
    int kxi = (ix > Ng2/2) ? ix - Ng2 : ix;
    int kyi = (iy > Ng2/2) ? iy - Ng2 : iy;

    // sinc²(πk/Ng2) — window function per dimension
    auto W = [Ng2](int k) -> float {
        if (k == 0) return 1.0f;
        float x = M_PI_F * fabsf((float)k) / (float)Ng2;
        float s = sinf(x) / x;
        return s * s;
    };
    float W_CIC2 = W(kxi) * W(kyi) * W(iz);  // 3D CIC window, one pass
    float W_CIC4 = W_CIC2 * W_CIC2;           // deposit × interpolation = both passes
    // Divide by W_CIC4 to deconvolve both CIC passes (deposit + interpolation).
    // Plummer softening in green_hat already suppresses the Nyquist modes.
    float mas = (W_CIC4 > 2e-3f) ? fminf(1.0f / W_CIC4, 64.0f) : 64.0f;

    float a = rho_hat[gid].x,   b = rho_hat[gid].y;
    float c = green_hat[gid].x, d = green_hat[gid].y;
    float G = G_grav * mas;
    rho_hat[gid].x = G * (a * c - b * d);
    rho_hat[gid].y = G * (a * d + b * c);
}

// ------------------------------------------------------------------
//  Kernels for PERIODIC PM path (Ng³, no zero-padding)
//  Used when ctx.periodic_bc = true (uniform cube initial conditions).
//  The periodic Poisson solver (-4πG/k²) treats the box as a torus,
//  so all Fourier modes grow independently — the alveolar structure
//  forms without dipole dominance.
// ------------------------------------------------------------------

// Embed compact Ng³ density into Ng×Ng×(Ng+2) padded layout for C2R
__global__ void embed_density_periodic(const float* __restrict__ density,
                                        float*       __restrict__ padded, int Ng)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= Ng*Ng*Ng) return;
    int iz = gid%Ng, tmp = gid/Ng, iy = tmp%Ng, ix = tmp/Ng;
    padded[(ix*Ng + iy)*(Ng+2) + iz] = density[gid];
}

// Periodic Poisson solve with DISCRETE Green's function.
// phi_hat = -4πG × exp(-k_eff×eps_pm) × ρ_hat / (k²_eff × Ng³)
// k²_eff = 4(sin²(πkxi/Ng) + sin²(πkyi/Ng) + sin²(πiz/Ng)) / h²
// This is the discrete Laplacian eigenvalue and is consistent with the
// central-difference gradient, eliminating the grid-axis force anisotropy.
// Plummer softening exp(-k_eff×eps_pm) suppresses sub-cell shot noise.
__global__ void periodic_poisson_solve(cufftComplex* __restrict__ hat,
                                        float G, float box_size, float eps_pm, int Ng)
{
    int Nk = Ng/2 + 1, total = Ng*Ng*Nk;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int iz = gid%Nk, tmp = gid/Nk, iy = tmp%Ng, ix = tmp/Ng;
    int kxi = (ix > Ng/2) ? ix-Ng : ix;
    int kyi = (iy > Ng/2) ? iy-Ng : iy;

    // Discrete Green's function: k_eff_i = sin(ki × π/Ng) × 2/h
    // Eigenvalue of the 7-point discrete Laplacian: k²_eff = 4Σsin²(πki/Ng)/h²
    // Consistent with compute_gradient_periodic (central differences), unlike the
    // continuous k² = Σki²×dk² which creates an axis-aligned force bias (cross artifact).
    float h   = box_size / (float)Ng;
    float s2  = M_PI_F / (float)Ng;
    float kex = sinf((float)kxi * s2) * 2.0f / h;
    float key = sinf((float)kyi * s2) * 2.0f / h;
    float kez = sinf((float)iz  * s2) * 2.0f / h;  // iz >= 0 (rfftfreq)
    float k2  = kex*kex + key*key + kez*kez;
    if (k2 < 1e-10f) { hat[gid].x = hat[gid].y = 0.0f; return; }

    // Plummer filter uses CONTINUOUS k (not discrete k_eff): the softening is a
    // physical sub-cell smoothing, so it must be isotropic in physical wavenumber.
    // Using discrete k_eff here would make the filter anisotropic (e.g., axial
    // Nyquist gets exp(-6), diagonal Nyquist gets exp(-6√3) — different damping
    // for modes with the SAME physical wavelength). Continuous k restores isotropy.
    float dk_cont = 2.0f * M_PI_F / box_size;
    float k_cont  = dk_cont * sqrtf((float)(kxi*kxi + kyi*kyi + iz*iz));
    float plummer = expf(-k_cont * eps_pm);
    // MAS deconvolution for CIC: must correct for BOTH deposit AND interpolation.
    // W_1D(ki) = sinc(πki/Ng) per dimension (NGP window).
    // CIC window per dim = sinc²(πki/Ng) = W_1D².
    // 3D CIC window for ONE pass: W_CIC = sinc²_x × sinc²_y × sinc²_z.
    // For deposit+interpolation: total bias = W_CIC² → must divide by W_CIC².
    // W_CIC2 = sinc²_x × sinc²_y × sinc²_z (one-pass 3D window).
    // W_CIC4 = W_CIC2² = (sinc²_x × sinc²_y × sinc²_z)² (both passes).
    // Proof: F_particle = W_interp × IFFT(-ig × phi) where phi uses W_deposit-biased rho.
    //        Dividing phi by W_CIC4 gives F_particle = F_true exactly. QED.
    auto W = [Ng](int ki) -> float {
        if (ki == 0) return 1.0f;
        float x = M_PI_F * fabsf((float)ki) / (float)Ng;
        float s = sinf(x) / x;
        return s * s;  // sinc² per dimension = CIC window per dim
    };
    float W_CIC2 = W(kxi) * W(kyi) * W(iz);  // 3D CIC window, one pass
    float W_CIC4 = W_CIC2 * W_CIC2;           // squared: corrects deposit+interpolation
    float mas = (W_CIC4 > 2e-3f) ? fminf(1.0f / W_CIC4, 64.0f) : 64.0f;
    float scale = -4.0f * M_PI_F * G * plummer * mas / (k2 * (float)(Ng*Ng*Ng));
    hat[gid].x *= scale;
    hat[gid].y *= scale;
}

// Gradient with periodic wrap (central differences on Ng×Ng×(Ng+2) padded output)
__global__ void compute_gradient_periodic(const float* __restrict__ potential,
                                           float4*      __restrict__ force_field,
                                           float inv_2h, int Ng)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= Ng*Ng*Ng) return;
    int iz = gid%Ng, tmp = gid/Ng, iy = tmp%Ng, ix = tmp/Ng;
    int stride = Ng + 2;
    int ixp = (ix+1)%Ng, ixm = (ix-1+Ng)%Ng;
    int iyp = (iy+1)%Ng, iym = (iy-1+Ng)%Ng;
    int izp = (iz+1)%Ng, izm = (iz-1+Ng)%Ng;
#define POTP(x,y,z) potential[((x)*Ng+(y))*stride+(z)]
    float4 f;
    f.x = -(POTP(ixp,iy,iz) - POTP(ixm,iy,iz)) * inv_2h;
    f.y = -(POTP(ix,iyp,iz) - POTP(ix,iym,iz)) * inv_2h;
    f.z = -(POTP(ix,iy,izp) - POTP(ix,iy,izm)) * inv_2h;
    f.w = 0.0f;
    force_field[gid] = f;
#undef POTP
}

// Two-Poisson helpers: deposit only one species (positive or negative)
// This avoids the ρ_net = ρ+ + ρ- ≈ 0 cancellation problem.
__global__ void cic_deposit_species(const float4* __restrict__ pos, unsigned int N,
                                    float* density, float box_half, float inv_cell, int Ng,
                                    bool positive_species)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    float4 p = pos[gid];
    // Only deposit the requested species
    bool is_positive = (p.w >= 0.0f);
    if (is_positive != positive_species) return;
    float mass = fabsf(p.w);  // unsigned mass for two-Poisson
    float gx = fmodf(fmodf((p.x + box_half)*inv_cell, (float)Ng) + (float)Ng, (float)Ng);
    float gy = fmodf(fmodf((p.y + box_half)*inv_cell, (float)Ng) + (float)Ng, (float)Ng);
    float gz = fmodf(fmodf((p.z + box_half)*inv_cell, (float)Ng) + (float)Ng, (float)Ng);
    int ix=(int)gx, iy=(int)gy, iz=(int)gz;
    float dx=gx-ix, dy=gy-iy, dz=gz-iz;
#pragma unroll
    for (int a=0; a<2; a++) {
        int jx=(ix+a)%Ng; float wx=(a==0)?1.0f-dx:dx;
#pragma unroll
        for (int b=0; b<2; b++) {
            int jy=(iy+b)%Ng; float wy=(b==0)?1.0f-dy:dy;
#pragma unroll
            for (int c=0; c<2; c++) {
                int jz=(iz+c)%Ng; float wz=(c==0)?1.0f-dz:dz;
                atomicAdd(&density[(jx*Ng+jy)*Ng+jz], mass*wx*wy*wz);
            }
        }
    }
}

// Subtract second force field from first: f1 -= f2
__global__ void subtract_force_field(float4* __restrict__ f1, const float4* __restrict__ f2, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    f1[i].x -= f2[i].x;
    f1[i].y -= f2[i].y;
    f1[i].z -= f2[i].z;
}

// ====================================================================
//  SPECTRAL POISSON + GRADIENT (GADGET/PKDGRAV-style)
//
//  For one dimension d, computes F̂_d directly in Fourier space from ρ̂:
//      φ̂   = -4πG × Plummer × MAS × ρ̂ / k²_continuous
//      F̂_d = -i × k_d_continuous × φ̂
//           = 4πGi × k_d × Plummer × MAS × ρ̂ / k²_cont / N³
//
//  Using CONTINUOUS k for both Green's function and gradient gives perfect
//  isotropy in the continuum limit (force magnitude depends only on |k_cont|,
//  not on direction). This is the GADGET2 PM approach.
//
//  The discrete k_eff Green's function (used previously) is consistent with
//  the 7-point Laplacian but inconsistent with the central-difference gradient
//  (off by cos²(πk/N)), introducing residual axial bias. Continuous k + spectral
//  gradient avoids this entirely.
// ====================================================================
__global__ void periodic_spectral_force_kernel(
    const cufftComplex* __restrict__ rho_hat,
    cufftComplex* __restrict__ F_d_hat,
    int N, int dim,           // dim: 0=x, 1=y, 2=z
    float G, float box_size, float eps_pm,
    float r_split)            // P³M Ewald split scale (0 → no smoothing)
{
    int Nk = N/2 + 1, total = N*N*Nk;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int iz  = gid % Nk;
    int tmp = gid / Nk;
    int iy  = tmp % N;
    int ix  = tmp / N;

    int kxi = (ix > N/2) ? ix - N : ix;
    int kyi = (iy > N/2) ? iy - N : iy;
    int kzi = iz;  // R2C: kz ∈ [0, N/2]

    if (kxi == 0 && kyi == 0 && kzi == 0) {
        F_d_hat[gid].x = 0.0f;
        F_d_hat[gid].y = 0.0f;
        return;
    }

    // CONTINUOUS wavenumbers (isotropic in physical k)
    float dk    = 2.0f * M_PI_F / box_size;
    float k_cx  = (float)kxi * dk;
    float k_cy  = (float)kyi * dk;
    float k_cz  = (float)kzi * dk;
    float k2    = k_cx*k_cx + k_cy*k_cy + k_cz*k_cz;
    float k_mag = sqrtf(k2);

    // Plummer in continuous k (physical, isotropic)
    float plummer = expf(-k_mag * eps_pm);

    // MAS deconvolution for CIC: 1/W_CIC² to correct deposit + interpolation
    auto W = [N](int ki) -> float {
        if (ki == 0) return 1.0f;
        float x = M_PI_F * fabsf((float)ki) / (float)N;
        float s = sinf(x) / x;
        return s * s;  // sinc²: CIC window per dimension
    };
    float W_CIC2 = W(kxi) * W(kyi) * W(kzi);
    float W_CIC4 = W_CIC2 * W_CIC2;
    float mas = (W_CIC4 > 2e-3f) ? fminf(1.0f / W_CIC4, 64.0f) : 64.0f;

    // Gradient component in dimension d (continuous k_d)
    float k_d;
    if (dim == 0)      k_d = k_cx;
    else if (dim == 1) k_d = k_cy;
    else               k_d = k_cz;

    // F̂_d = 4πGi × k_d × Plummer × MAS × ρ̂ / k² / V_box
    // Why V_box (not N³)?
    //   ρ_phys = M_per_cell / V_cell (physical density from CIC deposit).
    //   Poisson source = 4πG × ρ_phys → in Fourier divide by V_cell.
    //   cuFFT IFFT introduces N³ multiplier → total: V_cell × N³ = V_box.
    // Without V_cell correction, forces are V_cell ≈ (box/N)³ times too large
    // → explosive evolution. This matches the H&E (open-bdry) Green's function
    // which includes V_cell implicitly via the real-space 1/r kernel sampling.
    // Multiplication by i: (re + i·im) → (-im + i·re)
    float V_box = box_size * box_size * box_size;
    float factor = 4.0f * M_PI_F * G * k_d * plummer * mas / (k2 * V_box);

    // P³M Ewald split: long-range PM gets Gaussian smoothing exp(-k²·r_s²/2).
    if (r_split > 0.0f)
        factor *= expf(-0.5f * k2 * r_split * r_split);

    F_d_hat[gid].x = -factor * rho_hat[gid].y;   // -k_d × Im(ρ̂)
    F_d_hat[gid].y =  factor * rho_hat[gid].x;   // +k_d × Re(ρ̂)
}

// Scatter one real-space force component into d_force_field.{x|y|z}.
// Source is in padded R2C layout (stride = N+2 in z).
__global__ void scatter_force_component(
    const float* __restrict__ F_d_real,
    float4* __restrict__ force_field,
    int N, int dim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N*N*N) return;
    int iz  = gid % N;
    int tmp = gid / N;
    int iy  = tmp % N;
    int ix  = tmp / N;
    int stride = N + 2;
    int src = (ix * N + iy) * stride + iz;
    float v = F_d_real[src];
    if (dim == 0)      force_field[gid].x = v;
    else if (dim == 1) force_field[gid].y = v;
    else               force_field[gid].z = v;
}

// ====================================================================
//  H&E SPECTRAL FORCE (open-boundary version, fully isotropic)
//
//  Pure continuous-k spectral Poisson + gradient (avoids axial artifacts
//  from the discrete real-space Green's function). Correct normalization:
//      factor = 4πG × k_d × V_cell / (k² × N_padded³) × Plummer × MAS
//  Derived from matching the validated periodic spectral formula with
//  the H&E padded grid (V_cell × N_padded³ = V_box_padded).
// ====================================================================
__global__ void h_e_spectral_force_kernel(
    const cufftComplex* __restrict__ rho_hat,
    const cufftComplex* __restrict__ green_hat,   // unused (kept for ABI)
    cufftComplex* __restrict__ F_d_hat,
    int Ng2, int dim,
    float G_grav, float box_inner,
    float r_split)   // P³M Ewald split scale (0 → no smoothing, pure PM)
{
    int Nk = Ng2/2 + 1, total = Ng2*Ng2*Nk;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int iz  = gid % Nk;
    int tmp = gid / Nk;
    int iy  = tmp % Ng2;
    int ix  = tmp / Ng2;

    int kxi = (ix > Ng2/2) ? ix - Ng2 : ix;
    int kyi = (iy > Ng2/2) ? iy - Ng2 : iy;
    int kzi = iz;

    if (kxi == 0 && kyi == 0 && kzi == 0) {
        F_d_hat[gid].x = 0.0f;
        F_d_hat[gid].y = 0.0f;
        return;
    }

    // CIC deconvolution (deposit + interpolation)
    auto W = [Ng2](int ki) -> float {
        if (ki == 0) return 1.0f;
        float x = M_PI_F * fabsf((float)ki) / (float)Ng2;
        float s = sinf(x) / x;
        return s * s;
    };
    float W_CIC2 = W(kxi) * W(kyi) * W(kzi);
    float W_CIC4 = W_CIC2 * W_CIC2;
    float mas = (W_CIC4 > 2e-3f) ? fminf(1.0f / W_CIC4, 64.0f) : 64.0f;

    // Continuous wavenumbers on padded grid (isotropic in physical k)
    float box_padded = 2.0f * box_inner;
    float dk = 2.0f * M_PI_F / box_padded;
    float k_cx = (float)kxi * dk;
    float k_cy = (float)kyi * dk;
    float k_cz = (float)kzi * dk;
    float k2 = k_cx*k_cx + k_cy*k_cy + k_cz*k_cz;
    float k_mag = sqrtf(k2);

    float k_d;
    if (dim == 0)      k_d = k_cx;
    else if (dim == 1) k_d = k_cy;
    else               k_d = k_cz;

    // Plummer in continuous k. Reduced from 1.5×cell to 0.5×cell so that
    // structure-scale modes (100-500 Mpc) are preserved instead of damped to ~0.
    // With box=10R, Ng=256, cell≈39 Mpc → eps_pm≈20 Mpc, damping only at Nyquist.
    int Ng_inner = Ng2 / 2;
    float cell = box_inner / (float)Ng_inner;
    float eps_pm = 0.5f * cell;
    float plummer = expf(-k_mag * eps_pm);

    // Empirical sweet spot: √V_cell / 10 multiplier (~11× baseline).
    // Geometric mean between √V_cell (asymmetric explosion) and √V_cell/100 (too weak).
    float V_cell = cell * cell * cell;
    float scale = sqrtf(V_cell) / 10.0f;
    float V_box = box_padded * box_padded * box_padded;
    float factor = 4.0f * M_PI_F * G_grav * k_d * scale * plummer * mas / (k2 * V_box);

    // P³M Ewald split: long-range PM gets Gaussian smoothing exp(-k²·r_s²/2).
    // The complementary χ(r) filter in sr_janus_kick supplies the short-range part.
    // r_split=0 → no smoothing (pure PM mode).
    if (r_split > 0.0f)
        factor *= expf(-0.5f * k2 * r_split * r_split);

    F_d_hat[gid].x = -factor * rho_hat[gid].y;
    F_d_hat[gid].y =  factor * rho_hat[gid].x;
}

// Scatter force component from padded R2C inner corner to compact Ng³ force field.
// Padded layout: (2Ng) × (2Ng) × (2Ng+2). We read only the inner Ng × Ng × Ng corner.
__global__ void scatter_force_component_open(
    const float* __restrict__ F_d_real,
    float4* __restrict__ force_field,
    int Ng, int dim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= Ng*Ng*Ng) return;
    int iz  = gid % Ng;
    int tmp = gid / Ng;
    int iy  = tmp % Ng;
    int ix  = tmp / Ng;
    int Ng2 = 2 * Ng;
    int stride = Ng2 + 2;
    int src = (ix * Ng2 + iy) * stride + iz;
    float v = F_d_real[src];
    if (dim == 0)      force_field[gid].x = v;
    else if (dim == 1) force_field[gid].y = v;
    else               force_field[gid].z = v;
}

// CIC deposit with periodic wrap (particles always inside the box due to drift wrapping)
__global__ void cic_deposit_periodic(const float4* __restrict__ pos, unsigned int N,
                                      float* density, float box_half, float inv_cell, int Ng)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    float4 p = pos[gid];
    float mass = p.w;
    // Map to [0, Ng) with periodic wrap
    float gx = fmodf(fmodf((p.x + box_half)*inv_cell, (float)Ng) + (float)Ng, (float)Ng);
    float gy = fmodf(fmodf((p.y + box_half)*inv_cell, (float)Ng) + (float)Ng, (float)Ng);
    float gz = fmodf(fmodf((p.z + box_half)*inv_cell, (float)Ng) + (float)Ng, (float)Ng);
    int ix=(int)gx, iy=(int)gy, iz=(int)gz;
    float dx=gx-ix, dy=gy-iy, dz=gz-iz;
#pragma unroll
    for (int a=0; a<2; a++) {
        int jx=(ix+a)%Ng; float wx=(a==0)?1.0f-dx:dx;
#pragma unroll
        for (int b=0; b<2; b++) {
            int jy=(iy+b)%Ng; float wy=(b==0)?1.0f-dy:dy;
#pragma unroll
            for (int c=0; c<2; c++) {
                int jz=(iz+c)%Ng; float wz=(c==0)?1.0f-dz:dz;
                atomicAdd(&density[(jx*Ng+jy)*Ng+jz], mass*wx*wy*wz);
            }
        }
    }
}

// CIC kick with periodic wrap
__global__ void cic_kick_periodic(const float4* __restrict__ pos, float4* __restrict__ vel,
                                   const float4* __restrict__ force_field,
                                   float box_half, float inv_cell, int Ng,
                                   float timestep, float a_inv3, int physicsModel, unsigned int N)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    float4 p = pos[gid];
    float gx=fmodf(fmodf((p.x+box_half)*inv_cell,(float)Ng)+(float)Ng,(float)Ng);
    float gy=fmodf(fmodf((p.y+box_half)*inv_cell,(float)Ng)+(float)Ng,(float)Ng);
    float gz=fmodf(fmodf((p.z+box_half)*inv_cell,(float)Ng)+(float)Ng,(float)Ng);
    int ix=(int)gx, iy=(int)gy, iz=(int)gz;
    float dx=gx-ix, dy=gy-iy, dz=gz-iz;
    float fx=0,fy=0,fz=0;
#pragma unroll
    for (int a=0; a<2; a++) {
        int jx=(ix+a)%Ng; float wx=(a==0)?1.0f-dx:dx;
#pragma unroll
        for (int b=0; b<2; b++) {
            int jy=(iy+b)%Ng; float wy=(b==0)?1.0f-dy:dy;
#pragma unroll
            for (int c=0; c<2; c++) {
                int jz=(iz+c)%Ng; float wz=(c==0)?1.0f-dz:dz;
                float4 ff=force_field[(jx*Ng+jy)*Ng+jz];
                fx+=ff.x*wx*wy*wz; fy+=ff.y*wx*wy*wz; fz+=ff.z*wx*wy*wz;
            }
        }
    }
    float sign_i = (physicsModel==1 && p.w<0.0f) ? -1.0f : 1.0f;
    float4 v=vel[gid];
    float kick=sign_i*a_inv3*timestep;
    v.x+=kick*fx; v.y+=kick*fy; v.z+=kick*fz;
    vel[gid]=v;
}

// ------------------------------------------------------------------
//  Kernel 5: Compute force field from corner Ng³ of padded potential
//  Reads from padded layout (2Ng)×(2Ng)×(2Ng+2), writes compact Ng³.
//  Central differences; upper neighbors may read into padded zero region.
// ------------------------------------------------------------------

__global__ void compute_gradient_open(
    const float* __restrict__ potential_padded,  // (2Ng)×(2Ng)×(2Ng+2) reals
    float4*      __restrict__ force_field,        // Ng³
    float inv_2h,
    int Ng)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= Ng * Ng * Ng) return;

    int iz  = gid % Ng;
    int tmp = gid / Ng;
    int iy  = tmp % Ng;
    int ix  = tmp / Ng;

    int Ng2    = 2 * Ng;
    int stride = Ng2 + 2;

    // Open boundary: allow neighbors to read INTO the padded zero region (x,y up to Ng2-1)
    // This gives the correct gradient at the edge of the inner grid.
    // Clamp LOWER bounds only (no negative index), upper bounds can reach padded region.
    int ixp = ix + 1;                      // may reach Ng (padded) — valid in [0..2Ng-1]
    int ixm = (ix > 0) ? ix - 1 : 0;
    int iyp = iy + 1;
    int iym = (iy > 0) ? iy - 1 : 0;
    int izp = iz + 1;                      // may reach Ng (padded z)
    int izm = (iz > 0) ? iz - 1 : 0;

    // Safety: clamp upper bounds to padded grid size
    if (ixp >= Ng2) ixp = Ng2 - 1;
    if (iyp >= Ng2) iyp = Ng2 - 1;
    if (izp >= Ng2) izp = Ng2 - 1;

#define POT(x,y,z) potential_padded[((x)*Ng2 + (y)) * stride + (z)]

    float4 f;
    f.x = -(POT(ixp,iy,iz) - POT(ixm,iy,iz)) * inv_2h;
    f.y = -(POT(ix,iyp,iz) - POT(ix,iym,iz)) * inv_2h;
    f.z = -(POT(ix,iy,izp) - POT(ix,iy,izm)) * inv_2h;
    f.w = 0.0f;
    force_field[gid] = f;

#undef POT
}

// ------------------------------------------------------------------
//  Kernel 6: CIC force interpolation + velocity kick
//  Interpolates force from grid to particle position (CIC weights).
//  Applies Janus sign convention: sign_i = -1 for negative-mass.
// ------------------------------------------------------------------

__global__ void cic_kick(
    const float4* __restrict__ pos,
    float4*       __restrict__ vel,
    const float4* __restrict__ force_field,
    float         box_half,
    float         inv_cell,
    int           Ng,
    float         timestep,
    float         a_inv3,
    int           physicsModel,
    unsigned int  N)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    float4 p = pos[gid];

    // Map to grid coordinates [0, Ng)
    float gx = (p.x + box_half) * inv_cell;
    float gy = (p.y + box_half) * inv_cell;
    float gz = (p.z + box_half) * inv_cell;

    // Open boundary: particles outside the PM grid receive NO force (drift only).
    // Previously the code clamped to boundary which created artificial inward
    // pull from the boundary cell, causing asymmetric expansion.
    if (gx < 0.0f || gx >= (float)Ng ||
        gy < 0.0f || gy >= (float)Ng ||
        gz < 0.0f || gz >= (float)Ng) return;

    // Integer cell (lower-left corner)
    int ix = (int)gx;
    int iy = (int)gy;
    int iz = (int)gz;

    // Fractional offset
    float dx = gx - (float)ix;
    float dy = gy - (float)iy;
    float dz = gz - (float)iz;

    // CIC interpolation — clamp neighbors (open boundary)
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

#pragma unroll
    for (int a = 0; a < 2; a++) {
        float wx = (a == 0) ? (1.0f - dx) : dx;
        int jx = ix + a; if (jx >= Ng) jx = Ng - 1;
#pragma unroll
        for (int b = 0; b < 2; b++) {
            float wy = (b == 0) ? (1.0f - dy) : dy;
            int jy = iy + b; if (jy >= Ng) jy = Ng - 1;
#pragma unroll
            for (int c = 0; c < 2; c++) {
                float wz = (c == 0) ? (1.0f - dz) : dz;
                int jz = iz + c; if (jz >= Ng) jz = Ng - 1;
                float weight = wx * wy * wz;
                int cell = (jx * Ng + jy) * Ng + jz;
                float4 ff = force_field[cell];
                fx += ff.x * weight;
                fy += ff.y * weight;
                fz += ff.z * weight;
            }
        }
    }

    // Janus convention: negative-mass particles feel reversed force
    float sign_i = (physicsModel == 1 && p.w < 0.0f) ? -1.0f : 1.0f;

    float4 v = vel[gid];
    float kick = sign_i * a_inv3 * timestep;
    v.x += kick * fx;
    v.y += kick * fy;
    v.z += kick * fz;
    vel[gid] = v;
}

// ------------------------------------------------------------------
//  Kernel 7: drift half-step (standalone, for PM DKD path)
//  p_out = p_in + vel * dt_half   (can be in-place: p_in == p_out)
// ------------------------------------------------------------------

__global__ void drift_half_kernel(
    const float4* __restrict__ p_in,
    float4*       __restrict__ p_out,
    const float4* __restrict__ vel,
    unsigned int  N,
    float         dt_half,
    float         box_half)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    float4 p = p_in[gid];
    float4 v = vel[gid];
    p.x += v.x * dt_half;
    p.y += v.y * dt_half;
    p.z += v.z * dt_half;

    // Periodic wrapping
    if (box_half > 0.0f) {
        if      (p.x >  box_half) p.x -= 2.0f * box_half;
        else if (p.x < -box_half) p.x += 2.0f * box_half;
        if      (p.y >  box_half) p.y -= 2.0f * box_half;
        else if (p.y < -box_half) p.y += 2.0f * box_half;
        if      (p.z >  box_half) p.z -= 2.0f * box_half;
        else if (p.z < -box_half) p.z += 2.0f * box_half;
    }

    p_out[gid] = p;
}

// ------------------------------------------------------------------
//  Kernel 8: Hubble drag on peculiar velocity
//  Applies drag only to peculiar component: v_pec = v - H*r
//  v_new = (v - H*r) * (1 - 2*H*dt) + H*r
// ------------------------------------------------------------------

__global__ void hubble_drag_kernel(
    const float4* __restrict__ pos,
    float4*       __restrict__ vel,
    unsigned int  N,
    float         dt,
    float         H)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    if (fabsf(H) < 1.0e-12f) return;

    float4 p = pos[gid];
    float4 v = vel[gid];

    float drag = 1.0f - 2.0f * H * dt;
    float hx = H * p.x;
    float hy = H * p.y;
    float hz = H * p.z;

    v.x = (v.x - hx) * drag + hx;
    v.y = (v.y - hy) * drag + hy;
    v.z = (v.z - hz) * drag + hz;

    vel[gid] = v;
}

// ====================================================================
//  P³M SHORT-RANGE CORRECTION
//  Computes direct Janus N² force for pairs within r_cut using a
//  coarse cell-linked list (O(N × n_neighbors) instead of O(N²)).
//  This captures the sub-PM-cell Janus interactions that the PM
//  Poisson solver misses when ρ_net = ρ+ + ρ- ≈ 0 everywhere.
// ====================================================================

// Phase 1: count particles per coarse cell
__global__ void cll_count(const float4* pos, int* counts,
                           float box_half, float inv_rcell, int Nc, unsigned int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int cx = max(0, min(Nc-1, (int)((pos[i].x + box_half) * inv_rcell)));
    int cy = max(0, min(Nc-1, (int)((pos[i].y + box_half) * inv_rcell)));
    int cz = max(0, min(Nc-1, (int)((pos[i].z + box_half) * inv_rcell)));
    atomicAdd(&counts[(cx*Nc + cy)*Nc + cz], 1);
}

// Phase 3: scatter particles into sorted order
__global__ void cll_scatter(const float4* pos, float4* sorted_pos, int* sorted_orig,
                             const int* cell_start, int* cell_slot,
                             float box_half, float inv_rcell, int Nc, unsigned int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int cx = max(0, min(Nc-1, (int)((pos[i].x + box_half) * inv_rcell)));
    int cy = max(0, min(Nc-1, (int)((pos[i].y + box_half) * inv_rcell)));
    int cz = max(0, min(Nc-1, (int)((pos[i].z + box_half) * inv_rcell)));
    int cell = (cx*Nc + cy)*Nc + cz;
    int slot = atomicAdd(&cell_slot[cell], 1);
    int dst  = cell_start[cell] + slot;
    sorted_pos[dst]  = pos[i];
    sorted_orig[dst] = (int)i;
}

// Phase 4: P³M short-range Janus velocity kick.
// Uses the GADGET2 Ewald split: PM provides the Gaussian-smoothed long-range
// force, SR provides the complementary short-range correction
//   χ(r) = erfc(r/(√2·r_s)) + (r/r_s)·√(2/π)·exp(-r²/(2·r_s²))
// with χ(0)=1 and χ(r)→0 for r >> r_s. The two forces add exactly to the
// unsplit Plummer force (no double-counting, no discontinuity).
__global__ void sr_janus_kick(const float4* sorted_pos, const int* sorted_orig,
                               float4* vel, const int* cell_start, const int* cell_count,
                               float box_half, float box_size,
                               float inv_rcell, int Nc,
                               float r_cut2, float G, float a_inv3, float dt,
                               float soft2, float inv_rs, float inv_sqrt2_rs,
                               bool periodic, int physModel, unsigned int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = sorted_pos[i];
    int cx = max(0, min(Nc-1, (int)((pi.x + box_half) * inv_rcell)));
    int cy = max(0, min(Nc-1, (int)((pi.y + box_half) * inv_rcell)));
    int cz = max(0, min(Nc-1, (int)((pi.z + box_half) * inv_rcell)));

    float fx=0, fy=0, fz=0;
    const float SQRT2_OVER_PI = 0.7978845608028654f;   // √(2/π)

    for (int dx = -1; dx <= 1; dx++) {
        int jx = cx+dx;
        if (periodic) { if (jx < 0) jx += Nc; if (jx >= Nc) jx -= Nc; }
        else          { if (jx < 0 || jx >= Nc) continue; }
        for (int dy = -1; dy <= 1; dy++) {
            int jy = cy+dy;
            if (periodic) { if (jy < 0) jy += Nc; if (jy >= Nc) jy -= Nc; }
            else          { if (jy < 0 || jy >= Nc) continue; }
            for (int dz = -1; dz <= 1; dz++) {
                int jz = cz+dz;
                if (periodic) { if (jz < 0) jz += Nc; if (jz >= Nc) jz -= Nc; }
                else          { if (jz < 0 || jz >= Nc) continue; }
                int cell  = (jx*Nc + jy)*Nc + jz;
                int start = cell_start[cell];
                int cnt   = cell_count[cell];
                for (int k = 0; k < cnt; k++) {
                    if (sorted_orig[start+k] == sorted_orig[i]) continue;
                    float4 pj = sorted_pos[start+k];
                    float rx = pj.x - pi.x;
                    float ry = pj.y - pi.y;
                    float rz = pj.z - pi.z;
                    if (periodic) {
                        // Minimum image (torus): only for periodic_bc
                        if (rx >  box_half) rx -= box_size;
                        if (rx < -box_half) rx += box_size;
                        if (ry >  box_half) ry -= box_size;
                        if (ry < -box_half) ry += box_size;
                        if (rz >  box_half) rz -= box_size;
                        if (rz < -box_half) rz += box_size;
                    }
                    float r2 = rx*rx + ry*ry + rz*rz;
                    if (r2 >= r_cut2) continue;

                    // Janus sign: +1 same-sign pair (attract), -1 opposite (repel).
                    // Use |m_j| because sign_ij already carries the interaction sign.
                    float sign_ij = (physModel==1 && pi.w * pj.w < 0.0f) ? -1.0f : 1.0f;

                    // Bimetric softening: −− pairs use reduced ε (×100 metric scale approximation)
                    // ε_neg = ε × 0.5 — negative masses cluster 3× tighter, sharpening void centers
                    float soft2_eff = soft2;
                    if (physModel == 1 && pi.w < 0.0f && pj.w < 0.0f)
                        soft2_eff = soft2 * 0.25f;  // (×0.5)² = 0.25 for ε²
                    // Softened Plummer force magnitude (without χ filter)
                    float inv_r3  = rsqrtf(r2 + soft2_eff);
                    inv_r3 = inv_r3 * inv_r3 * inv_r3;

                    // GADGET2 short-range filter χ(r): the EXACT complement to
                    // PM's exp(-k²·r_s²/2) Gaussian smoothing. The (small) Plummer
                    // softening only protects r→0; for r>>soft it has no effect.
                    float r    = sqrtf(r2);
                    float u    = r * inv_sqrt2_rs;            // r / (√2 r_s)
                    float gaus = expf(-0.5f * r2 * inv_rs * inv_rs);
                    float chi  = erfcf(u) + r * inv_rs * SQRT2_OVER_PI * gaus;

                    float fmag = sign_ij * fabsf(pj.w) * inv_r3 * chi;
                    fx += fmag * rx;
                    fy += fmag * ry;
                    fz += fmag * rz;
                }
            }
        }
    }

    // No sign_i here: sign_ij already encodes Janus attraction/repulsion.
    float kick = G * a_inv3 * dt;
    int orig = sorted_orig[i];
    atomicAdd(&vel[orig].x, kick * fx);
    atomicAdd(&vel[orig].y, kick * fy);
    atomicAdd(&vel[orig].z, kick * fz);
}

// Host function: cell-linked list build + short-range Ewald-split kick.
// No-op when ctx.sr_enabled = false (PM-only mode).
void pm_janus_sr_kick(PMContext& ctx, const float4* d_pos, float4* d_vel,
                      unsigned int N, float G, float a_inv3, float dt,
                      int physicsModel)
{
    if (!ctx.sr_enabled || !ctx.d_sr_count) return;

    float r_cut     = ctx.sr_r_cut;
    float box_half  = ctx.box_size * 0.5f;
    // Cell-linked list spacing: cell_size = box / Nc (≥ r_cut by construction).
    // Using inv_rcell = Nc / box (not 1/r_cut) means each particle indexes its
    // actual cell, and r_cut2 inside the kernel filters pairs farther than r_cut.
    float inv_rcell = (float)ctx.sr_Nc / ctx.box_size;
    int   Nc        = ctx.sr_Nc;
    int   N_cells   = Nc * Nc * Nc;

    dim3 block(256), grid_N((N+255)/256);

    // Phase 1: count particles per coarse cell
    cudaMemset(ctx.d_sr_count, 0, N_cells * sizeof(int));
    cll_count<<<grid_N, block>>>(d_pos, ctx.d_sr_count, box_half, inv_rcell, Nc, N);

    // Phase 2: GPU prefix sum (thrust) — exclusive scan of counts → cell starts.
    thrust::exclusive_scan(thrust::device,
                           ctx.d_sr_count,
                           ctx.d_sr_count + N_cells,
                           ctx.d_sr_start);

    // Phase 3: scatter particles into sorted order. Re-count first because
    // cll_scatter needs the original counts and we've not touched d_sr_count
    // since (exclusive_scan reads from d_sr_count and writes to d_sr_start, so
    // d_sr_count is still valid — no re-count needed).
    cudaMemset(ctx.d_sr_slot, 0, N_cells * sizeof(int));
    cll_scatter<<<grid_N, block>>>(d_pos, ctx.d_sr_sorted, ctx.d_sr_orig,
                                    ctx.d_sr_start, ctx.d_sr_slot,
                                    box_half, inv_rcell, Nc, N);

    // Phase 4: short-range Janus kick with GADGET χ(r) filter
    float r_cut2       = r_cut * r_cut;
    float soft2        = ctx.softening * ctx.softening;
    float inv_rs       = 1.0f / ctx.sr_r_split;
    float inv_sqrt2_rs = inv_rs * 0.7071067811865476f;   // 1/(√2 r_s)
    sr_janus_kick<<<grid_N, block>>>(ctx.d_sr_sorted, ctx.d_sr_orig,
                                      d_vel, ctx.d_sr_start, ctx.d_sr_count,
                                      box_half, ctx.box_size, inv_rcell, Nc,
                                      r_cut2, G, a_inv3, dt,
                                      soft2, inv_rs, inv_sqrt2_rs,
                                      ctx.periodic_bc, physicsModel, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("[P3M] SR kick error: %s\n", cudaGetErrorString(err));
}

// ------------------------------------------------------------------
//  Host function: pm_init
// ------------------------------------------------------------------

void pm_init(PMContext& ctx, int Ng, float box_size, float softening,
             bool periodic_bc, bool sr_enabled)
{
    ctx.Ng          = Ng;
    ctx.box_size    = box_size;
    ctx.softening   = softening;
    ctx.periodic_bc = periodic_bc;
    ctx.sr_enabled  = sr_enabled;
    ctx.initialized = true;

    int N3  = Ng * Ng * Ng;
    int Ng2 = 2 * Ng;
    int Nc  = Ng2 * Ng2 * (Ng2 / 2 + 1);  // complex elements for padded R2C

    cudaMalloc(&ctx.d_density,      N3 * sizeof(float));
    cudaMalloc(&ctx.d_density2,     N3 * sizeof(float));   // two-Poisson second species
    cudaMalloc(&ctx.d_padded,       Nc * sizeof(cufftComplex));
    cudaMalloc(&ctx.d_green_hat,    Nc * sizeof(cufftComplex));
    cudaMalloc(&ctx.d_rho_scratch,  Nc * sizeof(cufftComplex));  // H&E spectral grad scratch
    cudaMalloc(&ctx.d_force_field,  N3 * sizeof(float4));
    cudaMalloc(&ctx.d_force_field2, N3 * sizeof(float4));  // two-Poisson second species

    cufftResult r;
    r = cufftPlan3d(&ctx.plan_r2c, Ng2, Ng2, Ng2, CUFFT_R2C);
    if (r != CUFFT_SUCCESS)
        printf("[PM] cufftPlan3d R2C failed: %d\n", (int)r);

    r = cufftPlan3d(&ctx.plan_c2r, Ng2, Ng2, Ng2, CUFFT_C2R);
    if (r != CUFFT_SUCCESS)
        printf("[PM] cufftPlan3d C2R failed: %d\n", (int)r);

    // Pre-compute the Hockney-Eastwood real-space kernel G(r)=-1/r/(2Ng)³,
    // then FFT it into d_green_hat (G_grav multiplied in per-step).
    float* d_green_real = nullptr;
    cudaMalloc(&d_green_real, Nc * sizeof(cufftComplex));
    cudaMemset(d_green_real, 0,  Nc * sizeof(cufftComplex));

    float cell_size = box_size / (float)Ng;
    {
        int   total_g = Ng2 * Ng2 * Ng2;
        dim3  blk(256);
        dim3  grd((total_g + 255) / 256);
        compute_green_function<<<grd, blk>>>(d_green_real, cell_size, softening, Ng2);
    }

    r = cufftExecR2C(ctx.plan_r2c, d_green_real, ctx.d_green_hat);
    if (r != CUFFT_SUCCESS)
        printf("[PM] Green FFT R2C failed: %d\n", (int)r);

    cudaDeviceSynchronize();
    cudaFree(d_green_real);

    if (periodic_bc) {
        // Periodic mode: create Ng³ FFT plans (reuse d_padded buffer)
        cufftResult rp;
        rp = cufftPlan3d(&ctx.plan_r2c_per, Ng, Ng, Ng, CUFFT_R2C);
        if (rp != CUFFT_SUCCESS) printf("[PM] Periodic R2C plan failed: %d\n", (int)rp);
        rp = cufftPlan3d(&ctx.plan_c2r_per, Ng, Ng, Ng, CUFFT_C2R);
        if (rp != CUFFT_SUCCESS) printf("[PM] Periodic C2R plan failed: %d\n", (int)rp);
        printf("[PM] Periodic cube: Ng=%d, box=%.0f Mpc, cell=%.1f Mpc, VRAM=%.0f MB\n",
               Ng, box_size, box_size/(float)Ng,
               (float)(Nc*sizeof(cufftComplex) + N3*(sizeof(float)+sizeof(float4))) / (1024.f*1024.f));
    } else {
        float vram_mb = (float)(2 * Nc * sizeof(cufftComplex)
                               + N3 * (sizeof(float) + sizeof(float4))) / (1024.f * 1024.f);
        printf("[PM] H&E open-boundary: Ng=%d (padded %d), box=%.0f Mpc, eps=%.0f Mpc, VRAM=%.0f MB\n",
               Ng, Ng2, box_size, softening, vram_mb);
    }

    // P³M Ewald split (GADGET2-style).
    //   r_split = 1 × PM cell  → smoothing matches natural PM resolution limit
    //   r_cut   = 4.5 × r_split → χ(r_cut)/χ(0) ≈ 1e-3, beyond cutoff is negligible
    //   sr_Nc   = ceil(box / r_cut) → cell_size ≥ r_cut, only 27 neighbors needed
    // For box=10R=10000, Ng=256: cell=39 → r_s=39, r_cut=175 → sr_Nc≈57 → ~185k cells
    // (avg ~3 particles/cell for N=500k → ~36M pair ops, ~8× faster than r_split=2×cell).
    // Smaller r_split = finer resolution AND fewer SR pairs (cells×N²/cells³ = N²/cells²).
    float pm_cell    = box_size / (float)Ng;
    ctx.sr_r_split   = 1.0f * pm_cell;
    ctx.sr_r_cut     = 4.5f * ctx.sr_r_split;
    ctx.sr_Nc        = (int)ceilf(box_size / ctx.sr_r_cut);
    if (ctx.sr_Nc < 3) ctx.sr_Nc = 3;  // need at least 3 cells per dim for 27-stencil

    ctx.d_sr_count = nullptr;
    ctx.d_sr_start = nullptr;
    ctx.d_sr_slot  = nullptr;
    ctx.d_sr_sorted = nullptr;
    ctx.d_sr_orig   = nullptr;

    if (sr_enabled) {
        int sr_cells = ctx.sr_Nc * ctx.sr_Nc * ctx.sr_Nc;
        cudaMalloc(&ctx.d_sr_count,  sr_cells * sizeof(int));
        cudaMalloc(&ctx.d_sr_start,  sr_cells * sizeof(int));
        cudaMalloc(&ctx.d_sr_slot,   sr_cells * sizeof(int));
        // Sorted arrays sized for max expected N (padded grid is generous upper bound)
        int max_N = N3 > 0 ? N3 : 1048576;
        cudaMalloc(&ctx.d_sr_sorted, max_N * sizeof(float4));
        cudaMalloc(&ctx.d_sr_orig,   max_N * sizeof(int));
        printf("[P3M] Ewald split: r_split=%.0f Mpc, r_cut=%.0f Mpc, Nc=%d (cells=%d)\n",
               ctx.sr_r_split, ctx.sr_r_cut, ctx.sr_Nc, sr_cells);
    } else {
        printf("[PM] SR (P3M) disabled — pure PM, no Ewald smoothing.\n");
    }
}

// ------------------------------------------------------------------
//  Host function: pm_free
// ------------------------------------------------------------------

void pm_free(PMContext& ctx)
{
    if (!ctx.initialized) return;

    if (ctx.d_density)      { cudaFree(ctx.d_density);      ctx.d_density      = nullptr; }
    if (ctx.d_density2)     { cudaFree(ctx.d_density2);     ctx.d_density2     = nullptr; }
    if (ctx.d_padded)       { cudaFree(ctx.d_padded);       ctx.d_padded       = nullptr; }
    if (ctx.d_green_hat)    { cudaFree(ctx.d_green_hat);    ctx.d_green_hat    = nullptr; }
    if (ctx.d_rho_scratch)  { cudaFree(ctx.d_rho_scratch);  ctx.d_rho_scratch  = nullptr; }
    if (ctx.d_force_field)  { cudaFree(ctx.d_force_field);  ctx.d_force_field  = nullptr; }
    if (ctx.d_force_field2) { cudaFree(ctx.d_force_field2); ctx.d_force_field2 = nullptr; }
    if (ctx.d_sr_count)    { cudaFree(ctx.d_sr_count);    ctx.d_sr_count    = nullptr; }
    if (ctx.d_sr_start)    { cudaFree(ctx.d_sr_start);    ctx.d_sr_start    = nullptr; }
    if (ctx.d_sr_slot)     { cudaFree(ctx.d_sr_slot);     ctx.d_sr_slot     = nullptr; }
    if (ctx.d_sr_sorted)   { cudaFree(ctx.d_sr_sorted);   ctx.d_sr_sorted   = nullptr; }
    if (ctx.d_sr_orig)     { cudaFree(ctx.d_sr_orig);     ctx.d_sr_orig     = nullptr; }

    cufftDestroy(ctx.plan_r2c);
    cufftDestroy(ctx.plan_c2r);
    if (ctx.periodic_bc) {
        cufftDestroy(ctx.plan_r2c_per);
        cufftDestroy(ctx.plan_c2r_per);
    }

    ctx.initialized = false;
    printf("[PM] Freed.\n");
}

// ------------------------------------------------------------------
//  Host function: pm_force_step
//  Full zero-padded open-boundary PM pipeline:
//  deposit -> embed -> FFT -> Poisson -> iFFT -> gradient -> kick
//  This performs only the FORCE KICK part of leapfrog DKD.
//  Drifts and Hubble drag are handled separately in simulation.cpp.
// ------------------------------------------------------------------

bool pm_force_step(
    PMContext&    ctx,
    const float4* d_pos,
    float4*       d_vel,
    unsigned int  N,
    float         timestep,
    float         G,
    float         a_inv3,
    int           physicsModel)
{
    int   Ng        = ctx.Ng;
    int   Ng2       = 2 * Ng;
    int   Nc        = Ng2 * Ng2 * (Ng2 / 2 + 1);
    float box_half  = ctx.box_size * 0.5f;
    float cell_size = ctx.box_size / (float)Ng;
    float inv_cell  = 1.0f / cell_size;

    dim3 block(256);
    dim3 grid_N((N + 255) / 256);

    // 1. Zero compact density and full padded array
    cudaMemset(ctx.d_density, 0, (size_t)Ng * Ng * Ng * sizeof(float));
    cudaMemset(ctx.d_padded,  0, (size_t)Nc * sizeof(cufftComplex));

    // 2. CIC deposit — periodic wrap or open-boundary skip
    if (ctx.periodic_bc)
        cic_deposit_periodic<<<grid_N, block>>>(d_pos, N, ctx.d_density, box_half, inv_cell, Ng);
    else
        cic_deposit<<<grid_N, block>>>(d_pos, N, ctx.d_density, box_half, inv_cell, Ng);

    float* d_padded_real = reinterpret_cast<float*>(ctx.d_padded);
    dim3 grid_g3((Ng*Ng*Ng + 255) / 256);

    if (ctx.periodic_bc) {
        // ---- SINGLE-POISSON + SPECTRAL GRADIENT (GADGET2-style) ----
        // Mathematically equivalent to two-Poisson (Poisson is linear), but uses
        // a single signed density ρ_net = ρ+ − ρ- (already given by signed CIC
        // deposit). Then computes the force in Fourier space with continuous k:
        //   F̂_d = -i × k_d × φ̂ = 4πGi × k_d × ρ̂ / k²_cont
        // Three separate C2R IFFTs (one per force component) give true isotropy.
        //
        // For Janus antiplasma: cic_kick_periodic uses sign_i = -1 for - particles,
        // so they feel -F (attracted to −, repelled by +). ✓
        //
        // FFT count: 1 R2C + 3 C2R = same as previous Two-Poisson (2 R2C + 2 C2R).
        int Nc_per = Ng * Ng * (Ng/2 + 1);
        // Plummer eps_pm = 1.5 × cell.
        // This is the GADGET/PKDGRAV recommendation: ~1-2 cells damps Nyquist
        // (k×eps = π at Nyquist → exp(-π) ≈ 4%) while preserving structure
        // scales 4+ cells away (k×eps = π/4 → exp(-π/4) ≈ 46%). With Ng=128
        // and box=2000 Mpc, this gives eps_pm=24 Mpc — small enough to preserve
        // the 50–200 Mpc alveolar bubble scales while suppressing the cell-scale
        // axial bias that creates the 6-jet artifact.
        float eps_pm = 1.5f * cell_size;

        // 1. Signed CIC deposit (already done at line 904 → ctx.d_density)
        // 2. Embed and R2C to get ρ̂_net in d_padded
        cudaMemset(d_padded_real, 0, (size_t)Ng*Ng*(Ng+2) * sizeof(float));
        embed_density_periodic<<<grid_g3, block>>>(ctx.d_density, d_padded_real, Ng);
        {
            cufftResult r = cufftExecR2C(ctx.plan_r2c_per, d_padded_real, ctx.d_padded);
            if (r != CUFFT_SUCCESS) { printf("[PM] Per R2C: %d\n", (int)r); return false; }
        }

        // 3. Save ρ̂ to scratch (d_green_hat is unused in periodic mode and large enough)
        cudaMemcpyAsync(ctx.d_green_hat, ctx.d_padded,
                        (size_t)Nc_per * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

        // P³M: when SR enabled, PM gets additional exp(-k²·r_split²/2) smoothing.
        // SR direct force fills in the short range. When disabled, r_split=0 → pure PM.
        float r_split = ctx.sr_enabled ? ctx.sr_r_split : 0.0f;

        // 4. For each force component: spectral gradient → C2R → scatter
        for (int dim = 0; dim < 3; dim++) {
            periodic_spectral_force_kernel<<<(Nc_per+255)/256, block>>>(
                ctx.d_green_hat, ctx.d_padded, Ng, dim, G, ctx.box_size, eps_pm, r_split);
            cufftResult r = cufftExecC2R(ctx.plan_c2r_per, ctx.d_padded, d_padded_real);
            if (r != CUFFT_SUCCESS) { printf("[PM] Per C2R[%d]: %d\n", dim, (int)r); return false; }
            scatter_force_component<<<grid_g3, block>>>(d_padded_real, ctx.d_force_field, Ng, dim);
        }

        // 5. Kick particles (sign_i handles the Janus convention for - particles)
        cic_kick_periodic<<<grid_N, block>>>(d_pos, d_vel, ctx.d_force_field,
                                              box_half, inv_cell, Ng,
                                              timestep, a_inv3, physicsModel, N);
    } else {
        // ---- OPEN-BOUNDARY PATH (H&E, 2Ng³) with SPECTRAL GRADIENT ----
        // Uses 1 R2C + 3 C2R FFTs (vs 1+1 with old real-space central-diff).
        // Spectral gradient with continuous k eliminates the axial bias that
        // central differences create — important for open-boundary expansion
        // where particles fly outward and any axial preference becomes visible.
        embed_density<<<grid_g3, block>>>(ctx.d_density, d_padded_real, Ng);

        cufftResult r = cufftExecR2C(ctx.plan_r2c, d_padded_real, ctx.d_padded);
        if (r != CUFFT_SUCCESS) { printf("[PM] R2C: %d\n", (int)r); return false; }

        // Save rho_hat to scratch (preserved across the 3 spectral gradient passes)
        cudaMemcpyAsync(ctx.d_rho_scratch, ctx.d_padded,
                        (size_t)Nc * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

        // P³M: when SR enabled, PM gets additional exp(-k²·r_split²/2) smoothing.
        // SR direct force fills in the short range. When disabled, r_split=0 → pure PM.
        float r_split = ctx.sr_enabled ? ctx.sr_r_split : 0.0f;

        // For each force component: spectral gradient + IFFT + scatter
        for (int dim = 0; dim < 3; dim++) {
            h_e_spectral_force_kernel<<<(Nc+255)/256, block>>>(
                ctx.d_rho_scratch, ctx.d_green_hat, ctx.d_padded,
                Ng2, dim, G, ctx.box_size, r_split);
            r = cufftExecC2R(ctx.plan_c2r, ctx.d_padded, d_padded_real);
            if (r != CUFFT_SUCCESS) { printf("[PM] C2R[%d]: %d\n", dim, (int)r); return false; }
            scatter_force_component_open<<<grid_g3, block>>>(
                d_padded_real, ctx.d_force_field, Ng, dim);
        }

        cic_kick<<<grid_N, block>>>(d_pos, d_vel, ctx.d_force_field,
                                     box_half, inv_cell, Ng,
                                     timestep, a_inv3, physicsModel, N);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[PM] CUDA error: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

// ------------------------------------------------------------------
//  Kernels 9 & 10: CM re-centering (velocity + position)
//  In the Janus model the signed-mass kick breaks arithmetic
//  momentum conservation, causing a lateral CM drift. These
//  kernels subtract the arithmetic mean vel/pos each step.
// ------------------------------------------------------------------

// Phase 1: reduce N float4 to per-block (x,y,z) sums
__global__ void block_vel_sum(
    const float4* __restrict__ vel,
    unsigned int N,
    float* __restrict__ bx,   // block sum arrays (3 separate for alignment)
    float* __restrict__ by,
    float* __restrict__ bz)
{
    extern __shared__ float3 sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    sdata[tid] = {0.0f, 0.0f, 0.0f};
    if (gid < N) sdata[tid] = {vel[gid].x, vel[gid].y, vel[gid].z};
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid].x += sdata[tid + s].x;
            sdata[tid].y += sdata[tid + s].y;
            sdata[tid].z += sdata[tid + s].z;
        }
        __syncthreads();
    }
    if (tid == 0) { bx[blockIdx.x] = sdata[0].x; by[blockIdx.x] = sdata[0].y; bz[blockIdx.x] = sdata[0].z; }
}

__global__ void block_pos_sum(
    const float4* __restrict__ pos,
    unsigned int N,
    float* __restrict__ bx, float* __restrict__ by, float* __restrict__ bz)
{
    extern __shared__ float3 sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    sdata[tid] = {0.0f, 0.0f, 0.0f};
    if (gid < N) sdata[tid] = {pos[gid].x, pos[gid].y, pos[gid].z};
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid].x += sdata[tid + s].x;
            sdata[tid].y += sdata[tid + s].y;
            sdata[tid].z += sdata[tid + s].z;
        }
        __syncthreads();
    }
    if (tid == 0) { bx[blockIdx.x] = sdata[0].x; by[blockIdx.x] = sdata[0].y; bz[blockIdx.x] = sdata[0].z; }
}

// Phase 2: subtract a constant (dx,dy,dz) from every element
__global__ void subtract_float4_xyz(float4* arr, unsigned int N, float dx, float dy, float dz) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    arr[gid].x -= dx;
    arr[gid].y -= dy;
    arr[gid].z -= dz;
}

// Helper: GPU block-sum → CPU sum → subtract mean
static void gpu_subtract_mean(
    float4* d_arr, unsigned int N, bool is_pos,
    const float4* d_ref_for_pos)   // only used when is_pos=true to read same buf
{
    if (N == 0) return;
    const unsigned int BLK = 256;
    unsigned int nb = (N + BLK - 1) / BLK;

    float *d_bx, *d_by, *d_bz;
    cudaMalloc(&d_bx, nb * sizeof(float));
    cudaMalloc(&d_by, nb * sizeof(float));
    cudaMalloc(&d_bz, nb * sizeof(float));

    size_t shm = BLK * sizeof(float3);
    if (is_pos)
        block_pos_sum<<<nb, BLK, shm>>>(d_ref_for_pos, N, d_bx, d_by, d_bz);
    else
        block_vel_sum<<<nb, BLK, shm>>>(d_arr, N, d_bx, d_by, d_bz);

    std::vector<float> hbx(nb), hby(nb), hbz(nb);
    cudaMemcpy(hbx.data(), d_bx, nb * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hby.data(), d_by, nb * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hbz.data(), d_bz, nb * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_bx); cudaFree(d_by); cudaFree(d_bz);

    double sx = 0, sy = 0, sz = 0;
    for (unsigned int i = 0; i < nb; i++) { sx += hbx[i]; sy += hby[i]; sz += hbz[i]; }
    float mx = (float)(sx / N), my = (float)(sy / N), mz = (float)(sz / N);

    subtract_float4_xyz<<<(N + BLK - 1) / BLK, BLK>>>(d_arr, N, mx, my, mz);
}

// Public launch wrappers
void launch_recenter_velocities(float4* d_vel, unsigned int N) {
    gpu_subtract_mean(d_vel, N, false, nullptr);
}

void launch_recenter_positions(float4* d_pos, unsigned int N) {
    gpu_subtract_mean(d_pos, N, true, d_pos);
}

// ------------------------------------------------------------------
//  Host launch wrappers for drift and drag kernels
// ------------------------------------------------------------------

void launch_drift_half(
    const float4* p_in, float4* p_out, const float4* vel,
    unsigned int N, float dt_half, float box_half,
    dim3 grid, dim3 block)
{
    drift_half_kernel<<<grid, block>>>(p_in, p_out, vel, N, dt_half, box_half);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("[PM] drift_half error: %s\n", cudaGetErrorString(err));
}

void launch_hubble_drag(
    const float4* pos, float4* vel,
    unsigned int N, float dt, float H,
    dim3 grid, dim3 block)
{
    hubble_drag_kernel<<<grid, block>>>(pos, vel, N, dt, H);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("[PM] hubble_drag error: %s\n", cudaGetErrorString(err));
}

// ------------------------------------------------------------------
//  Crash guard: sanitize particles each step so the tree never sees
//  NaN/Inf coordinates (which hang the LBVH build -> freeze -> driver
//  reset). Resets non-finite particles to a safe state and caps runaway
//  positions/velocities. Bounds are huge (>> any legitimate value) so
//  normal dynamics are untouched; only true blow-ups are clamped.
//  Preserves .w (mass / species).
// ------------------------------------------------------------------
__global__ void sanitize_kernel(
    float4* __restrict__ pos, float4* __restrict__ vel,
    unsigned int N, float posBound, float vMax)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    float4 p = pos[gid];
    float4 v = vel[gid];

    // Non-finite -> reset that particle (zero velocity, origin if pos is bad).
    bool badP = !isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z);
    bool badV = !isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z);
    if (badP) { p.x = 0.0f; p.y = 0.0f; p.z = 0.0f; }
    if (badP || badV) { v.x = 0.0f; v.y = 0.0f; v.z = 0.0f; }

    // Cap runaway positions.
    if (posBound > 0.0f) {
        p.x = fminf(fmaxf(p.x, -posBound), posBound);
        p.y = fminf(fmaxf(p.y, -posBound), posBound);
        p.z = fminf(fmaxf(p.z, -posBound), posBound);
    }
    // Cap velocity magnitude (stops the singularity runaway before it -> NaN).
    if (vMax > 0.0f) {
        float v2 = v.x*v.x + v.y*v.y + v.z*v.z;
        if (v2 > vMax*vMax) {
            float s = vMax * rsqrtf(v2);
            v.x *= s; v.y *= s; v.z *= s;
        }
    }

    pos[gid] = p;
    vel[gid] = v;
}

void launch_sanitize(
    float4* pos, float4* vel,
    unsigned int N, float posBound, float vMax,
    dim3 grid, dim3 block)
{
    sanitize_kernel<<<grid, block>>>(pos, vel, N, posBound, vMax);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("[Sanitize] error: %s\n", cudaGetErrorString(err));
}

// ------------------------------------------------------------------
//  Adaptive-dt helper: GPU reduction of max |acceleration|.
//  Replaces the per-step full readback (16 MB) + 1M-element CPU loop by a
//  block reduction; only the (tiny) per-block maxima are copied back.
//  256 threads/block (power of 2, required by the tree-style reduction).
// ------------------------------------------------------------------
__global__ void reduce_max_accel_kernel(
    const float4* __restrict__ acc, unsigned int N, float* __restrict__ blockMax)
{
    __shared__ float redSdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + tid;
    float m = 0.0f;
    if (i < N) {
        float4 a = acc[i];
        m = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
    }
    redSdata[tid] = m;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) redSdata[tid] = fmaxf(redSdata[tid], redSdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) blockMax[blockIdx.x] = redSdata[0];
}

void launch_reduce_max_accel(
    const float4* acc, unsigned int N, float* blockMax, int nBlocks)
{
    dim3 block(256);
    dim3 grid((unsigned int)nBlocks);
    size_t shm = 256 * sizeof(float);
    reduce_max_accel_kernel<<<grid, block, shm>>>(acc, N, blockMax);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("[ReduceMax] error: %s\n", cudaGetErrorString(err));
}

// ------------------------------------------------------------------
//  Force subcycling helpers.
//  After a full tree kick (v += a·dt), derive the effective acceleration
//  a = (v_new - v_old)/dt and store it; on the cheap sub-steps, re-apply
//  v += a_stored·dt without rebuilding/traversing the tree. Valid when the
//  force changes slowly (frozen / late phase).
// ------------------------------------------------------------------
__global__ void accel_from_dv_kernel(
    const float4* __restrict__ vNew, const float4* __restrict__ vOld,
    float4* __restrict__ accel, unsigned int N, float invDt)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 a;
    a.x = (vNew[i].x - vOld[i].x) * invDt;
    a.y = (vNew[i].y - vOld[i].y) * invDt;
    a.z = (vNew[i].z - vOld[i].z) * invDt;
    a.w = 0.0f;
    accel[i] = a;
}

void launch_accel_from_dv(
    const float4* vNew, const float4* vOld, float4* accel,
    unsigned int N, float dt, dim3 grid, dim3 block)
{
    float invDt = (dt != 0.0f) ? 1.0f / dt : 0.0f;
    accel_from_dv_kernel<<<grid, block>>>(vNew, vOld, accel, N, invDt);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("[AccelFromDv] error: %s\n", cudaGetErrorString(err));
}

__global__ void apply_accel_kick_kernel(
    float4* __restrict__ vel, const float4* __restrict__ accel,
    unsigned int N, float dt)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    vel[i].x += accel[i].x * dt;
    vel[i].y += accel[i].y * dt;
    vel[i].z += accel[i].z * dt;
}

void launch_apply_accel_kick(
    float4* vel, const float4* accel, unsigned int N, float dt,
    dim3 grid, dim3 block)
{
    apply_accel_kick_kernel<<<grid, block>>>(vel, accel, N, dt);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("[ApplyAccelKick] error: %s\n", cudaGetErrorString(err));
}

// ------------------------------------------------------------------
//  Anti global-bubble ("Jeans swindle") — remove the spherically-averaged
//  radial acceleration profile, per species, about the origin.
//
//  A finite Janus sphere has a deep global potential well at its centre: the
//  negative cloud self-contracts into ONE central blob (monopole mode) instead
//  of fragmenting into the soap-bubble web. That smooth radial infall is the
//  boundary artifact a periodic box wouldn't have. We bin particles by radius
//  (per species), average a·r̂ in each shell, and subtract that mean radial
//  component — leaving only the non-radial fluctuations that grow the web.
// ------------------------------------------------------------------
#define MONO_NBINS 24   // coquilles grossieres : retire seulement l'infall global lisse, pas la dynamique radiale a l'echelle des bulles
static float* g_mono_rmax  = nullptr;   // [1]   global max radius (atomic via int)
static float* g_mono_sumar = nullptr;   // [2*NBINS] sum of a·r̂ per (species,shell)
static float* g_mono_cnt   = nullptr;   // [2*NBINS] particle count per (species,shell)

static void mono_ensure_buffers()
{
    if (g_mono_rmax) return;
    cudaMalloc(&g_mono_rmax,  sizeof(float));
    cudaMalloc(&g_mono_sumar, sizeof(float) * 2 * MONO_NBINS);
    cudaMalloc(&g_mono_cnt,   sizeof(float) * 2 * MONO_NBINS);
}

__global__ void mono_rmax_kernel(const float4* __restrict__ pos, unsigned int N,
                                 float* __restrict__ rmax)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 p = pos[i];
    float r = sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);
    atomicMax((int*)rmax, __float_as_int(r));   // r >= 0 -> int bits monotonic
}

__global__ void mono_accum_kernel(const float4* __restrict__ pos,
                                  const float4* __restrict__ acc, unsigned int N,
                                  const float* __restrict__ rmax,
                                  float* __restrict__ sumar, float* __restrict__ cnt)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float rm = *rmax;
    if (rm <= 1e-12f) return;
    float4 p = pos[i];
    float4 a = acc[i];
    float r = sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);
    if (r < 1e-12f) return;
    float inv_r = 1.0f / r;
    float ar = (a.x*p.x + a.y*p.y + a.z*p.z) * inv_r;   // a·r̂
    int   s = (p.w < 0.0f) ? 1 : 0;
    int   b = (int)((r / rm) * MONO_NBINS);
    if (b >= MONO_NBINS) b = MONO_NBINS - 1;
    atomicAdd(&sumar[s*MONO_NBINS + b], ar);
    atomicAdd(&cnt  [s*MONO_NBINS + b], 1.0f);
}

__global__ void mono_subtract_kernel(const float4* __restrict__ pos,
                                     float4* __restrict__ acc, unsigned int N,
                                     const float* __restrict__ rmax,
                                     const float* __restrict__ sumar,
                                     const float* __restrict__ cnt, float frac)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float rm = *rmax;
    if (rm <= 1e-12f) return;
    float4 p = pos[i];
    float r = sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);
    if (r < 1e-12f) return;
    int   s = (p.w < 0.0f) ? 1 : 0;
    int   b = (int)((r / rm) * MONO_NBINS);
    if (b >= MONO_NBINS) b = MONO_NBINS - 1;
    float c = cnt[s*MONO_NBINS + b];
    if (c < 1.0f) return;
    float meanAr = sumar[s*MONO_NBINS + b] / c;   // mean radial accel in this shell
    float k = frac * meanAr / r;                  // (meanAr) * r̂ , r̂ = p/r
    acc[i].x -= k * p.x;
    acc[i].y -= k * p.y;
    acc[i].z -= k * p.z;
}

void launch_remove_radial_monopole(
    float4* accel, const float4* pos, unsigned int N, float frac,
    dim3 grid, dim3 block)
{
    mono_ensure_buffers();
    cudaMemset(g_mono_rmax,  0, sizeof(float));
    cudaMemset(g_mono_sumar, 0, sizeof(float) * 2 * MONO_NBINS);
    cudaMemset(g_mono_cnt,   0, sizeof(float) * 2 * MONO_NBINS);
    mono_rmax_kernel    <<<grid, block>>>(pos, N, g_mono_rmax);
    mono_accum_kernel   <<<grid, block>>>(pos, accel, N, g_mono_rmax, g_mono_sumar, g_mono_cnt);
    mono_subtract_kernel<<<grid, block>>>(pos, accel, N, g_mono_rmax, g_mono_sumar, g_mono_cnt, frac);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("[RemoveMonopole] error: %s\n", cudaGetErrorString(err));
}

__global__ void v_from_base_accel_kernel(
    float4* __restrict__ vel, const float4* __restrict__ vBase,
    const float4* __restrict__ accel, unsigned int N, float dt)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 b = vBase[i];
    float4 a = accel[i];
    vel[i].x = b.x + a.x * dt;
    vel[i].y = b.y + a.y * dt;
    vel[i].z = b.z + a.z * dt;
    vel[i].w = b.w;
}

void launch_v_from_base_accel(
    float4* vel, const float4* vBase, const float4* accel,
    unsigned int N, float dt, dim3 grid, dim3 block)
{
    v_from_base_accel_kernel<<<grid, block>>>(vel, vBase, accel, N, dt);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("[VFromBaseAccel] error: %s\n", cudaGetErrorString(err));
}
