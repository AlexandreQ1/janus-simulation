#include "universe_generator.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ *
 *  Cosmological Initial Conditions Generator
 *  - Power spectrum P(k) with Eisenstein-Hu transfer function
 *  - Linear growth factor D(a)
 *  - Zel'dovich approximation for displacements
 *  - Super-comoving coordinates (Martel & Shapiro 1998)
 * ------------------------------------------------------------------ */

// ---- simple 1D complex FFT (Cooley-Tukey, iterative, in-place) ----
struct Cplx { float re, im; };

static void fft1d(Cplx* data, int n, bool inverse) {
    // bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { Cplx tmp = data[i]; data[i] = data[j]; data[j] = tmp; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2 * (float)M_PI / len * (inverse ? -1 : 1);
        float wlen_c = cosf(ang), wlen_s = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float w_c = 1.0f, w_s = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                Cplx u = data[i + j];
                Cplx v = { w_c * data[i + j + len / 2].re - w_s * data[i + j + len / 2].im,
                           w_c * data[i + j + len / 2].im + w_s * data[i + j + len / 2].re };
                data[i + j]           = { u.re + v.re, u.im + v.im };
                data[i + j + len / 2] = { u.re - v.re, u.im - v.im };
                float nw_c = w_c * wlen_c - w_s * wlen_s;
                float nw_s = w_c * wlen_s + w_s * wlen_c;
                w_c = nw_c; w_s = nw_s;
            }
        }
    }
    if (inverse) {
        float inv_n = 1.0f / (float)n;
        for (int i = 0; i < n; i++) { data[i].re *= inv_n; data[i].im *= inv_n; }
    }
}

// ---- 3D FFT : transform along x, then y, then z ----
static void fft3d(Cplx* cube, int nx, int ny, int nz, bool inverse) {
    std::vector<Cplx> line;
    // x
    line.resize(nx);
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) line[x] = cube[(z * ny + y) * nx + x];
            fft1d(line.data(), nx, inverse);
            for (int x = 0; x < nx; x++) cube[(z * ny + y) * nx + x] = line[x];
        }
    }
    // y
    line.resize(ny);
    for (int z = 0; z < nz; z++) {
        for (int x = 0; x < nx; x++) {
            for (int y = 0; y < ny; y++) line[y] = cube[(z * ny + y) * nx + x];
            fft1d(line.data(), ny, inverse);
            for (int y = 0; y < ny; y++) cube[(z * ny + y) * nx + x] = line[y];
        }
    }
    // z
    line.resize(nz);
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int z = 0; z < nz; z++) line[z] = cube[(z * ny + y) * nx + x];
            fft1d(line.data(), nz, inverse);
            for (int z = 0; z < nz; z++) cube[(z * ny + y) * nx + x] = line[z];
        }
    }
}

// ---- random helpers (PCG32, O'Neill 2014, period 2^64) ----
static unsigned int g_seed = 123456789;          // legacy LCG state (kept for setSeed API)
static uint64_t g_pcg_state = 0x853c49e6748fea9bULL;
static uint64_t g_pcg_inc   = 0xda3e39cb94b95bdbULL;

static uint32_t pcg32_next() {
    uint64_t oldstate = g_pcg_state;
    g_pcg_state = oldstate * 6364136223846793005ULL + g_pcg_inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-(int32_t)rot) & 31));
}

static void pcg32_seed_from(uint64_t seed) {
    g_pcg_state = 0;
    g_pcg_inc   = (seed << 1u) | 1u;
    pcg32_next();
    g_pcg_state += seed;
    pcg32_next();
}

static float randf() {
    return (float)pcg32_next() / (float)0xFFFFFFFFu;
}

static float gauss01() {
    float u1 = randf(); if (u1 < 1e-7f) u1 = 1e-7f;
    float u2 = randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// ---- Eisenstein-Hu transfer function (no baryons, simplified) ----
static float T_EH(float k, float omegaM, float h) {
    float q = k / (omegaM * h * h);   // in Mpc/h units ; here adimensionalised
    float L = logf(2.0f * expf(1.0f) + 1.8f * q);
    float C = 14.2f + 731.0f / (1.0f + 62.5f * q);
    return L / (L + C * q * q);
}

// ---- power spectrum P(k) = A * k^ns * T^2(k) * D^2(a) ----
static float powerSpectrum(float k, float ns, float omegaM, float h, float D) {
    float Tk = T_EH(k, omegaM, h);
    // Normalisation via sigma8 built-in : we keep A=1 and scale later
    float Pk = powf(k, ns) * Tk * Tk * D * D;
    // Harrison-Zel'dovich normalisation for k->0 : P ~ k^1
    // Small-k cutoff to avoid divergence at k=0
    if (k < 1e-6f) Pk = 0.0f;
    return Pk;
}

// ---- linear growth factor D(a) for LCDM ----
//  D(a) ~ a * 2F1(...) ; accurate approximation:
static float growthFactor(float a, float omegaM, float omegaL) {
    float om = omegaM / (omegaM + omegaL * a * a * a);
    float ol = 1.0f - om;
    return a * powf(om, 0.545f);   // Linder approximation (good to ~0.1%)
}

// Compute sigma8 of the unnormalized power spectrum P(k)=k^ns*T^2(k) (D=1)
// via numerical integration: σ8² = (1/2π²) ∫ P(k) W²(kR8) k² dk
static float computeSigma8(float ns, float omegaM) {
    const float h   = 0.7f;
    const float R8  = 8.0f / h;   // 8 Mpc/h in Mpc
    const int   N   = 2000;
    const float lnk_min = logf(1e-4f), lnk_max = logf(1e2f);
    const float dlnk    = (lnk_max - lnk_min) / (float)N;
    float s2 = 0.0f;
    for (int i = 0; i < N; i++) {
        float k  = expf(lnk_min + (i + 0.5f) * dlnk);
        float kR = k * R8;
        float W  = (kR < 0.001f) ? 1.0f
                   : 3.0f * (sinf(kR) - kR * cosf(kR)) / (kR * kR * kR);
        float Pk = powerSpectrum(k, ns, omegaM, h, 1.0f);   // D=1
        s2 += Pk * W * W * k * k * k * dlnk;                // k³ P(k) dlnk
    }
    return sqrtf(s2 / (2.0f * (float)M_PI * (float)M_PI));
}

// ------------------------------------------------------------------ *
UniverseGenerator::UniverseGenerator(const UniverseParams& p)
    : m_p(p), m_total(p.numPosiPart + p.numNegaPart)
    , m_omegaM(0.3f), m_omegaL(0.7f), m_ns(0.96f), m_sigma8(0.8f), m_aStart(0.01f)
    , m_nGrid(64)
{
    // Seed PCG32 from time + per-instance counter (so back-to-back Apply&Generate
    // calls within the same second get different sequences).
    static uint64_t ctr = 0;
    ctr += 2654435761ULL;
    uint64_t s = (uint64_t)time(NULL) ^ ctr;
    g_seed = (unsigned int)s;
    pcg32_seed_from(s);
}

void UniverseGenerator::setSeed(unsigned int seed) {
    if (seed != 0) {
        g_seed = seed;
        pcg32_seed_from((uint64_t)seed);
    }
}

void UniverseGenerator::setCosmology(float omegaM, float omegaL, float ns, float sigma8, float aStart) {
    m_omegaM = omegaM;
    m_omegaL = omegaL;
    m_ns     = ns;
    m_sigma8 = sigma8;
    m_aStart = aStart;
}

void UniverseGenerator::setGridResolution(int nGrid) {
    m_nGrid = nGrid;
}

/* ------------------------------------------------------------------ *
 * generate() — Cosmological ICs with Zel'dovich approximation
 * ------------------------------------------------------------------ *
 * 1. Build a regular Lagrangian grid of N³ cells
 * 2. Generate δ(k) modes with random phases, amplitude ∝ √P(k)
 * 3. FFT inverse → δ(x) on the grid
 * 4. Compute displacement field s = -∇Ψ where ∇²Ψ = δ
 *    (in Fourier space:  s̃(k) = i k δ̃(k) / k²)
 * 5. Displace particles:  x = q + D(a) · s(q)   (Zel'dovich)
 * 6. Velocities:  v = a · Ḋ · s(q)   ≈  a · H · f · D · s(q)
 *    with f = d ln D / d ln a ≈ Ω_m^0.545
 * ------------------------------------------------------------------ */
void UniverseGenerator::generate(std::vector<cl_float4>& pos,
                                   std::vector<cl_float4>& vel)
{
    if (m_p.freeExplosion) {
        int nPart = m_total;
        pos.resize(nPart);
        vel.resize(nPart);
        float H0_sim = m_p.H0;
        int nPosi = m_p.numPosiPart;
        int nNega = m_p.numNegaPart;

        if (m_p.periodicBox) {
            // ----------------------------------------------------------------
            // Level-1 perturbed lattice ICs (Two-Poisson Janus PM validation).
            //
            // + and − each placed on their OWN cubic lattice with small
            // INDEPENDENT Gaussian perturbations (different RNG states).
            // Seeds are different because + perturbations consume RNG before −.
            //
            // Why independent (not paired anti-symmetric)?
            //   Paired:      pos− = x0 − δ  → ρ+ ≈ ρ−  → F+ ≈ F−  → net ≈ 0
            //   Independent: pos− = x0 + δ' → ρ+ ≠ ρ−  → F+ ≠ F−  → structure forms
            //
            // Lattice reduces shot noise vs pure random (Poisson-distributed);
            // small perturbation amplitude (pertAmp × lat) excites only large-k
            // modes safely below Nyquist.
            // ----------------------------------------------------------------
            int N_lat = (int)round(cbrtf((float)nPosi));
            float lat  = 2.0f * m_p.initRadius / (float)N_lat;
            float eps  = lat * m_p.pertAmp;   // Gaussian sigma = pertAmp × cell

            // + species: perturbed lattice with current RNG state
            int placed = 0;
            for (int ix = 0; ix < N_lat && placed < nPosi; ix++) {
                for (int iy = 0; iy < N_lat && placed < nPosi; iy++) {
                    for (int iz = 0; iz < N_lat && placed < nPosi; iz++) {
                        float x0 = -m_p.initRadius + (ix + 0.5f) * lat;
                        float y0 = -m_p.initRadius + (iy + 0.5f) * lat;
                        float z0 = -m_p.initRadius + (iz + 0.5f) * lat;
                        float dx = eps * gauss01();
                        float dy = eps * gauss01();
                        float dz = eps * gauss01();
                        pos[placed] = { x0+dx, y0+dy, z0+dz, m_p.posiMass };
                        vel[placed] = { 0.0f, 0.0f, 0.0f, 0.0f };
                        placed++;
                    }
                }
            }

            // − species: SAME lattice geometry but with INDEPENDENT perturbations
            // (RNG already advanced by + placements → different sequence).
            int placed_neg = 0;
            for (int ix = 0; ix < N_lat && placed_neg < nNega; ix++) {
                for (int iy = 0; iy < N_lat && placed_neg < nNega; iy++) {
                    for (int iz = 0; iz < N_lat && placed_neg < nNega; iz++) {
                        float x0 = -m_p.initRadius + (ix + 0.5f) * lat;
                        float y0 = -m_p.initRadius + (iy + 0.5f) * lat;
                        float z0 = -m_p.initRadius + (iz + 0.5f) * lat;
                        float dx = eps * gauss01();   // independent from + dx
                        float dy = eps * gauss01();
                        float dz = eps * gauss01();
                        int j = nPosi + placed_neg;
                        pos[j] = { x0+dx, y0+dy, z0+dz, m_p.negaMass };
                        vel[j] = { 0.0f, 0.0f, 0.0f, 0.0f };
                        placed_neg++;
                    }
                }
            }
        } else if (false && m_p.periodicBox) {
            // (old paired lattice kept for reference — no longer used)
            // ----------------------------------------------------------------
            // Janus cosmological IC (anti-Poisson lattice + paired ±).
            //
            // Problem with pure Poisson: 500k particles in 1000 Mpc box →
            // 0.015 particles/cell → 100% fluctuations at PM cell scale →
            // artificial noise amplified 8000× at aStart=0.05 → white cube.
            //
            // Solution: place + and − on a regular lattice with OPPOSITE small
            // displacements δ.  Net density ρ+−ρ− is exactly zero on the
            // lattice nodes and carries only the physical dipole field at scale
            // δ.  No Poisson shot noise → clean alveolar structure formation.
            // ----------------------------------------------------------------
            int N_lat = (int)round(cbrtf((float)nPosi));
            float lat  = 2.0f * m_p.initRadius / (float)N_lat;
            float eps  = lat * 0.3f * m_p.pertAmp;  // displacement amplitude

            int placed = 0;
            for (int ix = 0; ix < N_lat && placed < nPosi; ix++) {
                for (int iy = 0; iy < N_lat && placed < nPosi; iy++) {
                    for (int iz = 0; iz < N_lat && placed < nPosi; iz++) {
                        float x0 = -m_p.initRadius + (ix + 0.5f) * lat;
                        float y0 = -m_p.initRadius + (iy + 0.5f) * lat;
                        float z0 = -m_p.initRadius + (iz + 0.5f) * lat;
                        float dx = eps * (randf() * 2.0f - 1.0f);
                        float dy = eps * (randf() * 2.0f - 1.0f);
                        float dz = eps * (randf() * 2.0f - 1.0f);
                        // Positive: displaced +δ
                        pos[placed] = { x0+dx, y0+dy, z0+dz, m_p.posiMass };
                        vel[placed] = { 0.0f, 0.0f, 0.0f, 0.0f };
                        // Negative: displaced −δ (same node, opposite direction)
                        int j = nPosi + placed;
                        if (j < nPart) {
                            pos[j] = { x0-dx, y0-dy, z0-dz, m_p.negaMass };
                            vel[j] = { 0.0f, 0.0f, 0.0f, 0.0f };
                        }
                        placed++;
                    }
                }
            }
        } else {
            // Free-explosion sphere with SEEDED cosmic-web ICs (no periodic box).
            //
            // The web needs rho+ != rho- with power at a definite LARGE scale, else
            // the global Janus mode (negatives collapse to centre, positives form a
            // shell = one bubble) dominates and you get a central blob, not a web.
            //
            // So we seed TWO scales:
            //  (1) LARGE-SCALE anti-correlated displacement (the "cluster scale"):
            //      a coarse random field D(x); + is pushed +D, - is pushed -D, so
            //      the two Janus fluids SEPARATE into filaments/voids everywhere at
            //      once (Zel'dovich-style). This is what makes the uniform web.
            //  (2) small cell-scale jitter for fine structure.
            // Open boundary (tree) + gentle Hubble flow lets the sphere breathe.
            // pertAmp scales BOTH seeds; webCells sets the number of bubbles.
            const float SPHERE_FRAC = 0.5236f;          // (4/3 pi r^3)/(2r)^3 = pi/6
            int   N_lat = (int)floorf(cbrtf((float)nPosi / SPHERE_FRAC));
            if (N_lat < 1) N_lat = 1;
            float lat    = 2.0f * m_p.initRadius / (float)N_lat;
            float epsFine= lat * m_p.pertAmp * 0.5f;    // fine cell jitter
            float R2     = m_p.initRadius * m_p.initRadius;
            float vtherm = H0_sim * m_p.initRadius * 0.02f;

            // PHYSICAL P(k) seed (Zel'dovich) — the STANDARD cosmological IC recipe,
            // not a hand-tuned knob: draw a Gaussian field delta(k) ~ sqrt(P(k)) with
            // a realistic CDM-like P(k), build the Zel'dovich displacement
            // s = i k delta / k^2, FFT to real space, then push + by +s and - by -s
            // (Janus anti-correlated). pertAmp sets the displacement RMS (in cells);
            // webCells sets the small-scale cutoff (more cells -> finer modes kept).
            // FFT grid resolution: scale it with the particle lattice (N_lat) so the
            // seed is as fine as the particles can sample — more particles = more
            // definition, automatically. Power of 2 (fft1d requirement), clamped.
            int ngz = 64;
            while (ngz < N_lat && ngz < 512) ngz <<= 1;   // 1M->128, 5M->256, >=9M->512 (cap)
            const int   ngz3 = ngz * ngz * ngz;
            const float boxz = 2.0f * m_p.initRadius;    // sphere fits in [-R,R]
            const float dkz  = 2.0f * (float)M_PI / boxz;
            const float cellz= boxz / (float)ngz;
            const float kNyq = dkz * (float)ngz * 0.5f;
            // Web cells 4..100 -> small-scale cutoff fraction of Nyquist (more = finer).
            float kCutFrac = (float)((m_p.webCells >= 2) ? m_p.webCells : 12) / 100.0f;
            if (kCutFrac < 0.08f) kCutFrac = 0.08f;
            if (kCutFrac > 0.92f) kCutFrac = 0.92f;
            float kCut = kNyq * kCutFrac;
            // Mode "mousse Petit" : spectre piqué -> bulles UNIFORMES (webCells = nb de bulles par axe),
            // au lieu du P(k) hiérarchique. kpeak = webCells*dkz (taille bulle = 2R/webCells), borné par Nyquist.
            const bool  foam  = m_p.foamMode;
            float       kPeak = (float)m_p.webCells * dkz;
            if (kPeak > 0.80f * kNyq) kPeak = 0.80f * kNyq;  // plus de bulles -> augmenter le nb de particules (grille adaptative)
            const float kSig  = kPeak * 0.30f;               // bande resserree autour de l'echelle bulle
            // Plancher bas-k HAUT (0.65*kPic) : PAS de puissance grande-echelle -> pas de
            // structure geante qui domine (les 1-2 grosses bulles), seulement beaucoup de
            // bulles a l'echelle kPic. Les coquilles sont evitees par la vitesse de Zel'dovich.
            float kLoF = kPeak - 1.2f * kSig; if (kLoF < 0.65f * kPeak) kLoF = 0.65f * kPeak;
            float kHiF = kPeak + 1.5f * kSig; if (kHiF > 0.92f * kNyq)  kHiF = 0.92f * kNyq;
            std::vector<Cplx> sxk(ngz3), syk(ngz3), szk(ngz3);
            for (int iz = 0; iz < ngz; iz++) { int kz = iz; if (kz > ngz/2) kz -= ngz;
              for (int iy = 0; iy < ngz; iy++) { int ky = iy; if (ky > ngz/2) ky -= ngz;
                for (int ix = 0; ix < ngz; ix++) { int kx = ix; if (kx > ngz/2) kx -= ngz;
                  int idx = (iz*ngz + iy)*ngz + ix;
                  float kMag = dkz * sqrtf((float)(kx*kx + ky*ky + kz*kz));
                  float kLo  = foam ? kLoF : 1e-8f;
                  float kHi  = foam ? kHiF : kCut;
                  if (kMag < kLo || kMag > kHi) { sxk[idx]={0,0}; syk[idx]={0,0}; szk[idx]={0,0}; continue; }
                  float Pk;
                  if (foam) { float d = (kMag - kPeak) / kSig; Pk = expf(-0.5f * d * d); } // bande gaussienne piquee
                  else      { Pk = powerSpectrum(kMag, m_ns, m_omegaM, 0.7f, 1.0f); }       // P(k) CDM hierarchique
                  float amp = sqrtf(0.5f * Pk / (float)ngz3);
                  Cplx d = { amp * gauss01(), amp * gauss01() };
                  float invK2 = 1.0f / (kMag * kMag);
                  sxk[idx] = { -d.im*kx*dkz*invK2, d.re*kx*dkz*invK2 };
                  syk[idx] = { -d.im*ky*dkz*invK2, d.re*ky*dkz*invK2 };
                  szk[idx] = { -d.im*kz*dkz*invK2, d.re*kz*dkz*invK2 };
                }}}
            fft3d(sxk.data(), ngz, ngz, ngz, true);
            fft3d(syk.data(), ngz, ngz, ngz, true);
            fft3d(szk.data(), ngz, ngz, ngz, true);
            // Normalize displacement RMS to pertAmp * cell (intuitive, unit-free).
            {
                double ms = 0.0;
                for (int i = 0; i < ngz3; i++)
                    ms += (double)sxk[i].re*sxk[i].re + (double)syk[i].re*syk[i].re + (double)szk[i].re*szk[i].re;
                float rms = (float)sqrt(ms / (3.0 * (double)ngz3));
                float target = m_p.pertAmp * cellz;
                float sc = (rms > 1e-20f) ? target / rms : 0.0f;
                for (int i = 0; i < ngz3; i++) { sxk[i].re *= sc; syk[i].re *= sc; szk[i].re *= sc; }
            }
            auto sampleD = [&](float x, float y, float z, float& ox, float& oy, float& oz) {
                float gx = (x + m_p.initRadius) / cellz;
                float gy = (y + m_p.initRadius) / cellz;
                float gz = (z + m_p.initRadius) / cellz;
                int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
                float fx = gx - ix, fy = gy - iy, fz = gz - iz;
                auto wp = [&](int v){ v %= ngz; return v < 0 ? v + ngz : v; };
                ox = oy = oz = 0.0f;
                for (int b = 0; b < 8; b++) {
                    int dx2 = b & 1, dy2 = (b >> 1) & 1, dz2 = (b >> 2) & 1;
                    float w = (dx2 ? fx : 1 - fx) * (dy2 ? fy : 1 - fy) * (dz2 ? fz : 1 - fz);
                    int idx = (wp(iz + dz2)*ngz + wp(iy + dy2))*ngz + wp(ix + dx2);
                    ox += w * sxk[idx].re; oy += w * syk[idx].re; oz += w * szk[idx].re;
                }
            };

            // In-sphere lattice node centres (shared base for + and -).
            std::vector<float> nx, ny, nz;
            nx.reserve((size_t)nPosi + 4096);
            for (int ix = 0; ix < N_lat; ix++) {
                float x0 = -m_p.initRadius + (ix + 0.5f) * lat;
                for (int iy = 0; iy < N_lat; iy++) {
                    float y0 = -m_p.initRadius + (iy + 0.5f) * lat;
                    for (int iz = 0; iz < N_lat; iz++) {
                        float z0 = -m_p.initRadius + (iz + 0.5f) * lat;
                        if (x0*x0 + y0*y0 + z0*z0 <= R2) {
                            // Jitter aléatoire ±½ maille : CASSE le réseau cubique (sinon
                            // plans/rangées alignés -> bande lumineuse + rayons vus de profil).
                            // "Glass-like" : densité uniforme, pas de symétrie de grille.
                            nx.push_back(x0 + (randf()-0.5f)*lat);
                            ny.push_back(y0 + (randf()-0.5f)*lat);
                            nz.push_back(z0 + (randf()-0.5f)*lat);
                        }
                    }
                }
            }
            int nNodes = (int)nx.size();

            // sign = +1 for positives (pushed +D), -1 for negatives (pushed -D).
            auto placeSpecies = [&](int start, int count, float mass, float sign) {
                for (int k = 0; k < count; k++) {
                    float x0, y0, z0;
                    if (k < nNodes) { x0 = nx[k]; y0 = ny[k]; z0 = nz[k]; }
                    else {
                        float rr = m_p.initRadius * powf(randf(), 1.0f / 3.0f);
                        float th = randf() * 2.0f * (float)M_PI;
                        float ph = acosf(2.0f * randf() - 1.0f);
                        x0 = rr*sinf(ph)*cosf(th); y0 = rr*sinf(ph)*sinf(th); z0 = rr*cosf(ph);
                    }
                    float dX, dY, dZ; sampleD(x0, y0, z0, dX, dY, dZ);
                    float x = x0 + sign * dX + epsFine * gauss01();
                    float y = y0 + sign * dY + epsFine * gauss01();
                    float z = z0 + sign * dZ + epsFine * gauss01();
                    // Zel'dovich PECULIAR velocity: PARALLEL to the displacement (matter
                    // converges onto the forming walls in 3D). Without it the only coherent
                    // motion is the radial Hubble outflow from the centre => concentric
                    // caustic shells ("Russian dolls"). Amplitude ~ growth rate × disp,
                    // chosen comparable to the local Hubble shear so peculiar flows compete.
                    const float zelVFac = 2.0f * H0_sim;
                    float vx = H0_sim * x + zelVFac * sign * dX + vtherm * (randf() * 2.0f - 1.0f);
                    float vy = H0_sim * y + zelVFac * sign * dY + vtherm * (randf() * 2.0f - 1.0f);
                    float vz = H0_sim * z + zelVFac * sign * dZ + vtherm * (randf() * 2.0f - 1.0f);
                    pos[start + k] = { x, y, z, mass };
                    vel[start + k] = { vx, vy, vz, 0.0f };
                }
            };
            placeSpecies(0,     nPosi, m_p.posiMass, +1.0f);
            placeSpecies(nPosi, nNega, m_p.negaMass, -1.0f);   // opposite displacement
        }
    } else {
        // Original cosmological generation with box
        int ng = m_nGrid;
        int ng3 = ng * ng * ng;
        float box = m_p.initRadius * 3.0f;     // box size in sim units
        float dk  = 2.0f * (float)M_PI / box;    // fundamental mode
        float cell = box / (float)ng;

        // --- Allocate 3D grids ---
        std::vector<Cplx> delta(ng3);
        std::vector<Cplx> dispX(ng3), dispY(ng3), dispZ(ng3);

        // --- Growth factor at start ---
        float D1 = growthFactor(m_aStart, m_omegaM, m_omegaL);
        float f  = powf(m_omegaM / (m_omegaM + m_omegaL * m_aStart * m_aStart * m_aStart), 0.545f);
        // Use cosmoScale for velocity to be consistent with Friedmann integration
        float H0_zel = m_p.H0 * m_p.cosmoScale;
        float aDot   = H0_zel * m_aStart * sqrtf(m_omegaM * powf(m_aStart, -3.0f) + m_omegaL);
        float vFactor = aDot * f;   // D1 removed: sx is now s0 (D=1 in powerSpectrum)

        // --- Fill Fourier modes δ̃(k) ---
        for (int iz = 0; iz < ng; iz++) {
            int kz = iz;
            if (kz > ng / 2) kz -= ng;      // wrap to [-N/2, N/2]
            for (int iy = 0; iy < ng; iy++) {
                int ky = iy;
                if (ky > ng / 2) ky -= ng;
                for (int ix = 0; ix < ng; ix++) {
                    int kx = ix;
                    if (kx > ng / 2) kx -= ng;

                    int idx = (iz * ng + iy) * ng + ix;
                    float kMag = dk * sqrtf((float)(kx * kx + ky * ky + kz * kz));
                    if (kMag < 1e-8f || kMag > dk * ng * 0.5f * 0.8f) {
                        delta[idx] = { 0, 0 };
                        continue;
                    }

                    float Pk = powerSpectrum(kMag, m_ns, m_omegaM, 0.7f, 1.0f); // D=1: σ8 scale applied separately
                    float amp = sqrtf(0.5f * Pk / (float)ng3);   // factor ½ per mode

                    float phase = 2.0f * (float)M_PI * randf();
                    float g1 = gauss01();
                    float g2 = gauss01();
                    // Complex Gaussian : mean 0, variance 1 per real/imag
                    delta[idx] = { amp * g1, amp * g2 };

                    // Displacement field in Fourier space :  s̃_i(k) = i k_i δ̃(k) / k²
                    float invK2 = 1.0f / (kMag * kMag);
                    dispX[idx] = { -delta[idx].im * kx * dk * invK2,
                                     delta[idx].re * kx * dk * invK2 };
                    dispY[idx] = { -delta[idx].im * ky * dk * invK2,
                                     delta[idx].re * ky * dk * invK2 };
                    dispZ[idx] = { -delta[idx].im * kz * dk * invK2,
                                     delta[idx].re * kz * dk * invK2 };
                }
            }
        }

        // --- FFT inverse : δ(x), s_x(x), s_y(x), s_z(x) ---
        fft3d(delta.data(), ng, ng, ng, true);
        fft3d(dispX.data(), ng, ng, ng, true);
        fft3d(dispY.data(), ng, ng, ng, true);
        fft3d(dispZ.data(), ng, ng, ng, true);

        // ── Sigma8 normalization ──────────────────────────────────────────
        // Scale displacement field so that σ8 = m_sigma8 at z=0 (D=1).
        // At aStart: σ8(aStart) = m_sigma8 × D1, well within the linear regime.
        {
            float sigma8_raw   = computeSigma8(m_ns, m_omegaM);
            float sigma8_scale = (sigma8_raw > 1e-12f) ? m_sigma8 / sigma8_raw : 1.0f;
            for (auto& c : dispX) c.re *= sigma8_scale;
            for (auto& c : dispY) c.re *= sigma8_scale;
            for (auto& c : dispZ) c.re *= sigma8_scale;
        }

        // --- Sample particles from the grid ---
        int nPart = m_total;
        pos.resize(nPart);
        vel.resize(nPart);

        int nPosi = m_p.numPosiPart;
        int nNega = m_p.numNegaPart;

        // Janus anti-correlated Zel'dovich ICs:
        // +mass at q + D1·s(q),  −mass at q − D1·s(q)
        // → ρ_net(x) = 2·ρ_mean·δ(x) from the Gaussian random field.
        // This gives a physical, isotropic density spectrum and non-zero
        // PM density from the start (no ρ_net≈0 pathology of paired-lattice).
        bool janus = (m_p.negaMass < 0.0f && nNega > 0);

        // Grid sized for one species so each Lagrangian node hosts one pair
        int nForGrid = janus ? nPosi : nPart;
        int nPerDim = (int)ceilf(powf((float)nForGrid, 1.0f / 3.0f));
        if (nPerDim < 2) nPerDim = 2;

        int placed = 0;
        for (int iz = 0; iz < nPerDim && placed < nPosi; iz++) {
            for (int iy = 0; iy < nPerDim && placed < nPosi; iy++) {
                for (int ix = 0; ix < nPerDim && placed < nPosi; ix++) {
                    // Lagrangian position on regular grid, centred in box
                    float qx = (ix + 0.5f) / nPerDim * box - box * 0.5f;
                    float qy = (iy + 0.5f) / nPerDim * box - box * 0.5f;
                    float qz = (iz + 0.5f) / nPerDim * box - box * 0.5f;

                    // Interpolate displacement from FFT grid (trilinear)
                    float gx = (qx + box * 0.5f) / cell;
                    float gy = (qy + box * 0.5f) / cell;
                    float gz = (qz + box * 0.5f) / cell;
                    int ix0 = (int)floorf(gx) % ng; if (ix0 < 0) ix0 += ng;
                    int iy0 = (int)floorf(gy) % ng; if (iy0 < 0) iy0 += ng;
                    int iz0 = (int)floorf(gz) % ng; if (iz0 < 0) iz0 += ng;
                    int ix1 = (ix0 + 1) % ng;
                    int iy1 = (iy0 + 1) % ng;
                    int iz1 = (iz0 + 1) % ng;
                    float tx = gx - floorf(gx);
                    float ty = gy - floorf(gy);
                    float tz = gz - floorf(gz);

                    auto trilin = [&](const std::vector<Cplx>& field) -> float {
                        int i000 = (iz0 * ng + iy0) * ng + ix0;
                        int i001 = (iz1 * ng + iy0) * ng + ix0;
                        int i010 = (iz0 * ng + iy1) * ng + ix0;
                        int i011 = (iz1 * ng + iy1) * ng + ix0;
                        int i100 = (iz0 * ng + iy0) * ng + ix1;
                        int i101 = (iz1 * ng + iy0) * ng + ix1;
                        int i110 = (iz0 * ng + iy1) * ng + ix1;
                        int i111 = (iz1 * ng + iy1) * ng + ix1;
                        float c00 = field[i000].re * (1 - tx) + field[i100].re * tx;
                        float c01 = field[i001].re * (1 - tx) + field[i101].re * tx;
                        float c10 = field[i010].re * (1 - tx) + field[i110].re * tx;
                        float c11 = field[i011].re * (1 - tx) + field[i111].re * tx;
                        float c0  = c00 * (1 - ty) + c10 * ty;
                        float c1  = c01 * (1 - ty) + c11 * ty;
                        return c0 * (1 - tz) + c1 * tz;
                    };

                    float sx = trilin(dispX);
                    float sy = trilin(dispY);
                    float sz = trilin(dispZ);

                    // + mass: standard Zel'dovich displacement
                    float x = qx + D1 * sx;
                    float y = qy + D1 * sy;
                    float z = qz + D1 * sz;
                    float vx = vFactor * sx;
                    float vy = vFactor * sy;
                    float vz = vFactor * sz;

                    float vtherm = H0_zel * cell * 0.05f;
                    vx += vtherm * (randf() * 2.0f - 1.0f);
                    vy += vtherm * (randf() * 2.0f - 1.0f);
                    vz += vtherm * (randf() * 2.0f - 1.0f);

                    pos[placed] = { x, y, z, m_p.posiMass };
                    vel[placed] = { vx, vy, vz, 0.0f };

                    if (janus) {
                        // − mass: ANTI-correlated displacement q − D1·s(q)
                        // Creates ρ_net = 2·ρ_mean·δ: + goes to overdense regions,
                        // − goes to underdense regions → isotropic Janus bubble ICs.
                        int j = nPosi + placed;
                        if (j < nPart) {
                            pos[j] = { qx - D1*sx, qy - D1*sy, qz - D1*sz, m_p.negaMass };
                            vel[j] = { -vFactor*sx, -vFactor*sy, -vFactor*sz, 0.0f };
                        }
                    }

                    placed++;
                }
            }
        }

        // Fill remaining slots if grid didn't cover all
        for (int i = placed; i < nPart; i++) {
            bool isNeg = (i >= nPosi);
            float mass = isNeg ? m_p.negaMass : m_p.posiMass;
            float r = m_p.initRadius * powf(randf(), 1.0f / 3.0f);
            float theta = randf() * 2.0f * (float)M_PI;
            float phi   = acosf(2.0f * randf() - 1.0f);
            pos[i] = { r * sinf(phi) * cosf(theta),
                       r * sinf(phi) * sinf(theta),
                       r * cosf(phi), mass };
            vel[i] = { H0_zel * pos[i].x, H0_zel * pos[i].y, H0_zel * pos[i].z, 0.0f };
        }
    }
}
