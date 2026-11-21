#include "galaxy_generator.h"
#include <cmath>
#include <cstdint>
#include <ctime>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
//  PCG32 random number generator (O'Neill 2014)
//  Period 2^64, 32-bit output. ~100× more unique values than Windows rand()
//  (which is 15-bit, only ~32k unique outputs).
// ---------------------------------------------------------------------------
static uint64_t g_pcg_state = 0x853c49e6748fea9bULL;
static uint64_t g_pcg_inc   = 0xda3e39cb94b95bdbULL;

static uint32_t pcg32_next() {
    uint64_t oldstate = g_pcg_state;
    g_pcg_state = oldstate * 6364136223846793005ULL + g_pcg_inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-(int32_t)rot) & 31));
}

static void pcg32_seed(uint64_t seed) {
    g_pcg_state = 0;
    g_pcg_inc   = (seed << 1u) | 1u;
    pcg32_next();
    g_pcg_state += seed;
    pcg32_next();
}

static float randf()  { return (float)pcg32_next() / (float)0xFFFFFFFFu; }
static float randf2() { return randf() * 2.0f - 1.0f; }

// Box-Muller: Gaussian N(0, stddev²)
static float gauss(float stddev)
{
    float u1 = randf();
    if (u1 < 1e-7f) u1 = 1e-7f;
    float u2 = randf();
    return stddev * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

GalaxyGenerator::GalaxyGenerator(const GalaxyParams& params)
    : m_params(params)
    , m_totalCount(params.numPosiStars + params.numNegaStars + 1)
{
    // Seed PCG32 from time. Mix with a per-instance counter so successive calls
    // in the same second still get different sequences.
    static uint64_t ctr = 0;
    ctr += 2654435761ULL;
    pcg32_seed((uint64_t)time(NULL) ^ ctr);
}

void GalaxyGenerator::generate(std::vector<cl_float4>& positions,
                                std::vector<cl_float4>& velocities)
{
    const GalaxyParams& p = m_params;
    positions.resize(m_totalCount);
    velocities.resize(m_totalCount);

    float massPerPosiStar = (p.numPosiStars > 0)
                            ? p.totalSolarMass / (float)p.numPosiStars
                            : 1.0f;

    // negaDensity is treated as TOTAL halo mass (negative), distributed equally
    // across all negative particles. This keeps the physical halo mass fixed
    // when the user changes the number of negative particles, preventing the
    // Janus repulsion from scaling out of control at high N.
    float totalNegaMass  = p.negaDensity;   // total halo mass (negative)
    float massPerNegaStar = (p.numNegaStars > 0)
                            ? totalNegaMass / (float)p.numNegaStars
                            : 0.0f;
    float haloRadius     = p.radius * 1.5f;

    // ── 0 : Central black hole ────────────────────────────────────────────────
    positions [0] = { 0.0f, 0.0f, 0.0f, p.blackHoleMass };
    velocities[0] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // ── 1 : Positive stars — thin disk ───────────────────────────────────────
    for (int i = 1; i <= p.numPosiStars; i++)
    {
        float r     = p.radius * sqrtf(randf());      // uniform surface density
        float theta = randf() * 2.0f * (float)M_PI;
        float z     = p.radius * 0.02f * randf2();    // thin disk

        float x = r * cosf(theta);
        float y = r * sinf(theta);

        float frac     = r / p.radius;
        float fracHalo = fminf(r / haloRadius, 1.0f);

        // Janus-corrected enclosed mass: positive disk + BH attract,
        // negative halo (uniform sphere, mass < 0) repels via Newton shell theorem.
        // Ignoring the halo term over-rotates the disk and ejects particles → ring transient.
        float mEnc = p.blackHoleMass
                   + p.totalSolarMass * frac * frac
                   + totalNegaMass * fracHalo * fracHalo * fracHalo;
        if (mEnc < p.blackHoleMass) mEnc = p.blackHoleMass;

        // Plummer-softened Keplerian: smooths the 1/r singularity and matches
        // what the tree code actually computes (softening ~ disk thickness here).
        // Without softening here, IC over-estimates the available centripetal
        // force → centrifuge wins → particles fly out → ring transient.
        const float eps = 10.0f;  // matches default UI softening
        float r_eff2 = r * r + eps * eps;
        float vCirc = (r > 0.5f && mEnc > 0.0f)
                        ? sqrtf(p.grav * mEnc * r * r / (r_eff2 * sqrtf(r_eff2)))
                        : 0.0f;

        // Rotation speed scale (disk_factor removed — was inflating vCirc by up
        // to +41% near the center, causing radial ejection and ring formation).
        float v_base = vCirc * (p.rotSpeed / 10.0f);

        // Thermal dispersion: scale with LOCAL vCirc, not global vCircNominal.
        // Global scaling blasts inner particles outward (where local vCirc ≪ global).
        float v_thermal = p.initSpeed * p.scaleUnit * vCirc;

        float vx = -sinf(theta) * v_base + v_thermal * randf2();
        float vy =  cosf(theta) * v_base + v_thermal * randf2();
        float vz =  v_thermal * randf2() * 0.1f;

        positions [i] = { x, y, z, massPerPosiStar };
        velocities[i] = { vx, vy, vz, 0.0f };
    }

    // ── 2 : Negative stars — 3D spherical halo (uniform volume, ρ = const) ───
    // Uniform volume density keeps the halo stable and avoids central concentration.
    // Shell theorem: outer halo exerts zero net force on interior → disk unperturbed.
    // massPerNegaStar already computed above from total halo mass.
    for (int i = p.numPosiStars + 1; i < m_totalCount; i++)
    {
        // Rejection sampling: uniform in sphere volume (ρ ∝ const)
        float x, y, z, r;
        do {
            x = haloRadius * randf2();
            y = haloRadius * randf2();
            z = haloRadius * randf2();
            r = sqrtf(x*x + y*y + z*z);
        } while (r > haloRadius || r < 0.1f);

        // Effective enclosed mass for velocity dispersion
        float fracPos   = fminf(r / p.radius, 1.0f);
        float fracNeg   = fminf(r / haloRadius, 1.0f);
        float mPosEnc   = p.blackHoleMass + p.totalSolarMass * fracPos * fracPos;
        float mNegEnc   = totalNegaMass * powf(fracNeg, 3.0f);  // uniform volume
        float mTotalEnc = fmaxf(mPosEnc + fabsf(mNegEnc), p.blackHoleMass);

        // Maxwell-Boltzmann: σ = vEsc / 3
        float vEsc  = (r > 0.5f) ? sqrtf(2.0f * p.grav * mTotalEnc / r) : 0.0f;
        float sigma = vEsc / 3.0f;

        positions [i] = { x, y, z, massPerNegaStar };
        velocities[i] = { gauss(sigma), gauss(sigma), gauss(sigma), 0.0f };
    }
}
