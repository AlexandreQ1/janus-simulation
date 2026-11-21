#include "galaxy_generator.h"
#include <cmath>
#include <cstdlib>
#include <ctime>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

static float randf()  { return (float)rand() / (float)RAND_MAX; }
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
    srand((unsigned int)time(NULL));
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

    float totalNegaMass  = (float)p.numNegaStars * p.negaDensity;
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

        float frac  = r / p.radius;
        float mEnc  = p.blackHoleMass + p.totalSolarMass * frac * frac;

        // Keplerian circular velocity (standard, no Janus correction on IC)
        float vCirc = (r > 0.5f) ? sqrtf(p.grav * mEnc / r) : 0.0f;

        // Self-gravitating disk correction + rotation speed scale
        float disk_factor  = sqrtf(fmaxf(2.0f - frac * frac, 0.001f));
        float v_base       = vCirc * disk_factor * (p.rotSpeed / 10.0f);

        // Thermal dispersion
        float vCircNominal = sqrtf(p.grav * p.totalSolarMass / p.radius);
        float v_thermal    = p.initSpeed * p.scaleUnit * vCircNominal;

        float vx = -sinf(theta) * v_base + v_thermal * randf2();
        float vy =  cosf(theta) * v_base + v_thermal * randf2();
        float vz =  v_thermal * randf2() * 0.1f;

        positions [i] = { x, y, z, massPerPosiStar };
        velocities[i] = { vx, vy, vz, 0.0f };
    }

    // ── 2 : Negative stars — 3D spherical halo (uniform volume, ρ = const) ───
    // Uniform volume density keeps the halo stable and avoids central concentration.
    // Shell theorem: outer halo exerts zero net force on interior → disk unperturbed.
    float massPerNegaStar = p.negaDensity;

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
