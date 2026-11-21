// simulation.cpp — CUDA Runtime implementation
// Cosmological N-body with leapfrog DKD, Friedmann integration,
// adaptive timestep, and periodic boundaries.

#include "simulation.h"
#include "cuda_kernels.cuh"
#include "../gpu-nbody/pm_force.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cmath>

static inline float4* f4(void* p) { return reinterpret_cast<float4*>(p); }

// ------------------------------------------------------------------
// Constructor / Destructor
// ------------------------------------------------------------------

Simulation::Simulation()
    : m_dPStart(nullptr), m_dPEnd(nullptr), m_dVStart(nullptr)
    , m_dVEnd(nullptr), m_dPShared(nullptr), m_dA(nullptr), m_dBlockMax(nullptr)
    , m_glSharing(false)
    , m_count(0), m_global(0), m_local(256)
    , m_soften(10000.0f), m_timestep(1.0f), m_dtMin(0.001f), m_dtMax(10.0f), m_eta(0.2f)
    , m_grav(0.0000000000667408f)
    , m_simTime(0.0f)
    , m_a(1.0f), m_da(0.0f)
    , m_omegaM(0.3f), m_omegaL(0.7f)
    , m_adaptiveDt(false)
    , m_spAmplitude(0.0f), m_spPatternSpeed(0.01f), m_spPitchTan(0.268f), m_spArms(2)
    , m_physicsModel(1)
    , m_hubble(0.0f)
    , m_cosmoScale(1.0f)
    , m_rhoMean(0.0f)
    , m_boxSize(0.0f)
    , m_currentFrame(0), m_totalFrames(30)
    , m_gridX(1), m_gridY(1), m_gridZ(1)
    , m_blockX(256), m_blockY(1), m_blockZ(1)
    , m_usePM(false)
    , m_pmNg(128)
{
    m_pmPeriodic          = false;
    m_pmEnableSR          = true;
    m_useTree             = false;
    m_treeTheta           = 0.5f;
    m_treeCtx.initialized = false;
    m_pmCtx.initialized   = false;
    m_pmCtx.softening     = 0.0f;
    m_pmCtx.periodic_bc   = false;
    m_pmCtx.sr_enabled    = false;
    m_pmCtx.d_density      = nullptr;
    m_pmCtx.d_density2     = nullptr;
    m_pmCtx.d_padded       = nullptr;
    m_pmCtx.d_green_hat    = nullptr;
    m_pmCtx.d_rho_scratch  = nullptr;
    m_pmCtx.d_force_field  = nullptr;
    m_pmCtx.d_force_field2 = nullptr;
    m_pmCtx.d_sr_count    = nullptr;
    m_pmCtx.d_sr_start    = nullptr;
    m_pmCtx.d_sr_slot     = nullptr;
    m_pmCtx.d_sr_sorted   = nullptr;
    m_pmCtx.d_sr_orig     = nullptr;
    m_pmCtx.sr_r_split    = 0.0f;
    m_pmCtx.sr_r_cut      = 0.0f;
    m_pmCtx.sr_Nc         = 0;
}

Simulation::~Simulation()
{
    shutdown();
}

// ------------------------------------------------------------------
// Shutdown
// ------------------------------------------------------------------

void Simulation::shutdown()
{
    if (m_dPShared) { m_dPShared = nullptr; m_glSharing = false; }
    if (m_dPStart)  { cudaFree(m_dPStart);  m_dPStart  = nullptr; }
    if (m_dPEnd)    { cudaFree(m_dPEnd);    m_dPEnd    = nullptr; }
    if (m_dVStart)  { cudaFree(m_dVStart);  m_dVStart  = nullptr; }
    if (m_dVEnd)    { cudaFree(m_dVEnd);    m_dVEnd    = nullptr; }
    if (m_dA)       { cudaFree(m_dA);       m_dA       = nullptr; }
    if (m_dBlockMax){ cudaFree(m_dBlockMax);m_dBlockMax= nullptr; }
    m_glSharing = false;
    if (m_pmCtx.initialized) { pm_free(m_pmCtx); m_pmCtx.initialized = false; }
    if (m_treeCtx.initialized) { tree_free(m_treeCtx); m_treeCtx.initialized = false; }
}

// ------------------------------------------------------------------
// Init
// ------------------------------------------------------------------

bool Simulation::init(const char* kernelFile, const char* kernelName, int preferredDevice)
{
    (void)kernelFile;
    (void)kernelName;

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0)
    {
        std::cerr << "No CUDA device found: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        std::cout << "[CUDA] Device " << i << ": " << p.name
                  << " | SM " << p.major << "." << p.minor
                  << " | " << (p.totalGlobalMem >> 20) << " MB\n";
    }

    int chosenDevice = 0;
    if (preferredDevice >= 0 && preferredDevice < deviceCount)
    {
        chosenDevice = preferredDevice;
        std::cout << "[CUDA] Using forced device " << chosenDevice << "\n";
    }
    else
    {
        int bestSM = -1;
        size_t bestMem = 0;
        for (int i = 0; i < deviceCount; i++)
        {
            cudaDeviceProp p;
            cudaGetDeviceProperties(&p, i);
            int sm = p.major * 10 + p.minor;
            if (sm > bestSM || (sm == bestSM && p.totalGlobalMem > bestMem))
            {
                bestSM       = sm;
                bestMem      = p.totalGlobalMem;
                chosenDevice = i;
            }
        }
    }

    err = cudaSetDevice(chosenDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaSetDevice(" << chosenDevice << ") failed: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Synchronisation bloquante plutôt qu'en spin-wait : le CPU dort pendant les
    // synchros GPU au lieu de tourner à 100 % d'un cœur (CPU, chaleur, batterie).
    // Doit être posé AVANT la création du contexte (première allocation).
    err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (err != cudaSuccess)
        std::cerr << "[CUDA] cudaSetDeviceFlags(BlockingSync) failed: "
                  << cudaGetErrorString(err) << "\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, chosenDevice);
    std::cout << "[CUDA] Selected device " << chosenDevice << ": " << prop.name
              << " | SM " << prop.major << "." << prop.minor
              << " | " << (prop.totalGlobalMem >> 20) << " MB\n";
    return true;
}

// ------------------------------------------------------------------
// Parameters
// ------------------------------------------------------------------

bool Simulation::setParameters(int numStars, float timeStep, float softening, float grav)
{
    m_count    = numStars;
    m_timestep = (timeStep > 0.0f) ? timeStep : 1.0f;
    m_dtBase   = m_timestep;   // base dt for the slow-mo multiplier
    float eps  = (softening < 0.01f ? 0.01f : softening);
    m_soften   = eps * eps;
    m_grav     = grav;
    m_simTime  = 0.0f;
    m_currentFrame = 0;

    m_local  = 256;
    m_global = ((numStars + m_local - 1) / m_local) * m_local;

    m_blockX = (unsigned int)m_local;
    m_blockY = 1;
    m_blockZ = 1;
    m_gridX  = (unsigned int)((m_global + m_local - 1) / m_local);
    m_gridY  = 1;
    m_gridZ  = 1;

    return true;
}

void Simulation::setCosmology(float omegaM, float omegaL, float aStart)
{
    m_omegaM  = omegaM;
    m_omegaL  = omegaL;
    m_a       = aStart;
    float H0 = m_hubble;
    if (H0 < 1e-12f) H0 = 0.003f;
    // da/dt cosmologique = H0 × cosmoScale × a × sqrt(...)
    // cosmoScale découple la vitesse d'expansion Friedmann du N-body :
    // avec cosmoScale=0.02, l'expansion cosmologique est 50× plus lente,
    // laissant le temps aux structures de se former avant que a ne grandisse.
    float H0_cosmo = H0 * m_cosmoScale;
    m_da = m_a * H0_cosmo * sqrtf(m_omegaM / (m_a * m_a * m_a) + m_omegaL);
}

void Simulation::setDtAdaptive(float dtMin, float dtMax, float eta)
{
    m_adaptiveDt = true;
    m_dtMin = dtMin;
    m_dtMax = dtMax;
    m_eta   = eta;
}

void Simulation::setSpiralParams(float amplitude, float patternSpeed, float pitchTan, unsigned int arms)
{
    m_spAmplitude    = amplitude;
    m_spPatternSpeed = patternSpeed;
    m_spPitchTan     = pitchTan;
    m_spArms         = arms;
}

void Simulation::setPhysicsModel(int model)
{
    m_physicsModel = model;
}

// ------------------------------------------------------------------
// Upload
// ------------------------------------------------------------------

bool Simulation::uploadInitialData(const std::vector<cl_float4>& positions,
                                   const std::vector<cl_float4>& velocities)
{
    size_t bytes = sizeof(float4) * m_count;

    if (m_dPStart) cudaFree(m_dPStart);
    if (m_dPEnd)   cudaFree(m_dPEnd);
    if (m_dVStart) cudaFree(m_dVStart);
    if (m_dVEnd)   cudaFree(m_dVEnd);
    if (m_dA)      cudaFree(m_dA);
    if (m_dBlockMax) cudaFree(m_dBlockMax);
    cudaError_t err;

    // Block-max buffer for the GPU adaptive-dt reduction (256 threads/block).
    int redBlocks = (int)((m_count + 255) / 256);
    if (redBlocks < 1) redBlocks = 1;
    err = cudaMalloc(&m_dBlockMax, redBlocks * sizeof(float));
    if (err != cudaSuccess) { std::cerr << "cudaMalloc dBlockMax failed\n"; return false; }
    m_hBlockMax.resize(redBlocks);

    err = cudaMalloc(&m_dPStart, bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc dPStart failed\n"; return false; }
    err = cudaMalloc(&m_dPEnd,   bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc dPEnd failed\n"; return false; }
    err = cudaMalloc(&m_dVStart, bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc dVStart failed\n"; return false; }
    err = cudaMalloc(&m_dVEnd,   bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc dVEnd failed\n"; return false; }
    err = cudaMalloc(&m_dA,      bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc dA failed\n"; return false; }
    cudaMemset(m_dA, 0, bytes);  // zéro -> 1er pas en mode Tree (a_max=0 -> dtMax, pas de lecture non-init)

    err = cudaMemcpy(m_dPStart, positions.data(),  bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy dPStart failed\n"; return false; }
    err = cudaMemcpy(m_dVStart, velocities.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy dVStart failed\n"; return false; }

    return true;
}

// ------------------------------------------------------------------
//  Friedmann equation : evolve a(t)
// ------------------------------------------------------------------
void Simulation::stepFriedmann()
{
    if (m_a <= 0.0f) m_a = 1e-6f;

    // Equation de Friedmann avec cosmoScale² :
    //   a'' = (H0 × cosmoScale)² × ( -OmegaM/(2a²) + OmegaL × a )
    // cosmoScale ralentit l'évolution cosmologique sans affecter le N-body,
    // permettant la formation des structures avant l'expansion significative de a.
    float H0       = (m_hubble > 1e-12f) ? m_hubble : 0.003f;
    float H0_cosmo = H0 * m_cosmoScale;
    float H0sq     = H0_cosmo * H0_cosmo;

    float a_inv3 = 1.0f / (m_a * m_a * m_a);
    float dda    = H0sq * (-0.5f * m_omegaM * a_inv3 * m_a + m_omegaL * m_a);

    float da_half  = m_da + 0.5f * dda * m_timestep;
    float a_half   = m_a  + 0.5f * m_da * m_timestep;

    float a_inv3_h = 1.0f / (a_half * a_half * a_half);
    float dda_h    = H0sq * (-0.5f * m_omegaM * a_inv3_h * a_half + m_omegaL * a_half);

    m_a  = m_a  + da_half * m_timestep;
    m_da = m_da + dda_h   * m_timestep;
    if (m_a < 1e-8f) m_a = 1e-8f;
}

// For old state files that didn't store the scale factor: integrate the (smooth,
// particle-independent) Friedmann ODE from the current a (= aStart, set by
// setCosmology) up to targetSimTime, recovering a(t).
void Simulation::reconstructScaleFactor(float targetSimTime)
{
    if (m_timestep <= 0.0f) return;
    long n = (long)(targetSimTime / m_timestep + 0.5f);
    if (n < 0) n = 0;
    if (n > 50000000L) n = 50000000L;   // safety bound
    for (long i = 0; i < n; i++) stepFriedmann();
}

// ------------------------------------------------------------------
//  Adaptive timestep : dt = eta * sqrt(epsilon / |a_max|)
// ------------------------------------------------------------------
float Simulation::computeNextDt()
{
    if (!m_adaptiveDt || !m_dPStart || !m_dA) return m_timestep;

    dim3 grid(m_gridX, m_gridY, m_gridZ);
    dim3 block(m_blockX, m_blockY, m_blockZ);
    size_t sharedBytes = block.x * sizeof(float4);

    float a_inv3 = 1.0f / (m_a * m_a * m_a);

    // Compute forces pour l'estimation de a_max.
    // En mode TREE, m_dA contient déjà l'accélération du pas précédent (remplie par
    // launch_accel_from_dv après le tree kick) -> on la RÉUTILISE au lieu de refaire une
    // passe N² O(N²) à chaque pas (qui annulait tout le gain du Tree à grand N).
    // En N² pur (et PM), on calcule la passe directe comme avant.
    // Tree path (incl. forced tree for Yukawa/MOND) already filled m_dA via the
    // previous kick -> reuse it. Only the N²/PM paths need a fresh force pass.
    bool treeFillsAccel = m_useTree || m_physicsModel == 0 || m_physicsModel == 2 || m_physicsModel == 3;
    if (!treeFillsAccel) {
        launch_nbody_force(
            f4(m_dPStart), f4(m_dA),
            m_soften, m_grav, (unsigned int)m_count,
            a_inv3, m_physicsModel, m_boxSize * 0.5f,
            grid, block, sharedBytes);
    }

    // Max |acceleration| via a GPU reduction. Only the (small) per-block maxima are
    // copied back — NOT the full 16 MB acceleration array + a 1M-element CPU loop on
    // every step (which dominated the cost during long/sub-stepped runs).
    int redBlocks = (int)((m_count + 255) / 256);
    if (redBlocks < 1) redBlocks = 1;
    launch_reduce_max_accel(f4(m_dA), (unsigned int)m_count, (float*)m_dBlockMax, redBlocks);
    if ((int)m_hBlockMax.size() < redBlocks) m_hBlockMax.resize(redBlocks);
    cudaError_t err = cudaMemcpy(m_hBlockMax.data(), m_dBlockMax,
                                 redBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return m_timestep;

    float aMax = 0.0f;
    for (int b = 0; b < redBlocks; b++) if (m_hBlockMax[b] > aMax) aMax = m_hBlockMax[b];
    if (!std::isfinite(aMax)) return m_dtMin;   // NaN/Inf guard
    if (aMax < 1e-12f) return m_dtMax;

    float dt = m_eta * sqrtf(m_soften / aMax);
    if (!std::isfinite(dt)) return m_dtMin;
    dt = std::max(dt, m_dtMin);
    dt = std::min(dt, m_dtMax);

    // CFL constraint for PM: v_max × dt < cell_size.
    // With Hubble flow v = H0 × R, particles move H0×R×dt per step.
    // If this exceeds one PM cell, the CIC deposit becomes inaccurate → lateral propagation.
    // dt_CFL = cell_size / v_max = (box/Ng) / (H0 × R) = 3/(Ng × H0) [for box=3R].
    if (m_usePM && m_pmCtx.initialized && m_pmCtx.box_size > 0.0f) {
        float cell      = m_pmCtx.box_size / (float)m_pmCtx.Ng;
        float H_eff     = (m_hubble > 1e-12f) ? m_hubble : 0.001f;
        float v_max_hub = H_eff * m_pmCtx.box_size * 0.5f;  // Hubble v at box edge
        if (v_max_hub > 1e-6f) {
            float dt_cfl = cell / v_max_hub;
            if (dt_cfl < dt) dt = std::max(dt_cfl, m_dtMin);
        }
    }

    // Contrainte de stabilité Friedmann (avec H0² — même correction que stepFriedmann).
    if (m_a > 1e-6f) {
        float H0loc   = (m_hubble > 1e-12f) ? m_hubble : 0.003f;
        float H0sq    = H0loc * H0loc;
        float a_inv3_loc = 1.0f / (m_a * m_a * m_a);
        float dda_loc = H0sq * (-0.5f * m_omegaM * a_inv3_loc * m_a + m_omegaL * m_a);
        if (fabsf(dda_loc) > 1e-12f) {
            float dt_fried = sqrtf(0.2f * m_a / fabsf(dda_loc));
            if (dt_fried < dt)
                dt = std::max(dt_fried, 1e-5f);
        }
    }

    return dt;
}

// ------------------------------------------------------------------
//  Leapfrog DKD step with cosmology
// ------------------------------------------------------------------
bool Simulation::stepLeapfrog()
{
    if (!m_dPStart || !m_dPEnd) return false;

    // Force subcycling: full-force step every K steps (K=1 => every step).
    bool fullStep = (m_forceSubcycle <= 1) || (m_subCounter == 0);

    // Newton (0), Yukawa (2) and MOND (3) are implemented (correctly, positive
    // matter only) in the Barnes-Hut tree kernel -> force the tree path for them
    // (and disable PM, which can't do them). Janus (1) and Custom (4) keep the
    // user-selected path.
    bool nonStdModel = (m_physicsModel == 0 || m_physicsModel == 2 || m_physicsModel == 3);
    bool useTreeEff  = m_useTree || nonStdModel;
    bool usePMEff    = m_usePM && !nonStdModel;

    // Adaptive timestep — recomputed only on full-force steps, held on sub-steps.
    if (m_adaptiveDt && fullStep) m_dtBase = computeNextDt();
    m_timestep = m_dtBase * m_dtMul;   // apply slow-motion scaling

    // Advance scale factor
    stepFriedmann();

    // Temps physique (non comobile) : dt, pas dt/a.
    // dt/a = temps comobile → sature en univers dominé énergie noire.
    m_simTime += m_timestep;

    dim3 grid(m_gridX, m_gridY, m_gridZ);
    dim3 block(m_blockX, m_blockY, m_blockZ);
    size_t sharedBytes = block.x * sizeof(float4);

    if (useTreeEff) {
        // ------------------------------------------------------------------
        //  Tree path: pure GPU Barnes-Hut O(N log N) DKD leapfrog.
        //  Bypasses PM completely. Two trees (per species) for Janus.
        // ------------------------------------------------------------------
        float a_inv3   = 1.0f / (m_a * m_a * m_a);
        float H        = (m_a > 1.0e-8f) ? m_da / m_a : 0.0f;
        float dt_half  = m_timestep * 0.5f;
        float eps      = sqrtf(m_soften);

        if (!m_treeCtx.initialized || m_treeCtx.capacity < (int)m_count
                                   || fabsf(m_treeCtx.softening - eps) > 0.1f) {
            if (m_treeCtx.initialized) tree_free(m_treeCtx);
            tree_init(m_treeCtx, (int)m_count, eps);
        }
        m_treeCtx.theta             = m_treeTheta;  // live-tunable
        m_treeCtx.use_quadrupole    = m_treeQuad;
        m_treeCtx.use_warp_coherent = m_treeWarp;

        // D1: drift half-step (no wrap — open boundary in tree mode)
        launch_drift_half(f4(m_dPStart), f4(m_dPEnd), f4(m_dVStart),
                          (unsigned int)m_count, dt_half, 0.0f, grid, block);

        // K: force kick. Full step -> Barnes-Hut tree (then store the effective
        // accel for reuse). Sub-step -> reuse the stored accel (no tree build).
        if (fullStep) {
            cudaMemcpy(m_dVEnd, m_dVStart, sizeof(float4) * m_count, cudaMemcpyDeviceToDevice); // v_before
            tree_sr_kick(m_treeCtx, f4(m_dPEnd), f4(m_dVStart),
                         (unsigned int)m_count, m_grav, a_inv3, m_timestep,
                         m_physicsModel,
                         m_yukawaAlpha, m_yukawaLambda, m_mondA0);
            launch_accel_from_dv(f4(m_dVStart), f4(m_dVEnd), f4(m_dA),
                                 (unsigned int)m_count, m_timestep, grid, block);  // a = Δv/dt
            // Anti-monopole (Jeans swindle) : retire l'infall radial moyen de l'accélération
            // (par espèce) puis ré-applique le kick -> plus d'effondrement central des négatives.
            if (m_removeMonopole) {
                launch_remove_radial_monopole(f4(m_dA), f4(m_dPEnd),
                                              (unsigned int)m_count, m_monopoleFrac, grid, block);
                launch_v_from_base_accel(f4(m_dVStart), f4(m_dVEnd), f4(m_dA),
                                         (unsigned int)m_count, m_timestep, grid, block);
            }
        } else {
            launch_apply_accel_kick(f4(m_dVStart), f4(m_dA),
                                    (unsigned int)m_count, m_timestep, grid, block); // v += a·dt
        }

        // Hubble drag on v_start
        if (fabsf(H) > 1.0e-12f) {
            launch_hubble_drag(f4(m_dPEnd), f4(m_dVStart),
                               (unsigned int)m_count, m_timestep, H, grid, block);
        }

        // D2: second drift
        launch_drift_half(f4(m_dPEnd), f4(m_dPEnd), f4(m_dVStart),
                          (unsigned int)m_count, dt_half, 0.0f, grid, block);

        // Crash guard: sanitize NaN/Inf + cap runaways so the next tree build
        // never receives invalid coordinates (over-collapse -> freeze/crash).
        launch_sanitize(f4(m_dPEnd), f4(m_dVStart),
                        (unsigned int)m_count, 1.0e7f, 1.0e5f, grid, block);

        cudaMemcpy(m_dVEnd, m_dVStart, sizeof(float4) * m_count,
                   cudaMemcpyDeviceToDevice);

        // Advance the subcycle counter (tree path only).
        if (m_forceSubcycle > 1) m_subCounter = (m_subCounter + 1) % m_forceSubcycle;

    } else if (usePMEff) {
        // ------------------------------------------------------------------
        //  PM path: Particle-Mesh O(N + Ng^3 log Ng) DKD leapfrog
        //  Uses separate drift / force-kick / drag kernels.
        // ------------------------------------------------------------------
        float a_inv3 = 1.0f / (m_a * m_a * m_a);
        float H = (m_a > 1.0e-8f) ? m_da / m_a : 0.0f;
        float box_half = m_boxSize * 0.5f;
        float dt_half  = m_timestep * 0.5f;
        // Open-boundary PM: no periodic wrapping in drift — particles escape freely.
        // Wrapping would create an artificial cube-shell when fast particles hit the box edge.
        // Initialize PM context lazily (re-init if box, grid, softening, or mode changed)
        float box = (m_boxSize > 0.0f) ? m_boxSize : 2000.0f;
        float eps = sqrtf(m_soften);
        bool  per = m_pmPeriodic;
        if (!m_pmCtx.initialized || m_pmCtx.Ng != m_pmNg ||
            fabsf(m_pmCtx.box_size   - box) > 1.0f ||
            fabsf(m_pmCtx.softening  - eps) > 0.1f ||
            m_pmCtx.periodic_bc != per ||
            m_pmCtx.sr_enabled  != m_pmEnableSR)
        {
            if (m_pmCtx.initialized) pm_free(m_pmCtx);
            pm_init(m_pmCtx, m_pmNg, box, eps, per, m_pmEnableSR);
        }

        // Periodic PM: wrap particles at box boundary (torus topology).
        // Open-boundary PM: no wrap (particles drift freely out of the grid).
        float drift_box = per ? box_half : 0.0f;

        // D1: first half-drift  p_end = p_start + v_start * dt/2
        launch_drift_half(f4(m_dPStart), f4(m_dPEnd), f4(m_dVStart),
                          (unsigned int)m_count, dt_half, drift_box,
                          grid, block);

        // K: PM force kick on v_start (in-place) using drifted positions p_end
        pm_force_step(m_pmCtx, f4(m_dPEnd), f4(m_dVStart),
                      (unsigned int)m_count, m_timestep, m_grav, a_inv3,
                      m_physicsModel);
        // P³M: add short-range direct Janus force for r < softening.
        // Captures local ±mass separation that PM misses when ρ_net ≈ 0.
        pm_janus_sr_kick(m_pmCtx, f4(m_dPEnd), f4(m_dVStart),
                         (unsigned int)m_count, m_grav, a_inv3, m_timestep,
                         m_physicsModel);

        // Hubble drag (peculiar velocity only) on v_start (in-place)
        if (fabsf(H) > 1.0e-12f) {
            launch_hubble_drag(f4(m_dPEnd), f4(m_dVStart),
                               (unsigned int)m_count, m_timestep, H,
                               grid, block);
        }

        // D2: second half-drift  p_end = p_end + v_start * dt/2  (in-place)
        launch_drift_half(f4(m_dPEnd), f4(m_dPEnd), f4(m_dVStart),
                          (unsigned int)m_count, dt_half, drift_box,
                          grid, block);

        // v_end = v_start (already updated in-place above)
        cudaMemcpy(m_dVEnd, m_dVStart, sizeof(float4) * m_count,
                   cudaMemcpyDeviceToDevice);

        // NOTE: velocity re-centering deliberately removed.
        // Subtracting the mean velocity each step damps both positive-mass
        // infall AND negative-mass expansion — the two mechanisms that create
        // the Janus alveolar structure.  The lateral CM drift is physical
        // and expected in a finite sphere; it does not prevent the alveolar
        // pattern from forming over 7–14 Gyr.

    } else {
        // ------------------------------------------------------------------
        //  Direct N^2 path (original monolithic DKD kernel)
        // ------------------------------------------------------------------
        launch_nbody_cosmo(
            f4(m_dPStart), f4(m_dPEnd), f4(m_dVStart), f4(m_dVEnd),
            m_soften, m_timestep, m_grav, (unsigned int)m_count,
            m_a, m_da, m_omegaM, m_omegaL, m_rhoMean,
            m_physicsModel, m_boxSize * 0.5f, grid, block, sharedBytes);
    }

    std::swap(m_dPStart, m_dPEnd);
    std::swap(m_dVStart, m_dVEnd);

    m_currentFrame++;
    return true;
}

// ------------------------------------------------------------------
//  Main step dispatcher
// ------------------------------------------------------------------
bool Simulation::step()
{
    // PM always uses the leapfrog path (even with a=1 fixed, da=0).
    // The N² good Janus result used a=1 fixed (non-cosmological, pure
    // Janus gravity). stepLeapfrog handles this: a stays at 1, a_inv3=1,
    // H=0, no Hubble drag — identical to the N² non-cosmological mode.
    // Newton/Yukawa/MOND must use the leapfrog/tree path (the legacy nbody kernel
    // below does not implement the corrected, positive-matter-only models).
    bool nonStdModel = (m_physicsModel == 0 || m_physicsModel == 2 || m_physicsModel == 3);
    if (m_usePM || nonStdModel || fabsf(m_a - 1.0f) > 1e-4f || fabsf(m_da) > 1e-8f)
        return stepLeapfrog();

    // Fallback to legacy non-cosmological kernel (galaxy mode)
    if (!m_dPStart || !m_dPEnd) return false;
    m_timestep = m_dtBase * m_dtMul;   // apply slow-motion scaling
    m_simTime += m_timestep;

    dim3 grid(m_gridX, m_gridY, m_gridZ);
    dim3 block(m_blockX, m_blockY, m_blockZ);
    size_t sharedBytes = block.x * sizeof(float4);

    launch_nbody(
        f4(m_dPStart), f4(m_dPEnd), f4(m_dVStart), f4(m_dVEnd),
        m_soften, m_timestep, m_grav, (unsigned int)m_count,
        m_simTime, m_spAmplitude, m_spPatternSpeed, m_spPitchTan, m_spArms,
        m_physicsModel, m_hubble, 0.0f, grid, block, sharedBytes);

    std::swap(m_dPStart, m_dPEnd);
    std::swap(m_dVStart, m_dVEnd);

    m_currentFrame++;
    return true;
}

// ------------------------------------------------------------------
// Read back
// ------------------------------------------------------------------

bool Simulation::readPositions(std::vector<cl_float4>& out)
{
    out.resize(m_count);
    void* src = m_dPShared ? m_dPShared : m_dPStart;

    cudaError_t err = cudaMemcpy(out.data(), src, sizeof(float4) * m_count, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy readPositions failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    return true;
}

bool Simulation::readVelocities(std::vector<cl_float4>& out)
{
    out.resize(m_count);
    cudaError_t err = cudaMemcpy(out.data(), m_dVStart, sizeof(float4) * m_count, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy readVelocities failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    return true;
}

void Simulation::setSimTime(float t)
{
    m_simTime = t;
}

float Simulation::getHubbleParam() const
{
    if (m_a < 1e-8f) return 0.0f;
    return m_da / m_a;
}

// ------------------------------------------------------------------
// VRAM info
// ------------------------------------------------------------------

void Simulation::getVRAMInfo(size_t& freeBytes, size_t& totalBytes) const
{
    cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
    if (err != cudaSuccess) { freeBytes = 0; totalBytes = 0; }
}

// ------------------------------------------------------------------
// CUDA-OpenGL interop
// ------------------------------------------------------------------

bool Simulation::shareGLBuffer(unsigned int vbo)
{
    (void)vbo;
    return false;
}