// simulation.cpp — CUDA Runtime implementation

#include "simulation.h"
#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Helper to cast void* → float4*
static inline float4* f4(void* p) { return reinterpret_cast<float4*>(p); }

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

Simulation::Simulation()
    : m_dPStart(nullptr), m_dPEnd(nullptr), m_dVStart(nullptr)
    , m_dVEnd(nullptr), m_dPShared(nullptr)
    , m_glSharing(false)
    , m_count(0), m_global(0), m_local(256)
    , m_soften(10000.0f), m_timestep(1.0f), m_grav(0.0000000000667408f)
    , m_simTime(0.0f)
    , m_spAmplitude(0.0f), m_spPatternSpeed(0.01f), m_spPitchTan(0.268f), m_spArms(2)
    , m_physicsModel(1)
    , m_currentFrame(0), m_totalFrames(30)
    , m_gridX(1), m_gridY(1), m_gridZ(1)
    , m_blockX(256), m_blockY(1), m_blockZ(1)
    , m_useTree(false), m_treeTheta(0.5f)
{
    m_treeCtx.initialized = false;
}

Simulation::~Simulation()
{
    shutdown();
}

// ---------------------------------------------------------------------------
// Shutdown — free all CUDA allocations
// ---------------------------------------------------------------------------

void Simulation::shutdown()
{
    if (m_dPShared) { m_dPShared = nullptr; m_glSharing = false; }
    if (m_dPStart)  { cudaFree(m_dPStart);  m_dPStart  = nullptr; }
    if (m_dPEnd)    { cudaFree(m_dPEnd);    m_dPEnd    = nullptr; }
    if (m_dVStart)  { cudaFree(m_dVStart);  m_dVStart  = nullptr; }
    if (m_dVEnd)    { cudaFree(m_dVEnd);    m_dVEnd    = nullptr; }
    if (m_treeCtx.initialized) { tree_free(m_treeCtx); m_treeCtx.initialized = false; }
    m_glSharing = false;
}

// ---------------------------------------------------------------------------
// Init — select best CUDA device and verify availability
// ---------------------------------------------------------------------------

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

    // List all available CUDA devices
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
        // Auto-select: highest SM version first, then most VRAM as tiebreaker
        int bestSM     = -1;
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
    // synchros GPU au lieu de tourner à 100 % d'un cœur. Moins de CPU, moins de
    // chaleur dans un châssis partagé, au prix d'une latence de réveil minime.
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

// ---------------------------------------------------------------------------
// Parameters — compute launch configuration
// ---------------------------------------------------------------------------

bool Simulation::setParameters(int numStars, float timeStep, float softening, float grav)
{
    m_count    = numStars;
    m_timestep = (timeStep > 0.0f) ? timeStep : 1.0f;
    float eps  = (softening < 1.0f ? 1.0f : softening);
    m_soften   = eps * eps;  // kernel receives ε², UI exposes ε
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

// ---------------------------------------------------------------------------
// Upload initial data
// ---------------------------------------------------------------------------

bool Simulation::uploadInitialData(const std::vector<cl_float4>& positions,
                                   const std::vector<cl_float4>& velocities)
{
    size_t bytes = sizeof(float4) * m_count;

    if (m_dPStart) cudaFree(m_dPStart);
    if (m_dPEnd)   cudaFree(m_dPEnd);
    if (m_dVStart) cudaFree(m_dVStart);
    if (m_dVEnd)   cudaFree(m_dVEnd);

    cudaError_t err;

    err = cudaMalloc(&m_dPStart, bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc dPStart failed\n"; return false; }
    err = cudaMalloc(&m_dPEnd,   bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc dPEnd failed\n"; return false; }
    err = cudaMalloc(&m_dVStart, bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc dVStart failed\n"; return false; }
    err = cudaMalloc(&m_dVEnd,   bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc dVEnd failed\n"; return false; }

    err = cudaMemcpy(m_dPStart, positions.data(),  bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy dPStart failed\n"; return false; }
    err = cudaMemcpy(m_dVStart, velocities.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy dVStart failed\n"; return false; }

    return true;
}

// ---------------------------------------------------------------------------
// Step — N-body kernel launch (N² exact)
// ---------------------------------------------------------------------------

bool Simulation::step()
{
    if (!m_dPStart || !m_dPEnd) return false;

    m_simTime += m_timestep;

    dim3 grid(m_gridX, m_gridY, m_gridZ);
    dim3 block(m_blockX, m_blockY, m_blockZ);
    size_t sharedBytes = block.x * sizeof(float4);

    if (m_useTree) {
        // ── GPU Barnes-Hut Tree path (O(N log N), two-trees Janus) ────────
        // Symplectic DKD leapfrog : drift dt/2, kick dt, drift dt/2.
        // Symplectic property eliminates the energy drift seen with forward
        // Euler at large N (which manifested as artificial ring formation).
        float eps = sqrtf(m_soften);
        if (!m_treeCtx.initialized || m_treeCtx.capacity < (int)m_count
                                   || fabsf(m_treeCtx.softening - eps) > 0.1f) {
            if (m_treeCtx.initialized) tree_free(m_treeCtx);
            tree_init(m_treeCtx, (int)m_count, eps);
        }
        m_treeCtx.theta          = m_treeTheta;
        m_treeCtx.use_quadrupole = m_treeQuad;
        m_treeCtx.use_warp_coherent = m_treeWarp;
        float dt      = m_timestep;
        float dt_half = dt * 0.5f;

        // D1 : drift half-step : p_end = p_start + v_start * dt/2
        tree_drift_full(f4(m_dPStart), f4(m_dPEnd), f4(m_dVStart),
                        (unsigned int)m_count, dt_half);

        // K  : full kick using forces at p_end (mid-step positions).
        //      v_start updated in place. a_inv3 = 1.0 (no cosmology).
        tree_sr_kick(m_treeCtx, f4(m_dPEnd), f4(m_dVStart),
                     (unsigned int)m_count, m_grav, 1.0f, dt, m_physicsModel);

        // Spiral density-wave forcing (tree has no spiral term — add it here so
        // density waves work in Tree mode too). No-op when amplitude is 0.
        launch_spiral_kick(f4(m_dPEnd), f4(m_dVStart),
                           (unsigned int)m_count, m_grav, dt, m_simTime,
                           m_spAmplitude, m_spPatternSpeed, m_spPitchTan, m_spArms);

        // D2 : drift half-step : p_end = p_end + v_start * dt/2  (in place)
        tree_drift_full(f4(m_dPEnd), f4(m_dPEnd), f4(m_dVStart),
                        (unsigned int)m_count, dt_half);

        // v_end = v_start (just updated)
        cudaMemcpy(m_dVEnd, m_dVStart, sizeof(float4) * m_count,
                   cudaMemcpyDeviceToDevice);
    } else {
        launch_nbody(
            f4(m_dPStart), f4(m_dPEnd), f4(m_dVStart), f4(m_dVEnd),
            m_soften, m_timestep, m_grav, (unsigned int)m_count,
            m_simTime, m_spAmplitude, m_spPatternSpeed, m_spPitchTan, m_spArms,
            m_physicsModel, grid, block, sharedBytes);
    }

    std::swap(m_dPStart, m_dPEnd);
    std::swap(m_dVStart, m_dVEnd);

    m_currentFrame++;
    return true;
}

// ---------------------------------------------------------------------------
// Read back positions / velocities
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// VRAM info
// ---------------------------------------------------------------------------

void Simulation::getVRAMInfo(size_t& freeBytes, size_t& totalBytes) const
{
    cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
    if (err != cudaSuccess) { freeBytes = 0; totalBytes = 0; }
}

// ---------------------------------------------------------------------------
// CUDA-OpenGL interop (disabled — GL header conflict with CUDA)
// ---------------------------------------------------------------------------

bool Simulation::shareGLBuffer(unsigned int vbo)
{
    (void)vbo;
    return false;
}
