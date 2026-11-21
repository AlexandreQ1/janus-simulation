#ifndef SIMULATION_H
# define SIMULATION_H

# include <vector>
# include <string>
# include <algorithm>

# include "cl_float4_compat.h"
# include "../gpu-nbody/pm_force.cuh"
# include "../gpu-nbody/treepm_force.cuh"

class Simulation
{
public:
    Simulation();
    ~Simulation();

    bool init(const char* kernelFile, const char* kernelName, int preferredDevice = -1);
    void shutdown();

    bool setParameters(int numStars, float timeStep, float softening, float grav);
    void setCosmology(float omegaM, float omegaL, float aStart);
    void setDtAdaptive(float dtMin, float dtMax, float eta);
    void setSpiralParams(float amplitude, float patternSpeed, float pitchTan, unsigned int arms);
    void setPhysicsModel(int model);
    void setHubble(float H0)         { m_hubble = H0; }
    void setCosmoScale(float scale)  { m_cosmoScale = std::max(1e-6f, std::min(1.0f, scale)); }
    void setRhoMean(float rho)       { m_rhoMean = rho; }
    void setBoxSize(float box)       { m_boxSize = box; }
    void setUsePM(bool use, int Ng = 128, bool periodic = false, bool enableSR = true)
         { m_usePM = use; m_pmNg = Ng; m_pmPeriodic = periodic; m_pmEnableSR = enableSR; }
    void setUseTree(bool use) { m_useTree = use; }
    // Anti-monopole (Jeans swindle) : retire l'infall radial moyen par espèce -> supprime
    // l'effondrement central de la masse négative (artefact de sphère finie).
    void setMonopoleRemoval(bool on, float frac) { m_removeMonopole = on; m_monopoleFrac = frac; }
    void setTreeTheta(float theta) {
        m_treeTheta = theta;
        if (m_treeCtx.initialized) m_treeCtx.theta = theta;
    }
    void setTreeQuadrupole(bool on)   { m_treeQuad = on; }
    void setTreeWarpCoherent(bool on) { m_treeWarp = on; }
    // Model parameters (Yukawa range/strength, MOND acceleration scale a0). Sim units.
    void setYukawa(float alpha, float lambda) { m_yukawaAlpha = alpha; m_yukawaLambda = lambda; }
    void setMondA0(float a0) { m_mondA0 = a0; }
    void setForceSubcycle(int k) { m_forceSubcycle = (k < 1) ? 1 : k; if (m_forceSubcycle == 1) m_subCounter = 0; }
    // Slow-motion: multiplies the integration dt (<1) so each frame advances a small
    // fraction of a step -> smooth slow-mo instead of skipping whole steps.
    void setDtMul(float m) { m_dtMul = (m > 0.0f) ? m : 1.0f; }
    bool isUsingPM() const { return m_usePM; }
    bool isUsingTree() const { return m_useTree; }
    float getSimTime() const { return m_simTime; }
    float getTimestep() const { return m_timestep; }   // current (adaptive) dt
    float getScaleFactor() const { return m_a; }
    float getHubbleParam() const;

    bool uploadInitialData(const std::vector<cl_float4>& positions,
                           const std::vector<cl_float4>& velocities);
    bool step();
    bool readPositions(std::vector<cl_float4>& out);
    bool readVelocities(std::vector<cl_float4>& out);
    void  setSimTime(float t);
    void  setTimestep(float dt) { if (dt > 0.0f) m_timestep = dt; }
    void  setScaleFactor(float a) { m_a = a; }
    float getDa() const { return m_da; }
    void  setCosmoState(float a, float da) { m_a = a; m_da = da; }  // restore on state load
    void  reconstructScaleFactor(float targetSimTime);             // recover a(t) for old state files

    // CUDA-OpenGL interop
    bool shareGLBuffer(unsigned int vbo);
    bool isGLSharing()  const { return m_glSharing; }
    bool hasSharedVBO() const { return m_dPShared != nullptr; }
    // Device buffers (current after the last step) for the interop vertex kernel.
    const void* getDevicePositions()  const { return m_dPStart; }
    const void* getDeviceVelocities() const { return m_dVStart; }
    int         getParticleCount()    const { return m_count; }

    int  getCurrentFrame() const { return m_currentFrame; }
    int  getTotalFrames()  const { return m_totalFrames; }
    void getVRAMInfo(size_t& freeBytes, size_t& totalBytes) const;

private:
    bool stepLeapfrog();   // DKD leapfrog with cosmology
    void stepFriedmann();  // evolve a(t)
    float computeNextDt(); // adaptive timestep from accelerations

    void* m_dPStart;
    void* m_dPEnd;
    void* m_dVStart;
    void* m_dVEnd;
    void* m_dPShared;   // mapped OpenGL VBO (interop, currently disabled)
    void* m_dA;         // accelerations buffer (float4)
    void* m_dBlockMax;  // per-block max|accel| for the GPU adaptive-dt reduction
    std::vector<float> m_hBlockMax;  // host staging for the (small) block maxima

    bool  m_removeMonopole = false;  // anti-monopole (Jeans swindle)
    float m_monopoleFrac   = 1.0f;   // fraction du monopôle radial soustraite (0..1)

    bool   m_glSharing;
    size_t m_count;
    size_t m_global;
    size_t m_local;
    float  m_soften;
    float  m_timestep;
    float  m_dtMin, m_dtMax, m_eta;
    float  m_grav;
    float  m_simTime;
    float  m_a, m_da;      // scale factor and da/dt
    float  m_omegaM, m_omegaL;
    bool   m_adaptiveDt;

    float        m_spAmplitude;
    float        m_spPatternSpeed;
    float        m_spPitchTan;
    unsigned int m_spArms;
    int          m_physicsModel;
    float        m_yukawaAlpha  = 0.5f;     // Yukawa strength  : F *= (1 + alpha·e^{-r/lambda})
    float        m_yukawaLambda = 500.0f;   // Yukawa range (sim units)
    float        m_mondA0       = 0.05f;    // MOND acceleration scale a0 (sim accel units)
    float        m_dtBase  = 1.0f;          // unscaled base dt (adaptive or fixed)
    float        m_dtMul   = 1.0f;          // slow-motion multiplier (<1 = slower)
    float        m_hubble;      // H0 reference in sim units
    float        m_cosmoScale;  // découple timescale Friedmann / N-body (0<s<=1)
    float        m_rhoMean;     // net mass density for δρ background correction
    float        m_boxSize;     // periodic box side length (0 = disabled, sphere mode)

    int m_currentFrame;
    int m_totalFrames;

    unsigned int m_gridX, m_gridY, m_gridZ;
    unsigned int m_blockX, m_blockY, m_blockZ;

    // Particle-Mesh solver (cuFFT)
    bool      m_usePM;
    bool      m_pmPeriodic;   // true = periodic cube (correct for Janus)
    bool      m_pmEnableSR;   // P³M short-range Ewald correction (vs pure PM)
    int       m_pmNg;
    PMContext m_pmCtx;

    // Barnes-Hut Tree (pure O(N log N), bypasses PM completely)
    bool        m_useTree;
    float       m_treeTheta;
    bool        m_treeQuad = true;   // order-2 multipole correction
    bool        m_treeWarp = true;   // warp-coherent traversal
    int         m_forceSubcycle = 1; // recompute tree force 1/K steps (K=1 = every step)
    int         m_subCounter = 0;    // 0 = full force step, else reuse stored accel
    TreeContext m_treeCtx;
};

#endif