#ifndef SIMULATION_H
# define SIMULATION_H

# include <vector>
# include <string>

# include "cl_float4_compat.h"
# include "../gpu-nbody/treepm_force.cuh"

class Simulation
{
public:
    Simulation();
    ~Simulation();

    bool init(const char* kernelFile, const char* kernelName, int preferredDevice = -1);
    void shutdown();

    bool setParameters(int numStars, float timeStep, float softening, float grav);
    void setSpiralParams(float amplitude, float patternSpeed, float pitchTan, unsigned int arms);
    void setPhysicsModel(int model);  // 0=Newton 1=Janus 2=Yukawa 3=MOND 4=Custom
    bool uploadInitialData(const std::vector<cl_float4>& positions,
                           const std::vector<cl_float4>& velocities);
    bool step();
    bool readPositions(std::vector<cl_float4>& out);
    bool readVelocities(std::vector<cl_float4>& out);
    void  setSimTime(float t);
    void  setTimestep(float dt) { if (dt > 0.0f) m_timestep = dt; }
    float getSimTime() const { return m_simTime; }

    // GPU Barnes-Hut Tree (O(N log N) replacement for direct N²)
    void  setUseTree(bool use) { m_useTree = use; }
    void  setTreeTheta(float th) {
        m_treeTheta = th;
        if (m_treeCtx.initialized) m_treeCtx.theta = th;
    }
    void  setTreeQuadrupole(bool on)   { m_treeQuad = on; }
    void  setTreeWarpCoherent(bool on) { m_treeWarp = on; }
    bool  isUsingTree() const { return m_useTree; }

    // CUDA-OpenGL interop
    bool shareGLBuffer(unsigned int vbo);
    bool isGLSharing()  const { return m_glSharing; }
    bool hasSharedVBO() const { return m_dPShared != nullptr; }
    // Device buffers (current after the last step's swap) for the interop vertex kernel.
    const void* getDevicePositions()  const { return m_dPStart; }
    const void* getDeviceVelocities() const { return m_dVStart; }
    int         getParticleCount()    const { return m_count; }

    int  getCurrentFrame() const { return m_currentFrame; }
    int  getTotalFrames()  const { return m_totalFrames; }
    void getVRAMInfo(size_t& freeBytes, size_t& totalBytes) const;

private:
    // Device pointers stored as void* to avoid CUDA headers in this header
    void* m_dPStart;
    void* m_dPEnd;
    void* m_dVStart;
    void* m_dVEnd;
    void* m_dPShared;   // mapped OpenGL VBO (interop, currently disabled)

    bool   m_glSharing;

    size_t m_count;
    size_t m_global;
    size_t m_local;
    float  m_soften;
    float  m_timestep;
    float  m_grav;
    float  m_simTime;

    float        m_spAmplitude;
    float        m_spPatternSpeed;
    float        m_spPitchTan;
    unsigned int m_spArms;
    int          m_physicsModel;

    int m_currentFrame;
    int m_totalFrames;

    unsigned int m_gridX, m_gridY, m_gridZ;
    unsigned int m_blockX, m_blockY, m_blockZ;

    // Barnes-Hut Tree (GPU LBVH, two-trees Janus)
    bool        m_useTree;
    float       m_treeTheta;
    bool        m_treeQuad = true;   // order-2 multipole correction
    bool        m_treeWarp = true;   // warp-coherent traversal
    TreeContext m_treeCtx;
};

#endif
