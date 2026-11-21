// volume_renderer.h — SPLATTING rendering (no voxels). Each particle = Gaussian
// colored by its velocity (nebula palette), accumulated in HDR then tone-mapped.
#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>

class VolumeRenderer
{
public:
    bool init();
    void shutdown();

    void uploadSpeeds(const float* data, int n);   // velocity per particle -> VBO

    // Halo: additive Gaussian splats colored by velocity -> HDR FBO -> tone-map
    // (hue preservation + whitening) composited into the current FBO.
    // posVbo = PointRenderer VBO (6 floats/vertex: pos offset 0, size-sign offset 5).
    void renderHalo(unsigned posVbo, int count, int fbW, int fbH,
                    const glm::mat4& mvp, float velRef, const glm::vec3 pal[32],
                    float haloSize, float intensity, float exposure,
                    bool colorByVel, bool showNeg);

    // Stars: small sharp additive splats directly into the current FBO.
    void renderStars(unsigned posVbo, int count, const glm::mat4& mvp,
                     float velRef, const glm::vec3 pal[32],
                     float pointSize, float brightness, bool colorByVel, bool showNeg);

private:
    void ensureHDR(int w, int h);
    void drawSplats(unsigned posVbo, int count, const glm::mat4& mvp, float velRef,
                    const glm::vec3 pal[32], float size, float intensity,
                    bool perspective, bool colorByVel, bool showNeg);

    GLuint m_splatProg = 0, m_toneProg = 0;
    GLuint m_hdrFbo = 0, m_hdrTex = 0;
    int    m_hw = 0, m_hh = 0;
    GLuint m_vao = 0, m_fsvao = 0;
    GLuint m_speedVbo = 0;
    int    m_speedN = 0;
};
