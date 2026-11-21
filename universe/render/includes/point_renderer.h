#ifndef POINT_RENDERER_H
# define POINT_RENDERER_H

# include <GL/glew.h>
# include <string>
# include <vector>
# include <glm/glm.hpp>
# include <glm/gtc/matrix_transform.hpp>

class PointRenderer
{
public:
    PointRenderer(int width, int height);
    ~PointRenderer();

    bool init();
    void resize(int width, int height);

    void beginFrame();
    void uploadParticles(const std::vector<float>& data, size_t count, bool subData = false);
    // Allocate the VBO to `count` vertices WITHOUT CPU data, for the CUDA-GL interop
    // path (a kernel fills it directly). Also sets the draw count.
    void reserveVBO(size_t count);
    void setDrawCount(size_t count) { m_particleCount = count; }
    void render(const glm::mat4& viewProj, float pointSize, float particleSizeNeg,
                const glm::vec3 posPalette[32], const glm::vec3 negPalette[32],
                float maxMass, bool showNegaMass = true, float galaxyRadius = 0.0f);

    // ── 3D Stereo rendering ──────────────────────────────────────────────────
    // stereoMode: 0=mono, 1=anaglyphe, 2=side-by-side
    // ipd: inter-pupillary distance in simulation units
    void renderStereo(const glm::mat4& viewProj,
                      const glm::vec3& camRight,
                      float pointSize, float particleSizeNeg,
                      const glm::vec3 posPalette[32], const glm::vec3 negPalette[32],
                      float maxMass, bool showNegaMass, float galaxyRadius,
                      int stereoMode, float ipd);
    void endFrame();

    void setGlow(bool enabled, float blurRadius, float glowIntensity);
    void setSoftHalo(bool on) { m_softHalo = on ? 1.0f : 0.0f; }  // noyau doux gaussien (rendu CFD/SPH)
    // SPH splat resolution scale (0.4-1.0). <1 renders the density splat at lower
    // res -> much less overdraw (4x at 0.5) -> faster. Smooth field hides the loss.
    void setSPHScale(float s);

    // Rendu SPH volumétrique (style py-sphviewer/Illustris) : chaque particule est
    // splattée par un noyau gaussien de TAILLE ADAPTATIVE (longueur de lissage, attr 3)
    // dans un buffer flottant. R=Σw·valeur, G=Σw. Puis : teinte = propriété MOYENNE
    // (R/G) via palette, luminosité = densité de colonne (log G). Champ continu lisse.
    // Remplace render() quand activé ; le glow s'applique par-dessus.
    void renderSPH(const glm::mat4& viewProj, float ptSize, bool showNegaMass,
                   const glm::vec3 palette[32], float valMax, float lumScale,
                   bool colorByDensity = false, float fovDeg = 45.0f, float smoothPx = 0.0f,
                   float asinhCeil = 40.0f);

    // Plans de coupe X/Y/Z : pour chaque axe, enable (0/1), position, sign (+1=masque
    // au-dessus, -1=en-dessous). Une particule est masquée si UN axe actif la coupe.
    void setClipBox(const float en[3], const float pos[3], const float sign[3]) {
        for (int k=0;k<3;k++){ m_clipEn[k]=en[k]; m_clipPos3[k]=pos[k]; m_clipSign[k]=sign[k]; }
    }

    GLuint getFrameTexture() const { return m_glowEnabled ? m_glowTex : m_colorTex; }
    GLuint getVBO()    const { return m_vbo; }
    int getWidth()  const { return m_width; }
    int getHeight() const { return m_height; }

    // Pick a particle by screen coordinates. Returns -1 if none, or particle index.
    int pickParticle(const glm::mat4& viewProj, float pointSize, float particleSizeNeg,
                     float maxMass, bool showNegaMass, int screenX, int screenY, float galaxyRadius = 0.0f);

private:
    bool        compileShader(GLuint shader, const std::string& path);
    bool        linkProgram(GLuint program);
    std::string loadFile(const std::string& path);
    void        applyGlow();
    void        createGlowFBOs(int w, int h);
    void        deleteGlowFBOs();
    void        createDensityFBO(int w, int h);
    void        deleteDensityFBO();

    int    m_width;
    int    m_height;
    size_t m_particleCount;

    GLuint m_fbo;
    GLuint m_colorTex;

    GLuint m_vao, m_vbo;
    GLuint m_program, m_vertShader, m_fragShader;
    GLint  m_uViewProj, m_uPointSize, m_uPointSizeNeg, m_uPosPalette, m_uNegPalette, m_uMaxMass;
    GLint  m_uShowNegaMass;
    GLint  m_uClipEnable = -1, m_uClipPos = -1, m_uClipSign = -1;          // plans de coupe (points)
    GLint  m_uDsClipEnable = -1, m_uDsClipPos = -1, m_uDsClipSign = -1;    // plans de coupe (splat SPH)
    // État des 3 plans de coupe (X/Y/Z), partagé par les deux programmes
    float  m_clipEn[3]   = {0,0,0};
    float  m_clipPos3[3] = {0,0,0};
    float  m_clipSign[3] = {1,1,1};
    GLint  m_uSoftHalo;      // soft gaussian sprite (CFD look)
    float  m_softHalo = 0.0f;
    GLint  m_uGalaxyRadius;  // for negative mass transparency in disk
    GLint  m_uPickMode;      // marker picking

    bool   m_glowEnabled;
    float  m_blurRadius;
    float  m_glowIntensity;

    GLuint m_dummyVao;

    // Density-field pipeline (HDR splat -> log colormap)
    GLuint m_densFbo = 0, m_densTex = 0;
    float  m_densScale = 1.0f;       // splat resolution scale (<1 = faster)
    int    m_densW = 0, m_densH = 0; // actual density-buffer dimensions
    GLuint m_densSplatProgram = 0, m_densSplatVert = 0, m_densSplatFrag = 0;
    GLint  m_uDsViewProj = -1, m_uDsPtSize = -1, m_uDsShowNeg = -1, m_uDsSizeScale = -1;
    GLuint m_densColorProgram = 0, m_densColorFrag = 0;
    GLint  m_uDcAccum = -1, m_uDcPalette = -1, m_uDcValMax = -1, m_uDcLumScale = -1, m_uDcColorByDensity = -1, m_uDcAccumBlur = -1, m_uDcAsinhCeil = -1;

    GLuint m_blurFbo1;
    GLuint m_blurTex1;
    GLuint m_blurFbo2;
    GLuint m_blurTex2;
    GLuint m_glowFbo;
    GLuint m_glowTex;

    // Picking FBO (same size, separate from main FBO)
    GLuint m_pickTex;
    GLuint m_pickFbo;

    GLuint m_blurProgram;
    GLuint m_blurVert, m_blurFrag;
    GLint  m_uBlurTex, m_uBlurDir;

    GLuint m_compositeProgram;
    GLuint m_compositeFrag;
    GLint  m_uCompScene, m_uCompGlow, m_uCompIntensity;
    // ── 3D Stereo uniforms ─────────────────────────────────────────────────
    GLint  m_uStereoMode;
    GLint  m_uCompSceneLeft, m_uCompSceneRight;
    GLint  m_uCompGlowLeft, m_uCompGlowRight;

    // ── Stereo FBOs ─────────────────────────────────────────────────────────
    GLuint m_fboLeft, m_colorTexLeft;
    GLuint m_fboRight, m_colorTexRight;
    GLuint m_blurFbo1L, m_blurTex1L, m_blurFbo2L, m_blurTex2L, m_glowFboL, m_glowTexL;
    GLuint m_blurFbo1R, m_blurTex1R, m_blurFbo2R, m_blurTex2R, m_glowFboR, m_glowTexR;

    void createStereoFBOs(int w, int h);
    void deleteStereoFBOs();
    void applyGlowStereo(GLuint srcTex, GLuint dstFbo1, GLuint dstTex1,
                         GLuint dstFbo2, GLuint dstTex2);
    void renderEye(const glm::mat4& viewProj, float pointSize, float particleSizeNeg,
                   const glm::vec3 posPalette[32], const glm::vec3 negPalette[32],
                   float maxMass, bool showNegaMass, float galaxyRadius,
                   GLuint fbo);
};

#endif
