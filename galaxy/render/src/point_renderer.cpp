#include "../includes/point_renderer.h"
#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

PointRenderer::PointRenderer(int width, int height)
    : m_width(width), m_height(height), m_particleCount(0)
    , m_fbo(0), m_colorTex(0)
    , m_vao(0), m_vbo(0)
    , m_program(0), m_vertShader(0), m_fragShader(0)
    , m_uViewProj(-1), m_uPointSize(-1), m_uPointSizeNeg(-1), m_uPosPalette(-1)
    , m_uNegPalette(-1), m_uMaxMass(-1), m_uShowNegaMass(-1)
    , m_uGalaxyRadius(-1)
    , m_uPickMode(-1)
    , m_glowEnabled(false), m_blurRadius(2.0f), m_glowIntensity(1.5f)
    , m_dummyVao(0)
    , m_blurFbo1(0), m_blurTex1(0)
    , m_blurFbo2(0), m_blurTex2(0)
    , m_glowFbo(0), m_glowTex(0)
    , m_pickTex(0), m_pickFbo(0)
    , m_blurProgram(0), m_blurVert(0), m_blurFrag(0)
    , m_uBlurTex(-1), m_uBlurDir(-1)
    , m_compositeProgram(0), m_compositeFrag(0)
    , m_uCompScene(-1), m_uCompGlow(-1), m_uCompIntensity(-1)
    , m_uStereoMode(-1)
    , m_uCompSceneLeft(-1), m_uCompSceneRight(-1)
    , m_uCompGlowLeft(-1), m_uCompGlowRight(-1)
    , m_fboLeft(0), m_colorTexLeft(0)
    , m_fboRight(0), m_colorTexRight(0)
    , m_blurFbo1L(0), m_blurTex1L(0), m_blurFbo2L(0), m_blurTex2L(0)
    , m_glowFboL(0), m_glowTexL(0)
    , m_blurFbo1R(0), m_blurTex1R(0), m_blurFbo2R(0), m_blurTex2R(0)
    , m_glowFboR(0), m_glowTexR(0)
{}

PointRenderer::~PointRenderer()
{
    if (m_vbo)     glDeleteBuffers(1, &m_vbo);
    if (m_vao)     glDeleteVertexArrays(1, &m_vao);
    if (m_dummyVao) glDeleteVertexArrays(1, &m_dummyVao);

    if (m_program) {
        glDetachShader(m_program, m_vertShader);
        glDetachShader(m_program, m_fragShader);
        glDeleteShader(m_vertShader);
        glDeleteShader(m_fragShader);
        glDeleteProgram(m_program);
    }

    if (m_colorTex) glDeleteTextures(1, &m_colorTex);
    if (m_fbo)      glDeleteFramebuffers(1, &m_fbo);
    if (m_pickTex)  glDeleteTextures(1, &m_pickTex);
    if (m_pickFbo)  glDeleteFramebuffers(1, &m_pickFbo);

    deleteGlowFBOs();
    deleteDensityFBO();
    if (m_densSplatProgram) glDeleteProgram(m_densSplatProgram);
    if (m_densColorProgram) glDeleteProgram(m_densColorProgram);

    if (m_blurProgram) {
        glDetachShader(m_blurProgram, m_blurVert);
        glDetachShader(m_blurProgram, m_blurFrag);
        glDeleteShader(m_blurVert);
        glDeleteShader(m_blurFrag);
        glDeleteProgram(m_blurProgram);
    }
    if (m_compositeProgram) {
        glDetachShader(m_compositeProgram, m_blurVert);
        glDetachShader(m_compositeProgram, m_compositeFrag);
        glDeleteShader(m_compositeFrag);
        glDeleteProgram(m_compositeProgram);
    }
}

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------

std::string PointRenderer::loadFile(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[PointRenderer] Cannot open: " << path << "\n";
        return "";
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool PointRenderer::compileShader(GLuint shader, const std::string& path)
{
    std::string src = loadFile(path);
    if (src.empty()) return false;
    const char* csrc = src.c_str();
    glShaderSource(shader, 1, &csrc, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "[PointRenderer] Compile error in " << path << ":\n" << log << "\n";
        return false;
    }
    return true;
}

bool PointRenderer::linkProgram(GLuint program)
{
    glLinkProgram(program);
    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::cerr << "[PointRenderer] Link error:\n" << log << "\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Glow FBO helpers
// ---------------------------------------------------------------------------

void PointRenderer::createGlowFBOs(int w, int h)
{
    // blurTex1 / blurFbo1
    glGenTextures(1, &m_blurTex1);
    glBindTexture(GL_TEXTURE_2D, m_blurTex1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &m_blurFbo1);
    glBindFramebuffer(GL_FRAMEBUFFER, m_blurFbo1);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_blurTex1, 0);

    // blurTex2 / blurFbo2
    glGenTextures(1, &m_blurTex2);
    glBindTexture(GL_TEXTURE_2D, m_blurTex2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &m_blurFbo2);
    glBindFramebuffer(GL_FRAMEBUFFER, m_blurFbo2);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_blurTex2, 0);

    // glowTex / glowFbo
    glGenTextures(1, &m_glowTex);
    glBindTexture(GL_TEXTURE_2D, m_glowTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &m_glowFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_glowFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_glowTex, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void PointRenderer::deleteGlowFBOs()
{
    if (m_blurTex1) { glDeleteTextures(1, &m_blurTex1); m_blurTex1 = 0; }
    if (m_blurFbo1) { glDeleteFramebuffers(1, &m_blurFbo1); m_blurFbo1 = 0; }
    if (m_blurTex2) { glDeleteTextures(1, &m_blurTex2); m_blurTex2 = 0; }
    if (m_blurFbo2) { glDeleteFramebuffers(1, &m_blurFbo2); m_blurFbo2 = 0; }
    if (m_glowTex)  { glDeleteTextures(1, &m_glowTex);  m_glowTex  = 0; }
    if (m_glowFbo)  { glDeleteFramebuffers(1, &m_glowFbo);  m_glowFbo  = 0; }
    if (m_pickTex)  { glDeleteTextures(1, &m_pickTex);  m_pickTex  = 0; }
    if (m_pickFbo)  { glDeleteFramebuffers(1, &m_pickFbo);  m_pickFbo  = 0; }
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

bool PointRenderer::init()
{
    // ---- particle program ----
    m_vertShader = glCreateShader(GL_VERTEX_SHADER);
    m_fragShader = glCreateShader(GL_FRAGMENT_SHADER);

    if (!compileShader(m_vertShader, "render/shaders/point.vert")) return false;
    if (!compileShader(m_fragShader, "render/shaders/point.frag")) return false;

    m_program = glCreateProgram();
    glAttachShader(m_program, m_vertShader);
    glAttachShader(m_program, m_fragShader);
    if (!linkProgram(m_program)) return false;

    m_uViewProj    = glGetUniformLocation(m_program, "uViewProj");
    m_uPointSize   = glGetUniformLocation(m_program, "uPointSize");
    m_uPointSizeNeg= glGetUniformLocation(m_program, "uPointSizeNeg");
    m_uPosPalette  = glGetUniformLocation(m_program, "uPosPalette");
    m_uNegPalette  = glGetUniformLocation(m_program, "uNegPalette");
    m_uMaxMass     = glGetUniformLocation(m_program, "uMaxMass");
    m_uShowNegaMass= glGetUniformLocation(m_program, "uShowNegaMass");
    m_uGalaxyRadius= glGetUniformLocation(m_program, "uGalaxyRadius");
    m_uPickMode    = glGetUniformLocation(m_program, "uPickMode");

    // ---- VAO / VBO ----
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          6 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE,
                          6 * sizeof(float), (void*)(3 * sizeof(float)));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE,
                          6 * sizeof(float), (void*)(4 * sizeof(float)));

    glEnableVertexAttribArray(3);   // adaptive size (SPH smoothing length)
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE,
                          6 * sizeof(float), (void*)(5 * sizeof(float)));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // ---- off-screen FBO for particles ----
    glGenTextures(1, &m_colorTex);
    glBindTexture(GL_TEXTURE_2D, m_colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F,
                 m_width, m_height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, m_colorTex, 0);

    // ---- picking FBO (RGBA8 for encoded IDs) ----
    glGenTextures(1, &m_pickTex);
    glBindTexture(GL_TEXTURE_2D, m_pickTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &m_pickFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_pickFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, m_pickTex, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "[PointRenderer] Picking FBO incomplete.\n";
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // ---- dummy VAO for full-screen quad passes ----
    glGenVertexArrays(1, &m_dummyVao);

    // ---- blur program ----
    m_blurVert = glCreateShader(GL_VERTEX_SHADER);
    m_blurFrag = glCreateShader(GL_FRAGMENT_SHADER);

    if (!compileShader(m_blurVert, "render/shaders/blur.vert")) return false;
    if (!compileShader(m_blurFrag, "render/shaders/blur.frag")) return false;

    m_blurProgram = glCreateProgram();
    glAttachShader(m_blurProgram, m_blurVert);
    glAttachShader(m_blurProgram, m_blurFrag);
    if (!linkProgram(m_blurProgram)) return false;

    m_uBlurTex = glGetUniformLocation(m_blurProgram, "uTex");
    m_uBlurDir = glGetUniformLocation(m_blurProgram, "uDir");

    // ---- composite program ----
    m_compositeFrag = glCreateShader(GL_FRAGMENT_SHADER);
    if (!compileShader(m_compositeFrag, "render/shaders/composite.frag")) return false;

    m_compositeProgram = glCreateProgram();
    glAttachShader(m_compositeProgram, m_blurVert);
    glAttachShader(m_compositeProgram, m_compositeFrag);
    if (!linkProgram(m_compositeProgram)) return false;

    m_uCompScene     = glGetUniformLocation(m_compositeProgram, "uScene");
    m_uCompGlow      = glGetUniformLocation(m_compositeProgram, "uGlow");
    m_uCompIntensity = glGetUniformLocation(m_compositeProgram, "uGlowIntensity");

    // ---- density splat program (adaptive gaussian splat -> RG accumulation) ----
    m_densSplatVert = glCreateShader(GL_VERTEX_SHADER);
    m_densSplatFrag = glCreateShader(GL_FRAGMENT_SHADER);
    if (!compileShader(m_densSplatVert, "render/shaders/density_splat.vert")) return false;
    if (!compileShader(m_densSplatFrag, "render/shaders/density_splat.frag")) return false;
    m_densSplatProgram = glCreateProgram();
    glAttachShader(m_densSplatProgram, m_densSplatVert);
    glAttachShader(m_densSplatProgram, m_densSplatFrag);
    if (!linkProgram(m_densSplatProgram)) return false;
    m_uDsViewProj  = glGetUniformLocation(m_densSplatProgram, "uViewProj");
    m_uDsPtSize    = glGetUniformLocation(m_densSplatProgram, "uPtSize");
    m_uDsShowNeg   = glGetUniformLocation(m_densSplatProgram, "uShowNeg");
    m_uDsSizeScale = glGetUniformLocation(m_densSplatProgram, "uSizeScale");

    // ---- density colormap program (fullscreen blur.vert + density_color.frag) ----
    m_densColorFrag = glCreateShader(GL_FRAGMENT_SHADER);
    if (!compileShader(m_densColorFrag, "render/shaders/density_color.frag")) return false;
    m_densColorProgram = glCreateProgram();
    glAttachShader(m_densColorProgram, m_blurVert);
    glAttachShader(m_densColorProgram, m_densColorFrag);
    if (!linkProgram(m_densColorProgram)) return false;
    m_uDcAccum          = glGetUniformLocation(m_densColorProgram, "uAccum");
    m_uDcPalette        = glGetUniformLocation(m_densColorProgram, "uPalette");
    m_uDcValMax         = glGetUniformLocation(m_densColorProgram, "uValMax");
    m_uDcLumScale       = glGetUniformLocation(m_densColorProgram, "uLumScale");
    m_uDcColorByDensity = glGetUniformLocation(m_densColorProgram, "uColorByDensity");
    m_uDcAccumBlur      = glGetUniformLocation(m_densColorProgram, "uAccumBlur");
    m_uDcAsinhCeil      = glGetUniformLocation(m_densColorProgram, "uAsinhCeil");

    // ---- glow + density FBOs ----
    createGlowFBOs(m_width, m_height);
    createDensityFBO(m_width, m_height);

    return true;
}

// ── SPH density-field FBO (HDR splat target) ─────────────────────────────────
void PointRenderer::createDensityFBO(int w, int h)
{
    m_densW = (int)(w * m_densScale); if (m_densW < 1) m_densW = 1;
    m_densH = (int)(h * m_densScale); if (m_densH < 1) m_densH = 1;
    w = m_densW; h = m_densH;
    glGenTextures(1, &m_densTex);
    glBindTexture(GL_TEXTURE_2D, m_densTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &m_densFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_densFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_densTex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void PointRenderer::deleteDensityFBO()
{
    if (m_densTex) { glDeleteTextures(1, &m_densTex); m_densTex = 0; }
    if (m_densFbo) { glDeleteFramebuffers(1, &m_densFbo); m_densFbo = 0; }
}

void PointRenderer::setSPHScale(float s)
{
    if (s < 0.25f) s = 0.25f; if (s > 1.0f) s = 1.0f;
    if (fabsf(s - m_densScale) < 0.01f) return;
    m_densScale = s;
    deleteDensityFBO();
    createDensityFBO(m_width, m_height);
}

// ── SPH volumetric render : adaptive splat -> blur -> log colormap ───────────
void PointRenderer::renderSPH(const glm::mat4& viewProj, float ptSize, bool showNegaMass,
                              const glm::vec3 palette[32], float valMax, float lumScale,
                              bool colorByDensity, float fovDeg, float smoothPx, float asinhCeil)
{
    if (m_particleCount == 0) return;

    // 1) Additive gaussian splat into the (scaled) HDR density buffer.
    glBindFramebuffer(GL_FRAMEBUFFER, m_densFbo);
    glViewport(0, 0, m_densW, m_densH);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    glUseProgram(m_densSplatProgram);
    glUniformMatrix4fv(m_uDsViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
    glUniform1f(m_uDsPtSize, ptSize);
    glUniform1f(m_uDsShowNeg, showNegaMass ? 1.0f : 0.0f);
    float sizeScale = 0.5f * (float)m_densH / tanf(fovDeg * 3.14159265f / 360.0f);
    glUniform1f(m_uDsSizeScale, sizeScale);
    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(m_particleCount));
    glBindVertexArray(0);

    glDisable(GL_BLEND);
    glDisable(GL_PROGRAM_POINT_SIZE);

    // 1b) Detail-preserving blur into a SEPARATE buffer (m_glowTex). Keeps densTex sharp.
    bool haveBlur = (smoothPx > 0.01f);
    if (haveBlur) {
        int   iters = 1 + (int)(smoothPx);
        float sx = smoothPx / (float)m_width;
        float sy = smoothPx / (float)m_height;
        glUseProgram(m_blurProgram);
        glUniform1i(m_uBlurTex, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindVertexArray(m_dummyVao);
        for (int it = 0; it < iters; ++it) {
            GLuint srcTex = (it == 0) ? m_densTex : m_glowTex;
            glBindFramebuffer(GL_FRAMEBUFFER, m_blurFbo1);
            glViewport(0, 0, m_width, m_height);
            glBindTexture(GL_TEXTURE_2D, srcTex);
            glUniform2f(m_uBlurDir, sx, 0.0f);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glBindFramebuffer(GL_FRAMEBUFFER, m_glowFbo);
            glBindTexture(GL_TEXTURE_2D, m_blurTex1);
            glUniform2f(m_uBlurDir, 0.0f, sy);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        glBindVertexArray(0);
    }

    // 2) Colormap : hue = density (palette), brightness = log column density.
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glViewport(0, 0, m_width, m_height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(m_densColorProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_densTex);
    glUniform1i(m_uDcAccum, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, haveBlur ? m_glowTex : m_densTex);
    glUniform1i(m_uDcAccumBlur, 1);
    glUniform3fv(m_uDcPalette, 32, glm::value_ptr(palette[0]));
    glUniform1f(m_uDcValMax, valMax);
    glUniform1f(m_uDcLumScale, lumScale);
    glUniform1i(m_uDcColorByDensity, colorByDensity ? 1 : 0);
    glUniform1f(m_uDcAsinhCeil, asinhCeil);
    glBindVertexArray(m_dummyVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    glUseProgram(0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ---------------------------------------------------------------------------
// resize
// ---------------------------------------------------------------------------

void PointRenderer::resize(int width, int height)
{
    m_width  = width;
    m_height = height;

    glBindTexture(GL_TEXTURE_2D, m_colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F,
                 width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    if (m_pickTex) {
        glBindTexture(GL_TEXTURE_2D, m_pickTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                     width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    deleteGlowFBOs();
    createGlowFBOs(width, height);
    deleteDensityFBO();
    createDensityFBO(width, height);
}

// ---------------------------------------------------------------------------
// setGlow
// ---------------------------------------------------------------------------

void PointRenderer::setGlow(bool enabled, float blurRadius, float glowIntensity)
{
    m_glowEnabled   = enabled;
    m_blurRadius    = blurRadius;
    m_glowIntensity = glowIntensity;
}

// ---------------------------------------------------------------------------
// beginFrame / endFrame
// ---------------------------------------------------------------------------

void PointRenderer::beginFrame()
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glViewport(0, 0, m_width, m_height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void PointRenderer::endFrame()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    applyGlow();
}

// ---------------------------------------------------------------------------
// uploadParticles
// ---------------------------------------------------------------------------

void PointRenderer::uploadParticles(const std::vector<float>& data,
                                    size_t count, bool subData)
{
    m_particleCount = count;

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    if (subData) {
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        count * 6 * sizeof(float), data.data());
    } else {
        glBufferData(GL_ARRAY_BUFFER,
                     count * 6 * sizeof(float),
                     data.data(), GL_DYNAMIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Allocate VBO storage for `count` vertices (no CPU data) — the CUDA interop kernel
// fills it. Must be called before (re)registering the VBO with CUDA.
void PointRenderer::reserveVBO(size_t count)
{
    m_particleCount = count;
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, count * 6 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

void PointRenderer::render(const glm::mat4& viewProj,
                           float pointSize, float particleSizeNeg,
                           const glm::vec3 posPalette[32],
                           const glm::vec3 negPalette[32],
                           float maxMass,
                           bool showNegaMass,
                           float galaxyRadius)
{
    if (m_particleCount == 0) return;

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glUseProgram(m_program);

    glUniformMatrix4fv(m_uViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
    glUniform1f(m_uPointSize, pointSize);
    glUniform1f(m_uPointSizeNeg, particleSizeNeg);
    glUniform3fv(m_uPosPalette, 32, glm::value_ptr(posPalette[0]));
    glUniform3fv(m_uNegPalette, 32, glm::value_ptr(negPalette[0]));
    glUniform1f(m_uMaxMass, maxMass);
    glUniform1f(m_uShowNegaMass, showNegaMass ? 1.0f : 0.0f);
    glUniform1f(m_uGalaxyRadius, galaxyRadius);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(m_particleCount));
    glBindVertexArray(0);

    glUseProgram(0);

    glDisable(GL_BLEND);
    glDisable(GL_PROGRAM_POINT_SIZE);
}

// ---------------------------------------------------------------------------
// pickParticle — one-pass color picking
// ---------------------------------------------------------------------------

int PointRenderer::pickParticle(const glm::mat4& viewProj, float pointSize, float particleSizeNeg,
                                  float maxMass, bool showNegaMass,
                                  int screenX, int screenY,
                                  float galaxyRadius)
{
    if (m_particleCount == 0 || m_uPickMode < 0) return -1;

    // Save current GL state
    GLint prevFbo = 0; glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
    GLint prevViewport[4]; glGetIntegerv(GL_VIEWPORT, prevViewport);

    // 1. Render particles in picking mode to the PICKING FBO (not the main FBO!)
    // Use 2x larger points for easier clicking on distant/excentric particles
    glBindFramebuffer(GL_FRAMEBUFFER, m_pickFbo);
    glViewport(0, 0, m_width, m_height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_PROGRAM_POINT_SIZE);

    glUseProgram(m_program);

    glUniformMatrix4fv(m_uViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
    glUniform1f(m_uPointSize, pointSize * 2.5f);   // 2.5x larger hit area
    glUniform1f(m_uPointSizeNeg, particleSizeNeg * 2.5f);
    // palettes not needed in picking mode, but keep uniforms valid
    glUniform3fv(m_uPosPalette, 32, glm::value_ptr(glm::vec3(1,0,0)));
    glUniform3fv(m_uNegPalette, 32, glm::value_ptr(glm::vec3(0,0,1)));
    glUniform1f(m_uMaxMass, maxMass);
    glUniform1f(m_uShowNegaMass, showNegaMass ? 1.0f : 0.0f);
    glUniform1f(m_uGalaxyRadius, galaxyRadius);
    glUniform1i(m_uPickMode, 1);   // enable picking encoding

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(m_particleCount));
    glBindVertexArray(0);

    glUniform1i(m_uPickMode, 0);   // restore normal mode
    glUseProgram(0);
    glDisable(GL_PROGRAM_POINT_SIZE);

    // 2. Search in a 9x9 zone around cursor for the nearest non-black pixel
    int searchR = 4;
    int readX = screenX - searchR;
    int readY = (m_height - 1 - screenY) - searchR;
    if (readX < 0) readX = 0;
    if (readY < 0) readY = 0;
    int readW = searchR * 2 + 1;
    int readH = searchR * 2 + 1;
    if (readX + readW > m_width)  readW = m_width - readX;
    if (readY + readH > m_height) readH = m_height - readY;

    int zonePixels = readW * readH;
    std::vector<GLubyte> pixels(zonePixels * 4, 0);
    glReadPixels(readX, readY, readW, readH, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    int bestId = -1;
    int bestDistSq = 99999;
    for (int py = 0; py < readH; py++) {
        for (int px = 0; px < readW; px++) {
            int idx = (py * readW + px) * 4;
            unsigned int id = pixels[idx] | (pixels[idx+1] << 8) | (pixels[idx+2] << 16);
            if (id == 0 || id > (unsigned int)m_particleCount) continue;
            int dx = px - searchR;
            int dy = py - searchR;
            int distSq = dx*dx + dy*dy;
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestId = (int)(id - 1);
            }
        }
    }

    // Restore GL state
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
    glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);

    return bestId;
}

// ---------------------------------------------------------------------------
// applyGlow — two-pass Gaussian blur then additive composite
// ---------------------------------------------------------------------------

void PointRenderer::applyGlow()
{
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(m_dummyVao);

    if (m_glowEnabled && m_blurRadius > 0.0f)
    {
        // 2-pass H+V blur with step = blurRadius/4.
        // A single pass with step = blurRadius creates discrete point rings
        // visible at high intensity. Dividing the step by 4 and doing
        // 2 passes gives the same effective radius without the artifact.
        glUseProgram(m_blurProgram);
        glUniform1i(m_uBlurTex, 0);

        float stepH = (m_blurRadius * 0.25f) / static_cast<float>(m_width);
        float stepV = (m_blurRadius * 0.25f) / static_cast<float>(m_height);

        // -- Pass 1 H: colorTex → blurTex1 --
        glBindFramebuffer(GL_FRAMEBUFFER, m_blurFbo1);
        glViewport(0, 0, m_width, m_height);
        glClear(GL_COLOR_BUFFER_BIT);
        glUniform2f(m_uBlurDir, stepH, 0.0f);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_colorTex);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // -- Pass 1 V: blurTex1 → blurTex2 --
        glBindFramebuffer(GL_FRAMEBUFFER, m_blurFbo2);
        glClear(GL_COLOR_BUFFER_BIT);
        glUniform2f(m_uBlurDir, 0.0f, stepV);
        glBindTexture(GL_TEXTURE_2D, m_blurTex1);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // -- Pass 2 H: blurTex2 → blurTex1 --
        glBindFramebuffer(GL_FRAMEBUFFER, m_blurFbo1);
        glClear(GL_COLOR_BUFFER_BIT);
        glUniform2f(m_uBlurDir, stepH, 0.0f);
        glBindTexture(GL_TEXTURE_2D, m_blurTex2);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // -- Pass 2 V: blurTex1 → blurTex2 (final result) --
        glBindFramebuffer(GL_FRAMEBUFFER, m_blurFbo2);
        glClear(GL_COLOR_BUFFER_BIT);
        glUniform2f(m_uBlurDir, 0.0f, stepV);
        glBindTexture(GL_TEXTURE_2D, m_blurTex1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    else
    {
        // Pas de blur : blurTex2 reste noir (pas de glow)
        glBindFramebuffer(GL_FRAMEBUFFER, m_blurFbo2);
        glViewport(0, 0, m_width, m_height);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    // -- Composite: (bg + colorTex + blurTex2 * intensity) → glowTex --
    glBindFramebuffer(GL_FRAMEBUFFER, m_glowFbo);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(m_compositeProgram);
    glUniform1i(m_uCompScene,     0);
    glUniform1i(m_uCompGlow,      1);
    glUniform1f(m_uCompIntensity, m_glowEnabled ? m_glowIntensity : 0.0f);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_colorTex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_blurTex2);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    // -- Cleanup --
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glUseProgram(0);
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ---------------------------------------------------------------------------
// 3D Stereo Implementation
// ---------------------------------------------------------------------------

static void createOneFBO(int w, int h, GLuint& fbo, GLuint& tex)
{
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

static void deleteOneFBO(GLuint& fbo, GLuint& tex)
{
    if (tex) { glDeleteTextures(1, &tex); tex = 0; }
    if (fbo) { glDeleteFramebuffers(1, &fbo); fbo = 0; }
}

void PointRenderer::createStereoFBOs(int w, int h)
{
    createOneFBO(w, h, m_fboLeft, m_colorTexLeft);
    createOneFBO(w, h, m_fboRight, m_colorTexRight);

    // Blur chain left
    createOneFBO(w, h, m_blurFbo1L, m_blurTex1L);
    createOneFBO(w, h, m_blurFbo2L, m_blurTex2L);
    createOneFBO(w, h, m_glowFboL,  m_glowTexL);

    // Blur chain right
    createOneFBO(w, h, m_blurFbo1R, m_blurTex1R);
    createOneFBO(w, h, m_blurFbo2R, m_blurTex2R);
    createOneFBO(w, h, m_glowFboR,  m_glowTexR);
}

void PointRenderer::deleteStereoFBOs()
{
    deleteOneFBO(m_fboLeft, m_colorTexLeft);
    deleteOneFBO(m_fboRight, m_colorTexRight);
    deleteOneFBO(m_blurFbo1L, m_blurTex1L);
    deleteOneFBO(m_blurFbo2L, m_blurTex2L);
    deleteOneFBO(m_glowFboL,  m_glowTexL);
    deleteOneFBO(m_blurFbo1R, m_blurTex1R);
    deleteOneFBO(m_blurFbo2R, m_blurTex2R);
    deleteOneFBO(m_glowFboR,  m_glowTexR);
}

void PointRenderer::renderEye(const glm::mat4& viewProj, float pointSize, float particleSizeNeg,
                               const glm::vec3 posPalette[32], const glm::vec3 negPalette[32],
                               float maxMass, bool showNegaMass, float galaxyRadius,
                               GLuint fbo)
{
    if (m_particleCount == 0) return;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, m_width, m_height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glUseProgram(m_program);

    glUniformMatrix4fv(m_uViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
    glUniform1f(m_uPointSize, pointSize);
    glUniform1f(m_uPointSizeNeg, particleSizeNeg);
    glUniform3fv(m_uPosPalette, 32, glm::value_ptr(posPalette[0]));
    glUniform3fv(m_uNegPalette, 32, glm::value_ptr(negPalette[0]));
    glUniform1f(m_uMaxMass, maxMass);
    glUniform1f(m_uShowNegaMass, showNegaMass ? 1.0f : 0.0f);
    glUniform1f(m_uGalaxyRadius, galaxyRadius);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(m_particleCount));
    glBindVertexArray(0);

    glUseProgram(0);
    glDisable(GL_BLEND);
    glDisable(GL_PROGRAM_POINT_SIZE);
}

void PointRenderer::applyGlowStereo(GLuint srcTex, GLuint dstFbo1, GLuint dstTex1,
                                     GLuint dstFbo2, GLuint dstTex2)
{
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(m_dummyVao);

    if (m_glowEnabled && m_blurRadius > 0.0f)
    {
        // Pass H: srcTex → dstTex1
        glBindFramebuffer(GL_FRAMEBUFFER, dstFbo1);
        glViewport(0, 0, m_width, m_height);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(m_blurProgram);
        glUniform1i(m_uBlurTex, 0);
        glUniform2f(m_uBlurDir, m_blurRadius / (float)m_width, 0.0f);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, srcTex);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // Pass V: dstTex1 → dstTex2
        glBindFramebuffer(GL_FRAMEBUFFER, dstFbo2);
        glClear(GL_COLOR_BUFFER_BIT);

        glUniform2f(m_uBlurDir, 0.0f, m_blurRadius / (float)m_height);
        glBindTexture(GL_TEXTURE_2D, dstTex1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    else
    {
        glBindFramebuffer(GL_FRAMEBUFFER, dstFbo2);
        glViewport(0, 0, m_width, m_height);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PointRenderer::renderStereo(const glm::mat4& viewProj,
                                  const glm::vec3& camRight,
                                  float pointSize, float particleSizeNeg,
                                  const glm::vec3 posPalette[32], const glm::vec3 negPalette[32],
                                  float maxMass, bool showNegaMass, float galaxyRadius,
                                  int stereoMode, float ipd)
{
    // First time stereo: create FBOs
    if (m_fboLeft == 0) {
        createStereoFBOs(m_width, m_height);
        // Re-link composite to get stereo uniforms
        if (m_compositeProgram) {
            glDetachShader(m_compositeProgram, m_compositeFrag);
            glDeleteShader(m_compositeFrag);
            m_compositeFrag = glCreateShader(GL_FRAGMENT_SHADER);
            if (compileShader(m_compositeFrag, "render/shaders/composite.frag")) {
                glAttachShader(m_compositeProgram, m_compositeFrag);
                linkProgram(m_compositeProgram);
                m_uCompScene      = glGetUniformLocation(m_compositeProgram, "uScene");
                m_uCompGlow       = glGetUniformLocation(m_compositeProgram, "uGlow");
                m_uCompIntensity  = glGetUniformLocation(m_compositeProgram, "uGlowIntensity");
                m_uStereoMode     = glGetUniformLocation(m_compositeProgram, "uStereoMode");
                m_uCompSceneLeft  = glGetUniformLocation(m_compositeProgram, "uSceneLeft");
                m_uCompSceneRight = glGetUniformLocation(m_compositeProgram, "uSceneRight");
                m_uCompGlowLeft   = glGetUniformLocation(m_compositeProgram, "uGlowLeft");
                m_uCompGlowRight  = glGetUniformLocation(m_compositeProgram, "uGlowRight");
            }
        }
    }

    // ── STEREO CORRECTION : viewProj * T(world_offset) ────────────────────
    // viewProj = proj * view. We want to move the camera by ±offset.
    // Moving the camera = translating the objects the opposite way BEFORE the view.
    // view * T_world(offset) = R * T(-camPos) * T(offset) = R * T(offset - camPos)
    // → effective camera at camPos - offset. Correct!
    glm::vec3 eyeOffset = camRight * (ipd * 0.5f);
    glm::mat4 worldTranslateLeft  = glm::translate(glm::mat4(1.0f), -eyeOffset);
    glm::mat4 worldTranslateRight = glm::translate(glm::mat4(1.0f), +eyeOffset);
    glm::mat4 vpLeft  = viewProj * worldTranslateLeft;
    glm::mat4 vpRight = viewProj * worldTranslateRight;

    // Render left eye + blur
    renderEye(vpLeft, pointSize, particleSizeNeg, posPalette, negPalette,
              maxMass, showNegaMass, galaxyRadius, m_fboLeft);
    applyGlowStereo(m_colorTexLeft, m_blurFbo1L, m_blurTex1L,
                    m_blurFbo2L, m_blurTex2L);

    // Render right eye + blur
    renderEye(vpRight, pointSize, particleSizeNeg, posPalette, negPalette,
              maxMass, showNegaMass, galaxyRadius, m_fboRight);
    applyGlowStereo(m_colorTexRight, m_blurFbo1R, m_blurTex1R,
                    m_blurFbo2R, m_blurTex2R);

    // Final composite to m_fbo (same pipeline as mono — renderGUI will display it)
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glViewport(0, 0, m_width, m_height);
    glClearColor(0.031f, 0.063f, 0.188f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(m_compositeProgram);
    glUniform1i(m_uStereoMode, stereoMode);
    glUniform1f(m_uCompIntensity, m_glowEnabled ? m_glowIntensity : 0.0f);

    // ── ENVOYER les uniforms sampler2D (le bug !) ─────────────────────────
    glUniform1i(m_uCompSceneLeft,  0);  // uSceneLeft  → texture unit 0
    glUniform1i(m_uCompGlowLeft,   1);  // uGlowLeft   → texture unit 1
    glUniform1i(m_uCompSceneRight, 2);  // uSceneRight → texture unit 2
    glUniform1i(m_uCompGlowRight,  3);  // uGlowRight  → texture unit 3

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_colorTexLeft);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_blurTex2L);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_colorTexRight);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, m_blurTex2R);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0);

    glUseProgram(0);
    glBindVertexArray(0);
}
