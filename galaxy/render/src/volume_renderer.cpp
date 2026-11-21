// volume_renderer.cpp — velocity-colored splatting (no voxels)
#include "volume_renderer.h"
#include <glm/gtc/type_ptr.hpp>
#include <cstdio>

static GLuint compile(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; glGetShaderInfoLog(s, 2048, nullptr, log); fprintf(stderr, "[volume] shader: %s\n", log); }
    return s;
}
static GLuint link(GLuint vs, GLuint fs)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, 2048, nullptr, log); fprintf(stderr, "[volume] link: %s\n", log); }
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

// --- splat: pos (loc0) + speed (loc1) + size-sign (loc2) ---
static const char* SPLAT_VS = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in float aSpeed;
layout(location=2) in float aSign;
uniform mat4 uMVP; uniform float uSize, uVpH, uVelRef;
uniform int uShowNeg, uColorByVel, uPersp; uniform vec3 uPal[32];
out vec3 vCol;
void main(){
    if(uShowNeg==0 && aSign<0.0){ gl_Position=vec4(2.0,2.0,2.0,1.0); gl_PointSize=0.0; return; }
    vec4 cp = uMVP*vec4(aPos,1.0);
    gl_Position = cp;
    float ps = (uPersp==1) ? uSize*uVpH/max(cp.w,0.001) : uSize;
    gl_PointSize = clamp(ps, 1.0, 220.0);
    float idx = (uColorByVel==1) ? clamp(aSpeed/(uVelRef*1.8),0.0,1.0) : 0.85;
    float t=idx*31.0; int i=int(t); float f=t-float(i); int j=min(i+1,31);
    vCol = mix(uPal[i],uPal[j],f);
})";
static const char* SPLAT_FS = R"(#version 330 core
in vec3 vCol; out vec4 frag; uniform float uIntensity;
void main(){
    vec2 d = gl_PointCoord*2.0-1.0; float r2=dot(d,d);
    if(r2>1.0) discard;
    float g = exp(-r2*3.0);
    frag = vec4(vCol*g*uIntensity, g*uIntensity);
})";

// --- full-screen tone-map: preserves the hue + whitens the bright zones ---
static const char* TONE_VS = R"(#version 330 core
out vec2 vUV;
void main(){ vec2 p=vec2((gl_VertexID==2)?3.0:-1.0,(gl_VertexID==1)?3.0:-1.0); vUV=p*0.5+0.5; gl_Position=vec4(p,0.0,1.0); })";
static const char* TONE_FS = R"(#version 330 core
in vec2 vUV; out vec4 frag; uniform sampler2D uHDR; uniform float uExposure;
void main(){
    vec3 col = texture(uHDR, vUV).rgb;
    float lum = max(dot(col, vec3(0.299,0.587,0.114)), 1e-5);
    float tl = 1.0 - exp(-lum*uExposure);
    float wb = smoothstep(0.75, 1.10, tl);          // bright cores -> white
    vec3 c = clamp(mix(col*(tl/lum), vec3(1.0), wb)*tl, 0.0, 1.0);
    frag = vec4(c, clamp(tl,0.0,1.0));              // alpha=coverage -> background visible
})";

bool VolumeRenderer::init()
{
    m_splatProg = link(compile(GL_VERTEX_SHADER, SPLAT_VS), compile(GL_FRAGMENT_SHADER, SPLAT_FS));
    m_toneProg  = link(compile(GL_VERTEX_SHADER, TONE_VS),  compile(GL_FRAGMENT_SHADER, TONE_FS));
    glGenVertexArrays(1, &m_vao);
    glGenVertexArrays(1, &m_fsvao);
    glGenBuffers(1, &m_speedVbo);
    glEnable(GL_PROGRAM_POINT_SIZE);
    return m_splatProg && m_toneProg;
}
void VolumeRenderer::shutdown()
{
    if (m_hdrTex) glDeleteTextures(1, &m_hdrTex);
    if (m_hdrFbo) glDeleteFramebuffers(1, &m_hdrFbo);
    if (m_speedVbo) glDeleteBuffers(1, &m_speedVbo);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
    if (m_fsvao) glDeleteVertexArrays(1, &m_fsvao);
    if (m_splatProg) glDeleteProgram(m_splatProg);
    if (m_toneProg) glDeleteProgram(m_toneProg);
}

void VolumeRenderer::uploadSpeeds(const float* data, int n)
{
    if (!data || n <= 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, m_speedVbo);
    if (n != m_speedN) { glBufferData(GL_ARRAY_BUFFER, (size_t)n * sizeof(float), data, GL_DYNAMIC_DRAW); m_speedN = n; }
    else glBufferSubData(GL_ARRAY_BUFFER, 0, (size_t)n * sizeof(float), data);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void VolumeRenderer::ensureHDR(int w, int h)
{
    if (w == m_hw && h == m_hh && m_hdrFbo) return;
    m_hw = w; m_hh = h;
    if (!m_hdrFbo) glGenFramebuffers(1, &m_hdrFbo);
    if (!m_hdrTex) glGenTextures(1, &m_hdrTex);
    glBindTexture(GL_TEXTURE_2D, m_hdrTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindFramebuffer(GL_FRAMEBUFFER, m_hdrFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_hdrTex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void VolumeRenderer::drawSplats(unsigned posVbo, int count, const glm::mat4& mvp, float velRef,
                                const glm::vec3 pal[32], float size, float intensity,
                                bool perspective, bool colorByVel, bool showNeg)
{
    glUseProgram(m_splatProg);
    auto U = [&](const char* n){ return glGetUniformLocation(m_splatProg, n); };
    glUniformMatrix4fv(U("uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1f(U("uSize"), size);
    glUniform1f(U("uVpH"), (float)m_hh);
    glUniform1f(U("uVelRef"), velRef);
    glUniform1f(U("uIntensity"), intensity);
    glUniform3fv(U("uPal"), 32, glm::value_ptr(pal[0]));
    glUniform1i(U("uShowNeg"), showNeg ? 1 : 0);
    glUniform1i(U("uColorByVel"), colorByVel ? 1 : 0);
    glUniform1i(U("uPersp"), perspective ? 1 : 0);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, posVbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(5 * sizeof(float)));
    glBindBuffer(GL_ARRAY_BUFFER, m_speedVbo);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glDrawArrays(GL_POINTS, 0, count);
    glBindVertexArray(0);
}

void VolumeRenderer::renderHalo(unsigned posVbo, int count, int fbW, int fbH,
                                const glm::mat4& mvp, float velRef, const glm::vec3 pal[32],
                                float haloSize, float intensity, float exposure,
                                bool colorByVel, bool showNeg)
{
    if (!posVbo || count <= 0) return;
    GLint prevFbo = 0; glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
    ensureHDR(fbW, fbH);

    // 1) additive splats into the HDR
    glBindFramebuffer(GL_FRAMEBUFFER, m_hdrFbo);
    glViewport(0, 0, fbW, fbH);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_ONE);
    // haloSize is already in PIXELS (computed app-side via wuPerPx) -> perspective=false
    drawSplats(posVbo, count, mvp, velRef, pal, haloSize, intensity, false, colorByVel, showNeg);

    // 2) tone-map -> current FBO (composite, alpha = coverage)
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prevFbo);
    glViewport(0, 0, fbW, fbH);
    glDisable(GL_BLEND);
    glUseProgram(m_toneProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTex);
    glUniform1i(glGetUniformLocation(m_toneProg, "uHDR"), 0);
    glUniform1f(glGetUniformLocation(m_toneProg, "uExposure"), exposure);
    glBindVertexArray(m_fsvao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

void VolumeRenderer::renderStars(unsigned posVbo, int count, const glm::mat4& mvp,
                                 float velRef, const glm::vec3 pal[32],
                                 float pointSize, float brightness, bool colorByVel, bool showNeg)
{
    if (!posVbo || count <= 0) return;
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_ONE);
    // fixed size in pixels (uPersp=0), small sharp points
    drawSplats(posVbo, count, mvp, velRef, pal, pointSize, brightness, false, colorByVel, showNeg);
    glDisable(GL_BLEND);
}
