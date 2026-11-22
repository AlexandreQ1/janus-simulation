#ifdef _WIN32
# define NOMINMAX
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
# include <commdlg.h>
# include <shellapi.h>
# undef NEAR
# undef FAR
# undef near
# undef far
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "app.h"
#include "gl_interop.cuh"   // CUDA-OpenGL interop : GPU-built vertex buffer
#include "volume_grid.cuh"  // 3D grid CIC deposit for the volumetric rendering
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cstdlib>   // std::system
#include <filesystem>
#include <unordered_map>
#include <cstdint>
#include <functional>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_opengl3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ------------------------------------------------------------------
// Static GUI helpers
// ------------------------------------------------------------------

static void SliderFloatWithInput(const char* label, float* value,
                                  float v_min, float v_max,
                                  const char* format)
{
    float avail = ImGui::GetContentRegionAvail().x;
    float labelW = 90.0f;
    float sliderW = (avail - labelW) * 0.55f;
    float inputW  = (avail - labelW) * 0.38f;

    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::SameLine(labelW);

    std::string s_id = std::string("##s_") + label;
    std::string i_id = std::string("##i_") + label;

    ImGui::PushItemWidth(sliderW);
    ImGui::SliderFloat(s_id.c_str(), value, v_min, v_max, format);
    ImGui::PopItemWidth();

    ImGui::SameLine();
    ImGui::PushItemWidth(inputW);
    ImGui::InputFloat(i_id.c_str(), value, 0, 0, format);
    ImGui::PopItemWidth();
}

static void SliderIntWithInput(const char* label, int* value,
                                int v_min, int v_max)
{
    float avail = ImGui::GetContentRegionAvail().x;
    float labelW = 90.0f;
    float sliderW = (avail - labelW) * 0.55f;
    float inputW  = (avail - labelW) * 0.38f;

    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::SameLine(labelW);

    std::string s_id = std::string("##s_") + label;
    std::string i_id = std::string("##i_") + label;

    ImGui::PushItemWidth(sliderW);
    ImGui::SliderInt(s_id.c_str(), value, v_min, v_max, "%d");
    ImGui::PopItemWidth();

    ImGui::SameLine();
    ImGui::PushItemWidth(inputW);
    int temp = *value;
    if (ImGui::InputInt(i_id.c_str(), &temp, 0, 0))
    {
        if (temp < v_min) temp = v_min;
        if (temp > v_max) temp = v_max;
        *value = temp;
    }
    ImGui::PopItemWidth();
}

static void SectionTitle(const char* text)
{
    ImGui::NewLine();
    float window_width = ImGui::GetWindowSize().x;
    float text_width = ImGui::CalcTextSize(text).x;
    ImGui::SetCursorPosX((window_width - text_width) * 0.5f);
    ImGui::TextUnformatted(text);
    ImGui::Separator();
}

// ------------------------------------------------------------------
// App class
// ------------------------------------------------------------------

App::App(int w, int h)
    : m_width(w), m_height(h)
    , m_window(nullptr), m_glContext(nullptr), m_keystate(nullptr)
    , m_renderer(nullptr), m_camera(nullptr), m_simulation(nullptr)
    , m_radius(500.0f)
    , m_scaleUnit(0.0005f)
    , m_timeStep(1.0f)
    , m_numNegaStars(50000)
    , m_numPosiStars(50000)
    , m_negaDensity(-1.0e5f)
    , m_totalSolarMass(100000.0f)
    , m_blackHoleMass(35.0f)
    , m_rotSpeed(0.0f)
    , m_initSpeed(0.0f)
    , m_softening(7.0f)
    , m_gravConstant(0.0000000000667408f)
    , m_useTree(false)
    , m_treeTheta(0.5f)
    , m_simRunning(false)
    , m_spAmplitude(0.0f)
    , m_spPatternSpeed(0.01f)
    , m_spPitchAngle(15.0f)
    , m_spArms(2)
    , m_densityWavesEnabled(true)
    , m_glowEnabled(true)
    , m_blurRadius(4.0f)
    , m_glowIntensity(2.5f)
    , m_colorByVelocity(false)
    , m_velAutoScale(true)
    , m_velMaxForColor(1.0f)
    , m_particleSize(0.4f)
    , m_particleSizeNeg(0.4f)
    , m_isGenerating(false)
    , m_generateCountdown(0)
    , m_phaseShowNeg(false)
    , m_showConservation(false)
    , m_showEscape(false)
    , m_showCorrelation(false)
    , m_showSersic(false)
    , m_corrDone(false)
    , m_showHaloFinder(false)
    , m_fofLinkingScale(0.2f), m_fofMinMembers(20)
    , m_fofIncludeNeg(false), m_fofOverlay(true), m_fofDone(false)
    , m_corrRMax(0.0f)
    , m_consHead(0), m_consCount(0)
    , m_escHead(0),  m_escCount(0)
    , m_prevNegVel(0.0f), m_prevEscTime(0.0f)
    , m_showAbout(false)
    , m_simMyPerSec(0.0f)
    , m_mySecLastSimTime(0.0)
    , m_mySecLastRealMs(0)
    , m_showDispersion(false)
    , m_needReshare(false)
    , m_showGrid(true)
    , m_showVelocityField(false), m_velFieldScale(1.0f), m_velFieldRelative(true)
    , m_velSelectMode(false), m_velSelActive(false), m_velSelDragging(false)
    , m_velSelPending(false)
    , m_velSelX0(0), m_velSelY0(0), m_velSelX1(0), m_velSelY1(0)
    , m_annotTool(0), m_annotStage(0), m_annotX0(0), m_annotY0(0)
    , m_annotEditIdx(-1)
    , m_screenshotPending(false)
    , m_lyPerUnit(100.0f)
    , m_showRotCurve(false)
    , m_showDensityProfile(false)
    , m_showPhaseSpace(false)
    , m_showEnergyPlot(false)
    , m_rotAutoRefresh(false)
    , m_rotRefreshInterval(50)
    , m_lastRefreshFrame(-9999)
    , m_energyHead(0), m_energyCount(0)
    , m_showNegaMass(true)
    , m_edgeOnView(false)
    , m_stereoEnabled(false)
    , m_stereoMode(0)
    , m_ipd(65.0f)
    , m_rightMouseDown(false)
    , m_lastMouseX(0)
    , m_lastMouseY(0)
    , m_showBarycenters(false)
    , m_baryHead(0), m_baryCount(0)
    , m_baryPosX(0), m_baryPosY(0), m_baryPosZ(0)
    , m_baryNegX(0), m_baryNegY(0), m_baryNegZ(0)
    , m_baryPosVx(0), m_baryPosVy(0), m_baryPosVz(0)
    , m_baryNegVx(0), m_baryNegVy(0), m_baryNegVz(0)
    , m_showFourier(false)
    , m_fourierHead(0), m_fourierCount(0)
    , m_showDarkMatter(false)
    , m_nfwRs(250.0f), m_nfwRho0(1.0f)
    , m_simPaused(false)
    , m_rotCurveRMax(100000.0f)
    , m_showEquationEditor(false)
    , m_eqCompileOk(true)
    , m_eqCompileTime(0.0f)
    , m_eqNeedReload(false)
    , m_eqSelectedModel(1)  // Janus = default
    , m_cudaDevice(-1)      // auto-select
    , m_selectedMarker(-1)
    , m_markerPickPending(false)
    , m_markerPickX(0), m_markerPickY(0)
    , m_currentFrame(0)
    , m_totalFrames(30)
    , m_simSpeed(1.0f)
    , m_skipCounter(0)
    , m_particleCount(0)
{
    const char* userProfile = getenv("USERPROFILE");
    if (userProfile)
        snprintf(m_stateFilePath, sizeof(m_stateFilePath),
                 "%s\\Desktop\\jgravity_state.bin", userProfile);
    else
        snprintf(m_stateFilePath, sizeof(m_stateFilePath), "jgravity_state.bin");

    m_presetName[0] = '\0';

    m_annotColor[0] = 1.0f; m_annotColor[1] = 0.85f; m_annotColor[2] = 0.2f;  // amber default
    m_annotTextBuf[0] = '\0';
    m_screenshotPath[0] = '\0';
}

// ---------------------------------------------------------------------------
// Conservation window (angular momentum + total energy)
// ---------------------------------------------------------------------------

void App::renderConservation()
{
    if (!m_showConservation) return;
    ImGui::SetNextWindowSize(ImVec2(620, 460), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(360, 30),   ImGuiCond_FirstUseEver);
    ImGui::Begin("Conservation", &m_showConservation,
        ImGuiWindowFlags_NoCollapse);

    if (ImGui::Button("Refresh") && m_simulation && !m_positions.empty()) refreshAnalytics();
    ImGui::SameLine();
    ImGui::TextDisabled("Lz/M (specific angular momentum)  |  KE + PE_est");

    if (m_consCount < 2) {
        ImGui::TextDisabled("Enable Auto-Refresh to record data");
        ImGui::End(); return;
    }

    ImVec2 cvs = ImGui::GetContentRegionAvail();
    float hH = cvs.y * 0.5f - 4;
    float pW = cvs.x;

    auto plotHistory = [&](float* hist, double* times, int cnt, int head,
                           int histSize, const char* label, ImU32 col)
    {
        ImVec2 cp = ImGui::GetCursorScreenPos();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float mL=60,mR=10,mT=8,mB=20;
        float pw = pW-mL-mR, ph = hH-mT-mB;
        float px = cp.x+mL, py = cp.y+mT;
        dl->AddRectFilled(cp, ImVec2(cp.x+pW,cp.y+hH), IM_COL32(8,16,48,255));

        // Linear copy, then Gaussian smoothing
        std::vector<float> vis(cnt);
        for (int i=0;i<cnt;i++) {
            int idx=(head-cnt+i+histSize)%histSize;
            vis[i] = hist[idx];
        }
        int sigB  = std::max(3, std::min(30, cnt / 20));
        int halfK = sigB * 3;
        std::vector<float> sm(cnt);
        for (int i=0;i<cnt;i++) {
            double sumW=0.0, sumWV=0.0;
            for (int k=-halfK;k<=halfK;k++) {
                int j=i+k; if (j<0||j>=cnt) continue;
                double w = exp(-0.5*(double)(k*k)/(double)(sigB*sigB));
                sumW += w; sumWV += w * vis[j];
            }
            sm[i] = (sumW>0.0) ? (float)(sumWV/sumW) : vis[i];
        }
        float vMin=1e30f, vMax=-1e30f;
        for (float v : sm) { if (v<vMin) vMin=v; if (v>vMax) vMax=v; }
        if (vMax<=vMin) vMax=vMin+1.0f;

        // Clip lines to plot area
        dl->PushClipRect(ImVec2(px, py), ImVec2(px+pw, py+ph), true);
        ImVec2 prev={0,0}; bool first=true;
        for (int i=0;i<cnt;i++) {
            float sx = px + (float)i/(cnt>1?cnt-1:1)*pw;
            float sy = py + (1.0f-(sm[i]-vMin)/(vMax-vMin))*ph;
            ImVec2 cur(sx,sy);
            if(!first) dl->AddLine(prev,cur,col,1.8f);
            prev=cur; first=false;
        }
        dl->PopClipRect();
        char buf[32];
        snprintf(buf,sizeof(buf),"%.3g",vMax);
        dl->AddText(ImVec2(px-54,py),IM_COL32(160,160,160,255),buf);
        snprintf(buf,sizeof(buf),"%.3g",vMin);
        dl->AddText(ImVec2(px-54,py+ph-12),IM_COL32(160,160,160,255),buf);
        dl->AddText(ImVec2(px+4,py+2),col,label);
        ImGui::Dummy(ImVec2(pW,hH));
    };

    plotHistory(m_angMomHistory, m_consTimesH, m_consCount, m_consHead,
                CONS_HISTORY, "Lz / M (sim units)", IM_COL32(100,220,255,220));
    ImGui::Dummy(ImVec2(0,4));
    plotHistory(m_totEHistory,   m_consTimesH, m_consCount, m_consHead,
                CONS_HISTORY, "Total E = KE + PE_est", IM_COL32(255,180,80,220));
    ImGui::End();
}

// ---------------------------------------------------------------------------
// Escape velocity window (negative mass average speed)
// ---------------------------------------------------------------------------

void App::renderEscapeVelocity()
{
    if (!m_showEscape) return;
    ImGui::SetNextWindowSize(ImVec2(620, 360), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(360, 30),   ImGuiCond_FirstUseEver);
    ImGui::Begin("Negative Mass Escape", &m_showEscape,
        ImGuiWindowFlags_NoCollapse);

    if (ImGui::Button("Refresh") && m_simulation && !m_positions.empty()) refreshAnalytics();
    ImGui::SameLine();
    ImGui::TextDisabled("<|v_neg|> — average speed of negative mass particles");

    if (m_escCount >= 1) {
        int last = (m_escHead - 1 + ESC_HISTORY) % ESC_HISTORY;
        ImGui::Text("<|v_neg|> = %.4f sim-vel    d/dt = %.4f sim-vel/MY",
            m_negVelHistory[last], m_negAccHistory[last]);
    }

    if (m_escCount < 1) { ImGui::TextDisabled("Enable Auto-Refresh to record data"); ImGui::End(); return; }

    ImVec2 cvs = ImGui::GetContentRegionAvail();
    ImVec2 cp  = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float mL=60,mR=10,mT=8,mB=20;
    float pw=cvs.x-mL-mR, ph=cvs.y-mT-mB;
    float px=cp.x+mL, py=cp.y+mT;
    dl->AddRectFilled(cp, ImVec2(cp.x+cvs.x,cp.y+cvs.y), IM_COL32(8,16,48,255));

    // Linear copy + Gaussian smoothing
    std::vector<float> visV(m_escCount);
    for (int i=0;i<m_escCount;i++) {
        int idx=(m_escHead-m_escCount+i+ESC_HISTORY)%ESC_HISTORY;
        visV[i] = m_negVelHistory[idx];
    }
    int sigB = std::max(3, std::min(30, m_escCount / 20));
    int halfK = sigB * 3;
    std::vector<float> sm(m_escCount);
    for (int i=0;i<m_escCount;i++) {
        double sumW=0.0, sumWV=0.0;
        for (int k=-halfK;k<=halfK;k++) {
            int j=i+k; if (j<0||j>=m_escCount) continue;
            double w = exp(-0.5*(double)(k*k)/(double)(sigB*sigB));
            sumW += w; sumWV += w * visV[j];
        }
        sm[i] = (sumW>0.0) ? (float)(sumWV/sumW) : visV[i];
    }
    float vMin=1e30f, vMax=-1e30f;
    for (float v : sm) { if (v<vMin) vMin=v; if (v>vMax) vMax=v; }
    if (vMax<=vMin) vMax=vMin+0.001f;

    dl->PushClipRect(ImVec2(px, py), ImVec2(px+pw, py+ph), true);
    ImVec2 prev={0,0}; bool first=true;
    for (int i=0; i<m_escCount; i++) {
        float sx=px+(float)i/(m_escCount>1?m_escCount-1:1)*pw;
        float sy=py+(1.0f-(sm[i]-vMin)/(vMax-vMin))*ph;
        ImVec2 cur(sx,sy);
        if (!first) dl->AddLine(prev,cur,IM_COL32(255,140,40,220),2.0f);
        prev=cur; first=false;
    }
    dl->PopClipRect();
    char buf[32];
    snprintf(buf,sizeof(buf),"%.4g",vMax);
    dl->AddText(ImVec2(px-56,py),IM_COL32(160,160,160,255),buf);
    snprintf(buf,sizeof(buf),"%.4g",vMin);
    dl->AddText(ImVec2(px-56,py+ph-12),IM_COL32(160,160,160,255),buf);
    dl->AddText(ImVec2(px+4,py+4),IM_COL32(255,160,60,220),"<|v_neg|>");
    ImGui::Dummy(cvs);
    ImGui::End();
}

// ---------------------------------------------------------------------------
// Two-point correlation ξ(r) — positive vs negative mass
// ---------------------------------------------------------------------------

void App::renderCorrelation()
{
    if (!m_showCorrelation) return;
    ImGui::SetNextWindowSize(ImVec2(620, 460), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(360, 30),   ImGuiCond_FirstUseEver);
    ImGui::Begin("Two-Point Correlation xi(r)", &m_showCorrelation,
        ImGuiWindowFlags_NoCollapse);

    const int RBINS = 50;

    ImGui::TextDisabled("xi(r) = DD/RR - 1  (pos-neg cross-correlation, sample 800 each)");
    ImGui::SameLine();
    if (ImGui::Button("Refresh##corr") && m_simulation && !m_positions.empty())
        refreshAnalytics();

    if (!m_corrDone) {
        ImGui::TextDisabled("Enable Auto-Refresh or click Refresh");
        ImGui::End(); return;
    }

    // Show first/mid values for verification
    if (!m_corrXi.empty()) {
        int mid = (int)m_corrXi.size() / 2;
        ImGui::Text("xi[0]=%.3f  xi[%d]=%.3f  xi[%d]=%.3f",
            m_corrXi[0], mid, m_corrXi[mid],
            (int)m_corrXi.size()-1, m_corrXi.back());
        ImGui::TextDisabled("xi > 0: clustered  |  xi < 0: anti-correlated (Janus repulsion)  |  xi = 0: random");
    }

    ImGui::Separator();

    // Gaussian smoothing of xi(r)
    int N = (int)m_corrXi.size();
    std::vector<float> xiSm(N);
    if (N > 0) {
        int sigB = std::max(2, std::min(8, N / 10));
        int halfK = sigB * 3;
        for (int i=0;i<N;i++) {
            double sumW=0.0, sumWV=0.0;
            for (int k=-halfK;k<=halfK;k++) {
                int j=i+k; if (j<0||j>=N) continue;
                double w = exp(-0.5*(double)(k*k)/(double)(sigB*sigB));
                sumW += w; sumWV += w * m_corrXi[j];
            }
            xiSm[i] = (sumW>0.0) ? (float)(sumWV/sumW) : m_corrXi[i];
        }
    }

    ImVec2 plotSize = ImGui::GetContentRegionAvail();
    plotSize.y -= 24;

    float xiMin = -1.0f, xiMax = 2.0f;
    for (float v : xiSm) {
        if (v > xiMax) xiMax = v * 1.1f;
        if (v < xiMin) xiMin = v;
    }

    char overlay[64];
    snprintf(overlay, sizeof(overlay), "xi(r)  [%.0f LY range]", m_corrRMax);
    ImGui::PlotLines("##xi_plot",
        xiSm.data(), (int)xiSm.size(), 0,
        overlay, xiMin, xiMax, plotSize);

    // Axis labels
    ImGui::Text("0 LY");
    ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.5f - 20);
    ImGui::Text("%.0f LY", m_corrRMax * 0.5f);
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 60);
    ImGui::Text("%.0f LY", m_corrRMax);

    ImGui::End();
}

// ---------------------------------------------------------------------------
// Sersic profile fitting
// ---------------------------------------------------------------------------

void App::renderSersic()
{
    if (!m_showSersic) return;
    ImGui::SetNextWindowSize(ImVec2(620, 460), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(360, 30),   ImGuiCond_FirstUseEver);
    ImGui::Begin("Sersic Profile", &m_showSersic,
        ImGuiWindowFlags_NoCollapse);

    ImGui::TextDisabled("I(R) = I0 * exp(-b_n * ((R/Re)^(1/n) - 1))   [n=1: flat disk  |  n=4: concentrated spherical bulge]");

    if (m_positions.empty()) { ImGui::TextDisabled("No data"); ImGui::End(); return; }

    const int BINS = 60;
    float rMax = m_radius * m_lyPerUnit * 3.0f;
    float bins[BINS] = {};
    int   cnt [BINS] = {};

    for (size_t i = 0; i < m_positions.size(); i++) {
        if (m_positions[i].w <= 0) continue;
        // Recenter on positive COM (galaxy may have drifted from origin)
        float dx = m_positions[i].x - m_baryPosX;
        float dy = m_positions[i].y - m_baryPosY;
        float r = sqrtf(dx*dx + dy*dy) * m_lyPerUnit;
        int b=(int)(r/rMax*BINS); if(b<0) b=0; if(b>=BINS) b=BINS-1;
        bins[b] += 1.0f; cnt[b]++;
    }

    // Surface brightness: divide by ring area
    float I[BINS]={};
    float iMax=0;
    for (int b=0;b<BINS;b++) {
        float r0=(float)b/BINS*rMax, r1=(float)(b+1)/BINS*rMax;
        float area=3.14159f*(r1*r1-r0*r0);
        I[b]= (area>0 && cnt[b]>0) ? bins[b]/area : 0;
        if(I[b]>iMax) iMax=I[b];
    }
    // Find half-light radius Re
    float cumSum=0, total=0;
    for(int b=0;b<BINS;b++) total+=I[b];
    float Re=rMax*0.3f;
    for(int b=0;b<BINS;b++){cumSum+=I[b]; if(cumSum>=total*0.5f){Re=(b+0.5f)/BINS*rMax;break;}}

    // Fit Sersic index n by minimising residuals (n=0.5 to 8)
    float bestN=1.0f, bestRes=1e30f;
    for(float nt=0.5f;nt<=8.0f;nt+=0.25f){
        float bn=2.0f*nt-0.327f; // Ciotti approximation
        float res=0;
        for(int b=0;b<BINS;b++){
            if(I[b]<1e-6f) continue;
            float R=(b+0.5f)/BINS*rMax;
            float Ifit = iMax * expf(-bn*(powf(R/Re,1.0f/nt)-1.0f));
            float d=logf(I[b]+0.1f)-logf(Ifit+0.1f);
            res+=d*d;
        }
        if(res<bestRes){bestRes=res;bestN=nt;}
    }
    float bn=2.0f*bestN-0.327f;
    ImGui::Text("Fitted Sersic index n = %.2f   |   Effective radius Re = %.0f LY", bestN, Re);
    ImGui::TextDisabled("n ~ 1: flat exponential disk   |   n ~ 4: concentrated spherical bulge (elliptical galaxy)");

    // Plot
    ImVec2 cvs=ImGui::GetContentRegionAvail();
    ImVec2 cp=ImGui::GetCursorScreenPos();
    ImDrawList* dl=ImGui::GetWindowDrawList();
    const float mL=60,mR=10,mT=8,mB=24;
    float pw=cvs.x-mL-mR, ph=cvs.y-mT-mB;
    float px=cp.x+mL, py=cp.y+mT;
    dl->AddRectFilled(cp, ImVec2(cp.x+cvs.x,cp.y+cvs.y), IM_COL32(8,16,48,255));

    // Data
    ImVec2 prev={0,0}; bool first=true;
    for(int b=0;b<BINS;b++){
        if(I[b]<1e-6f){first=true;continue;}
        float sx=px+(b+0.5f)/BINS*pw;
        float sy=py+(1.0f-I[b]/iMax)*ph;
        ImVec2 cur(sx,sy);
        if(!first) dl->AddLine(prev,cur,IM_COL32(100,180,255,220),1.5f);
        prev=cur; first=false;
    }
    // Fitted Sersic
    prev={0,0}; first=true;
    for(int b=0;b<BINS;b++){
        float R=(b+0.5f)/BINS*rMax;
        float Ifit=iMax*expf(-bn*(powf(R/Re,1.0f/bestN)-1.0f));
        float sx=px+(b+0.5f)/BINS*pw;
        float sy=py+(1.0f-fminf(Ifit/iMax,1.0f))*ph;
        ImVec2 cur(sx,sy);
        if(!first) dl->AddLine(prev,cur,IM_COL32(255,180,50,180),1.5f);
        prev=cur; first=false;
    }
    float lx=px+4, ly=py+4;
    dl->AddLine(ImVec2(lx,ly+8),ImVec2(lx+20,ly+8),IM_COL32(100,180,255,220),2);
    dl->AddText(ImVec2(lx+22,ly),IM_COL32(140,180,255,220),"Data");
    dl->AddLine(ImVec2(lx,ly+22),ImVec2(lx+20,ly+22),IM_COL32(255,180,50,180),2);
    dl->AddText(ImVec2(lx+22,ly+14),IM_COL32(255,200,80,220),"Sersic fit");
    char buf[32];
    snprintf(buf,sizeof(buf),"%.0f LY",rMax);
    {ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px+pw-ts.x,py+ph+2),IM_COL32(160,160,160,200),buf);}
    dl->AddText(ImVec2(px,py+ph+2),IM_COL32(160,160,160,200),"0");
    ImGui::Dummy(cvs);
    ImGui::End();
}

// ---------------------------------------------------------------------------
//  Friends-of-Friends halo finder (#3) + per-cluster virial ratio (#4)
//  Grid-accelerated union-find. Linking length b = scale × mean inter-particle
//  separation (Davis et al. 1985, standard b=0.2). Runs on demand (CPU).
// ---------------------------------------------------------------------------
// Runs on a worker thread. Reads the snapshot buffers (m_fofPosCopy/Vel) and
// snapshot params — NEVER the live m_positions which the sim mutates. Writes
// m_fofResult; the main thread swaps it into m_fofClusters when ready.
void App::computeFoF()
{
    const std::vector<cl_float4>& POS = m_fofPosCopy;
    const std::vector<cl_float4>& VEL = m_fofVelCopy;
    m_fofResult.clear();
    if (POS.empty()) return;

    // Gather indices of the selected species
    std::vector<int> idx;
    idx.reserve(POS.size());
    for (int i = 0; i < (int)POS.size(); i++) {
        float w = POS[i].w;
        if (m_fofNegSnap) { if (w < 0.0f) idx.push_back(i); }
        else              { if (w > 0.0f) idx.push_back(i); }
    }
    int N = (int)idx.size();
    if (N < 2) return;

    // Bounding box → mean inter-particle separation → linking length b
    float mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
    for (int k = 0; k < N; k++) {
        const cl_float4& p = POS[idx[k]];
        mn[0]=fminf(mn[0],p.x); mx[0]=fmaxf(mx[0],p.x);
        mn[1]=fminf(mn[1],p.y); mx[1]=fmaxf(mx[1],p.y);
        mn[2]=fminf(mn[2],p.z); mx[2]=fmaxf(mx[2],p.z);
    }
    float Lx = fmaxf(mx[0]-mn[0], 1e-3f);
    float Ly = fmaxf(mx[1]-mn[1], 1e-3f);
    float Lz = fmaxf(mx[2]-mn[2], 1e-3f);
    float vol = Lx * Ly * Lz;
    float meanSep = cbrtf(vol / (float)N);
    float b = m_fofLinkSnap * meanSep;
    if (b < 1e-4f) b = 1e-4f;
    float b2 = b * b;

    // Spatial hash grid, cell size = b. Only occupied cells stored.
    float invB = 1.0f / b;
    auto cellKey = [&](float x, float y, float z) -> uint64_t {
        // Offset to non-negative, 21 bits per axis
        uint64_t cx = (uint64_t)(uint32_t)(int32_t)floorf((x - mn[0]) * invB);
        uint64_t cy = (uint64_t)(uint32_t)(int32_t)floorf((y - mn[1]) * invB);
        uint64_t cz = (uint64_t)(uint32_t)(int32_t)floorf((z - mn[2]) * invB);
        return (cx & 0x1FFFFF) | ((cy & 0x1FFFFF) << 21) | ((cz & 0x1FFFFF) << 42);
    };
    std::unordered_map<uint64_t, std::vector<int>> grid;
    grid.reserve(N * 2);
    for (int k = 0; k < N; k++) {
        const cl_float4& p = POS[idx[k]];
        grid[cellKey(p.x, p.y, p.z)].push_back(k);
    }

    // Union-Find (local indices 0..N-1)
    std::vector<int> parent(N), rank_(N, 0);
    for (int k = 0; k < N; k++) parent[k] = k;
    std::function<int(int)> find = [&](int a) {
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        return a;
    };
    auto unite = [&](int a, int b_) {
        a = find(a); b_ = find(b_);
        if (a == b_) return;
        if (rank_[a] < rank_[b_]) std::swap(a, b_);
        parent[b_] = a;
        if (rank_[a] == rank_[b_]) rank_[a]++;
    };

    // Link particles within b: for each, scan 27 neighbor cells
    for (int k = 0; k < N; k++) {
        if ((k & 0xFFFF) == 0 && m_fofCancel.load()) { m_fofResult.clear(); return; }
        const cl_float4& p = POS[idx[k]];
        int32_t bx = (int32_t)floorf((p.x - mn[0]) * invB);
        int32_t by = (int32_t)floorf((p.y - mn[1]) * invB);
        int32_t bz = (int32_t)floorf((p.z - mn[2]) * invB);
        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            uint64_t cx = (uint64_t)(uint32_t)(bx + dx);
            uint64_t cy = (uint64_t)(uint32_t)(by + dy);
            uint64_t cz = (uint64_t)(uint32_t)(bz + dz);
            uint64_t key = (cx & 0x1FFFFF) | ((cy & 0x1FFFFF) << 21) | ((cz & 0x1FFFFF) << 42);
            auto it = grid.find(key);
            if (it == grid.end()) continue;
            for (int m : it->second) {
                if (m <= k) continue;  // each pair once
                const cl_float4& q = POS[idx[m]];
                float ddx=q.x-p.x, ddy=q.y-p.y, ddz=q.z-p.z;
                if (ddx*ddx + ddy*ddy + ddz*ddz <= b2) unite(k, m);
            }
        }
    }

    // Collect group members
    std::unordered_map<int, std::vector<int>> groups;
    for (int k = 0; k < N; k++) groups[find(k)].push_back(k);

    float G_eff = m_fofGsnap * 1.0e10f;
    bool haveVel = (VEL.size() >= POS.size());

    for (auto& g : groups) {
        if (m_fofCancel.load()) { m_fofResult.clear(); return; }
        std::vector<int>& mem = g.second;
        if ((int)mem.size() < m_fofMinSnap) continue;

        FoFCluster c{};
        c.count = (int)mem.size();
        double cx=0, cy=0, cz=0, cm=0, cvx=0, cvy=0, cvz=0;
        for (int li : mem) {
            const cl_float4& p = POS[idx[li]];
            float am = fabsf(p.w);
            cx += p.x*am; cy += p.y*am; cz += p.z*am; cm += am;
            if (haveVel) {
                const cl_float4& v = VEL[idx[li]];
                cvx += v.x*am; cvy += v.y*am; cvz += v.z*am;
            }
        }
        if (cm <= 0) continue;
        c.mass = (float)cm;
        c.cx = (float)(cx/cm); c.cy = (float)(cy/cm); c.cz = (float)(cz/cm);
        c.vx = (float)(cvx/cm); c.vy = (float)(cvy/cm); c.vz = (float)(cvz/cm);

        // rms radius
        double r2sum = 0;
        for (int li : mem) {
            const cl_float4& p = POS[idx[li]];
            float ddx=p.x-c.cx, ddy=p.y-c.cy, ddz=p.z-c.cz;
            r2sum += ddx*ddx + ddy*ddy + ddz*ddz;
        }
        c.radius = sqrtf((float)(r2sum / mem.size()));

        // Kinetic energy relative to cluster bulk velocity
        double KE = 0;
        if (haveVel) {
            for (int li : mem) {
                const cl_float4& p = POS[idx[li]];
                const cl_float4& v = VEL[idx[li]];
                float dvx=v.x-c.vx, dvy=v.y-c.vy, dvz=v.z-c.vz;
                KE += 0.5 * fabsf(p.w) * (dvx*dvx + dvy*dvy + dvz*dvz);
            }
        }

        // Potential energy: exact for small clusters, sampled for large ones
        double PE = 0;
        int M = (int)mem.size();
        float soft2 = m_fofSoftSnap * m_fofSoftSnap;
        const int PE_CAP = 1500;
        if (M <= PE_CAP) {
            for (int a = 0; a < M; a++) {
                const cl_float4& pa = POS[idx[mem[a]]];
                for (int bb = a+1; bb < M; bb++) {
                    const cl_float4& pb = POS[idx[mem[bb]]];
                    float ddx=pa.x-pb.x, ddy=pa.y-pb.y, ddz=pa.z-pb.z;
                    float r = sqrtf(ddx*ddx+ddy*ddy+ddz*ddz + soft2);
                    PE -= G_eff * fabsf(pa.w) * fabsf(pb.w) / r;
                }
            }
        } else {
            // Sample PE_CAP pairs uniformly and scale to full M(M-1)/2
            uint64_t seed = 0x2545F4914F6CDD1DULL;
            auto rnd = [&]() { seed ^= seed<<13; seed ^= seed>>7; seed ^= seed<<17; return (int)((seed>>33) % M); };
            const int SAMPLES = 200000;
            double sampleSum = 0; int got = 0;
            for (int s = 0; s < SAMPLES; s++) {
                int a = rnd(), bb = rnd(); if (a == bb) continue;
                const cl_float4& pa = POS[idx[mem[a]]];
                const cl_float4& pb = POS[idx[mem[bb]]];
                float ddx=pa.x-pb.x, ddy=pa.y-pb.y, ddz=pa.z-pb.z;
                float r = sqrtf(ddx*ddx+ddy*ddy+ddz*ddz + soft2);
                sampleSum -= G_eff * fabsf(pa.w) * fabsf(pb.w) / r;
                got++;
            }
            double totalPairs = (double)M * (M-1) * 0.5;
            PE = (got > 0) ? sampleSum * (totalPairs / got) : 0.0;
        }

        c.virial = (fabs(PE) > 1e-30) ? (float)(KE / fabs(PE)) : 0.0f;
        // Store global particle indices for live tracking + member highlighting
        c.members.reserve(mem.size());
        for (int li : mem) c.members.push_back(idx[li]);
        m_fofResult.push_back(c);
    }

    // Sort by mass descending
    std::sort(m_fofResult.begin(), m_fofResult.end(),
              [](const FoFCluster& a, const FoFCluster& b){ return a.mass > b.mass; });
}

void App::renderHaloFinder()
{
    if (!m_showHaloFinder) return;
    ImGui::SetNextWindowSize(ImVec2(560, 460), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(360, 30), ImGuiCond_FirstUseEver);
    ImGui::Begin("Halo Finder (FoF)", &m_showHaloFinder, ImGuiWindowFlags_NoCollapse);

    // Viewport overlay: circle each detected cluster, colored by virial state
    if (m_fofOverlay && m_fofDone && !m_fofClusters.empty() && m_camera) {
        ImGuiViewport* gMV = ImGui::GetMainViewport();
        const float gMVX = gMV->Pos.x, gMVY = gMV->Pos.y;
        const float panelW = 340.0f;
        const float vpX = gMVX + panelW;
        const float vpW = (float)m_width - panelW, vpH = (float)m_height;
        if (vpW > 0 && vpH > 0) {
            float vp[16]; m_camera->getViewProjMatrix(vp);
            ImDrawList* dl = ImGui::GetBackgroundDrawList(gMV);
            dl->PushClipRect(ImVec2(vpX, gMVY), ImVec2(vpX+vpW, gMVY+vpH), false);
            bool liveOK = (m_positions.size() > 0);
            const float SR = 9.0f;   // small FIXED circle (px) — the variable size added nothing
            for (size_t i = 0; i < m_fofClusters.size(); i++) {
                const FoFCluster& c = m_fofClusters[i];
                if (c.virial > 1.0f) continue;   // hide the UNBOUND groups (KE/|PE| > 1)
                // Live centroid (follows the structure); circle radius = FIXED.
                float gx = c.cx, gy = c.cy, gz = c.cz;
                if (liveOK && !c.members.empty()) {
                    double sx=0, sy=0, sz=0, sm=0;
                    for (int gi : c.members) {
                        if (gi < 0 || gi >= (int)m_positions.size()) continue;
                        const cl_float4& p = m_positions[gi];
                        float am = fabsf(p.w);
                        sx += p.x*am; sy += p.y*am; sz += p.z*am; sm += am;
                    }
                    if (sm > 0) { gx = (float)(sx/sm); gy = (float)(sy/sm); gz = (float)(sz/sm); }
                }
                float ccx = vp[0]*gx + vp[4]*gy + vp[8]*gz + vp[12];
                float ccy = vp[1]*gx + vp[5]*gy + vp[9]*gz + vp[13];
                float ccw = vp[3]*gx + vp[7]*gy + vp[11]*gz + vp[15];
                if (ccw < 1e-3f) continue;
                ImVec2 sc(vpX + (ccx/ccw + 1.0f)*0.5f*vpW, gMVY + (1.0f - ccy/ccw)*0.5f*vpH);
                float v = c.virial;
                ImU32 col = (v < 0.4f) ? IM_COL32(120,180,255,220)    // collapsing = bleu
                                       : IM_COL32(120,255,120,220);   // virialized = green
                dl->AddCircle(sc, SR, col, 0, 1.8f);
                char tag[16]; snprintf(tag, sizeof(tag), "%zu", i+1);
                dl->AddText(ImVec2(sc.x + SR + 2, sc.y - 6), col, tag);
            }

            // Highlight the selected cluster's member particles
            if (m_fofSelected >= 0 && m_fofSelected < (int)m_fofClusters.size()) {
                const FoFCluster& sc2 = m_fofClusters[m_fofSelected];
                for (int gi : sc2.members) {
                    if (gi < 0 || gi >= (int)m_positions.size()) continue;
                    const cl_float4& p = m_positions[gi];
                    float px = vp[0]*p.x + vp[4]*p.y + vp[8]*p.z + vp[12];
                    float py = vp[1]*p.x + vp[5]*p.y + vp[9]*p.z + vp[13];
                    float pw = vp[3]*p.x + vp[7]*p.y + vp[11]*p.z + vp[15];
                    if (pw < 1e-3f) continue;
                    ImVec2 s(vpX + (px/pw + 1.0f)*0.5f*vpW, gMVY + (1.0f - py/pw)*0.5f*vpH);
                    dl->AddRectFilled(ImVec2(s.x-1.0f, s.y-1.0f), ImVec2(s.x+1.0f, s.y+1.0f),
                                      IM_COL32(255, 255, 0, 230));
                }
            }
            dl->PopClipRect();
        }
    }

    ImGui::TextDisabled("Friends-of-Friends (Davis 1985): b = scale x mean separation");
    SliderFloatWithInput("Linking b/sep", &m_fofLinkingScale, 0.05f, 0.5f, "%.2f");
    ImGui::SetNextItemWidth(160);
    ImGui::SliderInt("Min members", &m_fofMinMembers, 5, 20000, "%d", ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("Negative species", &m_fofIncludeNeg);
    ImGui::SameLine();
    ImGui::Checkbox("Overlay on viewport", &m_fofOverlay);

    // Collect the worker's result once it signals completion
    if (m_fofResultReady.load()) {
        if (m_fofThread.joinable()) m_fofThread.join();
        // Always adopt the worker's output. computeFoF() clears m_fofResult when it
        // aborts on cancel, so this is empty for an aborted run and full for a
        // completed one. (The old `if (!m_fofCancel)` guard could discard a result
        // that finished just as the cancel flag flipped, forcing a confusing 2nd Run.)
        m_fofClusters = std::move(m_fofResult);
        m_fofDone = true;
        m_fofResult.clear();
        m_fofResultReady = false;
        m_fofCancel = false;
        // Keep the selection across live re-detections (clusters are mass-sorted,
        // so the same row tracks roughly the same structure). Only drop it if the
        // catalog shrank past it. Resetting to -1 here made the white highlight flash.
        if (m_fofSelected >= (int)m_fofClusters.size()) m_fofSelected = -1;
    }

    // Run = persistent live mode (button red), click again = stop.
    if (!m_fofLive) {
        // Green "Run FoF"
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.20f,0.55f,0.25f,1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f,0.68f,0.30f,1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.16f,0.45f,0.20f,1.0f));
        if (ImGui::Button("Run FoF", ImVec2(-1, 26)) && m_simulation) {
            m_fofLive = true;
            m_fofLiveLastFrame = -100000;   // trigger first detection immediately
        }
        ImGui::PopStyleColor(3);
    } else {
        // Red "Stop FoF"
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.65f,0.18f,0.18f,1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.80f,0.24f,0.24f,1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.52f,0.14f,0.14f,1.0f));
        if (ImGui::Button("Stop FoF", ImVec2(-1, 26))) { m_fofLive = false; m_fofCancel = true; }
        ImGui::PopStyleColor(3);
        if (m_fofComputing.load()) ImGui::TextDisabled("running (background)... (%zu pts)", m_fofPosCopy.size());
        else                       ImGui::TextDisabled("live - re-detects every %d frames", m_fofLiveInterval);
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90);
    ImGui::SliderInt("every##fofint", &m_fofLiveInterval, 5, 300, "%d fr");

    // Drive the background worker while live: relaunch every N frames once the
    // previous detection has finished. Keeps the catalog/overlay updating on their own.
    if (m_fofLive && !m_fofComputing.load() && m_simulation &&
        (m_currentFrame - m_fofLiveLastFrame >= m_fofLiveInterval)) {
        m_simulation->readPositions(m_positions);
        if (m_rotVelocities.size() < m_positions.size())
            m_simulation->readVelocities(m_rotVelocities);
        if (!m_positions.empty()) {
            m_fofLiveLastFrame = m_currentFrame;
            m_fofPosCopy = m_positions;
            m_fofVelCopy = m_rotVelocities;
            m_fofGsnap   = m_gravConstant;
            m_fofSoftSnap= m_softening;
            m_fofLinkSnap= m_fofLinkingScale;
            m_fofMinSnap = m_fofMinMembers;
            m_fofNegSnap = m_fofIncludeNeg;
            if (m_fofThread.joinable()) m_fofThread.join();
            m_fofCancel      = false;
            m_fofResultReady = false;
            m_fofComputing   = true;
            m_fofThread = std::thread([this]{
                // try/catch: a bad_alloc on very large N (FoF grid) would otherwise escape
                // the thread -> std::terminate -> frozen app (force-kill). We exit cleanly.
                try { computeFoF(); }
                catch (...) { m_fofResult.clear(); }
                m_fofResultReady = true;
                m_fofComputing   = false;
            });
        }
    }
    ImGui::Separator();

    if (!m_fofDone) { ImGui::TextDisabled("Click 'Run FoF' to detect clusters"); ImGui::End(); return; }

    ImGui::Text("%zu clusters found (>= %d members)", m_fofClusters.size(), m_fofMinMembers);
    ImGui::SameLine();
    if (ImGui::SmallButton("Export CSV") && !m_fofClusters.empty()) exportFoFCSV();
    if (m_fofSelected >= 0) {
        ImGui::SameLine();
        if (ImGui::SmallButton("Unselect")) m_fofSelected = -1;
    }
    ImGui::TextDisabled("Click a row to highlight its member particles (yellow).");
    ImGui::TextDisabled("Virial KE/|PE|:  ~0.5 = virialized  |  <0.5 = collapsing  |  >1 = unbound");

    if (ImGui::BeginTable("fof", 6,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupColumn("#");
        ImGui::TableSetupColumn("Members");
        ImGui::TableSetupColumn("Mass");
        ImGui::TableSetupColumn("R_rms (LY)");
        ImGui::TableSetupColumn("Center (LY)");
        ImGui::TableSetupColumn("KE/|PE|");
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < m_fofClusters.size(); i++) {
            const FoFCluster& c = m_fofClusters[i];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            char rowlbl[16]; snprintf(rowlbl, sizeof(rowlbl), "%zu", i+1);
            if (ImGui::Selectable(rowlbl, m_fofSelected == (int)i,
                    ImGuiSelectableFlags_SpanAllColumns))
                m_fofSelected = (m_fofSelected == (int)i) ? -1 : (int)i;
            ImGui::TableNextColumn(); ImGui::Text("%d", c.count);
            ImGui::TableNextColumn(); ImGui::Text("%.3g", c.mass);
            ImGui::TableNextColumn(); ImGui::Text("%.0f", c.radius * m_lyPerUnit);
            ImGui::TableNextColumn();
            ImGui::Text("(%.0f, %.0f)", c.cx * m_lyPerUnit, c.cy * m_lyPerUnit);
            ImGui::TableNextColumn();
            float v = c.virial;
            ImVec4 col = (v < 0.4f) ? ImVec4(0.5f,0.7f,1.0f,1.0f)     // collapsing (blue)
                       : (v > 1.0f) ? ImVec4(1.0f,0.5f,0.4f,1.0f)     // unbound (red)
                                    : ImVec4(0.5f,1.0f,0.5f,1.0f);    // virialized (green)
            ImGui::TextColored(col, "%.2f", v);
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

// ---------------------------------------------------------------------------
// Reset to defaults
// ---------------------------------------------------------------------------

void App::resetToDefaults()
{
    m_radius         = 500.0f;
    m_scaleUnit      = 1.0f;
    m_numPosiStars   = 50000;
    m_numNegaStars   = 50000;
    m_totalSolarMass = 100000.0f;
    m_negaDensity    = -1.0e5f;  // total halo mass (~ disk mass for balanced Janus)
    m_blackHoleMass  = 1000.0f;
    m_rotSpeed       = 0.0f;
    m_initSpeed      = 0.0f;
    m_timeStep       = 0.5f;
    m_softening      = 10.0f;
    m_gravConstant   = 6.67408e-11f;
    m_useTree        = false;
    m_treeTheta      = 0.5f;
    m_simSpeed       = 1.0f;
    m_spAmplitude    = 0.0f;
    m_spPatternSpeed = 0.01f;
    m_spPitchAngle   = 15.0f;
    m_spArms         = 2;
    m_glowEnabled    = true;  // Match Universe : default ON
    m_blurRadius     = 4.0f;
    m_glowIntensity  = 2.5f;
    m_showNegaMass   = true;
    m_colorByVelocity = false;
    m_velAutoScale   = true;
    m_showGrid       = true;
    m_lyPerUnit      = 10.0f;
    m_particleSize   = 0.4f;
    m_particleSizeNeg= 0.4f;
    m_rotCurveRMax   = 100000.0f;
    m_eqSelectedModel = 1;
    m_cudaDevice     = -1;
    if (m_simulation) m_simulation->setPhysicsModel(1);
}

// ---------------------------------------------------------------------------
// CSV Export
// ---------------------------------------------------------------------------

void App::exportRotationCurveCSV()
{
#ifdef _WIN32
    char filename[MAX_PATH] = "rotation_curve.csv";
    char desktopPath[MAX_PATH] = {};
    const char* up = getenv("USERPROFILE");
    if (up) snprintf(desktopPath, MAX_PATH, "%s\\Desktop", up);
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn); ofn.lpstrFilter = "CSV (*.csv)\0*.csv\0";
    ofn.lpstrFile = filename; ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = "csv"; ofn.lpstrTitle = "Export Rotation Curve";
    ofn.lpstrInitialDir = desktopPath[0] ? desktopPath : NULL;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetSaveFileNameA(&ofn)) return;
#else
    const char* filename = "rotation_curve.csv";
#endif
    const int BINS = 80;
    float sumV[BINS] = {}; int cnt[BINS] = {};
    float rLyMax = (m_rotCurveRMax > 0) ? m_rotCurveRMax : m_radius * m_lyPerUnit * 3.0f;
    float velConv = (m_lyPerUnit * 9.461e12f) / (m_timeStep * 3.156e13f);
    size_t n = m_rotVelocities.size() < m_positions.size() ? m_rotVelocities.size() : m_positions.size();
    for (size_t i = 0; i < n; i++) {
        if (m_positions[i].w <= 0) continue;
        float x = m_positions[i].x, y = m_positions[i].y;
        float r = sqrtf(x*x+y*y); if (r < 1) continue;
        float vt = fabsf((-y * m_rotVelocities[i].x + x * m_rotVelocities[i].y) / r * velConv);
        float rLy = r * m_lyPerUnit; if (rLy > rLyMax) continue;
        int b = (int)(rLy / rLyMax * BINS); if (b >= BINS) b = BINS-1;
        sumV[b] += vt; cnt[b]++;
    }
    std::ofstream f(filename);
    f << "Radius_LY,Velocity_kms\n";
    for (int b = 0; b < BINS; b++) {
        if (cnt[b] < 3) continue;
        f << (b + 0.5f) / BINS * rLyMax << "," << sumV[b] / cnt[b] << "\n";
    }
}

// Save the FoF cluster catalog as a CSV (one row per cluster).
void App::exportFoFCSV()
{
    if (m_fofClusters.empty()) return;
#ifdef _WIN32
    char filename[MAX_PATH] = "fof_clusters.csv";
    char desktopPath[MAX_PATH] = {};
    const char* up = getenv("USERPROFILE");
    if (up) snprintf(desktopPath, MAX_PATH, "%s\\Desktop", up);
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn); ofn.lpstrFilter = "CSV (*.csv)\0*.csv\0";
    ofn.lpstrFile = filename; ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = "csv"; ofn.lpstrTitle = "Export FoF Clusters";
    ofn.lpstrInitialDir = desktopPath[0] ? desktopPath : NULL;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetSaveFileNameA(&ofn)) return;
#else
    const char* filename = "fof_clusters.csv";
#endif
    std::ofstream f(filename);
    f << "Index,Members,Mass,R_rms_LY,Cx_LY,Cy_LY,Cz_LY,Vx,Vy,Vz,Virial_KE_over_PE,State\n";
    for (size_t i = 0; i < m_fofClusters.size(); i++) {
        const FoFCluster& c = m_fofClusters[i];
        const char* state = (c.virial < 0.4f) ? "collapsing"
                          : (c.virial > 1.0f) ? "unbound" : "virialized";
        f << (i+1) << "," << c.count << "," << c.mass << ","
          << (c.radius * m_lyPerUnit) << ","
          << (c.cx * m_lyPerUnit) << "," << (c.cy * m_lyPerUnit) << "," << (c.cz * m_lyPerUnit) << ","
          << c.vx << "," << c.vy << "," << c.vz << ","
          << c.virial << "," << state << "\n";
    }
}

void App::exportEnergyCSV()
{
#ifdef _WIN32
    char filename[MAX_PATH] = "energy_history.csv";
    char desktopPath[MAX_PATH] = {};
    const char* up = getenv("USERPROFILE");
    if (up) snprintf(desktopPath, MAX_PATH, "%s\\Desktop", up);
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn); ofn.lpstrFilter = "CSV (*.csv)\0*.csv\0";
    ofn.lpstrFile = filename; ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = "csv"; ofn.lpstrTitle = "Export Energy History";
    ofn.lpstrInitialDir = desktopPath[0] ? desktopPath : NULL;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetSaveFileNameA(&ofn)) return;
#else
    const char* filename = "energy_history.csv";
#endif
    std::ofstream f(filename);
    f << "Time_MY,KineticEnergy\n";
    for (int i = 0; i < m_energyCount; i++) {
        int idx = (m_energyHead - m_energyCount + i + ENERGY_HISTORY) % ENERGY_HISTORY;
        f << m_energyTimes[idx] << "," << m_energyHistory[idx] << "\n";
    }
}

void App::exportDensityCSV()
{
#ifdef _WIN32
    char filename[MAX_PATH] = "density_profile.csv";
    char desktopPath[MAX_PATH] = {};
    const char* up = getenv("USERPROFILE");
    if (up) snprintf(desktopPath, MAX_PATH, "%s\\Desktop", up);
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn); ofn.lpstrFilter = "CSV (*.csv)\0*.csv\0";
    ofn.lpstrFile = filename; ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = "csv"; ofn.lpstrTitle = "Export Density Profile";
    ofn.lpstrInitialDir = desktopPath[0] ? desktopPath : NULL;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetSaveFileNameA(&ofn)) return;
#else
    const char* filename = "density_profile.csv";
#endif
    const int BINS = 80;
    float binPos[BINS]={}, binNeg[BINS]={};
    float rMax = m_radius * m_lyPerUnit * 4.0f;
    for (size_t i = 0; i < m_positions.size(); i++) {
        float r = sqrtf(m_positions[i].x*m_positions[i].x + m_positions[i].y*m_positions[i].y) * m_lyPerUnit;
        int b = (int)(r / rMax * BINS); if (b < 0) b=0; if (b >= BINS) b=BINS-1;
        if (m_positions[i].w > 0) binPos[b]++; else if (m_positions[i].w < 0) binNeg[b]++;
    }
    std::ofstream f(filename);
    f << "Radius_LY,Count_Positive,Count_Negative\n";
    for (int b = 0; b < BINS; b++)
        f << (b + 0.5f) / BINS * rMax << "," << (int)binPos[b] << "," << (int)binNeg[b] << "\n";
}

void App::exportBarycentersCSV()
{
#ifdef _WIN32
    char filename[MAX_PATH] = "barycenters.csv";
    char desktopPath[MAX_PATH] = {};
    const char* up = getenv("USERPROFILE");
    if (up) snprintf(desktopPath, MAX_PATH, "%s\\Desktop", up);
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn); ofn.lpstrFilter = "CSV (*.csv)\0*.csv\0";
    ofn.lpstrFile = filename; ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = "csv"; ofn.lpstrTitle = "Export Barycenters";
    ofn.lpstrInitialDir = desktopPath[0] ? desktopPath : NULL;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetSaveFileNameA(&ofn)) return;
#else
    const char* filename = "barycenters.csv";
#endif
    std::ofstream f(filename);
    f << "Time_MY,Distance_LY\n";
    for (int i = 0; i < m_baryCount; i++) {
        int idx = (m_baryHead - m_baryCount + i + BARY_HISTORY) % BARY_HISTORY;
        f << m_baryTimes[idx] << "," << m_baryDist[idx] << "\n";
    }
}

void App::exportFourierCSV()
{
#ifdef _WIN32
    char filename[MAX_PATH] = "fourier_modes.csv";
    char desktopPath[MAX_PATH] = {};
    const char* up = getenv("USERPROFILE");
    if (up) snprintf(desktopPath, MAX_PATH, "%s\\Desktop", up);
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn); ofn.lpstrFilter = "CSV (*.csv)\0*.csv\0";
    ofn.lpstrFile = filename; ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = "csv"; ofn.lpstrTitle = "Export Fourier Modes";
    ofn.lpstrInitialDir = desktopPath[0] ? desktopPath : NULL;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetSaveFileNameA(&ofn)) return;
#else
    const char* filename = "fourier_modes.csv";
#endif
    std::ofstream f(filename);
    f << "Time_MY,A1,A2,A3,A4\n";
    for (int i = 0; i < m_fourierCount; i++) {
        int idx = (m_fourierHead - m_fourierCount + i + FOURIER_HISTORY) % FOURIER_HISTORY;
        f << m_fourierTimes[idx]
          << "," << m_fourierA[0][idx]
          << "," << m_fourierA[1][idx]
          << "," << m_fourierA[2][idx]
          << "," << m_fourierA[3][idx] << "\n";
    }
}

// ---------------------------------------------------------------------------
// About dialog
// ---------------------------------------------------------------------------

void App::renderAbout()
{
    if (!m_showAbout) return;
    ImGui::SetNextWindowSize(ImVec2(560, 520), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2((float)m_width*0.5f - 280, (float)m_height*0.5f - 260), ImGuiCond_FirstUseEver);
    ImGui::Begin("About Janus Galaxy", &m_showAbout,
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

    ImGui::TextColored(ImVec4(0.4f,0.8f,1.0f,1), "Janus Galaxy");
    ImGui::Separator();

    ImGui::TextColored(ImVec4(0.8f,0.8f,0.4f,1), "Project origin");
    ImGui::TextWrapped("42 School Silicon Valley");
    if (ImGui::SmallButton("https://j-gravity.github.io/web-landing/index.html"))
        SDL_OpenURL("https://j-gravity.github.io/web-landing/index.html");
    ImGui::Dummy(ImVec2(0,6));

    ImGui::TextColored(ImVec4(0.8f,0.8f,0.4f,1), "Janus Cosmological Model — Jean-Pierre Petit");
    ImGui::TextWrapped("Publications:");
    auto link = [](const char* url) {
        if (ImGui::SmallButton(url))
            SDL_OpenURL(url);
    };
    link("http://www.jp-petit.org/papers/cosmo/2026-Mathematical-and-Geometrical-Inconsistency-of-the-Black-Hole-Model.Part%20I.pdf");
    link("http://www.jp-petit.org/papers/cosmo/2025-10-14-JMP-Alternative-to-Black-Holes-Gravastars-and-Plugstars.pdf");
    link("http://www.jp-petit.org/papers/cosmo/2025-10-14-Astroph-and-Space-Sci-Contribution-of-Kinetic-Theory-to-Galactic-Dynamics.pdf");
    link("http://www.jp-petit.org/papers/cosmo/2024-11-EPJC.pdf");
    link("http://www.jp-petit.org/papers/cosmo/2024-12-RMC-study-of-symmetries-through-the-action-on-torsors-of-the-janus-symplectic-group.pdf");
    link("https://januscosmologicalmodel.com/");
    link("https://hal.science/hal-04583560v1/document");
    ImGui::Dummy(ImVec2(0,6));

    ImGui::TextColored(ImVec4(0.8f,0.8f,0.4f,1), "Windows Port");
    ImGui::TextWrapped("Alexandre");

    ImGui::Dummy(ImVec2(0,8));
    if (ImGui::Button("Close", ImVec2(-1, 0))) m_showAbout = false;
    ImGui::End();
}

// ---------------------------------------------------------------------------
// Presets — named parameter configurations
// ---------------------------------------------------------------------------

static std::string presetsDir()
{
#ifdef _WIN32
    char exePath[MAX_PATH] = {};
    GetModuleFileNameA(NULL, exePath, MAX_PATH);
    std::string dir(exePath);
    size_t sep = dir.find_last_of("\\/");
    if (sep != std::string::npos) dir = dir.substr(0, sep);
    return dir + "\\presets";
#else
    return "presets";
#endif
}

std::vector<std::string> App::listPresets()
{
    std::vector<std::string> result;
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directory(presetsDir(), ec);
    for (const auto& entry : fs::directory_iterator(presetsDir(), ec))
    {
        if (entry.path().extension() == ".cfg")
            result.push_back(entry.path().stem().string());
    }
    std::sort(result.begin(), result.end());
    return result;
}

void App::savePreset(const std::string& name)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directory(presetsDir(), ec);
    saveSettings(presetsDir() + "/" + name + ".cfg");
}

void App::loadPreset(const std::string& name)
{
    loadSettings(presetsDir() + "/" + name + ".cfg");
    if (m_simulation)
        m_simulation->setPhysicsModel(m_eqSelectedModel);
}

// ---------------------------------------------------------------------------

void App::saveSettings(const std::string& path)
{
    std::ofstream f(path);
    if (!f.is_open()) { std::cerr << "Cannot write settings: " << path << "\n"; return; }

    f << "# Janus Galaxy settings\n";
    f << "[Galaxy]\n";
    f << "radius="          << m_radius          << "\n";
    f << "scale_unit="      << m_scaleUnit        << "\n";
    f << "num_posi_stars="  << m_numPosiStars     << "\n";
    f << "num_nega_stars="  << m_numNegaStars     << "\n";
    f << "total_solar_mass="<< m_totalSolarMass   << "\n";
    f << "nega_density="    << m_negaDensity      << "\n";
    f << "black_hole_mass=" << m_blackHoleMass    << "\n";
    f << "rot_speed="       << m_rotSpeed         << "\n";
    f << "init_speed="      << m_initSpeed        << "\n";
    f << "[Simulation]\n";
    f << "time_step="       << m_timeStep         << "\n";
    f << "softening="       << m_softening        << "\n";
    f << "grav_constant="   << m_gravConstant     << "\n";
    f << "useTree="         << (m_useTree ? 1 : 0) << "\n";
    f << "treeTheta="       << m_treeTheta        << "\n";
    f << "treeQuad="        << (m_treeQuad ? 1 : 0) << "\n";
    f << "sim_speed="       << m_simSpeed         << "\n";
    f << "[DensityWaves]\n";
    f << "sp_amplitude="       << m_spAmplitude            << "\n";
    f << "density_waves_on=" << (m_densityWavesEnabled ? 1 : 0) << "\n";
    f << "sp_pattern_speed="<< m_spPatternSpeed   << "\n";
    f << "sp_pitch_angle="  << m_spPitchAngle     << "\n";
    f << "sp_arms="         << m_spArms           << "\n";
    f << "[Glow]\n";
    f << "glow_enabled="    << (m_glowEnabled ? 1 : 0) << "\n";
    f << "blur_radius="     << m_blurRadius       << "\n";
    f << "glow_intensity="  << m_glowIntensity    << "\n";
    f << "[RayVolume]\n";
    f << "ray_trace="       << (m_rayTrace ? 1 : 0) << "\n";
    f << "vol_grid_n="      << m_volGridN         << "\n";
    f << "vol_steps="       << m_volSteps         << "\n";
    f << "vol_gain="        << m_volGain          << "\n";
    f << "vol_contrast="    << m_volContrast      << "\n";
    f << "vol_absorb="      << m_volAbsorb        << "\n";
    f << "vol_exposure="    << m_volExposure      << "\n";
    f << "vol_smooth="      << m_volSmooth        << "\n";
    f << "vol_box_sigma="   << m_volBoxSigma      << "\n";
    f << "vol_galaxy_pal="  << (m_volGalaxyPalette ? 1 : 0) << "\n";
    f << "vol_color_vel="   << (m_volColorByVel ? 1 : 0) << "\n";
    f << "vol_stars="       << (m_volStars ? 1 : 0) << "\n";
    f << "vol_star_scale="  << m_volStarScale     << "\n";
    f << "vol_star_bright=" << m_volStarBright     << "\n";
    f << "vol_halo_size="   << m_volHaloSize       << "\n";
    f << "[Display]\n";
    f << "show_nega_mass="  << (m_showNegaMass ? 1 : 0) << "\n";
    f << "color_by_vel="    << (m_colorByVelocity ? 1 : 0) << "\n";
    f << "show_grid="       << (m_showGrid ? 1 : 0) << "\n";
    f << "ly_per_unit="     << m_lyPerUnit        << "\n";
    f << "particle_size="    << m_particleSize    << "\n";
    f << "particle_size_neg="<< m_particleSizeNeg << "\n";
    f << "[Analytics]\n";
    f << "show_rot_curve="  << (m_showRotCurve       ? 1 : 0) << "\n";
    f << "show_density="    << (m_showDensityProfile ? 1 : 0) << "\n";
    f << "show_phase="      << (m_showPhaseSpace     ? 1 : 0) << "\n";
    f << "show_energy="     << (m_showEnergyPlot     ? 1 : 0) << "\n";
    f << "show_dispersion=" << (m_showDispersion     ? 1 : 0) << "\n";
    f << "auto_refresh="    << (m_rotAutoRefresh     ? 1 : 0) << "\n";
    f << "refresh_interval="<< m_rotRefreshInterval << "\n";
    f << "[State]\n";
    f << "state_path="      << m_stateFilePath     << "\n";
    f << "[CUDA]\n";
    f << "cuda_device="     << m_cudaDevice        << "\n";
}

void App::loadSettings(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) return;

    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#' || line[0] == '[') continue;
        auto sep = line.find('=');
        if (sep == std::string::npos) continue;
        std::string key = line.substr(0, sep);
        std::string val = line.substr(sep + 1);
        if (val.empty()) continue;

        try {
            if      (key == "radius")          m_radius          = std::stof(val);
            else if (key == "scale_unit")      m_scaleUnit       = std::stof(val);
            else if (key == "num_posi_stars")  m_numPosiStars    = std::stoi(val);
            else if (key == "num_nega_stars")  m_numNegaStars    = std::stoi(val);
            else if (key == "total_solar_mass")m_totalSolarMass  = std::stof(val);
            else if (key == "nega_density")    m_negaDensity     = std::stof(val);
            else if (key == "black_hole_mass") m_blackHoleMass   = std::stof(val);
            else if (key == "rot_speed")       m_rotSpeed        = std::stof(val);
            else if (key == "init_speed")      m_initSpeed       = std::stof(val);
            else if (key == "time_step")       m_timeStep        = std::stof(val);
            else if (key == "softening")       m_softening       = std::stof(val);
            else if (key == "grav_constant")   m_gravConstant    = std::stof(val);
            else if (key == "useTree")         m_useTree         = std::stoi(val) != 0;
            else if (key == "treeTheta")       m_treeTheta       = std::stof(val);
            else if (key == "treeQuad")        m_treeQuad        = std::stoi(val) != 0;
            else if (key == "sim_speed")       m_simSpeed        = std::stof(val);
            else if (key == "sp_amplitude")       m_spAmplitude          = std::stof(val);
            else if (key == "density_waves_on")   m_densityWavesEnabled  = std::stoi(val) != 0;
            else if (key == "sp_pattern_speed")m_spPatternSpeed  = std::stof(val);
            else if (key == "sp_pitch_angle")  m_spPitchAngle    = std::stof(val);
            else if (key == "sp_arms")         m_spArms          = std::max(1, std::min(20, std::stoi(val)));
            else if (key == "glow_enabled")    m_glowEnabled     = std::stoi(val) != 0;
            else if (key == "blur_radius")     m_blurRadius      = std::stof(val);
            else if (key == "glow_intensity")  m_glowIntensity   = std::stof(val);
            else if (key == "ray_trace")       m_rayTrace        = std::stoi(val) != 0;
            else if (key == "vol_grid_n")      m_volGridN        = std::stoi(val);
            else if (key == "vol_steps")       m_volSteps        = std::stoi(val);
            else if (key == "vol_gain")        m_volGain         = std::stof(val);
            else if (key == "vol_contrast")    m_volContrast     = std::stof(val);
            else if (key == "vol_absorb")      m_volAbsorb       = std::stof(val);
            else if (key == "vol_exposure")    m_volExposure     = std::stof(val);
            else if (key == "vol_smooth")      m_volSmooth       = std::stof(val);
            else if (key == "vol_box_sigma")   m_volBoxSigma     = std::stof(val);
            else if (key == "vol_galaxy_pal")  m_volGalaxyPalette = std::stoi(val) != 0;
            else if (key == "vol_color_vel")   m_volColorByVel    = std::stoi(val) != 0;
            else if (key == "vol_stars")       m_volStars         = std::stoi(val) != 0;
            else if (key == "vol_star_scale")  m_volStarScale     = std::stof(val);
            else if (key == "vol_star_bright") m_volStarBright     = std::stof(val);
            else if (key == "vol_halo_size")   m_volHaloSize       = std::stof(val);
            else if (key == "show_nega_mass")  m_showNegaMass    = std::stoi(val) != 0;
            else if (key == "color_by_vel")    m_colorByVelocity = std::stoi(val) != 0;
            else if (key == "show_grid")       m_showGrid        = std::stoi(val) != 0;
            else if (key == "ly_per_unit")     m_lyPerUnit       = std::stof(val);
            else if (key == "particle_size")     m_particleSize    = std::stof(val);
            else if (key == "particle_size_neg") m_particleSizeNeg = std::stof(val);
            else if (key == "show_rot_curve")    m_showRotCurve    = std::stoi(val) != 0;
            else if (key == "show_density")    m_showDensityProfile = std::stoi(val) != 0;
            else if (key == "show_phase")      m_showPhaseSpace  = std::stoi(val) != 0;
            else if (key == "show_energy")     m_showEnergyPlot  = std::stoi(val) != 0;
            else if (key == "show_dispersion") m_showDispersion  = std::stoi(val) != 0;
            else if (key == "auto_refresh")    m_rotAutoRefresh  = std::stoi(val) != 0;
            else if (key == "refresh_interval")m_rotRefreshInterval = std::stoi(val);
            else if (key == "state_path")      strncpy(m_stateFilePath, val.c_str(), sizeof(m_stateFilePath)-1);
            else if (key == "cuda_device")     m_cudaDevice = std::stoi(val);
        } catch (...) {}
    }
}

App::~App()
{
    saveSettings("janusgalaxy.cfg");

    if (m_videoRecording) stopVideoRecording();      // finalize MP4 if still recording
    m_fofCancel = true;                              // signal abort BEFORE the join -> no hang on shutdown
    if (m_fofThread.joinable()) m_fofThread.join();  // wait for any FoF worker
    if (m_densThread.joinable()) m_densThread.join(); // wait for any density worker
    if (m_interopRes) interopUnregister(m_interopRes);
    if (m_dSmooth)    interopFree(m_dSmooth);
    if (m_simulation) delete m_simulation;
    if (m_camera) delete m_camera;
    if (m_renderer) delete m_renderer;

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    if (m_glContext) SDL_GL_DeleteContext(m_glContext);
    if (m_window) SDL_DestroyWindow(m_window);
    SDL_Quit();
}

bool App::init()
{
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
    {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return false;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    m_window = SDL_CreateWindow("Janus Galaxy",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        m_width, m_height,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN);
    if (!m_window) { std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl; return false; }

    m_glContext = SDL_GL_CreateContext(m_window);
    if (!m_glContext) { std::cerr << "SDL_GL_CreateContext Error: " << SDL_GetError() << std::endl; return false; }
    SDL_GL_MakeCurrent(m_window, m_glContext);
    SDL_GL_SetSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) { std::cerr << "GLEW Error: " << glewGetErrorString(glewErr) << std::endl; return false; }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // TTF font (Windows Segoe UI) with EXTENDED glyphs -> replaces ProggyClean
    // embedded font (accents, em-dash, σ, ȧ, arrows display instead of « ? »).
    {
        static const ImWchar ranges[] = {
            0x0020, 0x00FF,   // Basic Latin + supplement (accents)
            0x0100, 0x024F,   // Latin Extended A/B
            0x0370, 0x03FF,   // Grec (σ …)
            0x2000, 0x206F,   // General punctuation (em-dash, …)
            0x2190, 0x21FF,   // Arrows
            0,
        };
        ImFont* f = io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\segoeui.ttf", 16.0f, nullptr, ranges);
        if (!f) io.Fonts->AddFontDefault();   // fallback si la police est absente
    }

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.TabRounding = 7.f; style.GrabRounding = 7.f; style.ChildRounding = 7.f;
    style.FrameRounding = 7.f; style.PopupRounding = 7.f; style.WindowRounding = 7.f;
    style.ScrollbarRounding = 7.f;
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }
    // Barres de menu (CollapsingHeaders) en vert
    style.Colors[ImGuiCol_Header]        = ImVec4(0.13f, 0.42f, 0.20f, 0.90f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.18f, 0.55f, 0.27f, 0.95f);
    style.Colors[ImGuiCol_HeaderActive]  = ImVec4(0.20f, 0.62f, 0.30f, 1.00f);

    ImGui_ImplSDL2_InitForOpenGL(m_window, m_glContext);
    ImGui_ImplOpenGL3_Init("#version 330");

    m_renderer = new PointRenderer(m_width, m_height);
    if (!m_renderer->init()) { std::cerr << "Failed to init PointRenderer\n"; return false; }

    int vpW0 = m_width - 340;
    int vpH0 = m_height;
    m_camera = new Camera(vpW0 > 0 ? vpW0 : m_width, vpH0,
        new Vector(0, 0, 2000.0), new Matrix(ROTATION, 0.0, 0.0, 0.0),
        60.0, 10.0, 1.0e9f);

    Color::init_color_table();
    Color::ninit_color_table();
    Color::vel_init_color_table();
    Color::getPosPalette(m_posPalette);
    Color::getNegPalette(m_negPalette);
    Color::getVelPalette(m_velPalette);

    // "true galaxy colors" palette (Hubble look) for the Soft Halo: the HUE
    // encodes the column density. Low -> high: dark blue (diffuse halo) -> blue
    // (spiral arms, young stars) -> cream-white (disk) -> warm gold (bulge) ->
    // golden-white (bright core).
    {
        const int NS = 8;
        const float st[NS][4] = {
            {0.00f, 0.01f, 0.02f, 0.06f},
            {0.14f, 0.05f, 0.13f, 0.42f},
            {0.30f, 0.20f, 0.40f, 0.82f},
            {0.46f, 0.55f, 0.72f, 0.95f},
            {0.60f, 0.90f, 0.92f, 0.93f},
            {0.74f, 0.98f, 0.94f, 0.80f},
            {0.88f, 1.00f, 0.86f, 0.55f},
            {1.00f, 1.00f, 0.97f, 0.88f},
        };
        for (int i = 0; i < 32; i++) {
            float t = i / 31.0f;
            int seg = 0;
            while (seg < NS-2 && t > st[seg+1][0]) seg++;
            float t0 = st[seg][0], t1 = st[seg+1][0];
            float u = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0f;
            for (int k = 0; k < 3; k++)
                m_galaxyPalette[i][k] = st[seg][k+1] + (st[seg+1][k+1] - st[seg][k+1]) * u;
        }
    }

    // Palette "nebuleuse" dediee au ray-traced : diffus FROID (faible densite) ->
    // coeur CREME (forte densite). 6 teintes fournies, ordre sombre -> brillant.
    {
        const int NS = 6;
        const float ns[NS][3] = {
            {0.40f, 0.16f, 0.42f},   // violet  (diffus le plus tenu)
            {0.17f, 0.17f, 0.42f},   // indigo
            {0.16f, 0.40f, 0.56f},   // bleu acier
            {0.16f, 0.54f, 0.60f},   // teal
            {0.32f, 0.66f, 0.54f},   // vert d'eau
            {0.97f, 0.85f, 0.58f},   // creme   (coeur dense)
        };
        for (int i = 0; i < 32; i++) {
            float t = i / 31.0f * (NS - 1);
            int seg = (int)t; if (seg > NS - 2) seg = NS - 2;
            float u = t - seg;
            for (int k = 0; k < 3; k++)
                m_nebulaPalette[i][k] = ns[seg][k] + (ns[seg+1][k] - ns[seg][k]) * u;
        }
    }

    m_simulation = new Simulation();
    if (!m_simulation->init("gpu-nbody/pair3.cl", "nbody", m_cudaDevice))
    {
#ifdef _WIN32
        MessageBoxA(NULL,
            "No CUDA-capable GPU was found.\n\n"
            "j-gravity requires an NVIDIA GPU with CUDA support.\n"
            "The simulation will not run.",
            "CUDA Initialization Error",
            MB_OK | MB_ICONERROR);
#endif
        std::cerr << "Failed to init CUDA simulation\n";
    }

    m_keystate = (Uint8*)SDL_GetKeyboardState(NULL);

    loadSettings("janusgalaxy.cfg");

    startSimulation();
    m_simRunning = false;
    m_simPaused = true;
    return true;
}

void App::processEvents(bool& running)
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        ImGui_ImplSDL2_ProcessEvent(&event);

        if (event.type == SDL_QUIT) running = false;

        if (event.type == SDL_WINDOWEVENT
            && event.window.event == SDL_WINDOWEVENT_CLOSE
            && event.window.windowID == SDL_GetWindowID(m_window))
            running = false;

        if (event.type == SDL_KEYDOWN)
        {
            if (event.key.keysym.sym == SDLK_ESCAPE) running = false;
            if (event.key.keysym.sym == SDLK_SPACE && !ImGui::GetIO().WantCaptureKeyboard)
                m_simPaused = !m_simPaused;
        }

        if (event.type == SDL_MOUSEWHEEL && !ImGui::GetIO().WantCaptureMouse)
        {
            if (m_autoFit) {
                // Auto-fit active: the wheel adjusts the FRAMING (apparent size
                // target) instead of the distance — otherwise it would fight against the auto-fit.
                m_autoFitFill *= (event.wheel.y > 0) ? 1.08f : 0.92f;
                m_autoFitFill = std::min(std::max(m_autoFitFill, 0.10f), 0.95f);
                m_autoFitLastFrame = -100000;   // recadre tout de suite
            } else {
                float factor = (event.wheel.y > 0) ? 0.92f : 1.08f;
                int mx, my;
                SDL_GetMouseState(&mx, &my);
                // SDL coords are window-relative — just subtract the controls panel
                mx -= 340;
                m_camera->zoom(factor, mx, my);
            }
        }

        if (event.type == SDL_MOUSEBUTTONDOWN
            && event.button.button == SDL_BUTTON_LEFT
            && !ImGui::GetIO().WantCaptureMouse)
        {
            int mx, my;
            SDL_GetMouseState(&mx, &my);
            // SDL coords are window-relative — viewport starts at x=340
            int vpX = 340;
            int vpW = m_width - 340;
            int vpH = m_height;
            if (mx >= vpX && mx < vpX + vpW && my >= 0 && my < vpH)
            {
                if (m_annotTool != 0)
                {
                    // ── Annotation placement (window-relative coords) ──────────
                    if (m_annotTool == 2) {
                        // Text: single click → place + open text editor
                        Annotation a{};
                        a.type = 2; a.x0 = (float)mx; a.y0 = (float)my;
                        a.x1 = (float)mx; a.y1 = (float)my;
                        a.col = IM_COL32((int)(m_annotColor[0]*255),(int)(m_annotColor[1]*255),
                                         (int)(m_annotColor[2]*255),255);
                        a.text[0] = '\0';
                        m_annotations.push_back(a);
                        m_annotEditIdx = (int)m_annotations.size() - 1;
                        m_annotTextBuf[0] = '\0';
                    } else if (m_annotStage == 0) {
                        // Arrow/rect/circle: record first point
                        m_annotX0 = (float)mx; m_annotY0 = (float)my;
                        m_annotStage = 1;
                    } else {
                        // Second point → commit annotation
                        Annotation a{};
                        a.type = m_annotTool;
                        a.x0 = m_annotX0; a.y0 = m_annotY0;
                        a.x1 = (float)mx; a.y1 = (float)my;
                        a.col = IM_COL32((int)(m_annotColor[0]*255),(int)(m_annotColor[1]*255),
                                         (int)(m_annotColor[2]*255),255);
                        a.text[0] = '\0';
                        m_annotations.push_back(a);
                        m_annotStage = 0;
                        if (m_annotTool == 1) {   // arrow → ask for label text
                            m_annotEditIdx = (int)m_annotations.size() - 1;
                            m_annotTextBuf[0] = '\0';
                        }
                    }
                }
                else if (m_showVelocityField && m_velSelectMode)
                {
                    // Begin region selection for streamlines (window-relative coords)
                    m_velSelDragging = true;
                    m_velSelActive   = false;
                    m_velSelX0 = m_velSelX1 = (float)mx;
                    m_velSelY0 = m_velSelY1 = (float)my;
                }
                else
                {
                    m_markerPickPending = true;
                    m_markerPickX = mx - vpX;
                    m_markerPickY = my;
                }
            }
        }

        // ── Rectangle selection (left click-drag) for streamlines ──────────
        if (event.type == SDL_MOUSEMOTION && m_velSelDragging)
        {
            m_velSelX1 = (float)event.motion.x;
            m_velSelY1 = (float)event.motion.y;
        }
        if (event.type == SDL_MOUSEBUTTONUP
            && event.button.button == SDL_BUTTON_LEFT
            && m_velSelDragging)
        {
            m_velSelDragging = false;
            // Valid selection only if the box is big enough (avoids stray clicks).
            // Capture happens next frame in renderVelocityField (needs the camera
            // projection): the box selects a SET OF STARS, then the box vanishes
            // and those stars' arrows follow them over time.
            float w = fabsf(m_velSelX1 - m_velSelX0);
            float h = fabsf(m_velSelY1 - m_velSelY0);
            m_velSelPending = (w > 6.0f && h > 6.0f);
            m_velSelActive  = false;
        }

        // ── Right click: orbit the camera ────────────────────────────────────
        if (event.type == SDL_MOUSEBUTTONDOWN
            && event.button.button == SDL_BUTTON_RIGHT
            && !ImGui::GetIO().WantCaptureMouse)
        {
            m_rightMouseDown = true;
            m_lastMouseX = event.button.x;
            m_lastMouseY = event.button.y;
        }
        if (event.type == SDL_MOUSEBUTTONUP
            && event.button.button == SDL_BUTTON_RIGHT)
        {
            m_rightMouseDown = false;
        }
        if (event.type == SDL_MOUSEMOTION && m_rightMouseDown)
        {
            // Orbit type CAO/Blender : tourner autour du centre (0,0,0)
            float dx = (float)(event.motion.x - m_lastMouseX);
            float dy = (float)(event.motion.y - m_lastMouseY);
            // dx = yaw horizontal, dy = pitch vertical
            // ~0.3° par pixel
            m_camera->orbit(dx * 0.005f, dy * 0.005f);
            m_lastMouseX = event.motion.x;
            m_lastMouseY = event.motion.y;
        }

        // ── Middle button (wheel): PAN (translation in the view plane) ──
        if (event.type == SDL_MOUSEBUTTONDOWN
            && event.button.button == SDL_BUTTON_MIDDLE
            && !ImGui::GetIO().WantCaptureMouse)
        {
            m_middleMouseDown = true;
            m_lastMouseX = event.button.x;
            m_lastMouseY = event.button.y;
        }
        if (event.type == SDL_MOUSEBUTTONUP
            && event.button.button == SDL_BUTTON_MIDDLE)
        {
            m_middleMouseDown = false;
        }
        if (event.type == SDL_MOUSEMOTION && m_middleMouseDown)
        {
            float dx = (float)(event.motion.x - m_lastMouseX);
            float dy = (float)(event.motion.y - m_lastMouseY);
            // View-aligned pan via the camera (works in face-on, edge-on AND orbit/3D).
            m_camera->panScreen(dx, dy);
            m_lastMouseX = event.motion.x;
            m_lastMouseY = event.motion.y;
        }

        if (event.type == SDL_WINDOWEVENT
            && event.window.windowID == SDL_GetWindowID(m_window))
        {
            bool needResize =
                event.window.event == SDL_WINDOWEVENT_RESIZED      ||
                event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED  ||
                event.window.event == SDL_WINDOWEVENT_DISPLAY_CHANGED;

            if (needResize)
            {
                // Read actual window size (reliable across DPI changes)
                SDL_GetWindowSize(m_window, &m_width, &m_height);
                if (m_width  < 1) m_width  = 1;
                if (m_height < 1) m_height = 1;
                m_renderer->resize(m_width, m_height);

                // Rebuild camera with correct new dimensions
                bool wasEdgeOn = m_edgeOnView;
                delete m_camera;
                if (wasEdgeOn) {
                    m_camera = new Camera(m_width, m_height,
                        new Vector(0, -2000.0, 0), new Matrix(ROTATION,0,0,0),
                        60.0, 10.0, 1.0e9f);
                    m_camera->setEdgeOn(true);
                } else {
                    m_camera = new Camera(m_width, m_height,
                        new Vector(0, 0, 2000.0), new Matrix(ROTATION,0,0,0),
                        60.0, 10.0, 1.0e9f);
                    m_camera->setOrbitMode(false);
                }
            }
        }
    }
}

void App::updateCamera()
{
    if (m_keystate[SDL_SCANCODE_W]) m_camera->mod_angles( M_PI/64.0, 0.0, 0.0);
    if (m_keystate[SDL_SCANCODE_S]) m_camera->mod_angles(-M_PI/64.0, 0.0, 0.0);
    if (m_keystate[SDL_SCANCODE_A]) m_camera->mod_angles(0.0, -M_PI/64.0, 0.0);
    if (m_keystate[SDL_SCANCODE_D]) m_camera->mod_angles(0.0,  M_PI/64.0, 0.0);
    if (m_keystate[SDL_SCANCODE_Q]) m_camera->mod_angles(0.0, 0.0, -M_PI/64.0);
    if (m_keystate[SDL_SCANCODE_E]) m_camera->mod_angles(0.0, 0.0,  M_PI/64.0);
    if (m_keystate[SDL_SCANCODE_UP])    m_camera->mod_location(0.0, -5000.0, 0.0);
    if (m_keystate[SDL_SCANCODE_DOWN])  m_camera->mod_location(0.0,  5000.0, 0.0);
    if (m_keystate[SDL_SCANCODE_LEFT])  m_camera->mod_location(-5000.0, 0.0, 0.0);
    if (m_keystate[SDL_SCANCODE_RIGHT]) m_camera->mod_location( 5000.0, 0.0, 0.0);
}

// SmoothDamp (critically-damped spring, Unity/Game Programming Gems 4 style): brings
// `current` toward `target` with a continuous VELOCITY (C1) -> no jolt even when
// the target changes in steps. `vel` is the internal state (kept between frames).
static float smoothDamp(float current, float target, float& vel,
                        float smoothTime, float dt)
{
    smoothTime = (smoothTime < 0.0001f) ? 0.0001f : smoothTime;
    float omega = 2.0f / smoothTime;
    float x     = omega * dt;
    float expo  = 1.0f / (1.0f + x + 0.48f*x*x + 0.235f*x*x*x);
    float change = current - target;
    float temp   = (vel + omega*change) * dt;
    vel  = (vel - omega*temp) * expo;
    float out = target + (change + temp) * expo;
    if (((target - current) > 0.0f) == (out > target)) {  // anti-overshoot
        out = target;
        vel = (out - target) / dt;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Auto-fit: keeps the structure at constant apparent size while it
// expands. ROBUST radius (P90, ignores ejecta) -> target distance, then:
//   (a) continuous low-pass on the target -> erases the staircase of recomputations;
//   (b) SmoothDamp with continuous velocity, real dt -> smooth motion without jolts.
// NB: m_positions is already refreshed each frame by stepSimulation() here, so
//      no extra GPU readback (zero hitch).
// ---------------------------------------------------------------------------
void App::updateAutoFit()
{
    if (!m_autoFit || !m_simulation || !m_simRunning) return;

    unsigned int now = SDL_GetTicks();
    float dt = (m_autoFitLastTicks == 0) ? (1.0f/60.0f)
                                         : (now - m_autoFitLastTicks) * 0.001f;
    m_autoFitLastTicks = now;
    if (dt < 0.0005f) dt = 0.0005f;
    if (dt > 0.05f)   dt = 0.05f;

    if (m_currentFrame - m_autoFitLastFrame >= m_autoFitInterval) {
        m_autoFitLastFrame = m_currentFrame;
        const size_t n = m_positions.size();
        if (n > 0) {
            std::vector<float> rr;
            rr.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                const cl_float4& p = m_positions[i];
                if (p.w == 0.0f) continue;
                rr.push_back(sqrtf(p.x*p.x + p.y*p.y + p.z*p.z));
            }
            if (!rr.empty()) {
                size_t k = (size_t)(0.90 * (double)(rr.size() - 1));
                std::nth_element(rr.begin(), rr.begin() + k, rr.end());
                m_autoFitRadius = rr[k];
                const float tanH = tanf(60.0f * 3.14159265f / 360.0f); // tan(FOV/2), FOV=60°
                float fill = (m_autoFitFill > 0.05f) ? m_autoFitFill : 0.05f;
                if (m_autoFitRadius > 0.0f)
                    m_autoFitTargetRaw = m_autoFitRadius / (fill * tanH);
            }
        }
    }

    if (m_autoFitTargetRaw <= 0.0f) return;

    if (m_autoFitTargetDist <= 0.0f) {
        m_autoFitTargetDist = m_autoFitTargetRaw;   // init : pas de swoop depuis 0
    } else {
        float aT = 1.0f - expf(-dt / (0.5f * m_autoFitSmoothTime));
        m_autoFitTargetDist += (m_autoFitTargetRaw - m_autoFitTargetDist) * aT;
    }

    float cur  = (float)m_camera->getDistance();
    float next = smoothDamp(cur, m_autoFitTargetDist, m_autoFitDistVel,
                            m_autoFitSmoothTime, dt);
    m_camera->setDistance((double)next);
}

void App::uploadParticlesToGPU(bool subData)
{
    // Soft Halo: density scheduler (light; the heavy work runs on a background task).
    if (m_softHalo) computeLocalDensity();

    // INTEROP path: the vertex buffer is built on the GPU directly into the
    // mapped VBO (no readback->rebuild->CPU upload). CPU fallback if unavailable.
    if (uploadParticlesInterop()) return;

    // CPU fallback: it will reallocate the VBO (glBufferData) -> unregister the interop
    // (otherwise the CUDA registration would point to stale storage).
    if (m_interopRes) { interopUnregister(m_interopRes); m_interopRes = nullptr; }
    m_interopOk = false; m_interopVboCount = 0;

    m_particleData.clear();
    m_particleData.reserve(m_positions.size() * 6);
    size_t count = 0;
    m_velMaxForColor = 1.0f;

    if (m_colorByVelocity && !m_rotVelocities.empty()) {
        std::vector<float> speeds;
        speeds.reserve(m_positions.size());
        for (size_t i = 0; i < m_rotVelocities.size() && i < m_positions.size(); i++) {
            if (m_positions[i].w <= 0.0f) continue;
            const cl_float4& v = m_rotVelocities[i];
            float spd = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
            speeds.push_back(spd);
        }
        if (!speeds.empty()) {
            size_t p98 = (size_t)(speeds.size() * 0.98f);
            std::nth_element(speeds.begin(), speeds.begin() + p98, speeds.end());
            m_velMaxForColor = speeds[p98] > 1e-6f ? speeds[p98] : 1.0f;
        }
    }

    for (size_t i = 0; i < m_positions.size(); i++)
    {
        const cl_float4& p = m_positions[i];
        if (!m_showNegaMass && p.w < 0.0f) continue;
        m_particleData.push_back(p.x);
        m_particleData.push_back(p.y);
        m_particleData.push_back(p.z);
        if (m_colorByVelocity && !m_rotVelocities.empty() && i < m_rotVelocities.size()) {
            const cl_float4& v = m_rotVelocities[i];
            float spd = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
            m_particleData.push_back(p.w > 0.0f ? spd : -spd);
        } else {
            m_particleData.push_back(p.w);
        }
        m_particleData.push_back((float)i);
        // 6th float: SPH smoothing length (Soft Halo); 1.0 otherwise. The SIGN encodes
        // the species (negative = negative mass) -> robust detection in the shaders.
        float sz6 = (m_softHalo && i < m_smoothLength.size()) ? m_smoothLength[i] : 1.0f;
        m_particleData.push_back((p.w < 0.0f) ? -sz6 : sz6);
        count++;
    }
    m_particleCount = count;
    m_renderer->uploadParticles(m_particleData, m_particleCount, subData);
}

// 98th percentile of |v| over the positive particles, from the host copy.
// Used to scale the velocity palette; refreshed a few times per second only.
static float computeVelMaxP98(const std::vector<cl_float4>& pos,
                              const std::vector<cl_float4>& vel)
{
    const size_t n = (pos.size() < vel.size()) ? pos.size() : vel.size();
    std::vector<float> speeds;
    speeds.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (pos[i].w <= 0.0f) continue;
        const cl_float4& v = vel[i];
        speeds.push_back(sqrtf(v.x*v.x + v.y*v.y + v.z*v.z));
    }
    if (speeds.empty()) return 1.0f;
    size_t k = (size_t)(speeds.size() * 0.98f);
    if (k >= speeds.size()) k = speeds.size() - 1;
    std::nth_element(speeds.begin(), speeds.begin() + k, speeds.end());
    return speeds[k] > 1e-6f ? speeds[k] : 1.0f;
}

// Builds the vertex buffer on the GPU (CUDA kernel -> mapped VBO), from the
// the sim's device buffers. Avoids the CPU round-trip. Returns false (CPU fallback) if:
// interop disabled/failed, or no sim.
bool App::uploadParticlesInterop()
{
    if (!m_useInterop || !m_simulation) return false;
    // Replay / scrub : les positions viennent du buffer hôte (m_positions), pas
    // du device -> il faut le chemin CPU, sinon on réaffiche la frame live.
    if (m_replayMode) return false;
    int n = m_simulation->getParticleCount();
    if (n <= 0) return false;
    const void* dpos = m_simulation->getDevicePositions();
    if (!dpos) return false;

    // (Re)allocates + (re)registers the VBO when the particle count changes.
    if (!m_interopOk || (size_t)n != m_interopVboCount) {
        if (m_interopRes) { interopUnregister(m_interopRes); m_interopRes = nullptr; }
        m_renderer->reserveVBO((size_t)n);          // allocate the storage (then register)
        m_interopOk = interopRegister(&m_interopRes, m_renderer->getVBO());
        if (!m_interopOk) { m_interopVboCount = 0; return false; }
        m_interopVboCount = (size_t)n;
        if (m_dSmooth) { interopFree(m_dSmooth); m_dSmooth = nullptr; }
        m_dSmooth = interopAllocFloats(n); m_dSmoothCount = m_dSmooth ? n : 0;
        m_smoothDirty = true;
        m_velMaxTick = 0;                           // new particle count -> refresh scale
    }

    // SPH smoothing length -> device (only when it changed).
    const void* dsm = nullptr;
    if (m_softHalo && m_dSmooth && (int)m_smoothLength.size() == n) {
        if (m_smoothDirty) { interopUploadFloats(m_dSmooth, m_smoothLength.data(), n); m_smoothDirty = false; }
        dsm = m_dSmooth;
    }

    // Colour-by-velocity: the kernel writes signed |v| (same encoding as the CPU
    // path). The palette scale comes from the host copy, refreshed every 10th
    // interop upload (~6 Hz at 60 fps) instead of every frame.
    const void* dvel = nullptr;
    int colorByVel = 0;
    if (m_colorByVelocity) {
        dvel = m_simulation->getDeviceVelocities();
        colorByVel = (dvel != nullptr) ? 1 : 0;
        if (colorByVel && m_velMaxTick <= 0) {
            if (!m_rotVelocities.empty() && !m_positions.empty()) {
                m_velMaxForColor = computeVelMaxP98(m_positions, m_rotVelocities);
                m_velMaxTick = 10;
            }
        }
        if (colorByVel && m_velMaxTick > 0) --m_velMaxTick;
    }

    bool ok = interopFillVertices(m_interopRes, dpos, dvel, dsm, n, colorByVel, m_softHalo ? 1 : 0);
    if (!ok) { m_interopOk = false; m_interopVboCount = 0; return false; }
    m_renderer->setDrawCount((size_t)n);            // draws all n (the shader masks the neg.)
    m_particleCount = (size_t)n;
    return true;
}

// SCHEDULER (each frame, light): adopts the ready result / relaunches the worker. The
// heavy work (64³ grid) runs on a background task -> no hitch at large N.
void App::computeLocalDensity()
{
    if (m_densReady.load()) {
        if (m_densThread.joinable()) m_densThread.join();
        m_density         = std::move(m_densityBack);
        m_smoothLength    = std::move(m_smoothLengthBack);
        m_densMaxForColor = m_densMaxBack;
        m_densReady = false; m_densComputing = false;
        m_densLastLaunchMs = SDL_GetTicks64();
        m_smoothDirty = true;   // re-upload smoothLength to the interop device buffer
    }
    if (!m_densComputing.load() && !m_positions.empty() &&
        (SDL_GetTicks64() - m_densLastLaunchMs) >= (uint64_t)m_densIntervalMs) {
        m_densLastLaunchMs = SDL_GetTicks64();
        m_densPosCopy = m_positions;
        m_densComputing = true; m_densReady = false;
        m_densThread = std::thread(&App::computeLocalDensityWorker, this);
    }
}

// WORKER (background thread): 3D density (64³ grid per species) -> SPH smoothing
// adaptive length h ∝ (dRef/d)^(1/3). Writes the BACK buffers (swapped by the scheduler).
void App::computeLocalDensityWorker()
{
    const std::vector<cl_float4>& POS = m_densPosCopy;
    const size_t n = POS.size();
    m_densityBack.assign(n, 0.0f);
    m_smoothLengthBack.assign(n, 1.0f);
    m_densMaxBack = 1.0f;
    if (n == 0) { m_densReady = true; return; }

    // Robust box = mean ± 3σ (ignores ejecta that would inflate the grid).
    double m0=0,m1=0,m2=0;
    for (size_t i=0;i<n;i++){ m0+=POS[i].x; m1+=POS[i].y; m2+=POS[i].z; }
    m0/=(double)n; m1/=(double)n; m2/=(double)n;
    double v0=0,v1=0,v2=0;
    for (size_t i=0;i<n;i++){ double dx=POS[i].x-m0,dy=POS[i].y-m1,dz=POS[i].z-m2;
        v0+=dx*dx; v1+=dy*dy; v2+=dz*dz; }
    float s0=(float)sqrt(v0/(double)n), s1=(float)sqrt(v1/(double)n), s2=(float)sqrt(v2/(double)n);
    const float K=3.0f;
    float mn[3]={(float)m0-K*s0,(float)m1-K*s1,(float)m2-K*s2};
    float mx[3]={(float)m0+K*s0,(float)m1+K*s1,(float)m2+K*s2};
    float ext[3]={mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2]};
    for (int k=0;k<3;k++) if (ext[k]<1e-6f) ext[k]=1e-6f;

    const int NG=64, NG3=NG*NG*NG;
    auto clampi=[&](int v){ return v<0?0:(v>=NG?NG-1:v); };
    std::vector<float> rhoP(NG3,0.0f), rhoN(NG3,0.0f);
    for (size_t i=0;i<n;i++){ const cl_float4&p=POS[i];
        int cx=clampi((int)((p.x-mn[0])/ext[0]*NG));
        int cy=clampi((int)((p.y-mn[1])/ext[1]*NG));
        int cz=clampi((int)((p.z-mn[2])/ext[2]*NG));
        ((p.w>=0.0f)?rhoP:rhoN)[(cz*NG+cy)*NG+cx]+=1.0f; }
    std::vector<float> tmp(NG3);
    auto smooth=[&](std::vector<float>&g){ for(int pass=0;pass<3;pass++){
        for(int z=0;z<NG;z++)for(int y=0;y<NG;y++){int b=(z*NG+y)*NG;
            for(int x=0;x<NG;x++){int xm=x>0?x-1:x,xp=x<NG-1?x+1:x; tmp[b+x]=0.25f*(g[b+xm]+2.0f*g[b+x]+g[b+xp]);} }
        for(int z=0;z<NG;z++)for(int y=0;y<NG;y++)for(int x=0;x<NG;x++){int ym=y>0?y-1:y,yp=y<NG-1?y+1:y;
            g[(z*NG+y)*NG+x]=0.25f*(tmp[(z*NG+ym)*NG+x]+2.0f*tmp[(z*NG+y)*NG+x]+tmp[(z*NG+yp)*NG+x]);}
        for(int z=0;z<NG;z++)for(int y=0;y<NG;y++)for(int x=0;x<NG;x++){int zm=z>0?z-1:z,zp=z<NG-1?z+1:z;
            tmp[(z*NG+y)*NG+x]=0.25f*(g[(zm*NG+y)*NG+x]+2.0f*g[(z*NG+y)*NG+x]+g[(zp*NG+y)*NG+x]);}
        g.swap(tmp);} };
    smooth(rhoP); smooth(rhoN);
    auto sample=[&](const std::vector<float>&g,float px,float py,float pz)->float{
        float gx=(px-mn[0])/ext[0]*NG-0.5f, gy=(py-mn[1])/ext[1]*NG-0.5f, gz=(pz-mn[2])/ext[2]*NG-0.5f;
        int x0=(int)floorf(gx),y0=(int)floorf(gy),z0=(int)floorf(gz);
        float fx=gx-x0,fy=gy-y0,fz=gz-z0;
        int X0=clampi(x0),Y0=clampi(y0),Z0=clampi(z0),X1=clampi(x0+1),Y1=clampi(y0+1),Z1=clampi(z0+1);
        auto V=[&](int xi,int yi,int zi){ return g[(zi*NG+yi)*NG+xi]; };
        float c00=V(X0,Y0,Z0)*(1-fx)+V(X1,Y0,Z0)*fx, c10=V(X0,Y1,Z0)*(1-fx)+V(X1,Y1,Z0)*fx;
        float c01=V(X0,Y0,Z1)*(1-fx)+V(X1,Y0,Z1)*fx, c11=V(X0,Y1,Z1)*(1-fx)+V(X1,Y1,Z1)*fx;
        float c0=c00*(1-fy)+c10*fy, c1=c01*(1-fy)+c11*fy; return c0*(1-fz)+c1*fz; };
    std::vector<float> rawD(n,0.0f), vals; vals.reserve(n);
    for (size_t i=0;i<n;i++){ const cl_float4&p=POS[i];
        float d=sample((p.w>=0.0f)?rhoP:rhoN, p.x,p.y,p.z);
        rawD[i]=d; m_densityBack[i]=logf(1.0f+d); vals.push_back(m_densityBack[i]); }
    { std::vector<float> dd; dd.reserve(n);
      for (size_t i=0;i<n;i++) if (POS[i].w>=0.0f) dd.push_back(rawD[i]);
      float dRef=1.0f;
      if (!dd.empty()){ size_t mid=dd.size()/2; std::nth_element(dd.begin(),dd.begin()+mid,dd.end()); dRef=dd[mid]>0.2f?dd[mid]:1.0f; }
      for (size_t i=0;i<n;i++){ float di=rawD[i]>0.15f?rawD[i]:0.15f;
          float h=2.6f*cbrtf(dRef/di); m_smoothLengthBack[i]=h<0.35f?0.35f:(h>70.0f?70.0f:h); } }
    if (!vals.empty()){ size_t p99=(size_t)(vals.size()*0.99f);
        std::nth_element(vals.begin(),vals.begin()+p99,vals.end());
        m_densMaxBack=vals[p99]>1e-4f?vals[p99]:1.0f; }

    // Frees the large locals BEFORE signaling "ready" (otherwise the scheduler join
    // would wait for their destruction -> hitch). Same lesson as the FoF worker.
    std::vector<float>().swap(rawD);
    std::vector<float>().swap(vals);
    std::vector<float>().swap(rhoP);
    std::vector<float>().swap(rhoN);
    std::vector<float>().swap(tmp);
    m_densReady = true;
}

void App::startSimulation()
{
    if (!m_simulation) return;
    m_isGenerating = true;

    GalaxyParams params;
    params.radius = m_radius;
    params.scaleUnit = m_scaleUnit;
    params.timeStep = m_timeStep;
    params.numNegaStars = m_numNegaStars;
    params.numPosiStars = m_numPosiStars;
    params.negaDensity = m_negaDensity;
    params.totalSolarMass = m_totalSolarMass;
    params.blackHoleMass = m_blackHoleMass;
    params.rotSpeed = m_rotSpeed;
    params.initSpeed = m_initSpeed;
    params.grav = m_gravConstant * 1.0e10f;

    GalaxyGenerator gen(params);
    gen.generate(m_positions, m_velocities);

    int totalCount = gen.getTotalCount();
    m_simulation->setParameters(totalCount, m_timeStep, m_softening, m_gravConstant * 1.0e10f);
    m_simulation->setPhysicsModel(m_eqSelectedModel);
    float pitchTan = 1.0f / tanf(m_spPitchAngle * 3.14159265f / 180.0f); // cot(pitch) = coeff de ln(r) d'une spirale log
    m_simulation->setSpiralParams(m_densityWavesEnabled ? m_spAmplitude : 0.0f, m_spPatternSpeed, pitchTan, (unsigned int)m_spArms);
    m_simulation->uploadInitialData(m_positions, m_velocities);
    // Copie hôte des vitesses : l'échelle de couleur est juste dès la première
    // frame, sans attendre le premier pas de simulation.
    m_rotVelocities = m_velocities;
    m_velMaxTick    = 0;

    uploadParticlesToGPU();

    if (m_simulation->isGLSharing())
        m_needReshare = true;

    m_currentFrame     = 0;
    m_lastRefreshFrame = -9999;

    // Reset analytics histories
    m_energyHead = 0;  m_energyCount = 0;
    m_baryHead   = 0;  m_baryCount   = 0;
    m_fourierHead = 0; m_fourierCount = 0;
    m_consHead   = 0;  m_consCount   = 0;
    m_escHead    = 0;  m_escCount    = 0;
    m_prevNegVel = 0.0f; m_prevEscTime = 0.0f;
    m_corrDone   = false; m_corrXi.clear();

    // Clear particle markers (indices no longer valid after regeneration)
    m_markers.clear();
    m_selectedMarker = -1;
    m_markerPathEnabled.clear();
    m_markerPaths.clear();

    m_simRunning   = true;
    m_simPaused    = false;
    m_isGenerating = false;
}

void App::resetSimulation()
{
    // Abort any running FoF worker and wipe its catalog (stale after reset)
    m_fofCancel = true;
    if (m_fofThread.joinable()) m_fofThread.join();
    if (m_densThread.joinable()) m_densThread.join();   // density worker too
    m_densComputing = false; m_densReady = false; m_smoothLength.clear();
    m_fofClusters.clear();
    m_fofResult.clear();
    m_fofDone = false;
    m_fofComputing = false;
    m_fofResultReady = false;
    m_fofCancel = false;
    m_fofSelected = -1;
    m_fofLive = false;

    // Replay buffer is invalid after regeneration (particle order changes)
    m_replayFrames.clear(); m_replayTimes.clear();
    m_replayMode = false; m_replayPlaying = false; m_replayCursor = 0;
    m_replayLastRecFrame = -1000000;

    m_simRunning = false;
    m_simPaused = true;
    m_currentFrame = 0;
    m_simWallSec = 0.0;          // reset real run-time counter
    startSimulation();
    m_simPaused = true;
}

// La copie hôte ne sert qu'aux consommateurs CPU. Quand l'interop construit le
// VBO depuis le device, elle n'est nécessaire que pour les marqueurs, le clic,
// le replay et le soft halo ; sinon 10 Hz suffisent (échelle de couleur, auto-fit,
// exports). En repli CPU, elle redevient obligatoire à chaque frame.
void App::readHostParticlesIfNeeded()
{
    if (!m_simulation) return;
    const bool interopRenders = m_useInterop && m_interopOk && !m_replayMode;
    const bool countChanged   = m_interopVboCount != (size_t)m_simulation->getParticleCount();
    const bool forced = !interopRenders || countChanged || !m_markers.empty() ||
                        m_markerPickPending || m_replayRecord || m_softHalo ||
                        m_showVelocityField;
    if (!forced && m_hostReadTick > 0) { --m_hostReadTick; return; }

    m_simulation->readPositions(m_positions);
    if (m_colorByVelocity) m_simulation->readVelocities(m_rotVelocities);
    m_hostReadTick = 9;   // une copie tous les 10 pas, pas plus
}

void App::stepSimulation()
{
    if (!m_simulation || !m_simRunning || m_simPaused) return;

    if (m_simSpeed >= 1.0f)
    {
        int nSteps = (int)(m_simSpeed + 0.5f);
        if (nSteps < 1) nSteps = 1;
        m_simulation->setTimestep(m_timeStep);
        for (int s = 0; s < nSteps; s++) m_simulation->step();
        m_currentFrame += nSteps;
        readHostParticlesIfNeeded();
    }
    else
    {
        // SLOW: one step EVERY frame with a reduced dt (x Sim Speed) -> smooth slow-motion.
        m_simulation->setTimestep(m_timeStep * m_simSpeed);
        m_simulation->step();
        readHostParticlesIfNeeded();
        m_currentFrame++;
    }
    uploadParticlesToGPU();

    // Replay timeline: snapshot the positions every N frames
    if (m_replayRecord && (m_currentFrame - m_replayLastRecFrame) >= m_replayInterval) {
        m_replayLastRecFrame = m_currentFrame;
        replayRecordSnapshot();
    }

    bool anyAnalytics = m_showRotCurve || m_showDensityProfile ||
                        m_showPhaseSpace || m_showEnergyPlot || m_showDispersion ||
                        m_showBarycenters || m_showFourier || m_showDarkMatter ||
                        m_showConservation || m_showEscape || m_showCorrelation;
    if (m_rotAutoRefresh && anyAnalytics &&
        (m_currentFrame - m_lastRefreshFrame >= m_rotRefreshInterval))
    {
        m_lastRefreshFrame = m_currentFrame;
        refreshAnalytics();
    }

    updateMarkerPaths();
}

void App::refreshAnalytics()
{
    if (!m_simulation) return;
    m_simulation->readPositions(m_positions);
    m_simulation->readVelocities(m_rotVelocities);

    // Auto-scale color by velocity range
    if (m_colorByVelocity && m_velAutoScale && !m_rotVelocities.empty())
    {
        std::vector<float> speeds;
        speeds.reserve(m_rotVelocities.size());
        for (const auto& v : m_rotVelocities)
            speeds.push_back(sqrtf(v.x*v.x + v.y*v.y + v.z*v.z));
        size_t p98 = (size_t)(speeds.size() * 0.98f);
        std::nth_element(speeds.begin(), speeds.begin() + p98, speeds.end());
        m_velMaxForColor = speeds[p98] > 1e-6f ? speeds[p98] : 1.0f;
        uploadParticlesToGPU();
    }

    float ke = 0.0f;
    size_t nv = m_rotVelocities.size();
    size_t np = m_positions.size();
    size_t n  = (nv < np) ? nv : np;
    for (size_t i = 0; i < n; i++)
    {
        if (m_positions[i].w <= 0.0f) continue;
        const cl_float4& v = m_rotVelocities[i];
        ke += 0.5f * m_positions[i].w * (v.x*v.x + v.y*v.y + v.z*v.z);
    }
    m_energyHistory[m_energyHead] = ke;
    m_energyTimes  [m_energyHead] = (double)m_currentFrame * (double)m_timeStep;
    m_energyHead = (m_energyHead + 1) % ENERGY_HISTORY;
    if (m_energyCount < ENERGY_HISTORY) m_energyCount++;

    {
        double px=0,py=0,pz=0,pm=0, nx=0,ny=0,nz=0,nm=0;
        double pvx=0,pvy=0,pvz=0,    nvx=0,nvy=0,nvz=0;
        size_t nvLoc = (m_rotVelocities.size() < np) ? m_rotVelocities.size() : np;
        for (size_t i = 0; i < np; i++) {
            const cl_float4& p = m_positions[i];
            if (p.w > 0.f) {
                px+=p.x*p.w; py+=p.y*p.w; pz+=p.z*p.w; pm+=p.w;
                if (i < nvLoc) {
                    const cl_float4& v = m_rotVelocities[i];
                    pvx+=v.x*p.w; pvy+=v.y*p.w; pvz+=v.z*p.w;
                }
            } else if (p.w < 0.f) {
                double am=fabsf(p.w);
                nx+=p.x*am; ny+=p.y*am; nz+=p.z*am; nm+=am;
                if (i < nvLoc) {
                    const cl_float4& v = m_rotVelocities[i];
                    nvx+=v.x*am; nvy+=v.y*am; nvz+=v.z*am;
                }
            }
        }
        if (pm > 0) { px/=pm; py/=pm; pz/=pm; pvx/=pm; pvy/=pm; pvz/=pm; }
        if (nm > 0) { nx/=nm; ny/=nm; nz/=nm; nvx/=nm; nvy/=nm; nvz/=nm; }
        m_baryPosX = (float)px; m_baryPosY = (float)py; m_baryPosZ = (float)pz;
        m_baryNegX = (float)nx; m_baryNegY = (float)ny; m_baryNegZ = (float)nz;
        m_baryPosVx = (float)pvx; m_baryPosVy = (float)pvy; m_baryPosVz = (float)pvz;
        m_baryNegVx = (float)nvx; m_baryNegVy = (float)nvy; m_baryNegVz = (float)nvz;
        float dist = sqrtf((float)((px-nx)*(px-nx)+(py-ny)*(py-ny)+(pz-nz)*(pz-nz))) * m_lyPerUnit;
        float posMag = sqrtf((float)(px*px + py*py + pz*pz)) * m_lyPerUnit;
        float negMag = sqrtf((float)(nx*nx + ny*ny + nz*nz)) * m_lyPerUnit;
        m_baryDist  [m_baryHead] = dist;
        m_baryPosMag[m_baryHead] = posMag;
        m_baryNegMag[m_baryHead] = negMag;
        m_baryTimes [m_baryHead] = (float)m_currentFrame * m_timeStep;
        m_baryHead = (m_baryHead + 1) % BARY_HISTORY;
        if (m_baryCount < BARY_HISTORY) m_baryCount++;
    }

    {
        double re[4]={}, im_[4]={};
        int cnt = 0;
        for (size_t i = 0; i < np; i++) {
            const cl_float4& p = m_positions[i];
            if (p.w <= 0.f) continue;
            // Recenter on positive COM: post-merger the galaxy can drift far from
            // origin, and m=1 mode would otherwise spike to ~1 purely from the offset.
            float dx = p.x - m_baryPosX, dy = p.y - m_baryPosY;
            float r = sqrtf(dx*dx + dy*dy);
            if (r < 5.f) continue;
            float theta = atan2f(dy, dx);
            for (int m = 0; m < 4; m++) {
                re[m]  += cos((m+1) * theta);
                im_[m] += sin((m+1) * theta);
            }
            cnt++;
        }
        for (int m = 0; m < 4; m++)
            m_fourierA[m][m_fourierHead] = (cnt>0) ? sqrtf((float)(re[m]*re[m]+im_[m]*im_[m]))/(float)cnt : 0.f;
        m_fourierTimes[m_fourierHead] = (float)m_currentFrame * m_timeStep;
        m_fourierHead = (m_fourierHead + 1) % FOURIER_HISTORY;
        if (m_fourierCount < FOURIER_HISTORY) m_fourierCount++;
    }

    // ── Point 1 : Angular momentum Lz + estimated PE ─────────────────────────
    {
        double Lz = 0.0, totalM = 0.0;
        // PE estimate via random sample (N_s particles)
        const int N_S = std::min((int)n, 800);
        double peSum = 0.0;
        int step = (n > (size_t)N_S) ? (int)(n / N_S) : 1;

        for (size_t i = 0; i < n; i++)
        {
            const cl_float4& pos = m_positions[i];
            if (fabsf(pos.w) < 1e-10f) continue;
            const cl_float4& vel = m_rotVelocities[i];
            // Recenter on positive COM and subtract bulk velocity — Lz is
            // frame-dependent, the systemic drift would dominate after a kick.
            float dx = pos.x - m_baryPosX, dy = pos.y - m_baryPosY;
            float dvx = vel.x - m_baryPosVx, dvy = vel.y - m_baryPosVy;
            Lz     += pos.w * (dx * dvy - dy * dvx);
            totalM += fabsf(pos.w);
        }
        float Lz_per_mass = (totalM > 0) ? (float)(Lz / totalM) : 0.0f;

        // Sampled PE (Newtonian, signed masses as in Janus)
        float G_eff = m_gravConstant * 1.0e10f;
        for (int si = 0; si < N_S; si++)
        {
            size_t ii = (size_t)(si * step) % n;
            for (int sj = si + 1; sj < N_S; sj++)
            {
                size_t jj = (size_t)(sj * step) % n;
                float dx = m_positions[ii].x - m_positions[jj].x;
                float dy = m_positions[ii].y - m_positions[jj].y;
                float dz = m_positions[ii].z - m_positions[jj].z;
                float r  = sqrtf(dx*dx + dy*dy + dz*dz + m_softening);
                // Janus PE: sign(m_i) * G * m_j / r
                float sign_i = (m_positions[ii].w >= 0) ? 1.0f : -1.0f;
                peSum -= sign_i * G_eff * m_positions[jj].w / r;
            }
        }
        // Scale sampled PE to full system
        float scaleFactor = (N_S > 1) ? (float)(n * n) / (N_S * (N_S - 1)) : 1.0f;
        float pe_est = (float)(peSum * scaleFactor);

        m_angMomHistory[m_consHead] = Lz_per_mass;
        m_totEHistory  [m_consHead] = ke + pe_est;
        m_consTimesH   [m_consHead] = (double)m_currentFrame * (double)m_timeStep;
        m_consHead = (m_consHead + 1) % CONS_HISTORY;
        if (m_consCount < CONS_HISTORY) m_consCount++;
    }

    // ── Point 3 : Negative mass escape velocity ───────────────────────────────
    {
        double sumVneg = 0.0;
        int    cntNeg  = 0;
        for (size_t i = 0; i < n; i++)
        {
            if (m_positions[i].w >= 0.0f) continue;
            const cl_float4& v = m_rotVelocities[i];
            // Subtract bulk velocity of the negative halo — otherwise a systemic
            // drift (post-merger recoil of the halo) inflates the "escape" metric.
            float dvx = v.x - m_baryNegVx;
            float dvy = v.y - m_baryNegVy;
            float dvz = v.z - m_baryNegVz;
            sumVneg += sqrtf(dvx*dvx + dvy*dvy + dvz*dvz);
            cntNeg++;
        }
        float avgVneg = (cntNeg > 0) ? (float)(sumVneg / cntNeg) : 0.0f;
        float tNow    = (float)m_currentFrame * m_timeStep;

        float acc = 0.0f;
        if (m_prevEscTime > 0.0f && (tNow - m_prevEscTime) > 1e-6f)
            acc = (avgVneg - m_prevNegVel) / (tNow - m_prevEscTime);

        m_negVelHistory[m_escHead] = avgVneg;
        m_negAccHistory[m_escHead] = acc;
        m_escTimes     [m_escHead] = tNow;
        m_escHead = (m_escHead + 1) % ESC_HISTORY;
        if (m_escCount < ESC_HISTORY) m_escCount++;

        m_prevNegVel  = avgVneg;
        m_prevEscTime = tNow;
    }

    // ── Two-point correlation ξ(r) — computed only when window is open ────────
    if (m_showCorrelation && !m_positions.empty())
    {
        const int RBINS = 50;
        const int NS = 800;
        std::vector<cl_float4> posP, posN;
        for (size_t i = 0; i < n; i++) {
            if (m_positions[i].w > 0 && (int)posP.size() < NS) posP.push_back(m_positions[i]);
            if (m_positions[i].w < 0 && (int)posN.size() < NS) posN.push_back(m_positions[i]);
        }

        float maxNegR = 0, maxPosR = 0;
        for (const auto& q : posN) { float r=sqrtf(q.x*q.x+q.y*q.y+q.z*q.z); if(r>maxNegR) maxNegR=r; }
        for (const auto& p : posP) { float r=sqrtf(p.x*p.x+p.y*p.y+p.z*p.z); if(r>maxPosR) maxPosR=r; }
        float rMaxSim = (maxNegR + maxPosR) * 1.05f;
        if (rMaxSim < 1.0f) rMaxSim = m_radius * 4.0f;
        float rMax = rMaxSim * m_lyPerUnit;
        m_corrRMax = rMax;

        float dR = rMax / RBINS;
        std::vector<float> DD(RBINS, 0), RR(RBINS, 0);
        for (const auto& p : posP)
            for (const auto& q : posN) {
                float dx=p.x-q.x, dy=p.y-q.y, dz=p.z-q.z;
                float r = sqrtf(dx*dx+dy*dy+dz*dz) * m_lyPerUnit;
                int b=(int)(r/dR); if(b>=0&&b<RBINS) DD[b]++;
            }
        float vol = 4.0f/3.0f * 3.14159f * rMax*rMax*rMax;
        float nP=(float)posP.size(), nN=(float)posN.size();
        for (int b=0;b<RBINS;b++) {
            float r0=b*dR, r1=r0+dR;
            float shellVol = 4.0f/3.0f*3.14159f*(r1*r1*r1-r0*r0*r0);
            RR[b] = nP * nN * shellVol / vol;
            if (RR[b] < 0.01f) RR[b] = 0.01f;
        }
        m_corrXi.resize(RBINS);
        for (int b=0;b<RBINS;b++)
            m_corrXi[b] = DD[b] / RR[b] - 1.0f;
        m_corrDone = true;
    }
}

void App::saveSimState()
{
    if (!m_simulation || !m_simRunning) return;
    std::vector<cl_float4> pos, vel;
    m_simulation->readPositions(pos);
    m_simulation->readVelocities(vel);
    std::ofstream f(m_stateFilePath, std::ios::binary);
    if (!f.is_open()) { std::cerr << "Cannot write state: " << m_stateFilePath << "\n"; return; }

    uint32_t magic      = 0x4A475354u;
    uint32_t version    = 2u;  // ← v2 : inclut les marqueurs
    uint32_t nParticles = (uint32_t)pos.size();
    int32_t  appFrame   = m_currentFrame;
    float    simTime    = m_simulation->getSimTime();
    float    timeStep   = m_timeStep;

    f.write(reinterpret_cast<const char*>(&magic),      4);
    f.write(reinterpret_cast<const char*>(&version),    4);
    f.write(reinterpret_cast<const char*>(&nParticles), 4);
    f.write(reinterpret_cast<const char*>(&appFrame),   4);
    f.write(reinterpret_cast<const char*>(&simTime),    4);
    f.write(reinterpret_cast<const char*>(&timeStep),   4);
    f.write(reinterpret_cast<const char*>(pos.data()),  pos.size() * sizeof(cl_float4));
    f.write(reinterpret_cast<const char*>(vel.data()),  vel.size() * sizeof(cl_float4));

    // ── Marqueurs (v2) ─────────────────────────────────────────────────
    uint32_t nMarkers = (uint32_t)m_markers.size();
    f.write(reinterpret_cast<const char*>(&nMarkers), 4);
    if (nMarkers > 0) {
        f.write(reinterpret_cast<const char*>(m_markers.data()), nMarkers * sizeof(int32_t));
        // Path enabled flags
        std::vector<uint8_t> pathFlags(nMarkers);
        for (size_t i = 0; i < nMarkers; i++)
            pathFlags[i] = (i < m_markerPathEnabled.size() && m_markerPathEnabled[i]) ? 1 : 0;
        f.write(reinterpret_cast<const char*>(pathFlags.data()), nMarkers * sizeof(uint8_t));
        // Paths data
        for (size_t i = 0; i < nMarkers; i++) {
            uint32_t pathLen = 0;
            if (i < m_markerPaths.size() && m_markerPathEnabled[i])
                pathLen = (uint32_t)m_markerPaths[i].size();
            f.write(reinterpret_cast<const char*>(&pathLen), 4);
            if (pathLen > 0) {
                f.write(reinterpret_cast<const char*>(m_markerPaths[i].data()),
                        pathLen * sizeof(cl_float4));
            }
        }
    }
}

void App::saveSimStateTo()
{
#ifdef _WIN32
    // Dossier initial : Bureau de l'utilisateur
    char desktopPath[MAX_PATH] = {};
    const char* userProfile = getenv("USERPROFILE");
    if (userProfile)
        snprintf(desktopPath, MAX_PATH, "%s\\Desktop", userProfile);

    char filename[MAX_PATH] = "jgravity_state.bin";

    OPENFILENAMEA ofn = {};
    ofn.lStructSize   = sizeof(ofn);
    ofn.hwndOwner     = NULL;
    ofn.lpstrFilter   = "Janus Galaxy State (*.bin)\0*.bin\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile     = filename;
    ofn.nMaxFile      = MAX_PATH;
    ofn.lpstrDefExt   = "bin";
    ofn.lpstrTitle    = "Save Simulation State";
    ofn.lpstrInitialDir = desktopPath[0] ? desktopPath : NULL;
    ofn.Flags         = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetSaveFileNameA(&ofn))
    {
        strncpy(m_stateFilePath, filename, sizeof(m_stateFilePath) - 1);
        m_stateFilePath[sizeof(m_stateFilePath) - 1] = '\0';
        saveSimState();
    }
#else
    saveSimState();
#endif
}

void App::loadSimState()
{
    if (!m_simulation) return;
    std::ifstream f(m_stateFilePath, std::ios::binary);
    if (!f.is_open()) { std::cerr << "Cannot read state: " << m_stateFilePath << "\n"; return; }

    uint32_t magic, version, nParticles;
    int32_t  appFrame;
    float    simTime, timeStep;

    f.read(reinterpret_cast<char*>(&magic),      4);
    f.read(reinterpret_cast<char*>(&version),    4);
    f.read(reinterpret_cast<char*>(&nParticles), 4);
    f.read(reinterpret_cast<char*>(&appFrame),   4);
    f.read(reinterpret_cast<char*>(&simTime),    4);
    f.read(reinterpret_cast<char*>(&timeStep),   4);

    if (magic != 0x4A475354u || (version != 1u && version != 2u)) { std::cerr << "Invalid state file\n"; return; }

    std::vector<cl_float4> pos(nParticles), vel(nParticles);
    f.read(reinterpret_cast<char*>(pos.data()), nParticles * sizeof(cl_float4));
    f.read(reinterpret_cast<char*>(vel.data()), nParticles * sizeof(cl_float4));
    if (!f) { std::cerr << "Truncated state file\n"; return; }

    m_positions  = pos;
    m_velocities = vel;

    m_simulation->setParameters((int)nParticles, m_timeStep, m_softening,
                                 m_gravConstant * 1.0e10f);
    m_simulation->setPhysicsModel(m_eqSelectedModel);   // (was missing: otherwise stale model on reload)
    float pitchTan = 1.0f / tanf(m_spPitchAngle * 3.14159265f / 180.0f); // cot(pitch) = coeff de ln(r) d'une spirale log
    m_simulation->setSpiralParams(m_densityWavesEnabled ? m_spAmplitude : 0.0f, m_spPatternSpeed, pitchTan, (unsigned int)m_spArms);
    m_simulation->uploadInitialData(m_positions, m_velocities);
    m_simulation->setSimTime(simTime);
    // Même raison qu'à la génération : état chargé en pause, aucun pas ne
    // rafraîchira les vitesses hôte avant le premier rendu.
    m_rotVelocities = m_velocities;
    m_velMaxTick    = 0;

    uploadParticlesToGPU();
    if (m_simulation->isGLSharing())
        m_needReshare = true;

    m_currentFrame = appFrame;
    m_simRunning   = true;
    m_simPaused    = true;

    // ── Marqueurs (v2 uniquement) ──────────────────────────────────────
    m_markers.clear();
    m_markerPathEnabled.clear();
    m_markerPaths.clear();
    m_selectedMarker = -1;
    if (version == 2u) {
        uint32_t nMarkers = 0;
        f.read(reinterpret_cast<char*>(&nMarkers), 4);
        if (nMarkers > 0 && nMarkers <= 16) {
            m_markers.resize(nMarkers);
            f.read(reinterpret_cast<char*>(m_markers.data()), nMarkers * sizeof(int32_t));
            // Path enabled flags
            std::vector<uint8_t> pathFlags(nMarkers);
            f.read(reinterpret_cast<char*>(pathFlags.data()), nMarkers * sizeof(uint8_t));
            m_markerPathEnabled.resize(nMarkers);
            for (size_t i = 0; i < nMarkers; i++)
                m_markerPathEnabled[i] = (pathFlags[i] != 0);
            // Paths data
            m_markerPaths.resize(nMarkers);
            for (size_t i = 0; i < nMarkers; i++) {
                uint32_t pathLen = 0;
                f.read(reinterpret_cast<char*>(&pathLen), 4);
                if (pathLen > 0 && pathLen < 1000000) {
                    m_markerPaths[i].resize(pathLen);
                    f.read(reinterpret_cast<char*>(m_markerPaths[i].data()),
                           pathLen * sizeof(cl_float4));
                }
            }
        }
    }
}

void App::saveScreenshot()
{
#ifdef _WIN32
    // The native dialog writes the selected path into this buffer.
    char filename[sizeof(m_screenshotPath)] = "jgravity_screenshot.png";
    char desktopPath[MAX_PATH] = {};
    const char* userProfile = getenv("USERPROFILE");
    if (userProfile)
        snprintf(desktopPath, MAX_PATH, "%s\\Desktop", userProfile);

    OPENFILENAMEA ofn = {};
    ofn.lStructSize  = sizeof(ofn);
    ofn.hwndOwner    = NULL;
    ofn.lpstrFilter  = "PNG Image (*.png)\0*.png\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile    = filename;
    ofn.nMaxFile     = sizeof(filename);
    ofn.lpstrDefExt  = "png";
    ofn.lpstrTitle   = "Save Screenshot";
    ofn.lpstrInitialDir = desktopPath[0] ? desktopPath : NULL;
    ofn.Flags        = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (!GetSaveFileNameA(&ofn))
        return;  // user cancelled
#else
    const char* filename = "jgravity_screenshot.png";
#endif

    // Defer the actual pixel capture to the end of the frame, AFTER ImGui has
    // rendered — otherwise overlays (grid, markers, annotations) are missing.
    strncpy(m_screenshotPath, filename, sizeof(m_screenshotPath)-1);
    m_screenshotPath[sizeof(m_screenshotPath)-1] = '\0';
    m_screenshotPending = true;
}

// Capture the viewport region (excludes the Controls panel) once everything —
// particles + ImGui overlays + annotations — is on the back buffer.
void App::captureScreenshotNow()
{
    if (!m_screenshotPending) return;
    m_screenshotPending = false;
    if (m_screenshotPath[0] == '\0') return;

    int w, h;
    SDL_GL_GetDrawableSize(m_window, &w, &h);
    // Crop out the left Controls panel (340 logical px, scaled to drawable px).
    float scaleX = (m_width > 0) ? (float)w / (float)m_width : 1.0f;
    int panelPx = (int)(340.0f * scaleX + 0.5f);
    if (panelPx < 0) panelPx = 0; if (panelPx >= w) panelPx = 0;
    int cw = w - panelPx;
    if (cw <= 0) { cw = w; panelPx = 0; }

    std::vector<unsigned char> pixels((size_t)cw * h * 3);
    glReadPixels(panelPx, 0, cw, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // Flip vertically (GL origin is bottom-left)
    for (int row = 0; row < h / 2; row++) {
        unsigned char* top = pixels.data() + (size_t)row * cw * 3;
        unsigned char* bot = pixels.data() + (size_t)(h - 1 - row) * cw * 3;
        for (int c = 0; c < cw * 3; c++) std::swap(top[c], bot[c]);
    }
    stbi_write_png(m_screenshotPath, cw, h, 3, pixels.data(), cw * 3);
    m_screenshotPath[0] = '\0';
}

// ─── MP4 video recording (viewport streamed to ffmpeg via a pipe) ────────────
// Locate ffmpeg.exe: next to our exe, in the known D:\SIMULATION folder, or PATH.
std::string App::findFFmpeg()
{
#ifdef _WIN32
    const char* cands[] = {
        "ffmpeg.exe",
        "D:\\SIMULATION\\ffmpeg-master-latest-win64-gpl-shared\\bin\\ffmpeg.exe",
        "ffmpeg\\bin\\ffmpeg.exe",
    };
    for (const char* c : cands) {
        DWORD a = GetFileAttributesA(c);
        if (a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY))
            return std::string(c);
    }
    // Fall back to bare name (resolved via PATH by CreateProcess search)
    return std::string("ffmpeg.exe");
#else
    return std::string("ffmpeg");
#endif
}

bool App::startVideoRecording()
{
#ifdef _WIN32
    if (m_videoRecording) return true;

    // Ask for the output path
    char filename[MAX_PATH] = "jgravity_video.mp4";
    char desktopPath[MAX_PATH] = {};
    const char* up = getenv("USERPROFILE");
    if (up) snprintf(desktopPath, MAX_PATH, "%s\\Desktop", up);
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "MP4 Video (*.mp4)\0*.mp4\0";
    ofn.lpstrFile = filename; ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = "mp4"; ofn.lpstrTitle = "Record MP4";
    ofn.lpstrInitialDir = desktopPath[0] ? desktopPath : NULL;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetSaveFileNameA(&ofn)) return false;
    strncpy(m_videoPath, filename, sizeof(m_videoPath)-1);

    // Lock the viewport size (in drawable px), rounded to even for h264
    int w, h; SDL_GL_GetDrawableSize(m_window, &w, &h);
    float scaleX = (m_width > 0) ? (float)w / (float)m_width : 1.0f;
    int panelPx = (int)(340.0f * scaleX + 0.5f);
    if (panelPx < 0 || panelPx >= w) panelPx = 0;
    m_recW = (w - panelPx) & ~1;
    m_recH = h & ~1;
    if (m_recW <= 0 || m_recH <= 0) return false;

    // Build the ffmpeg command. Reads raw rgb24 from stdin, vflip (GL bottom-up),
    // encodes h264 yuv420p. -crf 18 = high quality.
    std::string ff = findFFmpeg();
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "\"%s\" -y -f rawvideo -pixel_format rgb24 -video_size %dx%d -framerate %d "
        "-i - -vf vflip -c:v libx264 -pix_fmt yuv420p -crf 18 -preset veryfast \"%s\"",
        ff.c_str(), m_recW, m_recH, m_videoFps, m_videoPath);

    // Create a pipe; child reads its stdin from the read end.
    SECURITY_ATTRIBUTES sa = {}; sa.nLength = sizeof(sa); sa.bInheritHandle = TRUE;
    HANDLE rd = NULL, wr = NULL;
    if (!CreatePipe(&rd, &wr, &sa, 1 << 22)) return false;     // 4 MB pipe
    SetHandleInformation(wr, HANDLE_FLAG_INHERIT, 0);          // keep our write end private

    STARTUPINFOA si = {}; si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdInput  = rd;
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi = {};
    BOOL ok = CreateProcessA(NULL, cmd, NULL, NULL, TRUE,
                             CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    CloseHandle(rd);   // child owns the read end now
    if (!ok) { CloseHandle(wr); printf("[Video] ffmpeg launch failed (%lu)\n", GetLastError()); return false; }

    m_ffmpegStdin = wr;
    m_ffmpegProc  = pi.hProcess;
    CloseHandle(pi.hThread);
    m_recBuf.resize((size_t)m_recW * m_recH * 3);
    m_videoRecording = true;
    printf("[Video] recording %dx%d @ %d fps → %s\n", m_recW, m_recH, m_videoFps, m_videoPath);
    return true;
#else
    return false;
#endif
}

void App::captureVideoFrame()
{
#ifdef _WIN32
    if (!m_videoRecording || !m_ffmpegStdin) return;
    int w, h; SDL_GL_GetDrawableSize(m_window, &w, &h);
    float scaleX = (m_width > 0) ? (float)w / (float)m_width : 1.0f;
    int panelPx = (int)(340.0f * scaleX + 0.5f);
    if (panelPx < 0 || panelPx >= w) panelPx = 0;
    // Read exactly the locked recording rectangle (clamp if the window changed)
    int rx = panelPx, ry = 0, rw = m_recW, rh = m_recH;
    if (rx + rw > w) rw = w - rx;
    if (rh > h) rh = h;
    if (rw <= 0 || rh <= 0) return;
    if ((int)m_recBuf.size() < rw * rh * 3) m_recBuf.resize((size_t)rw * rh * 3);
    glReadPixels(rx, ry, rw, rh, GL_RGB, GL_UNSIGNED_BYTE, m_recBuf.data());
    DWORD written = 0;
    WriteFile((HANDLE)m_ffmpegStdin, m_recBuf.data(),
              (DWORD)((size_t)m_recW * m_recH * 3), &written, NULL);
#endif
}

void App::stopVideoRecording()
{
#ifdef _WIN32
    if (!m_videoRecording) return;
    m_videoRecording = false;
    if (m_ffmpegStdin) { CloseHandle((HANDLE)m_ffmpegStdin); m_ffmpegStdin = nullptr; }
    if (m_ffmpegProc) {
        WaitForSingleObject((HANDLE)m_ffmpegProc, 15000);  // let ffmpeg finalize
        CloseHandle((HANDLE)m_ffmpegProc); m_ffmpegProc = nullptr;
    }
    printf("[Video] saved %s\n", m_videoPath);
#endif
}

// ─── Replay timeline : ring buffer of position snapshots ─────────────────────
void App::replayRecordSnapshot()
{
    if (m_positions.empty()) return;
    if (m_replayStream) {
        replayWriteDiskFrame();   // unlimited, persisted
        return;
    }
    m_replayFrames.push_back(m_positions);                 // copy current positions
    m_replayTimes.push_back(m_simulation ? m_simulation->getSimTime() / 100.0f : 0.0f);
    while ((int)m_replayFrames.size() > m_replayMaxFrames) {
        m_replayFrames.erase(m_replayFrames.begin());
        m_replayTimes.erase(m_replayTimes.begin());
    }
}

// Scrub/playback: shows the snapshot at m_replayCursor on the GPU. Called each
// frame while m_replayMode is on (freezes the live simulation).
void App::tickReplay()
{
    int n = replayCount();
    if (n <= 0) return;
    if (m_replayPlaying) {
        static float acc = 0.0f;
        acc += m_replayPlaySpeed;
        while (acc >= 1.0f) { acc -= 1.0f; m_replayCursor++; }
        if (m_replayCursor >= n) { m_replayCursor = n - 1; m_replayPlaying = false; }
    }
    if (m_replayCursor < 0) m_replayCursor = 0;
    if (m_replayCursor >= n) m_replayCursor = n - 1;
    if (m_replayCursor != m_replayShownCursor) {
        replaySetFrame(m_replayCursor);
        m_replayShownCursor = m_replayCursor;
    }
}

// ── Replay persistence: single-file (RAM buffer) + streaming-to-folder ────────
int App::replayCount() const {
    return m_replayStream ? (int)m_replayTimes.size() : (int)m_replayFrames.size();
}

void App::replaySetFrame(int idx) {
    int n = replayCount();
    if (n <= 0) return;
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    if (m_replayStream) { if (!replayReadDiskFrame(idx)) return; }
    else                { m_positions = m_replayFrames[idx]; }
    uploadParticlesToGPU();
}

void App::saveReplay() {
    if (m_replayFrames.empty()) { std::cerr << "[Replay] nothing to save\n"; return; }
    std::ofstream f(m_replayFilePath, std::ios::binary);
    if (!f.is_open()) { std::cerr << "[Replay] cannot write: " << m_replayFilePath << "\n"; return; }
    uint32_t magic = 0x4A525031u;
    uint32_t nSnaps = (uint32_t)m_replayFrames.size();
    uint32_t nParts = (uint32_t)(m_replayFrames.empty() ? 0 : m_replayFrames[0].size());
    f.write(reinterpret_cast<const char*>(&magic),  4);
    f.write(reinterpret_cast<const char*>(&nSnaps), 4);
    f.write(reinterpret_cast<const char*>(&nParts), 4);
    for (uint32_t i = 0; i < nSnaps; i++) {
        float t = (i < m_replayTimes.size()) ? m_replayTimes[i] : 0.0f;
        f.write(reinterpret_cast<const char*>(&t), 4);
        f.write(reinterpret_cast<const char*>(m_replayFrames[i].data()),
                m_replayFrames[i].size() * sizeof(cl_float4));
    }
    printf("[Replay] saved %u snapshots -> %s\n", nSnaps, m_replayFilePath);
}

void App::loadReplay() {
    std::ifstream f(m_replayFilePath, std::ios::binary);
    if (!f.is_open()) { std::cerr << "[Replay] cannot open: " << m_replayFilePath << "\n"; return; }
    uint32_t magic = 0, nSnaps = 0, nParts = 0;
    f.read(reinterpret_cast<char*>(&magic),  4);
    f.read(reinterpret_cast<char*>(&nSnaps), 4);
    f.read(reinterpret_cast<char*>(&nParts), 4);
    if (magic != 0x4A525031u || nParts == 0 || nSnaps == 0) { std::cerr << "[Replay] invalid file\n"; return; }
    m_replayStream = false;
    m_replayFrames.clear(); m_replayTimes.clear();
    for (uint32_t i = 0; i < nSnaps; i++) {
        float t = 0.0f; f.read(reinterpret_cast<char*>(&t), 4);
        std::vector<cl_float4> snap(nParts);
        f.read(reinterpret_cast<char*>(snap.data()), (size_t)nParts * sizeof(cl_float4));
        if (!f) break;
        m_replayFrames.push_back(std::move(snap));
        m_replayTimes.push_back(t);
    }
    m_replayCursor = 0; m_replayShownCursor = -1;
    m_replayMode = !m_replayFrames.empty();
    if (m_replayMode) { m_replayPlaying = false; m_simPaused = true; }
    printf("[Replay] loaded %zu snapshots from %s\n", m_replayFrames.size(), m_replayFilePath);
}

void App::replayWriteDiskFrame() {
    if (m_positions.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(m_replayDir, ec);
    char fname[640];
    snprintf(fname, sizeof(fname), "%s\\snap_%06d.bin", m_replayDir, m_replayDiskCount);
    std::ofstream f(fname, std::ios::binary);
    if (!f.is_open()) { std::cerr << "[Replay] cannot write " << fname << "\n"; return; }
    float    t      = m_simulation ? m_simulation->getSimTime() / 100.0f : 0.0f;
    uint32_t nParts = (uint32_t)m_positions.size();
    f.write(reinterpret_cast<const char*>(&t), 4);
    f.write(reinterpret_cast<const char*>(&nParts), 4);
    f.write(reinterpret_cast<const char*>(m_positions.data()), m_positions.size() * sizeof(cl_float4));
    m_replayTimes.push_back(t);
    m_replayDiskCount++;
}

bool App::replayReadDiskFrame(int idx) {
    if (idx < 0) return false;
    char fname[640];
    snprintf(fname, sizeof(fname), "%s\\snap_%06d.bin", m_replayDir, idx);
    std::ifstream f(fname, std::ios::binary);
    if (!f.is_open()) return false;
    float t = 0.0f; uint32_t nParts = 0;
    f.read(reinterpret_cast<char*>(&t), 4);
    f.read(reinterpret_cast<char*>(&nParts), 4);
    if (!f || nParts == 0 || nParts > 50000000u) return false;
    m_positions.resize(nParts);
    f.read(reinterpret_cast<char*>(m_positions.data()), (size_t)nParts * sizeof(cl_float4));
    return (bool)f;
}

void App::replayScanDiskFolder() {
    m_replayTimes.clear(); m_replayFrames.clear();
    int idx = 0;
    for (;;) {
        char fname[640];
        snprintf(fname, sizeof(fname), "%s\\snap_%06d.bin", m_replayDir, idx);
        std::ifstream f(fname, std::ios::binary);
        if (!f.is_open()) break;
        float t = 0.0f; f.read(reinterpret_cast<char*>(&t), 4);
        if (!f) break;
        m_replayTimes.push_back(t); idx++;
    }
    m_replayDiskCount = idx;
    if (m_replayDiskCount > 0) {
        m_replayStream = true; m_replayCursor = 0; m_replayShownCursor = -1;
        m_replayMode = true; m_replayPlaying = false; m_simPaused = true;
        replaySetFrame(0);
    }
    printf("[Replay] folder scan: %d snapshots in %s\n", m_replayDiskCount, m_replayDir);
}

static bool beginPlotWindow(const char* title, bool* open,
    float& plotW, float& plotH, ImVec2& canvasPos,
    const char* xlabel, const char* ylabel)
{
    ImGui::SetNextWindowSize(ImVec2(600,460), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(title, open, ImGuiWindowFlags_NoCollapse)) { ImGui::End(); return false; }
    canvasPos  = ImGui::GetCursorScreenPos();
    ImVec2 sz  = ImGui::GetContentRegionAvail();
    if (sz.x < 100) sz.x = 100;
    if (sz.y < 100) sz.y = 100;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(canvasPos, ImVec2(canvasPos.x+sz.x, canvasPos.y+sz.y), IM_COL32(8,16,48,255));
    dl->AddRect(canvasPos, ImVec2(canvasPos.x+sz.x, canvasPos.y+sz.y), IM_COL32(80,80,80,255));

    ImVec2 yts = ImGui::CalcTextSize(ylabel);
    const float mL = fmaxf(68.0f, yts.x + 8.0f);
    const float mB=34.0f, mT=10.0f, mR=10.0f;

    plotW = sz.x - mL - mR;
    plotH = sz.y - mB - mT;
    canvasPos.x += mL; canvasPos.y += mT;
    ImU32 ca = IM_COL32(160,160,160,255);
    dl->AddLine(ImVec2(canvasPos.x, canvasPos.y+plotH), ImVec2(canvasPos.x+plotW, canvasPos.y+plotH), ca);
    dl->AddLine(ImVec2(canvasPos.x, canvasPos.y),       ImVec2(canvasPos.x, canvasPos.y+plotH), ca);
    ImU32 cg = IM_COL32(40,40,40,255);
    for (int i=1;i<=4;i++) {
        float xg = canvasPos.x + plotW*i/4.0f, yg = canvasPos.y + plotH*i/4.0f;
        dl->AddLine(ImVec2(xg,canvasPos.y), ImVec2(xg,canvasPos.y+plotH), cg);
        dl->AddLine(ImVec2(canvasPos.x,yg), ImVec2(canvasPos.x+plotW,yg), cg);
    }
    { ImVec2 ts=ImGui::CalcTextSize(xlabel); dl->AddText(ImVec2(canvasPos.x+(plotW-ts.x)*0.5f, canvasPos.y+plotH+16), ca, xlabel); }
    dl->AddText(ImVec2(canvasPos.x - yts.x - 4, canvasPos.y + plotH*0.5f - yts.y*0.5f), ca, ylabel);
    return true;
}

// ─── Rotation Curve ──────────────────────────────────────────────────────────
void App::renderRotationCurve()
{
    if (!m_showRotCurve) return;

    ImGui::SetNextWindowSize(ImVec2(580, 460), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(350, 30),  ImGuiCond_FirstUseEver);
    ImGui::Begin("Rotation Curve", &m_showRotCurve,
        ImGuiWindowFlags_NoCollapse);

    if (ImGui::Button("Refresh") && m_simulation && !m_positions.empty())
        m_simulation->readVelocities(m_rotVelocities);

    ImGui::SameLine();
    ImGui::TextDisabled("(%zu particles)", m_rotVelocities.size());
    ImGui::SameLine();
    float velConv = (m_lyPerUnit * 9.461e12f) / (m_timeStep * 3.156e13f);
    ImGui::TextDisabled("1 sim = %.1f km/s", velConv);

    ImGui::Separator();
    SliderFloatWithInput("R max (LY)", &m_rotCurveRMax, 0.0f, 200000.0f, "%.0f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("0 = auto (3× galaxy radius)");
    ImGui::Separator();

    ImVec2 canvasPos  = ImGui::GetCursorScreenPos();
    ImVec2 canvasSize = ImGui::GetContentRegionAvail();
    if (canvasSize.x < 100) canvasSize.x = 100;
    if (canvasSize.y < 100) canvasSize.y = 100;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(canvasPos,
        ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
        IM_COL32(8, 16, 48, 255));
    dl->AddRect(canvasPos,
        ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
        IM_COL32(100, 100, 100, 200));

    if (m_rotVelocities.empty() || m_positions.empty())
    {
        dl->AddText(ImVec2(canvasPos.x + 10, canvasPos.y + 10),
            IM_COL32(150,150,150,255), "Click 'Refresh' to compute rotation curve");
        ImGui::Dummy(canvasSize);
        ImGui::End();
        return;
    }

    float rLyMax = (m_rotCurveRMax > 0.0f) ? m_rotCurveRMax : (m_radius * m_lyPerUnit * 3.0f);
    float vtMax   = 0.0f;

    struct Point { float r, vt; };
    std::vector<Point> pts;
    pts.reserve(m_numPosiStars + 1);

    size_t nVel = m_rotVelocities.size();
    size_t nPos = m_positions.size();
    size_t n    = (nVel < nPos) ? nVel : nPos;

    for (size_t i = 0; i < n; i++)
    {
        const cl_float4& pos = m_positions[i];
        const cl_float4& vel = m_rotVelocities[i];
        if (pos.w <= 0.0f) continue;

        // Recenter on positive COM (post-merger drift) + subtract bulk velocity
        float x  = pos.x - m_baryPosX,  y  = pos.y - m_baryPosY;
        float vx = vel.x - m_baryPosVx, vy = vel.y - m_baryPosVy;
        float r  = sqrtf(x*x + y*y);
        if (r < 1.0f) continue;

        float vt = (vy*x - vx*y) / r;
        float vtKms = vt * velConv;
        float rLy   = r  * m_lyPerUnit;

        if (rLy > rLyMax) continue;
        if (vtKms < 0.0f) vtKms = -vtKms;
        if (vtKms > vtMax) vtMax = vtKms;

        pts.push_back({rLy, vtKms});
    }

    if (vtMax < 1.0f) vtMax = 1.0f;

    float yLabelMaxW = 0.0f;
    char bufTest[32];
    snprintf(bufTest, sizeof(bufTest), "%.0f km/s", vtMax);
    yLabelMaxW = fmaxf(yLabelMaxW, ImGui::CalcTextSize(bufTest).x);
    yLabelMaxW = fmaxf(yLabelMaxW, ImGui::CalcTextSize("0").x);
    yLabelMaxW = fmaxf(yLabelMaxW, ImGui::CalcTextSize("km/s").x);

    const float marginL = fmaxf(55.0f, yLabelMaxW + 8.0f);
    const float marginB = 38.0f, marginT = 12.0f, marginR = 12.0f;
    float plotW = canvasSize.x - marginL - marginR;
    float plotH = canvasSize.y - marginB - marginT;

    auto toScreen = [&](float r, float vt) -> ImVec2 {
        float sx = canvasPos.x + marginL + (r  / rLyMax) * plotW;
        float sy = canvasPos.y + marginT + (1.0f - vt / vtMax) * plotH;
        return ImVec2(sx, sy);
    };

    ImU32 colGrid = IM_COL32(50, 50, 50, 200);
    for (int i = 1; i <= 4; i++) {
        float xg = canvasPos.x + marginL + plotW * i / 4.0f;
        float yg = canvasPos.y + marginT + plotH * i / 4.0f;
        dl->AddLine(ImVec2(xg, canvasPos.y + marginT),
                    ImVec2(xg, canvasPos.y + marginT + plotH), colGrid);
        dl->AddLine(ImVec2(canvasPos.x + marginL, yg),
                    ImVec2(canvasPos.x + marginL + plotW, yg), colGrid);
    }

    ImU32 colAxis = IM_COL32(180, 180, 180, 255);
    dl->AddLine(ImVec2(canvasPos.x + marginL, canvasPos.y + marginT + plotH),
                ImVec2(canvasPos.x + marginL + plotW, canvasPos.y + marginT + plotH), colAxis);
    dl->AddLine(ImVec2(canvasPos.x + marginL, canvasPos.y + marginT),
                ImVec2(canvasPos.x + marginL, canvasPos.y + marginT + plotH), colAxis);

    char buf[32];
    float axL = canvasPos.x + marginL;
    snprintf(buf, sizeof(buf), "%.0f LY", rLyMax);
    { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(axL+plotW-ts.x, canvasPos.y+marginT+plotH+2), colAxis, buf); }
    dl->AddText(ImVec2(axL, canvasPos.y+marginT+plotH+2), colAxis, "0");
    { ImVec2 ts=ImGui::CalcTextSize("Distance from Center (LY)");
      dl->AddText(ImVec2(axL+(plotW-ts.x)*0.5f, canvasPos.y+marginT+plotH+18), IM_COL32(200,200,200,200), "Distance from Center (LY)"); }
    snprintf(buf, sizeof(buf), "%.0f km/s", vtMax);
    { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(axL-ts.x-2, canvasPos.y+marginT), colAxis, buf); }
    { ImVec2 ts=ImGui::CalcTextSize("0"); dl->AddText(ImVec2(axL-ts.x-2, canvasPos.y+marginT+plotH-12), colAxis, "0"); }
    dl->AddText(ImVec2(canvasPos.x+1, canvasPos.y+marginT+plotH*0.4f), IM_COL32(200,200,200,200), "km/s");

    ImU32 colDot = IM_COL32(0, 200, 80, 60);
    for (const auto& p : pts)
    {
        ImVec2 sc = toScreen(p.r, p.vt);
        dl->AddCircleFilled(sc, 1.5f, colDot);
    }

    const int   nBins   = 60;
    std::vector<float> sumV(nBins, 0.0f);
    std::vector<int>   cnt (nBins, 0);

    for (const auto& p : pts)
    {
        int bin = (int)(p.r / rLyMax * nBins);
        if (bin < 0) bin = 0;
        if (bin >= nBins) bin = nBins - 1;
        sumV[bin] += p.vt;
        cnt [bin]++;
    }

    // Gaussian smoothing of velocity per bin
    const int sigB = 2, halfK = 6;
    std::vector<float> vSmooth(nBins, 0.0f);
    for (int b = 0; b < nBins; b++)
    {
        double sumW = 0.0, sumWV = 0.0;
        for (int k = -halfK; k <= halfK; k++) {
            int bb = b + k;
            if (bb < 0 || bb >= nBins || cnt[bb] < 3) continue;
            double w = exp(-0.5 * (double)(k*k) / (double)(sigB*sigB));
            sumW  += w;
            sumWV += w * (sumV[bb] / cnt[bb]);
        }
        vSmooth[b] = (sumW > 0.0) ? (float)(sumWV / sumW) : 0.0f;
    }

    // Clip curve to plot area
    {
        float clipX0 = canvasPos.x + marginL;
        float clipY0 = canvasPos.y + marginT;
        dl->PushClipRect(ImVec2(clipX0, clipY0),
                         ImVec2(clipX0 + plotW, clipY0 + plotH), true);
        ImVec2 prev = ImVec2(0,0);
        bool   first = true;
        ImU32  colCurve = IM_COL32(100, 180, 255, 240);
        for (int b = 0; b < nBins; b++)
        {
            if (cnt[b] < 3) { first = true; continue; }
            float rMid = (b + 0.5f) / nBins * rLyMax;
            ImVec2 cur = toScreen(rMid, vSmooth[b]);
            if (!first) dl->AddLine(prev, cur, colCurve, 2.0f);
            prev  = cur;
            first = false;
        }
        dl->PopClipRect();
    }

    {
        float G_eff = m_gravConstant * 1.0e10f;
        float Mtot  = m_totalSolarMass;
        // Clip Keplerian curve to plot area so it never overflows the window
        float clipX0 = canvasPos.x + marginL;
        float clipY0 = canvasPos.y + marginT;
        dl->PushClipRect(ImVec2(clipX0, clipY0),
                         ImVec2(clipX0 + plotW, clipY0 + plotH), true);
        ImVec2 kPrev = ImVec2(0,0); bool kFirst = true;
        for (int ki=1; ki<=200; ki++) {
            float rly  = rLyMax * ki / 200.0f;
            float rSim = rly / m_lyPerUnit;
            float frac = rSim / m_radius; if (frac > 1.0f) frac = 1.0f;
            float Menc = m_blackHoleMass + Mtot * frac * frac;
            float vk   = sqrtf(G_eff * Menc / fmaxf(rSim, 0.1f)) * velConv;
            ImVec2 kCur = toScreen(rly, vk);
            if (!kFirst) dl->AddLine(kPrev, kCur, IM_COL32(255,180,0,180), 1.5f);
            kPrev = kCur; kFirst = false;
        }
        dl->PopClipRect();
        float lx = canvasPos.x + marginL + 4, ly = canvasPos.y + marginT + 4;
        ImU32 colCurve = IM_COL32(100, 180, 255, 240);
        dl->AddLine(ImVec2(lx,ly+6),   ImVec2(lx+22,ly+6),   colCurve,              2.0f);
        dl->AddText(ImVec2(lx+24,ly),  IM_COL32(140,200,255,220), "Janus (sim)");
        dl->AddLine(ImVec2(lx,ly+20),  ImVec2(lx+22,ly+20),  IM_COL32(255,180,0,180),2.0f);
        dl->AddText(ImVec2(lx+24,ly+14),IM_COL32(255,200,80,220),"Keplerian");

        // Compteur de temps de simulation
        double elapsed = (double)m_currentFrame * (double)m_timeStep;
        char timeBuf[64];
        if (elapsed < 1000.0)
            snprintf(timeBuf, sizeof(timeBuf), "T = %.2f MY", elapsed);
        else
            snprintf(timeBuf, sizeof(timeBuf), "T = %.3f BY", elapsed / 1000.0);
        dl->AddText(ImVec2(lx, ly+30), IM_COL32(180,220,180,220), timeBuf);
    }

    ImGui::Dummy(canvasSize);
    ImGui::End();
}

// ─── Density Profile ─────────────────────────────────────────────────────────
void App::renderDensityProfile()
{
    if (!m_showDensityProfile) return;

    ImGui::SetNextWindowSize(ImVec2(640, 460), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Density Profile (3D)", &m_showDensityProfile,
                      ImGuiWindowFlags_NoCollapse)) {
        ImGui::End(); return;
    }

    if (m_positions.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f),
                           "No particles loaded. Click 'Apply & Generate' first.");
        ImGui::End(); return;
    }

    if (ImGui::Button("Refresh") && m_simulation) m_simulation->readPositions(m_positions);
    ImGui::SameLine();
    static bool logScale = true;
    ImGui::Checkbox("Log scale (Y)", &logScale);
    ImGui::SameLine();
    ImGui::TextDisabled("3D shell-normalized density");
    ImGui::Separator();

    // Plot canvas
    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImVec2 sz        = ImGui::GetContentRegionAvail();
    if (sz.x < 100) sz.x = 100;
    if (sz.y < 100) sz.y = 100;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(canvasPos, ImVec2(canvasPos.x+sz.x, canvasPos.y+sz.y), IM_COL32(8,16,48,255));
    dl->AddRect    (canvasPos, ImVec2(canvasPos.x+sz.x, canvasPos.y+sz.y), IM_COL32(80,80,80,255));
    const float mL=78.0f, mR=14.0f, mT=14.0f, mB=34.0f;
    float plotW = sz.x - mL - mR;
    float plotH = sz.y - mB - mT;
    ImVec2 cp(canvasPos.x + mL, canvasPos.y + mT);
    ImU32 ca = IM_COL32(160,160,160,255);
    dl->AddLine(ImVec2(cp.x, cp.y+plotH), ImVec2(cp.x+plotW, cp.y+plotH), ca);
    dl->AddLine(ImVec2(cp.x, cp.y),       ImVec2(cp.x, cp.y+plotH), ca);
    ImU32 cg = IM_COL32(40,40,40,255);
    for (int i=1;i<=4;i++) {
        float xg = cp.x + plotW*i/4.0f, yg = cp.y + plotH*i/4.0f;
        dl->AddLine(ImVec2(xg,cp.y), ImVec2(xg,cp.y+plotH), cg);
        dl->AddLine(ImVec2(cp.x,yg), ImVec2(cp.x+plotW,yg), cg);
    }
    const char* xlabel = "Radius (LY)";
    const char* ylabel = logScale ? "log10 Density (N/LY^3)" : "Density (N/LY^3)";
    { ImVec2 ts=ImGui::CalcTextSize(xlabel); dl->AddText(ImVec2(cp.x+(plotW-ts.x)*0.5f, cp.y+plotH+16), ca, xlabel); }
    { ImVec2 yts=ImGui::CalcTextSize(ylabel); dl->AddText(ImVec2(cp.x - yts.x - 6, cp.y + plotH*0.5f - yts.y*0.5f), ca, ylabel); }

    const int BINS = 80;
    float countPos[BINS]={}, countNeg[BINS]={};

    // Per-species recentering: each profile is measured around its own COM
    // (post-merger the + core and − halo can be spatially separated).
    size_t np = m_positions.size();
    float rMaxSim = 0.0f;
    for (size_t i = 0; i < np; i++) {
        float dx, dy, dz;
        if (m_positions[i].w > 0) {
            dx = m_positions[i].x - m_baryPosX;
            dy = m_positions[i].y - m_baryPosY;
            dz = m_positions[i].z - m_baryPosZ;
        } else if (m_positions[i].w < 0) {
            dx = m_positions[i].x - m_baryNegX;
            dy = m_positions[i].y - m_baryNegY;
            dz = m_positions[i].z - m_baryNegZ;
        } else continue;
        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 > rMaxSim) rMaxSim = r2;
    }
    rMaxSim = sqrtf(rMaxSim);
    if (rMaxSim < 1.0f) rMaxSim = m_radius * 1.2f;
    float rMax = rMaxSim * m_lyPerUnit;

    // Bin particles by TRUE 3D radius from their species' COM
    for (size_t i = 0; i < np; i++) {
        float dx, dy, dz;
        bool isPos = m_positions[i].w > 0;
        bool isNeg = m_positions[i].w < 0;
        if (isPos) {
            dx = m_positions[i].x - m_baryPosX;
            dy = m_positions[i].y - m_baryPosY;
            dz = m_positions[i].z - m_baryPosZ;
        } else if (isNeg) {
            dx = m_positions[i].x - m_baryNegX;
            dy = m_positions[i].y - m_baryNegY;
            dz = m_positions[i].z - m_baryNegZ;
        } else continue;
        float r3d = sqrtf(dx*dx + dy*dy + dz*dz) * m_lyPerUnit;
        int b = (int)(r3d / rMax * BINS);
        if (b < 0) b = 0; if (b >= BINS) b = BINS-1;
        if      (isPos) countPos[b]++;
        else if (isNeg) countNeg[b]++;
    }

    // Normalize by spherical shell volume → real density
    float binPos[BINS]={}, binNeg[BINS]={};
    const float FOUR_THIRDS_PI = 4.18879020478f;
    float binW = rMax / BINS;
    for (int b = 0; b < BINS; b++) {
        float r0 = b * binW;
        float r1 = (b + 1) * binW;
        float vol = FOUR_THIRDS_PI * (r1*r1*r1 - r0*r0*r0);
        if (vol < 1e-12f) vol = 1e-12f;
        binPos[b] = countPos[b] / vol;
        binNeg[b] = countNeg[b] / vol;
    }

    // Gaussian smoothing (σ = 3 bins)
    float smoothPos[BINS]={}, smoothNeg[BINS]={};
    const int sigBins = 3;
    const int halfK   = sigBins * 3;
    for (int b = 0; b < BINS; b++) {
        double sumW=0.0, sumWP=0.0, sumWN=0.0;
        for (int k = -halfK; k <= halfK; k++) {
            int bb = b + k; if (bb < 0 || bb >= BINS) continue;
            double w = exp(-0.5 * (double)(k*k) / (double)(sigBins*sigBins));
            sumW += w; sumWP += w * binPos[bb]; sumWN += w * binNeg[bb];
        }
        smoothPos[b] = (sumW>0.0) ? (float)(sumWP/sumW) : binPos[b];
        smoothNeg[b] = (sumW>0.0) ? (float)(sumWN/sumW) : binNeg[b];
    }

    // Transform : log10 or linear, find min/max
    float plotPos[BINS]={}, plotNeg[BINS]={};
    float minVal =  1e30f, maxVal = -1e30f;
    if (logScale) {
        const float EPS = 1e-15f;
        for (int b = 0; b < BINS; b++) {
            plotPos[b] = log10f(smoothPos[b] + EPS);
            plotNeg[b] = log10f(smoothNeg[b] + EPS);
            if (smoothPos[b] > 0.0f && plotPos[b] < minVal) minVal = plotPos[b];
            if (smoothNeg[b] > 0.0f && plotNeg[b] < minVal) minVal = plotNeg[b];
            if (smoothPos[b] > 0.0f && plotPos[b] > maxVal) maxVal = plotPos[b];
            if (smoothNeg[b] > 0.0f && plotNeg[b] > maxVal) maxVal = plotNeg[b];
        }
    } else {
        for (int b = 0; b < BINS; b++) {
            plotPos[b] = smoothPos[b]; plotNeg[b] = smoothNeg[b];
            if (plotPos[b] > maxVal) maxVal = plotPos[b];
            if (plotNeg[b] > maxVal) maxVal = plotNeg[b];
        }
        minVal = 0.0f;
    }
    if (maxVal <= minVal) maxVal = minVal + 1.0f;
    float yRange = maxVal - minVal;

    auto yMap = [&](float v) -> float {
        float t = (v - minVal) / yRange;
        if (t < 0.0f) t = 0.0f; if (t > 1.0f) t = 1.0f;
        return cp.y + (1.0f - t) * plotH;
    };
    float baseline = cp.y + plotH;

    // Positive (red)
    for (int b = 1; b < BINS; b++) {
        float x0 = cp.x + (float)(b-1) / BINS * plotW;
        float x1 = cp.x + (float)b     / BINS * plotW;
        float y0p = yMap(plotPos[b-1]);
        float y1p = yMap(plotPos[b]);
        dl->AddQuadFilled(
            ImVec2(x0, baseline), ImVec2(x0, y0p),
            ImVec2(x1, y1p),      ImVec2(x1, baseline),
            IM_COL32(200, 60, 60, 60));
        dl->AddLine(ImVec2(x0, y0p), ImVec2(x1, y1p), IM_COL32(255, 100, 100, 230), 1.8f);
    }

    // Negative (fuchsia)
    for (int b = 1; b < BINS; b++) {
        float x0 = cp.x + (float)(b-1) / BINS * plotW;
        float x1 = cp.x + (float)b     / BINS * plotW;
        float y0n = yMap(plotNeg[b-1]);
        float y1n = yMap(plotNeg[b]);
        dl->AddQuadFilled(
            ImVec2(x0, baseline), ImVec2(x0, y0n),
            ImVec2(x1, y1n),      ImVec2(x1, baseline),
            IM_COL32(220, 40, 220, 50));
        dl->AddLine(ImVec2(x0, y0n), ImVec2(x1, y1n), IM_COL32(255, 100, 255, 230), 1.8f);
    }

    // Legend centered top
    {
        float legW = 230, legH = 22;
        float lx = cp.x + (plotW - legW) * 0.5f;
        float ly = cp.y + 4;
        dl->AddRectFilled(ImVec2(lx, ly), ImVec2(lx + legW, ly + legH), IM_COL32(0,0,0,160));
        float ix = lx + 8;
        dl->AddLine(ImVec2(ix, ly + 11), ImVec2(ix + 18, ly + 11), IM_COL32(255, 100, 100, 230), 2.5f);
        dl->AddText(ImVec2(ix + 24, ly + 4), IM_COL32(240, 200, 200, 255), "Positive");
        ix += 110;
        dl->AddLine(ImVec2(ix, ly + 11), ImVec2(ix + 18, ly + 11), IM_COL32(255, 100, 255, 230), 2.5f);
        dl->AddText(ImVec2(ix + 24, ly + 4), IM_COL32(240, 180, 240, 255), "Negative");
    }

    // Axis labels
    char buf[48];
    snprintf(buf, sizeof(buf), "%.0f LY", rMax);
    { ImVec2 ts = ImGui::CalcTextSize(buf);
      dl->AddText(ImVec2(cp.x + plotW - ts.x, cp.y + plotH + 2), ca, buf); }
    dl->AddText(ImVec2(cp.x, cp.y + plotH + 2), ca, "0");
    if (logScale) {
        snprintf(buf, sizeof(buf), "1e%+.0f", maxVal);
        { ImVec2 ts = ImGui::CalcTextSize(buf);
          dl->AddText(ImVec2(cp.x - ts.x - 4, cp.y - 2), ca, buf); }
        snprintf(buf, sizeof(buf), "1e%+.0f", minVal);
        { ImVec2 ts = ImGui::CalcTextSize(buf);
          dl->AddText(ImVec2(cp.x - ts.x - 4, cp.y + plotH - 12), ca, buf); }
    } else {
        snprintf(buf, sizeof(buf), "%.2e", maxVal);
        { ImVec2 ts = ImGui::CalcTextSize(buf);
          dl->AddText(ImVec2(cp.x - ts.x - 4, cp.y - 2), ca, buf); }
        dl->AddText(ImVec2(cp.x - 14, cp.y + plotH - 12), ca, "0");
    }

    ImGui::Dummy(sz);
    ImGui::End();
}

// ─── Phase Space ─────────────────────────────────────────────────────────────
void App::renderPhaseSpace()
{
    if (!m_showPhaseSpace) return;
    float plotW, plotH; ImVec2 cp;
    if (!beginPlotWindow("Phase Space  (r, v_radial)", &m_showPhaseSpace,
                         plotW, plotH, cp, "Radius (LY)", "v_r (km/s)")) return;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (m_rotVelocities.empty()) {
        dl->AddText(ImVec2(cp.x+4,cp.y+4), IM_COL32(150,150,150,255), "Click 'Refresh Analytics'");
        ImGui::Dummy(ImVec2(plotW+60,plotH+38)); ImGui::End(); return;
    }

    ImGui::Checkbox("Show Negative Mass", &m_phaseShowNeg);

    float velConv = (m_lyPerUnit*9.461e12f)/(m_timeStep*3.156e13f);
    float rMax    = m_radius * m_lyPerUnit * 3.0f;
    float vrAbsMax = 0;

    struct Pt { float r, vr; bool neg; };
    std::vector<Pt> pts;
    pts.reserve(m_numPosiStars + (m_phaseShowNeg ? m_numNegaStars : 0));
    size_t n = m_rotVelocities.size() < m_positions.size() ? m_rotVelocities.size() : m_positions.size();
    for (size_t i = 0; i < n; i++) {
        bool isNeg = m_positions[i].w < 0;
        if (isNeg && !m_phaseShowNeg) continue;
        if (m_positions[i].w == 0) continue;
        // Per-species recentering + bulk-velocity subtraction
        float cx  = isNeg ? m_baryNegX  : m_baryPosX;
        float cy  = isNeg ? m_baryNegY  : m_baryPosY;
        float vcx = isNeg ? m_baryNegVx : m_baryPosVx;
        float vcy = isNeg ? m_baryNegVy : m_baryPosVy;
        float x = m_positions[i].x - cx, y = m_positions[i].y - cy;
        float r = sqrtf(x*x+y*y); if (r < 1) continue;
        float vx = m_rotVelocities[i].x - vcx;
        float vy = m_rotVelocities[i].y - vcy;
        float vr = (vx*x + vy*y) / r * velConv;
        float rLy = r * m_lyPerUnit;
        if (rLy > rMax) continue;
        if (fabsf(vr) > vrAbsMax) vrAbsMax = fabsf(vr);
        pts.push_back({rLy, vr, isNeg});
    }
    if (vrAbsMax < 1) vrAbsMax = 1;

    float y0 = cp.y + 0.5f*plotH;
    dl->AddLine(ImVec2(cp.x,y0), ImVec2(cp.x+plotW,y0), IM_COL32(100,100,60,180), 1.0f);

    for (const auto& p : pts) {
        float sx = cp.x + p.r / rMax * plotW;
        float sy = cp.y + (1.0f - (p.vr+vrAbsMax)/(2*vrAbsMax)) * plotH;
        ImU32 col = p.neg ? IM_COL32(255,140,40,60) : IM_COL32(80,220,100,50);
        dl->AddCircleFilled(ImVec2(sx,sy), 1.2f, col);
    }

    // Legend
    if (m_phaseShowNeg) {
        float lx = cp.x + 4, ly = cp.y + 4;
        dl->AddRectFilled(ImVec2(lx,ly), ImVec2(lx+100,ly+32), IM_COL32(0,0,0,140));
        dl->AddCircleFilled(ImVec2(lx+8,ly+9),  3.f, IM_COL32(80,220,100,180));
        dl->AddText(ImVec2(lx+14,ly+2),  IM_COL32(180,220,180,220), "Positive");
        dl->AddCircleFilled(ImVec2(lx+8,ly+23), 3.f, IM_COL32(255,140,40,180));
        dl->AddText(ImVec2(lx+14,ly+16), IM_COL32(255,180,100,220), "Negative");
    }

    char buf[32];
    snprintf(buf,sizeof(buf),"%.0f km/s",vrAbsMax);
    { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(cp.x-ts.x-3,cp.y),IM_COL32(160,160,160,255),buf); }
    snprintf(buf,sizeof(buf),"-%.0f",vrAbsMax);
    { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(cp.x-ts.x-3,cp.y+plotH-12),IM_COL32(160,160,160,255),buf); }
    snprintf(buf,sizeof(buf),"%.0f LY",rMax);
    { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(cp.x+plotW-ts.x,cp.y+plotH+2),IM_COL32(160,160,160,255),buf); }
    { ImVec2 ts=ImGui::CalcTextSize("0"); dl->AddText(ImVec2(cp.x-ts.x-3,cp.y+0.5f*plotH-6),IM_COL32(100,100,80,200),"0"); }

    ImGui::Dummy(ImVec2(plotW+60,plotH+38)); ImGui::End();
}

// ─── Energy Plot ─────────────────────────────────────────────────────────────
void App::renderEnergyPlot()
{
    if (!m_showEnergyPlot) return;
    ImGui::SetNextWindowSize(ImVec2(620, 460), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(360, 30),  ImGuiCond_FirstUseEver);
    ImGui::Begin("Kinetic Energy", &m_showEnergyPlot,
        ImGuiWindowFlags_NoCollapse);

    if (ImGui::Button("Refresh") && m_simulation && !m_positions.empty())
        refreshAnalytics();

    ImGui::SameLine();
    ImGui::TextDisabled("KE evolution");
    if (m_energyCount < 2) {
        ImGui::TextDisabled("Enable Auto-Refresh to record energy");
        ImGui::End();
        return;
    }

    ImVec2 canvasPos  = ImGui::GetCursorScreenPos();
    ImVec2 canvasSize = ImGui::GetContentRegionAvail();
    if (canvasSize.x < 100) canvasSize.x = 100;
    if (canvasSize.y < 100) canvasSize.y = 100;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(canvasPos,
        ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
        IM_COL32(8, 16, 48, 255));
    dl->AddRect(canvasPos,
        ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
        IM_COL32(100, 100, 100, 200));

    const float marginL = 80.0f, marginR = 10.0f, marginT = 10.0f, marginB = 32.0f;
    float plotW = canvasSize.x - marginL - marginR;
    float plotH = canvasSize.y - marginT - marginB;
    float px = canvasPos.x + marginL;
    float py = canvasPos.y + marginT;

    float keMin=1e30f, keMax=-1e30f;
    double tMin=1e30,  tMax=-1e30;
    int cnt = m_energyCount;
    int head = m_energyHead;
    for (int i=0;i<cnt;i++) {
        int idx = (head - cnt + i + ENERGY_HISTORY) % ENERGY_HISTORY;
        float ke = m_energyHistory[idx];
        double t = m_energyTimes[idx];
        if (ke<keMin) keMin=ke; if (ke>keMax) keMax=ke;
        if (t<tMin) tMin=t;   if (t>tMax) tMax=t;
    }
    if (keMax<=keMin) keMax=keMin+1;
    if (tMax<=tMin)   tMax=tMin+1;

    ImU32 colGrid = IM_COL32(50, 50, 50, 200);
    for (int i = 1; i <= 4; i++) {
        float xg = px + plotW * i / 4.0f;
        float yg = py + plotH * i / 4.0f;
        dl->AddLine(ImVec2(xg, py), ImVec2(xg, py + plotH), colGrid);
        dl->AddLine(ImVec2(px, yg), ImVec2(px + plotW, yg), colGrid);
    }
    ImU32 colAxis = IM_COL32(180, 180, 180, 255);
    dl->AddLine(ImVec2(px, py + plotH), ImVec2(px + plotW, py + plotH), colAxis);
    dl->AddLine(ImVec2(px, py), ImVec2(px, py + plotH), colAxis);

    // Linear copy + Gaussian smoothing
    std::vector<float> visKE(cnt);
    std::vector<double> visT(cnt);
    for (int i=0;i<cnt;i++) {
        int idx = (head - cnt + i + ENERGY_HISTORY) % ENERGY_HISTORY;
        visKE[i] = m_energyHistory[idx];
        visT[i]  = m_energyTimes[idx];
    }
    int sigBe = std::max(3, std::min(30, cnt / 20));
    int halfKe = sigBe * 3;
    std::vector<float> smKE(cnt);
    for (int i=0;i<cnt;i++) {
        double sumW=0.0, sumWV=0.0;
        for (int k=-halfKe;k<=halfKe;k++) {
            int j=i+k; if (j<0||j>=cnt) continue;
            double w = exp(-0.5*(double)(k*k)/(double)(sigBe*sigBe));
            sumW += w; sumWV += w * visKE[j];
        }
        smKE[i] = (sumW>0.0) ? (float)(sumWV/sumW) : visKE[i];
    }
    dl->PushClipRect(ImVec2(px, py), ImVec2(px+plotW, py+plotH), true);
    ImVec2 prev=ImVec2(0,0); bool first=true;
    for (int i=0;i<cnt;i++) {
        float sx = px + (float)i / (cnt > 1 ? cnt - 1 : 1) * plotW;
        float sy = py + (1.0f-(smKE[i]-keMin)/(keMax-keMin))*plotH;
        ImVec2 cur(sx,sy);
        if (!first) dl->AddLine(prev, cur, IM_COL32(100,220,255,220), 1.8f);
        prev=cur; first=false;
    }
    dl->PopClipRect();

    char buf[48];
    double span = tMax-tMin;
    const char* unit = span<1000?"MY":"BY";
    double div = span<1000?1.0:1000.0;
    snprintf(buf,sizeof(buf),"%.1f %s",tMin/div,unit);
    dl->AddText(ImVec2(px,py+plotH+2),IM_COL32(160,160,160,255),buf);
    snprintf(buf,sizeof(buf),"%.1f %s",tMax/div,unit);
    { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px+plotW-ts.x,py+plotH+2),IM_COL32(160,160,160,255),buf); }
    snprintf(buf,sizeof(buf),"%.2e",keMax);
    { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px-ts.x-3,py),IM_COL32(160,160,160,255),buf); }
    snprintf(buf,sizeof(buf),"%.2e",keMin);
    { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px-ts.x-3,py+plotH-12),IM_COL32(160,160,160,255),buf); }

    ImGui::Dummy(canvasSize);
    ImGui::End();
}

// ─── Dispersion & Toomre Q ───────────────────────────────────────────────────
void App::renderDispersionPlot()
{
    if (!m_showDispersion) return;

    ImGui::SetNextWindowSize(ImVec2(620, 540), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(350, 30),   ImGuiCond_FirstUseEver);
    ImGui::Begin("Velocity Dispersion & Toomre Q", &m_showDispersion,
                 ImGuiWindowFlags_NoCollapse);

    if (m_rotVelocities.empty() || m_positions.empty()) {
        ImGui::TextDisabled("Click 'Refresh Analytics' to compute");
        ImGui::End(); return;
    }

    const int   BINS   = 60;
    const float rMax   = m_radius * m_lyPerUnit * 3.0f;
    float velConv      = (m_lyPerUnit * 9.461e12f) / (m_timeStep * 3.156e13f);
    float G_eff        = m_gravConstant * 1.0e10f;
    float massPP       = (m_numPosiStars > 0) ? m_totalSolarMass / (float)m_numPosiStars : 1.0f;

    size_t n = m_rotVelocities.size() < m_positions.size()
               ? m_rotVelocities.size() : m_positions.size();

    // Use precomputed positive COM + bulk velocity from refreshAnalytics —
    // the galaxy can drift after mergers, and bulk motion would contaminate
    // intrinsic sigma_r / sigma_t if not subtracted.
    float sumVr[BINS]={}, sumVr2[BINS]={};
    float sumVt[BINS]={}, sumVt2[BINS]={};
    float sumMassAnn[BINS]={};
    int   cnt[BINS]={};

    for (size_t i = 0; i < n; i++) {
        if (m_positions[i].w <= 0.0f) continue;
        float x = m_positions[i].x - m_baryPosX, y = m_positions[i].y - m_baryPosY;
        float r = sqrtf(x*x + y*y);
        if (r < 1.0f) continue;
        float rLy = r * m_lyPerUnit;
        if (rLy > rMax) continue;
        int b = (int)(rLy / rMax * BINS);
        if (b >= BINS) b = BINS - 1;
        float vx = m_rotVelocities[i].x - m_baryPosVx;
        float vy = m_rotVelocities[i].y - m_baryPosVy;
        float vr = (vx*x + vy*y) / r;
        float vt = (vy*x - vx*y) / r;
        sumVr[b]  += vr;  sumVr2[b] += vr*vr;
        sumVt[b]  += vt;  sumVt2[b] += vt*vt;
        sumMassAnn[b] += massPP;
        cnt[b]++;
    }

    float sigR[BINS]={}, sigT[BINS]={}, vcirc[BINS]={};
    float sigMax = 1.0f;
    for (int b = 0; b < BINS; b++) {
        if (cnt[b] < 3) continue;
        float invN = 1.0f / cnt[b];
        float mVr  = sumVr[b] * invN, mVt = sumVt[b] * invN;
        sigR[b]   = sqrtf(fmaxf(sumVr2[b]*invN - mVr*mVr, 0.0f)) * velConv;
        sigT[b]   = sqrtf(fmaxf(sumVt2[b]*invN - mVt*mVt, 0.0f)) * velConv;
        vcirc[b]  = fabsf(mVt);
        if (sigR[b] > sigMax) sigMax = sigR[b];
        if (sigT[b] > sigMax) sigMax = sigT[b];
    }

    float Q[BINS]={};
    float Qmax  = 5.0f;
    bool  anyQ  = false;
    for (int b = 0; b < BINS; b++) {
        if (cnt[b] < 3 || sigR[b] < 1e-6f) continue;
        float rMid  = (b + 0.5f) / BINS * rMax / m_lyPerUnit;
        float kappa = 1.41421356f * vcirc[b] / fmaxf(rMid, 1.0f);
        float rIn   = (float)b / BINS * rMax / m_lyPerUnit;
        float rOut  = (float)(b+1) / BINS * rMax / m_lyPerUnit;
        float area  = 3.14159265f * (rOut*rOut - rIn*rIn);
        if (area < 1e-20f) continue;
        float Sigma = sumMassAnn[b] / area;
        if (Sigma < 1e-30f) continue;
        float sigR_sim = sigR[b] / velConv;
        Q[b] = sigR_sim * kappa / (3.36f * G_eff * Sigma);
        anyQ = true;
        if (Q[b] < 50.0f && Q[b] > Qmax) Qmax = Q[b];
    }

    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImVec2 avail  = ImGui::GetContentRegionAvail();
    if (avail.x < 80) avail.x = 80;
    if (avail.y < 80) avail.y = 80;

    const float mL=70, mR=10, mT=10, mB=30;
    float halfH = avail.y * 0.5f - 8.0f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImU32 ca = IM_COL32(160,160,160,255);
    ImU32 cg = IM_COL32(40,40,40,255);

    auto drawAxes = [&](float ox, float oy, float bw, float bh,
                        float& px, float& py, float& pw, float& ph) {
        dl->AddRectFilled(ImVec2(ox,oy), ImVec2(ox+bw,oy+bh), IM_COL32(8,16,48,255));
        dl->AddRect(ImVec2(ox,oy),       ImVec2(ox+bw,oy+bh), IM_COL32(80,80,80,255));
        pw = bw - mL - mR;  ph = bh - mT - mB;
        px = ox + mL;       py = oy + mT;
        for (int i=1;i<=4;i++) {
            float xg = px + pw*i/4.0f, yg = py + ph*i/4.0f;
            dl->AddLine(ImVec2(xg,py),ImVec2(xg,py+ph),cg);
            dl->AddLine(ImVec2(px,yg),ImVec2(px+pw,yg),cg);
        }
        dl->AddLine(ImVec2(px,py+ph),ImVec2(px+pw,py+ph),ca);
        dl->AddLine(ImVec2(px,py),   ImVec2(px,py+ph),   ca);
    };

    {
        float px, py, pw, ph;
        drawAxes(origin.x, origin.y, avail.x, halfH, px, py, pw, ph);

        // Gaussian smoothing of sigR, sigT
        const int sigB = 2, halfK = 6;
        float smR[BINS]={}, smT[BINS]={};
        for (int b = 0; b < BINS; b++) {
            double sumW=0.0, sumWR=0.0, sumWT=0.0;
            for (int k=-halfK;k<=halfK;k++) {
                int j=b+k; if (j<0||j>=BINS||cnt[j]<3) continue;
                double w = exp(-0.5*(double)(k*k)/(double)(sigB*sigB));
                sumW += w; sumWR += w * sigR[j]; sumWT += w * sigT[j];
            }
            smR[b] = (sumW>0.0) ? (float)(sumWR/sumW) : sigR[b];
            smT[b] = (sumW>0.0) ? (float)(sumWT/sumW) : sigT[b];
        }
        dl->PushClipRect(ImVec2(px,py), ImVec2(px+pw,py+ph), true);
        ImVec2 prevR = ImVec2(0,0), prevT = ImVec2(0,0);
        bool firstR = true, firstT = true;
        for (int b = 0; b < BINS; b++) {
            if (cnt[b] < 3) { firstR = true; firstT = true; continue; }
            float rLy = (b + 0.5f) / BINS * rMax;
            float sx  = px + rLy / rMax * pw;
            float syR = py + (1.0f - smR[b]/sigMax) * ph;
            float syT = py + (1.0f - smT[b]/sigMax) * ph;
            if (!firstR) dl->AddLine(prevR, ImVec2(sx,syR), IM_COL32(255,255,255,200), 1.8f);
            if (!firstT) dl->AddLine(prevT, ImVec2(sx,syT), IM_COL32(80,220,255,200),  1.8f);
            prevR = ImVec2(sx,syR); prevT = ImVec2(sx,syT);
            firstR = false; firstT = false;
        }
        dl->PopClipRect();
        char buf[32];
        snprintf(buf,sizeof(buf),"%.0f km/s",sigMax);
        { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px-ts.x-2,py), ca, buf); }
        { ImVec2 ts=ImGui::CalcTextSize("0"); dl->AddText(ImVec2(px-ts.x-2,py+ph-12), ca, "0"); }
        { ImVec2 ts=ImGui::CalcTextSize("Radius (LY)"); dl->AddText(ImVec2(px+(pw-ts.x)*0.5f,py+ph+14), ca, "Radius (LY)"); }
        snprintf(buf,sizeof(buf),"%.0f LY",rMax);
        { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px+pw-ts.x,py+ph+2), ca, buf); }
        dl->AddText(ImVec2(px,py+ph+2), ca, "0");
        float lx=px+4, ly=py+4;
        dl->AddLine(ImVec2(lx,ly+6), ImVec2(lx+18,ly+6), IM_COL32(255,255,255,200),2);
        dl->AddText(ImVec2(lx+20,ly),   IM_COL32(220,220,220,230), "sr (radial)");
        dl->AddLine(ImVec2(lx,ly+20),ImVec2(lx+18,ly+20),IM_COL32(80,220,255,200),2);
        dl->AddText(ImVec2(lx+20,ly+14),IM_COL32(80,220,255,230),"st (tangential)");
    }

    {
        float oy2 = origin.y + halfH + 12.0f;
        float px, py, pw, ph;
        drawAxes(origin.x, oy2, avail.x, halfH, px, py, pw, ph);

        if (Qmax >= 1.0f) {
            float y1 = py + (1.0f - 1.0f/Qmax) * ph;
            dl->AddLine(ImVec2(px,y1), ImVec2(px+pw,y1), IM_COL32(255,80,80,160), 1.0f);
            dl->AddText(ImVec2(origin.x+2, y1-8), IM_COL32(255,100,100,200), "Q=1");
        }

        if (anyQ) {
            // Gaussian smoothing of Q
            const int sigB = 2, halfK = 6;
            float smQ[BINS]={};
            for (int b = 0; b < BINS; b++) {
                double sumW=0.0, sumWV=0.0;
                for (int k=-halfK;k<=halfK;k++) {
                    int j=b+k; if (j<0||j>=BINS||cnt[j]<3||Q[j]<=0.0f) continue;
                    double w = exp(-0.5*(double)(k*k)/(double)(sigB*sigB));
                    sumW += w; sumWV += w * Q[j];
                }
                smQ[b] = (sumW>0.0) ? (float)(sumWV/sumW) : Q[b];
            }
            dl->PushClipRect(ImVec2(px,py), ImVec2(px+pw,py+ph), true);
            ImVec2 prev={}; bool first=true;
            for (int b = 0; b < BINS; b++) {
                if (cnt[b] < 3 || Q[b] <= 0.0f) { first=true; continue; }
                float rLy = (b + 0.5f) / BINS * rMax;
                float Qcl = smQ[b] < Qmax ? smQ[b] : Qmax;
                float sx  = px + rLy / rMax * pw;
                float sy  = py + (1.0f - Qcl/Qmax) * ph;
                if (!first) dl->AddLine(prev, ImVec2(sx,sy), IM_COL32(255,220,60,220), 1.8f);
                prev=ImVec2(sx,sy); first=false;
            }
            dl->PopClipRect();
        } else {
            dl->AddText(ImVec2(px+4,py+4), IM_COL32(150,150,150,255),
                        "Q unavailable — enable Auto-Refresh");
        }
        char buf[32];
        snprintf(buf,sizeof(buf),"%.1f",Qmax);
        { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px-ts.x-2,py), ca, buf); }
        { ImVec2 ts=ImGui::CalcTextSize("0"); dl->AddText(ImVec2(px-ts.x-2,py+ph-12), ca, "0"); }
        { ImVec2 ts=ImGui::CalcTextSize("Radius (LY)"); dl->AddText(ImVec2(px+(pw-ts.x)*0.5f,py+ph+14), ca, "Radius (LY)"); }
        snprintf(buf,sizeof(buf),"%.0f LY",rMax);
        { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px+pw-ts.x,py+ph+2), ca, buf); }
        dl->AddText(ImVec2(px,py+ph+2), ca, "0");
        { ImVec2 ts=ImGui::CalcTextSize("Q"); dl->AddText(ImVec2(px-ts.x-2,py+ph*0.35f), ca, "Q"); }
    }

    ImGui::Dummy(ImVec2(avail.x, avail.y));
    ImGui::End();
}

// ─── Grid ────────────────────────────────────────────────────────────────────
void App::renderGrid()
{
    if (!m_showGrid || !m_camera) return;

    ImGuiViewport* gMV = ImGui::GetMainViewport();
    const float gMVX = gMV->Pos.x, gMVY = gMV->Pos.y;

    const float panelW  = 340.0f;
    const float vpX     = gMVX + panelW;
    const float vpW     = (float)m_width  - panelW;
    const float vpH     = (float)m_height;
    if (vpW <= 0 || vpH <= 0) return;

    Vector* orig  = m_camera->getOrigin();
    float cam_x   = (float)orig->x;
    float cam_y   = (float)orig->y;
    float cam_z   = (float)orig->z;
    bool  edgeOn  = m_camera->isEdgeOn();

    const float fovDeg = 60.0f;
    float tanH         = tanf(fovDeg * 3.14159265f / 360.0f);

    float camDist, worldHalfW, worldHalfH;
    if (edgeOn) {
        camDist    = fabsf(cam_y);
        worldHalfH = camDist * tanH;
        worldHalfW = worldHalfH * (vpW / vpH);
    } else if (m_camera->isOrbitMode()) {
        // Orbit mode : distance totale au centre (0,0,0)
        camDist    = sqrtf(cam_x*cam_x + cam_y*cam_y + cam_z*cam_z);
        worldHalfH = camDist * tanH;
        worldHalfW = worldHalfH * (vpW / vpH);
    } else {
        camDist    = fabsf(cam_z);
        worldHalfH = camDist * tanH;
        worldHalfW = worldHalfH * (vpW / vpH);
    }

    float worldWidthLy = 2.0f * worldHalfW * m_lyPerUnit;
    float idealLy      = worldWidthLy / 10.0f;

    static const float niceVals[] = {10,20,50,100,200,500,1000,2000,5000,10000,20000,50000};
    float gridLy = niceVals[sizeof(niceVals)/sizeof(float)-1];
    for (float nv : niceVals) { if (nv >= idealLy) { gridLy = nv; break; } }
    float gridUnits = gridLy / m_lyPerUnit;

    float vp[16];
    m_camera->getViewProjMatrix(vp);

    // Simple inline function for world-to-screen projection
    auto w2s = [&](float wx, float wy) -> ImVec2 {
        if (edgeOn) {
            return ImVec2(vpX + (wx / worldHalfW + 1.0f) * 0.5f * vpW,
                          gMVY + (1.0f - wy / worldHalfH) * 0.5f * vpH);
        }
        float cx = vp[0]*wx + vp[4]*wy + vp[12];
        float cy = vp[1]*wx + vp[5]*wy + vp[13];
        float cw = vp[3]*wx + vp[7]*wy + vp[15];
        if (fabsf(cw) < 1e-6f) cw = 1e-6f;
        return ImVec2(vpX + (cx/cw + 1.0f) * 0.5f * vpW,
                      gMVY + (1.0f - cy/cw) * 0.5f * vpH);
    };

    ImDrawList* dl = ImGui::GetBackgroundDrawList(ImGui::GetMainViewport());
    dl->PushClipRect(ImVec2(vpX, gMVY), ImVec2(vpX + vpW, gMVY + vpH), false);

    ImU32 colGrid  = IM_COL32(255, 255, 255,  35);
    ImU32 colAxisX = IM_COL32(255, 100, 100,  80);  // Rouge = X
    ImU32 colAxisY = IM_COL32(100, 255, 100,  80);  // Vert = Y

    float gridExtent = m_radius * 3.0f;
    // Variables for the label (must be defined in both branches)
    float xStart = 0.0f, hStart = 0.0f, hMax = 0.0f;

    if (m_camera->isOrbitMode()) {
        // ── Grille 3D type Blender ────────────────────────────────────────
        // Lines in the 3 directions: always visible from all angles
        float step = gridUnits;
        if (step < 1.0f) step = 1.0f;
        const int RANGE = 50;  // -50 to +50 lines = 101 lines
        float limit = RANGE * step;

        // ── Grid in the galaxy plane only (Z=0) ──────────────────────────
        // Helper: draws a 3D line with near-plane clipping
        auto line3dClip = [&](float x1, float y1, float z1, float x2, float y2, float z2, ImU32 col) {
            float cx1 = vp[0]*x1 + vp[4]*y1 + vp[8]*z1 + vp[12];
            float cy1 = vp[1]*x1 + vp[5]*y1 + vp[9]*z1 + vp[13];
            float cw1 = vp[3]*x1 + vp[7]*y1 + vp[11]*z1 + vp[15];

            float cx2 = vp[0]*x2 + vp[4]*y2 + vp[8]*z2 + vp[12];
            float cy2 = vp[1]*x2 + vp[5]*y2 + vp[9]*z2 + vp[13];
            float cw2 = vp[3]*x2 + vp[7]*y2 + vp[11]*z2 + vp[15];

            const float NEAR = 1e-3f;
            // If both points are behind the near plane, skip
            if (cw1 < NEAR && cw2 < NEAR) return;

            // If one point is behind, clip to the near plane
            if (cw1 < NEAR || cw2 < NEAR) {
                float t = (NEAR - cw1) / (cw2 - cw1 + 1e-9f);
                float xc = x1 + t*(x2-x1);
                float yc = y1 + t*(y2-y1);
                float zc = z1 + t*(z2-z1);
                if (cw1 < NEAR) { x1=xc; y1=yc; z1=zc; cx1 = vp[0]*x1+vp[4]*y1+vp[8]*z1+vp[12]; cy1 = vp[1]*x1+vp[5]*y1+vp[9]*z1+vp[13]; cw1=NEAR; }
                else            { x2=xc; y2=yc; z2=zc; cx2 = vp[0]*x2+vp[4]*y2+vp[8]*z2+vp[12]; cy2 = vp[1]*x2+vp[5]*y2+vp[9]*z2+vp[13]; cw2=NEAR; }
            }

            ImVec2 p1(vpX + (cx1/cw1 + 1.0f)*0.5f*vpW, gMVY + (1.0f - cy1/cw1)*0.5f*vpH);
            ImVec2 p2(vpX + (cx2/cw2 + 1.0f)*0.5f*vpW, gMVY + (1.0f - cy2/cw2)*0.5f*vpH);
            dl->AddLine(p1, p2, col, 1.0f);
        };

        // 2D grid in the Z=0 plane = galaxy plane
        // Same logic as the 2D mode: covers ±worldHalf * ext
        const float ext = 4.0f;
        float xMin = -worldHalfW * ext;
        float xMax =  worldHalfW * ext;
        float yMin = -worldHalfH * ext;
        float yMax =  worldHalfH * ext;

        float xStart = floorf(xMin / gridUnits) * gridUnits;
        float yStart = floorf(yMin / gridUnits) * gridUnits;

        // Vertical lines: X fixed, Y varies
        for (float x = xStart; x <= xMax + gridUnits*0.5f; x += gridUnits) {
            line3dClip(x, yMin, 0, x, yMax, 0, colGrid);
        }
        // Horizontal lines: Y fixed, X varies
        for (float y = yStart; y <= yMax + gridUnits*0.5f; y += gridUnits) {
            line3dClip(xMin, y, 0, xMax, y, 0, colGrid);
        }

        // Axes in the Z=0 plane (short, ~10% of the grid)
        float axisLen = fminf(worldHalfW, worldHalfH) * ext * 0.1f;
        line3dClip(0, 0, 0, axisLen, 0, 0, colAxisX);   // Rouge = X
        line3dClip(0, 0, 0, 0, axisLen, 0, colAxisY);   // Vert = Y
    } else {
        // ── Grille 2D standard (face-on / edge-on) ──────────────────────
        const float ext = 4.0f;
        float hOrig  = edgeOn ? 0.0f : cam_y;
        float xOrig  = edgeOn ? 0.0f : cam_x;
        float xMin   = xOrig - worldHalfW * ext;
        float xMax   = xOrig + worldHalfW * ext;
        float hMin   = hOrig - worldHalfH * ext;
        hMax         = hOrig + worldHalfH * ext;

        xStart = floorf(xMin / gridUnits) * gridUnits;
        hStart = floorf(hMin / gridUnits) * gridUnits;

        for (float x = xStart; x <= xMax + gridUnits * 0.5f; x += gridUnits)
        {
            ImVec2 p1 = w2s(x, hMin);
            ImVec2 p2 = w2s(x, hMax);
            ImU32  c  = (fabsf(x) < gridUnits * 0.01f) ? colAxisX : colGrid;
            dl->AddLine(p1, p2, c, 1.0f);
        }
        for (float h = hStart; h <= hMax + gridUnits * 0.5f; h += gridUnits)
        {
            ImVec2 p1 = w2s(xMin, h);
            ImVec2 p2 = w2s(xMax, h);
            ImU32  c  = (fabsf(h) < gridUnits * 0.01f) ? colAxisY : colGrid;
            dl->AddLine(p1, p2, c, 1.0f);
        }
    }

    // Fixed label at the top-left of the render window
    ImVec2 labelAnchor(vpX + 10, gMVY + 10);

    char label[64];
    snprintf(label, sizeof(label), "%.0f LY", gridLy);

    ImVec2 textSize = ImGui::CalcTextSize(label);
    dl->AddRectFilled(
        ImVec2(labelAnchor.x - 2, labelAnchor.y - 2),
        ImVec2(labelAnchor.x + textSize.x + 4, labelAnchor.y + textSize.y + 2),
        IM_COL32(0, 0, 0, 140));
    dl->AddText(labelAnchor, IM_COL32(220, 220, 100, 230), label);

    dl->PopClipRect();
}

// ─── Velocity Field / Streamlines (#10) ──────────────────────────────────────
// Viewport overlay: short arrows at a sampled subset of particles showing the
// local velocity. Reveals the Janus infall / runaway flow structure.
void App::renderVelocityField()
{
    if (!m_showVelocityField || !m_camera) return;
    if (m_positions.empty() || m_rotVelocities.size() < m_positions.size()) return;

    ImGuiViewport* gMV = ImGui::GetMainViewport();
    const float gMVX = gMV->Pos.x, gMVY = gMV->Pos.y;
    const float panelW = 340.0f;
    const float vpX = gMVX + panelW;
    const float vpW = (float)m_width - panelW;
    const float vpH = (float)m_height;
    if (vpW <= 0 || vpH <= 0) return;

    float vp[16];
    m_camera->getViewProjMatrix(vp);
    bool edgeOn = m_camera->isEdgeOn();

    // Project a 3D world point to screen; returns false if behind near plane.
    auto project = [&](float wx, float wy, float wz, ImVec2& out) -> bool {
        if (edgeOn) {
            // Edge-on uses an orthographic-style mapping (matches renderGrid)
            Vector* orig = m_camera->getOrigin();
            float camDist = fabsf((float)orig->y);
            float tanH = tanf(60.0f * 3.14159265f / 360.0f);
            float wHalfH = camDist * tanH;
            float wHalfW = wHalfH * (vpW / vpH);
            if (wHalfW < 1e-6f || wHalfH < 1e-6f) return false;
            out = ImVec2(vpX + (wx / wHalfW + 1.0f) * 0.5f * vpW,
                         gMVY + (1.0f - wz / wHalfH) * 0.5f * vpH);
            return true;
        }
        float cx = vp[0]*wx + vp[4]*wy + vp[8]*wz  + vp[12];
        float cy = vp[1]*wx + vp[5]*wy + vp[9]*wz  + vp[13];
        float cw = vp[3]*wx + vp[7]*wy + vp[11]*wz + vp[15];
        if (cw < 1e-3f) return false;
        out = ImVec2(vpX + (cx/cw + 1.0f) * 0.5f * vpW,
                     gMVY + (1.0f - cy/cw) * 0.5f * vpH);
        return true;
    };

    size_t n = m_positions.size();

    auto bulk = [&](bool isNeg, float& vcx, float& vcy, float& vcz) {
        if (!m_velFieldRelative) { vcx = vcy = vcz = 0.0f; return; }
        vcx = isNeg ? m_baryNegVx : m_baryPosVx;
        vcy = isNeg ? m_baryNegVy : m_baryPosVy;
        vcz = isNeg ? m_baryNegVz : m_baryPosVz;
    };

    // ── Capture pass: a freshly drawn box selects the SET OF STARS inside it now.
    // We store their indices; from then on the box disappears and those stars'
    // arrows follow them as they move. Indices are stable (global array order).
    if (m_velSelPending) {
        float bxMin = fminf(m_velSelX0, m_velSelX1), bxMax = fmaxf(m_velSelX0, m_velSelX1);
        float byMin = fminf(m_velSelY0, m_velSelY1), byMax = fmaxf(m_velSelY0, m_velSelY1);
        std::vector<int> hits;
        hits.reserve(8192);
        for (size_t i = 0; i < n; i++) {
            const cl_float4& p = m_positions[i];
            if (p.w == 0.0f) continue;
            if (p.w < 0.0f && !m_showNegaMass) continue;
            ImVec2 s0;
            if (!project(p.x, p.y, p.z, s0)) continue;
            float wx = s0.x - gMVX, wy = s0.y - gMVY;
            if (wx >= bxMin && wx <= bxMax && wy >= byMin && wy <= byMax)
                hits.push_back((int)i);
        }
        // Cap to MAX_ARROWS by uniform sub-sampling
        const size_t CAP = 3000;
        m_velSelIndices.clear();
        if (hits.size() <= CAP) {
            m_velSelIndices = std::move(hits);
        } else {
            size_t st = hits.size() / CAP;
            for (size_t k = 0; k < hits.size(); k += st) m_velSelIndices.push_back(hits[k]);
        }
        m_velSelActive  = !m_velSelIndices.empty();
        m_velSelPending = false;
    }

    bool useSel = m_velSelActive && !m_velSelIndices.empty();

    ImDrawList* dl = ImGui::GetBackgroundDrawList(gMV);
    dl->PushClipRect(ImVec2(vpX, gMVY), ImVec2(vpX + vpW, gMVY + vpH), false);

    // Reference speed (over the selected set) so arrow length scales to the regime.
    float speedRef = 0.0f; int sc = 0;
    if (useSel) {
        for (int i : m_velSelIndices) {
            if (i < 0 || i >= (int)m_rotVelocities.size()) continue;
            const cl_float4& v = m_rotVelocities[i];
            bool isNeg = m_positions[i].w < 0.0f;
            float vcx, vcy, vcz; bulk(isNeg, vcx, vcy, vcz);
            float dvx=v.x-vcx, dvy=v.y-vcy, dvz=v.z-vcz;
            speedRef += sqrtf(dvx*dvx+dvy*dvy+dvz*dvz); sc++;
        }
    }
    speedRef = (sc > 0 && speedRef > 1e-12f) ? speedRef / sc : 1.0f;

    float arrowWorldLen = m_radius * 0.03f * m_velFieldScale;
    float velToWorld = arrowWorldLen / speedRef;

    auto drawArrow = [&](int i) {
        const cl_float4& p = m_positions[i];
        if (p.w == 0.0f) return;
        bool isNeg = p.w < 0.0f;
        if (isNeg && !m_showNegaMass) return;
        const cl_float4& v = m_rotVelocities[i];
        float vcx, vcy, vcz; bulk(isNeg, vcx, vcy, vcz);
        float dvx = v.x - vcx, dvy = v.y - vcy, dvz = v.z - vcz;
        float sp = sqrtf(dvx*dvx + dvy*dvy + dvz*dvz);
        if (sp < 1e-12f) return;
        ImVec2 s0, s1;
        if (!project(p.x, p.y, p.z, s0)) return;
        if (!project(p.x + dvx*velToWorld, p.y + dvy*velToWorld, p.z + dvz*velToWorld, s1)) return;

        // Red palette: + = bright red-orange, − = amber/yellow (both warm/visible)
        float t = sp / (speedRef * 2.0f); if (t > 1.0f) t = 1.0f;
        int a = 110 + (int)(145.0f * t);
        ImU32 col = isNeg ? IM_COL32(255, 200, 40, a) : IM_COL32(255, 60, 30, a);
        dl->AddLine(s0, s1, col, 1.4f);
        float hx = s1.x - s0.x, hy = s1.y - s0.y;
        float hl = sqrtf(hx*hx + hy*hy);
        if (hl > 4.0f) {
            float ux = hx / hl, uy = hy / hl, ah = 4.0f;
            dl->AddLine(s1, ImVec2(s1.x - ah*(ux + 0.5f*uy), s1.y - ah*(uy - 0.5f*ux)), col, 1.4f);
            dl->AddLine(s1, ImVec2(s1.x - ah*(ux - 0.5f*uy), s1.y - ah*(uy + 0.5f*ux)), col, 1.4f);
        }
    };

    // Only draw streams for an explicit selection — never the whole galaxy
    // (avoids the clutter inside dense clumps the user flagged).
    if (useSel) {
        for (int i : m_velSelIndices)
            if (i >= 0 && i < (int)n && i < (int)m_rotVelocities.size()) drawArrow(i);
    }

    // Selection rectangle: live only WHILE dragging, then it vanishes.
    if (m_velSelDragging) {
        float rx0 = gMVX + fminf(m_velSelX0, m_velSelX1);
        float ry0 = gMVY + fminf(m_velSelY0, m_velSelY1);
        float rx1 = gMVX + fmaxf(m_velSelX0, m_velSelX1);
        float ry1 = gMVY + fmaxf(m_velSelY0, m_velSelY1);
        dl->AddRect(ImVec2(rx0, ry0), ImVec2(rx1, ry1), IM_COL32(255, 230, 80, 220), 0.0f, 0, 1.5f);
        dl->AddRectFilled(ImVec2(rx0, ry0), ImVec2(rx1, ry1), IM_COL32(255, 230, 80, 25));
    }

    // Label
    char lbl[96];
    if (useSel)
        snprintf(lbl, sizeof(lbl), "Velocity field %s  [tracking %zu stars]",
                 m_velFieldRelative ? "(relative to COM)" : "(absolute)", m_velSelIndices.size());
    else
        snprintf(lbl, sizeof(lbl), "Velocity field — drag a box to pick stars");
    ImVec2 anchor(vpX + 10, gMVY + 30);
    ImVec2 ts = ImGui::CalcTextSize(lbl);
    dl->AddRectFilled(ImVec2(anchor.x-2, anchor.y-2), ImVec2(anchor.x+ts.x+4, anchor.y+ts.y+2), IM_COL32(0,0,0,140));
    dl->AddText(anchor, IM_COL32(255,180,80,230), lbl);

    dl->PopClipRect();
}

// ─── Barycenters ─────────────────────────────────────────────────────────────
void App::renderBarycenters()
{
    if (!m_showBarycenters) return;
    ImGui::SetNextWindowSize(ImVec2(620, 460), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(360, 30),  ImGuiCond_FirstUseEver);
    ImGui::Begin("Barycenters", &m_showBarycenters,
        ImGuiWindowFlags_NoCollapse);

    float curDist = (m_baryCount > 0)
        ? m_baryDist[(m_baryHead - 1 + BARY_HISTORY) % BARY_HISTORY] : 0.f;
    ImGui::Text("G+ (%.1f, %.1f) | G- (%.1f, %.1f) | |G+-G-| = %.1f LY",
        m_baryPosX * m_lyPerUnit, m_baryPosY * m_lyPerUnit,
        m_baryNegX * m_lyPerUnit, m_baryNegY * m_lyPerUnit, curDist);
    ImGui::Separator();

    if (m_baryCount < 2) {
        ImGui::TextDisabled("Insufficient data — enable Auto-Refresh or click Refresh Analytics");
        ImGui::End();
        return;
    }

    ImVec2 canvasPos  = ImGui::GetCursorScreenPos();
    ImVec2 canvasSize = ImGui::GetContentRegionAvail();
    if (canvasSize.x < 100) canvasSize.x = 100;
    if (canvasSize.y < 100) canvasSize.y = 100;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(canvasPos,
        ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
        IM_COL32(8, 16, 48, 255));
    dl->AddRect(canvasPos,
        ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
        IM_COL32(100, 100, 100, 200));

    const float marginL = 70.0f, marginR = 10.0f, marginT = 10.0f, marginB = 32.0f;
    float plotW = canvasSize.x - marginL - marginR;
    float plotH = canvasSize.y - marginT - marginB;
    float px = canvasPos.x + marginL;
    float py = canvasPos.y + marginT;

    float tMax = 0.0f, dMax = 1e-6f;
    for (int i = 0; i < m_baryCount; i++) {
        int idx = (m_baryHead - m_baryCount + i + BARY_HISTORY) % BARY_HISTORY;
        if (m_baryTimes [idx] > tMax) tMax = m_baryTimes [idx];
        if (m_baryDist  [idx] > dMax) dMax = m_baryDist  [idx];
        if (m_baryPosMag[idx] > dMax) dMax = m_baryPosMag[idx];
        if (m_baryNegMag[idx] > dMax) dMax = m_baryNegMag[idx];
    }

    ImU32 colGrid = IM_COL32(50, 50, 50, 200);
    for (int i = 1; i <= 4; i++) {
        float xg = px + plotW * i / 4.0f;
        float yg = py + plotH * i / 4.0f;
        dl->AddLine(ImVec2(xg, py), ImVec2(xg, py + plotH), colGrid);
        dl->AddLine(ImVec2(px, yg), ImVec2(px + plotW, yg), colGrid);
    }
    ImU32 colAxis = IM_COL32(180, 180, 180, 255);
    dl->AddLine(ImVec2(px, py + plotH), ImVec2(px + plotW, py + plotH), colAxis);
    dl->AddLine(ImVec2(px, py), ImVec2(px, py + plotH), colAxis);

    char buf[32];
    snprintf(buf, sizeof(buf), "%.0f LY", dMax);
    { ImVec2 ts = ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px - ts.x - 2, py), colAxis, buf); }
    { ImVec2 ts = ImGui::CalcTextSize("0"); dl->AddText(ImVec2(px - ts.x - 2, py + plotH - 12), colAxis, "0"); }
    snprintf(buf, sizeof(buf), "%.0f MY", tMax);
    { ImVec2 ts = ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px + plotW - ts.x, py + plotH + 2), colAxis, buf); }
    dl->AddText(ImVec2(px, py + plotH + 2), colAxis, "0");
    { ImVec2 ts = ImGui::CalcTextSize("Distance (LY)"); dl->AddText(ImVec2(px + (plotW - ts.x) * 0.5f, py + plotH + 16), colAxis, "Distance (LY)"); }

    // Linear copy + Gaussian smoothing for all 3 series
    std::vector<float> visD(m_baryCount), visP(m_baryCount), visN(m_baryCount);
    for (int i = 0; i < m_baryCount; i++) {
        int idx = (m_baryHead - m_baryCount + i + BARY_HISTORY) % BARY_HISTORY;
        visD[i] = m_baryDist  [idx];
        visP[i] = m_baryPosMag[idx];
        visN[i] = m_baryNegMag[idx];
    }
    int sigBb = std::max(3, std::min(30, m_baryCount / 20));
    int halfKb = sigBb * 3;
    auto smooth = [&](const std::vector<float>& v) {
        std::vector<float> s(v.size());
        for (int i = 0; i < (int)v.size(); i++) {
            double sumW=0.0, sumWV=0.0;
            for (int k=-halfKb;k<=halfKb;k++) {
                int j=i+k; if (j<0||j>=(int)v.size()) continue;
                double w = exp(-0.5*(double)(k*k)/(double)(sigBb*sigBb));
                sumW += w; sumWV += w * v[j];
            }
            s[i] = (sumW>0.0) ? (float)(sumWV/sumW) : v[i];
        }
        return s;
    };
    std::vector<float> smD = smooth(visD), smP = smooth(visP), smN = smooth(visN);

    dl->PushClipRect(ImVec2(px, py), ImVec2(px+plotW, py+plotH), true);
    auto drawCurve = [&](const std::vector<float>& s, ImU32 col, float thickness) {
        ImVec2 prev = ImVec2(0,0); bool first = true;
        for (int i = 0; i < (int)s.size(); i++) {
            float sx = px + (float)i / (float)(s.size() > 1 ? s.size() - 1 : 1) * plotW;
            float sy = py + (1.0f - s[i] / dMax) * plotH;
            ImVec2 cur = ImVec2(sx, sy);
            if (!first) dl->AddLine(prev, cur, col, thickness);
            prev = cur; first = false;
        }
    };
    ImU32 colPos  = IM_COL32(100, 220, 255, 220);  // cyan = G+ drift
    ImU32 colNeg  = IM_COL32(255, 100, 255, 220);  // fuchsia = G- drift
    ImU32 colDist = IM_COL32(255, 200,  50, 230);  // amber = |G+ - G-| (Janus offset)
    drawCurve(smP, colPos,  1.6f);
    drawCurve(smN, colNeg,  1.6f);
    drawCurve(smD, colDist, 2.2f);
    dl->PopClipRect();

    // Legend
    {
        float lx = px + 8, ly = py + 6;
        float lw = 150, lh = 50;
        dl->AddRectFilled(ImVec2(lx, ly), ImVec2(lx + lw, ly + lh), IM_COL32(0,0,0,150));
        dl->AddLine(ImVec2(lx + 6, ly + 10), ImVec2(lx + 24, ly + 10), colPos, 2.f);
        dl->AddText(ImVec2(lx + 28, ly + 3), colPos,  "|G+|  drift +");
        dl->AddLine(ImVec2(lx + 6, ly + 24), ImVec2(lx + 24, ly + 24), colNeg, 2.f);
        dl->AddText(ImVec2(lx + 28, ly + 17), colNeg, "|G-|  drift -");
        dl->AddLine(ImVec2(lx + 6, ly + 38), ImVec2(lx + 24, ly + 38), colDist, 2.5f);
        dl->AddText(ImVec2(lx + 28, ly + 31), colDist, "|G+ - G-|  offset");
    }

    ImGui::Dummy(canvasSize);
    ImGui::End();
}

// ─── Fourier Modes ────────────────────────────────────────────────────────────
void App::renderFourierModes()
{
    if (!m_showFourier) return;
    ImGui::SetNextWindowSize(ImVec2(620, 460), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(360, 30),  ImGuiCond_FirstUseEver);
    ImGui::Begin("Fourier Modes", &m_showFourier,
        ImGuiWindowFlags_NoCollapse);

    if (ImGui::Button("Refresh") && m_simulation && !m_positions.empty())
        refreshAnalytics();

    ImGui::SameLine();
    ImGui::TextDisabled("A_m = |sum exp(i*m*theta)| / N");
    if (m_fourierCount < 2) {
        ImGui::TextDisabled("Insufficient data — enable Auto-Refresh or click Refresh");
        ImGui::End();
        return;
    }

    int hi = (m_fourierHead-1+FOURIER_HISTORY)%FOURIER_HISTORY;
    ImGui::Text("A1=%.3f  A2=%.3f  A3=%.3f  A4=%.3f  (t=%.1f MY)",
        m_fourierA[0][hi], m_fourierA[1][hi], m_fourierA[2][hi], m_fourierA[3][hi],
        m_fourierTimes[hi]);
    ImGui::Separator();

    ImVec2 canvasPos  = ImGui::GetCursorScreenPos();
    ImVec2 canvasSize = ImGui::GetContentRegionAvail();
    if (canvasSize.x < 100) canvasSize.x = 100;
    if (canvasSize.y < 100) canvasSize.y = 100;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(canvasPos,
        ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
        IM_COL32(8, 16, 48, 255));
    dl->AddRect(canvasPos,
        ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
        IM_COL32(100, 100, 100, 200));

    const float marginL = 70.0f, marginR = 10.0f, marginT = 10.0f, marginB = 32.0f;
    float plotW = canvasSize.x - marginL - marginR;
    float plotH = canvasSize.y - marginT - marginB;
    float px = canvasPos.x + marginL;
    float py = canvasPos.y + marginT;

    float tMax = 0.0f, aMax = 0.01f;
    for (int i = 0; i < m_fourierCount; i++) {
        int idx = (m_fourierHead - m_fourierCount + i + FOURIER_HISTORY) % FOURIER_HISTORY;
        if (m_fourierTimes[idx] > tMax) tMax = m_fourierTimes[idx];
        for (int m = 0; m < 4; m++)
            if (m_fourierA[m][idx] > aMax) aMax = m_fourierA[m][idx];
    }

    ImU32 colGrid = IM_COL32(50, 50, 50, 200);
    for (int i = 1; i <= 4; i++) {
        float xg = px + plotW * i / 4.0f;
        float yg = py + plotH * i / 4.0f;
        dl->AddLine(ImVec2(xg, py), ImVec2(xg, py + plotH), colGrid);
        dl->AddLine(ImVec2(px, yg), ImVec2(px + plotW, yg), colGrid);
    }
    ImU32 colAxis = IM_COL32(180, 180, 180, 255);
    dl->AddLine(ImVec2(px, py + plotH), ImVec2(px + plotW, py + plotH), colAxis);
    dl->AddLine(ImVec2(px, py), ImVec2(px, py + plotH), colAxis);

    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f", aMax);
    { ImVec2 ts = ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px - ts.x - 2, py), colAxis, buf); }
    { ImVec2 ts = ImGui::CalcTextSize("0"); dl->AddText(ImVec2(px - ts.x - 2, py + plotH - 12), colAxis, "0"); }
    snprintf(buf, sizeof(buf), "%.0f MY", tMax);
    { ImVec2 ts = ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px + plotW - ts.x, py + plotH + 2), colAxis, buf); }
    dl->AddText(ImVec2(px, py + plotH + 2), colAxis, "0");

    const ImU32 mCols[4] = {
        IM_COL32(255, 100, 100, 220),
        IM_COL32(100, 220, 255, 220),
        IM_COL32(100, 255, 100, 220),
        IM_COL32(255, 200, 50, 220)
    };
    const char* mNames[4] = {"m=1", "m=2", "m=3", "m=4"};

    int sigBf = std::max(3, std::min(30, m_fourierCount / 20));
    int halfKf = sigBf * 3;
    dl->PushClipRect(ImVec2(px, py), ImVec2(px+plotW, py+plotH), true);
    for (int m = 0; m < 4; m++) {
        // Gather + Gaussian smooth
        std::vector<float> visA(m_fourierCount);
        for (int i = 0; i < m_fourierCount; i++) {
            int idx = (m_fourierHead - m_fourierCount + i + FOURIER_HISTORY) % FOURIER_HISTORY;
            visA[i] = m_fourierA[m][idx];
        }
        std::vector<float> smA(m_fourierCount);
        for (int i = 0; i < m_fourierCount; i++) {
            double sumW=0.0, sumWV=0.0;
            for (int k=-halfKf;k<=halfKf;k++) {
                int j=i+k; if (j<0||j>=m_fourierCount) continue;
                double w = exp(-0.5*(double)(k*k)/(double)(sigBf*sigBf));
                sumW += w; sumWV += w * visA[j];
            }
            smA[i] = (sumW>0.0) ? (float)(sumWV/sumW) : visA[i];
        }
        ImVec2 prev = ImVec2(0,0);
        bool first = true;
        for (int i = 0; i < m_fourierCount; i++) {
            float sx = px + (float)i / (float)(m_fourierCount > 1 ? m_fourierCount - 1 : 1) * plotW;
            float sy = py + (1.0f - smA[i] / aMax) * plotH;
            ImVec2 cur = ImVec2(sx, sy);
            if (!first) dl->AddLine(prev, cur, mCols[m], 1.8f);
            prev = cur;
            first = false;
        }
        float lx = px + 4 + m * (plotW * 0.23f);
        dl->AddLine(ImVec2(lx, py + 10), ImVec2(lx + 18, py + 10), mCols[m], 2.0f);
        dl->AddText(ImVec2(lx + 20, py + 4), mCols[m], mNames[m]);
    }
    dl->PopClipRect();

    ImGui::Dummy(canvasSize);
    ImGui::End();
}

// ─── Equation Editor ─────────────────────────────────────────────────────────
void App::renderEquationEditor()
{
    if (!m_showEquationEditor) return;

    ImGui::SetNextWindowSize(ImVec2(640, 600), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(360, 30), ImGuiCond_FirstUseEver);
    ImGui::Begin("Equation Editor", &m_showEquationEditor,
        ImGuiWindowFlags_NoCollapse);

    static const char* modelNames[] = {
        "Newton (standard gravity)",
        "Janus (signed mass gravity)",
        "Yukawa (finite-range force)",
        "MOND (modified dynamics)",
        "Custom (user-defined)"
    };
    static const char* modelFiles[] = {
        "equations/newton.cl",
        "equations/janus.cl",
        "equations/yukawa.cl",
        "equations/mond.cl",
        "equations/custom.cl"
    };
    static const char* modelDesc[] = {
        "F = G*m/r^2  |  Standard inverse-square law",
        "F = sgn(m_i)*G*m/r^2  |  Same sign: attract  |  Opposite sign: repel",
        "F = G*m/r^2 * (1+alpha*exp(-r/lambda))  |  Short-range correction",
        "F = G*m/r^2 * mu(a/a0)  |  Modified Newtonian Dynamics",
        "User-defined equation  |  Edit equations/custom.cl"
    };

    // ── Model Selection ─────────────────────────────────────────────────────
    ImGui::TextDisabled("Physical Model");
    int prevModel = m_eqSelectedModel;
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo("##eqmodel", modelNames[m_eqSelectedModel]))
    {
        for (int i = 0; i < 5; i++)
        {
            bool isSelected = (m_eqSelectedModel == i);
            if (ImGui::Selectable(modelNames[i], isSelected))
                m_eqSelectedModel = i;
            if (isSelected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::TextDisabled("%s", modelDesc[m_eqSelectedModel]);

    // Model changed: apply immediately to GPU kernel (no restart needed)
    if (m_eqSelectedModel != prevModel)
    {
        m_eqCodeBuffer[0] = '\0';
        if (m_simulation)
            m_simulation->setPhysicsModel(m_eqSelectedModel);
        // Janus is the only model with physically consistent negative masses.
        bool janusNow = (m_eqSelectedModel == 1);
        if (!janusNow && m_showNegaMass) {
            m_showNegaMass = false;
            uploadParticlesToGPU();
        } else if (janusNow && !m_showNegaMass) {
            m_showNegaMass = true;
            uploadParticlesToGPU();
        }
    }

    ImGui::Separator();

    // ── Backend Information ────────────────────────────────────────
    ImVec2 infoPos = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(
        ImVec2(infoPos.x, infoPos.y),
        ImVec2(infoPos.x + ImGui::GetContentRegionAvail().x, infoPos.y + 68),
        IM_COL32(20, 40, 80, 255));
    dl->AddRect(
        ImVec2(infoPos.x, infoPos.y),
        ImVec2(infoPos.x + ImGui::GetContentRegionAvail().x, infoPos.y + 68),
        IM_COL32(80, 120, 200, 200));

    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4);
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "  i  Information");
    ImGui::TextDisabled("  This file contains the physical equations that run on the GPU.");
    ImGui::TextDisabled("  Language: OpenCL C (similar to standard C with GPU functions).");
    ImGui::Dummy(ImVec2(0, 4));

    // ── File Sections ────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("File Structure: pair3.cl", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(8.0f);
        
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "1. pair_force()");
        ImGui::TextDisabled("   Pair force: F_ij = G * m_j * r_ij / |r|^3");
        ImGui::TextDisabled("   This is where you modify the gravity law.");
        ImGui::Dummy(ImVec2(0, 2));

        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "2. nbody()");
        ImGui::TextDisabled("   Main kernel: loops over all pairs.");
        ImGui::TextDisabled("   Integrates force, spiral waves, and the integrator.");
        ImGui::Dummy(ImVec2(0, 2));

        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "3. Uniform Parameters");
        ImGui::TextDisabled("   softening, timestep, G, sp_amplitude, etc.");
        ImGui::TextDisabled("   These values are sent from the interface (Controls).");
        ImGui::Dummy(ImVec2(0, 2));

        ImGui::Unindent(8.0f);
    }

    // ── Code Preview ────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Template Preview (read-only)", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Load template content if buffer is empty
        if (m_eqCodeBuffer[0] == '\0')
        {
            std::ifstream f(modelFiles[m_eqSelectedModel]);
            if (f.is_open())
            {
                std::string content((std::istreambuf_iterator<char>(f)),
                                    std::istreambuf_iterator<char>());
                if (content.size() >= (size_t)EQ_CODE_SIZE - 1)
                    content.resize(EQ_CODE_SIZE - 1);
                std::strncpy(m_eqCodeBuffer, content.c_str(), EQ_CODE_SIZE - 1);
                m_eqCodeBuffer[EQ_CODE_SIZE - 1] = '\0';
            }
            else
            {
                std::strcpy(m_eqCodeBuffer, "// Template file not found.\n// Check the equations/ folder.");
            }
        }

        ImGui::InputTextMultiline("##eqcode", m_eqCodeBuffer, EQ_CODE_SIZE,
            ImVec2(-1, 220),
            ImGuiInputTextFlags_ReadOnly);
    }

    // ── Actions ────────────────────────────────────────────────────────────
    ImGui::Separator();
    
    // ── Workflow ───────────────────────────────────────────────────────────
    ImVec2 warnPos = ImGui::GetCursorScreenPos();
    dl->AddRectFilled(
        ImVec2(warnPos.x, warnPos.y),
        ImVec2(warnPos.x + ImGui::GetContentRegionAvail().x, warnPos.y + 90),
        IM_COL32(20, 50, 30, 255));
    dl->AddRect(
        ImVec2(warnPos.x, warnPos.y),
        ImVec2(warnPos.x + ImGui::GetContentRegionAvail().x, warnPos.y + 90),
        IM_COL32(80, 180, 100, 200));

    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4);
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "  Workflow: modify equations");
    ImGui::TextDisabled("  1. Select a model from the dropdown above");
    ImGui::TextDisabled("  2. The model switches instantly on the GPU (no restart needed)");
    ImGui::TextDisabled("  3. Click 'Open Template' to view or edit the equation file");
    ImGui::TextDisabled("  4. Click 'Apply Model' to confirm the active model");
    ImGui::Dummy(ImVec2(0, 4));

    float btnW = (ImGui::GetContentRegionAvail().x - 8) * 0.5f;
    if (ImGui::Button("Open Template", ImVec2(btnW, 26)))
    {
        std::string cmd = "notepad.exe \"";
        cmd += modelFiles[m_eqSelectedModel];
        cmd += "\"";
        std::system(cmd.c_str());
    }
    ImGui::SameLine(0, 8);
    if (ImGui::Button("Apply Model", ImVec2(btnW, 26)))
    {
        if (m_simulation)
        {
            m_simulation->setPhysicsModel(m_eqSelectedModel);
            m_eqCompileOk = true;
            if (m_eqSelectedModel == 4)
                std::strcpy(m_eqCompileMsg,
                    "Custom model active — to modify: edit gpu-nbody/pair3.cu (case 4) and recompile");
            else
                std::strcpy(m_eqCompileMsg, "Model active on GPU (runtime switch)");
        }
        else
        {
            m_eqCompileOk = false;
            std::strcpy(m_eqCompileMsg, "Error: simulation not initialized");
        }
    }

    if (m_eqCompileMsg[0] != '\0')
    {
        ImVec4 msgColor = m_eqCompileOk ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f) : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
        ImGui::TextColored(msgColor, "%s", m_eqCompileMsg);
    }

    if (ImGui::Button("Reload Preview", ImVec2(-1, 24)))
    {
        m_eqCodeBuffer[0] = '\0';
    }

    ImGui::End();
}

// ─── Dark Matter Equivalent Profile ──────────────────────────────────────────
void App::renderDarkMatterProfile()
{
    if (!m_showDarkMatter) return;
    ImGui::SetNextWindowSize(ImVec2(620, 480), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(360, 30),  ImGuiCond_FirstUseEver);
    ImGui::Begin("Dark Matter Equivalent", &m_showDarkMatter, ImGuiWindowFlags_NoCollapse);

    ImGui::TextDisabled("Negative density |rho-(r)| = negative mass per spherical shell vs NFW profile");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90); ImGui::SliderFloat("r_s##nfw",&m_nfwRs,10.f,1000.f,"%.0f");
    ImGui::SameLine();
    ImGui::TextDisabled("(r_s = NFW scale radius)");
    static bool dmLog = true;
    ImGui::Checkbox("Log scale (Y)", &dmLog);

    if (m_positions.empty()) { ImGui::TextDisabled("No data"); ImGui::End(); return; }

    const int BINS=60;

    // --- Adaptive radial extent (recentered on COM−) ----------------------
    // The negative cloud drifts and expands; a fixed rMax pushes the profile
    // off the right edge over time. Size the X-axis to the 97%-enclosing
    // radius of the negative mass so the curve always fills the plot.
    float rMaxNeg = 0.f;
    for (const auto& p : m_positions) {
        if (p.w >= 0.f) continue;
        float dx=p.x-m_baryNegX, dy=p.y-m_baryNegY, dz=p.z-m_baryNegZ;
        float r=sqrtf(dx*dx+dy*dy+dz*dz);
        if (r>rMaxNeg) rMaxNeg=r;
    }
    if (rMaxNeg < 1e-6f) rMaxNeg = m_radius * 3.0f;
    {
        const int CO=256;
        std::vector<double> coMass(CO,0.0); double totM=0.0;
        for (const auto& p : m_positions) {
            if (p.w >= 0.f) continue;
            float dx=p.x-m_baryNegX, dy=p.y-m_baryNegY, dz=p.z-m_baryNegZ;
            float r=sqrtf(dx*dx+dy*dy+dz*dz);
            int c=(int)(r/rMaxNeg*CO); if(c>=CO)c=CO-1; if(c<0)c=0;
            coMass[c]+=(-p.w); totM+=(-p.w);
        }
        if (totM>0.0) {
            double cum=0.0;
            for(int c=0;c<CO;c++){ cum+=coMass[c]; if(cum>=0.97*totM){ rMaxNeg=(float)(c+1)/CO*rMaxNeg; break; } }
        }
    }
    float rMax = rMaxNeg * 1.10f;
    if (rMax < m_radius*0.2f) rMax = m_radius*0.2f;
    // ----------------------------------------------------------------------

    float rhoNeg[BINS]={};
    float massNeg[BINS]={};
    int   cntBin[BINS]={};

    // Per-species recentering: negative density around COM−
    // (post-merger the two species can be spatially offset)
    for (const auto& p : m_positions) {
        if (p.w >= 0.f) continue;
        float dx = p.x - m_baryNegX, dy = p.y - m_baryNegY, dz = p.z - m_baryNegZ;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        if (r >= rMax) continue;
        int b = (int)(r / rMax * BINS);
        if (b >= BINS) b = BINS-1;
        massNeg[b] -= p.w; cntBin[b]++;
    }
    float rhoMax = 1e-10f;
    for (int b=0;b<BINS;b++) {
        float rIn  = (float)b/BINS * rMax;
        float rOut = (float)(b+1)/BINS * rMax;
        float vol  = (4.f/3.f) * 3.14159265f * (rOut*rOut*rOut - rIn*rIn*rIn);
        if (vol > 1e-20f) rhoNeg[b] = massNeg[b] / vol;
        if (rhoNeg[b] > rhoMax) rhoMax = rhoNeg[b];
    }
    m_nfwRho0 = rhoMax;

    ImVec2 canv=ImGui::GetCursorScreenPos();
    ImVec2 sz=ImGui::GetContentRegionAvail();
    if (sz.x<80) sz.x=80; if (sz.y<80) sz.y=80;
    ImDrawList* dl=ImGui::GetWindowDrawList();
    dl->AddRectFilled(canv,ImVec2(canv.x+sz.x,canv.y+sz.y),IM_COL32(8,16,48,255));
    dl->AddRect(canv,ImVec2(canv.x+sz.x,canv.y+sz.y),IM_COL32(80,80,80,255));

    const float mL=70,mR=10,mT=10,mB=32;
    float pw=sz.x-mL-mR, ph=sz.y-mT-mB;
    float ppx=canv.x+mL, ppy=canv.y+mT;

    ImU32 cg=IM_COL32(40,40,40,255), ca=IM_COL32(160,160,160,255);
    for (int i=1;i<=4;i++){float xg=ppx+pw*i/4.f; dl->AddLine(ImVec2(xg,ppy),ImVec2(xg,ppy+ph),cg);}
    for (int i=1;i<=4;i++){float yg=ppy+ph*i/4.f; dl->AddLine(ImVec2(ppx,yg),ImVec2(ppx+pw,yg),cg);}
    dl->AddLine(ImVec2(ppx,ppy+ph),ImVec2(ppx+pw,ppy+ph),ca);
    dl->AddLine(ImVec2(ppx,ppy),ImVec2(ppx,ppy+ph),ca);

    // Gaussian smoothing of rhoNeg
    float smRhoNeg[BINS]={};
    {
        const int sigB = 2, halfK = 6;
        for (int b = 0; b < BINS; b++) {
            double sumW = 0.0, sumWV = 0.0;
            for (int k = -halfK; k <= halfK; k++) {
                int j = b + k; if (j < 0 || j >= BINS) continue;
                double w = exp(-0.5 * (double)(k*k) / (double)(sigB*sigB));
                sumW += w; sumWV += w * rhoNeg[j];
            }
            smRhoNeg[b] = (sumW > 0.0) ? (float)(sumWV / sumW) : rhoNeg[b];
        }
    }

    // Y-axis scaling from the ACTUALLY PLOTTED (smoothed) curve, not the raw peak.
    // The negative halo collapses to a central spike in Janus → the smoothed peak
    // is far below the raw peak, which used to squash the curve to the bottom.
    float smMax = 1e-30f, smMin = 1e30f;
    for (int b=0;b<BINS;b++) {
        if (smRhoNeg[b] > smMax) smMax = smRhoNeg[b];
        if (smRhoNeg[b] > 0.0f && smRhoNeg[b] < smMin) smMin = smRhoNeg[b];
    }
    if (smMin >= smMax) smMin = smMax * 1e-3f;
    m_nfwRho0 = smMax;   // anchor NFW reference to the same scale

    // yMap: maps a density value to a screen y (linear or log10)
    float logHi = log10f(smMax + 1e-30f), logLo = log10f(smMin + 1e-30f);
    if (logHi - logLo < 1e-3f) logLo = logHi - 3.0f;   // ensure a visible range
    auto yMap = [&](float v)->float {
        float t;
        if (dmLog) {
            float lv = log10f(fmaxf(v, 1e-30f));
            t = (lv - logLo) / (logHi - logLo);
        } else {
            t = v / smMax;
        }
        if (t < 0.f) t = 0.f; if (t > 1.f) t = 1.f;
        return ppy + (1.f - t) * ph;
    };

    dl->PushClipRect(ImVec2(ppx, ppy), ImVec2(ppx+pw, ppy+ph), true);
    ImVec2 prev=ImVec2(0,0); bool first=true;
    for (int b=0;b<BINS;b++) {
        float rMid=(b+0.5f)/BINS*rMax;
        float sx=ppx+rMid/rMax*pw;
        float sy=yMap(smRhoNeg[b]);
        ImVec2 cur=ImVec2(sx,sy);
        if (!first) dl->AddLine(prev,cur,IM_COL32(80,180,255,220),1.8f);
        prev=cur; first=false;
    }
    dl->PopClipRect();

    dl->PushClipRect(ImVec2(ppx, ppy), ImVec2(ppx+pw, ppy+ph), true);
    prev=ImVec2(0,0); first=true;
    for (int b=0;b<BINS;b++) {
        float rMid=(b+0.5f)/BINS*rMax;
        float x = rMid / m_nfwRs;
        if (x < 0.01f) x = 0.01f;
        float rhoNFW = m_nfwRho0 / (x * (1.f+x) * (1.f+x));
        float sx=ppx+rMid/rMax*pw;
        float sy=yMap(rhoNFW);
        ImVec2 cur=ImVec2(sx,sy);
        if (!first) dl->AddLine(prev,cur,IM_COL32(255,160,50,220),1.5f);
        prev=cur; first=false;
    }
    dl->PopClipRect();

    dl->AddLine(ImVec2(ppx+8,ppy+14),ImVec2(ppx+28,ppy+14),IM_COL32(80,180,255,220),2.f);
    dl->AddText(ImVec2(ppx+30,ppy+7),ca,"|rho_neg(r)|");
    dl->AddLine(ImVec2(ppx+8,ppy+28),ImVec2(ppx+28,ppy+28),IM_COL32(255,160,50,220),2.f);
    dl->AddText(ImVec2(ppx+30,ppy+21),ca,"NFW fit");

    char buf[32];
    snprintf(buf,sizeof(buf),"%.2e",smMax);
    { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(ppx-ts.x-2,ppy),ca,buf); }
    if (dmLog) snprintf(buf,sizeof(buf),"%.2e",smMin); else snprintf(buf,sizeof(buf),"0");
    { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(ppx-ts.x-2,ppy+ph-12),ca,buf); }
    snprintf(buf,sizeof(buf),"%.0f LY",rMax*m_lyPerUnit);
    { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(ppx+pw-ts.x,ppy+ph+2),ca,buf); }
    dl->AddText(ImVec2(ppx,ppy+ph+2),ca,"0");

    ImGui::Dummy(sz);
    ImGui::End();
}

int App::pickParticleCPU(int screenX, int screenY)
{
    if (m_positions.empty() || !m_camera) return -1;

    float vp[16];
    m_camera->getViewProjMatrix(vp);

    const float panelW = 340.0f;
    const float vpW = (float)m_width - panelW;
    const float vpH = (float)m_height;
    if (vpW <= 0 || vpH <= 0) return -1;

    float margin = 50.0f;
    float minSX = -margin, maxSX = vpW + margin;
    float minSY = -margin, maxSY = vpH + margin;

    float targetX = (float)screenX;
    float targetY = (float)screenY;

    int bestIdx = -1;
    float bestDistSq = 2500.0f;

    for (size_t i = 0; i < m_positions.size(); i++)
    {
        if (!m_showNegaMass && m_positions[i].w < 0.0f) continue;

        float x = m_positions[i].x, y = m_positions[i].y, z = m_positions[i].z;

        float cx = vp[0]*x + vp[4]*y + vp[8]*z  + vp[12];
        float cy = vp[1]*x + vp[5]*y + vp[9]*z  + vp[13];
        float cw = vp[3]*x + vp[7]*y + vp[11]*z + vp[15];
        if (fabsf(cw) < 1e-6f) cw = 1e-6f;
        if (cw < 0.0f) continue;

        float ndcX = cx / cw;
        float ndcY = cy / cw;

        float sx = (ndcX + 1.0f) * 0.5f * vpW;
        float sy = (1.0f - ndcY) * 0.5f * vpH;

        if (sx < minSX || sx > maxSX || sy < minSY || sy > maxSY) continue;

        float dx = sx - targetX;
        float dy = sy - targetY;
        float distSq = dx*dx + dy*dy;

        if (distSq < bestDistSq)
        {
            bestDistSq = distSq;
            bestIdx = (int)i;
        }
    }

    return bestIdx;
}

void App::removeMarker(int index)
{
    if (index < 0 || index >= (int)m_markers.size()) return;
    m_markers.erase(m_markers.begin() + index);
}

void App::updateMarkerPaths()
{
    if (m_markers.empty() || m_positions.empty()) return;

    if ((int)m_markerPathEnabled.size() != (int)m_markers.size())
        m_markerPathEnabled.resize(m_markers.size(), false);
    if ((int)m_markerPaths.size() != (int)m_markers.size())
        m_markerPaths.resize(m_markers.size());

    for (size_t i = 0; i < m_markers.size(); i++)
    {
        if (!m_markerPathEnabled[i]) continue;
        int idx = m_markers[i];
        if (idx < 0 || idx >= (int)m_positions.size()) continue;

        std::vector<cl_float4>& path = m_markerPaths[i];
        path.push_back(m_positions[idx]);
    }
}

// ── Annotations overlay (screen-space 2D, for annotated screenshots) ─────────
void App::renderAnnotations()
{
    ImGuiViewport* gMV = ImGui::GetMainViewport();
    const float gMVX = gMV->Pos.x, gMVY = gMV->Pos.y;
    const float panelW = 340.0f;
    const float vpX = gMVX + panelW;
    const float vpW = (float)m_width - panelW, vpH = (float)m_height;
    if (vpW <= 0 || vpH <= 0) return;

    ImDrawList* dl = ImGui::GetForegroundDrawList(gMV);
    dl->PushClipRect(ImVec2(vpX, gMVY), ImVec2(vpX + vpW, gMVY + vpH), true);

    auto drawArrow = [&](ImVec2 a, ImVec2 b, ImU32 col) {
        dl->AddLine(a, b, col, 2.0f);
        float dx = b.x - a.x, dy = b.y - a.y;
        float len = sqrtf(dx*dx + dy*dy);
        if (len > 1.0f) {
            float ux = dx/len, uy = dy/len;
            float ah = 14.0f;          // arrowhead length
            float aw = 7.0f;           // half-width
            ImVec2 base(b.x - ux*ah, b.y - uy*ah);
            ImVec2 p1(base.x - uy*aw, base.y + ux*aw);
            ImVec2 p2(base.x + uy*aw, base.y - ux*aw);
            dl->AddTriangleFilled(b, p1, p2, col);
        }
    };

    // Committed annotations
    for (const auto& an : m_annotations) {
        ImVec2 p0(an.x0, an.y0), p1(an.x1, an.y1);
        switch (an.type) {
            case 1: { // arrow + optional label at tail
                drawArrow(p0, p1, an.col);
                if (an.text[0]) {
                    ImVec2 ts = ImGui::CalcTextSize(an.text);
                    ImVec2 tp(p0.x - ts.x*0.5f, p0.y - ts.y - 4);
                    dl->AddRectFilled(ImVec2(tp.x-3,tp.y-2), ImVec2(tp.x+ts.x+3,tp.y+ts.y+2), IM_COL32(0,0,0,150));
                    dl->AddText(tp, an.col, an.text);
                }
                break;
            }
            case 2: { // text
                if (an.text[0]) {
                    ImVec2 ts = ImGui::CalcTextSize(an.text);
                    dl->AddRectFilled(ImVec2(p0.x-3,p0.y-2), ImVec2(p0.x+ts.x+3,p0.y+ts.y+2), IM_COL32(0,0,0,150));
                    dl->AddText(p0, an.col, an.text);
                }
                break;
            }
            case 3: // rectangle
                dl->AddRect(ImVec2(fminf(p0.x,p1.x),fminf(p0.y,p1.y)),
                            ImVec2(fmaxf(p0.x,p1.x),fmaxf(p0.y,p1.y)), an.col, 0.0f, 0, 2.0f);
                break;
            case 4: { // circle (center p0, edge p1)
                float r = sqrtf((p1.x-p0.x)*(p1.x-p0.x) + (p1.y-p0.y)*(p1.y-p0.y));
                if (r < 2) r = 2;
                dl->AddCircle(p0, r, an.col, 64, 2.0f);
                break;
            }
        }
    }

    // Live preview of in-progress geometry (between 1st click and 2nd)
    if (m_annotTool != 0 && m_annotStage == 1) {
        ImVec2 mp = ImGui::GetMousePos();
        ImU32 col = IM_COL32((int)(m_annotColor[0]*255),(int)(m_annotColor[1]*255),
                             (int)(m_annotColor[2]*255),160);
        ImVec2 p0(m_annotX0, m_annotY0);
        if      (m_annotTool == 1) drawArrow(p0, mp, col);
        else if (m_annotTool == 3) dl->AddRect(ImVec2(fminf(p0.x,mp.x),fminf(p0.y,mp.y)),
                                               ImVec2(fmaxf(p0.x,mp.x),fmaxf(p0.y,mp.y)), col, 0.0f, 0, 1.5f);
        else if (m_annotTool == 4) {
            float r = sqrtf((mp.x-p0.x)*(mp.x-p0.x) + (mp.y-p0.y)*(mp.y-p0.y));
            dl->AddCircle(p0, r<2?2:r, col, 64, 1.5f);
        }
    }
    dl->PopClipRect();

    // Text editor popup for the annotation awaiting a label
    if (m_annotEditIdx >= 0 && m_annotEditIdx < (int)m_annotations.size()) {
        Annotation& an = m_annotations[m_annotEditIdx];
        ImGui::SetNextWindowPos(ImVec2(an.x0 + 12, an.y0 + 12), ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(ImVec2(260, 0), ImGuiCond_Appearing);
        ImGui::Begin("Annotation text", nullptr,
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::SetNextItemWidth(230);
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        bool entered = ImGui::InputText("##annottext", m_annotTextBuf, sizeof(m_annotTextBuf),
                                        ImGuiInputTextFlags_EnterReturnsTrue);
        bool ok     = ImGui::Button("OK");
        ImGui::SameLine();
        bool cancel = ImGui::Button("Cancel");
        if (entered || ok) {
            strncpy(an.text, m_annotTextBuf, sizeof(an.text)-1);
            an.text[sizeof(an.text)-1] = '\0';
            // If text annotation left empty, drop it
            if (an.type == 2 && an.text[0] == '\0')
                m_annotations.erase(m_annotations.begin() + m_annotEditIdx);
            m_annotEditIdx = -1;
        } else if (cancel) {
            // Arrow keeps its geometry without label; empty text drops
            if (an.type == 2 && an.text[0] == '\0')
                m_annotations.erase(m_annotations.begin() + m_annotEditIdx);
            m_annotEditIdx = -1;
        }
        ImGui::End();
    }
}

void App::renderMarkerPaths()
{
    if (m_markers.empty() || m_positions.empty() || !m_camera) return;

    float vp[16];
    m_camera->getViewProjMatrix(vp);

    ImGuiViewport* gMV = ImGui::GetMainViewport();
    if (!gMV) return;
    const float gMVX = gMV->Pos.x, gMVY = gMV->Pos.y;
    const float panelW = 340.0f;
    const float vpX = gMVX + panelW;
    const float vpW = (float)m_width - panelW;
    const float vpH = (float)m_height;
    if (vpW <= 0 || vpH <= 0) return;

    // Uses the background draw list to be ABOVE the 3D scene
    // but BELOW the ImGui windows
    ImDrawList* dl = ImGui::GetBackgroundDrawList(gMV);
    if (!dl) return;

    ImU32 colMagenta = IM_COL32(255, 0, 255, 220);

    for (size_t m = 0; m < m_markers.size(); m++)
    {
        if (!m_markerPathEnabled[m]) continue;
        const std::vector<cl_float4>& path = m_markerPaths[m];
        if (path.size() < 2) continue;

        ImVec2 prev = ImVec2(0,0);
        bool first = true;
        for (size_t i = 0; i < path.size(); i++)
        {
            const cl_float4& p = path[i];
            float cx = vp[0]*p.x + vp[4]*p.y + vp[8]*p.z  + vp[12];
            float cy = vp[1]*p.x + vp[5]*p.y + vp[9]*p.z  + vp[13];
            float cw = vp[3]*p.x + vp[7]*p.y + vp[11]*p.z + vp[15];
            if (fabsf(cw) < 1e-6f) cw = 1e-6f;
            if (cw < 0.0f) { first = true; continue; }

            float ndcX = cx / cw;
            float ndcY = cy / cw;
            float sx = vpX + (ndcX + 1.0f) * 0.5f * vpW;
            float sy = gMVY + (1.0f - ndcY) * 0.5f * vpH;

            if (sx < vpX || sx > vpX + vpW || sy < gMVY || sy > gMVY + vpH) { first = true; continue; }

            ImVec2 cur(sx, sy);
            if (!first)
                dl->AddLine(prev, cur, colMagenta, 2.0f);
            prev = cur;
            first = false;
        }
    }
}

// ── σ field (km/s) for the marker "temperature" ──────────────────────────────
// σ² = <|v|²> - |<v>|² par cellule (le flot d'ensemble — rotation du disque — est
// removed -> pure random agitation). PHYSICAL value in km/s (same conversion as
// the marker velocity display, which includes m_timeStep).
void App::computeMarkerSigmaField()
{
    m_markerSigKms.clear();
    const size_t n = m_positions.size();
    if (n == 0 || m_rotVelocities.size() < n) return;

    double m0=0,m1=0,m2=0; size_t np=0;
    for (size_t i=0;i<n;i++){ if(m_positions[i].w<0.0f) continue; m0+=m_positions[i].x; m1+=m_positions[i].y; m2+=m_positions[i].z; np++; }
    if (np==0) return;
    m0/=(double)np; m1/=(double)np; m2/=(double)np;

    std::vector<float> rad; rad.reserve(np);
    for (size_t i=0;i<n;i++){ const cl_float4&p=m_positions[i]; if(p.w<0.0f) continue;
        double dx=p.x-m0,dy=p.y-m1,dz=p.z-m2; rad.push_back((float)sqrt(dx*dx+dy*dy+dz*dz)); }
    float R=1.0f;
    if(!rad.empty()){ size_t kk=(size_t)(0.90*(double)(rad.size()-1));
        std::nth_element(rad.begin(),rad.begin()+kk,rad.end()); R=rad[kk]*1.1f; if(R<1e-3f)R=1e-3f; }
    m_markerSigMn[0]=(float)m0-R; m_markerSigMn[1]=(float)m1-R; m_markerSigMn[2]=(float)m2-R;
    m_markerSigExt[0]=m_markerSigExt[1]=m_markerSigExt[2]=2.0f*R;

    const int NG=MARKER_SIG_NG, NG3=NG*NG*NG;
    auto clampi=[&](int v){ return v<0?0:(v>=NG?NG-1:v); };
    auto cellOf=[&](const cl_float4&p)->int{
        int cx=clampi((int)((p.x-m_markerSigMn[0])/m_markerSigExt[0]*NG));
        int cy=clampi((int)((p.y-m_markerSigMn[1])/m_markerSigExt[1]*NG));
        int cz=clampi((int)((p.z-m_markerSigMn[2])/m_markerSigExt[2]*NG));
        return (cz*NG+cy)*NG+cx;
    };
    std::vector<float> sx(NG3,0),sy(NG3,0),sz(NG3,0),s2(NG3,0),cnt(NG3,0);
    for (size_t i=0;i<n;i++){
        if(m_positions[i].w<0.0f) continue;
        int c=cellOf(m_positions[i]);
        const cl_float4&v=m_rotVelocities[i];
        sx[c]+=v.x; sy[c]+=v.y; sz[c]+=v.z; s2[c]+=v.x*v.x+v.y*v.y+v.z*v.z; cnt[c]+=1.0f;
    }
    const float conv=(m_lyPerUnit*9.461e12f)/(m_timeStep*3.156e13f);  // 1 sim_vel -> km/s
    m_markerSigKms.assign(NG3,0.0f);
    for (int c=0;c<NG3;c++){
        if(cnt[c]<4.0f) continue;
        float inv=1.0f/cnt[c];
        float mvx=sx[c]*inv,mvy=sy[c]*inv,mvz=sz[c]*inv;
        float d=s2[c]*inv-(mvx*mvx+mvy*mvy+mvz*mvz);     // σ_3D² (sim)
        m_markerSigKms[c]= d>0.0f ? sqrtf(d)*conv : 0.0f; // σ_3D (km/s)
    }
}

float App::markerSigmaKms(int idx) const
{
    if (m_markerSigKms.empty() || idx<0 || idx>=(int)m_positions.size()) return -1.0f;
    const cl_float4&p=m_positions[idx];
    const int NG=MARKER_SIG_NG;
    auto clampi=[&](int v){ return v<0?0:(v>=NG?NG-1:v); };
    int cx=clampi((int)((p.x-m_markerSigMn[0])/m_markerSigExt[0]*NG));
    int cy=clampi((int)((p.y-m_markerSigMn[1])/m_markerSigExt[1]*NG));
    int cz=clampi((int)((p.z-m_markerSigMn[2])/m_markerSigExt[2]*NG));
    return m_markerSigKms[(cz*NG+cy)*NG+cx];   // 0 si voisinage trop pauvre
}

void App::renderMarkers()
{
    if (m_markers.empty() || m_positions.empty() || !m_camera) return;

    float vp[16];
    m_camera->getViewProjMatrix(vp);

    ImGuiViewport* gMV = ImGui::GetMainViewport();
    if (!gMV) return;
    const float gMVX = gMV->Pos.x, gMVY = gMV->Pos.y;
    const float panelW = 340.0f;
    const float vpX = gMVX + panelW;
    const float vpW = (float)m_width - panelW;
    const float vpH = (float)m_height;
    if (vpW <= 0 || vpH <= 0) return;

    // Uses the background draw list to be ABOVE the 3D scene
    // but BELOW the ImGui windows
    ImDrawList* dl = ImGui::GetBackgroundDrawList(gMV);
    if (!dl) return;

    float velConv = (m_lyPerUnit * 9.461e12f) / (m_timeStep * 3.156e13f);
    bool haveVels = !m_rotVelocities.empty() && m_rotVelocities.size() >= m_positions.size();

    // σ field for the kinetic temperature. When RUNNING the positions move every frame
    // -> recompute EVERY frame (otherwise the stale grid returns σ=0 -> the T disappears).
    // When PAUSED: throttled (static positions).
    if (!m_markers.empty() && haveVels) {
        bool running = m_simRunning && !m_simPaused;
        if (m_markerSigKms.empty() || running || (++m_markerSigTick % 20) == 0)
            computeMarkerSigmaField();
    }

    for (size_t m = 0; m < m_markers.size(); m++)
    {
        int idx = m_markers[m];
        if (idx < 0 || idx >= (int)m_positions.size()) continue;

        const cl_float4& p = m_positions[idx];
        float x = p.x, y = p.y, z = p.z;

        float cx = vp[0]*x + vp[4]*y + vp[8]*z  + vp[12];
        float cy = vp[1]*x + vp[5]*y + vp[9]*z  + vp[13];
        float cw = vp[3]*x + vp[7]*y + vp[11]*z + vp[15];
        if (fabsf(cw) < 1e-6f) cw = 1e-6f;

        float ndcX = cx / cw;
        float ndcY = cy / cw;
        if (cw < 0.0f) continue;

        float sx = vpX + (ndcX + 1.0f) * 0.5f * vpW;
        float sy = gMVY + (1.0f - ndcY) * 0.5f * vpH;
        if (sx < vpX || sx > vpX + vpW || sy < gMVY || sy > gMVY + vpH) continue;

        bool isSelected = (m_selectedMarker == (int)m);
        ImU32 colArrow = isSelected ? IM_COL32(255, 0, 0, 240)   : IM_COL32(255, 220, 0, 240);
        ImU32 colText  = isSelected ? IM_COL32(255, 100, 100, 255) : IM_COL32(255, 255, 0, 255);
        float arrowLen = 20.0f;
        float headSize = 6.0f;

        ImVec2 tip(sx, sy);
        ImVec2 base(sx, sy - arrowLen);

        dl->AddLine(base, tip, colArrow, 2.0f);
        dl->AddTriangleFilled(
            ImVec2(sx - headSize, sy - arrowLen + headSize),
            ImVec2(sx + headSize, sy - arrowLen + headSize),
            tip,
            colArrow);

        float spd = 0.0f;
        if (haveVels && idx < (int)m_rotVelocities.size())
        {
            const cl_float4& v = m_rotVelocities[idx];
            spd = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z) * velConv;
        }

        // Kinetic temperature = local σ (km/s) -> equivalent T (gas/virial):
        // k_B T = μ m_p σ_1D²  ->  T[K] ≈ 72.6·σ_1D[km/s]²  (μ≈0.6), σ_1D = σ_3D/√3.
        char tbuf[56] = "";
        float sigKms = markerSigmaKms(idx);
        if (sigKms > 0.0f) {
            float sig1d = sigKms / 1.7320508f;
            float Tk    = 72.6f * sig1d * sig1d;
            snprintf(tbuf, sizeof(tbuf), "T~%.1e K (sig %.0f km/s)", Tk, sigKms);
        }

        // Line 1 = velocity (km/s), Line 2 = temperature (below velocity). Centered on sx.
        char line1[40];
        if (spd > 0.0f)
            snprintf(line1, sizeof(line1), "#%zu %.1f km/s", m + 1, spd);
        else
            snprintf(line1, sizeof(line1), "#%zu", m + 1);

        auto drawCentered = [&](const char* txt, float topY) {
            ImVec2 ts = ImGui::CalcTextSize(txt);
            float lx = sx - ts.x * 0.5f;
            dl->AddRectFilled(ImVec2(lx - 2, topY - 1),
                              ImVec2(lx + ts.x + 2, topY + ts.y + 1),
                              IM_COL32(0, 0, 0, 180));
            dl->AddText(ImVec2(lx, topY), colText, txt);
        };

        float lineH = ImGui::GetTextLineHeight();
        bool  hasTemp = (tbuf[0] != '\0');
        float blockH = hasTemp ? (2.0f * lineH + 1.0f) : lineH;
        float topY   = base.y - blockH - 4.0f;
        drawCentered(line1, topY);
        if (hasTemp) drawCentered(tbuf, topY + lineH + 1.0f);
    }
}

void App::renderGUI()
{
    ImGuiIO& io = ImGui::GetIO();

    ImGuiViewport* mainVP = ImGui::GetMainViewport();
    const float mvX = mainVP->Pos.x;
    const float mvY = mainVP->Pos.y;

    const float panelW = 340.0f;
    float vpX = mvX + panelW;
    float vpW = (float)m_width  - panelW;
    float vpH = (float)m_height;

    if (vpW > 0 && vpH > 0 && ((int)vpW != m_renderer->getWidth() || (int)vpH != m_renderer->getHeight()))
    {
        m_renderer->resize((int)vpW, (int)vpH);
        m_camera->resize((int)vpW, (int)vpH);
    }

    GLuint tex = m_renderer->getFrameTexture();
    ImDrawList* bg = ImGui::GetBackgroundDrawList(ImGui::GetMainViewport());
    bg->AddImage((ImTextureID)(intptr_t)tex, ImVec2(vpX, mvY), ImVec2(vpX + vpW, mvY + vpH), ImVec2(0, 1), ImVec2(1, 0));

    ImGui::SetNextWindowViewport(mainVP->ID);
    ImGui::SetNextWindowPos(ImVec2(mvX, mvY), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelW, (float)m_height), ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 6));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6, 4));
    ImGui::Begin("Controls", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus);

    SectionTitle("Settings");


    ImGui::PushID("geom");
    if (ImGui::CollapsingHeader("Geometry", ImGuiTreeNodeFlags_DefaultOpen))
    {
        SliderFloatWithInput("Radius", &m_radius, 100.0f, 100000.0f, "%.0f");
        SliderFloatWithInput("Scale", &m_scaleUnit, 0.0001f, 10.0f, "%.4f");
    }
    ImGui::PopID();

    ImGui::PushID("mass");
    if (ImGui::CollapsingHeader("Mass", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Up to 10M per species → 20M total (large GPU configs)
        // Posi up to 20 M: a single-species model (Newton, no negative mass) can
        // put the whole budget on the positives. What limits (VRAM + compute) is the
        // TOTAL: 20 M positives alone ≈ 10 M + 10 M in bimetric.
        SliderIntWithInput("Posi Stars", &m_numPosiStars, 0, 20000000);
        SliderIntWithInput("Nega Stars", &m_numNegaStars, 0, 10000000);
        SliderFloatWithInput("Total Solar M", &m_totalSolarMass, 1.0e4f, 1.0e12f, "%.1e");
        // Nega Mass is now interpreted as TOTAL halo mass (negative), like
        // Total Solar M for positives. Halo strength is independent of N.
        // Recommended : |Nega Mass| ~ Total Solar M for balanced Janus galaxy.
        SliderFloatWithInput("Nega Mass", &m_negaDensity, -1.0e6f, 0.0f, "%.0f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Total halo mass (negative). Equivalent to 'Total Solar M' for the disk.\n"
                              "Per-particle mass = Nega Mass / Nega Stars.\n"
                              "Recommended : |Nega Mass| ~ Total Solar M for balanced disk/halo.\n"
                              "Too large -> Janus repulsion ejects disk into ring.");
        SliderFloatWithInput("SMBH Mass", &m_blackHoleMass, 1.0f, 1.0e12f, "%.0f");
    }
    ImGui::PopID();

    ImGui::PushID("dyn");
    if (ImGui::CollapsingHeader("Dynamics", ImGuiTreeNodeFlags_DefaultOpen))
    {
        SliderFloatWithInput("Rot Speed", &m_rotSpeed, 0.0f, 20.0f, "%.1f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Tangential velocity added to vCirc (sim-units). 0 = pure orbit.");
        SliderFloatWithInput("Init Speed", &m_initSpeed, 0.0f, 10000.0f, "%.0f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Thermal velocity dispersion (/ 1000 -> sim-units). 1200 = +/-1.2 sim-vel.");
    }
    ImGui::PopID();

    ImGui::PushID("sim");
    if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // ── GPU Barnes-Hut Tree (replaces O(N²) for large N) ─────────────
        ImGui::Checkbox("Use Tree (Barnes-Hut)", &m_useTree);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("GPU Barnes-Hut octree (Karras 2012 LBVH)\n"
                              "O(N log N) instead of O(N^2) — required for N > 200k\n"
                              "Two-trees per species: + and - handled separately\n"
                              "Same physics as direct N^2 (accurate to ~1%% with theta=0.5)\n"
                              "Requires Apply & Generate");
        if (m_useTree) {
            SliderFloatWithInput("Tree theta", &m_treeTheta, 0.3f, 1.0f, "%.2f");
            if (m_simulation) m_simulation->setTreeTheta(m_treeTheta);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Barnes-Hut opening angle (live-tunable).\n"
                                  "0.3 = very accurate, slow\n"
                                  "0.5 = accurate (~1%% error), DEFAULT\n"
                                  "0.7 = good trade-off (~3%% error), 1.5-2x faster\n"
                                  "1.0 = fast but artifacts (~5-10%% error)");
            ImGui::Checkbox("Quadrupole", &m_treeQuad);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Order-2 multipole correction. Lets you raise theta\n"
                                  "without losing accuracy → faster at equal quality.");
            ImGui::SameLine();
            ImGui::Checkbox("Warp-coherent", &m_treeWarp);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Warp-coherent tree traversal (32 lanes share one path).\n"
                                  "2-3x faster in dense regions. Same physics.\n"
                                  "Toggle off to compare against per-thread traversal.");
        }
        if (m_simulation) {
            m_simulation->setUseTree(m_useTree);
            m_simulation->setTreeTheta(m_treeTheta);
            m_simulation->setTreeQuadrupole(m_treeQuad);
            m_simulation->setTreeWarpCoherent(m_treeWarp);
        }

        SliderFloatWithInput("Time Step", &m_timeStep, 0.1f, 1000.0f, "%.2f");
        SliderFloatWithInput("Softening", &m_softening, 1.0f, 200.0f, "%.1f");
        if (m_softening < 1.0f) m_softening = 1.0f;
        SliderFloatWithInput("G Const", &m_gravConstant, 1.0e-12f, 1.0e-8f, "%.3e");
        SliderFloatWithInput("Sim Speed", &m_simSpeed, 0.02f, 60.0f, "%.2fx");

        // SOFTENING indicator = ε / inter-particle separation (like Universe).
        // Softening is for force RESOLUTION, not time-step stability:
        // too small -> numerical heating (close pairs); too large -> over-smoothed.
        float nPart   = (float)(m_numPosiStars + m_numNegaStars);
        float partSep = (nPart > 0.0f) ? 2.0f * m_radius / cbrtf(nPart) : 1.0f;
        float ratio   = (partSep > 0.0f) ? m_softening / partSep : 0.0f;
        ImVec4 ratioColor;
        const char* ratioLabel;
        if      (ratio < 0.05f) { ratioColor = ImVec4(1.0f,0.2f,0.2f,1.0f); ratioLabel = "too small (heating)"; }
        else if (ratio < 0.3f)  { ratioColor = ImVec4(0.4f,0.8f,1.0f,1.0f); ratioLabel = "fine resolution"; }
        else if (ratio <= 2.0f) { ratioColor = ImVec4(0.2f,1.0f,0.2f,1.0f); ratioLabel = "balanced"; }
        else if (ratio <= 5.0f) { ratioColor = ImVec4(1.0f,0.8f,0.0f,1.0f); ratioLabel = "over-smoothed"; }
        else                    { ratioColor = ImVec4(1.0f,0.4f,0.2f,1.0f); ratioLabel = "no structure possible"; }
        ImGui::TextColored(ratioColor, "e/sep : %.2f  [%s]", ratio, ratioLabel);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Softening / inter-particle separation\n"
                              "< 0.05    : too small (numerical heating from close pairs)\n"
                              "0.05-0.3  : fine resolution (slow but accurate)\n"
                              "0.3-2.0   : balanced (recommended)\n"
                              "2.0-5.0   : over-smoothed (fine structure lost)\n"
                              "> 5.0     : no structure possible\n"
                              "sep = 2*radius / N^(1/3) ~ %.1f sim units", partSep);

        ImGui::TextColored(ImVec4(1,0.6f,0,1), "N2: TimeStep <= 1 required");
    }
    ImGui::PopID();

    ImGui::PushID("spiral");
    if (ImGui::CollapsingHeader("Density Waves", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Standard checkbox (matches Universe style)
        ImGui::Checkbox("Enable Density Waves", &m_densityWavesEnabled);

        // Parameters greyed out when inactive
        if (!m_densityWavesEnabled) ImGui::BeginDisabled();
        SliderFloatWithInput("Amplitude",  &m_spAmplitude,    0.0f,  5.0f,  "%.3f");
        SliderFloatWithInput("Pattern Spd",&m_spPatternSpeed, 0.0f,  0.1f,  "%.4f");
        SliderFloatWithInput("Pitch Angle",&m_spPitchAngle,   5.0f, 45.0f, "%.1f");
        SliderIntWithInput("Spiral Arms",  &m_spArms,         1,    20);
        if (!m_densityWavesEnabled) ImGui::EndDisabled();

        if (m_simulation)
        {
            float pitchTan = 1.0f / tanf(m_spPitchAngle * 3.14159265f / 180.0f); // cot(pitch) = coeff de ln(r) d'une spirale log
            // Passes amplitude=0 if disabled — parameters kept for re-activation
            float amp = m_densityWavesEnabled ? m_spAmplitude : 0.0f;
            m_simulation->setSpiralParams(amp, m_spPatternSpeed, pitchTan, (unsigned int)m_spArms);
        }
    }
    ImGui::PopID();

    ImGui::PushID("gfx");
    if (ImGui::CollapsingHeader("Graphic Tools", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Enable Glow", &m_glowEnabled);
        if (m_glowEnabled)
        {
            SliderFloatWithInput("Blur Radius", &m_blurRadius,    1.0f, 20.0f, "%.1f");
            SliderFloatWithInput("Glow Intens", &m_glowIntensity, 0.0f,  8.0f, "%.2f");
        }
        bool prevHalo = m_softHalo;
        ImGui::Checkbox("Soft Halo (galaxy colors)", &m_softHalo);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Volumetric SPH render: adaptive gaussian splat -> continuous\n"
                              "field, colored with a realistic galaxy palette (blue arms ->\n"
                              "cream disk -> golden core, Hubble look). Hue = column density.");
        if (m_softHalo != prevHalo) { m_smoothLength.clear(); uploadParticlesToGPU(); }
        if (m_softHalo) {
            SliderFloatWithInput("Halo Bright", &m_sphBright, 0.02f, 8.0f, "%.3f");
            SliderFloatWithInput("Smooth", &m_sphSmooth, 0.0f, 8.0f, "%.1f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Gaussian blur of the density field (px) -> continuous filaments.\n"
                                  "0 = raw, 2-4 = smooth smoke.");
            SliderFloatWithInput("Core Detail", &m_sphCoreDetail, 40.0f, 600.0f, "%.0f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("asinh ceiling : 40 = white saturated cores, 200-600 = detail\n"
                                  "inside the dense cores (raise Halo Bright to relight the rest).");
            SliderFloatWithInput("Speed (quality)", &m_sphQuality, 0.35f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Splat resolution : lower = faster (less overdraw), little visible loss.");
        }
        // ── Velocity splatting: per-particle splats colored by velocity (no voxels) ──
        ImGui::Checkbox("Splatting velocity", &m_rayTrace);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Per-particle splatting (no 3D grid): each particle = gaussian\n"
                              "colored by its velocity (nebula palette). Light memory, no streaks.");
        if (m_rayTrace) {
            SliderFloatWithInput("Halo size", &m_volHaloSize, 0.5f, 40.0f, "%.1f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Halo splat radius (x softening). Small = thin filaments,\n"
                                  "large = diffuse cloud. Scales with zoom.");
            SliderFloatWithInput("Emission", &m_volGain, 0.01f, 1.0f, "%.3f");
            SliderFloatWithInput("Exposure", &m_volExposure, 0.2f, 4.0f, "%.2f");
            ImGui::Checkbox("Color by velocity (RT)", &m_volColorByVel);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Color = each particle's velocity (nebula palette, exact).\n"
                                  "Uncheck -> uniform warm tint.");
            ImGui::Checkbox("Star points (sharp)", &m_volStars);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Small sharp splats over the halo -> Hubble-style composition.");
            if (m_volStars) {
                SliderFloatWithInput("Star size (px)", &m_volStarScale, 0.5f, 5.0f, "%.2f");
                SliderFloatWithInput("Star bright", &m_volStarBright, 0.05f, 3.0f, "%.2f");
            }
        }

        if (ImGui::Checkbox("GPU vertex buffer (CUDA interop)", &m_useInterop)) {
            // En basculant, on force la reconstruction du VBO par le bon chemin.
            if (m_interopRes) { interopUnregister(m_interopRes); m_interopRes = nullptr; }
            m_interopOk = false; m_interopVboCount = 0;
            uploadParticlesToGPU();
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("ON  = render buffer built on the GPU (CUDA kernel -> VBO),\n"
                              "      without the CPU round-trip (readback/rebuild/upload) -> more FPS.\n"
                              "OFF = old CPU path. Toggle to compare FPS.\n"
                              "%s", m_interopOk ? "Status: interop ACTIVE" : "Status: CPU fallback");
        ImGui::Separator();
        SliderFloatWithInput("Pt Size", &m_particleSize, 0.05f, 3.0f, "%.2f");
        SliderFloatWithInput("Pt Size Nega", &m_particleSizeNeg, 0.05f, 3.0f, "%.2f");

        // ── Species display / color (moved here) ───────────────────────────
        {
            bool isJanusGT = (m_eqSelectedModel == 1);
            if (!isJanusGT) ImGui::BeginDisabled();
            bool prevShowGT = m_showNegaMass;
            ImGui::Checkbox("Show Nega Mass", &m_showNegaMass);
            if (!isJanusGT) { ImGui::SameLine(); ImGui::TextDisabled("(Janus only)"); ImGui::EndDisabled(); }
            if (m_showNegaMass != prevShowGT) uploadParticlesToGPU();

            bool prevCBVgt = m_colorByVelocity;
            ImGui::Checkbox("Color by Velocity", &m_colorByVelocity);
            if (m_colorByVelocity) { ImGui::SameLine(); ImGui::Checkbox("Auto-scale", &m_velAutoScale); }
            if (m_colorByVelocity != prevCBVgt) {
                if (m_colorByVelocity && m_simRunning) refreshAnalytics();
                uploadParticlesToGPU();
            }
        }

        // ── Auto-fit: the camera pulls back to keep the structure at constant size ──
        if (ImGui::Checkbox("Auto-fit (follow expansion)", &m_autoFit) && m_autoFit) {
            m_autoFitDistVel = 0.0f; m_autoFitLastTicks = 0;
            m_autoFitTargetDist = 0.0f; m_autoFitTargetRaw = 0.0f; m_autoFitLastFrame = -100000;
        }
        if (m_autoFit) {
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("Framing##autofit", &m_autoFitFill, 0.10f, 0.95f, "fill %.2f");
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("Smoothing##autofit", &m_autoFitSmoothTime, 0.15f, 2.50f, "%.2f s");
            ImGui::TextDisabled("Structure R (P90): %.0f  |  target dist: %.0f",
                                m_autoFitRadius, m_autoFitTargetDist);
            ImGui::TextDisabled("Wheel = adjust framing");
        }
        ImGui::Separator();

        ImGui::Checkbox("Show Grid", &m_showGrid);
        if (m_showGrid)
            SliderFloatWithInput("ly/unit", &m_lyPerUnit, 1.0f, 10000.0f, "%.0f");
        if (ImGui::Checkbox("Velocity Field", &m_showVelocityField)) {
            // Turning it on arms drag-select so the workflow is immediate.
            if (m_showVelocityField) m_velSelectMode = true;
        }
        if (m_showVelocityField)
        {
            SliderFloatWithInput("Arrow Scale", &m_velFieldScale, 0.1f, 10.0f, "%.1f");
            ImGui::Checkbox("Relative to COM", &m_velFieldRelative);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("On: subtract per-species bulk velocity (shows internal flow / infall)\nOff: absolute velocities (shows runaway drift)");
            ImGui::Checkbox("Select Region (drag)", &m_velSelectMode);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("On: left-drag on the viewport to box a region — the stars inside\nare captured and their arrows follow them. Off: left-click picks markers.");
            if (m_velSelActive) {
                ImGui::SameLine();
                ImGui::TextDisabled("(%zu stars)", m_velSelIndices.size());
                ImGui::SameLine();
                if (ImGui::SmallButton("Clear")) {
                    m_velSelActive = false;
                    m_velSelIndices.clear();
                }
            } else {
                ImGui::TextDisabled("Drag a box on the view to pick stars");
            }
        }

        // ── Annotations (for annotated screenshots, on pause) ─────────────────
        ImGui::Separator();
        ImGui::TextDisabled("Annotations");
        const char* tools[] = { "Off", "Arrow + text", "Text", "Rectangle", "Circle" };
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##annottool", &m_annotTool, tools, 5))
            m_annotStage = 0;  // reset in-progress geometry on tool change
        if (m_annotTool != 0) {
            ImGui::ColorEdit3("##annotcol", m_annotColor,
                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
            ImGui::SameLine();
            ImGui::TextDisabled("%s",
                m_annotTool == 2 ? "click to place text"
                                 : "click start, click end");
        }
        if (!m_annotations.empty()) {
            ImGui::Text("%zu placed", m_annotations.size());
            ImGui::SameLine();
            if (ImGui::SmallButton("Clear##annot")) { m_annotations.clear(); m_annotEditIdx = -1; }

            const char* tnames[] = { "", "Arrow", "Text", "Rect", "Circle" };
            int toDelete = -1;
            // Scrollable list; height grows with count up to ~6 rows
            float listH = ImGui::GetTextLineHeightWithSpacing() *
                          (float)std::min((int)m_annotations.size(), 6) + 8.0f;
            ImGui::BeginChild("##annotlist", ImVec2(-1, listH), true);
            for (int i = 0; i < (int)m_annotations.size(); i++) {
                const Annotation& a = m_annotations[i];
                ImGui::PushID(i);
                if (ImGui::SmallButton("x")) toDelete = i;
                ImGui::SameLine();
                // Color swatch
                ImVec4 sw = ImGui::ColorConvertU32ToFloat4(a.col);
                ImGui::ColorButton("##sw", sw,
                    ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker,
                    ImVec2(12, 12));
                ImGui::SameLine();
                const char* tn = (a.type >= 1 && a.type <= 4) ? tnames[a.type] : "?";
                if (a.text[0])
                    ImGui::Text("%d. %s \"%s\"", i + 1, tn, a.text);
                else
                    ImGui::Text("%d. %s", i + 1, tn);
                // Click a row (on the label) to re-edit its text (arrow/text only)
                if ((a.type == 1 || a.type == 2) && ImGui::IsItemClicked()) {
                    m_annotEditIdx = i;
                    strncpy(m_annotTextBuf, a.text, sizeof(m_annotTextBuf)-1);
                    m_annotTextBuf[sizeof(m_annotTextBuf)-1] = '\0';
                }
                ImGui::PopID();
            }
            ImGui::EndChild();
            if (toDelete >= 0) {
                m_annotations.erase(m_annotations.begin() + toDelete);
                if (m_annotEditIdx == toDelete) m_annotEditIdx = -1;
                else if (m_annotEditIdx > toDelete) m_annotEditIdx--;
            }
        }
    }
    ImGui::PopID();

    ImGui::Separator();
    if (!m_markers.empty())
    {
        ImGui::TextDisabled("Markers (%zu/16)", m_markers.size());
        float velConv = (m_lyPerUnit * 9.461e12f) / (m_timeStep * 3.156e13f);
        bool haveVels = !m_rotVelocities.empty() && m_rotVelocities.size() >= m_positions.size();
        if (!haveVels && m_simulation && m_simRunning)
        {
            m_simulation->readVelocities(m_rotVelocities);
            haveVels = !m_rotVelocities.empty() && m_rotVelocities.size() >= m_positions.size();
        }
        if ((int)m_markerPathEnabled.size() != (int)m_markers.size())
            m_markerPathEnabled.resize(m_markers.size(), false);
        if ((int)m_markerPaths.size() != (int)m_markers.size())
            m_markerPaths.resize(m_markers.size());
        for (size_t i = 0; i < m_markers.size(); i++)
        {
            int idx = m_markers[i];
            if (idx < 0 || idx >= (int)m_positions.size()) continue;
            float spd = 0.0f;
            if (haveVels && idx < (int)m_rotVelocities.size())
            {
                const cl_float4& v = m_rotVelocities[idx];
                spd = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z) * velConv;
            }
            bool isSelected = (m_selectedMarker == (int)i);

            char label[64];
            if (spd > 0.0f)
                snprintf(label, sizeof(label), "#%zu: p=%d  %.1f km/s", i+1, idx, spd);
            else
                snprintf(label, sizeof(label), "#%zu: p=%d", i+1, idx);

            float availW = ImGui::GetContentRegionAvail().x;
            float btnW = 32.0f;
            float labelW = availW - (btnW + 4) * 2;

            if (isSelected) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1,0.4f,0.4f,1));
            ImGui::PushItemWidth(labelW);
            if (ImGui::Selectable(label, isSelected, 0, ImVec2(labelW, 0))) {
                m_selectedMarker = isSelected ? -1 : (int)i;
            }
            ImGui::PopItemWidth();
            if (isSelected) ImGui::PopStyleColor();

            ImGui::SameLine();

            char pathBtnId[32]; snprintf(pathBtnId, sizeof(pathBtnId), "Path##%zu", i);
            bool pathEnabled = m_markerPathEnabled[i];
            if (pathEnabled) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.2f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.8f, 0.3f, 1.0f));
            }
            if (ImGui::Button(pathBtnId, ImVec2(btnW, 0))) {
                m_markerPathEnabled[i] = !m_markerPathEnabled[i];
                if (!m_markerPathEnabled[i]) m_markerPaths[i].clear();
            }
            if (pathEnabled) ImGui::PopStyleColor(2);

            ImGui::SameLine();

            char btnId[32]; snprintf(btnId, sizeof(btnId), "X##%zu", i);
            if (ImGui::Button(btnId, ImVec2(btnW, 0))) {
                removeMarker((int)i);
                if (m_selectedMarker == (int)i) m_selectedMarker = -1;
                else if (m_selectedMarker > (int)i) m_selectedMarker--;
                if (i < m_markerPathEnabled.size()) m_markerPathEnabled.erase(m_markerPathEnabled.begin() + i);
                if (i < m_markerPaths.size()) m_markerPaths.erase(m_markerPaths.begin() + i);
            }
        }
        // ── Distance tool : consecutive marker pairs (#1↔#2, #2↔#3, ...) ────
        if (m_markers.size() >= 2) {
            ImGui::Separator();
            ImGui::TextDisabled("Distances (consecutive pairs):");
            for (size_t i = 0; i + 1 < m_markers.size(); i++) {
                int a = m_markers[i];
                int b = m_markers[i+1];
                if (a < 0 || b < 0 || a >= (int)m_positions.size() || b >= (int)m_positions.size())
                    continue;
                const cl_float4& pa = m_positions[a];
                const cl_float4& pb = m_positions[b];
                float dx = (pb.x - pa.x) * m_lyPerUnit;
                float dy = (pb.y - pa.y) * m_lyPerUnit;
                float dz = (pb.z - pa.z) * m_lyPerUnit;
                float dist_ly = sqrtf(dx*dx + dy*dy + dz*dz);
                // Pretty-print with adaptive unit
                char distBuf[40];
                if      (dist_ly >= 1.0e9f) snprintf(distBuf, sizeof(distBuf), "%.2f Gly", dist_ly / 1.0e9f);
                else if (dist_ly >= 1.0e6f) snprintf(distBuf, sizeof(distBuf), "%.2f Mly", dist_ly / 1.0e6f);
                else if (dist_ly >= 1.0e3f) snprintf(distBuf, sizeof(distBuf), "%.2f kly", dist_ly / 1.0e3f);
                else                        snprintf(distBuf, sizeof(distBuf), "%.1f LY",  dist_ly);
                ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f),
                                   "#%zu ↔ #%zu : %s", i+1, i+2, distBuf);
            }
            ImGui::Separator();
        }
        if (ImGui::Button("Clear All Markers", ImVec2(-1, 20))) {
            m_markers.clear();
            m_selectedMarker = -1;
            m_markerPathEnabled.clear();
            m_markerPaths.clear();
        }
    }
    else
    {
        ImGui::TextDisabled("Click on viewport (pause) to add markers");
    }

    SectionTitle("Controls");

    ImGui::Text("Frame: %d", m_currentFrame);
    ImGui::Text("Parts: %lu", m_particleCount);
    ImGui::Text("FPS: %.1f", io.Framerate);

    // VRAM usage
    if (m_simulation)
    {
        size_t vramFree = 0, vramTotal = 0;
        m_simulation->getVRAMInfo(vramFree, vramTotal);
        size_t used = vramTotal - vramFree;
        float pct = (vramTotal > 0) ? (float)used / vramTotal * 100.0f : 0.0f;
        ImVec4 vramCol = (pct > 85.0f) ? ImVec4(1,0.3f,0.3f,1) :
                         (pct > 65.0f) ? ImVec4(1,0.8f,0.2f,1) :
                                         ImVec4(0.5f,0.9f,0.5f,1);
        ImGui::TextColored(vramCol, "VRAM: %zu / %zu MB (%.0f%%)",
            used >> 20, vramTotal >> 20, pct);
    }

    double elapsed_MY = (double)m_currentFrame * (double)m_timeStep;
    if (elapsed_MY < 1000.0)
        ImGui::Text("Time: %.2f MY", elapsed_MY);
    else
        ImGui::Text("Time: %.3f BY", elapsed_MY / 1000.0);

    ImGui::Separator();

    if (m_isGenerating)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Generating...");
    }
    else
    {
        // State indicator on the right, button always accessible
        if (m_simRunning && !m_simPaused)
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), "● Running");
        else if (m_simRunning && m_simPaused)
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "● Paused");

        if (ImGui::Button("Apply & Generate", ImVec2(-1, 26)))
        {
            m_generateCountdown = 2;
            m_isGenerating      = true;
        }
    }

    ImGui::Separator();

    {
        if (ImGui::Button(m_simPaused ? "Resume" : "Pause", ImVec2(-1, 26)))
            m_simPaused = !m_simPaused;
        if (ImGui::Button("Reset", ImVec2(-1, 26)))
            resetSimulation();
    }

    ImGui::Separator();

    if (ImGui::Button("Top View", ImVec2(-1, 22)))
    {
        // Preserve current camera distance so zoom is consistent across views
        double dist = 2000.0;
        if (m_camera) {
            Vector* o = m_camera->getOrigin();
            if (o) {
                dist = sqrt(o->x*o->x + o->y*o->y + o->z*o->z);
                if (dist < 100.0) dist = 100.0;
            }
        }
        m_edgeOnView = false;
        delete m_camera;
        m_camera = new Camera(m_renderer->getWidth(), m_renderer->getHeight(),
            new Vector(0, 0, dist), new Matrix(ROTATION, 0.0, 0.0, 0.0),
            60.0, 10.0, 1.0e9f);
        m_camera->setOrbitMode(false);
    }
    if (ImGui::Button("Toggle Edge-on View", ImVec2(-1, 22)))
    {
        // Preserve current camera distance across edge-on toggle
        double dist = 2000.0;
        if (m_camera) {
            Vector* o = m_camera->getOrigin();
            if (o) {
                dist = sqrt(o->x*o->x + o->y*o->y + o->z*o->z);
                if (dist < 100.0) dist = 100.0;
            }
        }
        m_edgeOnView = !m_edgeOnView;
        delete m_camera;
        if (m_edgeOnView) {
            m_camera = new Camera(m_renderer->getWidth(), m_renderer->getHeight(),
                new Vector(0, -dist, 0), new Matrix(ROTATION, 0.0, 0.0, 0.0),
                60.0, 10.0, 1.0e9f);
            m_camera->setEdgeOn(true);
        } else {
            m_camera = new Camera(m_renderer->getWidth(), m_renderer->getHeight(),
                new Vector(0, 0, dist), new Matrix(ROTATION, 0.0, 0.0, 0.0),
                60.0, 10.0, 1.0e9f);
        }
    }

    if (ImGui::Button("Screenshot", ImVec2(-1, 22))) saveScreenshot();

    // ── MP4 video recording ───────────────────────────────────────────────────
    if (!m_videoRecording) {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.65f,0.18f,0.18f,1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.80f,0.24f,0.24f,1.0f));
        if (ImGui::Button("Record MP4", ImVec2(-1, 22))) startVideoRecording();
        ImGui::PopStyleColor(2);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Records the viewport to MP4 (via ffmpeg).\nThe left panel is excluded. Click again to stop.");
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.85f,0.35f,0.10f,1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f,0.45f,0.15f,1.0f));
        if (ImGui::Button("Stop recording", ImVec2(-1, 22))) stopVideoRecording();
        ImGui::PopStyleColor(2);
        ImGui::TextDisabled("REC %dx%d @ %d fps", m_recW, m_recH, m_videoFps);
    }
    ImGui::SetNextItemWidth(120);
    ImGui::SliderInt("FPS##vid", &m_videoFps, 15, 60);

    // ── Replay timeline ───────────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::TextDisabled("Replay timeline");
    ImGui::Checkbox("Record snapshots", &m_replayRecord);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Captures positions every N frames.\n"
                          "RAM: ring buffer (oldest overwritten > Max frames, lost on close).\n"
                          "Stream to disk: each snapshot written to a folder -> UNLIMITED, never lost.");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::SliderInt("every##rep", &m_replayInterval, 5, 1000);

    bool prevStream = m_replayStream;
    ImGui::Checkbox("Stream to disk (unlimited)", &m_replayStream);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Writes EVERY snapshot to the folder below (unlimited, never lost).\n"
                          "Resume a session with 'Load folder'.");
    if (m_replayStream && !prevStream) { m_replayDiskCount = 0; m_replayTimes.clear(); m_replayShownCursor = -1; }

    if (m_replayStream) {
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##replaydir", m_replayDir, sizeof(m_replayDir));
        float halfW = (ImGui::GetContentRegionAvail().x - 6.0f) * 0.5f;
        if (ImGui::Button("Load folder", ImVec2(halfW, 22))) replayScanDiskFolder();
        ImGui::SameLine();
        if (ImGui::Button("New recording", ImVec2(halfW, 22))) {
            m_replayDiskCount = 0; m_replayTimes.clear();
            m_replayCursor = 0; m_replayShownCursor = -1; m_replayMode = false;
        }
        ImGui::TextDisabled("%d snapshots on disk", m_replayDiskCount);
    } else {
        ImGui::SetNextItemWidth(120);
        ImGui::SliderInt("Max frames##rep", &m_replayMaxFrames, 20, 1000);
        double mb = (double)m_replayFrames.size() * (m_positions.empty()?0:m_positions.size()) * sizeof(cl_float4) / 1e6;
        ImGui::SameLine(); ImGui::TextDisabled("(%zu snaps, ~%.0f MB)", m_replayFrames.size(), mb);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##replaypath", m_replayFilePath, sizeof(m_replayFilePath));
        float halfW = (ImGui::GetContentRegionAvail().x - 6.0f) * 0.5f;
        if (ImGui::Button("Save replay", ImVec2(halfW, 22))) saveReplay();
        ImGui::SameLine();
        if (ImGui::Button("Load replay", ImVec2(halfW, 22))) loadReplay();
    }

    if (replayCount() > 0) {
        if (ImGui::Checkbox("Replay mode", &m_replayMode)) {
            if (m_replayMode) { m_replayPlaying = false; m_simPaused = true; m_replayShownCursor = -1; }
        }
        if (m_replayMode) {
            int n = replayCount();
            float tcur = (m_replayCursor>=0 && m_replayCursor<(int)m_replayTimes.size()) ? m_replayTimes[m_replayCursor] : 0.0f;
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderInt("##scrub", &m_replayCursor, 0, n-1, "frame %d");
            ImGui::Text("t = %.1f MY", tcur);
            ImGui::SameLine();
            if (ImGui::Button(m_replayPlaying ? "Pause" : "Play")) m_replayPlaying = !m_replayPlaying;
            ImGui::SameLine();
            if (ImGui::Button("|<")) { m_replayCursor = 0; m_replayPlaying = false; }
            ImGui::SameLine();
            if (ImGui::Button(">|")) { m_replayCursor = n-1; m_replayPlaying = false; }
            ImGui::SetNextItemWidth(120);
            ImGui::SliderFloat("Speed##rep", &m_replayPlaySpeed, 0.1f, 5.0f, "%.1fx");
        }
        if (!m_replayStream) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Clear##rep")) {
                m_replayFrames.clear(); m_replayTimes.clear();
                m_replayCursor = 0; m_replayMode = false; m_replayPlaying = false;
            }
        }
    }

    ImGui::Separator();
    ImGui::TextDisabled("State I/O");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##statepath", m_stateFilePath, sizeof(m_stateFilePath));
    float thirdW = (ImGui::GetContentRegionAvail().x - 8) / 3.0f;
    if (ImGui::Button("Save State", ImVec2(thirdW, 22))) saveSimState();
    ImGui::SameLine(0, 4);
    if (ImGui::Button("Save to...", ImVec2(thirdW, 22))) saveSimStateTo();
    ImGui::SameLine(0, 4);
    if (ImGui::Button("Load State", ImVec2(thirdW, 22))) loadSimState();

    // ── Presets ───────────────────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::TextDisabled("Presets");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##presetname", m_presetName, sizeof(m_presetName));
    float halfPreset = (ImGui::GetContentRegionAvail().x - 4) * 0.5f;
    if (ImGui::Button("Save Preset", ImVec2(halfPreset, 22)))
    {
        if (m_presetName[0] != '\0')
            savePreset(m_presetName);
    }
    ImGui::SameLine(0, 4);
    {
        auto presets = listPresets();
        static int presetSel = 0;
        if (presetSel >= (int)presets.size()) presetSel = 0;
        const char* combo = presets.empty() ? "(no presets)" : presets[presetSel].c_str();
        ImGui::SetNextItemWidth(halfPreset);
        if (ImGui::BeginCombo("##presetlist", combo))
        {
            for (int i = 0; i < (int)presets.size(); i++)
            {
                bool sel = (presetSel == i);
                if (ImGui::Selectable(presets[i].c_str(), sel)) presetSel = i;
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (ImGui::Button("Load Preset", ImVec2(-1, 22)) && !presets.empty())
            loadPreset(presets[presetSel]);
    }
    ImGui::Separator();

    if (ImGui::Button(m_showRotCurve      ? "Hide Rot. Curve"   : "Rotation Curve",   ImVec2(-1,22))) m_showRotCurve      = !m_showRotCurve;
    if (ImGui::Button(m_showDensityProfile? "Hide Density"      : "Density Profile",  ImVec2(-1,22))) m_showDensityProfile= !m_showDensityProfile;
    if (ImGui::Button(m_showPhaseSpace    ? "Hide Phase Space"  : "Phase Space",      ImVec2(-1,22))) m_showPhaseSpace    = !m_showPhaseSpace;
    if (ImGui::Button(m_showEnergyPlot    ? "Hide Energy"       : "Energy",           ImVec2(-1,22))) m_showEnergyPlot    = !m_showEnergyPlot;
    if (ImGui::Button(m_showDispersion    ? "Hide Dispersion"   : "Dispersion & Q",   ImVec2(-1,22))) m_showDispersion    = !m_showDispersion;
    ImGui::Separator();
    if (ImGui::Button(m_showBarycenters   ? "Hide Barycenters"  : "Barycenters",      ImVec2(-1,22))) m_showBarycenters   = !m_showBarycenters;
    if (ImGui::Button(m_showFourier       ? "Hide Fourier"      : "Fourier Modes",    ImVec2(-1,22))) m_showFourier       = !m_showFourier;
    if (ImGui::Button(m_showDarkMatter    ? "Hide DM Profile"   : "DM Equivalent",    ImVec2(-1,22))) m_showDarkMatter    = !m_showDarkMatter;
    if (ImGui::Button(m_showEquationEditor ? "Hide Eq. Editor"   : "Equation Editor",  ImVec2(-1,22))) m_showEquationEditor = !m_showEquationEditor;
    ImGui::Separator();
    ImGui::TextDisabled("Scientific Analysis");
    if (ImGui::Button(m_showConservation ? "Hide Conservation" : "Conservation",     ImVec2(-1,22))) m_showConservation = !m_showConservation;
    if (ImGui::Button(m_showEscape       ? "Hide Escape Vel."  : "Escape Velocity",  ImVec2(-1,22))) m_showEscape       = !m_showEscape;
    if (ImGui::Button(m_showCorrelation  ? "Hide xi(r)"        : "Correlation xi(r)",ImVec2(-1,22))) m_showCorrelation  = !m_showCorrelation;
    if (ImGui::Button(m_showSersic       ? "Hide Sersic"       : "Sersic Profile",   ImVec2(-1,22))) m_showSersic       = !m_showSersic;
    if (ImGui::Button(m_showHaloFinder   ? "Hide Halo Finder"  : "Halo Finder (FoF)",ImVec2(-1,22))) m_showHaloFinder   = !m_showHaloFinder;
    ImGui::Separator();
    ImGui::Checkbox("Auto-Refresh", &m_rotAutoRefresh);
    if (m_rotAutoRefresh) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::InputInt("##ri", &m_rotRefreshInterval, 0);
        if (m_rotRefreshInterval < 10) m_rotRefreshInterval = 10;
    }
    if (!m_rotAutoRefresh && ImGui::Button("Refresh Analytics", ImVec2(-1,22)) && m_simulation && !m_positions.empty()) refreshAnalytics();

    ImGui::Separator();
    float hw = (ImGui::GetContentRegionAvail().x - 4) * 0.5f;
    if (ImGui::Button("Reset to Defaults", ImVec2(hw, 22))) resetToDefaults();
    ImGui::SameLine(0, 4);
    if (ImGui::Button("About", ImVec2(hw, 22))) m_showAbout = true;

    ImGui::Separator();
    ImGui::TextDisabled("Export CSV");
    if (ImGui::Button("Rot. Curve##csv",  ImVec2(hw, 22))) exportRotationCurveCSV();
    ImGui::SameLine(0, 4);
    if (ImGui::Button("Energy##csv",      ImVec2(hw, 22))) exportEnergyCSV();
    if (ImGui::Button("Density##csv",     ImVec2(hw, 22))) exportDensityCSV();
    ImGui::SameLine(0, 4);
    if (ImGui::Button("Barycenters##csv", ImVec2(hw, 22))) exportBarycentersCSV();
    if (ImGui::Button("Fourier Modes##csv", ImVec2(-1, 22))) exportFourierCSV();

    ImGui::Separator();
    if (ImGui::Button("Reset Cam", ImVec2(-1, 22)))
    {
        m_edgeOnView = false;
        delete m_camera;
        m_camera = new Camera(m_renderer->getWidth(), m_renderer->getHeight(),
            new Vector(0, 0, 2000.0), new Matrix(ROTATION, 0.0, 0.0, 0.0),
            60.0, 10.0, 1.0e9f);
        m_camera->setOrbitMode(false);
    }

    ImGui::End();
    ImGui::PopStyleVar(2);

    renderGrid();
    renderVelocityField();
    renderRotationCurve();
    renderDensityProfile();
    renderPhaseSpace();
    renderEnergyPlot();
    renderDispersionPlot();
    renderBarycenters();
    renderFourierModes();
    renderDarkMatterProfile();
    renderEquationEditor();
    renderConservation();
    renderEscapeVelocity();
    renderCorrelation();
    renderSersic();
    renderHaloFinder();
    renderAbout();
    renderMarkerPaths();
    renderMarkers();
    renderAnnotations();

    {
        const float pW = 340.0f;
        const float vX = mvX + pW;
        const float vW = (float)m_width - pW;

        double eMY = (double)m_currentFrame * (double)m_timeStep;
        char timeBuf[64];
        if (eMY < 1000.0)
            snprintf(timeBuf, sizeof(timeBuf), "Time : %.2f MY", eMY);
        else
            snprintf(timeBuf, sizeof(timeBuf), "Time : %.3f BY", eMY / 1000.0);

        // MY/sec calculation (updated every ~1 sec)
        Uint32 nowMs = SDL_GetTicks();
        if (nowMs - m_mySecLastRealMs >= 1000 && m_simRunning && !m_simPaused)
        {
            float dtSec = (nowMs - m_mySecLastRealMs) / 1000.0f;
            float dtMY  = (float)(eMY - m_mySecLastSimTime);
            m_simMyPerSec       = dtMY / dtSec;
            m_mySecLastSimTime  = eMY;
            m_mySecLastRealMs   = nowMs;
        }

        // Real wall-clock run time (only accumulates while running)
        if (m_wallPrevMs != 0 && m_simRunning && !m_simPaused)
            m_simWallSec += (nowMs - m_wallPrevMs) / 1000.0;
        m_wallPrevMs = nowMs;
        int wsec = (int)m_simWallSec;
        char wallBuf[32];
        snprintf(wallBuf, sizeof(wallBuf), "%02d:%02d:%02d", wsec / 3600, (wsec / 60) % 60, wsec % 60);

        // Physics model name
        const char* modelNames[] = {"Newton","Janus","Yukawa","MOND","Custom"};
        const char* modelName = (m_eqSelectedModel >= 0 && m_eqSelectedModel <= 4)
                                ? modelNames[m_eqSelectedModel] : "?";

        char overlayBuf[160];
        if (m_simMyPerSec > 0.0f)
            snprintf(overlayBuf, sizeof(overlayBuf), "%s  |  %s  |  %.1f MY/s  |  run %s",
                     timeBuf, modelName, m_simMyPerSec, wallBuf);
        else
            snprintf(overlayBuf, sizeof(overlayBuf), "%s  |  %s  |  run %s",
                     timeBuf, modelName, wallBuf);

        ImVec2 textSize = ImGui::CalcTextSize(overlayBuf);
        float  tx = vX + vW - textSize.x - 10.0f;
        float  ty = mvY + 8.0f;

        // Background draw list: the HUD stays above the 3D scene but goes UNDER the
        // option windows (the foreground used to cover them).
        ImDrawList* fg = ImGui::GetBackgroundDrawList(ImGui::GetMainViewport());
        fg->AddRectFilled(
            ImVec2(tx - 4, ty - 2),
            ImVec2(tx + textSize.x + 4, ty + textSize.y + 2),
            IM_COL32(0, 0, 0, 140));
        fg->AddText(ImVec2(tx, ty), IM_COL32(220, 220, 100, 255), overlayBuf);
    }
}


void App::run()
{
    bool running = true;
    const int TARGET_FPS = 60;
    const int FRAME_DELAY_MS = 1000 / TARGET_FPS;
    
    while (running)
    {
        Uint32 frameStart = SDL_GetTicks();
        
        processEvents(running);
        updateCamera();
        updateAutoFit();    // smooth auto-fit (follows the expansion)

        if (m_replayMode)
            tickReplay();                       // scrubbing recorded frames (sim frozen)
        else if (m_simRunning && !m_simPaused)
            stepSimulation();

        // In Soft Halo the SPH rendering is already bright -> we cut the post glow.
        m_renderer->setGlow(m_softHalo ? false : m_glowEnabled, m_blurRadius, m_glowIntensity);

        m_renderer->beginFrame();
        float viewProj[16];
        m_camera->getViewProjMatrix(viewProj);
        glm::vec3 posPal[32];
        glm::vec3 negPal[32];
        for (int i = 0; i < 32; i++)
        {
            // Universe-style : positives switch to vel palette when CBV is on,
            // but negatives ALWAYS use the fuchsia neg palette (indexed by
            // |speed| via the signed-mass attribute) so + and - remain visually
            // distinct even in velocity color mode.
            const float* pp = m_colorByVelocity ? m_velPalette[i] : m_posPalette[i];
            const float* np = m_negPalette[i];
            posPal[i] = glm::vec3(pp[0], pp[1], pp[2]);
            negPal[i] = glm::vec3(np[0], np[1], np[2]);
        }
        // In ORBIT, true distance to the center (otherwise |z|→0 in grazing z=0 view -> huge points).
        float cam_z;
        {
            const Vector* o = m_camera->getOrigin();
            if (m_camera->isEdgeOn())        cam_z = fabsf((float)o->y);
            else if (m_camera->isOrbitMode())cam_z = sqrtf((float)(o->x*o->x + o->y*o->y + o->z*o->z));
            else                             cam_z = fabsf((float)o->z);
        }
        float vpWf       = (float)(m_width - 340);
        float wuPerPx    = 2.0f * cam_z * tanf(60.0f * 3.14159265f / 360.0f) / vpWf;
        float softeningRef = (m_softening > 0.f) ? m_softening : 7.0f;
        float physSize   = m_particleSize    * softeningRef;
        float ptSize     = physSize / wuPerPx;
        ptSize = (ptSize < 0.5f) ? 0.5f : (ptSize > 16.0f) ? 16.0f : ptSize;

        float physSizeNeg = m_particleSizeNeg * softeningRef;
        float ptSizeNeg   = physSizeNeg / wuPerPx;
        ptSizeNeg = (ptSizeNeg < 0.5f) ? 0.5f : (ptSizeNeg > 16.0f) ? 16.0f : ptSizeNeg;

        float maxVal = m_colorByVelocity ? m_velMaxForColor : (float)Color::max_mass;

        if (m_rayTrace && m_simulation) {
            // ── Velocity-colored SPLATTING rendering (no voxels) ──
            if (!m_volumeReady) { m_volume.init(); m_volumeReady = true; }
            const void* dvel = m_simulation->getDeviceVelocities();
            int nPart = m_simulation->getParticleCount();
            volumeInit(nPart);                       // (re)allocate the velocity buffer
            volumeComputeSpeeds(dvel, nPart);        // |v| per particle + mean
            m_volume.uploadSpeeds(volumeSpeedData(), nPart);

            glm::mat4 vp = glm::make_mat4(viewProj);
            glm::vec3 pal[32];
            for (int i = 0; i < 32; i++)
                pal[i] = glm::vec3(m_nebulaPalette[i][0], m_nebulaPalette[i][1], m_nebulaPalette[i][2]);
            unsigned posVbo = m_renderer->getVBO();
            int fbW = m_renderer->getWidth(), fbH = m_renderer->getHeight();
            float velRef = volumeVelRef();
            // splat size in PIXELS: world radius (m_volHaloSize x softening) / wuPerPx
            float haloPx = (m_volHaloSize * softeningRef) / wuPerPx;
            haloPx = (haloPx < 1.0f) ? 1.0f : (haloPx > 200.0f ? 200.0f : haloPx);

            m_volume.renderHalo(posVbo, nPart, fbW, fbH, vp, velRef, pal,
                                haloPx, m_volGain, m_volExposure,
                                m_volColorByVel, m_showNegaMass);
            if (m_volStars)
                m_volume.renderStars(posVbo, nPart, vp, velRef, pal,
                                     m_volStarScale, m_volStarBright, m_volColorByVel, m_showNegaMass);
        } else if (m_softHalo) {
            // Volumetric SPH rendering with the "true galaxy colors" palette.
            glm::vec3 galPal[32];
            for (int i = 0; i < 32; i++)
                galPal[i] = glm::vec3(m_galaxyPalette[i][0], m_galaxyPalette[i][1], m_galaxyPalette[i][2]);
            m_renderer->setSPHScale(m_sphQuality);
            m_renderer->renderSPH(glm::make_mat4(viewProj), ptSize, m_showNegaMass,
                                  galPal, maxVal, m_sphBright, /*colorByDensity=*/true,
                                  60.0f, m_sphSmooth, m_sphCoreDetail);
        } else {
            m_renderer->render(glm::make_mat4(viewProj), ptSize, ptSizeNeg,
                               posPal, negPal, maxVal, m_showNegaMass, m_radius);
        }
        m_renderer->endFrame();

        if (m_markerPickPending && m_positions.size() > 0)
        {
            int picked = pickParticleCPU(m_markerPickX, m_markerPickY);
            if (picked >= 0 && picked < (int)m_positions.size())
            {
                bool already = false;
                for (int idx : m_markers) if (idx == picked) { already = true; break; }
                if (!already && m_markers.size() < 16)
                    m_markers.push_back(picked);
            }
            m_markerPickPending = false;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        renderGUI();
        ImGui::Render();

        int displayW, displayH;
        SDL_GL_GetDrawableSize(m_window, &displayW, &displayH);
        glViewport(0, 0, displayW, displayH);
        glClearColor(0.031f, 0.063f, 0.188f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Deferred screenshot: back buffer now has particles + overlays + annotations
        captureScreenshotNow();
        captureVideoFrame();   // stream this frame to ffmpeg if recording

        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            SDL_GL_MakeCurrent(m_window, m_glContext);
        }

        SDL_GL_SwapWindow(m_window);

        // Deferred generation countdown:
        // 2→1: button drawn this frame, text will show next
        // 1→0: "Generating..." shown this frame, now launch
        if (m_generateCountdown > 0)
        {
            --m_generateCountdown;
            if (m_generateCountdown == 0)
                startSimulation();
        }

        Uint32 frameTime = SDL_GetTicks() - frameStart;
        if (frameTime < FRAME_DELAY_MS)
            SDL_Delay(FRAME_DELAY_MS - frameTime);
    }
}
