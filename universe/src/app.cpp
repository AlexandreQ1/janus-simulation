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
#include "universe_generator.h"
#include "gl_interop.cuh"   // CUDA-OpenGL interop : VBO construit directement sur le GPU
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cstdlib>   // std::system
#include <filesystem>
#include <unordered_map>
#include <functional>
#include <cstdint>
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

// Format a distance in LY with appropriate unit (LY / kly / Mly / Gly)
static void fmtLY(float ly, char* buf, int n)
{
    if      (ly >= 1e9f)  snprintf(buf, n, "%.2f Gly", ly / 1e9f);
    else if (ly >= 1e6f)  snprintf(buf, n, "%.2f Mly", ly / 1e6f);
    else if (ly >= 1e3f)  snprintf(buf, n, "%.1f kly", ly / 1e3f);
    else                   snprintf(buf, n, "%.0f LY",  ly);
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
    , m_totalSolarMass(1.0f)        // masse par particule positive
    , m_negaDensity(-1.0f)          // Janus : |m-| = |m+| → répulsion nette nulle à symétrie, structures possibles
    , m_numPosiStars(100000)
    , m_numNegaStars(100000)
    , m_pertAmp(0.5f)               // legacy seed
    , m_H0(0.003f)                  // H0 légèrement < H0_crit (~0.004)
    , m_scaleUnit(0.0005f)
    , m_blackHoleMass(0.0f)
    , m_rotSpeed(0.0f)
    , m_initSpeed(0.0f)
    , m_timeStep(0.5f)
    , m_softening(1.0f)             // ε=1 → résolution suffisante pour halos ~100 particules
    , m_gravConstant(1e-12f)
    , m_omegaM(0.3f)
    , m_omegaL(0.7f)
    , m_aStart(0.12f)   // t0≈0.22 Gyr; D1≈0.064 so Zel'dovich pairs resolve on PM grid
    , m_cosmoScale(0.014f)
    , m_dtMin(0.01f)
    , m_dtMax(2.0f)
    , m_eta(0.2f)
    , m_useCosmology(true)
    , m_adaptiveDt(true)
    , m_usePM(false)
    , m_pmEnableSR(true)
    , m_pmPeriodic(false)
    , m_pmNg(128)
    , m_useTree(false)
    , m_treeTheta(0.5f)
    , m_treeQuad(true), m_treeWarp(true)
    , m_useManualSeed(false)
    , m_manualSeed(42)
    , m_batchActive(false)
    , m_batchN(20)
    , m_batchCurrentIdx(0)
    , m_batchStartSeed(1)
    , m_batchTargetMY(2000.0f)
    , m_batchState(0)
    , m_wallClockStartMs(0)
    , m_wallClockAccumMs(0)
    , m_galaxyCount(0)
    , m_galaxyMinPart(50)
    , m_firstGalaxyDetected(false)
    , m_firstGalaxyTime(0.0f)
    , m_showGalaxyMarkers(true)
    , m_simRunning(false)
    , m_spAmplitude(0.0f)
    , m_spPatternSpeed(0.01f)
    , m_spPitchAngle(15.0f)
    , m_spArms(2)
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
    , m_phaseRMaxLY(0.0f)        // 0 = auto-fit
    , m_energyTimeRangeMY(0.0f)  // 0 = show all
    , m_baryTimeRangeMY(0.0f)    // 0 = show all
    , m_showConservation(false)
    , m_showHaloFinder(false)
    , m_fofLinkingScale(0.2f), m_fofMinMembers(20)
    , m_fofIncludeNeg(false), m_fofOverlay(true), m_fofDone(false)
    , m_showEscape(false)
    , m_showCorrelation(false)
    , m_showSersic(false)
    , m_corrDone(false)
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
    , m_lyPerUnit(3260000.0f)  // 1 sim-unit = 1 Mpc → Init Radius 500 ≈ 500 Mpc ≈ 1.63 Gly
    , m_showRotCurve(false)
    , m_showDensityProfile(false)
    , m_showPhaseSpace(false)
    , m_showEnergyPlot(false)
    , m_rotAutoRefresh(true)
    , m_rotRefreshInterval(100)
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
    , m_baryPosX(0), m_baryPosY(0), m_baryNegX(0), m_baryNegY(0)
    , m_showFourier(false)
    , m_showPowerSpectrum(false)
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
}

//  Friends-of-Friends halo finder + per-cluster virial ratio (à la demande)
//  Grid-accelerated union-find. Linking length b = scale × mean inter-particle
//  separation (Davis et al. 1985, standard b=0.2).
// ---------------------------------------------------------------------------
// Runs on a worker thread. Reads snapshot buffers (m_fofPosCopy/Vel), writes
// m_fofResult; main thread swaps it into m_fofClusters when ready.
void App::computeFoF()
{
    const std::vector<cl_float4>& POS = m_fofPosCopy;
    const std::vector<cl_float4>& VEL = m_fofVelCopy;
    m_fofResult.clear();
    if (POS.empty()) return;

    std::vector<int> idx;
    idx.reserve(POS.size());
    for (int i = 0; i < (int)POS.size(); i++) {
        float w = POS[i].w;
        if (m_fofNegSnap) { if (w < 0.0f) idx.push_back(i); }
        else              { if (w > 0.0f) idx.push_back(i); }
    }
    int N = (int)idx.size();
    if (N < 2) return;

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

    float invB = 1.0f / b;
    std::unordered_map<uint64_t, std::vector<int>> grid;
    grid.reserve(N * 2);
    auto cellKey = [&](float x, float y, float z) -> uint64_t {
        uint64_t cx = (uint64_t)(uint32_t)(int32_t)floorf((x - mn[0]) * invB);
        uint64_t cy = (uint64_t)(uint32_t)(int32_t)floorf((y - mn[1]) * invB);
        uint64_t cz = (uint64_t)(uint32_t)(int32_t)floorf((z - mn[2]) * invB);
        return (cx & 0x1FFFFF) | ((cy & 0x1FFFFF) << 21) | ((cz & 0x1FFFFF) << 42);
    };
    for (int k = 0; k < N; k++) {
        const cl_float4& p = POS[idx[k]];
        grid[cellKey(p.x, p.y, p.z)].push_back(k);
    }

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
                if (m <= k) continue;
                const cl_float4& q = POS[idx[m]];
                float ddx=q.x-p.x, ddy=q.y-p.y, ddz=q.z-p.z;
                if (ddx*ddx + ddy*ddy + ddz*ddz <= b2) unite(k, m);
            }
        }
    }

    std::unordered_map<int, std::vector<int>> groups;
    for (int k = 0; k < N; k++) groups[find(k)].push_back(k);

    // G effectif COHÉRENT avec l'intégrateur (calibration-free) : le kick cosmologique
    // applique a_inv3 = 1/a³ à l'accélération, donc le G dynamique = G_sim/a³ avec
    // G_sim = g_sim×1e10. Utiliser le MÊME ici rend KE/|PE| dimensionnellement consistant
    // -> les nœuds relaxés lisent ~0.5 à TOUTE époque (l'ancien 1.1e11/a ne coïncidait avec
    // 1/a³ qu'à a≈0.3 / 1.66 Gyr, et dérivait -> tout rouge ailleurs).
    float a3   = m_fofAsnap * m_fofAsnap * m_fofAsnap;
    float G_eff = m_fofGsnap * 1.0e10f / a3;
    float H_fof = m_fofHsnap;   // retrait du flot de Hubble dans le KE (v_pec = v - H·x)
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
                const cl_float4& v = VEL[idx[li]];           // vitesse péculière = v - H·x
                cvx += (v.x - H_fof*p.x)*am; cvy += (v.y - H_fof*p.y)*am; cvz += (v.z - H_fof*p.z)*am;
            }
        }
        if (cm <= 0) continue;
        c.mass = (float)cm;
        c.cx = (float)(cx/cm); c.cy = (float)(cy/cm); c.cz = (float)(cz/cm);
        c.vx = (float)(cvx/cm); c.vy = (float)(cvy/cm); c.vz = (float)(cvz/cm);

        double r2sum = 0;
        for (int li : mem) {
            const cl_float4& p = POS[idx[li]];
            float ddx=p.x-c.cx, ddy=p.y-c.cy, ddz=p.z-c.cz;
            r2sum += ddx*ddx + ddy*ddy + ddz*ddz;
        }
        c.radius = sqrtf((float)(r2sum / mem.size()));

        double KE = 0;
        if (haveVel) {
            for (int li : mem) {
                const cl_float4& p = POS[idx[li]];
                const cl_float4& v = VEL[idx[li]];
                // dispersion PÉCULIÈRE interne : (v - H·x) - <v_pec> -> exclut l'expansion
                float dvx=(v.x - H_fof*p.x)-c.vx, dvy=(v.y - H_fof*p.y)-c.vy, dvz=(v.z - H_fof*p.z)-c.vz;
                KE += 0.5 * fabsf(p.w) * (dvx*dvx + dvy*dvy + dvz*dvz);
            }
        }

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
        // Store global particle indices for member highlighting
        c.members.reserve(mem.size());
        for (int li : mem) c.members.push_back(idx[li]);
        m_fofResult.push_back(c);
    }

    std::sort(m_fofResult.begin(), m_fofResult.end(),
              [](const FoFCluster& a, const FoFCluster& b){ return a.mass > b.mass; });
}

// ─── FoF cluster overlay (viewport circles) ──────────────────────────────────
// Drawn every frame from the main render loop so the markers stay visible whether
// or not the Halo Finder window is open. Background draw list = above the 3D scene
// but below the ImGui windows. Circles follow live member positions, colour by virial.
void App::renderFoFOverlay()
{
    if (!m_fofOverlay || !m_fofDone || m_fofClusters.empty() || !m_camera) return;

    ImGuiViewport* gMV = ImGui::GetMainViewport();
    const float gMVX = gMV->Pos.x, gMVY = gMV->Pos.y;
    const float panelW = 340.0f;
    const float vpX = gMVX + panelW;
    const float vpW = (float)m_width - panelW, vpH = (float)m_height;
    if (vpW <= 0 || vpH <= 0) return;

    float vp[16]; m_camera->getViewProjMatrix(vp);
    ImDrawList* dl = ImGui::GetBackgroundDrawList(gMV);
    dl->PushClipRect(ImVec2(vpX, gMVY), ImVec2(vpX+vpW, gMVY+vpH), false);
    bool liveOK = (m_positions.size() > 0);

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
            // White + larger so the selected cluster stands out in the dense field.
            dl->AddRectFilled(ImVec2(s.x-2.0f, s.y-2.0f), ImVec2(s.x+2.0f, s.y+2.0f),
                              IM_COL32(255, 255, 255, 255));
        }
    }

    const float SR = 9.0f;   // petit cercle FIXE (px) — la taille variable n'apportait rien
    for (size_t i = 0; i < m_fofClusters.size(); i++) {
        const FoFCluster& c = m_fofClusters[i];
        if (c.virial > 1.0f) continue;   // masquer les groupes NON LIÉS (KE/|PE| > 1)
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
        ImU32 col = (v < 0.4f) ? IM_COL32(120,190,255,255)    // collapsing = bleu
                               : IM_COL32(120,255,120,255);   // virialisé = vert
        dl->AddCircle(sc, SR, col, 0, 2.6f);
        dl->AddCircleFilled(sc, 2.0f, col);
        char tag[16]; snprintf(tag, sizeof(tag), "%zu", i+1);
        ImVec2 ts = ImGui::CalcTextSize(tag);
        dl->AddRectFilled(ImVec2(sc.x+SR+1, sc.y-ts.y*0.5f-1),
                          ImVec2(sc.x+SR+5+ts.x, sc.y+ts.y*0.5f+1), IM_COL32(0,0,0,150));
        dl->AddText(ImVec2(sc.x + SR + 3, sc.y - ts.y*0.5f), col, tag);
    }
    dl->PopClipRect();
}

void App::renderHaloFinder()
{
    if (!m_showHaloFinder) return;
    ImGui::SetNextWindowSize(ImVec2(560, 460), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(360, 30), ImGuiCond_FirstUseEver);
    ImGui::Begin("Halo Finder (FoF)", &m_showHaloFinder, ImGuiWindowFlags_NoCollapse);

    // (Viewport cluster circles are drawn by renderFoFOverlay() from the main loop.)

    ImGui::TextDisabled("Friends-of-Friends (Davis 1985): b = scale x mean separation");
    SliderFloatWithInput("Linking b/sep", &m_fofLinkingScale, 0.05f, 0.5f, "%.2f");
    ImGui::SetNextItemWidth(160);
    // Jusqu'à 20 000 : galaxie ≈ 200 membres -> amas (≥ dizaines de galaxies) ≈ 2 000-20 000.
    // Échelle logarithmique pour rester maniable sur une si large plage.
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
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.20f,0.55f,0.25f,1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f,0.68f,0.30f,1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.16f,0.45f,0.20f,1.0f));
        if (ImGui::Button("Run FoF", ImVec2(-1, 26)) && m_simulation) {
            m_fofLive = true;
            m_fofLiveLastFrame = -100000;   // trigger first detection immediately
        }
        ImGui::PopStyleColor(3);
    } else {
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
            m_fofHsnap   = (m_useCosmology && m_simulation) ? m_simulation->getHubbleParam() : 0.0f;
            m_fofAsnap   = (m_useCosmology && m_simulation) ? m_simulation->getScaleFactor() : 1.0f;
            if (m_fofAsnap < 1e-3f) m_fofAsnap = 1e-3f;
            m_fofLinkSnap= m_fofLinkingScale;
            m_fofMinSnap = m_fofMinMembers;
            m_fofNegSnap = m_fofIncludeNeg;
            if (m_fofThread.joinable()) m_fofThread.join();
            m_fofCancel      = false;
            m_fofResultReady = false;
            m_fofComputing   = true;
            m_fofThread = std::thread([this]{
                // try/catch : un bad_alloc sur très grand N (grille FoF) s'échapperait sinon
                // du thread -> std::terminate -> appli figée (force-kill). On sort proprement.
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
            ImVec4 col = (v < 0.4f) ? ImVec4(0.5f,0.7f,1.0f,1.0f)
                       : (v > 1.0f) ? ImVec4(1.0f,0.5f,0.4f,1.0f)
                                    : ImVec4(0.5f,1.0f,0.5f,1.0f);
            ImGui::TextColored(col, "%.2f", v);
        }
        ImGui::EndTable();
    }
    ImGui::End();
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

        // Récupération des données dans un buffer plat (linéaire en temps)
        std::vector<float> vis(cnt);
        for (int i=0;i<cnt;i++) {
            int idx=(head-cnt+i+histSize)%histSize;
            vis[i] = hist[idx];
        }
        // Lissage Gaussien (σ adaptatif)
        int sigB = std::max(3, std::min(30, cnt / 20));
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

        ImVec2 prev={0,0}; bool first=true;
        for (int i=0;i<cnt;i++) {
            float sx = px + (float)i/(cnt>1?cnt-1:1)*pw;
            float sy = py + (1.0f-(sm[i]-vMin)/(vMax-vMin))*ph;
            ImVec2 cur(sx,sy);
            if(!first) dl->AddLine(prev,cur,col,1.8f);
            prev=cur; first=false;
        }
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

    // Récupération buffer plat
    std::vector<float> visV(m_escCount);
    for (int i=0;i<m_escCount;i++) {
        int idx=(m_escHead-m_escCount+i+ESC_HISTORY)%ESC_HISTORY;
        visV[i] = m_negVelHistory[idx];
    }
    // Lissage Gaussien (σ adaptatif)
    int sigB  = std::max(3, std::min(30, m_escCount / 20));
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

    ImVec2 prev={0,0}; bool first=true;
    for (int i=0; i<m_escCount; i++) {
        float sx=px+(float)i/(m_escCount>1?m_escCount-1:1)*pw;
        float sy=py+(1.0f-(sm[i]-vMin)/(vMax-vMin))*ph;
        ImVec2 cur(sx,sy);
        if (!first) dl->AddLine(prev,cur,IM_COL32(255,140,40,220),2.0f);
        prev=cur; first=false;
    }
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

    // Lissage Gaussien des valeurs ξ(r) avant affichage
    int N = (int)m_corrXi.size();
    std::vector<float> xiSm(N);
    if (N > 0) {
        int sigB  = std::max(2, std::min(8, N / 10));
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

    // PlotLines — native ImGui widget, no coordinate issues
    ImVec2 plotSize = ImGui::GetContentRegionAvail();
    plotSize.y -= 24;  // leave room for axis label

    // Adaptive Y scale based on smoothed values
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
    char cBuf1[32], cBuf2[32];
    fmtLY(m_corrRMax * 0.5f, cBuf1, sizeof(cBuf1));
    fmtLY(m_corrRMax,        cBuf2, sizeof(cBuf2));
    ImGui::Text("0 LY");
    ImGui::SameLine(ImGui::GetContentRegionAvail().x * 0.5f - 20);
    ImGui::Text("%s", cBuf1);
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 60);
    ImGui::Text("%s", cBuf2);

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
        float r = sqrtf(m_positions[i].x*m_positions[i].x + m_positions[i].y*m_positions[i].y)
                  * m_lyPerUnit;
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
    { char reBuf[32]; fmtLY(Re, reBuf, sizeof(reBuf));
      ImGui::Text("Sersic n = %.2f   |   Re = %s", bestN, reBuf); }
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
    fmtLY(rMax, buf, sizeof(buf));
    {ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px+pw-ts.x,py+ph+2),IM_COL32(160,160,160,200),buf);}
    dl->AddText(ImVec2(px,py+ph+2),IM_COL32(160,160,160,200),"0");
    ImGui::Dummy(cvs);
    ImGui::End();
}

// ---------------------------------------------------------------------------
// Reset to defaults
// ---------------------------------------------------------------------------

void App::resetToDefaults()
{
    m_radius         = 500.0f;
    m_scaleUnit      = 1.0f;
    m_numPosiStars   = 100000;
    m_numNegaStars   = 100000;
    m_totalSolarMass = 1.0f;        // masse par particule positive (simulation univers)
    m_negaDensity    = -1.0f;       // Janus : |m-| = |m+| → répulsion nette nulle à symétrie
    m_blackHoleMass  = 0.0f;
    m_rotSpeed       = 0.0f;
    m_initSpeed      = 0.0f;
    m_timeStep       = 0.5f;
    m_softening      = 1.0f;        // ε=1 → résolution suffisante
    m_gravConstant   = 1e-12f;
    m_H0             = 0.003f;
    m_pertAmp        = 0.5f;
    // Cosmologie
    m_omegaM         = 0.3f;
    m_omegaL         = 0.7f;
    m_aStart         = 0.5f;
    m_cosmoScale     = 0.02f;
    m_dtMin          = 0.01f;
    m_dtMax          = 2.0f;
    m_eta            = 0.2f;
    m_useCosmology   = true;
    m_adaptiveDt     = true;
    m_usePM          = false;
    m_pmEnableSR     = true;
    m_pmPeriodic     = false;  // cube périodique retiré
    m_pmNg           = 128;
    m_useTree        = false;
    m_treeTheta      = 0.5f;
    m_treeQuad       = true;
    m_treeWarp       = true;
    m_useManualSeed  = false;
    m_manualSeed     = 42;
    m_batchActive    = false;
    m_batchN         = 20;
    m_batchCurrentIdx= 0;
    m_batchStartSeed = 1;
    m_batchTargetMY  = 2000.0f;
    m_batchState     = 0;
    strncpy(m_batchOutDir, "C:\\Users\\Administrateur\\Desktop\\batch_runs", sizeof(m_batchOutDir)-1);
    m_batchOutDir[sizeof(m_batchOutDir)-1] = '\0';
    m_simSpeed       = 1.0f;
    m_spAmplitude    = 0.0f;
    m_spPatternSpeed = 0.01f;
    m_spPitchAngle   = 15.0f;
    m_spArms         = 2;
    m_glowEnabled    = true;
    m_blurRadius     = 4.0f;
    m_glowIntensity  = 2.5f;
    m_showNegaMass   = true;
    m_colorByVelocity = false;
    m_colorByDensity  = false;
    m_colorByHalo     = false;
    m_colorByWeb      = false;
    m_colorByTemp     = false;
    m_velAutoScale   = true;
    m_velPeakHold    = false;
    m_showGrid       = true;
    m_lyPerUnit      = 3260000.0f;
    m_particleSize   = 0.4f;
    m_particleSizeNeg= 0.4f;
    m_rotCurveRMax   = 100000.0f;
    m_eqSelectedModel = 1;          // Janus par défaut
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
    float velConv = (m_lyPerUnit * 9.461e12f) / 3.156e13f;
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
    // Diagnostic : si pas de particules, on écrit juste un message d'erreur.
    if (m_positions.empty()) {
        std::ofstream f(filename);
        if (f.is_open())
            f << "# ERROR: no particles loaded (m_positions is empty).\n"
              << "# Click 'Apply & Generate' then 'Refresh Analytics' before exporting.\n";
        return;
    }

    const int BINS = 80;
    float countPos[BINS]={}, countNeg[BINS]={};

    // Étendue radiale 3D réelle (pas 2D)
    float rMaxSim = 0.0f;
    for (size_t i = 0; i < m_positions.size(); i++) {
        float r2 = m_positions[i].x*m_positions[i].x
                 + m_positions[i].y*m_positions[i].y
                 + m_positions[i].z*m_positions[i].z;
        if (r2 > rMaxSim) rMaxSim = r2;
    }
    rMaxSim = sqrtf(rMaxSim);
    if (rMaxSim < 1.0f) rMaxSim = m_radius * 1.2f;
    float rMax = rMaxSim * m_lyPerUnit;

    for (size_t i = 0; i < m_positions.size(); i++) {
        float r3d = sqrtf(m_positions[i].x*m_positions[i].x
                        + m_positions[i].y*m_positions[i].y
                        + m_positions[i].z*m_positions[i].z) * m_lyPerUnit;
        int b = (int)(r3d / rMax * BINS); if (b < 0) b=0; if (b >= BINS) b=BINS-1;
        if (m_positions[i].w > 0) countPos[b]++; else if (m_positions[i].w < 0) countNeg[b]++;
    }

    std::ofstream f(filename);
    if (!f.is_open()) {
        // Cannot open the chosen file. Caller will see no file or read-only error.
        return;
    }
    f << "Radius_LY,Count_Positive,Count_Negative,Density_Pos_per_LY3,Density_Neg_per_LY3,Shell_Volume_LY3\n";
    const float FOUR_THIRDS_PI = 4.18879020478f;
    float binW = rMax / BINS;
    for (int b = 0; b < BINS; b++) {
        float r0 = b * binW;
        float r1 = (b + 1) * binW;
        float vol = FOUR_THIRDS_PI * (r1*r1*r1 - r0*r0*r0);
        float densPos = (vol > 1e-12f) ? countPos[b] / vol : 0.0f;
        float densNeg = (vol > 1e-12f) ? countNeg[b] / vol : 0.0f;
        f << (b + 0.5f) * binW << ","
          << (int)countPos[b] << "," << (int)countNeg[b] << ","
          << densPos << "," << densNeg << ","
          << vol << "\n";
    }
    f.flush();
    f.close();
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
    ImGui::SetNextWindowSize(ImVec2(560, 520), ImGuiCond_Always);
    // Placée à DROITE (loin des panneaux d'options à gauche) et DÉPLAÇABLE (FirstUseEver)
    // -> ne masque plus les options ; l'utilisateur peut la bouger ou la fermer.
    float aboutX = (float)m_width - 580.0f; if (aboutX < 360.0f) aboutX = 360.0f;
    ImGui::SetNextWindowPos(ImVec2(aboutX, 40.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("About Janus Universe", &m_showAbout,
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

    ImGui::TextColored(ImVec4(0.4f,0.8f,1.0f,1), "Janus Universe - Cosmological Simulator");
    ImGui::Separator();

    ImGui::TextColored(ImVec4(0.8f,0.8f,0.4f,1), "Project origin");
    ImGui::TextWrapped("42 School Silicon Valley");
    if (ImGui::SmallButton("https://j-gravity.github.io/web-landing/index.html"))
        ShellExecuteA(NULL,"open","https://j-gravity.github.io/web-landing/index.html",NULL,NULL,SW_SHOWNORMAL);
    ImGui::Dummy(ImVec2(0,6));

    ImGui::TextColored(ImVec4(0.8f,0.8f,0.4f,1), "Janus Cosmological Model - Jean-Pierre Petit");
    ImGui::TextWrapped("Publications:");
    auto link = [](const char* url) {
        if (ImGui::SmallButton(url))
            ShellExecuteA(NULL,"open",url,NULL,NULL,SW_SHOWNORMAL);
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
#ifdef JANUS_WITH_3DCONNEXION
    ImGui::Dummy(ImVec2(0, 4));
    ImGui::TextWrapped("3D input device development tools and related technology are provided "
                       "under license from 3Dconnexion. \xC2\xA9 3Dconnexion 1992 - 2025. "
                       "All rights reserved.");
#endif

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

    f << "# J-Gravity settings\n";
    f << "[Galaxy]\n";
    f << "radius="          << m_radius          << "\n";
    f << "scale_unit="      << m_scaleUnit        << "\n";
    f << "num_posi_stars="  << m_numPosiStars     << "\n";
    f << "num_nega_stars="  << m_numNegaStars     << "\n";
    f << "total_solar_mass="<< m_totalSolarMass   << "\n";
    f << "nega_density="    << m_negaDensity      << "\n";
    f << "pert_amp="        << m_pertAmp          << "\n";
    f << "web_cells="       << m_webCells         << "\n";
    f << "foam_mode="       << (m_foamMode?1:0)   << "\n";
    f << "black_hole_mass=" << m_blackHoleMass    << "\n";
    f << "rot_speed="       << m_rotSpeed         << "\n";
    f << "init_speed="      << m_initSpeed        << "\n";
    f << "[Simulation]\n";
    f << "time_step="       << m_timeStep         << "\n";
    f << "softening="       << m_softening        << "\n";
    f << "grav_constant="   << m_gravConstant     << "\n";
    f << "sim_speed="       << m_simSpeed         << "\n";
    f << "[DensityWaves]\n";
    f << "sp_amplitude="    << m_spAmplitude      << "\n";
    f << "sp_pattern_speed="<< m_spPatternSpeed   << "\n";
    f << "sp_pitch_angle="  << m_spPitchAngle     << "\n";
    f << "sp_arms="         << m_spArms           << "\n";
    f << "[Glow]\n";
    f << "glow_enabled="    << (m_glowEnabled ? 1 : 0) << "\n";
    f << "soft_halo="       << (m_softHalo ? 1 : 0) << "\n";
    f << "sph_bright="      << m_sphBright        << "\n";
    f << "sph_viridis="     << (m_sphViridis ? 1 : 0) << "\n";
    f << "sph_palette="     << m_sphPalette << "\n";
    f << "sph_smooth="      << m_sphSmooth        << "\n";
    f << "sph_core_detail=" << m_sphCoreDetail    << "\n";
    f << "color_by_dens="   << (m_colorByDensity ? 1 : 0) << "\n";
    f << "color_by_halo="   << (m_colorByHalo ? 1 : 0) << "\n";
    f << "color_by_web="    << (m_colorByWeb ? 1 : 0) << "\n";
    f << "color_by_temp="   << (m_colorByTemp ? 1 : 0) << "\n";
    f << "blur_radius="     << m_blurRadius       << "\n";
    f << "glow_intensity="  << m_glowIntensity    << "\n";
    f << "[Display]\n";
    f << "show_nega_mass="  << (m_showNegaMass ? 1 : 0) << "\n";
    f << "color_by_vel="    << (m_colorByVelocity ? 1 : 0) << "\n";
    f << "show_grid="       << (m_showGrid ? 1 : 0) << "\n";
    f << "ly_per_unit="     << m_lyPerUnit        << "\n";
    f << "particle_size="    << m_particleSize    << "\n";
    f << "particle_size_neg="<< m_particleSizeNeg << "\n";
    f << "[Cosmology]\n";
    f << "H0="              << m_H0              << "\n";
    f << "omegaM="          << m_omegaM          << "\n";
    f << "omegaL="          << m_omegaL          << "\n";
    f << "aStart="          << m_aStart          << "\n";
    f << "dtMin="           << m_dtMin           << "\n";
    f << "dtMax="           << m_dtMax           << "\n";
    f << "eta="             << m_eta             << "\n";
    f << "cosmoScale="      << m_cosmoScale      << "\n";
    f << "useCosmology="    << (m_useCosmology ? 1 : 0) << "\n";
    f << "remove_monopole=" << (m_removeMonopole ? 1 : 0) << "\n";
    f << "monopole_frac="   << m_monopoleFrac     << "\n";
    f << "adaptiveDt="      << (m_adaptiveDt ? 1 : 0) << "\n";
    f << "usePM="           << (m_usePM ? 1 : 0) << "\n";
    f << "pmEnableSR="      << (m_pmEnableSR ? 1 : 0) << "\n";
    f << "pmPeriodic="      << (m_pmPeriodic ? 1 : 0) << "\n";
    f << "pmNg="            << m_pmNg << "\n";
    f << "useTree="         << (m_useTree ? 1 : 0) << "\n";
    f << "treeTheta="       << m_treeTheta << "\n";
    f << "treeQuad="        << (m_treeQuad ? 1 : 0) << "\n";
    f << "treeWarp="        << (m_treeWarp ? 1 : 0) << "\n";
    f << "galaxyMinPart="   << m_galaxyMinPart << "\n";
    f << "galaxyLinkScale=" << m_galaxyLinkScale << "\n";
    f << "showGalaxyMarkers="<< (m_showGalaxyMarkers ? 1 : 0) << "\n";
    f << "targetFps="       << m_targetFps << "\n";
    f << "useManualSeed="   << (m_useManualSeed ? 1 : 0) << "\n";
    f << "manualSeed="      << m_manualSeed << "\n";
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
            else if (key == "pert_amp")        m_pertAmp         = std::stof(val);
            else if (key == "web_cells")       m_webCells        = std::stoi(val);
            else if (key == "foam_mode")       m_foamMode        = (std::stoi(val)!=0);
            else if (key == "voronoi_foam")    { /* option retirée : ignorée pour compat. anciens .cfg */ }
            else if (key == "remove_monopole") m_removeMonopole = std::stoi(val) != 0;
            else if (key == "monopole_frac")   m_monopoleFrac   = std::stof(val);
            else if (key == "black_hole_mass") m_blackHoleMass   = std::stof(val);
            else if (key == "rot_speed")       m_rotSpeed        = std::stof(val);
            else if (key == "init_speed")      m_initSpeed       = std::stof(val);
            else if (key == "time_step")       m_timeStep        = std::stof(val);
            else if (key == "softening")       m_softening       = std::stof(val);
            else if (key == "grav_constant")   m_gravConstant    = std::stof(val);
            else if (key == "sim_speed")       m_simSpeed        = std::stof(val);
            else if (key == "sp_amplitude")    m_spAmplitude     = std::stof(val);
            else if (key == "sp_pattern_speed")m_spPatternSpeed  = std::stof(val);
            else if (key == "sp_pitch_angle")  m_spPitchAngle    = std::stof(val);
            else if (key == "sp_arms")         m_spArms          = std::max(1, std::min(20, std::stoi(val)));
            else if (key == "glow_enabled")    m_glowEnabled     = std::stoi(val) != 0;
            else if (key == "soft_halo")       m_softHalo        = std::stoi(val) != 0;
            else if (key == "sph_bright")      m_sphBright       = std::stof(val);
            else if (key == "sph_viridis")     m_sphViridis      = std::stoi(val) != 0;
            else if (key == "sph_thermal")     m_sphPalette      = (std::stoi(val) != 0) ? 1 : m_sphPalette; // migration ancienne clé
            else if (key == "sph_palette")     m_sphPalette      = std::max(0, std::min(2, std::stoi(val)));
            else if (key == "sph_smooth")      m_sphSmooth       = std::stof(val);
            else if (key == "sph_core_detail") m_sphCoreDetail   = std::stof(val);
            else if (key == "color_by_dens")   m_colorByDensity  = std::stoi(val) != 0;
            else if (key == "color_by_halo")   m_colorByHalo     = std::stoi(val) != 0;
            else if (key == "color_by_web")    m_colorByWeb      = std::stoi(val) != 0;
            else if (key == "color_by_temp")   m_colorByTemp     = std::stoi(val) != 0;
            else if (key == "blur_radius")     m_blurRadius      = std::stof(val);
            else if (key == "glow_intensity")  m_glowIntensity   = std::stof(val);
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
            else if (key == "H0")              m_H0     = std::stof(val);
            else if (key == "omegaM")          m_omegaM = std::stof(val);
            else if (key == "omegaL")          m_omegaL = std::stof(val);
            else if (key == "aStart")          m_aStart = std::stof(val);
            else if (key == "dtMin")           m_dtMin = std::stof(val);
            else if (key == "dtMax")           m_dtMax = std::stof(val);
            else if (key == "eta")             m_eta = std::stof(val);
            else if (key == "cosmoScale")      m_cosmoScale = std::stof(val);
            else if (key == "useCosmology")    m_useCosmology = std::stoi(val) != 0;
            else if (key == "adaptiveDt")      m_adaptiveDt = std::stoi(val) != 0;
            else if (key == "usePM")           m_usePM = std::stoi(val) != 0;
            else if (key == "pmEnableSR")      m_pmEnableSR = std::stoi(val) != 0;
            else if (key == "pmPeriodic")      m_pmPeriodic = std::stoi(val) != 0;
            else if (key == "pmNg")            m_pmNg = std::stoi(val);
            else if (key == "useTree")         m_useTree = std::stoi(val) != 0;
            else if (key == "treeTheta")       m_treeTheta = std::stof(val);
            else if (key == "treeQuad")        m_treeQuad = std::stoi(val) != 0;
            else if (key == "treeWarp")        m_treeWarp = std::stoi(val) != 0;
            else if (key == "galaxyMinPart")   m_galaxyMinPart = std::stoi(val);
            else if (key == "galaxyLinkScale") m_galaxyLinkScale = std::stof(val);
            else if (key == "showGalaxyMarkers") m_showGalaxyMarkers = std::stoi(val) != 0;
            else if (key == "targetFps")       m_targetFps = std::stoi(val);
            else if (key == "useManualSeed")   m_useManualSeed = std::stoi(val) != 0;
            else if (key == "manualSeed")      m_manualSeed = std::stoi(val);
            else if (key == "state_path")      strncpy(m_stateFilePath, val.c_str(), sizeof(m_stateFilePath)-1);
            else if (key == "cuda_device")     m_cudaDevice = std::stoi(val);
        } catch (...) {}
    }
}

App::~App()
{
    saveSettings("jgravity.cfg");

    if (m_videoRecording) stopVideoRecording();      // finalize MP4 if still recording
    m_fofCancel = true;                              // signal abort AVANT le join -> pas de blocage à la fermeture
    if (m_fofThread.joinable()) m_fofThread.join();  // wait for any FoF worker
    if (m_galThread.joinable()) m_galThread.join();  // wait for any galaxy-detect worker
    if (m_densThread.joinable()) m_densThread.join(); // wait for any density worker
    if (m_spaceMouse) { m_spaceMouse->disable(); delete m_spaceMouse; m_spaceMouse = nullptr; }
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

    m_window = SDL_CreateWindow("Janus Universe - Cosmological Model",
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

    // Police TTF (Segoe UI de Windows) avec glyphes ÉTENDUS -> remplace la ProggyClean
    // ASCII intégrée (accents, em-dash, σ, ȧ, flèches s'affichent au lieu de « ? »).
    {
        static const ImWchar ranges[] = {
            0x0020, 0x00FF,   // Latin de base + supplément (accents)
            0x0100, 0x024F,   // Latin étendu A/B (ȧ …)
            0x0370, 0x03FF,   // Grec (σ …)
            0x2000, 0x206F,   // Ponctuation générale (em-dash, …)
            0x2190, 0x21FF,   // Flèches
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
    // Green dropdown menus : collapsing-header bars + combo popups.
    style.Colors[ImGuiCol_Header]        = ImVec4(0.13f, 0.42f, 0.18f, 1.00f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.18f, 0.56f, 0.24f, 1.00f);
    style.Colors[ImGuiCol_HeaderActive]  = ImVec4(0.22f, 0.66f, 0.30f, 1.00f);
    style.Colors[ImGuiCol_PopupBg]       = ImVec4(0.06f, 0.16f, 0.09f, 0.98f);
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    ImGui_ImplSDL2_InitForOpenGL(m_window, m_glContext);
    ImGui_ImplOpenGL3_Init("#version 330");

    m_renderer = new PointRenderer(m_width, m_height);
    if (!m_renderer->init()) { std::cerr << "Failed to init PointRenderer\n"; return false; }

    int vpW0 = m_width - 340;
    int vpH0 = m_height;
    m_camera = new Camera(vpW0 > 0 ? vpW0 : m_width, vpH0,
        new Vector(0, 0, 2000.0), new Matrix(ROTATION, 0.0, 0.0, 0.0),
        60.0, 10.0, 1.0e10f);

    // 3Dconnexion (SpaceMouse) : créé seulement si le driver est présent (sinon no-op).
    m_spaceMouse = new SpaceMouseNav();

    Color::init_color_table();
    Color::ninit_color_table();
    Color::vel_init_color_table();
    Color::getPosPalette(m_posPalette);
    Color::getNegPalette(m_negPalette);
    Color::getVelPalette(m_velPalette);

    // Density palette ("dark matter" halo ramp) : bleu sombre -> bleu -> bleu clair
    // -> rose -> rouge -> blanc (cœur de halo). Calculée par densité locale 3D.
    {
        const float cs[6][3] = {
            {0.03f, 0.05f, 0.25f},  // vide : bleu sombre
            {0.10f, 0.38f, 0.95f},  // diffus : bleu
            {0.45f, 0.60f, 1.00f},  // filaments : bleu clair
            {0.92f, 0.28f, 0.72f},  // halos : rose/magenta
            {1.00f, 0.15f, 0.12f},  // cœurs : rouge (zone large)
            {1.00f, 0.95f, 0.85f}   // pic extreme seulement : blanc chaud
        };
        const float stops[6] = { 0.0f, 0.25f, 0.45f, 0.65f, 0.95f, 1.0f };
        for (int i = 0; i < 32; i++) {
            float t = i / 31.0f;
            int seg = 0; while (seg < 4 && t > stops[seg + 1]) seg++;
            float s = (t - stops[seg]) / (stops[seg + 1] - stops[seg] + 1e-6f);
            for (int k = 0; k < 3; k++)
                m_densPalette[i][k] = cs[seg][k] + (cs[seg + 1][k] - cs[seg][k]) * s;
        }
    }

    // Palette VIRIDIS (matplotlib) : violet foncé -> bleu -> teal -> vert -> jaune.
    // Pour le rendu SPH "carte de densité" façon py-sphviewer (la teinte encode la densité).
    {
        const float vc[11][3] = {
            {0.267f,0.005f,0.329f}, {0.283f,0.141f,0.458f}, {0.254f,0.265f,0.530f},
            {0.207f,0.372f,0.553f}, {0.164f,0.471f,0.558f}, {0.128f,0.567f,0.551f},
            {0.135f,0.659f,0.518f}, {0.267f,0.749f,0.441f}, {0.478f,0.821f,0.318f},
            {0.741f,0.873f,0.150f}, {0.993f,0.906f,0.144f}
        };
        for (int i = 0; i < 32; i++) {
            float t = i / 31.0f * 10.0f;     // 0..10 sur les 11 points de contrôle
            int   s = (int)t; if (s > 9) s = 9;
            float f = t - s;
            for (int k = 0; k < 3; k++)
                m_viridisPalette[i][k] = vc[s][k] + (vc[s+1][k] - vc[s][k]) * f;
        }
    }

    // Palette THERMIQUE (fond NOIR) : noir -> bleu nuit -> bleu -> cyan -> vert -> jaune ->
    // orange -> rouge -> blanc chaud. L'index 0 = noir -> vides et fond NOIRS automatiquement.
    {
        const float tc[10][3] = {
            {0.00f,0.00f,0.00f}, {0.04f,0.01f,0.18f}, {0.10f,0.10f,0.55f},
            {0.10f,0.45f,0.78f}, {0.10f,0.72f,0.60f}, {0.45f,0.86f,0.22f},
            {0.95f,0.86f,0.15f}, {1.00f,0.55f,0.06f}, {1.00f,0.22f,0.10f},
            {1.00f,0.92f,0.88f}
        };
        for (int i = 0; i < 32; i++) {
            float t = i / 31.0f * 9.0f;      // 0..9 sur les 10 points
            int   s = (int)t; if (s > 8) s = 8;
            float f = t - s;
            for (int k = 0; k < 3; k++)
                m_thermalPalette[i][k] = tc[s][k] + (tc[s+1][k] - tc[s][k]) * f;
        }
    }

    // Palette "GALAXY COLORS" (look Hubble) — répliquée du Soft Halo de la galaxie. La TEINTE
    // encode la densité de colonne. Faible -> forte : bleu sombre (halo diffus) -> bleu (bras
    // spiraux) -> blanc-crème (disque) -> or chaud (bulbe) -> blanc-doré (cœur brillant).
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

    // Palette T-web : 4 bandes distinctes (8 entrées chacune) — vide / paroi / filament / nœud.
    {
        const float wc[4][3] = {
            {0.04f, 0.05f, 0.12f},  // vide : sombre
            {0.15f, 0.40f, 0.95f},  // paroi (sheet) : bleu
            {0.20f, 0.90f, 0.45f},  // filament : vert
            {1.00f, 0.25f, 0.12f}   // nœud (knot) : rouge
        };
        for (int i = 0; i < 32; i++) {
            int band = i / 8; if (band > 3) band = 3;
            for (int k = 0; k < 3; k++) m_webPalette[i][k] = wc[band][k];
        }
    }

    // Palette température (froid -> chaud) : bleu sombre -> bleu -> cyan/vert -> jaune -> rouge -> blanc.
    {
        const float ts[6][3] = {
            {0.05f, 0.12f, 0.60f},  // froid : bleu
            {0.00f, 0.50f, 1.00f},  // bleu clair
            {0.00f, 0.85f, 0.80f},  // cyan
            {0.45f, 0.90f, 0.20f},  // vert
            {1.00f, 0.65f, 0.00f},  // orange/jaune
            {0.92f, 0.10f, 0.05f}   // chaud : ROUGE (pas de blanc)
        };
        const float st[6] = { 0.0f, 0.20f, 0.40f, 0.60f, 0.80f, 1.0f };
        for (int i = 0; i < 32; i++) {
            float t = i / 31.0f;
            int seg = 0; while (seg < 4 && t > st[seg + 1]) seg++;
            float s = (t - st[seg]) / (st[seg + 1] - st[seg] + 1e-6f);
            for (int k = 0; k < 3; k++)
                m_tempPalette[i][k] = ts[seg][k] + (ts[seg + 1][k] - ts[seg][k]) * s;
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

    loadSettings("jgravity.cfg");

    startSimulation();        // initialise CL/CUDA GPU buffers
    m_positions.clear();      // clear CPU + GL buffer → nothing to draw
    m_velocities.clear();
    uploadParticlesToGPU(false);  // m_particleCount = 0, VBO empty
    m_simRunning = false;
    m_simPaused  = false;
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
            if (event.key.keysym.sym == SDLK_SPACE && !ImGui::GetIO().WantCaptureKeyboard) {
                if (m_simRunning) togglePauseWithClock();
                else              m_simPaused = !m_simPaused;
            }
        }

        if (event.type == SDL_MOUSEWHEEL && !ImGui::GetIO().WantCaptureMouse)
        {
            if (m_autoFit) {
                // Auto-cadrage actif : la molette ajuste le CADRAGE (taille apparente
                // cible) au lieu de la distance — sinon elle lutterait contre l'auto-fit.
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
                        m_annotX0 = (float)mx; m_annotY0 = (float)my;
                        m_annotStage = 1;
                    } else {
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
                else if (event.button.clicks == 2)
                {
                    // Double-clic : FOCUS la caméra sur la particule/amas cliqué -> pivot
                    // d'orbite. Annule le marqueur en attente du 1er clic.
                    m_pendingMarker = false;
                    int picked = pickParticleCPU(mx - vpX, my);
                    if (picked >= 0 && picked < (int)m_positions.size()) {
                        const cl_float4& p = m_positions[picked];
                        m_camera->focusTarget(p.x, p.y, p.z);
                    }
                }
                else
                {
                    // Simple clic : marqueur DIFFÉRÉ (posé seulement si aucun 2e clic
                    // n'arrive dans le délai double-clic -> pas de marqueur parasite).
                    m_pendingMarker = true;
                    m_pendingMarkerX = mx - vpX;
                    m_pendingMarkerY = my;
                    m_pendingMarkerTime = SDL_GetTicks();
                }
            }
        }

        // ── Clic droit : orbiter la caméra ──────────────────────────────────
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
            // Orbit type CAO/Blender : tourner autour de la CIBLE (pivot)
            float dx = (float)(event.motion.x - m_lastMouseX);
            float dy = (float)(event.motion.y - m_lastMouseY);
            // dx = yaw horizontal, dy = pitch vertical
            // ~0.3° par pixel
            m_camera->orbit(dx * 0.005f, dy * 0.005f);
            m_lastMouseX = event.motion.x;
            m_lastMouseY = event.motion.y;
        }

        // ── Bouton du milieu (molette) : PAN / translation (type CAO) ───────────
        if (event.type == SDL_MOUSEBUTTONDOWN
            && event.button.button == SDL_BUTTON_MIDDLE
            && !ImGui::GetIO().WantCaptureMouse)
        {
            m_midMouseDown = true;
            m_lastMouseX = event.button.x;
            m_lastMouseY = event.button.y;
        }
        if (event.type == SDL_MOUSEBUTTONUP
            && event.button.button == SDL_BUTTON_MIDDLE)
        {
            m_midMouseDown = false;
        }
        if (event.type == SDL_MOUSEMOTION && m_midMouseDown)
        {
            float dx = (float)(event.motion.x - m_lastMouseX);
            float dy = (float)(event.motion.y - m_lastMouseY);
            m_camera->panPixels(dx, dy);
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
                        60.0, 10.0, 1.0e10f);
                    m_camera->setEdgeOn(true);
                } else {
                    m_camera = new Camera(m_width, m_height,
                        new Vector(0, 0, 2000.0), new Matrix(ROTATION,0,0,0),
                        60.0, 10.0, 1.0e10f);
                    m_camera->setOrbitMode(false);
                }
            }
        }
    }

    // Marqueur différé : si le délai double-clic est écoulé sans 2e clic, on pose
    // enfin le marqueur (sinon le double-clic focus l'aurait déjà annulé).
    if (m_pendingMarker && (SDL_GetTicks() - m_pendingMarkerTime) > 280u)
    {
        m_markerPickPending = true;
        m_markerPickX = m_pendingMarkerX;
        m_markerPickY = m_pendingMarkerY;
        m_pendingMarker = false;
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

// ---------------------------------------------------------------------------
// Navigation 3Dconnexion (SpaceMouse) : pousse la pose caméra vers le pilote,
// récupère la pose mise à jour (6 DoF) et l'applique. Appelé chaque frame.
// ---------------------------------------------------------------------------
void App::update3DMouse()
{
    if (!m_spaceMouse || !m_use3DMouse || !m_spaceMouse->isEnabled() || !m_camera) return;

    // bbox du modèle (throttlé) -> sert au pilote pour calibrer la vitesse de navigation.
    if ((m_smBBtick++ % 30) == 0 && !m_positions.empty()) {
        float mn[3] = {1e30f,1e30f,1e30f}, mx[3] = {-1e30f,-1e30f,-1e30f};
        for (const auto& p : m_positions) {
            mn[0]=std::min(mn[0],p.x); mx[0]=std::max(mx[0],p.x);
            mn[1]=std::min(mn[1],p.y); mx[1]=std::max(mx[1],p.y);
            mn[2]=std::min(mn[2],p.z); mx[2]=std::max(mx[2],p.z);
        }
        for (int k=0;k<3;k++){ m_smBBmin[k]=mn[k]; m_smBBmax[k]=mx[k]; }
    }

    // Pose caméra -> matrice camera-to-world (ROW vectors : row0=right, row1=up,
    // row2=back, row3=eye), même base que getViewProjMatrix (orbit).
    Vector* o = m_camera->getOrigin();
    double ex=o->x, ey=o->y, ez=o->z;
    double tx,ty,tz; m_camera->getTarget(tx,ty,tz);
    double fx=tx-ex, fy=ty-ey, fz=tz-ez;
    double fl=std::sqrt(fx*fx+fy*fy+fz*fz); if (fl<1e-9) { fx=0;fy=0;fz=-1;fl=1; } fx/=fl;fy/=fl;fz/=fl;
    double ux=0,uy=0,uz=1; if (std::fabs(fz)>0.99){ ux=0;uy=1;uz=0; }
    double rx=uy*fz-uz*fy, ry=uz*fx-ux*fz, rz=ux*fy-uy*fx;
    double rl=std::sqrt(rx*rx+ry*ry+rz*rz); if (rl<1e-9) rl=1; rx/=rl;ry/=rl;rz/=rl;
    double upx=fy*rz-fz*ry, upy=fz*rx-fx*rz, upz=fx*ry-fy*rx;
    // right négatif : corrige l'inversion droite/gauche (base camera-to-world propre RH).
    double cam[16] = { -rx,-ry,-rz,0,  upx,upy,upz,0,  -fx,-fy,-fz,0,  ex,ey,ez,1 };

    double bbMin[3]={m_smBBmin[0],m_smBBmin[1],m_smBBmin[2]};
    double bbMax[3]={m_smBBmax[0],m_smBBmax[1],m_smBBmax[2]};
    double pivot[3]={tx,ty,tz};
    double fovRad = m_camera->getFov() * 3.14159265358979 / 180.0;
    double aspect = (m_renderer && m_renderer->getHeight()>0)
                  ? (double)m_renderer->getWidth()/(double)m_renderer->getHeight() : 1.6;
    m_spaceMouse->pushCamera(cam, bbMin, bbMax, fovRad, pivot, aspect, 10.0, 1.0e9);

    // Récupère la pose mise à jour par la SpaceMouse et l'applique.
    double ncam[16], npiv[3]; bool pivChanged=false;
    if (m_spaceMouse->pullCamera(ncam, npiv, pivChanged)) {
        double nex=ncam[12], ney=ncam[13], nez=ncam[14];     // row3 = œil
        double nbx=ncam[8],  nby=ncam[9],  nbz=ncam[10];     // row2 = back = -forward
        double nfx=-nbx, nfy=-nby, nfz=-nbz;
        // cible = œil + forward * distance (garde le pivot ≈ centre de rotation).
        double cx,cy,cz; m_camera->getTarget(cx,cy,cz);
        double d=std::sqrt((cx-nex)*(cx-nex)+(cy-ney)*(cy-ney)+(cz-nez)*(cz-nez));
        if (d < 1.0) { d = m_camera->getDistance(); if (d < 1.0) d = 1000.0; }
        double ntx=nex+nfx*d, nty=ney+nfy*d, ntz=nez+nfz*d;
        if (pivChanged) { ntx=npiv[0]; nty=npiv[1]; ntz=npiv[2]; }
        m_camera->setEyeTarget(nex,ney,nez, ntx,nty,ntz);
    }
}

// SmoothDamp (ressort critique, façon Unity/Game Programming Gems 4) : ramène
// `current` vers `target` avec une VITESSE continue (C¹) -> aucun à-coup même quand
// la cible change par paliers. `vel` est l'état interne (conservé entre frames).
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
    // Anti-dépassement : ne franchit pas la cible.
    if (((target - current) > 0.0f) == (out > target)) {
        out = target;
        vel = (out - target) / dt;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Auto-cadrage : garde la structure à taille apparente constante pendant qu'elle
// s'étend (expansion). On mesure un rayon ROBUSTE (P90, pour ignorer les quelques
// particules éjectées au loin), on en déduit la distance caméra cible, puis :
//   (a) on LISSE la cible elle-même (passe-bas) -> gomme l'escalier des recalculs ;
//   (b) on applique un SmoothDamp à vitesse continue, avec le VRAI dt -> mouvement
//       parfaitement fluide, indépendant du framerate, sans à-coups.
// ---------------------------------------------------------------------------
void App::updateAutoFit()
{
    if (!m_autoFit || !m_simulation || !m_simRunning) return;

    // dt réel (s), clampé pour absorber un hitch ou la reprise après pause.
    unsigned int now = SDL_GetTicks();
    float dt = (m_autoFitLastTicks == 0) ? (1.0f/60.0f)
                                         : (now - m_autoFitLastTicks) * 0.001f;
    m_autoFitLastTicks = now;
    if (dt < 0.0005f) dt = 0.0005f;
    if (dt > 0.05f)   dt = 0.05f;

    // 1) Recalcul throttlé du rayon de la structure (lecture GPU) -> cible BRUTE.
    if (m_currentFrame - m_autoFitLastFrame >= m_autoFitInterval) {
        m_autoFitLastFrame = m_currentFrame;
        m_simulation->readPositions(m_positions);

        const size_t n = m_positions.size();
        if (n > 0) {
            std::vector<float> rr;
            rr.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                const cl_float4& p = m_positions[i];
                if (p.w == 0.0f) continue;   // particule inactive
                rr.push_back(sqrtf(p.x*p.x + p.y*p.y + p.z*p.z));
            }
            if (!rr.empty()) {
                // P90 : cadre le cœur de la structure sans se faire tirer par les éjectas.
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

    // 2a) Passe-bas continu de la cible : la cible lissée rattrape la brute en douceur,
    //     ce qui transforme l'escalier (recalcul tous les N frames) en rampe continue.
    if (m_autoFitTargetDist <= 0.0f) {
        m_autoFitTargetDist = m_autoFitTargetRaw;   // init : pas de swoop depuis 0
    } else {
        float aT = 1.0f - expf(-dt / (0.5f * m_autoFitSmoothTime));
        m_autoFitTargetDist += (m_autoFitTargetRaw - m_autoFitTargetDist) * aT;
    }

    // 2b) SmoothDamp de la distance réelle vers la cible lissée (vitesse continue).
    float cur  = (float)m_camera->getDistance();
    float next = smoothDamp(cur, m_autoFitTargetDist, m_autoFitDistVel,
                            m_autoFitSmoothTime, dt);
    m_camera->setDistance((double)next);
}

// SCHEDULER (cheap, every frame): adopt a finished density result, launch the worker
// when due. The heavy 64^3 density + smoothing runs off-thread so it never hitches the
// frame. uploadParticlesToGPU already guards stale m_density/m_smoothLength (size check).
void App::computeLocalDensity()
{
    if (m_densReady.load()) {
        if (m_densThread.joinable()) m_densThread.join();
        m_density         = std::move(m_densityBack);
        m_smoothLength    = std::move(m_smoothLengthBack);
        m_densMaxForColor = m_densMaxBack;
        ++m_densityVersion;
        m_smoothDirty = true;
        m_densReady = false; m_densComputing = false;
        m_densLastLaunchMs = SDL_GetTicks64();   // cooldown from completion
    }
    if (!m_densComputing.load() && !m_positions.empty() &&
        (SDL_GetTicks64() - m_densLastLaunchMs) >= (uint64_t)m_densIntervalMs) {
        m_densLastLaunchMs = SDL_GetTicks64();
        m_densPosCopy = m_positions;             // snapshot (worker reads its own copy)
        m_densComputing = true; m_densReady = false;
        m_densThread = std::thread(&App::computeLocalDensityWorker, this);
    }
}

// WORKER (background thread): 64³ density grid (per species) + Gaussian smooth +
// trilinear sample -> per-particle smoothing length. Writes the BACK buffers; the
// scheduler swaps them in. Operates on the snapshot m_densPosCopy only.
void App::computeLocalDensityWorker()
{
    const std::vector<cl_float4>& POS = m_densPosCopy;
    const size_t n = POS.size();
    m_densityBack.assign(n, 0.0f);
    m_smoothLengthBack.assign(n, 1.0f);
    m_densMaxBack = 1.0f;
    if (n == 0) { m_densReady = true; return; }

    // Boîte ROBUSTE = moyenne ± 3σ par axe (ignore les particules éjectées au loin
    // qui gonfleraient la boîte -> grille trop grossière -> tout bleu/blanc sans milieu).
    double m0=0, m1=0, m2=0;
    for (size_t i = 0; i < n; i++) { m0 += POS[i].x; m1 += POS[i].y; m2 += POS[i].z; }
    m0 /= (double)n; m1 /= (double)n; m2 /= (double)n;
    double v0=0, v1=0, v2=0;
    for (size_t i = 0; i < n; i++) {
        double dx=POS[i].x-m0, dy=POS[i].y-m1, dz=POS[i].z-m2;
        v0 += dx*dx; v1 += dy*dy; v2 += dz*dz;
    }
    float s0=(float)sqrt(v0/(double)n), s1=(float)sqrt(v1/(double)n), s2=(float)sqrt(v2/(double)n);
    const float K = 3.0f;
    float mn[3] = { (float)m0-K*s0, (float)m1-K*s1, (float)m2-K*s2 };
    float mx[3] = { (float)m0+K*s0, (float)m1+K*s1, (float)m2+K*s2 };
    float ext[3] = { mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2] };
    for (int k = 0; k < 3; k++) if (ext[k] < 1e-6f) ext[k] = 1e-6f;

    const int NG = 64, NG2 = NG*NG, NG3 = NG*NG*NG;
    auto clampi = [&](int v){ return v < 0 ? 0 : (v >= NG ? NG-1 : v); };
    std::vector<float> rhoP(NG3, 0.0f), rhoN(NG3, 0.0f);
    for (size_t i = 0; i < n; i++) {
        const cl_float4& p = POS[i];
        int cx = clampi((int)((p.x-mn[0])/ext[0]*NG));
        int cy = clampi((int)((p.y-mn[1])/ext[1]*NG));
        int cz = clampi((int)((p.z-mn[2])/ext[2]*NG));
        ((p.w >= 0.0f) ? rhoP : rhoN)[(cz*NG + cy)*NG + cx] += 1.0f;
    }
    // Lissage gaussien séparable ([1 2 1]/4, 3 passes) -> champ CONTINU : la densité
    // varie en dégradé (pas 1/centaines binaire) -> toute la rampe de couleur est utilisée.
    std::vector<float> tmp(NG3);
    auto smooth = [&](std::vector<float>& g) {
        for (int pass = 0; pass < 3; pass++) {
            for (int z=0;z<NG;z++) for (int y=0;y<NG;y++){ int b=(z*NG+y)*NG;
                for (int x=0;x<NG;x++){ int xm=x>0?x-1:x, xp=x<NG-1?x+1:x;
                    tmp[b+x]=0.25f*(g[b+xm]+2.0f*g[b+x]+g[b+xp]); } }
            for (int z=0;z<NG;z++) for (int y=0;y<NG;y++) for (int x=0;x<NG;x++){
                int ym=y>0?y-1:y, yp=y<NG-1?y+1:y;
                g[(z*NG+y)*NG+x]=0.25f*(tmp[(z*NG+ym)*NG+x]+2.0f*tmp[(z*NG+y)*NG+x]+tmp[(z*NG+yp)*NG+x]); }
            for (int z=0;z<NG;z++) for (int y=0;y<NG;y++) for (int x=0;x<NG;x++){
                int zm=z>0?z-1:z, zp=z<NG-1?z+1:z;
                tmp[(z*NG+y)*NG+x]=0.25f*(g[(zm*NG+y)*NG+x]+2.0f*g[(z*NG+y)*NG+x]+g[(zp*NG+y)*NG+x]); }
            g.swap(tmp);
        }
    };
    smooth(rhoP); smooth(rhoN);
    // Échantillonnage trilinéaire du champ lissé (densité douce, sans marches de cellule).
    auto sample = [&](const std::vector<float>& g, float px, float py, float pz) -> float {
        float gx=(px-mn[0])/ext[0]*NG-0.5f, gy=(py-mn[1])/ext[1]*NG-0.5f, gz=(pz-mn[2])/ext[2]*NG-0.5f;
        int x0=(int)floorf(gx), y0=(int)floorf(gy), z0=(int)floorf(gz);
        float fx=gx-x0, fy=gy-y0, fz=gz-z0;
        int X0=clampi(x0),Y0=clampi(y0),Z0=clampi(z0),X1=clampi(x0+1),Y1=clampi(y0+1),Z1=clampi(z0+1);
        auto V=[&](int xi,int yi,int zi)->float{ return g[(zi*NG+yi)*NG+xi]; };
        float c00=V(X0,Y0,Z0)*(1-fx)+V(X1,Y0,Z0)*fx, c10=V(X0,Y1,Z0)*(1-fx)+V(X1,Y1,Z0)*fx;
        float c01=V(X0,Y0,Z1)*(1-fx)+V(X1,Y0,Z1)*fx, c11=V(X0,Y1,Z1)*(1-fx)+V(X1,Y1,Z1)*fx;
        float c0=c00*(1-fy)+c10*fy, c1=c01*(1-fy)+c11*fy;
        return c0*(1-fz)+c1*fz;
    };
    std::vector<float> rawD(n, 0.0f);
    std::vector<float> vals; vals.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const cl_float4& p = POS[i];
        float d = sample((p.w >= 0.0f) ? rhoP : rhoN, p.x, p.y, p.z);
        rawD[i] = d;
        m_densityBack[i] = logf(1.0f + d);    // log -> peuple le milieu de la rampe
        vals.push_back(m_densityBack[i]);
    }
    // Longueur de lissage SPH : h ∝ (dRef / d)^(1/3). Particule DIFFUSE (d faible) -> GROS
    // halo (remplit les trous -> champ lisse) ; particule DENSE -> petit halo (net).
    {
        std::vector<float> dd; dd.reserve(n);
        for (size_t i = 0; i < n; i++) if (POS[i].w >= 0.0f) dd.push_back(rawD[i]);
        float dRef = 1.0f;
        if (!dd.empty()) { size_t mid = dd.size()/2; std::nth_element(dd.begin(), dd.begin()+mid, dd.end()); dRef = dd[mid] > 0.2f ? dd[mid] : 1.0f; }
        for (size_t i = 0; i < n; i++) {
            float di = rawD[i] > 0.15f ? rawD[i] : 0.15f;
            // Lissage adaptatif à LARGE dynamique (clé du look "fumée" py-sphviewer) :
            //   ×2.6 + plafond 70 -> les particules DIFFUSES ont de TRÈS gros noyaux qui se
            //   chevauchent et se fondent en fumée continue (tue le moiré de grille des vides) ;
            //   plancher 0.35 -> les particules DENSES ont de TOUT petits noyaux -> cœurs piqués.
            float h = 2.6f * cbrtf(dRef / di);
            m_smoothLengthBack[i] = h < 0.35f ? 0.35f : (h > 70.0f ? 70.0f : h);
        }
    }
    // Normalisation au 99e percentile -> cœurs en rouge, blanc reservé au tout dernier pic.
    if (!vals.empty()) {
        size_t p99 = (size_t)(vals.size() * 0.99f);
        std::nth_element(vals.begin(), vals.begin() + p99, vals.end());
        m_densMaxBack = vals[p99] > 1e-4f ? vals[p99] : 1.0f;
    }
    // Free the heavy locals on the worker BEFORE signaling ready, so the scheduler's
    // join() is instant (same lesson as the galaxy worker).
    std::vector<float>().swap(rawD);
    std::vector<float>().swap(vals);
    std::vector<float>().swap(rhoP);
    std::vector<float>().swap(rhoN);
    std::vector<float>().swap(tmp);
    m_densReady = true;
}

// FoF par-particule pour le CORIAGE (distinct du computeFoF existant qui produit
// le catalogue de clusters async). Friends-of-Friends b=0.2, halos >=200 parts
// (recette cosmo standard, cf Yu & Chen 2025). Remplit m_haloSize par particule.
void App::computeHaloColoring()
{
    ++m_haloVersion;
    const size_t n = m_positions.size();
    m_haloSize.assign(n, 0);
    m_haloMaxForColor = 1.0f;
    if (n == 0) return;

    std::vector<int> parent(n);
    for (size_t i = 0; i < n; i++) parent[i] = (int)i;
    auto find = [&](int x) -> int {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto uni = [&](int a, int b) { int ra = find(a), rb = find(b); if (ra != rb) parent[ra] = rb; };
    auto hkey = [](int cx, int cy, int cz) -> long long {
        return ((long long)cx * 73856093LL) ^ ((long long)cy * 19349663LL) ^ ((long long)cz * 83492791LL);
    };

    // FoF within ONE species only (positives never link to negatives -> components are
    // pure). Run it for both species so NEGATIVE-mass halos are detected too (shown pink).
    auto linkSpecies = [&](bool neg, int speciesCount) {
        float sep = 2.0f * m_radius / cbrtf((float)(speciesCount > 0 ? speciesCount : 1));
        if (sep < 1e-6f) sep = 1.0f;
        float ll = 0.15f * sep, ll2 = ll * ll, inv_ll = 1.0f / ll;
        std::unordered_map<long long, std::vector<int>> cells; cells.reserve(n);
        for (size_t i = 0; i < n; i++) {
            if ((m_positions[i].w < 0.0f) != neg) continue;
            const cl_float4& p = m_positions[i];
            cells[hkey((int)floorf(p.x*inv_ll),(int)floorf(p.y*inv_ll),(int)floorf(p.z*inv_ll))].push_back((int)i);
        }
        for (size_t i = 0; i < n; i++) {
            if ((m_positions[i].w < 0.0f) != neg) continue;
            const cl_float4& pi = m_positions[i];
            int cx=(int)floorf(pi.x*inv_ll), cy=(int)floorf(pi.y*inv_ll), cz=(int)floorf(pi.z*inv_ll);
            for (int dz=-1;dz<=1;dz++) for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++) {
                auto it = cells.find(hkey(cx+dx,cy+dy,cz+dz));
                if (it == cells.end()) continue;
                for (int j : it->second) {
                    if (j <= (int)i) continue;
                    const cl_float4& pj = m_positions[j];
                    float ax=pi.x-pj.x, ay=pi.y-pj.y, az=pi.z-pj.z;
                    if (ax*ax+ay*ay+az*az <= ll2) uni((int)i, j);
                }
            }
        }
    };
    linkSpecies(false, m_numPosiStars);   // positive-mass halos
    linkSpecies(true,  m_numNegaStars);    // negative-mass halos (pink)

    std::vector<int> cnt(n, 0);
    for (size_t i = 0; i < n; i++) cnt[find((int)i)]++;   // each component is single-species
    int maxSize = 1;
    for (size_t i = 0; i < n; i++) {
        int s = cnt[find((int)i)];
        m_haloSize[i] = (s >= 200) ? s : 0;   // halo minimum 200 particules
        if (m_haloSize[i] > maxSize) maxSize = m_haloSize[i];
    }
    m_haloMaxForColor = logf(1.0f + (float)maxSize);
}

// Valeurs propres d'une matrice symétrique 3x3 (méthode analytique de Smith).
static void eig3sym(float a, float b, float c, float d, float e, float f, float ev[3])
{
    float p1 = b*b + c*c + e*e;
    if (p1 < 1e-20f) { ev[0]=a; ev[1]=d; ev[2]=f; return; }
    float q  = (a + d + f) / 3.0f;
    float p2 = (a-q)*(a-q) + (d-q)*(d-q) + (f-q)*(f-q) + 2.0f*p1;
    float p  = sqrtf(p2 / 6.0f);
    if (p < 1e-20f) { ev[0]=ev[1]=ev[2]=q; return; }
    float ip = 1.0f / p;
    float b00=(a-q)*ip, b11=(d-q)*ip, b22=(f-q)*ip, b01=b*ip, b02=c*ip, b12=e*ip;
    float detB = b00*(b11*b22 - b12*b12) - b01*(b01*b22 - b12*b02) + b02*(b01*b12 - b11*b02);
    float r = detB * 0.5f;
    r = r < -1.0f ? -1.0f : (r > 1.0f ? 1.0f : r);
    float phi = acosf(r) / 3.0f;
    ev[0] = q + 2.0f*p*cosf(phi);
    ev[2] = q + 2.0f*p*cosf(phi + 2.0943951f);   // +2π/3
    ev[1] = 3.0f*q - ev[0] - ev[2];
}

// T-web : classification de la toile cosmique. Champ de densité lissé -> Hessian
// par différences finies -> valeurs propres. Le NOMBRE de valeurs propres NÉGATIVES
// (directions d'effondrement) donne la classe : 0=vide, 1=paroi, 2=filament, 3=nœud.
void App::computeTweb()
{
    ++m_webVersion;
    const size_t n = m_positions.size();
    m_webClass.assign(n, 0);
    if (n == 0) return;

    // Boîte robuste ±3σ (comme la densité)
    double m0=0,m1=0,m2=0;
    for (size_t i=0;i<n;i++){ m0+=m_positions[i].x; m1+=m_positions[i].y; m2+=m_positions[i].z; }
    m0/=(double)n; m1/=(double)n; m2/=(double)n;
    double v0=0,v1=0,v2=0;
    for (size_t i=0;i<n;i++){ double dx=m_positions[i].x-m0,dy=m_positions[i].y-m1,dz=m_positions[i].z-m2; v0+=dx*dx;v1+=dy*dy;v2+=dz*dz; }
    float s0=(float)sqrt(v0/(double)n), s1=(float)sqrt(v1/(double)n), s2=(float)sqrt(v2/(double)n);
    const float K=3.0f;
    float mn[3]={(float)m0-K*s0,(float)m1-K*s1,(float)m2-K*s2};
    float mx[3]={(float)m0+K*s0,(float)m1+K*s1,(float)m2+K*s2};
    float ext[3]={mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2]};
    for (int k=0;k<3;k++) if (ext[k]<1e-6f) ext[k]=1e-6f;

    const int NG=64, NG2=NG*NG, NG3=NG*NG*NG;
    auto cell3=[&](const cl_float4&p,int&cx,int&cy,int&cz){
        cx=(int)((p.x-mn[0])/ext[0]*NG); cx=cx<0?0:(cx>=NG?NG-1:cx);
        cy=(int)((p.y-mn[1])/ext[1]*NG); cy=cy<0?0:(cy>=NG?NG-1:cy);
        cz=(int)((p.z-mn[2])/ext[2]*NG); cz=cz<0?0:(cz>=NG?NG-1:cz);
    };
    std::vector<float> rho(NG3, 0.0f);
    for (size_t i=0;i<n;i++){ if (m_positions[i].w<0.0f) continue; int cx,cy,cz; cell3(m_positions[i],cx,cy,cz); rho[(cz*NG+cy)*NG+cx]+=1.0f; }

    // Lissage gaussien séparable ([1 2 1]/4), 2 passes -> réduit le bruit du Hessian.
    std::vector<float> tmp(NG3);
    for (int pass=0; pass<2; pass++) {
        for (int z=0;z<NG;z++) for (int y=0;y<NG;y++) for (int x=0;x<NG;x++){
            int xm=x>0?x-1:x, xp=x<NG-1?x+1:x; int id=(z*NG+y)*NG;
            tmp[id+x]=0.25f*(rho[id+xm]+2.0f*rho[id+x]+rho[id+xp]);
        }
        for (int z=0;z<NG;z++) for (int y=0;y<NG;y++) for (int x=0;x<NG;x++){
            int ym=y>0?y-1:y, yp=y<NG-1?y+1:y;
            rho[(z*NG+y)*NG+x]=0.25f*(tmp[(z*NG+ym)*NG+x]+2.0f*tmp[(z*NG+y)*NG+x]+tmp[(z*NG+yp)*NG+x]);
        }
        for (int z=0;z<NG;z++) for (int y=0;y<NG;y++) for (int x=0;x<NG;x++){
            int zm=z>0?z-1:z, zp=z<NG-1?z+1:z;
            tmp[(z*NG+y)*NG+x]=0.25f*(rho[(zm*NG+y)*NG+x]+2.0f*rho[(z*NG+y)*NG+x]+rho[(zp*NG+y)*NG+x]);
        }
        rho.swap(tmp);
    }

    // Hessian + valeurs propres + classe (cellules intérieures ; bords -> vide).
    std::vector<unsigned char> cls(NG3, 0);
    auto R=[&](int x,int y,int z)->float{ return rho[(z*NG+y)*NG+x]; };
    for (int z=1;z<NG-1;z++) for (int y=1;y<NG-1;y++) for (int x=1;x<NG-1;x++){
        float c0=R(x,y,z);
        float Hxx=R(x+1,y,z)-2.0f*c0+R(x-1,y,z);
        float Hyy=R(x,y+1,z)-2.0f*c0+R(x,y-1,z);
        float Hzz=R(x,y,z+1)-2.0f*c0+R(x,y,z-1);
        float Hxy=0.25f*(R(x+1,y+1,z)-R(x+1,y-1,z)-R(x-1,y+1,z)+R(x-1,y-1,z));
        float Hxz=0.25f*(R(x+1,y,z+1)-R(x+1,y,z-1)-R(x-1,y,z+1)+R(x-1,y,z-1));
        float Hyz=0.25f*(R(x,y+1,z+1)-R(x,y+1,z-1)-R(x,y-1,z+1)+R(x,y-1,z-1));
        float ev[3]; eig3sym(Hxx,Hxy,Hxz,Hyy,Hyz,Hzz, ev);
        int cnt = (ev[0]<0.0f?1:0)+(ev[1]<0.0f?1:0)+(ev[2]<0.0f?1:0); // neg = effondrement
        cls[(z*NG+y)*NG+x]=(unsigned char)cnt;
    }
    for (size_t i=0;i<n;i++){
        if (m_positions[i].w<0.0f){ m_webClass[i]=0; continue; }
        int cx,cy,cz; cell3(m_positions[i],cx,cy,cz);
        m_webClass[i]=cls[(cz*NG+cy)*NG+cx];
    }
}

// "Température" cinétique = dispersion locale des vitesses σ² (agitation aléatoire).
// Par cellule : σ² = <|v|²> - |<v>|². Chaud = amas virialisés / chocs ; froid = flot
// cohérent / vides. Champ lissé + échantillonné, normalisé au 98e percentile.
void App::computeTemperature()
{
    ++m_tempVersion;
    // Lire POSITIONS et VITESSES du MÊME frame. Sinon, au changement de mode en cours
    // de route, les positions sont à jour mais les vitesses périmées (ou en déphasage)
    // -> la σ² par cellule associe la vitesse d'une particule à la position d'une autre
    // -> gradient cohérent (split rouge/bleu gauche-droite). Les lire ensemble garantit
    // la cohérence et supprime ce bug (plus besoin de redémarrer la simu).
    if (m_simulation) {
        m_simulation->readPositions(m_positions);
        m_simulation->readVelocities(m_rotVelocities);
    }
    const size_t n = m_positions.size();
    m_temp.assign(n, 0.0f);
    m_tempMaxForColor = 1.0f;
    if (n == 0) return;
    if (m_rotVelocities.size() < n) return;

    double m0=0,m1=0,m2=0;
    for (size_t i=0;i<n;i++){ m0+=m_positions[i].x; m1+=m_positions[i].y; m2+=m_positions[i].z; }
    m0/=(double)n; m1/=(double)n; m2/=(double)n;
    // Boîte ROBUSTE : rayon P90 des POSITIFS depuis le centroïde. L'écart-type était
    // gonflé par les éjectas -> boîte géante -> cellules énormes -> le flot cohérent
    // n'était PAS retiré localement -> gradient. Avec une boîte serrée, chaque cellule
    // est un vrai VOISINAGE LOCAL (le flot cohérent y est ~constant -> retiré par la
    // moyenne de cellule -> σ² = dispersion thermique locale, à toutes les échelles).
    std::vector<float> rad; rad.reserve(n);
    for (size_t i=0;i<n;i++){ const cl_float4& p=m_positions[i]; if(p.w<0.0f) continue;
        double dx=p.x-m0,dy=p.y-m1,dz=p.z-m2; rad.push_back((float)sqrt(dx*dx+dy*dy+dz*dz)); }
    float R=1.0f;
    if (!rad.empty()){ size_t kk=(size_t)(0.90*(double)(rad.size()-1));
        std::nth_element(rad.begin(),rad.begin()+kk,rad.end()); R=rad[kk]*1.1f; if(R<1e-3f)R=1e-3f; }
    float mn[3]={(float)m0-R,(float)m1-R,(float)m2-R};
    float ext[3]={2.0f*R,2.0f*R,2.0f*R};

    const int NG=64, NG3=NG*NG*NG;   // grille fine -> cellules = voisinage local
    auto clampi=[&](int v){ return v<0?0:(v>=NG?NG-1:v); };
    auto cellOf=[&](const cl_float4&p)->int{
        int cx=clampi((int)((p.x-mn[0])/ext[0]*NG));
        int cy=clampi((int)((p.y-mn[1])/ext[1]*NG));
        int cz=clampi((int)((p.z-mn[2])/ext[2]*NG));
        return (cz*NG+cy)*NG+cx;
    };
    std::vector<float> sx(NG3,0),sy(NG3,0),sz(NG3,0),s2g(NG3,0),cnt(NG3,0);
    for (size_t i=0;i<n;i++){
        if (m_positions[i].w<0.0f) continue;   // matière positive
        int c=cellOf(m_positions[i]);
        const cl_float4& v=m_rotVelocities[i];
        sx[c]+=v.x; sy[c]+=v.y; sz[c]+=v.z;
        s2g[c]+=v.x*v.x+v.y*v.y+v.z*v.z; cnt[c]+=1.0f;
    }
    std::vector<float> sig(NG3,0.0f);
    for (int c=0;c<NG3;c++){
        if (cnt[c]<4.0f) continue;   // min de particules -> moyenne fiable (sinon bruit)
        float inv=1.0f/cnt[c];
        float mvx=sx[c]*inv, mvy=sy[c]*inv, mvz=sz[c]*inv;
        float d=s2g[c]*inv - (mvx*mvx+mvy*mvy+mvz*mvz);   // σ² (dispersion locale)
        sig[c]= d>0.0f ? d : 0.0f;
    }
    // lissage léger (1 passe séparable) -> champ continu
    std::vector<float> tmp(NG3);
    for (int z=0;z<NG;z++) for (int y=0;y<NG;y++){ int b=(z*NG+y)*NG;
        for (int x=0;x<NG;x++){ int xm=x>0?x-1:x,xp=x<NG-1?x+1:x; tmp[b+x]=0.25f*(sig[b+xm]+2*sig[b+x]+sig[b+xp]); } }
    for (int z=0;z<NG;z++) for (int y=0;y<NG;y++) for (int x=0;x<NG;x++){ int ym=y>0?y-1:y,yp=y<NG-1?y+1:y;
        sig[(z*NG+y)*NG+x]=0.25f*(tmp[(z*NG+ym)*NG+x]+2*tmp[(z*NG+y)*NG+x]+tmp[(z*NG+yp)*NG+x]); }
    for (int z=0;z<NG;z++) for (int y=0;y<NG;y++) for (int x=0;x<NG;x++){ int zm=z>0?z-1:z,zp=z<NG-1?z+1:z;
        tmp[(z*NG+y)*NG+x]=0.25f*(sig[(zm*NG+y)*NG+x]+2*sig[(z*NG+y)*NG+x]+sig[(zp*NG+y)*NG+x]); }
    sig.swap(tmp);

    std::vector<float> vals; vals.reserve(n);
    for (size_t i=0;i<n;i++){
        if (m_positions[i].w<0.0f){ m_temp[i]=0.0f; continue; }
        float T=sig[cellOf(m_positions[i])];
        m_temp[i]=logf(1.0f+T);   // log -> compresse l'énorme plage (amas très chauds)
        vals.push_back(m_temp[i]);
    }
    // Contraste : ancre BAS au 20e percentile (diffus -> bleu) et HAUT au 99e
    // (cœurs -> blanc). Étale toute la rampe au lieu de tout coincer au milieu (jaune).
    if (!vals.empty()){
        std::vector<float> sv(vals);
        size_t loi=(size_t)(sv.size()*0.20f), hii=(size_t)(sv.size()*0.99f);
        std::nth_element(sv.begin(), sv.begin()+loi, sv.end()); float lo=sv[loi];
        std::nth_element(sv.begin(), sv.begin()+hii, sv.end()); float hi=sv[hii];
        float range = hi - lo; if (range < 1e-4f) range = 1.0f;
        for (size_t i=0;i<n;i++){
            if (m_positions[i].w < 0.0f) continue;
            float t = m_temp[i] - lo; m_temp[i] = t > 0.0f ? t : 0.0f;
        }
        m_tempMaxForColor = range;
    }
}

// ── Champ σ (km/s) pour la "température" des marqueurs ────────────────────────
// Même principe que computeTemperature (σ² = <|v|²>-|<v>|² par cellule, flot
// d'ensemble retiré) mais on garde la valeur PHYSIQUE BRUTE en km/s (pas de log ni
// de normalisation), afin d'en déduire une température cinétique en kelvins.
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
    const float conv=(m_lyPerUnit*9.461e12f)/3.156e13f;  // 1 sim_vel -> km/s
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

// Construit le VBO de rendu sur le GPU (kernel CUDA -> VBO mappé) depuis les buffers
// device de la simulation. Couvre masse, vitesse et valeurs auxiliaires (température,
// T-web, halo, densité). Retourne false -> repli CPU (replay, interop coupé/échec,
// données de couleur absentes).
bool App::uploadParticlesInterop()
{
    if (!m_useInterop || !m_simulation || m_replayMode) return false;
    const int n = m_simulation->getParticleCount();
    if (n <= 0) return false;
    const size_t N = (size_t)n;
    if (m_positions.size() != N) return false;   // hôte/device désynchronisés
    const void* dpos = m_simulation->getDevicePositions();
    if (!dpos) return false;

    // Mode couleur, même priorité que le chemin CPU, mêmes gardes de taille.
    int mode = 0;                 // 0 masse, 1 vitesse, 2 auxiliaire
    int auxKind = 0;              // 1 temp, 2 web, 3 halo, 4 densité
    const void* dvel = nullptr;
    const float* hostAux = nullptr;
    if (m_colorByTemp && m_temp.size() == N)            auxKind = 1;
    else if (m_colorByWeb && m_webClass.size() == N)    auxKind = 2;
    else if (m_colorByHalo && m_haloSize.size() == N)   auxKind = 3;
    else if (m_colorByDensity && m_density.size() == N) auxKind = 4;
    else if (m_colorByVelocity && m_rotVelocities.size() >= N) {
        dvel = m_simulation->getDeviceVelocities();
        if (dvel) mode = 1;
    }
    if (auxKind != 0) mode = 2;

    if (auxKind == 1) hostAux = m_temp.data();
    else if (auxKind == 4) hostAux = m_density.data();
    else if (auxKind == 2 || auxKind == 3) {
        const int ver = (auxKind == 2) ? m_webVersion : m_haloVersion;
        if (m_auxValues.size() != N || m_auxValuesVersion != ver || m_auxValuesMode != auxKind) {
            m_auxValues.resize(N);
            if (auxKind == 2) {
                for (size_t i = 0; i < N; ++i) m_auxValues[i] = (float)m_webClass[i] + 0.5f;
            } else {
                for (size_t i = 0; i < N; ++i)
                    m_auxValues[i] = (m_haloSize[i] > 0) ? logf(1.0f + (float)m_haloSize[i]) : 0.0f;
            }
            m_auxValuesVersion = ver; m_auxValuesMode = auxKind;
        }
        hostAux = m_auxValues.data();
    }

    // (Ré)alloue + réenregistre le VBO quand le nombre de particules change.
    if (!m_interopOk || m_interopVboCount != N) {
        if (m_interopRes) { interopUnregister(m_interopRes); m_interopRes = nullptr; }
        m_renderer->reserveVBO(N);
        m_interopOk = interopRegister(&m_interopRes, m_renderer->getVBO());
        if (!m_interopOk) { m_interopVboCount = 0; return false; }
        m_interopVboCount = N;
        if (m_dSmooth) { interopFree(m_dSmooth); m_dSmooth = nullptr; }
        m_dSmooth = interopAllocFloats(n); m_dSmoothCount = m_dSmooth ? n : 0;
        m_smoothDirty = true;
        if (m_dAux) { interopFree(m_dAux); m_dAux = nullptr; }
        m_dAux = interopAllocFloats(n); m_dAuxCount = m_dAux ? n : 0;
        m_auxValuesVersion = -1; m_auxValuesMode = -1;
        m_auxUploadedVersion = -1; m_auxUploadedMode = -1;
    }

    // Longueurs de lissage SPH -> device (seulement quand le worker les recalcule).
    const bool useSph = m_softHalo && (m_smoothLength.size() == N) && (m_dSmooth != nullptr);
    const void* dsm = nullptr;
    if (useSph) {
        if (m_smoothDirty) { interopUploadFloats(m_dSmooth, m_smoothLength.data(), n); m_smoothDirty = false; }
        dsm = m_dSmooth;
    }

    // Valeurs auxiliaires -> device (seulement quand la source est recalculée).
    const void* daux = nullptr;
    if (mode == 2) {
        if (!m_dAux || !hostAux) return false;
        const int auxVer = (auxKind == 1) ? m_tempVersion
                         : (auxKind == 2) ? m_webVersion
                         : (auxKind == 3) ? m_haloVersion
                                          : m_densityVersion;
        if (m_auxUploadedVersion != auxVer || m_auxUploadedMode != auxKind) {
            interopUploadFloats(m_dAux, hostAux, n);
            m_auxUploadedVersion = auxVer;
            m_auxUploadedMode = auxKind;
        }
        daux = m_dAux;
    }

    bool ok = interopFillVertices(m_interopRes, dpos, dvel, dsm, daux, n, mode, useSph ? 1 : 0);
    if (!ok) { m_interopOk = false; m_interopVboCount = 0; return false; }
    m_renderer->setDrawCount(N);
    m_particleCount = N;
    return true;
}

void App::uploadParticlesToGPU(bool subData)
{
    // computeLocalDensity() rebuilds a 64^3 density field + smoothing on the CPU
    // (~tens of ms). The density evolves slowly, so THROTTLE it (1 frame in 12) for
    // BOTH Color-by-Density and Soft-Halo. Previously Color-by-Density recomputed it
    // EVERY frame -> that alone made Color-by-Density runs (e.g. Newton) far slower
    // than the throttled Soft-Halo runs (e.g. the smooth 3M Janus view).
    // Density scheduler (cheap): adopts the background result / launches the worker.
    // The buffer build below guards stale m_density / m_smoothLength by size.
    if (m_colorByDensity || m_softHalo) computeLocalDensity();
    // Garde-fous : si un mode couleur est actif mais ses données ne sont pas (encore)
    // calculées (ex. au démarrage en pause), on les calcule maintenant -> plus besoin
    // de décocher/recocher pour que ça marche.
    if (m_colorByTemp && m_temp.size()     != m_positions.size()) computeTemperature();
    if (m_colorByHalo && m_haloSize.size() != m_positions.size()) computeHaloColoring();
    if (m_colorByWeb  && m_webClass.size() != m_positions.size()) computeTweb();

    const bool useSph = m_softHalo && (m_smoothLength.size() == m_positions.size());
    m_particleData.clear();
    m_particleData.reserve(m_positions.size() * 6);
    size_t count = 0;

    // Velocity colour scale : Auto-scale ON -> échelle = p98 courant (suit up/down) ;
    // Peak-hold -> ne fait que MONTER (garde le pic) -> le refroidissement (baisse des
    // vitesses) devient visible tout seul ; OFF/OFF -> figée à la dernière valeur.
    // Vitesse TOTALE (Hubble inclus) : le péculier seul est trop uniforme/plat tôt -> bleu.
    if (m_colorByVelocity && (m_velAutoScale || m_velPeakHold) && !m_rotVelocities.empty()) {
        std::vector<float> speeds;
        speeds.reserve(m_positions.size());
        for (size_t i = 0; i < m_rotVelocities.size() && i < m_positions.size(); i++) {
            if (m_positions[i].w <= 0.0f) continue;
            const cl_float4& v = m_rotVelocities[i];
            speeds.push_back(sqrtf(v.x*v.x + v.y*v.y + v.z*v.z));
        }
        if (!speeds.empty()) {
            size_t p98 = (size_t)(speeds.size() * 0.98f);
            std::nth_element(speeds.begin(), speeds.begin() + p98, speeds.end());
            float cur = speeds[p98] > 1e-6f ? speeds[p98] : 1.0f;
            if (m_velPeakHold) { if (cur > m_velMaxForColor) m_velMaxForColor = cur; }
            else               { m_velMaxForColor = cur; }
        }
    }
    if (m_velMaxForColor < 1e-6f) m_velMaxForColor = 1.0f;

    // Interop : le kernel CUDA remplit le VBO directement depuis le device.
    if (uploadParticlesInterop()) return;
    // Repli CPU : le VBO va être réalloué -> désenregistrer l'interop.
    if (m_interopRes) { interopUnregister(m_interopRes); m_interopRes = nullptr; }
    m_interopOk = false; m_interopVboCount = 0;

    for (size_t i = 0; i < m_positions.size(); i++)
    {
        const cl_float4& p = m_positions[i];
        if (!m_showNegaMass && p.w < 0.0f) continue;
        m_particleData.push_back(p.x);
        m_particleData.push_back(p.y);
        m_particleData.push_back(p.z);
        if (m_colorByTemp && i < m_temp.size()) {
            float tv = m_temp[i];
            m_particleData.push_back(p.w > 0.0f ? tv : -tv);
        } else if (m_colorByWeb && i < m_webClass.size()) {
            // valeur = classe T-web + 0.5 (avec maxVal=4 -> bande de palette dédiée)
            float wv = (float)m_webClass[i] + 0.5f;
            m_particleData.push_back(p.w > 0.0f ? wv : -wv);
        } else if (m_colorByHalo && i < m_haloSize.size()) {
            // valeur = log(taille du halo FoF) ; 0 si pas dans un halo -> reste sombre
            float h = (m_haloSize[i] > 0) ? logf(1.0f + (float)m_haloSize[i]) : 0.0f;
            m_particleData.push_back(p.w > 0.0f ? h : -h);
        } else if (m_colorByDensity && i < m_density.size()) {
            float d = m_density[i];
            m_particleData.push_back(p.w > 0.0f ? d : -d);   // signe = espèce (palette pos/neg)
        } else if (m_colorByVelocity && !m_rotVelocities.empty() && i < m_rotVelocities.size()) {
            const cl_float4& v = m_rotVelocities[i];
            float spd = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);   // vitesse totale (Hubble inclus)
            m_particleData.push_back(p.w > 0.0f ? spd : -spd);
        } else {
            m_particleData.push_back(p.w);
        }
        m_particleData.push_back((float)i);
        // 6e float : taille adaptative (longueur de lissage SPH), 1.0 sinon. Le SIGNE encode
        // l'espèce (négatif = masse négative) -> détection robuste pour la teinte rose,
        // indépendante du mode couleur (≠ signe de la valeur qui peut être nul).
        float sz6 = useSph ? m_smoothLength[i] : 1.0f;
        m_particleData.push_back((p.w < 0.0f) ? -sz6 : sz6);
        count++;
    }
    m_particleCount = count;
    m_renderer->uploadParticles(m_particleData, m_particleCount, subData);
}

void App::startSimulation()
{
    if (!m_simulation) return;
    m_isGenerating = true;

     UniverseParams params;
     params.initRadius  = m_radius;
     params.numPosiPart = m_numPosiStars;
     params.numNegaPart = m_numNegaStars;
     params.posiMass    = m_totalSolarMass;
     params.negaMass    = m_negaDensity;
     params.pertAmp     = m_pertAmp;
     params.webCells    = m_webCells;
     params.foamMode    = m_foamMode;
     params.H0          = m_H0;
     params.cosmoScale  = m_useCosmology ? m_cosmoScale : 1.0f;
     // Zel'dovich ICs when cosmology ON (aStart < 0.2 ensures linear regime, no shell-crossing)
     // Zel'dovich ICs only for pure ΛCDM (no negative masses).
     // For Janus (physicsModel==1): always use free-explosion sphere — the Zel'dovich
     // approach assumes positive-only perturbations and is physically wrong with negative masses.
     bool janusMode = (m_eqSelectedModel == 1);
     // Janus PM: paired lattice ICs (freeExplosion=true + periodicBox=true).
     // Key property: paired lattice gives k=1 amplitude = EXACTLY ZERO (lattice symmetry).
     // → k=1 mode never collapses, even in periodic box!
     // aStart=0.3 required: a_inv3=37 (vs 8000) → dipole forces ×37 weaker → stable.
     // radius=1000, pertAmp=1.0: δ = 9.5 Mpc > cell/2 = 5.85 Mpc → ρ_net ≠ 0 → forces work.
     params.freeExplosion = !m_useCosmology || (m_aStart > 0.2f) || janusMode;
     // Cube périodique RETIRÉ (artefacts d'axes, pas de mousse stable). PM = bord ouvert.
     params.periodicBox   = false;

    UniverseGenerator gen(params);
    if (m_useCosmology && !params.freeExplosion) {
        // Reduced sigma8=0.4 to avoid shell-crossing at early aStart
        gen.setCosmology(m_omegaM, m_omegaL, 0.96f, 0.4f, m_aStart);
        gen.setGridResolution(64);
    }
    // Reproducibility: if user enabled manual seed, override the time-based default
    if (m_useManualSeed) gen.setSeed((unsigned int)m_manualSeed);
    gen.generate(m_positions, m_velocities);

    // Anti-correlated Zel'dovich ICs for Janus are handled inside the generator:
    // + at q+D1·s,  − at q−D1·s  →  ρ_net = 2·ρ_mean·δ(x) isotropic spectrum.
    // No co-location override needed (co-location made ρ_net≡0 → no forces).

// CM correction AFTER pairing — zeros net position and velocity of whole system
    {
        int n = (int)m_positions.size();
        if (n > 0) {
            double cmx=0,cmy=0,cmz=0,vcmx=0,vcmy=0,vcmz=0;
            for (int i = 0; i < n; i++) {
                cmx  += m_positions[i].x;  cmy  += m_positions[i].y;  cmz  += m_positions[i].z;
                vcmx += m_velocities[i].x; vcmy += m_velocities[i].y; vcmz += m_velocities[i].z;
            }
            cmx/=n; cmy/=n; cmz/=n; vcmx/=n; vcmy/=n; vcmz/=n;
            for (int i = 0; i < n; i++) {
                m_positions[i].x  -= (float)cmx;  m_positions[i].y  -= (float)cmy;  m_positions[i].z  -= (float)cmz;
                m_velocities[i].x -= (float)vcmx; m_velocities[i].y -= (float)vcmy; m_velocities[i].z -= (float)vcmz;
            }
        }
    }


    int totalCount = gen.getTotalCount();
    m_simulation->setParameters(totalCount, m_timeStep, m_softening, m_gravConstant * 1.0e10f);
    m_simulation->setHubble(m_H0);
    // En mode cosmologique : toujours Janus Bimétrique (modèle unique de Petit)
    int physModel = m_eqSelectedModel;   // modèle choisi dans l'Equation Editor (Janus par défaut)
    m_simulation->setPhysicsModel(physModel);
    m_simulation->setSpiralParams(0.0f, 0.0f, 0.0f, 0);

    if (m_useCosmology) {
        m_simulation->setCosmoScale(m_cosmoScale);
        m_simulation->setCosmology(m_omegaM, m_omegaL, m_aStart);
        m_simulation->setUsePM(m_usePM, m_pmNg, params.periodicBox, m_pmEnableSR);
        m_simulation->setUseTree(m_useTree);
        m_simulation->setMonopoleRemoval(m_removeMonopole, m_monopoleFrac);
        m_simulation->setTreeTheta(m_treeTheta);
    } else {
        m_simulation->setCosmoScale(1.0f);
        m_simulation->setCosmology(0.0f, 0.0f, 1.0f);  // non-cosmological: a=1, a_inv3=1
        // Non-cosmo uses pure N² (no PM): PM H&E creates stripe artifacts from sphere boundary
        // anisotropy. N² gives correct isotropic Janus forces → correct alveolar structure.
        m_simulation->setUsePM(false, m_pmNg);
        m_simulation->setUseTree(m_useTree);
        m_simulation->setMonopoleRemoval(m_removeMonopole, m_monopoleFrac);
        m_simulation->setTreeTheta(m_treeTheta);
    }

    // Janus cosmological: box = 2×radius so ICs fill the ENTIRE periodic box.
    // No empty space = no spurious k=1 forces, no cube boundary visible.
    // Janus cosmological: box = 2×radius so ICs fill the ENTIRE periodic box.
    // No empty space = no spurious k=1 forces, no cube boundary visible.
    // Cell size = 2R/Ng = 2×500/256 = 3.9 Mpc (good resolution for voids).
    // Open-boundary sphere PM: box = 3×radius → cell = 3R/Ng ≈ 5.9 Mpc for good resolution.
    // Periodic cosmological: box = 2×radius (particles fill the full periodic cube).
    float pmBoxSize = (!params.freeExplosion) ? m_radius * 3.0f
                    : (params.periodicBox)    ? m_radius * 2.0f
                    : (m_usePM)              ? m_radius * 3.0f
                    :                          0.0f;
    m_simulation->setBoxSize(pmBoxSize);

    // δρ correction: net mass density = (pos + neg masses) / V_box.
    // For symmetric Janus (N_pos×m_pos = N_neg×|m_neg|): rho_mean ≈ 0 → no-op.
    {
        float V_box = (4.0f / 3.0f) * 3.14159265f * m_radius * m_radius * m_radius;
        float mass_net = (float)m_numPosiStars * m_totalSolarMass
                       + (float)m_numNegaStars * m_negaDensity;  // m_negaDensity < 0
        float rho_mean = (V_box > 0.f) ? mass_net / V_box : 0.f;
        m_simulation->setRhoMean(rho_mean);
    }

    m_simulation->uploadInitialData(m_positions, m_velocities);
    // Copie hôte des vitesses : l'échelle de couleur est juste dès la première frame.
    m_rotVelocities = m_velocities;

    uploadParticlesToGPU();

    if (m_simulation->isGLSharing())
        m_needReshare = true;

    m_currentFrame     = 0;
    m_lastRefreshFrame = -9999;

    // Reset analytics histories
    m_galaxyCount = 0;
    m_firstGalaxyDetected = false;
    m_firstGalaxyTime     = 0.0f;
    m_galaxyCentroids.clear();
    m_galHistory.clear();   // galaxy-formation history plot
    m_expTimes.clear(); m_expA.clear(); m_expAdot.clear();   // historique d'expansion (a, ȧ)
    m_expThrottle = 4;  m_expTick = 0;
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
    m_rotVelocities.clear();   // force GPU re-read on next marker/analytics display

    // Cosmological mode: auto-reset camera to 3D perspective view
    if (m_useCosmology) {
        int vpW = m_renderer->getWidth();
        int vpH = m_renderer->getHeight();
        double camDist = std::max((double)m_radius * 20.0, 2000.0);
        delete m_camera;
        m_camera = new Camera(vpW, vpH,
            new Vector(0, 0, camDist), new Matrix(ROTATION, 0.0, 0.0, 0.0),
            60.0, 10.0, 1.0e10f);
        m_camera->orbit(-M_PI / 6.0, -M_PI / 3.0);  // 30° azimuth, 30° elevation
    }

    m_simRunning   = true;
    m_simPaused    = false;
    m_isGenerating = false;

    // Reset wall-clock timer pour la nouvelle simulation
    m_wallClockAccumMs = 0;
    m_wallClockStartMs = SDL_GetTicks64();
}

// Toggle pause/resume en tenant à jour le wall-clock timer.
// Le temps en pause n'est pas compté.
void App::togglePauseWithClock()
{
    uint64_t now = SDL_GetTicks64();
    if (!m_simPaused) {
        // En train de passer à PAUSED : accumuler le temps écoulé depuis le dernier resume
        m_wallClockAccumMs += now - m_wallClockStartMs;
    } else {
        // En train de passer à RUNNING : redémarrer le compteur de cette session
        m_wallClockStartMs = now;
    }
    m_simPaused = !m_simPaused;
}

void App::resetSimulation()
{
    // Abort any running FoF worker and wipe its catalog (stale after reset)
    m_fofCancel = true;
    if (m_fofThread.joinable()) m_fofThread.join();
    m_fofClusters.clear();
    m_fofResult.clear();
    m_fofDone = false;
    m_fofComputing = false;
    // Same for the galaxy-detect worker.
    if (m_galThread.joinable()) m_galThread.join();
    m_galComputing   = false;
    m_galResultReady = false;
    m_galaxyCentroids.clear();
    m_galaxyCount = 0;
    // And the density worker.
    if (m_densThread.joinable()) m_densThread.join();
    m_densComputing = false;
    m_densReady = false;
    m_galHistory.clear();   // galaxy-formation history plot
    m_fofResultReady = false;
    m_fofCancel = false;
    m_fofSelected = -1;
    m_fofLive = false;

    // Replay buffer invalid after regeneration (particle order changes)
    m_replayFrames.clear(); m_replayTimes.clear();
    m_replayMode = false; m_replayPlaying = false; m_replayCursor = 0;
    m_replayLastRecFrame = -1000000;

    m_simRunning = false;
    m_simPaused = true;
    m_currentFrame = 0;
    // Clear all persistent CPU buffers so no ghost particles from previous run
    // leak through into the new VBO upload.
    m_positions.clear();
    m_velocities.clear();
    m_rotVelocities.clear();
    m_particleData.clear();
    // Force a full VBO reallocation (subData=false) to flush any residual
    // device-side data that could be visible behind the new initial conditions.
    m_particleCount = 0;
    uploadParticlesToGPU(false);  // empty VBO
    startSimulation();             // generates new IC + re-allocates GPU buffers via setParameters
    m_simPaused = true;
}

// ─── Batch mode state machine ────────────────────────────────────────────────
// Run N simulations with seeds startSeed, startSeed+1, ... until each reaches
// batchTargetMY of simulated time, then export particles to CSV.
void App::exportBatchSnapshotCSV(const char* filename)
{
    if (!m_simulation || m_positions.empty()) return;
    m_simulation->readPositions(m_positions);
    std::ofstream f(filename);
    if (!f.is_open()) return;
    f << "# seed=" << (m_batchStartSeed + m_batchCurrentIdx)
      << "  N=" << m_positions.size()
      << "  simTime=" << (m_simulation->getSimTime() / 100.0) << " MY\n";
    f << "x,y,z,mass\n";
    for (const auto& p : m_positions) {
        f << p.x << "," << p.y << "," << p.z << "," << p.w << "\n";
    }
    f.close();
}

void App::tickBatchMode()
{
    if (!m_batchActive) return;

    switch (m_batchState) {
    case 1: {  // GENERATING
        // Set seed for this run
        m_manualSeed    = m_batchStartSeed + m_batchCurrentIdx;
        m_useManualSeed = true;
        // Make sure output dir exists (Windows)
        #ifdef _WIN32
        CreateDirectoryA(m_batchOutDir, NULL);
        #endif
        startSimulation();
        m_simPaused = false;  // ensure step is active
        m_batchState = 2;     // → RUNNING
        break;
    }
    case 2: {  // RUNNING : wait until target sim time
        if (!m_simulation) { m_batchState = 0; m_batchActive = false; break; }
        double simMY = (double)m_simulation->getSimTime() / 100.0;
        if (simMY >= (double)m_batchTargetMY) {
            m_simPaused = true;
            m_batchState = 3;  // → SAVING
        }
        break;
    }
    case 3: {  // SAVING
        char fname[512];
        int seed = m_batchStartSeed + m_batchCurrentIdx;
        snprintf(fname, sizeof(fname), "%s\\batch_seed_%04d.csv", m_batchOutDir, seed);
        exportBatchSnapshotCSV(fname);
        // Next run
        m_batchCurrentIdx++;
        if (m_batchCurrentIdx >= m_batchN) {
            // Batch complete
            m_batchActive = false;
            m_batchState  = 0;
        } else {
            m_batchState = 1;  // → next GENERATING
        }
        break;
    }
    default: break;
    }
}

void App::stepSimulation()
{
    if (!m_simulation || !m_simRunning || m_simPaused) return;

    if (m_simSpeed >= 1.0f)
    {
        // FAST: N STABLE sub-steps per frame (each at the normal/adaptive dt) instead
        // of one giant dt×speed step. Multiplying dt blows up dense nodes; sub-stepping
        // stays stable. GPU readback + upload + analytics run once per frame, amortized.
        m_simulation->setDtMul(1.0f);
        int nSub = (int)(m_simSpeed + 0.5f);
        if (nSub < 1)   nSub = 1;
        if (nSub > 200) nSub = 200;
        for (int s = 0; s < nSub; s++) {
            m_simulation->step();
            m_currentFrame++;
        }
        m_simulation->readPositions(m_positions);
        if (m_colorByVelocity)
            m_simulation->readVelocities(m_rotVelocities);
    }
    else
    {
        // SLOW: one step EVERY frame with a reduced dt (× Sim Speed) -> smooth
        // slow-motion. (The old skip-every-N-frames approach froze the structure
        // between steps -> stepwise/choppy motion.)
        m_simulation->setDtMul(m_simSpeed);
        m_simulation->step();
        m_simulation->readPositions(m_positions);
        if (m_colorByVelocity)
            m_simulation->readVelocities(m_rotVelocities);
        m_currentFrame++;
    }
    uploadParticlesToGPU();

    // Replay timeline: snapshot the positions every N frames
    if (m_replayRecord && (m_currentFrame - m_replayLastRecFrame) >= m_replayInterval) {
        m_replayLastRecFrame = m_currentFrame;
        replayRecordSnapshot();
    }

    // Galaxy detection: cheap SCHEDULER every frame (adopts the background result /
    // launches the worker every rotRefreshInterval). The heavy FoF runs off-thread.
    detectGalaxies();

    bool anyAnalytics = m_showDensityProfile || m_showPhaseSpace ||
                        m_showEnergyPlot    || m_showBarycenters  ||
                        m_showConservation  || m_showEscape       || m_showCorrelation ||
                        m_showFourier       || m_showDarkMatter   ||
                        m_showDispersion    || m_showSersic;
    if (m_rotAutoRefresh && anyAnalytics &&
        (m_currentFrame - m_lastRefreshFrame >= m_rotRefreshInterval))
    {
        m_lastRefreshFrame = m_currentFrame;
        refreshAnalytics();
    }

    // Historique d'expansion (a, ȧ) pour la fenêtre Expansion — léger, throttlé.
    if (m_simulation && m_useCosmology && m_simRunning && (++m_expTick % m_expThrottle == 0)) {
        float a = m_simulation->getScaleFactor();
        float H = m_simulation->getHubbleParam();
        m_expTimes.push_back((double)m_curAgeGyr);
        m_expA.push_back(a);
        m_expAdot.push_back(H * a);             // ȧ = vitesse d'expansion
        // Buffer NON bouclant : à saturation, décimation ×2 (garde TOUT le span,
        // y compris le creux de transition) + throttle doublé. Évite que le creux
        // de ȧ soit écrasé sur un très long run (sinon repères dégénérés).
        if (m_expTimes.size() >= 8192) {
            size_t w = 0;
            for (size_t r = 0; r < m_expTimes.size(); r += 2) {
                m_expTimes[w]=m_expTimes[r]; m_expA[w]=m_expA[r]; m_expAdot[w]=m_expAdot[r]; w++;
            }
            m_expTimes.resize(w); m_expA.resize(w); m_expAdot.resize(w);
            m_expThrottle *= 2;
        }
    }

    updateMarkerPaths();
}

// SCHEDULER (cheap, called every frame). Adopts a finished background result and
// launches a new detection when idle and due. The heavy FoF runs on a worker thread
// (detectGalaxiesWorker) so a collapsed structure (Newton: ~1 s of FoF) never blocks
// the frame — same pattern as the cluster Halo Finder.
void App::detectGalaxies()
{
    // 1) Adopt a finished worker result.
    if (m_galResultReady.load()) {
        if (m_galThread.joinable()) m_galThread.join();
        m_galaxyCentroids = std::move(m_galResult);
        m_galaxyCount     = m_galResultCount;
        m_galResult.clear();
        m_galResultReady = false;
        m_galComputing   = false;
        m_galLastLaunchMs = SDL_GetTicks64();   // cooldown measured from COMPLETION
        if (!m_firstGalaxyDetected && m_galaxyCount > 0 && m_simulation) {
            m_firstGalaxyDetected = true;
            m_firstGalaxyTime = (float)(m_simulation->getSimTime() / 100.0);
        }
    }
    // 2) Launch a new detection if idle and enough WALL-CLOCK time elapsed. Using
    //    real time (not m_currentFrame, which jumps by nSub with Sim Speed) keeps the
    //    cadence stable at any Sim Speed -> no over-launching / thread contention.
    if (!m_galComputing.load() && m_positions.size() >= 2 &&
        (SDL_GetTicks64() - m_galLastLaunchMs) >= (uint64_t)m_galIntervalMs) {
        m_galLastLaunchMs = SDL_GetTicks64();
        m_galPosCopy  = m_positions;          // snapshot (worker reads its own copy)
        m_galMinSnap  = m_galaxyMinPart;
        m_galLinkSnap = m_galaxyLinkScale;
        m_galComputing   = true;
        m_galResultReady = false;
        m_galThread = std::thread(&App::detectGalaxiesWorker, this);
    }
}

// WORKER (background thread). FoF (Friends-of-Friends) at SHORT linking length to
// isolate dense nodes (galaxies). Operates on the snapshot m_galPosCopy and writes
// m_galResult / m_galResultCount. Touches no main-thread state besides those.
void App::detectGalaxiesWorker()
{
    const std::vector<cl_float4>& P = m_galPosCopy;
    m_galResult.clear();
    m_galResultCount = 0;

    std::vector<int> idx;
    idx.reserve(P.size());
    for (int i = 0; i < (int)P.size(); i++)
        if (P[i].w > 0.0f) idx.push_back(i);
    int N = (int)idx.size();
    if (N < 2) { m_galResultReady = true; return; }

    float mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
    for (int k = 0; k < N; k++) {
        const cl_float4& p = P[idx[k]];
        mn[0]=fminf(mn[0],p.x); mx[0]=fmaxf(mx[0],p.x);
        mn[1]=fminf(mn[1],p.y); mx[1]=fmaxf(mx[1],p.y);
        mn[2]=fminf(mn[2],p.z); mx[2]=fmaxf(mx[2],p.z);
    }
    float Lx=fmaxf(mx[0]-mn[0],1e-3f), Ly=fmaxf(mx[1]-mn[1],1e-3f), Lz=fmaxf(mx[2]-mn[2],1e-3f);
    float meanSep = cbrtf((Lx*Ly*Lz) / (float)N);
    float b = m_galLinkSnap * meanSep;
    if (b < 1e-4f) b = 1e-4f;
    float b2 = b * b;
    float invB = 1.0f / b;

    std::unordered_map<uint64_t, std::vector<int>> grid;
    grid.reserve(N * 2);
    auto keyOf = [&](int32_t cx, int32_t cy, int32_t cz) -> uint64_t {
        return ((uint64_t)(uint32_t)cx & 0x1FFFFF)
             | (((uint64_t)(uint32_t)cy & 0x1FFFFF) << 21)
             | (((uint64_t)(uint32_t)cz & 0x1FFFFF) << 42);
    };
    for (int k = 0; k < N; k++) {
        const cl_float4& p = P[idx[k]];
        int32_t cx=(int32_t)floorf((p.x-mn[0])*invB), cy=(int32_t)floorf((p.y-mn[1])*invB), cz=(int32_t)floorf((p.z-mn[2])*invB);
        grid[keyOf(cx,cy,cz)].push_back(k);
    }

    std::vector<int> parent(N), rnk(N, 0);
    for (int k = 0; k < N; k++) parent[k] = k;
    std::function<int(int)> find = [&](int a){ while(parent[a]!=a){parent[a]=parent[parent[a]];a=parent[a];} return a; };
    auto unite = [&](int a, int c){ a=find(a); c=find(c); if(a==c)return; if(rnk[a]<rnk[c])std::swap(a,c); parent[c]=a; if(rnk[a]==rnk[c])rnk[a]++; };

    // Per-particle neighbour-check cap: bounds work to O(N·CAP) even when a collapsed
    // clump piles into one hash cell (otherwise O(N^2)). Connectivity preserved.
    const int NB_CAP = 256;
    for (int k = 0; k < N; k++) {
        const cl_float4& p = P[idx[k]];
        int32_t bx=(int32_t)floorf((p.x-mn[0])*invB), by=(int32_t)floorf((p.y-mn[1])*invB), bz=(int32_t)floorf((p.z-mn[2])*invB);
        int checks = 0;
        for (int dz=-1;dz<=1 && checks<=NB_CAP;dz++) for (int dy=-1;dy<=1 && checks<=NB_CAP;dy++) for (int dx=-1;dx<=1 && checks<=NB_CAP;dx++) {
            auto it = grid.find(keyOf(bx+dx,by+dy,bz+dz));
            if (it == grid.end()) continue;
            for (int m : it->second) {
                if (++checks > NB_CAP) break;
                if (m <= k) continue;
                const cl_float4& q = P[idx[m]];
                float ddx=q.x-p.x, ddy=q.y-p.y, ddz=q.z-p.z;
                if (ddx*ddx+ddy*ddy+ddz*ddz <= b2) unite(k, m);
            }
        }
    }

    // Group members by FoF root.
    std::unordered_map<int, std::vector<int>> groups;
    for (int k = 0; k < N; k++) groups[find(k)].push_back(k);

    int maxComp = (int)(N * 0.5f);
    int count = 0;
    for (auto& g : groups) {
        std::vector<int>& mem = g.second;
        int cnt = (int)mem.size();
        if (cnt < m_galMinSnap || cnt >= maxComp) continue;

        // Mass centroid, then SHRINKING-SPHERE refinement: re-center on the dense
        // CORE (the bright blob) instead of the plain mass centroid, which a filament
        // tail pulls off-peak.
        double cx=0, cy=0, cz=0, cm=0;
        for (int li : mem) { const cl_float4& p=P[idx[li]]; float am=fabsf(p.w);
            cx+=p.x*am; cy+=p.y*am; cz+=p.z*am; cm+=am; }
        if (cm <= 0) continue;
        float ccx=(float)(cx/cm), ccy=(float)(cy/cm), ccz=(float)(cz/cm);
        for (int iter=0; iter<2; iter++) {
            double r2s=0; for (int li:mem){ const cl_float4& p=P[idx[li]];
                double dx=p.x-ccx,dy=p.y-ccy,dz=p.z-ccz; r2s+=dx*dx+dy*dy+dz*dz; }
            float rcut2 = (0.6f*0.6f) * (float)(r2s / mem.size());
            double sx=0,sy=0,sz=0,sm=0; int inN=0;
            for (int li:mem){ const cl_float4& p=P[idx[li]];
                float dx=p.x-ccx,dy=p.y-ccy,dz=p.z-ccz;
                if (dx*dx+dy*dy+dz*dz <= rcut2){ float am=fabsf(p.w);
                    sx+=p.x*am; sy+=p.y*am; sz+=p.z*am; sm+=am; inN++; } }
            if (inN < 8 || sm <= 0) break;
            ccx=(float)(sx/sm); ccy=(float)(sy/sm); ccz=(float)(sz/sm);
        }
        // Representative member nearest the core -> its GLOBAL index. The marker is
        // drawn at this particle's CURRENT position each frame -> tracks the blob live
        // (no lag from the async snapshot). Stored in the .w field.
        int rep=-1; float best=1e30f;
        for (int li:mem){ const cl_float4& p=P[idx[li]];
            float dx=p.x-ccx,dy=p.y-ccy,dz=p.z-ccz; float d2=dx*dx+dy*dy+dz*dz;
            if (d2<best){ best=d2; rep=idx[li]; } }
        m_galResult.push_back({ ccx, ccy, ccz, (float)rep });
        count++;
    }
    m_galResultCount = count;

    // Free the heavy locals NOW, on the worker thread, BEFORE signaling ready (so the
    // scheduler's join() is instant — destroying the big maps would otherwise stall it).
    std::unordered_map<uint64_t, std::vector<int>>().swap(grid);
    std::unordered_map<int, std::vector<int>>().swap(groups);
    std::vector<int>().swap(parent);
    std::vector<int>().swap(rnk);
    std::vector<int>().swap(idx);

    m_galResultReady = true;   // signal the scheduler (release: result fully written)
}

void App::refreshAnalytics()
{
    if (!m_simulation) return;
    m_simulation->readPositions(m_positions);
    m_simulation->readVelocities(m_rotVelocities);
    detectGalaxies();

    // Velocity colour scale (auto / frozen) is handled inside uploadParticlesToGPU;
    // just refresh the GPU buffer when colouring by velocity with auto-scale.
    if (m_colorByVelocity && (m_velAutoScale || m_velPeakHold) && !m_rotVelocities.empty())
        uploadParticlesToGPU();
    // Density colour : recompute local density as the structure evolves.
    if (m_colorByDensity)
        uploadParticlesToGPU();
    // Halo colour (FoF) : lourd -> recalcul ~1 fois/60 frames (+ immédiat si vide).
    if (m_colorByHalo) {
        if (m_haloSize.size() != m_positions.size() || (++m_foFTick % 60) == 0) computeHaloColoring();
        uploadParticlesToGPU();
    }
    // T-web : lourd -> recalcul ~1/60 frames (+ immédiat si vide).
    if (m_colorByWeb) {
        if (m_webClass.size() != m_positions.size() || (++m_webTick % 60) == 0) computeTweb();
        uploadParticlesToGPU();
    }
    // Température : recalcul ~1/30 frames (+ IMMÉDIAT si vide -> couleur dès le lancement).
    if (m_colorByTemp) {
        if (m_temp.size() != m_positions.size() || (++m_tempTick % 30) == 0) computeTemperature();
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
        double px=0,py=0,pm=0, nx=0,ny=0,nm=0;
        for (size_t i = 0; i < np; i++) {
            const cl_float4& p = m_positions[i];
            if (p.w > 0.f) { px+=p.x*p.w; py+=p.y*p.w; pm+=p.w; }
            else if (p.w < 0.f) { double am=fabsf(p.w); nx+=p.x*am; ny+=p.y*am; nm+=am; }
        }
        if (pm > 0) { px/=pm; py/=pm; }
        if (nm > 0) { nx/=nm; ny/=nm; }
        m_baryPosX = (float)px; m_baryPosY = (float)py;
        m_baryNegX = (float)nx; m_baryNegY = (float)ny;
        float dist = sqrtf((float)((px-nx)*(px-nx)+(py-ny)*(py-ny))) * m_lyPerUnit;
        m_baryDist [m_baryHead] = dist;
        m_baryTimes[m_baryHead] = (float)m_currentFrame * m_timeStep;
        m_baryHead = (m_baryHead + 1) % BARY_HISTORY;
        if (m_baryCount < BARY_HISTORY) m_baryCount++;
    }

    {
        double re[4]={}, im_[4]={};
        int cnt = 0;
        for (size_t i = 0; i < np; i++) {
            const cl_float4& p = m_positions[i];
            if (p.w <= 0.f) continue;
            float r = sqrtf(p.x*p.x + p.y*p.y);
            if (r < 5.f) continue;
            float theta = atan2f(p.y, p.x);
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
            // Lz = m * (x*vy - y*vx)
            Lz     += pos.w * (pos.x * vel.y - pos.y * vel.x);
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
            sumVneg += sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
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
    uint32_t version    = 3u;  // ← v3 : + facteur d'échelle cosmologique (a, da)
    uint32_t nParticles = (uint32_t)pos.size();
    int32_t  appFrame   = m_currentFrame;
    float    simTime    = m_simulation->getSimTime();
    float    timeStep   = m_timeStep;
    float    cosmoA     = m_simulation->getScaleFactor();
    float    cosmoDa    = m_simulation->getDa();

    f.write(reinterpret_cast<const char*>(&magic),      4);
    f.write(reinterpret_cast<const char*>(&version),    4);
    f.write(reinterpret_cast<const char*>(&nParticles), 4);
    f.write(reinterpret_cast<const char*>(&appFrame),   4);
    f.write(reinterpret_cast<const char*>(&simTime),    4);
    f.write(reinterpret_cast<const char*>(&timeStep),   4);
    f.write(reinterpret_cast<const char*>(&cosmoA),     4);   // v3
    f.write(reinterpret_cast<const char*>(&cosmoDa),    4);   // v3
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
    ofn.lpstrFilter   = "J-Gravity State (*.bin)\0*.bin\0All Files (*.*)\0*.*\0";
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

    if (magic != 0x4A475354u || version < 1u || version > 3u) { std::cerr << "Invalid state file\n"; return; }

    // v3: cosmological scale factor (a, da), stored between the header and the data.
    float cosmoA = 0.0f, cosmoDa = 0.0f; bool haveCosmoState = false;
    if (version >= 3u) {
        f.read(reinterpret_cast<char*>(&cosmoA),  4);
        f.read(reinterpret_cast<char*>(&cosmoDa), 4);
        haveCosmoState = true;
    }

    std::vector<cl_float4> pos(nParticles), vel(nParticles);
    f.read(reinterpret_cast<char*>(pos.data()), nParticles * sizeof(cl_float4));
    f.read(reinterpret_cast<char*>(vel.data()), nParticles * sizeof(cl_float4));
    if (!f) { std::cerr << "Truncated state file\n"; return; }

    m_positions  = pos;
    m_velocities = vel;

    m_simulation->setParameters((int)nParticles, m_timeStep, m_softening,
                                 m_gravConstant * 1.0e10f);
    m_simulation->setHubble(m_H0);
    m_simulation->setPhysicsModel(m_eqSelectedModel);   // respecte le modèle choisi (Janus par défaut)
    float pitchTan = tanf(m_spPitchAngle * 3.14159265f / 180.0f);
    m_simulation->setSpiralParams(m_spAmplitude, m_spPatternSpeed, pitchTan, (unsigned int)m_spArms);

    // ── Restore the cosmology + solver setup (mirror generateParticles), otherwise
    //    the reloaded run uses the wrong scale factor / forces -> cube + blob artifacts.
    bool janusMode    = (m_eqSelectedModel == 1);
    bool periodicBox  = false;  // cube périodique retiré (artefacts d'axes)
    bool freeExplos   = !m_useCosmology || (m_aStart > 0.2f) || janusMode;
    if (m_useCosmology) {
        m_simulation->setCosmoScale(m_cosmoScale);
        m_simulation->setCosmology(m_omegaM, m_omegaL, m_aStart);   // sets a = aStart
        m_simulation->setUsePM(m_usePM, m_pmNg, periodicBox, m_pmEnableSR);
        m_simulation->setUseTree(m_useTree);
        m_simulation->setMonopoleRemoval(m_removeMonopole, m_monopoleFrac);
        m_simulation->setTreeTheta(m_treeTheta);
    } else {
        m_simulation->setCosmoScale(1.0f);
        m_simulation->setCosmology(0.0f, 0.0f, 1.0f);
        m_simulation->setUsePM(false, m_pmNg);
        m_simulation->setUseTree(m_useTree);
        m_simulation->setMonopoleRemoval(m_removeMonopole, m_monopoleFrac);
        m_simulation->setTreeTheta(m_treeTheta);
    }
    float pmBoxSize = (!freeExplos) ? m_radius * 3.0f
                    : (periodicBox) ? m_radius * 2.0f
                    : (m_usePM)     ? m_radius * 3.0f : 0.0f;
    m_simulation->setBoxSize(pmBoxSize);
    {
        float V_box = (4.0f / 3.0f) * 3.14159265f * m_radius * m_radius * m_radius;
        float mass_net = (float)m_numPosiStars * m_totalSolarMass
                       + (float)m_numNegaStars * m_negaDensity;
        m_simulation->setRhoMean((V_box > 0.f) ? mass_net / V_box : 0.f);
    }

    m_simulation->uploadInitialData(m_positions, m_velocities);
    m_simulation->setSimTime(simTime);

    // Restore the scale factor a(t): exact from the file (v3) or reconstructed from
    // simTime by re-integrating Friedmann (old v1/v2 files that didn't store it).
    if (m_useCosmology) {
        if (haveCosmoState) m_simulation->setCosmoState(cosmoA, cosmoDa);
        else                m_simulation->reconstructScaleFactor(simTime);
    }
    // Même raison qu'à la génération : état chargé en pause, aucun pas ne
    // rafraîchira les vitesses hôte avant le premier rendu.
    m_rotVelocities = m_velocities;

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
    if (version >= 2u) {
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
    // Open Save dialog — default to Desktop
    char filename[MAX_PATH] = "jgravity_screenshot.png";

#ifdef _WIN32
    char desktopPath[MAX_PATH] = {};
    const char* userProfile = getenv("USERPROFILE");
    if (userProfile)
        snprintf(desktopPath, MAX_PATH, "%s\\Desktop", userProfile);

    OPENFILENAMEA ofn = {};
    ofn.lStructSize  = sizeof(ofn);
    ofn.hwndOwner    = NULL;
    ofn.lpstrFilter  = "PNG Image (*.png)\0*.png\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile    = filename;
    ofn.nMaxFile     = MAX_PATH;
    ofn.lpstrDefExt  = "png";
    ofn.lpstrTitle   = "Save Screenshot";
    ofn.lpstrInitialDir = desktopPath[0] ? desktopPath : NULL;
    ofn.Flags        = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (!GetSaveFileNameA(&ofn))
        return;  // user cancelled
#endif

    // Defer the actual capture to the end of the frame, AFTER ImGui has rendered,
    // so overlays (grid, markers, annotations) are included in the PNG.
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
    float scaleX = (m_width > 0) ? (float)w / (float)m_width : 1.0f;
    int panelPx = (int)(340.0f * scaleX + 0.5f);
    if (panelPx < 0) panelPx = 0; if (panelPx >= w) panelPx = 0;
    int cw = w - panelPx;
    if (cw <= 0) { cw = w; panelPx = 0; }

    std::vector<unsigned char> pixels((size_t)cw * h * 3);
    glReadPixels(panelPx, 0, cw, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    for (int row = 0; row < h / 2; row++) {
        unsigned char* top = pixels.data() + (size_t)row * cw * 3;
        unsigned char* bot = pixels.data() + (size_t)(h - 1 - row) * cw * 3;
        for (int c = 0; c < cw * 3; c++) std::swap(top[c], bot[c]);
    }
    stbi_write_png(m_screenshotPath, cw, h, 3, pixels.data(), cw * 3);
    m_screenshotPath[0] = '\0';
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
            float ah = 14.0f, aw = 7.0f;
            ImVec2 base(b.x - ux*ah, b.y - uy*ah);
            ImVec2 p1(base.x - uy*aw, base.y + ux*aw);
            ImVec2 p2(base.x + uy*aw, base.y - ux*aw);
            dl->AddTriangleFilled(b, p1, p2, col);
        }
    };

    for (const auto& an : m_annotations) {
        ImVec2 p0(an.x0, an.y0), p1(an.x1, an.y1);
        switch (an.type) {
            case 1: {
                drawArrow(p0, p1, an.col);
                if (an.text[0]) {
                    ImVec2 ts = ImGui::CalcTextSize(an.text);
                    ImVec2 tp(p0.x - ts.x*0.5f, p0.y - ts.y - 4);
                    dl->AddRectFilled(ImVec2(tp.x-3,tp.y-2), ImVec2(tp.x+ts.x+3,tp.y+ts.y+2), IM_COL32(0,0,0,150));
                    dl->AddText(tp, an.col, an.text);
                }
                break;
            }
            case 2: {
                if (an.text[0]) {
                    ImVec2 ts = ImGui::CalcTextSize(an.text);
                    dl->AddRectFilled(ImVec2(p0.x-3,p0.y-2), ImVec2(p0.x+ts.x+3,p0.y+ts.y+2), IM_COL32(0,0,0,150));
                    dl->AddText(p0, an.col, an.text);
                }
                break;
            }
            case 3:
                dl->AddRect(ImVec2(fminf(p0.x,p1.x),fminf(p0.y,p1.y)),
                            ImVec2(fmaxf(p0.x,p1.x),fmaxf(p0.y,p1.y)), an.col, 0.0f, 0, 2.0f);
                break;
            case 4: {
                float r = sqrtf((p1.x-p0.x)*(p1.x-p0.x) + (p1.y-p0.y)*(p1.y-p0.y));
                if (r < 2) r = 2;
                dl->AddCircle(p0, r, an.col, 64, 2.0f);
                break;
            }
        }
    }

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
            if (an.type == 2 && an.text[0] == '\0')
                m_annotations.erase(m_annotations.begin() + m_annotEditIdx);
            m_annotEditIdx = -1;
        } else if (cancel) {
            if (an.type == 2 && an.text[0] == '\0')
                m_annotations.erase(m_annotations.begin() + m_annotEditIdx);
            m_annotEditIdx = -1;
        }
        ImGui::End();
    }
}

// ─── MP4 video recording (viewport streamed to ffmpeg via a pipe) ────────────
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
    return std::string("ffmpeg.exe");
#else
    return std::string("ffmpeg");
#endif
}

bool App::startVideoRecording()
{
#ifdef _WIN32
    if (m_videoRecording) return true;
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

    int w, h; SDL_GL_GetDrawableSize(m_window, &w, &h);
    float scaleX = (m_width > 0) ? (float)w / (float)m_width : 1.0f;
    int panelPx = (int)(340.0f * scaleX + 0.5f);
    if (panelPx < 0 || panelPx >= w) panelPx = 0;
    m_recW = (w - panelPx) & ~1;
    m_recH = h & ~1;
    if (m_recW <= 0 || m_recH <= 0) return false;

    std::string ff = findFFmpeg();
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "\"%s\" -y -f rawvideo -pixel_format rgb24 -video_size %dx%d -framerate %d "
        "-i - -vf vflip -c:v libx264 -pix_fmt yuv420p -crf 18 -preset veryfast \"%s\"",
        ff.c_str(), m_recW, m_recH, m_videoFps, m_videoPath);

    SECURITY_ATTRIBUTES sa = {}; sa.nLength = sizeof(sa); sa.bInheritHandle = TRUE;
    HANDLE rd = NULL, wr = NULL;
    if (!CreatePipe(&rd, &wr, &sa, 1 << 22)) return false;
    SetHandleInformation(wr, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si = {}; si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdInput  = rd;
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi = {};
    BOOL ok = CreateProcessA(NULL, cmd, NULL, NULL, TRUE,
                             CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    CloseHandle(rd);
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
        WaitForSingleObject((HANDLE)m_ffmpegProc, 15000);
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
        // Stream straight to disk: unlimited length, never lost, no RAM ring.
        replayWriteDiskFrame();
        return;
    }
    m_replayFrames.push_back(m_positions);
    m_replayTimes.push_back(m_simulation ? m_simulation->getSimTime() / 100.0f : 0.0f);
    while ((int)m_replayFrames.size() > m_replayMaxFrames) {
        m_replayFrames.erase(m_replayFrames.begin());
        m_replayTimes.erase(m_replayTimes.begin());
    }
}

// ── Persist the recorded replay to disk so it survives app close / overwrite ──
void App::saveReplay()
{
    if (m_replayFrames.empty()) { std::cerr << "[Replay] nothing to save\n"; return; }
    std::ofstream f(m_replayFilePath, std::ios::binary);
    if (!f.is_open()) { std::cerr << "[Replay] cannot write: " << m_replayFilePath << "\n"; return; }
    uint32_t magic = 0x4A525031u;  // "JRP1"
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

void App::loadReplay()
{
    std::ifstream f(m_replayFilePath, std::ios::binary);
    if (!f.is_open()) { std::cerr << "[Replay] cannot open: " << m_replayFilePath << "\n"; return; }
    uint32_t magic = 0, nSnaps = 0, nParts = 0;
    f.read(reinterpret_cast<char*>(&magic),  4);
    f.read(reinterpret_cast<char*>(&nSnaps), 4);
    f.read(reinterpret_cast<char*>(&nParts), 4);
    if (magic != 0x4A525031u || nParts == 0 || nSnaps == 0) { std::cerr << "[Replay] invalid file\n"; return; }
    m_replayFrames.clear(); m_replayTimes.clear();
    m_replayFrames.reserve(nSnaps); m_replayTimes.reserve(nSnaps);
    for (uint32_t i = 0; i < nSnaps; i++) {
        float t = 0.0f;
        f.read(reinterpret_cast<char*>(&t), 4);
        std::vector<cl_float4> snap(nParts);
        f.read(reinterpret_cast<char*>(snap.data()), (size_t)nParts * sizeof(cl_float4));
        if (!f) { std::cerr << "[Replay] truncated at snap " << i << "\n"; break; }
        m_replayFrames.push_back(std::move(snap));
        m_replayTimes.push_back(t);
    }
    m_replayCursor = 0;
    m_replayMode = !m_replayFrames.empty();
    if (m_replayMode) { m_replayPlaying = false; m_simPaused = true; }
    printf("[Replay] loaded %zu snapshots from %s\n", m_replayFrames.size(), m_replayFilePath);
}

// ── Streaming replay: each snapshot written straight to a folder (unlimited) ──
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
    m_replayTimes.clear();
    m_replayFrames.clear();
    int idx = 0;
    for (;;) {
        char fname[640];
        snprintf(fname, sizeof(fname), "%s\\snap_%06d.bin", m_replayDir, idx);
        std::ifstream f(fname, std::ios::binary);
        if (!f.is_open()) break;
        float t = 0.0f;
        f.read(reinterpret_cast<char*>(&t), 4);
        if (!f) break;
        m_replayTimes.push_back(t);
        idx++;
    }
    m_replayDiskCount = idx;
    if (m_replayDiskCount > 0) {
        m_replayStream  = true;
        m_replayCursor  = 0;
        m_replayMode    = true;
        m_replayPlaying = false;
        m_simPaused     = true;
        replaySetFrame(0);
    }
    printf("[Replay] folder scan: %d snapshots in %s\n", m_replayDiskCount, m_replayDir);
}

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
    if (m_replayCursor != m_replayShownCursor) {   // only (re)load when the frame changes
        replaySetFrame(m_replayCursor);
        m_replayShownCursor = m_replayCursor;
    }
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
    float velConv = (m_lyPerUnit * 9.461e12f) / 3.156e13f;
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

        float x  = pos.x, y = pos.y;
        float vx = vel.x, vy = vel.y;
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
    fmtLY(rLyMax, buf, sizeof(buf));
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

    // Lissage des bins par moyenne glissante (fenêtre ±4 bins)
    const int RC_SMOOTH = 4;
    std::vector<float> vSmooth(nBins, 0.0f);
    for (int b = 0; b < nBins; b++)
    {
        float sumV2 = 0.0f; int n2 = 0;
        for (int k = -RC_SMOOTH; k <= RC_SMOOTH; k++)
        {
            int bb = b + k;
            if (bb < 0 || bb >= nBins || cnt[bb] < 3) continue;
            sumV2 += sumV[bb] / cnt[bb]; n2++;
        }
        vSmooth[b] = (n2 > 0) ? sumV2 / n2 : 0.0f;
    }

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
        dl->AddLine(ImVec2(lx,ly+6),   ImVec2(lx+22,ly+6),   colCurve,              2.0f);
        dl->AddText(ImVec2(lx+24,ly),  IM_COL32(140,200,255,220), "Janus (sim)");
        dl->AddLine(ImVec2(lx,ly+20),  ImVec2(lx+22,ly+20),  IM_COL32(255,180,0,180),2.0f);
        dl->AddText(ImVec2(lx+24,ly+14),IM_COL32(255,200,80,220),"Keplerian");

        // Compteur de temps de simulation
        double elapsed = m_simulation ? (double)m_simulation->getSimTime() / 100.0 : 0.0;
        char timeBuf[64];
        if (elapsed < 1000.0)
            snprintf(timeBuf, sizeof(timeBuf), "T = %.2f MY", elapsed);
        else
            snprintf(timeBuf, sizeof(timeBuf), "T = %.3f Gy", elapsed / 1000.0);
        dl->AddText(ImVec2(lx, ly+30), IM_COL32(180,220,180,220), timeBuf);
    }

    ImGui::Dummy(canvasSize);
    ImGui::End();
}

// ─── Density Profile ─────────────────────────────────────────────────────────
void App::renderDensityProfile()
{
    if (!m_showDensityProfile) return;

    // Petite barre de contrôle au-dessus du plot (button Refresh + checkbox log)
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

    // Bouton Refresh (lecture des positions depuis GPU)
    if (ImGui::Button("Refresh") && m_simulation) m_simulation->readPositions(m_positions);
    ImGui::SameLine();
    static bool logScale = true;
    ImGui::Checkbox("Log scale (Y)", &logScale);
    ImGui::SameLine();
    ImGui::TextDisabled("3D shell-normalized density");
    ImGui::Separator();

    // Reproduce beginPlotWindow logic inline (we can't call it because we used some space)
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

    // Find actual 3D radial extent (use current positions, not init radius × 4)
    float rMaxSim = 0.0f;
    size_t np = m_positions.size();
    for (size_t i = 0; i < np; i++) {
        float r2 = m_positions[i].x*m_positions[i].x
                 + m_positions[i].y*m_positions[i].y
                 + m_positions[i].z*m_positions[i].z;
        if (r2 > rMaxSim) rMaxSim = r2;
    }
    rMaxSim = sqrtf(rMaxSim);
    if (rMaxSim < 1.0f) rMaxSim = m_radius * 1.2f;
    float rMax = rMaxSim * m_lyPerUnit;

    // Bin particles by TRUE 3D radius
    for (size_t i = 0; i < np; i++) {
        float r3d = sqrtf(m_positions[i].x*m_positions[i].x
                        + m_positions[i].y*m_positions[i].y
                        + m_positions[i].z*m_positions[i].z) * m_lyPerUnit;
        int b = (int)(r3d / rMax * BINS);
        if (b < 0) b = 0; if (b >= BINS) b = BINS-1;
        if      (m_positions[i].w > 0) countPos[b]++;
        else if (m_positions[i].w < 0) countNeg[b]++;
    }

    // Normalize by spherical shell volume → real density (N/LY^3)
    // V_shell = (4π/3) × (r_outer^3 - r_inner^3)
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

    // Gaussian smoothing (σ = 3 bins, kernel ±9 bins for 99.7% coverage)
    float smoothPos[BINS]={}, smoothNeg[BINS]={};
    const int sigBins = 3;
    const int halfK   = sigBins * 3;
    for (int b = 0; b < BINS; b++) {
        double sumW = 0.0, sumWP = 0.0, sumWN = 0.0;
        for (int k = -halfK; k <= halfK; k++) {
            int bb = b + k; if (bb < 0 || bb >= BINS) continue;
            double w = exp(-0.5 * (double)(k*k) / (double)(sigBins*sigBins));
            sumW  += w;
            sumWP += w * binPos[bb];
            sumWN += w * binNeg[bb];
        }
        smoothPos[b] = (sumW > 0.0) ? (float)(sumWP / sumW) : binPos[b];
        smoothNeg[b] = (sumW > 0.0) ? (float)(sumWN / sumW) : binNeg[b];
    }

    // Transformation : log10 si demandé, linéaire sinon
    // En log : on shift d'un epsilon pour éviter log(0) → -inf
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
            plotPos[b] = smoothPos[b];
            plotNeg[b] = smoothNeg[b];
            if (plotPos[b] > maxVal) maxVal = plotPos[b];
            if (plotNeg[b] > maxVal) maxVal = plotNeg[b];
        }
        minVal = 0.0f;
    }
    if (maxVal <= minVal) { maxVal = minVal + 1.0f; }
    float yRange = maxVal - minVal;

    auto yMap = [&](float v) -> float {
        float t = (v - minVal) / yRange;
        if (t < 0.0f) t = 0.0f; if (t > 1.0f) t = 1.0f;
        return cp.y + (1.0f - t) * plotH;
    };
    float baseline = cp.y + plotH;

    // Remplissage + contour — positives (rouge)
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

    // Remplissage + contour — négatives (fuchsia → cyan layout pour contraste)
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

    // ── Légende centrée en haut ───────────────────────────────────────
    {
        float legW = 230, legH = 22;
        float lx = cp.x + (plotW - legW) * 0.5f;
        float ly = cp.y + 4;
        dl->AddRectFilled(ImVec2(lx, ly), ImVec2(lx + legW, ly + legH),
                          IM_COL32(0, 0, 0, 160));
        // Item Positive
        float ix = lx + 8;
        dl->AddLine(ImVec2(ix, ly + 11), ImVec2(ix + 18, ly + 11),
                    IM_COL32(255, 100, 100, 230), 2.5f);
        dl->AddText(ImVec2(ix + 24, ly + 4), IM_COL32(240, 200, 200, 255), "Positive");
        // Item Negative
        ix += 110;
        dl->AddLine(ImVec2(ix, ly + 11), ImVec2(ix + 18, ly + 11),
                    IM_COL32(255, 100, 255, 230), 2.5f);
        dl->AddText(ImVec2(ix + 24, ly + 4), IM_COL32(240, 180, 240, 255), "Negative");
    }

    // ── Labels axes (max / min explicites) ────────────────────────────
    char buf[48];
    // X axis : rayon max
    fmtLY(rMax, buf, sizeof(buf));
    { ImVec2 ts = ImGui::CalcTextSize(buf);
      dl->AddText(ImVec2(cp.x + plotW - ts.x, cp.y + plotH + 2), ca, buf); }
    dl->AddText(ImVec2(cp.x, cp.y + plotH + 2), ca, "0");
    // Y axis : top et bottom values (avec exposant pour log)
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

    // Radius slider centered top + Auto button next to it, Show Negative Mass on right
    const float sliderW = 220.0f, autoBtnW = 50.0f, gap = 8.0f;
    float totalW = sliderW + gap + autoBtnW;
    float availW = ImGui::GetContentRegionAvail().x;
    float padL   = (availW - totalW) * 0.5f;
    if (padL < 0.0f) padL = 0.0f;
    ImGui::Dummy(ImVec2(padL, 0.0f)); ImGui::SameLine();
    ImGui::SetNextItemWidth(sliderW);
    ImGui::SliderFloat("Radius (LY)", &m_phaseRMaxLY, 0.0f, 5.0e10f, "%.0f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::SameLine();
    if (ImGui::Button("Auto", ImVec2(autoBtnW, 0))) m_phaseRMaxLY = 0.0f;
    // Show Negative Mass on right-most position
    ImGui::SameLine(availW - 165.0f);
    ImGui::Checkbox("Show Negative Mass", &m_phaseShowNeg);

    float velConv = (m_lyPerUnit*9.461e12f)/(m_timeStep*3.156e13f);
    // Auto-fit OR user-defined max radius
    float rMax;
    if (m_phaseRMaxLY > 0.0f) {
        rMax = m_phaseRMaxLY;
    } else {
        float rMaxSim2 = 0.0f;
        for (size_t i = 0; i < m_positions.size(); i++) {
            float r2 = m_positions[i].x*m_positions[i].x
                     + m_positions[i].y*m_positions[i].y
                     + m_positions[i].z*m_positions[i].z;
            if (r2 > rMaxSim2) rMaxSim2 = r2;
        }
        rMax = sqrtf(rMaxSim2) * m_lyPerUnit;
        if (rMax < 1.0f) rMax = m_radius * m_lyPerUnit * 3.0f;
    }
    float vrAbsMax = 0;

    struct Pt { float r, vr; bool neg; };
    std::vector<Pt> pts;
    pts.reserve(m_numPosiStars + (m_phaseShowNeg ? m_numNegaStars : 0));
    size_t n = m_rotVelocities.size() < m_positions.size() ? m_rotVelocities.size() : m_positions.size();
    for (size_t i = 0; i < n; i++) {
        bool isNeg = m_positions[i].w < 0;
        if (isNeg && !m_phaseShowNeg) continue;
        if (m_positions[i].w == 0) continue;
        float x = m_positions[i].x, y = m_positions[i].y, z = m_positions[i].z;
        // 3D radius and 3D radial velocity (was 2D bug)
        float r = sqrtf(x*x + y*y + z*z); if (r < 1) continue;
        float vr = (m_rotVelocities[i].x*x + m_rotVelocities[i].y*y + m_rotVelocities[i].z*z) / r * velConv;
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
    fmtLY(rMax, buf, sizeof(buf));
    { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(cp.x+plotW-ts.x,cp.y+plotH+2),IM_COL32(160,160,160,255),buf); }
    { ImVec2 ts=ImGui::CalcTextSize("0"); dl->AddText(ImVec2(cp.x-ts.x-3,cp.y+0.5f*plotH-6),IM_COL32(100,100,80,200),"0"); }

    ImGui::Dummy(ImVec2(plotW+60,plotH+38)); ImGui::End();
}

// ─── Expansion diagnostic : a(t), ȧ=H·a vs âge cosmique ──────────────────────
// ȧ qui MONTE = accélération (ä>0) ; PLAT = coasting linéaire (Big Freeze, Petit) ;
// DESCEND = décélération. q = paramètre de décélération (q<0 = accélère).
void App::renderExpansion()
{
    if (!m_showExpansion) return;
    ImGui::SetNextWindowSize(ImVec2(560, 420), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(380, 60),  ImGuiCond_FirstUseEver);
    ImGui::Begin("Expansion", &m_showExpansion, ImGuiWindowFlags_NoCollapse);

    float a    = m_simulation ? m_simulation->getScaleFactor() : 1.0f;
    float H    = m_simulation ? m_simulation->getHubbleParam() : 0.0f;
    float adot = H * a;
    ImGui::Text("a = %.4f      H = %.5f      a-dot (H*a) = %.5f", a, H, adot);

    int cnt = (int)m_expTimes.size();

    // Verdict : pente de ȧ sur la DERNIÈRE portion en TEMPS (25% du span) -> ä, q.
    const char* verdict = "(running...)"; ImVec4 vcol(0.7f,0.7f,0.7f,1);
    if (cnt >= 4) {
        double tmn=1e30, tmx=-1e30;
        for (int i=0;i<cnt;i++){ double t=m_expTimes[i]; if(t<tmn)tmn=t; if(t>tmx)tmx=t; }
        double t0 = tmx - 0.25*(tmx - tmn);
        int iF=-1; for (int i=0;i<cnt;i++) if (m_expTimes[i] >= t0){ iF=i; break; }
        int iL=cnt-1;
        if (iF>=0 && iF!=iL) {
            double dt = m_expTimes[iL]-m_expTimes[iF];
            float adF=m_expAdot[iF], adL=m_expAdot[iL];
            if (dt > 1e-9) {
                float slope = (adL-adF)/(float)dt;                 // ä ≈ d(ȧ)/dt
                float rel   = (fabsf(adF)>1e-12f) ? (adL-adF)/fabsf(adF) : 0.0f;
                float q     = (fabsf(adot)>1e-12f) ? -slope*a/(adot*adot) : 0.0f;
                if      (rel >  0.02f) { verdict="accelerating  (a-ddot > 0)";        vcol=ImVec4(0.3f,1.0f,0.4f,1); }
                else if (rel < -0.02f) { verdict="decelerating";                       vcol=ImVec4(1.0f,0.5f,0.3f,1); }
                else                   { verdict="coasting ~linear  (Big Freeze)";     vcol=ImVec4(0.4f,0.8f,1.0f,1); }
                ImGui::Text("q = %.3f", q); ImGui::SameLine();
            }
        }
    }
    ImGui::TextColored(vcol, "%s", verdict);
    ImGui::TextDisabled("a-dot rising = acceleration | flat = coasting | falling = deceleration");

    if (cnt < 2) { ImGui::End(); return; }

    ImVec2 cp = ImGui::GetCursorScreenPos();
    ImVec2 cs = ImGui::GetContentRegionAvail();
    if (cs.x<120) cs.x=120; if (cs.y<100) cs.y=100;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(cp, ImVec2(cp.x+cs.x,cp.y+cs.y), IM_COL32(8,16,48,255));
    dl->AddRect      (cp, ImVec2(cp.x+cs.x,cp.y+cs.y), IM_COL32(100,100,100,200));
    ImGui::SetWindowFontScale(1.3f);   // labels du graphe un peu agrandis (~21 px sur base 16)
    const float mL=72,mR=10,mT=10,mB=28;
    float pw=cs.x-mL-mR, ph=cs.y-mT-mB, px=cp.x+mL, py=cp.y+mT;

    double tMin=1e30,tMax=-1e30; float yMin=1e30f,yMax=-1e30f;
    for (int i=0;i<cnt;i++){ double t=m_expTimes[i]; float y=m_expAdot[i];
        if(t<tMin)tMin=t; if(t>tMax)tMax=t; if(y<yMin)yMin=y; if(y>yMax)yMax=y; }
    if (tMax<=tMin) tMax=tMin+1.0;
    if (yMin>0.0f) yMin=0.0f;                 // inclure 0 comme référence
    if (yMax<=yMin) yMax=yMin+1.0f;

    auto SX=[&](double t){ return px+(float)((t-tMin)/(tMax-tMin))*pw; };
    auto SY=[&](float  y){ return py+ph-(y-yMin)/(yMax-yMin)*ph; };

    ImU32 cg=IM_COL32(50,50,50,200), ca=IM_COL32(180,180,180,255);
    for (int i=1;i<=4;i++){ float xg=px+pw*i/4.0f, yg=py+ph*i/4.0f;
        dl->AddLine(ImVec2(xg,py),ImVec2(xg,py+ph),cg); dl->AddLine(ImVec2(px,yg),ImVec2(px+pw,yg),cg); }
    dl->AddLine(ImVec2(px,py+ph),ImVec2(px+pw,py+ph),ca);
    dl->AddLine(ImVec2(px,py),   ImVec2(px,py+ph),    ca);

    bool first=true; ImVec2 prev;
    for (int i=0;i<cnt;i++){ ImVec2 cur(SX(m_expTimes[i]), SY(m_expAdot[i]));
        if(!first) dl->AddLine(prev,cur,IM_COL32(120,220,255,235),2.0f); prev=cur; first=false; }

    // ── Repères : creux de ȧ (coasting) + reprise d'accélération ──────────────
    float adotMin=1e30f; int mMin=-1;
    for (int i=0;i<cnt;i++){ if(m_expAdot[i]<adotMin){adotMin=m_expAdot[i]; mMin=i;} }
    float thr = adotMin * 1.05f;
    // Repères dès qu'un creux INTÉRIEUR existe (au moins 2 points après -> ȧ remonte
    // déjà par définition). Pas de seuil de 5% ici, sinon les repères apparaissent
    // bien après que le verdict soit passé en « accelerating ». Rien au tout début
    // (creux = point courant -> mMin >= cnt-2 -> pas de repère).
    bool transitionHappened = (mMin>=1 && adotMin>1e-9f && mMin<cnt-2);
    if (transitionHappened) {
        int ri=mMin; while (ri<cnt-1 && m_expAdot[ri+1]<=thr) ri++;   // bord droit de la bande
        const ImU32 colC = IM_COL32(120,200,255,235), colA = IM_COL32(120,255,140,235);
        // Points sur la courbe (sans ligne de marqueur).
        dl->AddCircleFilled(ImVec2(SX(m_expTimes[mMin]), SY(m_expAdot[mMin])), 4.0f, colC);
        if (ri>mMin) dl->AddCircleFilled(ImVec2(SX(m_expTimes[ri]), SY(m_expAdot[ri])), 4.0f, colA);
        // Labels en HAUT-GAUCHE, empilés sous « a-dot … ».
        float lineH = ImGui::GetTextLineHeight();
        float ly = py + 4 + lineH + 2;
        char b[44];
        snprintf(b,sizeof(b),"coasting %.1f Gyr", m_expTimes[mMin]);
        dl->AddText(ImVec2(px+6, ly), colC, b);
        if (ri>mMin) {
            ly += lineH + 1;
            snprintf(b,sizeof(b),"accel %.1f Gyr", m_expTimes[ri]);
            dl->AddText(ImVec2(px+6, ly), colA, b);
        }
    }

    // ── Domaine spéculatif : au-delà de l'âge de l'univers (~14.8 Gyr) tout est
    // extrapolation. Trait pointillé vertical + label vertical, pour rester honnête
    // vis-à-vis de l'utilisateur (qui peut être un scientifique).
    const double SPEC_AGE = 14.8;
    if (tMax >= SPEC_AGE && tMin <= SPEC_AGE) {
        float sx = px + (float)((SPEC_AGE - tMin)/(tMax - tMin)) * pw;
        ImU32 colSpec = IM_COL32(255, 170, 60, 210);
        for (float yy = py; yy < py+ph; yy += 10.0f) {           // pointillé
            float ye = (yy+5.0f < py+ph) ? yy+5.0f : py+ph;
            dl->AddLine(ImVec2(sx, yy), ImVec2(sx, ye), colSpec, 1.4f);
        }
        // Texte tourné -90° (lettres perpendiculaires, lecture bas->haut), centré
        // verticalement. Helper : dessine à l'horizontale puis pivote les vertices.
        float chh = ImGui::GetTextLineHeight();
        auto vtext = [&](float anchorX, const char* s){
            float tw = ImGui::CalcTextSize(s).x;
            ImVec2 tpos(anchorX, py + ph*0.5f + tw*0.5f);
            int v0 = dl->VtxBuffer.Size;
            dl->AddText(tpos, colSpec, s);
            int v1 = dl->VtxBuffer.Size;
            for (int vi=v0; vi<v1; vi++){
                ImDrawVert& vt = dl->VtxBuffer[vi];
                float dx = vt.pos.x - tpos.x, dy = vt.pos.y - tpos.y;   // rot -90° : (x,y)->(x+dy, y-dx)
                vt.pos.x = tpos.x + dy;
                vt.pos.y = tpos.y - dx;
            }
        };
        vtext(sx + 5.0f,       "Speculative domain");   // à droite du trait
        vtext(sx - 5.0f - chh, "14.8 Gyr");             // à gauche du trait, aligné/centré
    }

    char lab[48];
    snprintf(lab,sizeof(lab),"a-dot %.4f",yMax); dl->AddText(ImVec2(px+4,py+2),     IM_COL32(160,200,255,220),lab);
    snprintf(lab,sizeof(lab),"%.4f",yMin);       dl->AddText(ImVec2(px+4,py+ph-ImGui::GetTextLineHeight()-3), IM_COL32(160,200,255,160),lab);
    snprintf(lab,sizeof(lab),"%.1f Gyr",tMax);   { ImVec2 ts=ImGui::CalcTextSize(lab); dl->AddText(ImVec2(px+pw-ts.x,py+ph+6),IM_COL32(180,180,180,200),lab); }
    snprintf(lab,sizeof(lab),"%.1f",tMin);       dl->AddText(ImVec2(px,py+ph+6),    IM_COL32(180,180,180,200),lab);

    ImGui::SetWindowFontScale(1.0f);   // restaure pour les autres usages de la fenêtre
    ImGui::End();
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
    // Time-range slider (0 = all history, max 50 Gyr = 50000 MY)
    ImGui::SetNextItemWidth(220);
    ImGui::SliderFloat("Time window (MY)", &m_energyTimeRangeMY, 0.0f, 50000.0f, "%.0f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::SameLine();
    if (ImGui::Button("All##e")) m_energyTimeRangeMY = 0.0f;

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

    // First pass: find global tMax
    int cnt = m_energyCount;
    int head = m_energyHead;
    double tGlobalMax = -1e30;
    for (int i=0;i<cnt;i++) {
        int idx = (head - cnt + i + ENERGY_HISTORY) % ENERGY_HISTORY;
        double t = m_energyTimes[idx];
        if (t > tGlobalMax) tGlobalMax = t;
    }
    // Time-window cutoff: if range > 0, only show last X MY
    double tCutoff = (m_energyTimeRangeMY > 0.0f) ? (tGlobalMax - (double)m_energyTimeRangeMY) : -1e30;

    // Second pass: stats only within window
    float keMin=1e30f, keMax=-1e30f;
    double tMin=1e30,  tMax=-1e30;
    int visibleCount = 0;
    for (int i=0;i<cnt;i++) {
        int idx = (head - cnt + i + ENERGY_HISTORY) % ENERGY_HISTORY;
        double t = m_energyTimes[idx];
        if (t < tCutoff) continue;
        float ke = m_energyHistory[idx];
        if (ke<keMin) keMin=ke; if (ke>keMax) keMax=ke;
        if (t<tMin) tMin=t;   if (t>tMax) tMax=t;
        visibleCount++;
    }
    if (visibleCount < 2) {
        ImGui::TextDisabled("(too few points in selected time window)");
        ImGui::End();
        return;
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

    // Collect visible data into a flat array (within tMin..tMax)
    std::vector<float> visKE; visKE.reserve(visibleCount);
    std::vector<double> visT;  visT.reserve(visibleCount);
    for (int i=0;i<cnt;i++) {
        int idx = (head - cnt + i + ENERGY_HISTORY) % ENERGY_HISTORY;
        double t = m_energyTimes[idx];
        if (t < tCutoff) continue;
        visKE.push_back(m_energyHistory[idx]);
        visT.push_back(t);
    }
    int nV = (int)visKE.size();

    // Gaussian smoothing — kernel sigma = adaptive (5% of visible points, min 3, max 30)
    int sigBins = std::max(3, std::min(30, nV / 20));
    int halfK   = sigBins * 3;  // 99.7% Gaussian coverage
    std::vector<float> smoothed(nV, 0.0f);
    for (int i = 0; i < nV; i++) {
        double sumW = 0.0, sumWKE = 0.0;
        for (int k = -halfK; k <= halfK; k++) {
            int j = i + k;
            if (j < 0 || j >= nV) continue;
            double w = exp(-0.5 * (double)(k*k) / (double)(sigBins*sigBins));
            sumW   += w;
            sumWKE += w * visKE[j];
        }
        smoothed[i] = (sumW > 0.0) ? (float)(sumWKE / sumW) : visKE[i];
    }

    // Draw smoothed curve
    ImVec2 prev=ImVec2(0,0); bool first=true;
    for (int i = 0; i < nV; i++) {
        float sx = px + (float)(visT[i] - tMin) / (float)(tMax - tMin) * plotW;
        float sy = py + (1.0f - (smoothed[i] - keMin) / (keMax - keMin)) * plotH;
        ImVec2 cur(sx, sy);
        if (!first) dl->AddLine(prev, cur, IM_COL32(100,220,255,220), 1.8f);
        prev = cur; first = false;
    }

    char buf[48];
    double span = tMax-tMin;
    const char* unit = span<1000?"MY":"Gy";
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
    float velConv      = (m_lyPerUnit * 9.461e12f) / 3.156e13f;
    float G_eff        = m_gravConstant * 1.0e10f;
    float massPP       = (m_numPosiStars > 0) ? m_totalSolarMass / (float)m_numPosiStars : 1.0f;

    float sumVr[BINS]={}, sumVr2[BINS]={};
    float sumVt[BINS]={}, sumVt2[BINS]={};
    float sumMassAnn[BINS]={};
    int   cnt[BINS]={};

    size_t n = m_rotVelocities.size() < m_positions.size()
               ? m_rotVelocities.size() : m_positions.size();
    for (size_t i = 0; i < n; i++) {
        if (m_positions[i].w <= 0.0f) continue;
        float x = m_positions[i].x, y = m_positions[i].y;
        float r = sqrtf(x*x + y*y);
        if (r < 1.0f) continue;
        float rLy = r * m_lyPerUnit;
        if (rLy > rMax) continue;
        int b = (int)(rLy / rMax * BINS);
        if (b >= BINS) b = BINS - 1;
        float vx = m_rotVelocities[i].x, vy = m_rotVelocities[i].y;
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

        ImVec2 prevR = ImVec2(0,0), prevT = ImVec2(0,0);
        bool firstR = true, firstT = true;
        for (int b = 0; b < BINS; b++) {
            if (cnt[b] < 3) { firstR = true; firstT = true; continue; }
            float rLy = (b + 0.5f) / BINS * rMax;
            float sx  = px + rLy / rMax * pw;
            float syR = py + (1.0f - sigR[b]/sigMax) * ph;
            float syT = py + (1.0f - sigT[b]/sigMax) * ph;
            if (!firstR) dl->AddLine(prevR, ImVec2(sx,syR), IM_COL32(255,255,255,200), 1.5f);
            if (!firstT) dl->AddLine(prevT, ImVec2(sx,syT), IM_COL32(80,220,255,200),  1.5f);
            prevR = ImVec2(sx,syR); prevT = ImVec2(sx,syT);
            firstR = false; firstT = false;
        }
        char buf[32];
        snprintf(buf,sizeof(buf),"%.0f km/s",sigMax);
        { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px-ts.x-2,py), ca, buf); }
        { ImVec2 ts=ImGui::CalcTextSize("0"); dl->AddText(ImVec2(px-ts.x-2,py+ph-12), ca, "0"); }
        { ImVec2 ts=ImGui::CalcTextSize("Radius (LY)"); dl->AddText(ImVec2(px+(pw-ts.x)*0.5f,py+ph+14), ca, "Radius (LY)"); }
        fmtLY(rMax, buf, sizeof(buf));
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
            ImVec2 prev={}; bool first=true;
            for (int b = 0; b < BINS; b++) {
                if (cnt[b] < 3 || Q[b] <= 0.0f) { first=true; continue; }
                float rLy = (b + 0.5f) / BINS * rMax;
                float Qcl = Q[b] < Qmax ? Q[b] : Qmax;
                float sx  = px + rLy / rMax * pw;
                float sy  = py + (1.0f - Qcl/Qmax) * ph;
                if (!first) dl->AddLine(prev, ImVec2(sx,sy), IM_COL32(255,220,60,220), 1.5f);
                prev=ImVec2(sx,sy); first=false;
            }
        } else {
            dl->AddText(ImVec2(px+4,py+4), IM_COL32(150,150,150,255),
                        "Q unavailable — enable Auto-Refresh");
        }
        char buf[32];
        snprintf(buf,sizeof(buf),"%.1f",Qmax);
        { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px-ts.x-2,py), ca, buf); }
        { ImVec2 ts=ImGui::CalcTextSize("0"); dl->AddText(ImVec2(px-ts.x-2,py+ph-12), ca, "0"); }
        { ImVec2 ts=ImGui::CalcTextSize("Radius (LY)"); dl->AddText(ImVec2(px+(pw-ts.x)*0.5f,py+ph+14), ca, "Radius (LY)"); }
        fmtLY(rMax, buf, sizeof(buf));
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

    static const float niceVals[] = {
        10, 20, 50, 100, 200, 500,
        1e3f, 2e3f, 5e3f, 10e3f, 20e3f, 50e3f,          // kLY
        100e3f, 200e3f, 500e3f,                            // 100-500 kLY
        1e6f, 2e6f, 5e6f, 10e6f, 20e6f, 50e6f,           // Mly
        100e6f, 200e6f, 500e6f,                            // 100-500 Mly
        1e9f, 2e9f, 5e9f, 10e9f                           // Gly
    };
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
    // Variables pour label (doivent être définies dans les deux branches)
    float xStart = 0.0f, hStart = 0.0f, hMax = 0.0f;

    if (m_camera->isOrbitMode()) {
        // ── Grille 3D type Blender ────────────────────────────────────────
        // Lignes dans les 3 directions : toujours visible sous tous les angles
        float step = gridUnits;
        if (step < 1.0f) step = 1.0f;
        const int RANGE = 50;  // -50 à +50 lignes = 101 lignes
        float limit = RANGE * step;

        // ── Grille dans le plan de la galaxie uniquement (Z=0) ────────────
        // Helper : dessine une ligne 3D avec clipping au near plane
        auto line3dClip = [&](float x1, float y1, float z1, float x2, float y2, float z2, ImU32 col) {
            // Fondu par distance : les lignes loin (qui convergent à l'horizon en vue
            // rasante) s'estompent -> plus d'empilement en bande lumineuse.
            {
                float mx = 0.5f*(x1+x2), my = 0.5f*(y1+y2), mz = 0.5f*(z1+z2);
                float ddx = mx-cam_x, ddy = my-cam_y, ddz = mz-cam_z;
                float dist = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
                float f0 = camDist*1.5f, f1 = camDist*4.0f;
                float fade = (f1>f0) ? 1.0f-(dist-f0)/(f1-f0) : 1.0f;
                fade = fade<0.0f?0.0f:(fade>1.0f?1.0f:fade);
                if (fade <= 0.02f) return;
                ImU32 al = (ImU32)((float)((col>>24)&0xFFu) * fade);
                col = (col & 0x00FFFFFFu) | (al << 24);
            }
            float cx1 = vp[0]*x1 + vp[4]*y1 + vp[8]*z1 + vp[12];
            float cy1 = vp[1]*x1 + vp[5]*y1 + vp[9]*z1 + vp[13];
            float cw1 = vp[3]*x1 + vp[7]*y1 + vp[11]*z1 + vp[15];

            float cx2 = vp[0]*x2 + vp[4]*y2 + vp[8]*z2 + vp[12];
            float cy2 = vp[1]*x2 + vp[5]*y2 + vp[9]*z2 + vp[13];
            float cw2 = vp[3]*x2 + vp[7]*y2 + vp[11]*z2 + vp[15];

            const float NEAR = 1e-3f;
            // Si les deux points sont derrière le near plane, on skip
            if (cw1 < NEAR && cw2 < NEAR) return;

            // Si un point est derrière, on clip au near plane
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

        // Grille 2D dans le plan Z=0 = plan de la galaxie
        // Même logique que le mode 2D : couvre ±worldHalf * ext
        const float ext = 4.0f;
        float xMin = -worldHalfW * ext;
        float xMax =  worldHalfW * ext;
        float yMin = -worldHalfH * ext;
        float yMax =  worldHalfH * ext;

        float xStart = floorf(xMin / gridUnits) * gridUnits;
        float yStart = floorf(yMin / gridUnits) * gridUnits;

        // Lignes verticales : X fixé, Y varie
        for (float x = xStart; x <= xMax + gridUnits*0.5f; x += gridUnits) {
            line3dClip(x, yMin, 0, x, yMax, 0, colGrid);
        }
        // Lignes horizontales : Y fixé, X varie
        for (float y = yStart; y <= yMax + gridUnits*0.5f; y += gridUnits) {
            line3dClip(xMin, y, 0, xMax, y, 0, colGrid);
        }

        // Axes dans le plan Z=0 (courts, ~10% de la grille)
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

    // Label fixe en haut à gauche de la fenêtre de rendu
    ImVec2 labelAnchor(vpX + 10, gMVY + 10);

    char label[64];
    if      (gridLy >= 1e9f)  snprintf(label, sizeof(label), "%.1f Gly", gridLy / 1e9f);
    else if (gridLy >= 1e6f)  snprintf(label, sizeof(label), "%.1f Mly", gridLy / 1e6f);
    else if (gridLy >= 1e3f)  snprintf(label, sizeof(label), "%.0f kly", gridLy / 1e3f);
    else                       snprintf(label, sizeof(label), "%.0f LY",  gridLy);

    ImVec2 textSize = ImGui::CalcTextSize(label);
    dl->AddRectFilled(
        ImVec2(labelAnchor.x - 2, labelAnchor.y - 2),
        ImVec2(labelAnchor.x + textSize.x + 4, labelAnchor.y + textSize.y + 2),
        IM_COL32(0, 0, 0, 140));
    dl->AddText(labelAnchor, IM_COL32(220, 220, 100, 230), label);

    dl->PopClipRect();
}

// ─── Galaxy Markers ──────────────────────────────────────────────────────────
void App::renderGalaxyMarkers()
{
    if (!m_showGalaxyMarkers || m_galaxyCentroids.empty() || !m_simRunning) return;

    ImGuiViewport* vp  = ImGui::GetMainViewport();
    const float panelW = 340.0f;
    const float vpX    = vp->Pos.x + panelW;
    const float vpW    = (float)m_width - panelW;
    const float vpH    = (float)m_height;
    const float mvY    = vp->Pos.y;

    float raw[16];
    m_camera->getViewProjMatrix(raw);
    glm::mat4 vpm = glm::make_mat4(raw);

    // Background draw list : au-dessus de la scène 3D mais EN-DESSOUS des fenêtres ImGui
    ImDrawList* dl = ImGui::GetBackgroundDrawList(vp);
    const ImU32 colCircle = IM_COL32(40, 255, 100, 220);
    const ImU32 colText   = IM_COL32(40, 255, 100, 255);
    const ImU32 colBg     = IM_COL32(0,  0,   0,   160);

    int idx = 1;
    for (const auto& c : m_galaxyCentroids)
    {
        // c.w holds the global index of the core-representative particle: draw at its
        // CURRENT position so the circle tracks the blob live (no async-snapshot lag).
        // Fall back to the stored centroid (c.xyz) if the index is stale/invalid.
        int rep = (int)c.w;
        float gxp = c.x, gyp = c.y, gzp = c.z;
        if (rep >= 0 && rep < (int)m_positions.size()) {
            gxp = m_positions[rep].x; gyp = m_positions[rep].y; gzp = m_positions[rep].z;
        }
        glm::vec4 clip = vpm * glm::vec4(gxp, gyp, gzp, 1.0f);
        if (clip.w <= 0.0f) { idx++; continue; }

        float ndcX =  clip.x / clip.w;
        float ndcY =  clip.y / clip.w;
        float ndcZ =  clip.z / clip.w;
        if (ndcZ < -1.f || ndcZ > 1.f)  { idx++; continue; }
        if (ndcX < -1.3f || ndcX > 1.3f) { idx++; continue; }
        if (ndcY < -1.3f || ndcY > 1.3f) { idx++; continue; }

        float sx = vpX + (ndcX + 1.0f) * 0.5f * vpW;
        float sy = mvY + (1.0f - ndcY) * 0.5f * vpH;

        dl->AddCircle(ImVec2(sx, sy), 10.0f, colCircle, 16, 1.5f);

        char lbl[12];
        snprintf(lbl, sizeof(lbl), "G%d", idx);
        ImVec2 ts = ImGui::CalcTextSize(lbl);
        float tx = sx + 12.0f, ty = sy - ts.y * 0.5f;
        dl->AddRectFilled(ImVec2(tx-2, ty-1), ImVec2(tx+ts.x+2, ty+ts.y+1), colBg);
        dl->AddText(ImVec2(tx, ty), colText, lbl);

        idx++;
    }
}

// ─── Galaxy Formation History (vs cosmic age) + JWST band ───────────────────
void App::renderGalaxyHistory()
{
    if (!m_showGalaxyHistory) return;
    ImGui::SetNextWindowSize(ImVec2(660, 480), ImGuiCond_FirstUseEver);
    // Resizable like the other windows ; NO scrollbar (we reserve a text area).
    if (!ImGui::Begin("Galaxy Formation History", &m_showGalaxyHistory,
                      ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                      ImGuiWindowFlags_NoScrollWithMouse)) { ImGui::End(); return; }
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Reserve a 2-line text strip at the bottom (verdict + Clear). The plot fills the rest.
    const float textH = ImGui::GetTextLineHeightWithSpacing() * 2.2f + 6.0f;
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImVec2 avail  = ImGui::GetContentRegionAvail();
    float fullW = avail.x, fullH = std::max(140.0f, avail.y - textH);

    dl->AddRectFilled(origin, ImVec2(origin.x+fullW, origin.y+fullH), IM_COL32(8,16,48,255));
    dl->AddRect      (origin, ImVec2(origin.x+fullW, origin.y+fullH), IM_COL32(80,80,80,255));

    // Plot area (with axis margins).
    const char* xlabel = "Cosmic age (Gyr)";  const char* ylabel = "Galaxies";
    ImVec2 yts = ImGui::CalcTextSize(ylabel);
    float mL = std::max(68.0f, yts.x + 8.0f), mR = 12.0f, mT = 12.0f, mB = 30.0f;
    ImVec2 cp(origin.x + mL, origin.y + mT);
    float plotW = fullW - mL - mR, plotH = fullH - mT - mB;
    ImU32 ca = IM_COL32(160,160,160,255), cg = IM_COL32(40,40,40,255);
    dl->AddLine(ImVec2(cp.x, cp.y+plotH), ImVec2(cp.x+plotW, cp.y+plotH), ca);
    dl->AddLine(ImVec2(cp.x, cp.y),       ImVec2(cp.x, cp.y+plotH),       ca);
    for (int i=1;i<=4;i++){ float xg=cp.x+plotW*i/4.f, yg=cp.y+plotH*i/4.f;
        dl->AddLine(ImVec2(xg,cp.y),ImVec2(xg,cp.y+plotH),cg);
        dl->AddLine(ImVec2(cp.x,yg),ImVec2(cp.x+plotW,yg),cg); }
    { ImVec2 ts=ImGui::CalcTextSize(xlabel); dl->AddText(ImVec2(cp.x+(plotW-ts.x)*0.5f, cp.y+plotH+12), ca, xlabel); }
    dl->AddText(ImVec2(cp.x-yts.x-4, cp.y+plotH*0.5f-yts.y*0.5f), ca, ylabel);

    float firstAge = -1.0f;
    const bool cosmo = m_useCosmology && m_omegaL > 1e-6f && m_omegaM > 1e-6f;

    if (m_galHistory.size() < 2) {
        dl->AddText(ImVec2(cp.x+6, cp.y+6), IM_COL32(150,150,150,255),
                    "Run the simulation to record the history.");
    } else {
        float ageMax = 0.1f; int galMax = 1;
        for (const auto& p : m_galHistory) { ageMax = std::max(ageMax, p.ageGyr); galMax = std::max(galMax, p.galaxies); }
        float gMaxF = (float)galMax * 1.10f;
        auto X = [&](float age){ return cp.x + (age / ageMax) * plotW; };
        auto Y = [&](float g)  { return cp.y + (1.0f - g / gMaxF) * plotH; };

        // JWST observed epoch : z ~ 10-14  ->  cosmic age ~ 0.30-0.49 Gyr.
        if (cosmo) {
            const float jwstLo = 0.30f, jwstHi = 0.49f;
            float x0 = X(std::min(jwstLo, ageMax)), x1 = X(std::min(jwstHi, ageMax));
            dl->AddRectFilled(ImVec2(x0, cp.y), ImVec2(x1, cp.y+plotH), IM_COL32(255,180,40,45));
            dl->AddLine(ImVec2(x0, cp.y), ImVec2(x0, cp.y+plotH), IM_COL32(255,200,90,160), 1.0f);
            dl->AddLine(ImVec2(x1, cp.y), ImVec2(x1, cp.y+plotH), IM_COL32(255,200,90,160), 1.0f);
            dl->AddText(ImVec2(x0+3, cp.y+3), IM_COL32(255,210,110,220), "JWST z~10-14");
        }

        // SMOOTH curve : moving average over the count series removes the integer
        // staircase -> clean line. AddPolyline is anti-aliased.
        int n = (int)m_galHistory.size();
        int win = std::min(50, std::max(3, n / 60));   // adaptive half-window
        std::vector<ImVec2> poly; poly.reserve(n);
        for (int i = 0; i < n; i++) {
            int a = std::max(0, i-win), b = std::min(n-1, i+win);
            float sum = 0.0f; for (int k=a;k<=b;k++) sum += (float)m_galHistory[k].galaxies;
            poly.push_back(ImVec2(X(m_galHistory[i].ageGyr), Y(sum / (float)(b-a+1))));
        }
        dl->AddPolyline(poly.data(), (int)poly.size(), IM_COL32(80,255,140,235), 0, 2.5f);

        for (const auto& p : m_galHistory) if (p.galaxies > 0) { firstAge = p.ageGyr; break; }
        if (firstAge >= 0.0f)
            dl->AddLine(ImVec2(X(firstAge), cp.y), ImVec2(X(firstAge), cp.y+plotH), IM_COL32(120,220,255,160), 1.0f);

        char buf[40];
        snprintf(buf, sizeof(buf), "%d", galMax);
        dl->AddText(ImVec2(cp.x+3, cp.y+1), IM_COL32(160,160,160,255), buf);
        snprintf(buf, sizeof(buf), "%.2f Gyr", ageMax);
        { ImVec2 ts=ImGui::CalcTextSize(buf); dl->AddText(ImVec2(cp.x+plotW-ts.x-2, cp.y+plotH-13), IM_COL32(160,160,160,255), buf); }
    }

    // Advance past the plot area, then draw the reserved text strip.
    ImGui::Dummy(ImVec2(fullW, fullH));
    if (firstAge >= 0.0f && cosmo) {
        if (firstAge < 0.49f)
            ImGui::TextColored(ImVec4(0.5f,1.0f,0.6f,1.0f),
                "1st galaxy at %.3f Gyr -> within/earlier than the JWST window (z~10-14).", firstAge);
        else
            ImGui::TextColored(ImVec4(0.9f,0.8f,0.4f,1.0f),
                "1st galaxy at %.3f Gyr (after the JWST window).", firstAge);
    } else if (firstAge >= 0.0f) {
        ImGui::TextDisabled("1st galaxy at %.3f (sim time, cosmology off).", firstAge);
    } else {
        ImGui::TextDisabled("No galaxy yet.");
    }
    if (ImGui::SmallButton("Clear##galhist")) m_galHistory.clear();

    ImGui::End();
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
    { char dBuf[32]; fmtLY(curDist, dBuf, sizeof(dBuf));
      ImGui::Text("G+(%.0f,%.0f)  G-(%.0f,%.0f)  |d|=%s",
        m_baryPosX * m_lyPerUnit, m_baryPosY * m_lyPerUnit,
        m_baryNegX * m_lyPerUnit, m_baryNegY * m_lyPerUnit, dBuf); }
    // Time-range slider (0 = all history, max 50 Gyr)
    ImGui::SetNextItemWidth(220);
    ImGui::SliderFloat("Time window (MY)", &m_baryTimeRangeMY, 0.0f, 50000.0f, "%.0f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::SameLine();
    if (ImGui::Button("All##b")) m_baryTimeRangeMY = 0.0f;
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

    // Global tMax first, then filtered stats
    float tGlobalMax = 0.0f;
    for (int i = 0; i < m_baryCount; i++) {
        int idx = (m_baryHead - m_baryCount + i + BARY_HISTORY) % BARY_HISTORY;
        if (m_baryTimes[idx] > tGlobalMax) tGlobalMax = m_baryTimes[idx];
    }
    float tCutoff = (m_baryTimeRangeMY > 0.0f) ? (tGlobalMax - m_baryTimeRangeMY) : -1e30f;

    float tMin = 1e30f, tMax = 0.0f, dMax = 1e-6f;
    int visibleCount = 0;
    for (int i = 0; i < m_baryCount; i++) {
        int idx = (m_baryHead - m_baryCount + i + BARY_HISTORY) % BARY_HISTORY;
        float t = m_baryTimes[idx];
        if (t < tCutoff) continue;
        if (t < tMin) tMin = t;
        if (t > tMax) tMax = t;
        if (m_baryDist[idx] > dMax) dMax = m_baryDist[idx];
        visibleCount++;
    }
    if (visibleCount < 2) {
        ImGui::TextDisabled("(too few points in selected time window)");
        ImGui::End();
        return;
    }
    if (tMax <= tMin) tMax = tMin + 1.0f;

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
    fmtLY(dMax, buf, sizeof(buf));
    { ImVec2 ts = ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px - ts.x - 2, py), colAxis, buf); }
    { ImVec2 ts = ImGui::CalcTextSize("0"); dl->AddText(ImVec2(px - ts.x - 2, py + plotH - 12), colAxis, "0"); }
    snprintf(buf, sizeof(buf), "%.0f MY", tMax);
    { ImVec2 ts = ImGui::CalcTextSize(buf); dl->AddText(ImVec2(px + plotW - ts.x, py + plotH + 2), colAxis, buf); }
    dl->AddText(ImVec2(px, py + plotH + 2), colAxis, "0");
    { ImVec2 ts = ImGui::CalcTextSize("|G+ - G-|"); dl->AddText(ImVec2(px + (plotW - ts.x) * 0.5f, py + plotH + 16), colAxis, "|G+ - G-|"); }

    // Gather visible data, then Gaussian smooth
    std::vector<float> visD; visD.reserve(visibleCount);
    std::vector<float> visT; visT.reserve(visibleCount);
    for (int i = 0; i < m_baryCount; i++) {
        int idx = (m_baryHead - m_baryCount + i + BARY_HISTORY) % BARY_HISTORY;
        float t = m_baryTimes[idx];
        if (t < tCutoff) continue;
        visD.push_back(m_baryDist[idx]);
        visT.push_back(t);
    }
    int nV = (int)visD.size();
    int sigBins = std::max(3, std::min(30, nV / 20));
    int halfK   = sigBins * 3;
    std::vector<float> smoothed(nV, 0.0f);
    for (int i = 0; i < nV; i++) {
        double sumW = 0.0, sumWD = 0.0;
        for (int k = -halfK; k <= halfK; k++) {
            int j = i + k;
            if (j < 0 || j >= nV) continue;
            double w = exp(-0.5 * (double)(k*k) / (double)(sigBins*sigBins));
            sumW  += w;
            sumWD += w * visD[j];
        }
        smoothed[i] = (sumW > 0.0) ? (float)(sumWD / sumW) : visD[i];
    }
    ImVec2 prev = ImVec2(0,0);
    bool first = true;
    for (int i = 0; i < nV; i++) {
        float sx = px + (visT[i] - tMin) / (tMax - tMin) * plotW;
        float sy = py + (1.0f - smoothed[i] / dMax) * plotH;
        ImVec2 cur(sx, sy);
        if (!first) dl->AddLine(prev, cur, IM_COL32(255, 200, 50, 220), 2.0f);
        prev = cur;
        first = false;
    }

    ImGui::Dummy(canvasSize);
    ImGui::End();
}

// ─── Power Spectrum P(k) ─────────────────────────────────────────────────────
// Standard cosmological diagnostic. Computes the 3D density field via CIC
// deposit on a small grid, applies a 3D FFT, then averages |rho_hat(k)|² over
// spherical shells in k-space. Plotted log-log.
namespace pkfft {
    struct Cplx { float re, im; };
    static void fft1d(Cplx* a, int n, bool inv) {
        // Bit reversal
        int j = 0;
        for (int i = 1; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { Cplx t = a[i]; a[i] = a[j]; a[j] = t; }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double angle = (inv ? 2.0 : -2.0) * 3.14159265358979323846 / len;
            float wr = (float)cos(angle), wi = (float)sin(angle);
            for (int i = 0; i < n; i += len) {
                float cr = 1.0f, ci = 0.0f;
                for (int k = 0; k < len/2; k++) {
                    Cplx u = a[i + k];
                    Cplx v = { cr * a[i+k+len/2].re - ci * a[i+k+len/2].im,
                               cr * a[i+k+len/2].im + ci * a[i+k+len/2].re };
                    a[i+k]         = { u.re + v.re, u.im + v.im };
                    a[i+k+len/2]   = { u.re - v.re, u.im - v.im };
                    float ncr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = ncr;
                }
            }
        }
    }
    static void fft3d(Cplx* cube, int N, bool inv) {
        std::vector<Cplx> line(N);
        // along X
        for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) line[x] = cube[(z*N + y)*N + x];
            fft1d(line.data(), N, inv);
            for (int x = 0; x < N; x++) cube[(z*N + y)*N + x] = line[x];
        }
        // along Y
        for (int z = 0; z < N; z++)
        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) line[y] = cube[(z*N + y)*N + x];
            fft1d(line.data(), N, inv);
            for (int y = 0; y < N; y++) cube[(z*N + y)*N + x] = line[y];
        }
        // along Z
        for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            for (int z = 0; z < N; z++) line[z] = cube[(z*N + y)*N + x];
            fft1d(line.data(), N, inv);
            for (int z = 0; z < N; z++) cube[(z*N + y)*N + x] = line[z];
        }
    }
}

void App::renderPowerSpectrum()
{
    if (!m_showPowerSpectrum) return;
    ImGui::SetNextWindowSize(ImVec2(640, 480), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(360, 30),   ImGuiCond_FirstUseEver);
    // No NoBringToFrontOnFocus flag : window can be dragged on top of the Controls panel
    ImGui::Begin("Power Spectrum P(k)", &m_showPowerSpectrum, ImGuiWindowFlags_NoCollapse);

    if (m_positions.empty()) {
        ImGui::TextColored(ImVec4(1.0f,0.6f,0.3f,1.0f),
                           "No particles. Click 'Apply & Generate' first.");
        ImGui::End(); return;
    }

    static int  NGRID    = 32;
    static int  speciesIdx = 0;  // 0=all signed, 1=positives only, 2=negatives only
    static bool autoRefresh = true;
    if (ImGui::Button("Refresh") && m_simulation) m_simulation->readPositions(m_positions);
    ImGui::SameLine();
    const char* gridLabels[] = {"32^3","64^3","128^3"};
    int gIdx = NGRID == 32 ? 0 : (NGRID == 64 ? 1 : 2);
    ImGui::SetNextItemWidth(80);
    if (ImGui::Combo("Grid##pk", &gIdx, gridLabels, 3))
        NGRID = (gIdx == 0) ? 32 : (gIdx == 1) ? 64 : 128;
    ImGui::SameLine();
    const char* spLabels[] = {"all (signed)","positive","negative"};
    ImGui::SetNextItemWidth(120);
    ImGui::Combo("Species##pk", &speciesIdx, spLabels, 3);
    ImGui::SameLine();
    ImGui::Checkbox("Auto", &autoRefresh);
    ImGui::Separator();

    // Bounding box (3D)
    float bbmin[3] = {  1e30f,  1e30f,  1e30f };
    float bbmax[3] = { -1e30f, -1e30f, -1e30f };
    for (const auto& p : m_positions) {
        if (p.x < bbmin[0]) bbmin[0] = p.x; if (p.x > bbmax[0]) bbmax[0] = p.x;
        if (p.y < bbmin[1]) bbmin[1] = p.y; if (p.y > bbmax[1]) bbmax[1] = p.y;
        if (p.z < bbmin[2]) bbmin[2] = p.z; if (p.z > bbmax[2]) bbmax[2] = p.z;
    }
    float boxL = fmaxf(fmaxf(bbmax[0]-bbmin[0], bbmax[1]-bbmin[1]), bbmax[2]-bbmin[2]);
    if (boxL < 1.0f) boxL = m_radius * 2.0f;
    boxL *= 1.05f;  // small margin
    float cx = 0.5f*(bbmin[0]+bbmax[0]);
    float cy = 0.5f*(bbmin[1]+bbmax[1]);
    float cz = 0.5f*(bbmin[2]+bbmax[2]);

    // CIC deposit (signed mass according to species filter)
    int N = NGRID;
    std::vector<pkfft::Cplx> grid((size_t)N * N * N, {0.0f, 0.0f});
    float h = boxL / (float)N;
    float invH = 1.0f / h;
    for (const auto& p : m_positions) {
        float mass = p.w;
        if (speciesIdx == 1 && mass <= 0.0f) continue;
        if (speciesIdx == 2 && mass >= 0.0f) continue;
        if (speciesIdx == 0) {} else mass = fabsf(mass);
        float gx = (p.x - cx + 0.5f*boxL) * invH;
        float gy = (p.y - cy + 0.5f*boxL) * invH;
        float gz = (p.z - cz + 0.5f*boxL) * invH;
        if (gx < 0 || gx >= N-1 || gy < 0 || gy >= N-1 || gz < 0 || gz >= N-1) continue;
        int ix=(int)gx, iy=(int)gy, iz=(int)gz;
        float dx=gx-ix, dy=gy-iy, dz=gz-iz;
        for (int a=0;a<2;a++) {
            float wx = (a==0) ? 1.0f-dx : dx;
            for (int b=0;b<2;b++) {
                float wy = (b==0) ? 1.0f-dy : dy;
                for (int c=0;c<2;c++) {
                    float wz = (c==0) ? 1.0f-dz : dz;
                    grid[((iz+c)*N + (iy+b))*N + (ix+a)].re += mass * wx*wy*wz;
                }
            }
        }
    }
    // Subtract mean (so we measure δ = ρ/ρ̄ - 1 essentially)
    double meanRho = 0.0;
    for (size_t i = 0; i < grid.size(); i++) meanRho += grid[i].re;
    meanRho /= (double)grid.size();
    if (meanRho != 0.0)
        for (size_t i = 0; i < grid.size(); i++)
            grid[i].re = (float)(grid[i].re / meanRho - 1.0);

    pkfft::fft3d(grid.data(), N, false);

    // Bin |rho_hat|² by |k|
    float dk = 2.0f * 3.14159265358979f / boxL;     // fundamental mode (sim units)
    int nBins = N / 2;
    std::vector<double> Psum(nBins, 0.0);
    std::vector<int>    Pcnt(nBins, 0);
    for (int kz = 0; kz < N; kz++) {
        int kzi = (kz > N/2) ? kz - N : kz;
        for (int ky = 0; ky < N; ky++) {
            int kyi = (ky > N/2) ? ky - N : ky;
            for (int kx = 0; kx < N; kx++) {
                int kxi = (kx > N/2) ? kx - N : kx;
                float km = sqrtf((float)(kxi*kxi + kyi*kyi + kzi*kzi));
                int b = (int)km;
                if (b < 0 || b >= nBins) continue;
                int idx = (kz*N + ky)*N + kx;
                double p = (double)grid[idx].re * grid[idx].re
                         + (double)grid[idx].im * grid[idx].im;
                Psum[b] += p;
                Pcnt[b]++;
            }
        }
    }
    std::vector<double> Pk(nBins, 0.0);
    std::vector<double> kArr(nBins, 0.0);
    int firstValid = -1, lastValid = -1;
    // Skip b=0 : DC mode has k=0 → log10(0) = -inf, breaks the log axis range.
    // Physically, P(k=0) is not measurable from a finite volume.
    for (int b = 1; b < nBins; b++) {
        if (Pcnt[b] > 0 && Psum[b] > 0.0) {
            Pk[b]   = Psum[b] / Pcnt[b];
            kArr[b] = (double)b * dk * m_lyPerUnit;  // 1/LY units
            if (firstValid < 0) firstValid = b;
            lastValid = b;
        }
    }
    if (firstValid < 0 || lastValid <= firstValid) {
        ImGui::TextDisabled("No power spectrum data (insufficient particles or all bins empty).");
        ImGui::End(); return;
    }

    // Plot log10(P) vs log10(k)
    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImVec2 sz        = ImGui::GetContentRegionAvail();
    if (sz.x < 100) sz.x = 100;
    if (sz.y < 100) sz.y = 100;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(canvasPos, ImVec2(canvasPos.x+sz.x, canvasPos.y+sz.y), IM_COL32(8,16,48,255));
    dl->AddRect    (canvasPos, ImVec2(canvasPos.x+sz.x, canvasPos.y+sz.y), IM_COL32(80,80,80,255));
    const float mL=80,mR=14,mT=14,mB=34;
    float pw = sz.x - mL - mR, ph = sz.y - mB - mT;
    ImVec2 cp(canvasPos.x + mL, canvasPos.y + mT);
    ImU32 ca = IM_COL32(160,160,160,255), cg = IM_COL32(40,40,40,255);
    dl->AddLine(ImVec2(cp.x, cp.y+ph), ImVec2(cp.x+pw, cp.y+ph), ca);
    dl->AddLine(ImVec2(cp.x, cp.y),    ImVec2(cp.x, cp.y+ph), ca);
    for (int i=1;i<=4;i++) {
        float xg = cp.x + pw*i/4.0f, yg = cp.y + ph*i/4.0f;
        dl->AddLine(ImVec2(xg,cp.y), ImVec2(xg,cp.y+ph), cg);
        dl->AddLine(ImVec2(cp.x,yg), ImVec2(cp.x+pw,yg), cg);
    }
    dl->AddText(ImVec2(cp.x + (pw - 80)*0.5f, cp.y+ph+16), ca, "log10 k (1/LY)");
    dl->AddText(ImVec2(cp.x - 70, cp.y + ph*0.5f - 6), ca, "log10 P(k)");

    // Find log-log ranges
    double logKMin = log10(fmax(kArr[firstValid], 1e-30));
    double logKMax = log10(fmax(kArr[lastValid],  1e-30));
    double logPMin =  1e30, logPMax = -1e30;
    for (int b = firstValid; b <= lastValid; b++) {
        if (Pk[b] <= 0.0) continue;
        double lp = log10(Pk[b]);
        if (lp < logPMin) logPMin = lp;
        if (lp > logPMax) logPMax = lp;
    }
    if (logPMax <= logPMin) { logPMax = logPMin + 1.0; }
    if (logKMax <= logKMin) { logKMax = logKMin + 1.0; }

    // Gaussian smoothing in log-log space for clean visualization
    int nValid = lastValid - firstValid + 1;
    std::vector<double> logPSm(nValid, 0.0);
    {
        int sigB  = std::max(1, std::min(6, nValid / 8));
        int halfK = sigB * 3;
        for (int i = 0; i < nValid; i++) {
            double sumW = 0.0, sumWV = 0.0;
            for (int k = -halfK; k <= halfK; k++) {
                int j = i + k;
                if (j < 0 || j >= nValid) continue;
                int b = firstValid + j;
                if (Pk[b] <= 0.0) continue;
                double w  = exp(-0.5 * (double)(k*k) / (double)(sigB*sigB));
                sumW  += w;
                sumWV += w * log10(Pk[b]);
            }
            int b = firstValid + i;
            logPSm[i] = (sumW > 0.0 && Pk[b] > 0.0)
                        ? (sumWV / sumW)
                        : (Pk[b] > 0.0 ? log10(Pk[b]) : logPMin);
        }
    }

    dl->PushClipRect(ImVec2(cp.x, cp.y), ImVec2(cp.x+pw, cp.y+ph), true);
    ImVec2 prev={}; bool first=true;
    for (int i = 0; i < nValid; i++) {
        int b = firstValid + i;
        if (Pk[b] <= 0.0) { first = true; continue; }
        double lk = log10(kArr[b]);
        double lp = logPSm[i];
        float tx = (float)((lk - logKMin) / (logKMax - logKMin));
        float ty = (float)((lp - logPMin) / (logPMax - logPMin));
        float sx = cp.x + tx * pw;
        float sy = cp.y + (1.0f - ty) * ph;
        ImVec2 cur(sx, sy);
        if (!first) dl->AddLine(prev, cur, IM_COL32(100, 220, 255, 230), 2.0f);
        prev = cur; first = false;
    }
    dl->PopClipRect();

    // Axis labels
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2e", pow(10.0, logKMin));
    dl->AddText(ImVec2(cp.x, cp.y+ph+2), ca, buf);
    snprintf(buf, sizeof(buf), "%.2e", pow(10.0, logKMax));
    { ImVec2 ts = ImGui::CalcTextSize(buf);
      dl->AddText(ImVec2(cp.x+pw-ts.x, cp.y+ph+2), ca, buf); }
    snprintf(buf, sizeof(buf), "%.2e", pow(10.0, logPMax));
    { ImVec2 ts = ImGui::CalcTextSize(buf);
      dl->AddText(ImVec2(cp.x-ts.x-4, cp.y-2), ca, buf); }
    snprintf(buf, sizeof(buf), "%.2e", pow(10.0, logPMin));
    { ImVec2 ts = ImGui::CalcTextSize(buf);
      dl->AddText(ImVec2(cp.x-ts.x-4, cp.y+ph-12), ca, buf); }

    ImGui::Dummy(sz);
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

    const int FSMOOTH   = std::max(3, std::min(60, m_fourierCount / 3));
    const int visCount  = m_fourierCount - FSMOOTH;  // points visibles après lissage
    for (int m = 0; m < 4; m++) {
        ImVec2 prev = ImVec2(0,0);
        bool first = true;
        for (int i = FSMOOTH/2; i < m_fourierCount - FSMOOTH/2; i++) {
            float sumA = 0.0f;
            for (int j = -FSMOOTH/2; j <= FSMOOTH/2; j++) {
                int idx = (m_fourierHead - m_fourierCount + i + j + FOURIER_HISTORY) % FOURIER_HISTORY;
                sumA += m_fourierA[m][idx];
            }
            float avgA = sumA / FSMOOTH;
            // X basé sur l'index : remplit toujours de gauche à droite sans offset
            float sx = px + (float)(i - FSMOOTH/2) / (float)(visCount > 1 ? visCount - 1 : 1) * plotW;
            float sy = py + (1.0f - avgA / aMax) * plotH;
            ImVec2 cur = ImVec2(sx, sy);
            if (!first) dl->AddLine(prev, cur, mCols[m], 1.5f);
            prev = cur;
            first = false;
        }
        float lx = px + 4 + m * (plotW * 0.23f);
        dl->AddLine(ImVec2(lx, py + 10), ImVec2(lx + 18, py + 10), mCols[m], 2.0f);
        dl->AddText(ImVec2(lx + 20, py + 4), mCols[m], mNames[m]);
    }

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
    // Menu TOUJOURS visible (même en cosmologie) : le changement de modèle se fait ici.
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
    if (m_useCosmology && m_eqSelectedModel != 1)
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                           "Cosmology + non-Janus model: experimental.");

    // ── Per-model parameters (live-tunable, sim units) ──────────────────────
    if (m_eqSelectedModel == 2) {          // Yukawa
        SliderFloatWithInput("Yukawa alpha", &m_yukawaAlpha, 0.0f, 5.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Strength of the short-range Yukawa boost:\n"
                              "F = G m/r^2 * (1 + alpha * exp(-r/lambda)).\n"
                              "0 = pure Newton, higher = stronger close-range attraction.");
        SliderFloatWithInput("Yukawa lambda", &m_yukawaLambda, 10.0f, 5000.0f, "%.0f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Range of the Yukawa term (sim units). Beyond ~lambda the\n"
                              "force reverts to standard Newtonian gravity.");
    } else if (m_eqSelectedModel == 3) {   // MOND
        SliderFloatWithInput("MOND a0", &m_mondA0, 0.0f, 1.0f, "%.4f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("MOND acceleration scale a0 (sim accel units). Where the\n"
                              "Newtonian acceleration g_N < a0, gravity is boosted to\n"
                              "g = sqrt(g_N * a0) (deep-MOND), giving flat rotation curves.\n"
                              "g_N >> a0 -> standard Newton. Tune until the effect is visible.");
    }
    if (m_eqSelectedModel == 0 || m_eqSelectedModel == 2 || m_eqSelectedModel == 3)
        ImGui::TextDisabled("Positive matter only (negative species inert).\n"
                            "Computed via the Barnes-Hut tree (forced for this model).");

    // Model changed: apply immediately to GPU kernel (no restart needed)
    if (m_eqSelectedModel != prevModel)
    {
        m_eqCodeBuffer[0] = '\0';
        if (m_simulation)
            m_simulation->setPhysicsModel(m_eqSelectedModel);
        // Janus est le seul modèle avec masses négatives physiquement cohérentes.
        // On désactive Show Nega Mass pour tous les autres modèles.
        bool janusNow = (m_eqSelectedModel == 1);
        // L'expansion Janus (Friedmann) est PILOTÉE par le modèle : ON en Janus, OFF sinon
        // (boîte statique a=1). Plus de case manuelle -> évite l'état bloqué quand on la
        // décochait sans régénérer.
        m_useCosmology = janusNow;
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

    // --- Adaptive radial extent (around origin) ---------------------------
    // The negative cloud expands/drifts; a fixed rMax pushes the profile off
    // the right edge over time. Size the X-axis to the 97%-enclosing radius
    // of the negative mass so the curve always fills the plot.
    float rMaxNeg = 0.f;
    for (const auto& p : m_positions) {
        if (p.w >= 0.f) continue;
        float r=sqrtf(p.x*p.x+p.y*p.y+p.z*p.z);
        if (r>rMaxNeg) rMaxNeg=r;
    }
    if (rMaxNeg < 1e-6f) rMaxNeg = m_radius * 3.0f;
    {
        const int CO=256;
        std::vector<double> coMass(CO,0.0); double totM=0.0;
        for (const auto& p : m_positions) {
            if (p.w >= 0.f) continue;
            float r=sqrtf(p.x*p.x+p.y*p.y+p.z*p.z);
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

    for (const auto& p : m_positions) {
        if (p.w >= 0.f) continue;
        float r = sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);
        if (r >= rMax) continue;
        int b = (int)(r / rMax * BINS);
        if (b >= BINS) b = BINS-1;
        massNeg[b] -= p.w; cntBin[b]++;
    }
    for (int b=0;b<BINS;b++) {
        float rIn  = (float)b/BINS * rMax;
        float rOut = (float)(b+1)/BINS * rMax;
        float vol  = (4.f/3.f) * 3.14159265f * (rOut*rOut*rOut - rIn*rIn*rIn);
        if (vol > 1e-20f) rhoNeg[b] = massNeg[b] / vol;
    }

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

    // Y-axis scaling from the actually-plotted (smoothed) curve, not the raw peak.
    float smMax = 1e-30f, smMin = 1e30f;
    for (int b=0;b<BINS;b++) {
        if (smRhoNeg[b] > smMax) smMax = smRhoNeg[b];
        if (smRhoNeg[b] > 0.0f && smRhoNeg[b] < smMin) smMin = smRhoNeg[b];
    }
    if (smMin >= smMax) smMin = smMax * 1e-3f;
    m_nfwRho0 = smMax;

    float logHi = log10f(smMax + 1e-30f), logLo = log10f(smMin + 1e-30f);
    if (logHi - logLo < 1e-3f) logLo = logHi - 3.0f;
    auto yMap = [&](float v)->float {
        float t;
        if (dmLog) { float lv = log10f(fmaxf(v, 1e-30f)); t = (lv - logLo) / (logHi - logLo); }
        else { t = v / smMax; }
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
    fmtLY(rMax * m_lyPerUnit, buf, sizeof(buf));
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

    // Utilise le background draw list pour être AU-DESSUS de la scène 3D
    // mais EN-DESSOUS des fenêtres ImGui
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

    // Utilise le background draw list pour être AU-DESSUS de la scène 3D
    // mais EN-DESSOUS des fenêtres ImGui
    ImDrawList* dl = ImGui::GetBackgroundDrawList(gMV);
    if (!dl) return;

    float velConv = (m_lyPerUnit * 9.461e12f) / 3.156e13f;
    // Hubble parameter H = aDot/a for peculiar velocity subtraction
    float H_now = (m_simulation && m_useCosmology) ? m_simulation->getHubbleParam() : 0.0f;
    bool haveVels = !m_rotVelocities.empty() && m_rotVelocities.size() >= m_positions.size();

    // Champ σ pour la température cinétique des marqueurs. En MARCHE les positions bougent
    // chaque frame -> recalcul CHAQUE frame (sinon la grille périmée renvoie une cellule vide
    // pour la particule qui a bougé -> σ=0 -> la T disparaît). En PAUSE : throttlé (statique).
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

        // Vitesse péculière en sim units (Hubble flow soustrait), puis conversion
        // physique : 1 sim_vel_unit = lyPerUnit × ly/MY = lyPerUnit × 0.30 km/s.
        float spdSim = 0.0f;
        if (haveVels && idx < (int)m_rotVelocities.size())
        {
            const cl_float4& v = m_rotVelocities[idx];
            float pvx = v.x - H_now * p.x;
            float pvy = v.y - H_now * p.y;
            float pvz = v.z - H_now * p.z;
            spdSim = sqrtf(pvx*pvx + pvy*pvy + pvz*pvz);
        }
        float spdKms = spdSim * m_lyPerUnit * 0.30f;  // → km/s

        // Température cinétique = σ locale (km/s) -> T équivalente (gaz/viriel) :
        // k_B T = μ m_p σ_1D²  ->  T[K] ≈ 72.6·σ_1D[km/s]²  (μ≈0.6), σ_1D = σ_3D/√3.
        char tbuf[56] = "";
        float sigKms = markerSigmaKms(idx);
        if (sigKms > 0.0f) {
            float sig1d = sigKms / 1.7320508f;
            float Tk    = 72.6f * sig1d * sig1d;
            snprintf(tbuf, sizeof(tbuf), "T~%.1e K (sig %.0f km/s)", Tk, sigKms);
        }

        // Ligne 1 = vitesse (km/s), Ligne 2 = température (sous la vitesse). Centrées sur sx.
        char line1[48];
        if (spdSim > 0.0f) {
            if (spdKms >= 1000.0f)
                snprintf(line1, sizeof(line1), "#%zu %.1f km/s (%.2fc)", m + 1, spdKms, spdKms / 299792.458f);
            else
                snprintf(line1, sizeof(line1), "#%zu %.1f km/s", m + 1, spdKms);
        } else {
            snprintf(line1, sizeof(line1), "#%zu", m + 1);
        }

        auto drawCentered = [&](const char* txt, float topY) -> float {
            ImVec2 ts = ImGui::CalcTextSize(txt);
            float lx = sx - ts.x * 0.5f;
            dl->AddRectFilled(ImVec2(lx - 2, topY - 1),
                              ImVec2(lx + ts.x + 2, topY + ts.y + 1),
                              IM_COL32(0, 0, 0, 180));
            dl->AddText(ImVec2(lx, topY), colText, txt);
            return ts.y;
        };

        float lineH = ImGui::GetTextLineHeight();
        bool  hasTemp = (tbuf[0] != '\0');
        // Bloc empilé au-dessus de la flèche : km/s en haut, température dessous.
        float blockH = hasTemp ? (2.0f * lineH + 1.0f) : lineH;
        float topY   = base.y - blockH - 4.0f;
        drawCentered(line1, topY);
        if (hasTemp) drawCentered(tbuf, topY + lineH + 1.0f);
    }
}

// Repère d'orientation XYZ (triade), type Blender/Maya/SolidWorks : trois axes
// colorés en bas-droite du viewport, orientés selon la caméra courante. Permet de
// savoir dans quel sens on regarde la structure. Indépendant de la position : on
// projette les DIRECTIONS de base du monde (w=0) par la view-proj.
void App::renderAxisGizmo()
{
    if (!m_showAxisGizmo || !m_camera) return;

    float vp[16];
    m_camera->getViewProjMatrix(vp);

    ImGuiViewport* gMV = ImGui::GetMainViewport();
    if (!gMV) return;
    const float panelW = 340.0f;
    const float vpX = gMV->Pos.x + panelW;
    const float vpW = (float)m_width - panelW;
    const float vpH = (float)m_height;
    if (vpW <= 0 || vpH <= 0) return;

    // Centre de la triade : coin bas-droite du viewport, avec une marge.
    const float radius = 42.0f;          // longueur des axes en pixels
    const float margin = radius + 24.0f;
    ImVec2 ctr(vpX + vpW - margin, gMV->Pos.y + vpH - margin);

    ImDrawList* dl = ImGui::GetForegroundDrawList(gMV);
    if (!dl) return;

    struct Axis { float dir[3]; ImU32 col; const char* name; };
    const Axis axes[3] = {
        { {1,0,0}, IM_COL32(225, 75, 75, 255),  "X" },   // X rouge
        { {0,1,0}, IM_COL32(90, 200, 75, 255),  "Y" },   // Y vert
        { {0,0,1}, IM_COL32(90, 120, 235, 255), "Z" },   // Z bleu
    };

    // Trois FLÈCHES (hampe + cône), sans cube. On projette les directions de base
    // du monde (w=0), puis on trie en profondeur (painter) pour que la flèche qui
    // pointe vers l'observateur passe devant celle qui s'éloigne.
    struct Arrow { ImVec2 tip; ImU32 col; float depth; const char* name; };
    Arrow ar[3];
    for (int a = 0; a < 3; a++) {
        const float* d = axes[a].dir;
        float cx = vp[0]*d[0] + vp[4]*d[1] + vp[8]*d[2];
        float cy = vp[1]*d[0] + vp[5]*d[1] + vp[9]*d[2];
        float cw = vp[3]*d[0] + vp[7]*d[1] + vp[11]*d[2];   // ~ -z_eye : profondeur
        float len = sqrtf(cx*cx + cy*cy);
        if (len < 1e-6f) len = 1e-6f;
        float ux = cx / len, uy = cy / len;
        ar[a] = { ImVec2(ctr.x + ux*radius, ctr.y - uy*radius), axes[a].col, cw, axes[a].name };
    }
    // tri painter : depth décroissante (loin d'abord)
    int order[3] = {0,1,2};
    for (int i=0;i<3;i++) for (int j=i+1;j<3;j++)
        if (ar[order[j]].depth > ar[order[i]].depth) { int t=order[i]; order[i]=order[j]; order[j]=t; }

    const float headLen = 13.0f, headW = 6.5f;
    for (int k = 0; k < 3; k++) {
        const Arrow& A = ar[order[k]];
        float dx = A.tip.x - ctr.x, dy = A.tip.y - ctr.y;
        float l = sqrtf(dx*dx + dy*dy); if (l < 1e-4f) l = 1e-4f;
        float sx = dx/l, sy = dy/l;             // direction écran centre->bout
        ImVec2 neck(A.tip.x - sx*headLen, A.tip.y - sy*headLen);   // base du cône
        ImVec2 perp(-sy, sx);
        // hampe
        dl->AddLine(ctr, neck, A.col, 2.6f);
        // cône (tête de flèche)
        dl->AddTriangleFilled(
            A.tip,
            ImVec2(neck.x + perp.x*headW, neck.y + perp.y*headW),
            ImVec2(neck.x - perp.x*headW, neck.y - perp.y*headW),
            A.col);
        // lettre X/Y/Z, légèrement au-delà du cône, avec liseré noir pour lisibilité
        ImVec2 ts = ImGui::CalcTextSize(A.name);
        ImVec2 lp(A.tip.x + sx*9.0f - ts.x*0.5f, A.tip.y + sy*9.0f - ts.y*0.5f);
        dl->AddText(ImVec2(lp.x+1, lp.y+1), IM_COL32(0,0,0,200), A.name);
        dl->AddText(lp, A.col, A.name);
    }
    // moyeu central
    dl->AddCircleFilled(ctr, 3.5f, IM_COL32(210, 210, 220, 255), 12);
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

    // Standby overlay (tant que l'utilisateur n'a pas cliqué Apply & Generate).
    // Drapeau explicite -> robuste (indépendant de m_particleCount / du chargement du cfg,
    // qui généraient des faux positifs faisant disparaître la fenêtre).
    if (!m_everGenerated) {
        // 1) Fond sombre plein viewport (BACKGROUND : cache les artefacts, derrière les fenêtres).
        bg->AddRectFilled(ImVec2(vpX, mvY), ImVec2(vpX + vpW, mvY + vpH),
                          IM_COL32(8, 12, 24, 255));
        // 2) Fenêtre de bienvenue : VRAIE fenêtre ImGui centrée (rendu garanti), NoInputs
        //    -> visible mais ne capte pas les clics (n'obstrue pas le panneau/l'éditeur).
        ImGui::SetNextWindowPos(ImVec2(vpX + vpW * 0.5f, mvY + vpH * 0.5f),
                                ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.07f, 0.14f, 0.33f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_Border,   ImVec4(0.35f, 0.59f, 1.0f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 2.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(26, 20));
        ImGui::Begin("##welcome", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoInputs |
            ImGuiWindowFlags_NoBringToFrontOnFocus);   // stays behind analytics windows (P(k), DM, Barycenters...)
        // Titre centré, en 18 (scale relatif à la police 16 de base).
        ImGui::SetWindowFontScale(18.0f / 16.0f);
        {
            const char* title = "Janus Universe";
            float tw = ImGui::CalcTextSize(title).x;
            float availW = ImGui::GetContentRegionAvail().x;
            if (availW > tw) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (availW - tw) * 0.5f);
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), title);
        }
        ImGui::SetWindowFontScale(1.0f);
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.82f, 0.82f, 0.82f, 1.0f), "Press  'Apply & Generate'  to start");
        ImGui::End();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);
    }

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


    ImGui::PushID("universe");
    if (ImGui::CollapsingHeader("Universe", ImGuiTreeNodeFlags_DefaultOpen))
    {
        SliderFloatWithInput("Init Radius", &m_radius, 1.f, 10000.f, "%.0f");
        // Limite haute = 2.5M par espèce → 5M total (compatible RTX 4090 / A6000)
        // Posi jusqu'à 20 M : un modèle mono-espèce (Newton/ΛCDM/MOND, pas de masse
        // négative) peut mettre tout le budget sur les positives. Ce qui limite (VRAM +
        // calcul) est le TOTAL : 20 M positives seules ≈ 10 M + 10 M en bimétrique.
        SliderIntWithInput  ("Posi Part",   &m_numPosiStars,  0, 20000000);
        SliderIntWithInput  ("Nega Part",   &m_numNegaStars,  0, 10000000);
        SliderFloatWithInput("Posi Mass",   &m_totalSolarMass, 0.001f, 100.f, "%.3f");
        SliderFloatWithInput("Nega Mass",   &m_negaDensity,  -100.f, -0.001f, "%.3f");
        SliderFloatWithInput("Pert Amp",    &m_pertAmp,        0.0f,   1.5f,  "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Amplitude of seeded perturbations, free-explosion sphere.\n"
                                                      "0.5 = soft, 1.0 = sharp filaments from shell crossing.");
        SliderIntWithInput  ("Web cells",   &m_webCells,       4,      100);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(m_foamMode
            ? "Foam: number of bubbles PER AXIS, uniform bubbles.\n"
              "Higher = more, smaller bubbles.\n"
              "Limited by seed-grid resolution -> more particles needed for more bubbles.\n"
              "Requires Apply & Generate."
            : "Hierarchical P(k): small-scale spectrum cutoff.\n"
              "Higher = more fine modes kept = finer structures.\n"
              "Requires Apply & Generate.");
        ImGui::Checkbox("Foam##foam", &m_foamMode);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Checked: peaked spectrum -> foam of equal-size bubbles, Petit honeycomb scheme.\n"
            "  Web cells = number of bubbles per axis.\n"
            "Unchecked: hierarchical CDM P(k) -> realistic observed web, clusters of varied sizes.\n"
            "Requires Apply & Generate.");
    }
    ImGui::PopID();

    ImGui::PushID("cosmo");
    // Force open and always show cosmology section
    bool cosmoOpen = ImGui::CollapsingHeader("Cosmology", ImGuiTreeNodeFlags_DefaultOpen);
    if (cosmoOpen)
    {
        // "Janus expansion" = état PILOTÉ par le modèle de l'Equation Editor (Janus = ON,
        // Friedmann ; autre modèle = OFF, boîte statique a=1). Plus de case : on affiche
        // seulement l'état pour lever toute ambiguïté (décocher figeait la simulation).
        ImGui::Text("Janus expansion :");
        ImGui::SameLine();
        if (m_useCosmology)
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), "ON  (Friedmann)");
        else
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "OFF (static box a=1)");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Driven by the Equation Editor model:\n"
                              "Janus  -> expansion ON (Friedmann).\n"
                              "Newton / Yukawa / MOND / Custom -> expansion OFF (static box a=1).\n"
                              "Regenerate (Apply & Generate) after changing the model.");
        {
            ImGui::Checkbox("Adaptive dt", &m_adaptiveDt);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Adaptive timestep: dt = eta * sqrt(epsilon / |a_max|)");
            if (m_adaptiveDt) {
                SliderFloatWithInput("eta", &m_eta, 0.05f, 1.0f, "%.2f");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Adaptive precision: 0.1 = accurate / slow,  0.4 = fast / approximate");
                SliderFloatWithInput("dtMax", &m_dtMax, 0.1f, 20.0f, "%.1f");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Maximum timestep (MY): raise to speed up (2 = default, 5 = fast)");
            }

            // ── GPU Barnes-Hut Tree (O(N log N), bypasses PM) ────────────────
            ImGui::Checkbox("Use Tree", &m_useTree);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("GPU Barnes-Hut octree (Karras 2012 LBVH)\n"
                                  "N^2-like forces but O(N log N) (faster for N > 100k)\n"
                                  "Two-trees per species -> avoids Janus PM cancellation\n"
                                  "When checked, BYPASSES the PM mode\n"
                                  "Requires Apply & Generate");

            // ── Anti-monopole (Jeans swindle) : supprime l'effondrement central des négatives ──
            ImGui::Checkbox("Anti-monopole", &m_removeMonopole);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Removes the MEAN radial infall per species around the center:\n"
                                  "suppresses the central collapse of negative mass,\n"
                                  "a finite-sphere artifact. Keeps only the fluctuations\n"
                                  "that form the web. Live, Tree path. Standard N-body cosmology.");
            if (m_removeMonopole)
                SliderFloatWithInput("Mono frac", &m_monopoleFrac, 0.0f, 1.0f, "%.2f");
            // Setter trivial -> appliqué en direct chaque frame (live-tunable, Tree).
            if (m_simulation) {
                m_simulation->setMonopoleRemoval(m_removeMonopole, m_monopoleFrac);
                m_simulation->setYukawa(m_yukawaAlpha, m_yukawaLambda);
                m_simulation->setMondA0(m_mondA0);
            }

            if (m_useTree) {
                SliderFloatWithInput("Tree theta", &m_treeTheta, 0.3f, 1.0f, "%.2f");
                if (m_simulation) m_simulation->setTreeTheta(m_treeTheta);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Barnes-Hut opening angle (live-tunable).\n"
                                      "0.3 = very accurate (~0.5%% error), slow\n"
                                      "0.5 = accurate (~1%% error), DEFAULT\n"
                                      "0.7 = good trade-off (~3%% error), 1.5-2x faster\n"
                                      "1.0 = fast but visible artifacts (~5-10%% error)\n"
                                      "Mostly affects FPS in dense clusters");
                ImGui::Checkbox("Quadrupole", &m_treeQuad);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Order-2 multipole correction: lets you raise theta\n"
                                      "without losing accuracy -> faster at equal quality.");
                ImGui::SameLine();
                ImGui::Checkbox("Warp-coherent", &m_treeWarp);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Warp-coherent traversal (Morton order).\n"
                                      "~3x faster in dense regions. Same physics.");
                ImGui::SetNextItemWidth(120);
                ImGui::SliderInt("Force subcycle##sub", &m_forceSubcycle, 1, 16);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Recomputes the tree force every K steps and REUSES it\n"
                                      "in between, the tree is the main cost. K=1 = normal.\n"
                                      "K=4-8: big speedup in the FROZEN phase, low velocities.\n"
                                      "Do NOT raise it in the active collapse phase -> inaccurate.");
                if (m_simulation) {
                    m_simulation->setTreeQuadrupole(m_treeQuad);
                    m_simulation->setTreeWarpCoherent(m_treeWarp);
                    m_simulation->setForceSubcycle(m_forceSubcycle);
                }
            }

            // ── Particle-Mesh solver ──────────────────────────────────────────
            if (m_useTree) ImGui::BeginDisabled();
            ImGui::Checkbox("Use PM", &m_usePM);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Particle-Mesh: O(N log N) instead of O(N^2)\n"
                                  "Ng^3 grid — ~100x faster for N > 200k\n"
                                  "Requires Apply & Generate to take effect");
            if (m_useTree) ImGui::EndDisabled();
            if (m_usePM) {
                const char* grids[] = { "64", "128", "256", "512" };
                int idx = (m_pmNg == 64) ? 0 : (m_pmNg == 128) ? 1 : (m_pmNg == 256) ? 2 : 3;
                if (ImGui::Combo("Grid##pm", &idx, grids, 4))
                    m_pmNg = (idx == 0) ? 64 : (idx == 1) ? 128 : (idx == 2) ? 256 : 512;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("PM resolution: 64^3 = 8MB, 128^3 = 50MB, 256^3 = 400MB\n"
                                      "512^3 = ~17 GiB + FFT workspace (24 GB+ VRAM; not for 16 GB cards)\n"
                                      "Use 256 for Janus soap-bubble structure on a 16 GB GPU");

                ImGui::Checkbox("Enable SR (P3M)", &m_pmEnableSR);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("P3M short-range correction (GADGET-style Ewald split)\n"
                                      "ON  = smoothed PM + direct SR, compact N^2-like clumps (slower)\n"
                                      "OFF = pure PM (fast, fuzzy structures limited to 1 cell)\n"
                                      "r_split = 1 x cell, r_cut = 4.5 x r_split\n"
                                      "Requires Apply & Generate");

                // « PM Periodic Cube » retiré : la boîte périodique produisait des
                // artefacts d'axes (matière sur les arêtes/coins) et ne formait pas de
                // mousse stable. PM = toujours bord ouvert (sphère).
                m_pmPeriodic = false;
            }
        }

        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "--- Matter Sectors ---");
        SliderFloatWithInput("OmegaM", &m_omegaM, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Positive sector density (Omega+ ~ 0.30)");
        SliderFloatWithInput("Omega- eff", &m_omegaL, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Effective contribution of negative mass\n= equivalent OmegaL in Janus (Petit)\nValue ~ 0.70 matches observations");

        {
            SliderFloatWithInput("aStart", &m_aStart, 0.01f, 1.0f, "%.3f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                "Initial scale factor  a = 1/(1+z)\n"
                "0.05 = z=19 (early universe, Zel'dovich ICs)\n"
                "1.00 = z=0  (current epoch, free-explosion ICs)");
            SliderFloatWithInput("Cosmo Scale", &m_cosmoScale, 0.001f, 1.0f, "%.3f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                "Decouples Friedmann from N-body timescale.\n"
                "0.02 : cosmic expansion 50x slower than N-body\n"
                "  -> structures at 30-50 MY, expansion visible at 150-300 MY\n"
                "1.0  : real velocities (too fast, no structures)\n"
                "0.001: negligible expansion");

            // ── Reproducibility : manual RNG seed ─────────────────────────────
            ImGui::Checkbox("Fixed Seed", &m_useManualSeed);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("ON  = use the seed below for IC RNG -> identical runs every Apply & Generate\n"
                                  "OFF = time-based seed (different result each generation)\n"
                                  "Useful for statistical ensembles : N runs with different seeds.");
            if (m_useManualSeed) {
                ImGui::SetNextItemWidth(140);
                ImGui::InputInt("Seed", &m_manualSeed);
                if (m_manualSeed < 1) m_manualSeed = 1;
                ImGui::SameLine();
                if (ImGui::Button("Randomize##seed"))
                    m_manualSeed = (int)(SDL_GetTicks64() & 0x7FFFFFFF);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Pick a fresh pseudo-random seed (still reproducible afterwards)");
            }

            // ── Batch Statistical Mode ───────────────────────────────────────
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.6f, 1.0f), "Batch statistical ensemble");
            if (!m_batchActive) {
                ImGui::SetNextItemWidth(70);
                ImGui::InputInt("N runs##batch", &m_batchN);
                if (m_batchN < 1) m_batchN = 1; if (m_batchN > 1000) m_batchN = 1000;
                ImGui::SameLine();
                ImGui::SetNextItemWidth(70);
                ImGui::InputInt("Start seed##batch", &m_batchStartSeed);
                if (m_batchStartSeed < 1) m_batchStartSeed = 1;
                ImGui::SetNextItemWidth(140);
                ImGui::InputFloat("Target (MY)##batch", &m_batchTargetMY, 0.0f, 0.0f, "%.0f");
                if (m_batchTargetMY < 1.0f) m_batchTargetMY = 1.0f;
                ImGui::TextDisabled("Out dir (CSV export):");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputText("##batchOutDir", m_batchOutDir, sizeof(m_batchOutDir));
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Directory where the batch exports CSV files: P(k), counts, etc.");
                if (ImGui::Button("Start Batch", ImVec2(-1, 24))) {
                    m_batchActive     = true;
                    m_batchCurrentIdx = 0;
                    m_batchState      = 1;  // GENERATING
                    m_useManualSeed   = true;
                    m_manualSeed      = m_batchStartSeed;
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Runs N simulations with seeds Start, Start+1, ...\n"
                                      "Each runs until Target MY, then exports particles to CSV.\n"
                                      "Useful for statistical ensembles (P(k), galaxy counts...)");
            } else {
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 1.0f, 1.0f),
                                   "Run %d / %d  seed=%d  state=%d",
                                   m_batchCurrentIdx + 1, m_batchN,
                                   m_batchStartSeed + m_batchCurrentIdx, m_batchState);
                if (ImGui::Button("Cancel Batch", ImVec2(-1, 24))) {
                    m_batchActive = false;
                    m_batchState  = 0;
                }
            }
        }

        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "--- Hubble & Threshold ---");

        float H0c = 0.f, H0c_matter = 0.f;
        {
            float R3    = m_radius * m_radius * m_radius;
            float vol   = (4.0f / 3.0f) * 3.14159f * R3;
            float rho   = (vol > 0.f) ? (float)m_numPosiStars * m_totalSolarMass / vol : 0.f;
            float G_eff = m_gravConstant * 1.0e10f;
            float H0c_matter_sq = (rho > 0.f) ? (8.0f * 3.14159f * G_eff * rho / 3.0f) : 0.f;
            H0c_matter = sqrtf(H0c_matter_sq);
            // Critical H0 = matter term ONLY. The old version added Omega-*H0^2, which
            // makes crit depend on H0 -> the displayed "crit" rises every time you set
            // H0 to it (endless escalation). Matter-only crit is a STABLE target.
            H0c = H0c_matter;
        }
        // Slider max from the FIXED matter-only crit (NOT H0c, which contains an
        // omegaL*H0^2 term that would feed back and blow H0 up to infinity).
        float H0sliderMax = std::max(0.05f, H0c_matter * 3.0f);
        if (!std::isfinite(m_H0) || m_H0 > H0sliderMax) m_H0 = H0c_matter;  // recover from runaway
        SliderFloatWithInput("H0", &m_H0, 0.0f, H0sliderMax, "%.4f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Hubble constant (sim units)  H0_crit = sqrt(8pi G rho/3)");

        // H0 indicator — compact 3-zone display
        {
            if (m_H0 < H0c * 0.6f)
                ImGui::TextColored(ImVec4(1.f, 0.3f, 0.1f, 1.f),
                    "Collapse!  crit=%.4f", H0c);
            else if (m_H0 > H0c * 1.05f)
                ImGui::TextColored(ImVec4(1.f, 0.7f, 0.1f, 1.f),
                    "Too fast!  crit=%.4f", H0c);
            else
                ImGui::TextColored(ImVec4(0.3f, 1.f, 0.3f, 1.f),
                    "Optimal    crit=%.4f", H0c);
        }

        ImGui::Separator();
        if (!m_simRunning)
            ImGui::TextDisabled("Galaxies: --");
        else if (m_galaxyCount == 0)
            ImGui::TextDisabled("Galaxies: waiting...");
        else if (m_firstGalaxyDetected)
            ImGui::TextColored(ImVec4(0.4f,1.f,0.6f,1.f),
                "Galaxies: %d (>= %d)  1st: %.1f MY",
                m_galaxyCount, m_galaxyMinPart, m_firstGalaxyTime);
        else
            ImGui::TextColored(ImVec4(0.4f,1.f,0.6f,1.f),
                "Galaxies: %d (>= %d)",
                m_galaxyCount, m_galaxyMinPart);
        // Extended range: with N=5M, clusters can have thousands of particles
        SliderIntWithInput("Galaxy min", &m_galaxyMinPart, 10, 10000);
        SliderFloatWithInput("Galaxy link", &m_galaxyLinkScale, 0.04f, 0.25f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Galaxy FoF linking length, x mean separation.\n"
                              "Short 0.05-0.10 = dense nodes well separated.\n"
                              "Long 0.15-0.20 = merges toward the cluster scale.");
        ImGui::Checkbox("Show Galaxy Markers", &m_showGalaxyMarkers);
    }
    ImGui::PopID();

    ImGui::PushID("sim");
    if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
    {
        SliderFloatWithInput("Time Step", &m_timeStep, 0.1f, 100.0f, "%.2f");
        SliderFloatWithInput("Softening", &m_softening, 1.0f, 5000.0f, "%.0f");
        if (m_softening < 1.0f) m_softening = 1.0f;
        SliderFloatWithInput("G Const", &m_gravConstant, 1.0e-14f, 1.0e-8f, "%.2e");

        // Softening / inter-particle separation ratio
        // - too small (< 0.05) : close-pair kicks → numerical heating
        // - 0.05-0.3            : fine resolution (slow but accurate)
        // - 0.3-2.0             : balanced (recommended for PM/Tree cosmology)
        // - > 2.0               : over-smoothed (structure invisible)
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

        // ── State I/O ───────────────────────────────────────────────────────
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

        // ── Presets ─────────────────────────────────────────────────────────
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
            const char* combo = presets.empty() ? "no presets" : presets[presetSel].c_str();
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
    }
    ImGui::PopID();

    ImGui::PushID("gfx");
    if (ImGui::CollapsingHeader("Graphic Tools", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // ── Particle display & color modes ──────────────────────────────────
        {
            bool isJ = (m_eqSelectedModel == 1);
            if (!isJ) ImGui::BeginDisabled();
            bool prevShow = m_showNegaMass;
            ImGui::Checkbox("Show Nega Mass", &m_showNegaMass);
            if (!isJ) { ImGui::SameLine(); ImGui::TextDisabled("Janus only"); ImGui::EndDisabled(); }
            if (m_showNegaMass != prevShow) uploadParticlesToGPU();

            bool prevCBV = m_colorByVelocity;
            ImGui::Checkbox("Color by Velocity", &m_colorByVelocity);
            if (m_colorByVelocity) {
                ImGui::Indent();
                ImGui::Checkbox("Auto-scale", &m_velAutoScale);
                bool prevPeak = m_velPeakHold;
                ImGui::Checkbox("Peak hold", &m_velPeakHold);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("L'echelle capture le pic de vitesse et le GARDE.\n"
                                      "-> la baisse des vitesses (refroidissement Big Freeze)\n"
                                      "devient visible TOUT SEUL au fil du temps.");
                // 0 + recalcul immédiat -> l'échelle se cale sur le PIC RÉEL courant
                // (le péculier est << 1.0 ; remettre 1.0 écrasait tout en bleu).
                if (m_velPeakHold && !prevPeak) { m_velMaxForColor = 0.0f; uploadParticlesToGPU(); }
                if (m_velPeakHold) { ImGui::SameLine(); if (ImGui::SmallButton("Reset scale")) { m_velMaxForColor = 0.0f; uploadParticlesToGPU(); } }
                ImGui::Unindent();
            }
            if (m_colorByVelocity != prevCBV) {
                if (m_colorByVelocity) { m_colorByDensity = false; m_colorByHalo = false; m_colorByWeb = false; m_colorByTemp = false; if (m_simRunning) refreshAnalytics(); }
                uploadParticlesToGPU();
            }

            bool prevCBW = m_colorByWeb;
            ImGui::Checkbox("Color by Web", &m_colorByWeb);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Cosmic-web classification, T-web from the density Hessian:\n"
                                  "VOID = dark, WALL = blue, FILAMENT = green, NODE = red.\n"
                                  "Gives an overview of the structure: voids/walls/filaments/clusters.\n"
                                  "Periodic recompute, heavy.");
            if (m_colorByWeb != prevCBW) {
                if (m_colorByWeb) { m_colorByVelocity = false; m_colorByDensity = false; m_colorByHalo = false; m_colorByTemp = false; computeTweb(); }
                uploadParticlesToGPU();
            }

            bool prevCBT = m_colorByTemp;
            ImGui::Checkbox("Color by Temperature", &m_colorByTemp);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Kinetic temperature = LOCAL velocity dispersion, sigma^2.\n"
                                  "HOT red/white = virialized clusters / shocks; COLD blue = coherent\n"
                                  "flow / voids. Physical and objective, close to the 'Gas Temperature' panel.\n"
                                  "Periodic recompute.");
            if (m_colorByTemp != prevCBT) {
                if (m_colorByTemp) { m_colorByVelocity = false; m_colorByDensity = false; m_colorByHalo = false; m_colorByWeb = false; computeTemperature(); }
                uploadParticlesToGPU();
            }
        }

        ImGui::Separator();
        ImGui::Checkbox("Enable Glow", &m_glowEnabled);
        if (m_glowEnabled)
        {
            SliderFloatWithInput("Blur Radius", &m_blurRadius,    1.0f, 20.0f, "%.1f");
            SliderFloatWithInput("Glow Intens", &m_glowIntensity, 0.0f,  8.0f, "%.2f");
        }
        ImGui::Checkbox("Soft Halo", &m_softHalo);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Volumetric SPH rendering: adaptive-size kernel splat -> continuous\n"
                              "field. Hue = mean property, temperature/density; brightness =\n"
                              "column density. py-sphviewer / Illustris look. Combine with a Color by mode.");
        if (m_softHalo) {
            SliderFloatWithInput("Halo Bright", &m_sphBright, 0.02f, 8.0f, "%.3f");
            SliderFloatWithInput("Smooth", &m_sphSmooth, 0.0f, 8.0f, "%.1f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Gaussian blur of the density field, px: turns points into\n"
                                  "CONTINUOUS filaments, py-sphviewer style. 0 = raw, pointillist,\n"
                                  "2-4 = smooth. This is the real particle 'smoothing'.");
            SliderFloatWithInput("Speed quality", &m_sphQuality, 0.35f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Splat resolution of the SPH field. Lower = FASTER\n"
                                  "(0.5 ~= 4x less fill-rate) with little visible loss\n"
                                  "since the field is smooth. 1.0 = full quality.");
            SliderFloatWithInput("Core Detail", &m_sphCoreDetail, 40.0f, 600.0f, "%.0f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("asinh compression ceiling, independent of Halo Bright.\n"
                                  "40 = white incandescent cores, saturated. Raise 200-600 to\n"
                                  "stretch the high range -> gradient/detail INSIDE the dense cores.\n"
                                  "Then raise Halo Bright to relight the surroundings.");
            ImGui::Checkbox("Color by density", &m_sphViridis);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("py-sphviewer-style density map: the HUE encodes the\n"
                                  "column density. Ignores the 'Color by' mode. Raise Halo Bright for the haze.");
            if (m_sphViridis) {
                ImGui::TextUnformatted("Palette:"); ImGui::SameLine();
                ImGui::RadioButton("Viridis", &m_sphPalette, 0); ImGui::SameLine();
                ImGui::RadioButton("Thermal", &m_sphPalette, 1); ImGui::SameLine();
                ImGui::RadioButton("Galaxy",  &m_sphPalette, 2);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("GALAXY COLORS (Hubble look): deep blue diffuse halo -> blue ->\n"
                                      "cream disk -> warm gold bulge -> golden-white cores.");
            }
        }

        ImGui::Separator();
        SliderFloatWithInput("Pt Size", &m_particleSize, 0.05f, 12.0f, "%.2f");
        SliderFloatWithInput("Pt Size Nega", &m_particleSizeNeg, 0.05f, 3.0f, "%.2f");
        ImGui::Checkbox("Show Grid", &m_showGrid);
        if (m_showGrid)
            SliderFloatWithInput("ly/unit", &m_lyPerUnit, 1.0f, 10000000.0f, "%.0f");

        // ── Annotations (for annotated screenshots, on pause) ─────────────────
        ImGui::Separator();
        ImGui::TextDisabled("Annotations");
        const char* tools[] = { "Off", "Arrow + text", "Text", "Rectangle", "Circle" };
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##annottool", &m_annotTool, tools, 5))
            m_annotStage = 0;
        if (m_annotTool != 0) {
            ImGui::ColorEdit3("##annotcol", m_annotColor,
                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
            ImGui::SameLine();
            ImGui::TextDisabled("%s", m_annotTool == 2 ? "click to place text" : "click start, click end");
        }
        if (!m_annotations.empty()) {
            ImGui::Text("%zu placed", m_annotations.size());
            ImGui::SameLine();
            if (ImGui::SmallButton("Clear##annot")) { m_annotations.clear(); m_annotEditIdx = -1; }
            const char* tnames[] = { "", "Arrow", "Text", "Rect", "Circle" };
            int toDelete = -1;
            float listH = ImGui::GetTextLineHeightWithSpacing() *
                          (float)std::min((int)m_annotations.size(), 6) + 8.0f;
            ImGui::BeginChild("##annotlist", ImVec2(-1, listH), true);
            for (int i = 0; i < (int)m_annotations.size(); i++) {
                const Annotation& a = m_annotations[i];
                ImGui::PushID(i);
                if (ImGui::SmallButton("x")) toDelete = i;
                ImGui::SameLine();
                ImVec4 sw = ImGui::ColorConvertU32ToFloat4(a.col);
                ImGui::ColorButton("##sw", sw,
                    ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker, ImVec2(12, 12));
                ImGui::SameLine();
                const char* tn = (a.type >= 1 && a.type <= 4) ? tnames[a.type] : "?";
                if (a.text[0]) ImGui::Text("%d. %s \"%s\"", i + 1, tn, a.text);
                else           ImGui::Text("%d. %s", i + 1, tn);
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

        // ── View / orientation ──────────────────────────────────────────────
        ImGui::Separator();
        ImGui::Checkbox("XYZ Axes", &m_showAxisGizmo);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Orientation triad, X red / Y green / Z blue, bottom-right,\n"
                              "oriented by the camera, like a 3D software.");

        if (m_spaceMouse && m_spaceMouse->available()) {
            static bool s_3dFail = false;
            if (ImGui::Checkbox("3DConnexion", &m_use3DMouse)) {
                if (m_use3DMouse) { s_3dFail = !m_spaceMouse->enable(); if (s_3dFail) m_use3DMouse = false; }
                else              { m_spaceMouse->disable(); s_3dFail = false; }
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("6-DoF SpaceMouse navigation: simultaneous pan/zoom/orbit\n"
                                  "around the pivot. Requires the 3Dconnexion driver.");
            if (s_3dFail)
                ImGui::TextColored(ImVec4(1.0f,0.6f,0.3f,1.0f), "No 3Dconnexion device/driver detected");
        }

        // ── Clip plane: 3 combinable axes -> cut a corner/octant ────────────
        if (ImGui::Checkbox("Clip Plane", &m_clipEnable) && m_clipEnable && !m_positions.empty()) {
            double cx=0, cy=0, cz=0;
            for (const auto& p : m_positions) { cx+=p.x; cy+=p.y; cz+=p.z; }
            double inv = 1.0/(double)m_positions.size();
            m_clipCenter[0]=(float)(cx*inv); m_clipCenter[1]=(float)(cy*inv); m_clipCenter[2]=(float)(cz*inv);
            float mxe = 1.0f;
            for (const auto& p : m_positions) {
                mxe = std::max(mxe, std::fabs(p.x - m_clipCenter[0]));
                mxe = std::max(mxe, std::fabs(p.y - m_clipCenter[1]));
                mxe = std::max(mxe, std::fabs(p.z - m_clipCenter[2]));
            }
            m_clipExtent = mxe;
        }
        if (m_clipEnable) {
            const char* clipAxes[3] = { "Clip X", "Clip Y", "Clip Z" };
            for (int a = 0; a < 3; a++) {
                ImGui::PushID(a);
                ImGui::Checkbox(clipAxes[a], &m_clipAxisOn[a]);
                if (m_clipAxisOn[a]) {
                    SliderFloatWithInput("pos", &m_clipAxisFrac[a], -1.0f, 1.0f, "%.3f");
                    ImGui::SameLine(); ImGui::Checkbox("inv", &m_clipAxisFlip[a]);
                }
                ImGui::PopID();
            }
        }

        // ── Auto-fit: camera eases back to keep the structure at constant
        //    apparent size as it expands. ──
        if (ImGui::Checkbox("Auto-fit", &m_autoFit) && m_autoFit) {
            m_autoFitDistVel   = 0.0f;
            m_autoFitLastTicks = 0;
            m_autoFitTargetDist= 0.0f;
            m_autoFitTargetRaw = 0.0f;
            m_autoFitLastFrame = -100000;   // reframe immediately
        }
        if (m_autoFit) {
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("Fill##autofit", &m_autoFitFill, 0.10f, 0.95f, "fills %.2f");
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("Smoothing##autofit", &m_autoFitSmoothTime, 0.15f, 2.50f, "%.2f s");
            ImGui::TextDisabled("Structure R (P90): %.0f  |  target dist: %.0f",
                                m_autoFitRadius, m_autoFitTargetDist);
            ImGui::TextDisabled("Mouse wheel adjusts the framing");
        }

        // ── Capture ─────────────────────────────────────────────────────────
        ImGui::Separator();
        if (ImGui::Button("Screenshot", ImVec2(-1, 22))) saveScreenshot();

        if (!m_videoRecording) {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.65f,0.18f,0.18f,1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.80f,0.24f,0.24f,1.0f));
            if (ImGui::Button("● Record MP4", ImVec2(-1, 22))) startVideoRecording();
            ImGui::PopStyleColor(2);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Records the viewport to MP4 via ffmpeg.\nThe left panel is excluded. Click again to stop.");
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.85f,0.35f,0.10f,1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f,0.45f,0.15f,1.0f));
            if (ImGui::Button("■ Stop recording", ImVec2(-1, 22))) stopVideoRecording();
            ImGui::PopStyleColor(2);
            ImGui::TextDisabled("REC %dx%d @ %d fps", m_recW, m_recH, m_videoFps);
        }
        ImGui::SetNextItemWidth(120);
        ImGui::SliderInt("FPS##vid", &m_videoFps, 15, 60);

        // ── Replay timeline ─────────────────────────────────────────────────
        ImGui::Separator();
        ImGui::TextDisabled("Replay timeline");
        ImGui::Checkbox("Record snapshots", &m_replayRecord);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Captures positions every N frames.\n"
                              "RAM: ring buffer, oldest overwritten past Max frames, lost on close.\n"
                              "Stream to disk: each snapshot written to a folder -> UNLIMITED, never lost.");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::SliderInt("every##rep", &m_replayInterval, 5, 1000);

        bool prevStream = m_replayStream;
        ImGui::Checkbox("Stream to disk", &m_replayStream);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Writes EACH snapshot to the folder below: unlimited, never lost.\n"
                              "Resume a session with 'Load folder'.");
        if (m_replayStream && !prevStream) {           // entering stream mode = fresh recording
            m_replayDiskCount = 0; m_replayTimes.clear(); m_replayShownCursor = -1;
        }

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
            ImGui::SameLine(); ImGui::TextDisabled("%zu snaps, ~%.0f MB", m_replayFrames.size(), mb);
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
    }
    ImGui::PopID();

    ImGui::Separator();
    if (!m_markers.empty())
    {
        ImGui::TextDisabled("Markers (%zu/16)", m_markers.size());
        float velConv = (m_lyPerUnit * 9.461e12f) / 3.156e13f;
        float H_panel = (m_simulation && m_useCosmology) ? m_simulation->getHubbleParam() : 0.0f;
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
            float spdKms = 0.0f;
            if (haveVels && idx < (int)m_rotVelocities.size())
            {
                const cl_float4& v = m_rotVelocities[idx];
                const cl_float4& pos = m_positions[idx];
                float pvx = v.x - H_panel * pos.x;
                float pvy = v.y - H_panel * pos.y;
                float pvz = v.z - H_panel * pos.z;
                float spdSim = sqrtf(pvx*pvx + pvy*pvy + pvz*pvz);
                // sim_vel × lyPerUnit × 0.30 = km/s (1 ly/MY = 0.30 km/s)
                spdKms = spdSim * m_lyPerUnit * 0.30f;
            }
            bool isSelected = (m_selectedMarker == (int)i);

            char label[80];
            if (spdKms > 0.0f) {
                if (spdKms >= 1000.0f)
                    snprintf(label, sizeof(label), "#%zu: p=%d  %.1f km/s (%.2fc)",
                             i+1, idx, spdKms, spdKms / 299792.458f);
                else
                    snprintf(label, sizeof(label), "#%zu: p=%d  %.1f km/s", i+1, idx, spdKms);
            } else {
                snprintf(label, sizeof(label), "#%zu: p=%d", i+1, idx);
            }

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

        if (ImGui::Button("Apply & Generate", ImVec2(-1, 26))) {
            startSimulation();
            m_everGenerated = true;   // masque la fenêtre de bienvenue
        }
    }

    ImGui::Separator();

    {
        if (ImGui::Button(m_simPaused ? "Resume" : "Pause", ImVec2(-1, 26))) {
            if (m_simRunning) togglePauseWithClock();
            else              m_simPaused = !m_simPaused;
        }
        if (ImGui::Button("Reset", ImVec2(-1, 26)))
            resetSimulation();
    }

    SliderFloatWithInput("Sim Speed", &m_simSpeed, 0.02f, 60.0f, "%.2fx");

    ImGui::Separator();
    ImGui::Text("Frame: %d", m_currentFrame);
    ImGui::Text("Parts: %lu", m_particleCount);
    ImGui::Text("FPS: %.1f", io.Framerate);
    {
        // Frame-rate cap : presets + unlimited. 0 = no cap (uncapped GPU).
        static const int   fpsVals[]  = { 30, 60, 120, 144, 240, 0 };
        static const char* fpsNames[] = { "30", "60", "120", "144", "240", "Unlimited" };
        int sel = 1;
        for (int k = 0; k < 6; k++) if (fpsVals[k] == m_targetFps) { sel = k; break; }
        ImGui::SetNextItemWidth(110);
        if (ImGui::Combo("Max FPS", &sel, fpsNames, 6)) m_targetFps = fpsVals[sel];
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Framerate cap. Match your monitor (e.g. 240 Hz) or set\n"
                              "Unlimited. Higher also advances the sim faster (1 step/frame),\n"
                              "but uses more GPU/power. 60 = default.");
    }

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

    double elapsed_MY = m_simulation ? (double)m_simulation->getSimTime() / 100.0 : 0.0;

    if (m_useCosmology && m_H0 > 1e-6f && m_omegaL > 1e-6f) {
        // Cosmological age = initial age at aStart + elapsed cosmological time
        // t(a) = (2/3) / sqrt(OmegaL) * asinh(sqrt(OmegaL/OmegaM * a^3)) * tH_Gyr
        double tH_Gyr = 14.0;  // Hubble time in real Gyr (calibrated)
        double argS = sqrtf((double)m_omegaL / (double)m_omegaM * powf((double)m_aStart, 3.0));
        double t0_Gyr = (2.0/3.0) / sqrt((double)m_omegaL) * asinh(argS) * tH_Gyr;
        // Elapsed cosmological time = N-body time × cosmoScale
        double t_elapsed_cosmo_Gyr = elapsed_MY * (double)m_cosmoScale * (double)m_H0 * 1400.0;
        double t_total_Gyr = t0_Gyr + t_elapsed_cosmo_Gyr;
        ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.5f, 1.0f),
            "Age: %.2f Gyr", t_total_Gyr);
        ImGui::TextDisabled("T_sim=%.2f MY  (t0=%.1f Gyr)", elapsed_MY, t0_Gyr);
    } else {
        // Non-cosmological: show simulation MY directly
        if (elapsed_MY < 1000.0)
            ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.5f, 1.0f), "T = %.2f MY", elapsed_MY);
        else
            ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.5f, 1.0f), "T = %.3f Gy", elapsed_MY / 1000.0);
    }

    if (m_useCosmology && m_simulation) {
        float a = m_simulation->getScaleFactor();
        float z = (a > 1e-6f) ? (1.0f / a - 1.0f) : 0.0f;
        float H = m_simulation->getHubbleParam();
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f),
            "a=%.4f  z=%.2f  H=%.5f", a, z, H);
    }

    // Big Crunch warning in Janus pure mode when H0 is sub-critical
    if (m_eqSelectedModel == 1 && !m_useCosmology && m_simRunning) {
        float R3     = m_radius * m_radius * m_radius;
        float vol    = (4.0f / 3.0f) * 3.14159f * R3;
        float rho    = (vol > 0.f) ? (float)m_numPosiStars * m_totalSolarMass / vol : 0.f;
        float G_eff  = m_gravConstant * 1.0e10f;
        float H0c_w  = (rho > 0.f) ? sqrtf(8.0f * 3.14159f * G_eff * rho / 3.0f) : 0.f;
        if (H0c_w > 0.f && m_H0 < H0c_w * 0.95f)
            ImGui::TextColored(ImVec4(1.f, 0.3f, 0.1f, 1.f),
                "!! Big Crunch (H0c=%.4f)", H0c_w);
    }

    if (ImGui::Button("Front View", ImVec2(-1, 22)))
    {
        // Preserve current camera distance from origin so zoom is consistent.
        double dist = 2000.0;
        if (m_camera) {
            Vector* o = m_camera->getOrigin();
            if (o) {
                dist = sqrt(o->x*o->x + o->y*o->y + o->z*o->z);
                if (dist < 100.0) dist = 100.0;
            }
        }
        delete m_camera;
        m_camera = new Camera(m_renderer->getWidth(), m_renderer->getHeight(),
            new Vector(0, 0, dist), new Matrix(ROTATION, 0.0, 0.0, 0.0),
            60.0, 10.0, 1.0e10f);
        m_camera->setOrbitMode(false);
    }

    ImGui::Separator();

    ImGui::TextDisabled("Analytics");
    if (ImGui::Button(m_showDensityProfile? "Hide Density"     : "Density Profile",  ImVec2(-1,22))) m_showDensityProfile= !m_showDensityProfile;
    if (ImGui::Button(m_showPhaseSpace    ? "Hide Phase Space" : "Phase Space",      ImVec2(-1,22))) m_showPhaseSpace    = !m_showPhaseSpace;
    if (ImGui::Button(m_showEnergyPlot    ? "Hide Energy"      : "Energy",           ImVec2(-1,22))) m_showEnergyPlot    = !m_showEnergyPlot;
    if (ImGui::Button(m_showExpansion     ? "Hide Expansion"   : "Expansion", ImVec2(-1,22))) m_showExpansion = !m_showExpansion;
    if (ImGui::Button(m_showBarycenters   ? "Hide Barycenters" : "Barycenters",      ImVec2(-1,22))) m_showBarycenters   = !m_showBarycenters;
    if (ImGui::Button(m_showDarkMatter    ? "Hide DM Profile"  : "DM Equivalent",    ImVec2(-1,22))) m_showDarkMatter    = !m_showDarkMatter;
    if (ImGui::Button(m_showGalaxyHistory ? "Hide Galaxy History" : "Galaxy History", ImVec2(-1,22))) m_showGalaxyHistory = !m_showGalaxyHistory;
    ImGui::Separator();
    ImGui::TextDisabled("Scientific Analysis");
    if (ImGui::Button(m_showConservation ? "Hide Conservation" : "Conservation",     ImVec2(-1,22))) m_showConservation = !m_showConservation;
    if (ImGui::Button(m_showEscape       ? "Hide Neg. Vel."   : "Neg. Mass Velocity",ImVec2(-1,22))) m_showEscape       = !m_showEscape;
    if (ImGui::Button(m_showCorrelation  ? "Hide xi(r)"       : "Correlation xi(r)",ImVec2(-1,22))) m_showCorrelation  = !m_showCorrelation;
    if (ImGui::Button(m_showHaloFinder   ? "Hide Halo Finder" : "Halo Finder",      ImVec2(-1,22))) m_showHaloFinder   = !m_showHaloFinder;
    if (ImGui::Button(m_showPowerSpectrum? "Hide P(k)"        : "Power Spectrum P(k)",ImVec2(-1,22))) m_showPowerSpectrum= !m_showPowerSpectrum;
    if (ImGui::Button(m_showEquationEditor? "Hide Eq. Editor"  : "Equation Editor",  ImVec2(-1,22))) m_showEquationEditor= !m_showEquationEditor;
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
    if (ImGui::Button("Energy##csv",      ImVec2(hw, 22))) exportEnergyCSV();
    ImGui::SameLine(0, 4);
    if (ImGui::Button("Density##csv",     ImVec2(hw, 22))) exportDensityCSV();
    if (ImGui::Button("Barycenters##csv", ImVec2(-1, 22))) exportBarycentersCSV();

    ImGui::Separator();
    if (ImGui::Button("Reset Cam", ImVec2(-1, 22)))
    {
        m_edgeOnView = false;
        delete m_camera;
        m_camera = new Camera(m_renderer->getWidth(), m_renderer->getHeight(),
            new Vector(0, 0, 2000.0), new Matrix(ROTATION, 0.0, 0.0, 0.0),
            60.0, 10.0, 1.0e10f);
        m_camera->setOrbitMode(false);
    }

    ImGui::End();
    ImGui::PopStyleVar(2);

    if (m_simRunning) {
        renderGrid();
        renderFourierModes();
        renderMarkerPaths();
        renderMarkers();
        renderGalaxyMarkers();
        renderFoFOverlay();
    }
    // Analytics WINDOWS render unconditionally (self-guarded by their m_show flags)
    // so they appear in front of the startup screen, like the other panels.
    renderBarycenters();
    renderPowerSpectrum();
    renderDarkMatterProfile();
    renderGalaxyHistory();
    renderAxisGizmo();
    renderRotationCurve();
    renderDensityProfile();
    renderPhaseSpace();
    renderEnergyPlot();
    renderExpansion();
    renderDispersionPlot();
    renderEquationEditor();
    renderConservation();
    renderEscapeVelocity();
    renderCorrelation();
    renderSersic();
    renderHaloFinder();
    renderAbout();

    {
        const float pW = 340.0f;
        const float vX = mvX + pW;
        const float vW = (float)m_width - pW;

        double eMY = m_simulation ? (double)m_simulation->getSimTime() / 100.0 : 0.0;

        char timeBuf[64];
        if (m_useCosmology && m_omegaL > 1e-6f && m_omegaM > 1e-6f) {
            // Âge cosmologique RÉEL depuis le facteur d'échelle a intégré par Friedmann
            // -> t(a) analytique. Robuste au cosmoScale (plus de divergence Age vs débit).
            double tH_Gyr = 14.0;
            double a_now = m_simulation ? (double)m_simulation->getScaleFactor() : (double)m_aStart;
            double argA = sqrt((double)m_omegaL / (double)m_omegaM * a_now*a_now*a_now);
            m_curAgeGyr = (2.0/3.0) / sqrt((double)m_omegaL) * asinh(argA) * tH_Gyr;
            snprintf(timeBuf, sizeof(timeBuf), "Age : %.2f Gyr", m_curAgeGyr);
        } else if (eMY < 1000.0)
            snprintf(timeBuf, sizeof(timeBuf), "T : %.2f MY", eMY);
        else
            snprintf(timeBuf, sizeof(timeBuf), "T : %.3f Gy", eMY / 1000.0);

        // Record the galaxy-formation history (for the JWST comparison plot).
        if (m_simRunning && !m_simPaused) {
            bool cosmo = m_useCosmology && m_omegaL > 1e-6f && m_omegaM > 1e-6f;
            float histAge = cosmo ? (float)m_curAgeGyr : (float)(eMY / 1000.0);
            if (m_galHistory.empty() || histAge - m_galHistory.back().ageGyr >= 0.005f) {
                m_galHistory.push_back({ histAge, m_galaxyCount });
                if (m_galHistory.size() > 40000) m_galHistory.erase(m_galHistory.begin());
            }
        }

        // Débit COHÉRENT avec l'affichage : cosmologie -> d(Age)/dt_réel (Gyr/s) ;
        // sinon -> temps N-corps (MY/s). Plus d'incohérence Age vs débit.
        Uint32 nowMs = SDL_GetTicks();
        if (nowMs - m_mySecLastRealMs >= 1000 && m_simRunning && !m_simPaused)
        {
            float dtSec = (nowMs - m_mySecLastRealMs) / 1000.0f;
            if (m_useCosmology) m_simMyPerSec = (float)((m_curAgeGyr - m_lastAgeGyr) / dtSec);
            else                m_simMyPerSec = (float)((eMY - m_mySecLastSimTime) / dtSec);
            m_lastAgeGyr        = m_curAgeGyr;
            m_mySecLastSimTime  = eMY;
            m_mySecLastRealMs   = nowMs;
        }

        // Physics model name
        const char* modelNames[] = {"Newton","Janus","Yukawa","MOND","Custom"};
        char modelBuf[32];
        if (m_useCosmology)
            snprintf(modelBuf, sizeof(modelBuf), "Janus bimetric model");
        else if (m_eqSelectedModel >= 0 && m_eqSelectedModel <= 4)
            snprintf(modelBuf, sizeof(modelBuf), "%s", modelNames[m_eqSelectedModel]);
        else
            snprintf(modelBuf, sizeof(modelBuf), "?");

        // Wall-clock elapsed time (excluding paused periods)
        uint64_t wallMs = m_wallClockAccumMs;
        if (!m_simPaused && m_wallClockStartMs > 0)
            wallMs += SDL_GetTicks64() - m_wallClockStartMs;
        uint64_t totalSec = wallMs / 1000;
        char wallBuf[32];
        if (totalSec < 3600) {
            // < 1 heure : format MM:SS
            snprintf(wallBuf, sizeof(wallBuf), "%llu:%02llu",
                     (unsigned long long)(totalSec / 60),
                     (unsigned long long)(totalSec % 60));
        } else {
            // ≥ 1 heure : format Hh MMm SSs
            snprintf(wallBuf, sizeof(wallBuf), "%lluh %02llum %02llus",
                     (unsigned long long)(totalSec / 3600),
                     (unsigned long long)((totalSec % 3600) / 60),
                     (unsigned long long)(totalSec % 60));
        }

        char overlayBuf[192];
        const char* rateUnit = m_useCosmology ? "Gyr/s" : "MY/s";
        if (m_simMyPerSec != 0.0f)
            snprintf(overlayBuf, sizeof(overlayBuf), "%s  |  %s  |  %.3f %s  |  Elapsed %s",
                     timeBuf, modelBuf, m_simMyPerSec, rateUnit, wallBuf);
        else
            snprintf(overlayBuf, sizeof(overlayBuf), "%s  |  %s  |  Elapsed %s",
                     timeBuf, modelBuf, wallBuf);

        // Background draw list : le HUD reste AU-DESSUS de la scène 3D (image ajoutée plus
        // tôt) mais passe SOUS les fenêtres d'options (foreground les mettait par-dessus).
        ImDrawList* fg = ImGui::GetBackgroundDrawList(ImGui::GetMainViewport());
        if (m_simRunning) {
            ImVec2 textSize = ImGui::CalcTextSize(overlayBuf);
            float  tx = vX + vW - textSize.x - 10.0f;
            float  ty = mvY + 8.0f;
            fg->AddRectFilled(
                ImVec2(tx - 4, ty - 2),
                ImVec2(tx + textSize.x + 4, ty + textSize.y + 2),
                IM_COL32(0, 0, 0, 140));
            fg->AddText(ImVec2(tx, ty), IM_COL32(220, 220, 100, 255), overlayBuf);
        }

        // Galaxy counter — top left, BELOW the grid-size label.
        // First-detection time is converted to the SAME cosmological age scale
        // used by the Age display in the top-right corner (t0 + eMY × cosmoScale × H0 × 1400),
        // so both values share a common temporal reference.
        float gyHud = mvY + 30.f;
        if (m_galaxyCount > 0)
        {
            char galBuf[96];
            if (m_firstGalaxyDetected && m_firstGalaxyTime > 0.0f) {
                // Map m_firstGalaxyTime (sim MY) to cosmological age in Gyr using
                // the same formula as the Age display.
                double firstGyr = 0.0;
                if (m_useCosmology && m_H0 > 1e-6f && m_omegaL > 1e-6f) {
                    double tH_Gyr = 14.0;
                    double argS = sqrtf((double)m_omegaL / (double)m_omegaM
                                        * powf((double)m_aStart, 3.0));
                    double t0   = (2.0/3.0) / sqrt((double)m_omegaL) * asinh(argS) * tH_Gyr;
                    firstGyr = t0 + (double)m_firstGalaxyTime
                                  * (double)m_cosmoScale * (double)m_H0 * 1400.0;
                    snprintf(galBuf, sizeof(galBuf), "Galaxies: %d  (>= %d, 1st: %.3f Gyr)",
                             m_galaxyCount, m_galaxyMinPart, firstGyr);
                } else {
                    // Non-cosmological fallback: raw MY/Gyr based on magnitude
                    if (m_firstGalaxyTime >= 1000.0f)
                        snprintf(galBuf, sizeof(galBuf), "Galaxies: %d  (>= %d, 1st: %.3f Gyr)",
                                 m_galaxyCount, m_galaxyMinPart, m_firstGalaxyTime / 1000.0f);
                    else
                        snprintf(galBuf, sizeof(galBuf), "Galaxies: %d  (>= %d, 1st: %.2f MY)",
                                 m_galaxyCount, m_galaxyMinPart, m_firstGalaxyTime);
                }
            } else {
                snprintf(galBuf, sizeof(galBuf), "Galaxies: %d", m_galaxyCount);
            }
            ImVec2 gs = ImGui::CalcTextSize(galBuf);
            float  gx = vpX + 8.f;
            fg->AddRectFilled(ImVec2(gx-4,gyHud-2), ImVec2(gx+gs.x+4,gyHud+gs.y+2), IM_COL32(0,0,0,160));
            fg->AddText(ImVec2(gx, gyHud), IM_COL32(100, 255, 160, 255), galBuf);
            gyHud += gs.y + 4.f;
        }
        // FoF cluster counter (live mode) — below the galaxy counter
        if (m_fofLive || m_fofDone)
        {
            char clBuf[64];
            snprintf(clBuf, sizeof(clBuf), "Clusters: %d", (int)m_fofClusters.size());
            ImVec2 cs = ImGui::CalcTextSize(clBuf);
            float  cx = vpX + 8.f;
            fg->AddRectFilled(ImVec2(cx-4,gyHud-2), ImVec2(cx+cs.x+4,gyHud+cs.y+2), IM_COL32(0,0,0,160));
            fg->AddText(ImVec2(cx, gyHud), IM_COL32(255, 210, 90, 255), clBuf);
        }
    }
}


void App::run()
{
    bool running = true;

    while (running)
    {
        Uint32 frameStart = SDL_GetTicks();
        
        processEvents(running);
        updateCamera();
        update3DMouse();    // navigation 3Dconnexion (SpaceMouse) si activée
        updateAutoFit();    // auto-cadrage fluide (suit l'expansion)

        // Hide cursor in viewport during standby (it shows as a "particle" on dark bg)
        if (!m_everGenerated) {
            int mx, my;
            SDL_GetMouseState(&mx, &my);
            SDL_ShowCursor(mx > 340 ? SDL_DISABLE : SDL_ENABLE);
        } else {
            SDL_ShowCursor(SDL_ENABLE);
        }

        // Batch mode state machine (auto-drives generation/run/save cycle)
        tickBatchMode();

        if (m_replayMode)
            tickReplay();                       // scrubbing recorded frames (sim frozen)
        else if (m_simRunning && !m_simPaused)
            stepSimulation();

        // En mode SPH, le rendu est déjà lumineux : on coupe le glow post-traitement
        // (sinon il AMPLIFIE la structure très brillante en vue rasante -> traînée énorme).
        m_renderer->setGlow(m_softHalo ? false : m_glowEnabled, m_blurRadius, m_glowIntensity);
        m_renderer->setSoftHalo(m_softHalo);
        m_renderer->setSPHScale(m_sphQuality);   // splat resolution (speed/quality)
        m_renderer->beginFrame();
        float viewProj[16];
        m_camera->getViewProjMatrix(viewProj);
        glm::vec3 posPal[32];
        glm::vec3 negPal[32];
        for (int i = 0; i < 32; i++)
        {
            const float* pp = m_colorByTemp ? m_tempPalette[i]
                            : (m_colorByWeb ? m_webPalette[i]
                            : ((m_colorByHalo || m_colorByDensity) ? m_densPalette[i]
                            : (m_colorByVelocity ? m_velPalette[i] : m_posPalette[i])));
            const float* np = m_negPalette[i];
            posPal[i] = glm::vec3(pp[0], pp[1], pp[2]);
            negPal[i] = glm::vec3(np[0], np[1], np[2]);
        }
        // Mode SPH densité : la teinte encode la densité -> palette dédiée (viridis ou thermique).
        const bool sphViridis = m_softHalo && m_sphViridis;
        if (sphViridis) {
            const auto& pal = (m_sphPalette == 2) ? m_galaxyPalette
                            : (m_sphPalette == 1) ? m_thermalPalette
                                                  : m_viridisPalette;
            for (int i = 0; i < 32; i++)
                posPal[i] = glm::vec3(pal[i][0], pal[i][1], pal[i][2]);
        }
        // Color by Temperature / Halo : tint the NEGATIVE species PINK so it stays
        // distinguishable (otherwise it shares the blue negative palette / value 0 and
        // blends with cold or halo-less regions). Ramp starts at a visible pink so
        // value-0 negatives (no halo) still read clearly.
        if (m_colorByTemp || m_colorByHalo) {
            for (int i = 0; i < 32; i++) {
                float t = i / 31.0f;
                negPal[i] = glm::mix(glm::vec3(0.55f, 0.10f, 0.32f),
                                     glm::vec3(1.0f, 0.45f, 0.85f), t);
            }
        }
        // Distance caméra pour la taille des points. En ORBITE, utiliser la VRAIE
        // distance au centre (sqrt(x²+y²+z²)) : sinon |z| s'annule quand on orbite dans
        // le plan z=0 (vue rasante) -> taille = physSize/≈0 -> particules ÉNORMES (bande).
        float cam_z;
        {
            const Vector* o = m_camera->getOrigin();
            if (m_camera->isEdgeOn())
                cam_z = fabsf((float)o->y);
            else if (m_camera->isOrbitMode())
                cam_z = sqrtf((float)(o->x*o->x + o->y*o->y + o->z*o->z));
            else
                cam_z = fabsf((float)o->z);
        }
        float vpWf       = (float)(m_width - 340);
        float wuPerPx    = 2.0f * cam_z * tanf(60.0f * 3.14159265f / 360.0f) / vpWf;
        float softeningRef = (m_softening > 0.f) ? std::min(m_softening, 20.0f) : 7.0f;
        float physSize   = m_particleSize    * softeningRef;
        float ptSize     = physSize / wuPerPx;
        ptSize = std::min(std::max(ptSize, 0.5f), 64.0f);

        float physSizeNeg = m_particleSizeNeg * softeningRef;
        float ptSizeNeg   = physSizeNeg / wuPerPx;
        ptSizeNeg = std::min(std::max(ptSizeNeg, 0.5f), 64.0f);

        float maxVal = m_colorByTemp ? m_tempMaxForColor
                     : (m_colorByWeb ? 4.0f
                     : (m_colorByHalo ? m_haloMaxForColor
                     : (m_colorByDensity ? m_densMaxForColor
                     : (m_colorByVelocity ? m_velMaxForColor : (float)Color::max_mass))));

        // Plans de coupe X/Y/Z : enable/pos/sign par axe (masque si un axe actif coupe).
        {
            float en[3]={0,0,0}, pos[3]={0,0,0}, sign[3]={1,1,1};
            if (m_clipEnable) for (int a=0; a<3; a++) {
                en[a]   = m_clipAxisOn[a] ? 1.0f : 0.0f;
                pos[a]  = m_clipCenter[a] + m_clipAxisFrac[a] * m_clipExtent;  // centré sur le barycentre
                sign[a] = m_clipAxisFlip[a] ? -1.0f : 1.0f;
            }
            m_renderer->setClipBox(en, pos, sign);
        }

        if (m_softHalo) {
            // Rendu SPH volumétrique : splat noyau adaptatif -> propriété moyenne (teinte)
            // + densité de colonne (luminosité log). Look py-sphviewer/Illustris.
            m_renderer->renderSPH(glm::make_mat4(viewProj), ptSize, m_showNegaMass,
                                  posPal, maxVal, m_sphBright, sphViridis,
                                  (float)m_camera->getFov(), m_sphSmooth, m_sphCoreDetail);
        } else {
            m_renderer->render(glm::make_mat4(viewProj), ptSize, ptSizeNeg,
                               posPal, negPal, maxVal, m_showNegaMass, 0.0f);
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
        renderAnnotations();   // screen-space overlay (foreground), drawn last
        ImGui::Render();

        int displayW, displayH;
        SDL_GL_GetDrawableSize(m_window, &displayW, &displayH);
        glViewport(0, 0, displayW, displayH);
        glClearColor(0.031f, 0.063f, 0.188f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        captureVideoFrame();   // stream this frame to ffmpeg if recording
        captureScreenshotNow();// deferred PNG (now back buffer has overlays/annotations)

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

        // Frame limiter : m_targetFps <= 0 -> unlimited (no cap).
        if (m_targetFps > 0) {
            Uint32 frameDelay = 1000u / (Uint32)m_targetFps;
            Uint32 frameTime  = SDL_GetTicks() - frameStart;
            if (frameTime < frameDelay)
                SDL_Delay(frameDelay - frameTime);
        }
    }
}