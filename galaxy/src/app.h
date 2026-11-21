#ifndef APP_H
# define APP_H

# include <SDL.h>
# include <GL/glew.h>
# include <imgui.h>
# include <imgui_impl_sdl2.h>
# include <imgui_impl_opengl3.h>
# include <string>
# include <vector>
# include <thread>
# include <atomic>
# include <mutex>

# include "point_renderer.h"
# include "volume_renderer.h"
# include "camera.h"
# include "color.h"
# include "vector.h"
# include "matrix.h"
# include "simulation.h"
# include "galaxy_generator.h"

class App
{
public:
    App(int w, int h);
    ~App();

    bool init();
    void run();

private:
    void processEvents(bool& running);
    void updateCamera();
    void updateAutoFit();   // auto-fit: smoothly pulls the camera back to follow the expansion
    void renderGUI();
    void uploadParticlesToGPU(bool subData = false);
    void startSimulation();
    void resetSimulation();
    void stepSimulation();
    void saveSettings(const std::string& path);
    void loadSettings(const std::string& path);
    void renderGrid();
    void renderVelocityField();      // #10 streamlines overlay
    bool  m_showVelocityField;
    float m_velFieldScale;           // arrow length multiplier
    bool  m_velFieldRelative;        // subtract per-species bulk velocity
    bool  m_velSelectMode;           // left-drag selects a region (vs marker pick)
    bool  m_velSelActive;            // a captured particle set exists (follows stars)
    bool  m_velSelDragging;          // drag in progress
    bool  m_velSelPending;           // box drawn, indices need capturing next frame
    float m_velSelX0, m_velSelY0;    // selection rect, window-relative coords
    float m_velSelX1, m_velSelY1;
    std::vector<int> m_velSelIndices; // captured particle indices (follow the stars)

    // ── Annotations (screen-space 2D overlay for annotated screenshots) ───────
    void renderAnnotations();
    struct Annotation {
        int   type;          // 1=arrow, 2=text, 3=rect, 4=circle
        float x0, y0, x1, y1;// window-relative coords (two points)
        char  text[128];
        unsigned int col;
    };
    std::vector<Annotation> m_annotations;
    int   m_annotTool;       // 0=none,1=arrow,2=text,3=rect,4=circle
    int   m_annotStage;      // 0=idle (waiting 1st point), 1=have 1st point
    float m_annotX0, m_annotY0;   // first point, window-relative
    float m_annotColor[3];   // current annotation color (RGB 0..1)
    int   m_annotEditIdx;    // annotation index whose text is being edited (-1=none)
    char  m_annotTextBuf[128];

    // Deferred screenshot: capture AFTER ImGui render so overlays/annotations show
    bool  m_screenshotPending;
    char  m_screenshotPath[512];
    void  captureScreenshotNow();    // viewport-only PNG, called at end of frame

    // ── MP4 video recording (viewport → ffmpeg pipe) ─────────────────────────
    bool  m_videoRecording = false;
    void* m_ffmpegStdin = nullptr;   // HANDLE (write end of pipe to ffmpeg)
    void* m_ffmpegProc  = nullptr;   // HANDLE to ffmpeg process
    int   m_recW = 0, m_recH = 0;    // locked viewport size for the recording
    int   m_videoFps = 30;
    char  m_videoPath[512] = "";
    std::vector<unsigned char> m_recBuf;     // scratch RGB frame buffer
    std::string findFFmpeg();                // locate ffmpeg.exe
    bool  startVideoRecording();             // opens dialog, spawns ffmpeg
    void  captureVideoFrame();               // end-of-frame: read viewport → pipe
    void  stopVideoRecording();              // close pipe, finalize MP4

    // ── Replay timeline (RAM ring buffer of position snapshots) ──────────────
    bool  m_replayRecord = false;            // capture snapshots while simulating
    int   m_replayInterval = 50;             // record every N frames
    int   m_replayMaxFrames = 120;           // ring-buffer cap
    bool  m_replayMode = false;              // true = scrubbing (frozen sim)
    bool  m_replayPlaying = false;           // auto-advance the cursor
    int   m_replayCursor = 0;                // current snapshot index
    float m_replayPlaySpeed = 1.0f;
    int   m_replayLastRecFrame = -1000000;   // frame of last snapshot
    std::vector<std::vector<cl_float4>> m_replayFrames;  // snapshots (positions)
    std::vector<float> m_replayTimes;        // sim time (MY) per snapshot
    void  replayRecordSnapshot();            // push current positions to buffer
    char  m_replayFilePath[256] = "C:\\Users\\Administrateur\\Desktop\\replay.jrp";
    void  saveReplay();   // persist the RAM ring-buffer snapshots to a single file
    void  loadReplay();
    // Streaming-to-disk replay: each snapshot written straight to a folder
    // (unlimited length, never lost). m_replayFrames stays empty in this mode.
    bool  m_replayStream = false;
    char  m_replayDir[512] = "D:\\SIMULATION\\replay";
    int   m_replayDiskCount = 0;
    int   m_replayShownCursor = -1;
    void  replayWriteDiskFrame();
    bool  replayReadDiskFrame(int idx);
    void  replayScanDiskFolder();
    int   replayCount() const;
    void  replaySetFrame(int idx);
    void  tickReplay();                      // scrub/playback recorded frames

    void renderRotationCurve();
    void renderDensityProfile();
    void renderPhaseSpace();
    void renderEnergyPlot();
    void refreshAnalytics();

    int m_width, m_height;
    SDL_Window* m_window;
    SDL_GLContext m_glContext;
    Uint8* m_keystate;

    PointRenderer* m_renderer;
    Camera* m_camera;
    Simulation* m_simulation;

    float m_radius;
    float m_scaleUnit;
    float m_timeStep;
    int   m_numNegaStars;
    int   m_numPosiStars;
    float m_negaDensity;
    float m_totalSolarMass;
    float m_blackHoleMass;
    float m_rotSpeed;
    float m_initSpeed;
    float m_softening;
    float m_gravConstant;

    // GPU Barnes-Hut tree (two-trees Janus, O(N log N))
    bool  m_useTree;
    float m_treeTheta;
    bool  m_treeQuad = true;   // order-2 multipole correction
    bool  m_treeWarp = true;   // warp-coherent traversal

    bool  m_simRunning;
    bool  m_simPaused;
    int   m_currentFrame;
    int   m_totalFrames;
    float m_simSpeed;
    int   m_skipCounter;

    std::vector<cl_float4> m_positions;
    std::vector<cl_float4> m_velocities;
    std::vector<float> m_particleData;

    // ── Auto-fit: the camera smoothly pulls back to keep the
    //    structure at constant apparent size while it expands. ──
    bool  m_autoFit           = false;  // enables automatic expansion tracking
    float m_autoFitFill       = 0.55f;  // fraction of the screen half-height occupied by the structure
    float m_autoFitSmoothTime = 0.6f;   // SmoothDamp smoothing time (s) — larger = softer
    int   m_autoFitInterval   = 15;     // recompute the structure radius every N frames
    int   m_autoFitLastFrame  = -100000;// last frame where the radius was recomputed
    float m_autoFitTargetRaw  = 0.0f;   // RAW target distance (last P90 computation, stepped)
    float m_autoFitTargetDist = 0.0f;   // SMOOTHED target (continuous low-pass of the raw target)
    float m_autoFitDistVel    = 0.0f;   // SmoothDamp internal velocity (C1 continuity -> no jolt)
    float m_autoFitRadius     = 0.0f;   // robust structure radius (P90), for display
    unsigned int m_autoFitLastTicks = 0;// timestamp for real dt (framerate-independent smoothing)
    unsigned long m_particleCount;

    float m_posPalette[32][3];
    float m_negPalette[32][3];
    float m_velPalette[32][3];   // jet: blue->cyan->yellow->red (velocity)

    float        m_spAmplitude;
    float        m_spPatternSpeed;
    float        m_spPitchAngle;
    int          m_spArms;
    bool         m_densityWavesEnabled;

    bool  m_glowEnabled;
    float m_blurRadius;
    float m_glowIntensity;

    bool  m_showGrid;
    float m_lyPerUnit;

    bool m_showRotCurve;
    bool m_showDensityProfile;
    bool m_showPhaseSpace;
    bool m_showEnergyPlot;
    bool m_rotAutoRefresh;
    int  m_rotRefreshInterval;
    int  m_lastRefreshFrame;
    std::vector<cl_float4> m_rotVelocities;
    float m_rotCurveRMax;  // max radius for rotation curve (in LY, 0 = auto = 3×radius)

    static const int ENERGY_HISTORY = 512;
    float  m_energyHistory[ENERGY_HISTORY];
    double m_energyTimes  [ENERGY_HISTORY];
    int    m_energyHead;
    int    m_energyCount;

    bool  m_colorByVelocity;
    bool  m_velAutoScale;       // auto-scale velocity color range at each refresh
    float m_velMaxForColor;
    float m_particleSize;
    float m_particleSizeNeg;

    // ── Soft Halo (volumetric SPH rendering, py-sphviewer style) ─────────────
    bool  m_softHalo = false;       // enables SPH rendering (otherwise points)
    float m_sphBright = 0.30f;      // brightness (column density)
    float m_sphSmooth = 2.0f;       // blur radius (px) -> continuous filaments
    float m_sphCoreDetail = 40.0f;  // asinh ceiling (40 = white cores .. 600 = internal detail)
    float m_sphQuality = 1.0f;      // splat resolution (1 = full, <1 = faster)
    std::vector<float> m_smoothLength;  // per-particle SPH smoothing length (∝ density^-1/3)
    std::vector<float> m_density;       // local density (not used for color here)
    float m_densMaxForColor = 1.0f;
    int   m_sphTick = 0;                // density recomputation throttle
    float m_galaxyPalette[32][3];       // "true galaxy colors" palette (blue->cream->gold)
    float m_nebulaPalette[32][3];       // nebula palette (diffuse cold -> cream core) for the RT
    void  computeLocalDensity();        // scheduler (adopts / launches the worker)

    // ── Ray-traced volume (Graphic Tools option: ray-march of a 3D grid) ──
    bool  m_rayTrace = false;
    VolumeRenderer m_volume;
    bool  m_volumeReady = false;
    float m_volHaloSize = 6.0f;         // halo splat radius (x softening, -> pixels)
    int   m_volGridN   = 128;           // grid resolution (64/96/128/192)
    int   m_volSteps   = 400;           // marching steps (quality/anti-banding)
    float m_volGain    = 0.08f;         // splat intensity
    float m_volContrast = 4.0f;         // density reference = N x mean (high = more dynamic range)
    float m_volAbsorb  = 1.0f;          // self-occlusion (front -> back)
    float m_volExposure = 1.6f;
    float m_volSmooth   = 1.5f;         // 3D Gaussian blur intensity (grain control, continuous)
    bool  m_volGalaxyPalette = true;    // density palette dedicated to the ray-traced view (nebula)
    bool  m_volColorByVel = true;       // color the volume by velocity (otherwise density)
    bool  m_volStars = true;            // overlays the sharp particles (stars) on the halo
    float m_volStarScale = 1.5f;        // star-point size (pixels)
    float m_volStarBright = 0.7f;       // star brightness
    float m_volBoxSigma = 3.0f;         // box half-size (fixed, not exposed)
    float m_volBboxMin[3] = {-1,-1,-1};
    float m_volBboxMax[3] = { 1, 1, 1};
    void  computeLocalDensityWorker();  // 64³ grid + smoothing, on a background thread
    std::thread            m_densThread;
    std::atomic<bool>      m_densComputing{false};
    std::atomic<bool>      m_densReady{false};
    std::vector<cl_float4> m_densPosCopy;
    std::vector<float>     m_densityBack, m_smoothLengthBack;
    float                  m_densMaxBack = 1.0f;
    uint64_t               m_densLastLaunchMs = 0;
    int                    m_densIntervalMs = 150;

    // ── CUDA-OpenGL interop : the render vertex buffer is built on the GPU directly
    //    into the mapped VBO (no CPU readback->rebuild->upload round-trip). ──
    bool   m_useInterop = true;
    bool   m_interopOk  = false;
    void*  m_interopRes = nullptr;     // cudaGraphicsResource for the VBO
    size_t m_interopVboCount = 0;      // registered VBO capacity (vertices)
    void*  m_dSmooth = nullptr;        // device SPH smoothing-length buffer
    int    m_dSmoothCount = 0;
    bool   m_smoothDirty = false;      // smoothLength changed -> re-upload to device
    int    m_velMaxTick = 0;           // interop: frames until the host p98 refresh (10)
    int    m_hostReadTick = 0;         // frames until the next GPU->CPU particle copy
    void   readHostParticlesIfNeeded();
    bool   uploadParticlesInterop();   // returns true if it built the VBO via interop

    void saveScreenshot();
    bool m_isGenerating;        // galaxy generation in progress
    int  m_generateCountdown;   // >0 = deferred generate (2=wait, 1=show text, 0=launch)

    // ── Phase Space options ───────────────────────────────────────────────
    bool m_phaseShowNeg;        // show negative mass in phase space

    // ── Presets ───────────────────────────────────────────────────────────
    void savePreset(const std::string& name);
    void loadPreset(const std::string& name);
    std::vector<std::string> listPresets();
    char m_presetName[128];

    void saveSimState();
    void saveSimStateTo();   // "Save As..." dialog box
    void loadSimState();
    char  m_stateFilePath[256];
    bool  m_needReshare;

    void renderDispersionPlot();
    bool  m_showDispersion;

    bool m_showNegaMass;
    bool m_edgeOnView;

    // ── 3D Stereo ─────────────────────────────────────────────────────────
    bool m_stereoEnabled;
    int  m_stereoMode;    // 0=mono, 1=anaglyphe, 2=sbs
    float m_ipd;          // inter-pupillary distance (mm)
    bool m_rightMouseDown;
    bool m_middleMouseDown = false;   // middle mouse button held -> PAN (translation)
    int  m_lastMouseX, m_lastMouseY;

    // ── Barycenters ──────────────────────────────────────────────────────────
    void renderBarycenters();
    bool m_showBarycenters;
    static const int BARY_HISTORY = 512;
    float  m_baryDist  [BARY_HISTORY]; // distance |G+ - G-| in LY
    float  m_baryPosMag[BARY_HISTORY]; // |G+| from origin in LY (drift +)
    float  m_baryNegMag[BARY_HISTORY]; // |G-| from origin in LY (drift −)
    float  m_baryTimes [BARY_HISTORY];
    int    m_baryHead, m_baryCount;
    float  m_baryPosX, m_baryPosY, m_baryPosZ;    // current positive center of mass (sim units)
    float  m_baryNegX, m_baryNegY, m_baryNegZ;    // current negative center of mass
    float  m_baryPosVx, m_baryPosVy, m_baryPosVz; // COM+ systemic velocity (bulk drift to subtract)
    float  m_baryNegVx, m_baryNegVy, m_baryNegVz; // COM− systemic velocity

    // ── Fourier modes ────────────────────────────────────────────────────────
    void renderFourierModes();
    bool m_showFourier;
    static const int FOURIER_HISTORY = 512;
    float  m_fourierA[4][FOURIER_HISTORY]; // m=1,2,3,4
    float  m_fourierTimes[FOURIER_HISTORY];
    int    m_fourierHead, m_fourierCount;

    // ── Equivalent dark-matter profile ───────────────────────────────────────
    void renderDarkMatterProfile();
    bool  m_showDarkMatter;
    float m_nfwRs;    // NFW characteristic radius (sim units)
    float m_nfwRho0;  // NFW central density (auto-normalized)

    // ── Scientific analytics ─────────────────────────────────────────────────
    void renderConservation();   // Angular momentum + total energy
    void renderEscapeVelocity(); // Negative mass average speed history
    void renderCorrelation();    // Two-point correlation ξ(r)
    void renderSersic();         // Surface brightness + Sersic fit

    bool  m_showConservation;
    bool  m_showEscape;
    bool  m_showCorrelation;
    bool  m_showSersic;

    // ── FoF halo finder (#3) + virial ratio per cluster (#4) ─────────────────
    void renderHaloFinder();
    void computeFoF();           // runs on a worker thread (reads snapshot buffers)
    bool  m_showHaloFinder;
    float m_fofLinkingScale;     // linking length = scale × mean inter-particle sep (default 0.2)
    int   m_fofMinMembers;       // ignore groups smaller than this (default 20)
    bool  m_fofIncludeNeg;       // run on negative species instead of positive
    bool  m_fofOverlay;          // draw cluster circles on the viewport
    struct FoFCluster {
        float cx, cy, cz;        // center of mass (sim units)
        float vx, vy, vz;        // bulk velocity
        float mass;              // total |mass|
        int   count;             // member particles
        float radius;            // rms radius (sim units)
        float virial;            // KE / |PE|  (0.5 = virialized)
        std::vector<int> members;// global particle indices (live tracking + highlight)
    };
    std::vector<FoFCluster> m_fofClusters;
    bool  m_fofDone;
    int   m_fofSelected = -1;    // selected cluster row (-1 = none)
    // Live mode: Run toggles a persistent background loop (button red), the worker
    // re-detects every N frames so the catalog/overlay update on their own.
    bool  m_fofLive = false;
    int   m_fofLiveInterval = 50;
    int   m_fofLiveLastFrame = -100000;
    // Background-thread execution (keeps UI responsive at millions of particles)
    std::thread             m_fofThread;
    std::atomic<bool>       m_fofComputing{false};
    std::atomic<bool>       m_fofResultReady{false};
    std::atomic<bool>       m_fofCancel{false};          // request worker abort
    std::vector<cl_float4>  m_fofPosCopy, m_fofVelCopy;  // worker-owned snapshots
    std::vector<FoFCluster> m_fofResult;                 // worker output
    float m_fofGsnap, m_fofSoftSnap, m_fofLinkSnap;      // param snapshots
    int   m_fofMinSnap;
    bool  m_fofNegSnap;
    void  exportFoFCSV();        // save cluster catalog to CSV

    // Stored correlation result (computed on demand, not every frame)
    std::vector<float> m_corrXi;
    bool               m_corrDone;
    float              m_corrRMax;

    static const int CONS_HISTORY = 512;
    float  m_angMomHistory[CONS_HISTORY];  // |L_z| per unit mass
    float  m_totEHistory  [CONS_HISTORY];  // KE + PE_estimated
    double m_consTimesH   [CONS_HISTORY];
    int    m_consHead, m_consCount;

    static const int ESC_HISTORY = 512;
    float  m_negVelHistory[ESC_HISTORY];   // <|v_neg|> average speed
    float  m_negAccHistory[ESC_HISTORY];   // d<|v_neg|>/dt  (MY⁻¹ units)
    float  m_escTimes     [ESC_HISTORY];
    int    m_escHead, m_escCount;
    float  m_prevNegVel;
    float  m_prevEscTime;

    // ── About dialog ─────────────────────────────────────────────────────────
    void renderAbout();
    bool m_showAbout;

    // ── Reset to defaults ────────────────────────────────────────────────────
    void resetToDefaults();

    // ── CSV export ───────────────────────────────────────────────────────────
    void exportRotationCurveCSV();
    void exportEnergyCSV();
    void exportDensityCSV();
    void exportBarycentersCSV();
    void exportFourierCSV();

    // ── Simulation speed (MY/sec) ─────────────────────────────────────────────
    float  m_simMyPerSec;
    double m_mySecLastSimTime;
    Uint32 m_mySecLastRealMs;

    // ── Real wall-clock run time (accumulated while running) ──────────────────
    double m_simWallSec  = 0.0;
    Uint32 m_wallPrevMs  = 0;

    // ── Equation Editor ──────────────────────────────────────────────────────
    void renderEquationEditor();
    bool  m_showEquationEditor;
    static const int EQ_CODE_SIZE = 16384;
    char  m_eqCodeBuffer[EQ_CODE_SIZE];
    char  m_eqCompileMsg[1024];
    bool  m_eqCompileOk;
    float m_eqCompileTime;
    bool  m_eqNeedReload;
    int   m_eqSelectedModel;  // 0=Newton, 1=Janus, 2=Yukawa, 3=MOND, 4=Custom
    int   m_cudaDevice;       // -1=auto, 0/1/2=device forced (from janusgalaxy.cfg)

    // ── Particle markers ──────────────────────────────────────────────────
    void renderMarkers();
    void renderMarkerPaths();
    void removeMarker(int index);
    int  pickParticleCPU(int screenX, int screenY);  // robust CPU ray-picking
    std::vector<int> m_markers;          // indices of the marked particles
    int  m_selectedMarker;               // index of the selected marker (-1 = none)
    bool m_markerPickPending;            // mouse click awaiting picking
    int  m_markerPickX, m_markerPickY;   // click coords

    // ── Marker kinetic temperature (local σ of the neighbourhood) ─────────────
    // A single particle has no T: we measure the velocity dispersion of its
    // neighbourhood (bulk flow removed) -> σ (km/s) -> equivalent kinetic T.
    static const int MARKER_SIG_NG = 48;
    std::vector<float> m_markerSigKms;   // σ_3D (km/s) per cell (0 if < 4 neighbours)
    float m_markerSigMn[3]  = {0,0,0};   // origin of the σ grid
    float m_markerSigExt[3] = {1,1,1};   // extent of the σ grid
    int   m_markerSigTick   = 0;         // recomputation throttle
    void  computeMarkerSigmaField();     // (re)computes the σ field (positive matter)
    float markerSigmaKms(int idx) const; // σ_3D km/s at the position of particle idx

    // ── Marker trajectories ─────────────────────────────────────────────────
    static const int PATH_HISTORY = 256;
    std::vector<bool> m_markerPathEnabled;   // trail active per marker
    std::vector<std::vector<cl_float4>> m_markerPaths; // position history
    void updateMarkerPaths();                // called after each simulation step

};

#endif
