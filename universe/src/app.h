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
# include "camera.h"
# include "color.h"
# include "vector.h"
# include "matrix.h"
# include "simulation.h"
# include "spacemouse_nav.h"
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
    void updateAutoFit();   // auto-cadrage : recule la caméra en douceur pour suivre l'expansion
    void renderGUI();
    void uploadParticlesToGPU(bool subData = false);
    void computeLocalDensity();        // scheduler (cheap): adopt result / launch worker
    void computeLocalDensityWorker();  // heavy 64^3 density + smoothing on a background thread
    // Background-thread density: the SPH smoothing-length recompute (~100+ ms at high N)
    // ran on the main thread (1 frame/12) -> periodic hitch in Soft Halo / Color-by-Density.
    // Threading it removes the hitch; the field evolves slowly so async update is invisible.
    std::thread            m_densThread;
    std::atomic<bool>      m_densComputing{false};
    std::atomic<bool>      m_densReady{false};
    std::vector<cl_float4> m_densPosCopy;          // snapshot fed to the worker
    std::vector<float>     m_densityBack, m_smoothLengthBack;
    float                  m_densMaxBack = 1.0f;
    uint64_t               m_densLastLaunchMs = 0;
    int                    m_densIntervalMs = 150; // min ms between density recomputes
    void startSimulation();
    void resetSimulation();
    void stepSimulation();
    void saveSettings(const std::string& path);
    void loadSettings(const std::string& path);
    void renderGrid();
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

     // ── Paramètres univers (réutilisent les noms galaxie pour compatibilité) ──
     float m_radius;         // = init radius de la boule primordiale
     float m_totalSolarMass; // = masse par particule positive
     float m_negaDensity;    // = masse par particule négative (valeur négative)
     int   m_numPosiStars;   // = nombre de particules positives
     int   m_numNegaStars;   // = nombre de particules négatives
     float m_pertAmp;        // legacy : graine aléatoire
     int   m_webCells = 12;  // sphère semée : nb de cellules du champ anti-corrélé (nb de bulles)
     bool  m_foamMode = false; // sphère semée : spectre piqué (bulles uniformes "mousse Petit") au lieu de P(k)
     float m_H0;             // constante de Hubble initiale (sim units)
     // conservés pour compatibilité UI (non utilisés physiquement)
     float m_scaleUnit;
     float m_blackHoleMass;
     float m_rotSpeed;
     float m_initSpeed;
     float m_timeStep;
     float m_softening;
     float m_gravConstant;
     // Cosmologie
     float m_omegaM;         // densité matière
     float m_omegaL;         // densité énergie noire
     float m_aStart;         // facteur d'échelle initial
     float m_cosmoScale;     // découplage timescale Friedmann / N-body (0<s<=1)
     float m_dtMin;          // pas de temps adaptatif min
     float m_dtMax;          // pas de temps adaptatif max
     float m_eta;            // critère Courant
     bool  m_useCosmology;   // activer cosmologie + leapfrog DKD
     bool  m_adaptiveDt;     // pas de temps adaptatif
     bool  m_usePM;          // Particle-Mesh cuFFT solver (O(N log N))
     bool  m_pmEnableSR;     // P³M short-range Ewald correction (GADGET-style)
     bool  m_pmPeriodic;     // PM cube périodique (off = open-boundary sphère)
     int   m_pmNg;           // PM grid dimension (64/128/256)
     bool  m_useTree;        // GPU Barnes-Hut tree O(N log N) — bypasses PM
     float m_treeTheta;      // Barnes-Hut opening angle (0.3=précis, 1.0=rapide)
     bool  m_treeQuad;       // order-2 multipole correction
     bool  m_treeWarp;       // warp-coherent traversal (~3x)
     int   m_forceSubcycle = 1;  // recompute tree force 1/K steps (frozen phase speedup)
     bool  m_useManualSeed;  // Use fixed RNG seed for reproducible runs
     int   m_manualSeed;     // Manual RNG seed value (when useManualSeed=true)

     // ── Batch mode (statistical ensemble) ────────────────────────────────
     bool  m_batchActive;        // currently running a batch
     int   m_batchN;             // number of seeds to run
     int   m_batchCurrentIdx;    // current seed index in [0, m_batchN)
     int   m_batchStartSeed;     // first seed (then ++ for next runs)
     float m_batchTargetMY;      // simulated MY to reach before snapshot
     char  m_batchOutDir[256];   // CSV output directory
     int   m_batchState;         // 0=idle, 1=generating, 2=running, 3=saving
     void  tickBatchMode();
     void  exportBatchSnapshotCSV(const char* filename);

     // Wall-clock timer for the simulation (ne compte pas le temps en pause)
     uint64_t m_wallClockStartMs;   // SDL ticks au dernier start/resume
     uint64_t m_wallClockAccumMs;   // temps total accumulé (en pause exclu)
     void togglePauseWithClock();   // pause/resume avec mise à jour du clock
    // galaxie détectée
    int   m_galaxyCount;
    int   m_galaxyMinPart;
    float m_galaxyLinkScale = 0.10f;  // longueur de liaison FoF galaxie (× sep moyenne) :
                                      // courte -> isole les noeuds denses sans percoler les filaments
    bool  m_firstGalaxyDetected;
    float m_firstGalaxyTime;
    bool  m_showGalaxyMarkers;
    std::vector<cl_float4> m_galaxyCentroids;  // centroïdes des galaxies détectées
    void  detectGalaxies();         // scheduler (cheap, every frame): adopt result / launch worker
    void  detectGalaxiesWorker();   // heavy FoF on the snapshot, runs on a background thread
    void  renderGalaxyMarkers();
    // Galaxy-formation history plot (galaxies vs cosmic age) + JWST z~10-14 band.
    bool  m_showGalaxyHistory = false;
    struct GalHistPt { float ageGyr; int galaxies; };
    std::vector<GalHistPt> m_galHistory;
    void  renderGalaxyHistory();
    // Background-thread galaxy detection. The FoF is O(N) when the structure is
    // spread (Janus web) but ~1 s when everything collapses into a few hash cells
    // (Newton). Running it on a worker keeps the frame smooth regardless of geometry.
    std::thread            m_galThread;
    std::atomic<bool>      m_galComputing{false};
    std::atomic<bool>      m_galResultReady{false};
    std::vector<cl_float4> m_galPosCopy;            // snapshot fed to the worker
    std::vector<cl_float4> m_galResult;             // centroids produced by the worker
    int                    m_galResultCount = 0;
    int                    m_galMinSnap = 50;       // params snapshot (read by worker)
    float                  m_galLinkSnap = 0.10f;
    uint64_t               m_galLastLaunchMs = 0;   // wall-clock gate (independent of Sim Speed)
    int                    m_galIntervalMs   = 250; // min ms between galaxy detections

    bool  m_simRunning;
    bool  m_simPaused;
    int   m_currentFrame;
    int   m_totalFrames;
    float m_simSpeed;
    int   m_skipCounter;

    std::vector<cl_float4> m_positions;
    std::vector<cl_float4> m_velocities;
    std::vector<float> m_particleData;
    unsigned long m_particleCount;

    float m_posPalette[32][3];
    float m_negPalette[32][3];
    float m_velPalette[32][3];   // jet : bleu->cyan->jaune->rouge (vitesse)

    float        m_spAmplitude;
    float        m_spPatternSpeed;
    float        m_spPitchAngle;
    int          m_spArms;

    bool  m_glowEnabled;
    float m_blurRadius;
    float m_glowIntensity;
    bool  m_softHalo = false;   // rendu SPH volumétrique (splat noyau adaptatif -> champ continu)
    float m_sphBright = 0.30f;  // luminosité du rendu SPH (densité de colonne -> gamma)
    bool  m_sphViridis = false; // SPH: teinte = DENSITÉ (palette indexée, fond rempli) — look py-sphviewer
    bool  m_sphThermal = false; // (déprécié, lu pour migration cfg) -> m_sphPalette
    int   m_sphPalette = 0;     // SPH: 0=viridis, 1=thermal, 2=galaxy colors (look Hubble bleu/or)
    float m_sphSmooth = 2.0f;   // SPH: rayon du flou gaussien du champ (px) -> filaments continus
    float m_sphCoreDetail = 40.0f; // SPH: plafond asinh (40=cœurs blancs saturés .. 600=détail interne)
    float m_sphQuality = 1.0f;  // SPH: résolution du splat (1=pleine, 0.5=moitié -> ~4x plus rapide)
    float m_viridisPalette[32][3]; // colormap viridis (violet -> bleu -> vert -> jaune)
    float m_thermalPalette[32][3]; // colormap thermique : NOIR -> bleu -> vert -> jaune -> rouge -> blanc
    float m_galaxyPalette[32][3];  // "galaxy colors" (look Hubble) : bleu sombre -> bleu -> crème -> or -> blanc-doré
    bool  m_colorByDensity = false; // colore chaque particule par sa densité locale 3D (style halos dark-matter)
    float m_densMaxForColor = 1.0f; // normalisation de la densité pour la colormap
    float m_densPalette[32][3];     // rampe densité : bleu (vide) -> rose -> rouge -> blanc (nœud/halo)
    std::vector<float> m_density;   // densité locale par particule (même indexation que m_positions)
    std::vector<float> m_smoothLength; // longueur de lissage SPH par particule (∝ densité^-1/3) : taille adaptative
    int   m_sphTick = 0;            // throttle du recalcul des longueurs de lissage (mode soft halo seul)

    // ── Auto-cadrage (auto-fit) : la caméra recule en douceur pour garder la
    //    structure à taille apparente constante pendant qu'elle s'étend. ──
    bool  m_autoFit           = false;  // active le suivi automatique de l'expansion
    float m_autoFitFill       = 0.55f;  // fraction de la demi-hauteur écran occupée par la structure
    float m_autoFitSmoothTime = 0.6f;   // temps de lissage (s) du SmoothDamp — grand = plus doux
    int   m_autoFitInterval   = 15;     // recalcule le rayon de la structure tous les N frames
    int   m_autoFitLastFrame  = -100000;// dernier frame où le rayon a été recalculé
    float m_autoFitTargetRaw  = 0.0f;   // distance cible BRUTE (dernier calcul P90, en escalier)
    float m_autoFitTargetDist = 0.0f;   // cible LISSÉE (passe-bas continu de la cible brute)
    float m_autoFitDistVel    = 0.0f;   // vitesse interne du SmoothDamp (continuité C¹ -> zéro à-coup)
    float m_autoFitRadius     = 0.0f;   // rayon robuste de la structure (P90), pour affichage
    unsigned int m_autoFitLastTicks = 0;// horodatage pour dt réel (lissage indépendant du framerate)

    bool  m_colorByHalo = false;    // colore par halo FoF (Friends-of-Friends, b=0.2, >=200 parts)
    std::vector<int> m_haloSize;    // taille du halo FoF de chaque particule (0 = pas dans un halo >=200)
    float m_haloMaxForColor = 1.0f; // normalisation (log de la plus grosse masse de halo)
    int   m_foFTick = 0;            // throttle du recalcul FoF (lourd)
    void  computeHaloColoring();    // FoF par-particule (b=0.2, >=200) -> m_haloSize (pour le coloriage)

    bool  m_colorByWeb = false;     // classification T-web : vide/paroi/filament/nœud (Hessian de densité)
    std::vector<int> m_webClass;    // classe T-web par particule (0=vide,1=paroi,2=filament,3=nœud)
    float m_webPalette[32][3];      // 4 bandes distinctes (vide sombre, paroi bleue, filament vert, nœud rouge)
    int   m_webTick = 0;            // throttle du recalcul T-web
    void  computeTweb();            // classification de la toile cosmique -> m_webClass

    bool  m_colorByTemp = false;    // "température" = dispersion locale des vitesses (chaud=virialisé, froid=flot/vide)
    std::vector<float> m_temp;      // température (σ²) par particule
    float m_tempMaxForColor = 1.0f; // normalisation
    float m_tempPalette[32][3];     // rampe froid->chaud (bleu sombre -> bleu -> vert -> jaune -> rouge -> blanc)
    int   m_tempTick = 0;           // throttle
    void  computeTemperature();     // dispersion locale des vitesses -> m_temp

    // ── CUDA-OpenGL interop : le VBO de rendu est construit par un kernel CUDA depuis
    //    les buffers device (pas de reconstruction/upload CPU par frame). Couvre la
    //    masse, la vitesse et les valeurs auxiliaires (température, T-web, halo,
    //    densité) envoyées au device quand elles sont recalculées. ──
    bool   m_useInterop = true;
    bool   m_interopOk  = false;
    void*  m_interopRes = nullptr;
    size_t m_interopVboCount = 0;
    void*  m_dSmooth = nullptr;        // longueurs de lissage SPH (device)
    int    m_dSmoothCount = 0;
    bool   m_smoothDirty = false;
    void*  m_dAux = nullptr;           // valeurs de couleur auxiliaires (device)
    int    m_dAuxCount = 0;
    int    m_tempVersion = 0, m_webVersion = 0, m_haloVersion = 0, m_densityVersion = 0;
    int    m_auxValuesVersion = -1, m_auxValuesMode = -1;
    int    m_auxUploadedVersion = -1, m_auxUploadedMode = -1;
    std::vector<float> m_auxValues;    // valeurs dérivées web/halo
    bool   uploadParticlesInterop();

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

    // ── Expansion diagnostic : a(t), ȧ=H·a vs cosmic age (test accélération/coasting) ──
    bool   m_showExpansion = false;
    std::vector<double> m_expTimes;   // âge cosmique (Gyr) — buffer NON bouclant (décimé à saturation)
    std::vector<float>  m_expA;       // facteur d'échelle a
    std::vector<float>  m_expAdot;    // ȧ = H·a (vitesse d'expansion)
    int    m_expTick = 0;             // throttle d'enregistrement
    int    m_expThrottle = 4;         // frames/échantillon (×2 à chaque décimation)
    void   renderExpansion();

    bool  m_colorByVelocity;
    bool  m_velAutoScale;       // auto-scale velocity color range at each refresh
    bool  m_velPeakHold = false;// peak-hold: l'échelle ne monte que (garde le pic) -> refroidissement visible
    float m_velMaxForColor;
    float m_particleSize;
    float m_particleSizeNeg;

    void saveScreenshot();
    bool m_isGenerating;        // galaxy generation in progress
    int  m_generateCountdown;   // >0 = deferred generate (2=wait, 1=show text, 0=launch)

    // ── Annotations (screen-space 2D overlay for annotated screenshots) ───────
    void renderAnnotations();
    struct Annotation {
        int   type;          // 1=arrow, 2=text, 3=rect, 4=circle
        float x0, y0, x1, y1;// window-relative coords (two points)
        char  text[128];
        unsigned int col;
    };
    std::vector<Annotation> m_annotations;
    int   m_annotTool = 0;       // 0=none,1=arrow,2=text,3=rect,4=circle
    int   m_annotStage = 0;      // 0=idle (waiting 1st point), 1=have 1st point
    float m_annotX0 = 0, m_annotY0 = 0;   // first point, window-relative
    float m_annotColor[3] = {1.0f, 0.85f, 0.2f};  // amber default
    int   m_annotEditIdx = -1;   // annotation index whose text is being edited
    char  m_annotTextBuf[128] = {0};
    // Deferred screenshot: capture AFTER ImGui render so overlays/annotations show
    bool  m_screenshotPending = false;
    char  m_screenshotPath[512] = {0};
    void  captureScreenshotNow();    // viewport-only PNG, called at end of frame

    // ── MP4 video recording (viewport → ffmpeg pipe) ─────────────────────────
    bool  m_videoRecording = false;
    void* m_ffmpegStdin = nullptr;
    void* m_ffmpegProc  = nullptr;
    int   m_recW = 0, m_recH = 0;
    int   m_videoFps = 30;
    char  m_videoPath[512] = "";
    std::vector<unsigned char> m_recBuf;
    std::string findFFmpeg();
    bool  startVideoRecording();
    void  captureVideoFrame();
    void  stopVideoRecording();

    // ── Replay timeline (RAM ring buffer of position snapshots) ──────────────
    bool  m_replayRecord = false;
    int   m_replayInterval = 50;
    int   m_replayMaxFrames = 120;
    bool  m_replayMode = false;
    bool  m_replayPlaying = false;
    int   m_replayCursor = 0;
    float m_replayPlaySpeed = 1.0f;
    int   m_replayLastRecFrame = -1000000;
    std::vector<std::vector<cl_float4>> m_replayFrames;
    std::vector<float> m_replayTimes;
    void  replayRecordSnapshot();
    char  m_replayFilePath[256] = "C:\\Users\\Administrateur\\Desktop\\replay.jrp";
    void  saveReplay();   // persist the RAM ring-buffer snapshots to a single file
    void  loadReplay();
    // Streaming-to-disk replay: each snapshot written straight to a folder
    // (unlimited length, never lost). m_replayFrames stays empty in this mode.
    bool  m_replayStream = false;
    char  m_replayDir[512] = "D:\\SIMULATION\\replay";
    int   m_replayDiskCount = 0;
    int   m_replayShownCursor = -1;   // last frame actually loaded (avoid redundant disk reads)
    void  replayWriteDiskFrame();      // write current m_positions as snap N
    bool  replayReadDiskFrame(int idx);// load snap idx -> m_positions
    void  replayScanDiskFolder();      // count snaps + read their times from the folder
    int   replayCount() const;         // RAM or disk snapshot count
    void  replaySetFrame(int idx);     // set m_positions from RAM/disk + upload
    void  tickReplay();

    // ── Phase Space options ───────────────────────────────────────────────
    bool  m_phaseShowNeg;       // show negative mass in phase space
    float m_phaseRMaxLY;        // X-axis max radius (LY). 0 = auto from rMax sim
    // ── Time-window sliders (Energy, Barycenter)  in MY ─────────────────────
    float m_energyTimeRangeMY;  // 0 = show all history, >0 = last X MY
    float m_baryTimeRangeMY;    // 0 = show all history, >0 = last X MY

    // ── Presets ───────────────────────────────────────────────────────────
    void savePreset(const std::string& name);
    void loadPreset(const std::string& name);
    std::vector<std::string> listPresets();
    char m_presetName[128];

    void saveSimState();
    void saveSimStateTo();   // boîte de dialogue "Enregistrer sous..."
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
    bool m_midMouseDown = false;   // bouton du milieu (molette) enfoncé -> pan
    int  m_lastMouseX, m_lastMouseY;
    // 3Dconnexion (SpaceMouse) — optionnel
    SpaceMouseNav* m_spaceMouse = nullptr;
    bool   m_use3DMouse = false;
    double m_smBBmin[3] = {-1000,-1000,-1000}, m_smBBmax[3] = {1000,1000,1000};
    int    m_smBBtick = 0;
    void   update3DMouse();
    // Plans de coupe (cross-section) — 3 axes indépendants combinables (X/Y/Z)
    bool   m_clipEnable = false;          // maître (affiche les sous-réglages)
    bool   m_clipAxisOn[3]  = {false,false,false};
    float  m_clipAxisFrac[3]= {0,0,0};    // position ∈ [-1,1] × extent (autour du centre)
    bool   m_clipAxisFlip[3]= {false,false,false};
    float  m_clipExtent = 1000.0f;        // demi-étendue (recalculée à l'activation)
    float  m_clipCenter[3] = {0,0,0};     // barycentre de la structure (la coupe est centrée dessus)
    bool   m_everGenerated = false;       // l'utilisateur a-t-il cliqué Apply & Generate ? (fenêtre de bienvenue)
    // Marqueur différé : on attend ~le délai double-clic avant de poser le marqueur,
    // pour qu'un double-clic (focus caméra) ne laisse pas un marqueur parasite.
    bool         m_pendingMarker = false;
    int          m_pendingMarkerX = 0, m_pendingMarkerY = 0;
    unsigned int m_pendingMarkerTime = 0;

    // ── Barycentres ──────────────────────────────────────────────────────────
    void renderBarycenters();
    bool m_showBarycenters;
    static const int BARY_HISTORY = 512;
    float  m_baryDist  [BARY_HISTORY]; // distance |G+ - G-| en LY
    float  m_baryTimes [BARY_HISTORY];
    int    m_baryHead, m_baryCount;
    float  m_baryPosX, m_baryPosY;    // centre de masse positif courant (unités sim)
    float  m_baryNegX, m_baryNegY;    // centre de masse négatif courant

    // ── Modes de Fourier ─────────────────────────────────────────────────────
    void renderFourierModes();
    bool m_showFourier;
    bool m_showPowerSpectrum;
    void renderPowerSpectrum();
    static const int FOURIER_HISTORY = 512;
    float  m_fourierA[4][FOURIER_HISTORY]; // m=1,2,3,4
    float  m_fourierTimes[FOURIER_HISTORY];
    int    m_fourierHead, m_fourierCount;

    // ── Profil matière noire équivalente ─────────────────────────────────────
    void renderDarkMatterProfile();
    bool  m_showDarkMatter;
    float m_nfwRs;    // rayon caractéristique NFW (unités sim)
    float m_nfwRho0;  // densité centrale NFW (normalisée auto)

    // ── Scientific analytics ─────────────────────────────────────────────────
    void renderConservation();   // Angular momentum + total energy
    void renderEscapeVelocity(); // Negative mass average speed history
    void renderCorrelation();    // Two-point correlation ξ(r)
    void renderSersic();         // Surface brightness + Sersic fit

    bool  m_showConservation;
    bool  m_showEscape;
    bool  m_showCorrelation;
    bool  m_showSersic;

    // ── FoF halo finder + per-cluster virial ratio (à la demande) ────────────
    void renderHaloFinder();
    void renderFoFOverlay();   // cluster circles on the viewport (drawn every frame)
    void computeFoF();
    bool  m_showHaloFinder;
    float m_fofLinkingScale;
    int   m_fofMinMembers;
    bool  m_fofIncludeNeg;
    bool  m_fofOverlay;
    struct FoFCluster {
        float cx, cy, cz;
        float vx, vy, vz;
        float mass;
        int   count;
        float radius;
        float virial;
        std::vector<int> members;   // global particle indices (for highlight)
    };
    std::vector<FoFCluster> m_fofClusters;
    bool  m_fofDone;
    int   m_fofSelected = -1;       // selected cluster row (-1 = none)
    // Live mode: Run toggles a persistent background loop (button red), the worker
    // re-detects every N frames so the catalog/overlay update on their own.
    bool  m_fofLive = false;
    int   m_fofLiveInterval = 50;
    int   m_fofLiveLastFrame = -100000;
    // Background-thread execution (keeps UI responsive at millions of particles)
    std::thread             m_fofThread;
    std::atomic<bool>       m_fofComputing{false};
    std::atomic<bool>       m_fofResultReady{false};
    std::atomic<bool>       m_fofCancel{false};
    std::vector<cl_float4>  m_fofPosCopy, m_fofVelCopy;
    std::vector<FoFCluster> m_fofResult;
    float m_fofGsnap, m_fofSoftSnap, m_fofLinkSnap;
    float m_fofHsnap = 0.0f;   // paramètre de Hubble au lancement (retrait du flot pour le KE)
    float m_fofAsnap = 1.0f;   // facteur d'échelle a au lancement (PE en distances physiques)
    int   m_fofMinSnap;
    bool  m_fofNegSnap;
    void  exportFoFCSV();

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

    // ── Simulation speed (MY/sec ou Gyr/s en cosmologie) ──────────────────────
    float  m_simMyPerSec;
    double m_mySecLastSimTime;
    Uint32 m_mySecLastRealMs;
    double m_curAgeGyr  = 0.0;   // âge cosmologique courant (depuis a), pour l'affichage
    double m_lastAgeGyr = 0.0;   // âge au dernier calcul de débit (pour Gyr/s)

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
    float m_yukawaAlpha  = 0.5f;    // Yukawa strength : F *= (1 + alpha·e^{-r/lambda})
    float m_yukawaLambda = 500.0f;  // Yukawa range (sim units)
    float m_mondA0       = 0.05f;   // MOND acceleration scale a0 (sim accel units)
    bool  m_removeMonopole = false; // anti-monopole (Jeans swindle) : supprime l'effondrement central
    float m_monopoleFrac   = 1.0f;  // fraction du monopôle radial soustraite
    int   m_cudaDevice;       // -1=auto, 0/1/2=device forcé (depuis jgravity.cfg)
    int   m_targetFps = 60;   // limiteur de framerate (0 = illimité) — réglable, persiste en cfg

    // ── Repère d'orientation XYZ (triade type logiciel 3D) ────────────────
    void renderAxisGizmo();
    bool m_showAxisGizmo = true;   // affiche le repère XYZ en coin d'écran

    // ── Marqueurs de particules ───────────────────────────────────────────
    void renderMarkers();
    void renderMarkerPaths();
    void removeMarker(int index);
    int  pickParticleCPU(int screenX, int screenY);  // robust CPU ray-picking
    std::vector<int> m_markers;          // indices des particules marquées
    int  m_selectedMarker;               // index du marqueur sélectionné (-1 = aucun)
    bool m_markerPickPending;            // clic souris en attente de picking
    int  m_markerPickX, m_markerPickY;   // coords du clic

    // ── Température cinétique des marqueurs (σ locale du voisinage) ────────────
    // Champ σ_3D (km/s) par cellule, flot d'ensemble retiré -> agitation aléatoire.
    // Une particule seule n'a pas de T : on mesure la dispersion de son voisinage.
    static const int MARKER_SIG_NG = 48;
    std::vector<float> m_markerSigKms;   // σ_3D (km/s) par cellule (0 si < 4 voisins)
    float m_markerSigMn[3]  = {0,0,0};   // origine de la grille σ
    float m_markerSigExt[3] = {1,1,1};   // étendue de la grille σ
    int   m_markerSigTick   = 0;         // throttle du recalcul
    void  computeMarkerSigmaField();     // (re)calcule le champ σ (positifs)
    float markerSigmaKms(int idx) const; // σ_3D km/s à la position de la particule idx

    // ── Trajectoires des marqueurs ──────────────────────────────────────────
    static const int PATH_HISTORY = 256;
    std::vector<bool> m_markerPathEnabled;   // trace active par marqueur
    std::vector<std::vector<cl_float4>> m_markerPaths; // historique des positions
    void updateMarkerPaths();                // appelé après chaque simulation step

};

#endif
