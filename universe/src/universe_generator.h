#pragma once
#include "cl_float4_compat.h"
#include <vector>

struct UniverseParams {
    float initRadius;    // rayon de la boule primordiale (ou demi-côté du box)
    int   numPosiPart;
    int   numNegaPart;
    float posiMass;
    float negaMass;      // valeur négative
    float pertAmp;       // legacy : graine aléatoire
    float H0;            // paramètre Hubble (sim units)
    float cosmoScale;    // découplage Friedmann/N-body — scale vFactor Zel'dovich
    bool freeExplosion;  // si vrai, générer une explosion libre sans boîte ni FFT
    bool periodicBox;    // si vrai, remplir le cube [-R,R]³ uniformément (PM périodique)
    int  webCells = 12;  // sphère semée : résolution du champ anti-corrélé (= nb de bulles ~ NC/2 par axe)
    bool foamMode = false; // sphère semée : spectre piqué (bulles uniformes "mousse Petit") au lieu de P(k) hiérarchique
};

class UniverseGenerator {
public:
    explicit UniverseGenerator(const UniverseParams& p);

    // Cosmological parameters (optional, LCDM defaults)
    void setCosmology(float omegaM, float omegaL, float ns, float sigma8, float aStart);
    void setGridResolution(int nGrid);
    // Manual RNG seed for reproducibility. seed=0 → keep time-based default.
    void setSeed(unsigned int seed);

    void generate(std::vector<cl_float4>& positions,
                  std::vector<cl_float4>& velocities);
    int getTotalCount() const { return m_total; }

private:
    UniverseParams m_p;
    int            m_total;
    float m_omegaM, m_omegaL, m_ns, m_sigma8, m_aStart;
    int   m_nGrid;
};
