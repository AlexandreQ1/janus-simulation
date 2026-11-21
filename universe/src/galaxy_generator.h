#pragma once
#include "cl_float4_compat.h"
#include <vector>

struct GalaxyParams {
    float radius;
    float scaleUnit;
    float timeStep;
    int   numNegaStars;
    int   numPosiStars;
    float negaDensity;      // masse de chaque étoile négative (valeur négative)
    float totalSolarMass;   // masse totale des étoiles positives
    float blackHoleMass;    // masse du trou noir central
    float rotSpeed;         // vitesse de rotation rad/Gy
    float initSpeed;        // vitesse initiale de base
    float grav;             // G * 1e10 (pré-multiplié)
};

class GalaxyGenerator {
public:
    explicit GalaxyGenerator(const GalaxyParams& params);

    void generate(std::vector<cl_float4>& positions,
                  std::vector<cl_float4>& velocities);

    int getTotalCount() const { return m_totalCount; }

private:
    GalaxyParams m_params;
    int          m_totalCount;
};
