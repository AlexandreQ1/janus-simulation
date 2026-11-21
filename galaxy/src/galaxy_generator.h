#pragma once
#include "cl_float4_compat.h"
#include <vector>

struct GalaxyParams {
    float radius;
    float scaleUnit;
    float timeStep;
    int   numNegaStars;
    int   numPosiStars;
    float negaDensity;      // mass of each negative star (negative value)
    float totalSolarMass;   // total mass of the positive stars
    float blackHoleMass;    // mass of the central black hole
    float rotSpeed;         // rotation speed rad/Gy
    float initSpeed;        // base initial speed
    float grav;             // G * 1e10 (pre-multiplied)
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
