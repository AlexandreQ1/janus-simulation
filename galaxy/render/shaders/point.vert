#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in float aMass;
layout(location = 2) in float aOrigIdx;  // original particle index (0-based)

uniform mat4  uViewProj;
uniform float uPointSize;
uniform float uPointSizeNeg;
uniform float uMaxMass;
uniform float uShowNegaMass;

out float vMass;
out float vDepth;
out float vPickID;  // packed particle index
out float vRadius;  // distance from center (for negative mass alpha)

void main()
{
    vPickID = aOrigIdx + 1.0;

    if (uShowNegaMass < 0.5 && aMass < 0.0)
    {
        gl_Position  = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        vMass  = 0.0;
        vDepth = 1.0;
        vRadius = 99999.0;
        return;
    }

    vec4 pos = uViewProj * vec4(aPos, 1.0);
    gl_Position  = pos;
    vMass  = aMass;
    vDepth = pos.w;
    vRadius = sqrt(aPos.x*aPos.x + aPos.y*aPos.y);

    // Perspective-correct point size — réduit les points avec la distance
    // (pos.w = profondeur clip-space). Empêche les particules <2px qui scintillent.
    float sizeBase = (aMass < 0.0) ? uPointSizeNeg : uPointSize;
    float perspectiveSize = sizeBase * (1500.0 / pos.w);
    gl_PointSize = clamp(perspectiveSize, 2.0, 64.0);
}
