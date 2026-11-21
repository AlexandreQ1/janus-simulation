#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in float aMass;
layout(location = 2) in float aOrigIdx;  // original particle index (0-based)
layout(location = 3) in float aSize;     // taille adaptative (longueur de lissage SPH ; 1.0 = uniforme)

uniform mat4  uViewProj;
uniform float uPointSize;
uniform float uPointSizeNeg;
uniform float uMaxMass;
uniform float uShowNegaMass;
uniform vec3  uClipEnable;    // plans de coupe X/Y/Z : 0/1 par axe
uniform vec3  uClipPos;       // position de coupe par axe (monde)
uniform vec3  uClipSign;      // +1 = masque au-dessus de la position, -1 = en-dessous

out float vMass;
out float vDepth;
out float vPickID;  // packed particle index
out float vRadius;  // distance from center (for negative mass alpha)
out float vNeg;     // 1.0 = negative species (robust : signe de aSize, pas de aMass=-0)

void main()
{
    // Pass original particle index for picking; 0 = no pick, id+1 = valid ID
    vPickID = aOrigIdx + 1.0;

    // Espèce = SIGNE de aSize (négatif = masse négative). Robuste, car la VALEUR couleur
    // (aMass) peut valoir -0.0 (ex. temp/halo = 0 sur une négative) -> aMass<0 serait FAUX
    // -> palette positive par erreur. aSize porte toujours le signe d'espèce.
    bool isNeg = (aSize < 0.0);
    vNeg = isNeg ? 1.0 : 0.0;

    // Plans de coupe X/Y/Z : CUTAWAY -> on retire le COIN où TOUS les axes actifs coupent
    // (1 axe = demi-espace, 2 = quart, 3 = octant retiré -> on voit jusqu'au centre).
    bool clAny = (uClipEnable.x > 0.5) || (uClipEnable.y > 0.5) || (uClipEnable.z > 0.5);
    bool clCut = true;
    if (uClipEnable.x > 0.5) clCut = clCut && (uClipSign.x*(aPos.x - uClipPos.x) > 0.0);
    if (uClipEnable.y > 0.5) clCut = clCut && (uClipSign.y*(aPos.y - uClipPos.y) > 0.0);
    if (uClipEnable.z > 0.5) clCut = clCut && (uClipSign.z*(aPos.z - uClipPos.z) > 0.0);
    if (clAny && clCut)
    {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0;
        vMass = 0.0; vDepth = 1.0; vRadius = 99999.0; return;
    }

    if (uShowNegaMass < 0.5 && isNeg)
    {
        gl_Position  = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        vMass  = 0.0;
        vDepth = 1.0;
        vRadius = 99999.0;
        return;
    }

    vec4 pos = uViewProj * vec4(aPos, 1.0);
    // Discard les particules DERRIÈRE/au ras de la caméra (pos.w <= 0) : sinon leur
    // projection s'inverse -> traînée/bande géante en vue rasante (caméra dans le plan).
    if (pos.w < 0.5)
    {
        gl_Position  = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        vMass  = 0.0;
        vDepth = 1.0;
        vRadius = 99999.0;
        return;
    }
    gl_Position  = pos;
    vMass  = aMass;
    vDepth = pos.w;
    // Distance from galaxy center (XY plane, before projection)
    vRadius = sqrt(aPos.x*aPos.x + aPos.y*aPos.y);
    
    // ── Perspective-correct point size (uniforme ; aSize sert UNIQUEMENT au rendu SPH) ──
    float sizeBase = isNeg ? uPointSizeNeg : uPointSize;
    float perspectiveSize = sizeBase * (1500.0 / pos.w);
    gl_PointSize = clamp(perspectiveSize, 2.0, 64.0);
}
