#version 330 core

layout(location = 0) in vec3  aPos;
layout(location = 1) in float aValue;  // valeur couleur (signée par espèce) = propriété (temp/densité/...)
layout(location = 3) in float aSize;   // longueur de lissage SPH (taille adaptative)

uniform mat4  uViewProj;
uniform float uPtSize;     // taille de base
uniform float uShowNeg;    // 0 = cacher masses neg
uniform float uSizeScale;  // focale en pixels = 0.5*hauteur / tan(fov/2) -> taille correcte à TOUT zoom
uniform vec3  uClipEnable; // plans de coupe X/Y/Z : 0/1 par axe
uniform vec3  uClipPos;    // position de coupe par axe (monde)
uniform vec3  uClipSign;   // +1 = masque au-dessus, -1 = en-dessous

out float vVal;          // |propriété| pour la moyenne pondérée
out float vSize;         // taille du sprite en pixels (pour la normalisation masse/h²)
out float vNeg;          // 1.0 si masse négative, 0.0 sinon (pour la teinte rose)

void main()
{
    bool clAny = (uClipEnable.x > 0.5) || (uClipEnable.y > 0.5) || (uClipEnable.z > 0.5);
    bool clCut = true;
    if (uClipEnable.x > 0.5) clCut = clCut && (uClipSign.x*(aPos.x - uClipPos.x) > 0.0);
    if (uClipEnable.y > 0.5) clCut = clCut && (uClipSign.y*(aPos.y - uClipPos.y) > 0.0);
    if (uClipEnable.z > 0.5) clCut = clCut && (uClipSign.z*(aPos.z - uClipPos.z) > 0.0);
    if (clAny && clCut) {
        gl_Position  = vec4(99.0, 99.0, 99.0, 1.0);
        gl_PointSize = 0.0;
        vVal = 0.0; vSize = 1.0; vNeg = 0.0;
        return;
    }
    if (uShowNeg < 0.5 && aSize < 0.0) {
        gl_Position  = vec4(99.0, 99.0, 99.0, 1.0);
        gl_PointSize = 0.0;
        vVal = 0.0; vSize = 1.0; vNeg = 0.0;
        return;
    }
    vec4 p = uViewProj * vec4(aPos, 1.0);
    // Discard les particules DERRIÈRE la caméra (p.w <= 0) : elles projettent de travers
    // -> traînées/bande en vue rasante.
    if (p.w < 0.5) {
        gl_Position  = vec4(99.0, 99.0, 99.0, 1.0);
        gl_PointSize = 0.0;
        vVal = 0.0; vSize = 1.0; vNeg = 0.0;
        return;
    }
    gl_Position  = p;
    // Taille écran = longueur de lissage WORLD (aSize) projetée par la VRAIE focale
    // (uSizeScale = 0.5*H/tan(fov/2)) / profondeur. Correct à tout zoom (avant : 1500 bidon).
    // aSize : magnitude = longueur de lissage ; SIGNE = espèce (négatif = masse négative).
    float sz = clamp(uPtSize * abs(aSize) * (uSizeScale / p.w), 2.0, 160.0);
    gl_PointSize = sz;
    vSize = sz;
    vVal  = abs(aValue);
    vNeg  = (aSize < 0.0) ? 1.0 : 0.0;   // espèce négative -> canal de teinte rose (robuste)
}
