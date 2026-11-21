#version 330 core

in  float vMass;
in  float vDepth;
in  float vPickID;   // packed particle index (0.0 = no pick, >0.0 = id+1)
in  float vRadius;   // distance from galaxy center
in  float vNeg;      // 1.0 = negative species (robuste, ≠ signe de vMass qui peut être -0)

uniform vec3  uPosPalette[32];
uniform vec3  uNegPalette[32];
uniform float uMaxMass;
uniform float uGalaxyRadius; // radius where negative mass becomes transparent
uniform int   uPickMode;   // 0 = normal, 1 = picking
uniform float uSoftHalo;   // >0.5 : noyau doux gaussien (rendu volumétrique/CFD) au lieu du disque dur

out vec4 fragColor;

void main() {
    // ── Coordonnées du fragment dans le point sprite [0,1] ─────────────────
    vec2  pc   = gl_PointCoord - vec2(0.5);
    float dist = dot(pc, pc);       // squared distance from centre

    if (uPickMode > 0) {
        // Picking mode: encode particle ID in RGBA
        if (dist > 0.25)            // only encode inside circle
            discard;
        uint id = uint(vPickID);
        float r = float(id & 0xFFu) / 255.0;
        float g = float((id >> 8u) & 0xFFu) / 255.0;
        float b = float((id >> 16u) & 0xFFu) / 255.0;
        fragColor = vec4(r, g, b, 1.0);
        return;
    }

    // ── Bord progressif (anti-scintillement) ─────────────────────────────
    // Au lieu de discard brutal qui crée des bords durs sub-pixel,
    // on utilise un alpha progressif qui fond le point dans le fond.
    // dist va de 0 (centre) à 0.5 (coin) ; on fait une transition douce.
    float alpha_circle;
    if (uSoftHalo > 0.5) {
        // Noyau DOUX et LARGE (gaussien étalé) : chaque particule = un halo vaporeux.
        // Accumulés en additif -> champ volumétrique lumineux, transitions de couleur
        // fondues (rendu type SPH/CFD, comme les visualisations de fluides).
        alpha_circle = 0.55 * exp(-dist * 4.0);   // dist = r² (0 centre -> 0.5 coin) ; large = vaporeux
    } else {
        float radius = sqrt(dist);
        alpha_circle = 1.0 - smoothstep(0.30, 0.48, radius);
    }
    if (alpha_circle <= 0.01)
        discard;  // vraiment invisible, on peut discard

    // Map mass magnitude to palette [0,1], INTERPOLÉ entre entrées -> dégradé lisse
    // (continu, pas 32 bandes discrètes).
    float absMass  = abs(vMass);
    float normMass = clamp(absMass / max(uMaxMass, 1e-6), 0.0, 1.0);
    float fidx     = normMass * 31.0;
    int   i0       = int(fidx);
    int   i1       = min(i0 + 1, 31);
    float fr       = fidx - float(i0);

    vec3  color;
    float alpha = 1.0;
    if (vNeg < 0.5)
        color = mix(uPosPalette[i0], uPosPalette[i1], fr);
    else
        color = mix(uNegPalette[i0], uNegPalette[i1], fr);

    fragColor = vec4(color, alpha * alpha_circle);
}
