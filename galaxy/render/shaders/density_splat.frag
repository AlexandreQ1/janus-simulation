#version 330 core

in float vVal;
in float vSize;   // taille du sprite (px)
in float vNeg;    // 1.0 si masse négative

// Accumulation pondérée (blending additif) :
//   R = Σ w·valeur   (pour la propriété MOYENNE = R/G)
//   G = Σ w          (densité de SURFACE = luminosité)
//   B = Σ w (NÉGATIVES seulement) -> fraction négative = B/G pour la teinte rose
out vec4 fragAccum;

void main()
{
    vec2  pc = gl_PointCoord - vec2(0.5);
    float d2 = dot(pc, pc) * 4.0;     // 0 centre -> 1 bord
    if (d2 > 1.0) discard;
    float k = exp(-d2 * 1.5);          // noyau gaussien doux

    // NORMALISATION (style py-sphviewer, version douce ∝ 1/vSize) : un sprite grand (diffus)
    // contribue MOINS par pixel qu'un petit (dense) -> les halos diffus deviennent un voile
    // faible au lieu de "pièces" brillantes uniformes, tout en restant visibles (≠ 1/vSize²
    // qui les rendait invisibles). Le flou post (renderSPH) fusionne ensuite en champ continu.
    // Réf 16 (~taille typique) -> valeurs en O(1).
    float w = k * 16.0 / vSize;
    fragAccum = vec4(w * vVal, w, w * vNeg, 0.0);
}
