#version 330 core

in  vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uAccum;     // champ NET (cœurs piqués) : R = Σ w·valeur , G = Σ w
uniform sampler2D uAccumBlur; // champ FLOUTÉ (voile lisse, mêmes canaux)
uniform vec3      uPalette[32];
uniform float     uValMax;    // normalisation de la propriété (couleur)
uniform float     uLumScale;  // luminosité (densité de colonne)
uniform int       uColorByDensity; // 1 = teinte indexée par la DENSITÉ (look viridis py-sphviewer)
uniform float     uAsinhCeil;      // plafond de la compression asinh : bas = cœurs blancs saturés,
                                   // haut = dynamique étirée -> détail/dégradé DANS les cœurs denses

// Échantillonne la palette 32 entrées en t∈[0,1]. Interpolation SMOOTHSTEP (C1) entre
// entrées -> supprime les "cassures de Mach" de l'interpolation linéaire (transitions douces).
vec3 paletteLookup(float t)
{
    t        = clamp(t, 0.0, 1.0);
    float fi = t * 31.0;
    int   i0 = int(fi);
    int   i1 = min(i0 + 1, 31);
    float fr = fi - float(i0);
    fr = fr * fr * (3.0 - 2.0 * fr);          // smoothstep -> dérivée continue aux jointures
    return mix(uPalette[i0], uPalette[i1], fr);
}

// Dither hash (bruit blanc par pixel) : casse le banding de quantification 8 bits de l'écran.
float ditherHash(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main()
{
    vec3 accS = texture(uAccum,     vTexCoord).rgb;   // net (cœurs)   : R=Σw·val G=Σw B=Σw(neg)
    vec3 accB = texture(uAccumBlur, vTexCoord).rgb;   // flouté (voile)

    // FLOU PRÉSERVANT LES DÉTAILS, gaté par les AMAS. Problème du max(net,flou) brut :
    // CHAQUE particule (encore sur le réseau initial) perce le voile -> grille de points.
    // Solution : ne révéler le champ NET (pics piqués) QUE là où la densité GRANDE-ÉCHELLE
    // (floutée gB) est élevée = les amas. Dans le voile (gB faible) -> flou pur, lisse,
    // SANS grille. coreW : 0 dans le voile -> 1 dans les amas (seuils liés à l'échelle asinh).
    float gS = accS.g, gB = accB.g;
    float coreW = smoothstep(2.0, 12.0, gB * uLumScale);
    float gMix  = mix(gB, max(gS, gB), coreW);

    // Densité de colonne tone-mappée ASINH (standard astro DS9/py-sphviewer).
    float g   = gMix * uLumScale;
    float lum = clamp(asinh(g) / asinh(uAsinhCeil), 0.0, 1.0);

    // Fraction de masse NÉGATIVE à ce pixel (B/G combinés) -> teinte rose.
    float negFrac = clamp((accS.b + accB.b) / (gS + gB + 1e-6), 0.0, 1.0);

    // Dither : ±1 LSB 8 bits réparti par pixel -> dégradés parfaitement lisses à l'écran.
    float dth = (ditherHash(gl_FragCoord.xy) - 0.5) / 255.0;

    // ── MODE py-sphviewer : la TEINTE encode la DENSITÉ (viridis), fond rempli, opaque ──
    if (uColorByDensity == 1) {
        vec3 dens = paletteLookup(lum);
        // Rampe ROSE (sombre->vif selon la densité) pour la masse NÉGATIVE, afin de la
        // distinguer du viridis/thermique des positives.
        vec3 pinkRamp = mix(vec3(0.10, 0.0, 0.06), vec3(1.0, 0.35, 0.80), lum);
        vec3 col = mix(dens, pinkRamp, smoothstep(0.30, 0.60, negFrac));
        fragColor = vec4(col + dth, 1.0);
        return;
    }

    if (gMix < 1e-6) { fragColor = vec4(0.0); return; }

    // ── MODE propriété : teinte = MOYENNE pondérée, lum = densité de colonne ──
    float avg = (accS.g > 1e-6) ? (accS.r / accS.g) : (accB.r / max(accB.g, 1e-6));
    vec3  col = paletteLookup(avg / max(uValMax, 1e-6));
    col = mix(col, vec3(1.0), 0.28);   // pastel laiteux

    fragColor = vec4(col * lum + dth, lum);
}
