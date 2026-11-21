#version 330 core

in  float vMass;
in  float vDepth;
in  float vPickID;   // packed particle index (0.0 = no pick, >0.0 = id+1)
in  float vRadius;   // distance from galaxy center

uniform vec3  uPosPalette[32];
uniform vec3  uNegPalette[32];
uniform float uMaxMass;
uniform float uGalaxyRadius; // radius where negative mass becomes transparent
uniform int   uPickMode;   // 0 = normal, 1 = picking

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
    float radius = sqrt(dist);
    float alpha_circle = 1.0 - smoothstep(0.30, 0.48, radius);
    if (alpha_circle <= 0.01)
        discard;

    // Map mass magnitude to palette index [0, 31]
    float absMass  = abs(vMass);
    float normMass = clamp(absMass / max(uMaxMass, 1e-6), 0.0, 1.0);
    int   idx      = int(normMass * 31.0);

    vec3  color;
    float alpha;

    if (vMass >= 0.0) {
        color = uPosPalette[idx];
        alpha = 1.0;
    } else {
        color = uNegPalette[idx];
        float rRatio = vRadius / max(uGalaxyRadius, 1.0);
        alpha = mix(0.005, 1.0, smoothstep(1.0, 2.0, rRatio));
    }

    fragColor = vec4(color, alpha * alpha_circle);
}
