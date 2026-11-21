#version 330 core

in  vec2 vTexCoord;

uniform sampler2D uScene;
uniform sampler2D uGlow;
uniform float     uGlowIntensity;

// ── 3D Stereo ─────────────────────────────────────────────────────────────
uniform int       uStereoMode;    // 0=mono, 1=anaglyphe, 2=sbs
uniform sampler2D uSceneLeft;
uniform sampler2D uSceneRight;
uniform sampler2D uGlowLeft;
uniform sampler2D uGlowRight;

out vec4 fragColor;

void main() {
    // uScene = particules sur fond noir (beginFrame clear=0)
    // uGlow  = blur(uScene) → uniquement le halo des particules
    // Fond bleu natif ajouté ici, jamais passé dans le blur
    vec3 bg = vec3(0.031, 0.063, 0.188);

    if (uStereoMode == 0) {
        // ── MONO (standard) ─────────────────────────────────────────────
        vec4 sceneColor = texture(uScene, vTexCoord);
        vec4 glowColor  = texture(uGlow,  vTexCoord);
        fragColor = vec4(bg, 1.0) + sceneColor + vec4(glowColor.rgb * uGlowIntensity, 0.0);
    }
    else if (uStereoMode == 1) {
        // ── ANAGLYPHE : R←gauche, GB←droit ──────────────────────────────
        // Rouge = canal L (profondeur)
        // Cyan = canaux R (relief)
        vec4 sceneL = texture(uSceneLeft,  vTexCoord);
        vec4 glowL  = texture(uGlowLeft,   vTexCoord);
        vec4 sceneR = texture(uSceneRight, vTexCoord);
        vec4 glowR  = texture(uGlowRight,  vTexCoord);

        vec3 colorL = bg + sceneL.rgb + glowL.rgb * uGlowIntensity;
        vec3 colorR = bg + sceneR.rgb + glowR.rgb * uGlowIntensity;

        float redL = dot(colorL, vec3(0.299, 0.587, 0.114)); // luminance gauche
        float redR = dot(colorR, vec3(0.299, 0.587, 0.114)); // luminance droite

        // Anaglyphe classique : R de gauche, G et B de droite
        // pour une meilleure fusion perceptive
        fragColor = vec4(redL, colorR.g, colorR.b, 1.0);
    }
    else {
        // ── SIDE-BY-SIDE : deux demi-images ───────────────────────────────
        if (vTexCoord.x < 0.5) {
            // Moitié gauche = œil gauche (scale x2)
            vec2 uv = vec2(vTexCoord.x * 2.0, vTexCoord.y);
            vec4 sceneColor = texture(uSceneLeft, uv);
            vec4 glowColor  = texture(uGlowLeft,  uv);
            fragColor = vec4(bg, 1.0) + sceneColor + vec4(glowColor.rgb * uGlowIntensity, 0.0);
        } else {
            // Moitié droite = œil droit (scale x2, offset -0.5)
            vec2 uv = vec2((vTexCoord.x - 0.5) * 2.0, vTexCoord.y);
            vec4 sceneColor = texture(uSceneRight, uv);
            vec4 glowColor  = texture(uGlowRight,  uv);
            fragColor = vec4(bg, 1.0) + sceneColor + vec4(glowColor.rgb * uGlowIntensity, 0.0);
        }
    }
}