#version 330 core

in  vec2 vTexCoord;

uniform sampler2D uTex;
uniform vec2      uDir;   // pass (1/width, 0) or (0, 1/height)

out vec4 fragColor;

// 9-tap Gaussian kernel (sigma ~2)
// Weights (unnormalised): 1 8 28 56 70 56 28 8 1  -> sum = 256
const float weight[5] = float[](70.0/256.0,
                                56.0/256.0,
                                28.0/256.0,
                                 8.0/256.0,
                                 1.0/256.0);

void main() {
    vec4 result = texture(uTex, vTexCoord) * weight[0];

    for (int i = 1; i < 5; ++i) {
        vec2 offset = uDir * float(i);
        result += texture(uTex, vTexCoord + offset) * weight[i];
        result += texture(uTex, vTexCoord - offset) * weight[i];
    }

    fragColor = result;
}
