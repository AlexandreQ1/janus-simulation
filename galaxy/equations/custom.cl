// ============================================================
// Custom Gravity Model — edit this file, then edit case 4 in
// gpu-nbody/pair3.cu and recompile to apply on GPU.
//
// Available variables:
//   pi = particle i (x,y,z position + mass in .w)
//   pj = particle j (x,y,z position + mass in .w)
//   ai = accumulated acceleration for particle i
//   softening = softening length squared (ε²)
//
// CUDA math functions available:
//   sqrtf(x)   rsqrtf(x)   expf(x)
//   sinf(x)    cosf(x)     logf(x)
//   fabsf(x)   fmaxf(x,y)  fminf(x,y)
//
// NOTE: this file is reference documentation only.
// The GPU runs the code in gpu-nbody/pair3.cu (case 4).
// ============================================================

static float4 pair_force(
    float4 pi,
    float4 pj,
    float4 ai,
    float softening)
{
    float dx = pj.x - pi.x;
    float dy = pj.y - pi.y;
    float dz = pj.z - pi.z;

    float distSq = dx*dx + dy*dy + dz*dz + softening;
    float invDist = rsqrtf(distSq);
    float invDistCube = invDist * invDist * invDist;

    // ── YOUR CUSTOM EQUATION HERE ──────────────────────────
    // Example: Yukawa-style short-range correction
    // float dist = sqrtf(distSq);
    // float correction = 1.0f + 0.5f * expf(-dist / 500.0f);
    // float s = pj.w * invDistCube * correction;

    float s = pj.w * invDistCube;  // default: Newtonian
    // ───────────────────────────────────────────────────────

    ai.x += dx * s;
    ai.y += dy * s;
    ai.z += dz * s;
    return ai;
}
