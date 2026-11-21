// ============================================================
// Newtonian Gravity
// F_ij = G * m_j * r_ij / |r|^3
// Standard inverse-square law (inverse-cube for acceleration)
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
    float s = pj.w * invDistCube;
    ai.x += dx * s;
    ai.y += dy * s;
    ai.z += dz * s;
    return ai;
}
