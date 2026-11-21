// ============================================================
// Janus Gravity (Jean-Pierre Petit)
// F_ij = G * m_j * r_ij / |r|^3 * sgn(m_i)
// Positive mass attracts positive mass (Newton)
// Negative mass repels negative mass (anti-Newton)
// Opposite masses repel each other
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

    // Janus: force direction depends on sign of receiver mass
    float sign_i = (pi.w > 0.0f) ? 1.0f : -1.0f;
    float s = pj.w * invDistCube * sign_i;

    ai.x += dx * s;
    ai.y += dy * s;
    ai.z += dz * s;
    return ai;
}
