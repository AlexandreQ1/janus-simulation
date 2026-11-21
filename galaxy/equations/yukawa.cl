// ============================================================
// Yukawa Gravity (Finite-range force)
// F_ij = G * m_j * r_ij / |r|^3 * (1 + alpha * exp(-r/lambda))
// Adds a short-range correction to Newtonian gravity
// alpha  = strength of correction (0.0 = pure Newton)
// lambda = range of the correction (in simulation units)
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
    float dist = sqrtf(distSq);
    float invDist = rsqrtf(distSq);
    float invDistCube = invDist * invDist * invDist;

    // Yukawa correction parameters (tune these values)
    const float alpha  = 0.5f;     // strength of correction
    const float lambda = 1000.0f;  // range in simulation units

    float yukawa = 1.0f + alpha * expf(-dist / lambda);
    float s = pj.w * invDistCube * yukawa;

    ai.x += dx * s;
    ai.y += dy * s;
    ai.z += dz * s;
    return ai;
}
