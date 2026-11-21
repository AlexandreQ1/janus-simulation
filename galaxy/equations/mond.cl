// ============================================================
// MOND (Modified Newtonian Dynamics)
// F_ij = G * m_j * r_ij / |r|^3 * mu(a/a0)
// where mu(x) = x / sqrt(1 + x^2)  (standard interpolating function)
// a0 = critical acceleration threshold
// When a >> a0 : mu -> 1  (Newtonian regime)
// When a << a0 : mu -> a/a0  (MOND regime)
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

    // MOND parameters (tune a0 to match your simulation units)
    const float a0 = 1.2e-10f;

    // Newtonian acceleration magnitude for this pair
    float a_newton = fabsf(pj.w) * invDistCube * dist;

    // MOND interpolating function: mu(x) = x / sqrt(1 + x^2)
    float x  = a_newton / a0;
    float mu = x * rsqrtf(1.0f + x * x);

    float s = pj.w * invDistCube * mu;

    ai.x += dx * s;
    ai.y += dy * s;
    ai.z += dz * s;
    return ai;
}
