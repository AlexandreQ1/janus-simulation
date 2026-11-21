static float4 pair_force(
    float4 pi,
    float4 pj,
    float4 ai,
    const unsigned int softening)
{
    float3 r;
    r.x = pj.x - pi.x;
    r.y = pj.y - pi.y;
    r.z = pj.z - pi.z;

    float distSquare = r.x * r.x + r.y * r.y + r.z * r.z + softening;
    float invDist = native_rsqrt(distSquare);
    float invDistCube = invDist * invDist * invDist;
    float s = pj.w * invDistCube;
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

kernel void nbody(
    __global float4* p_start,
    __global float4* p_end,
    __global float4* v_start,
    __global float4* v_end,
    __global float4* a,
    __local float4 *cached_stars,
    const unsigned int softening,
    const float timestep,
    const float G,
    const unsigned int N,
    const unsigned int unused_approx,
    const float sim_time,
    const float sp_amplitude,
    const float sp_pattern_speed,
    const float sp_pitch_tan,
    const unsigned int sp_arms)
{
    unsigned int globalid = get_global_id(0);
    unsigned int chunksize = get_local_size(0);
    unsigned int localid = get_local_id(0);
    unsigned int threadcount = get_global_size(0);

    for (;globalid < N; globalid += threadcount)
    {
        float4 pos = p_start[globalid];
        float4 vel = v_start[globalid];

        float4 force = {0,0,0,0};

        // N² complet — seul mode valide pour le modele Janus (masses mixtes +/-)
        int chunk = 0;
        for (int i = 0; i < N; i += chunksize, chunk++)
        {
            int local_pos = chunk * chunksize + localid;
            if (local_pos < N)
                cached_stars[localid] = p_start[local_pos];
            else
                cached_stars[localid] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

            barrier(CLK_LOCAL_MEM_FENCE);
            for (int j = 0; j < chunksize;)
            {
                force = pair_force(pos, cached_stars[j++], force, softening);
                force = pair_force(pos, cached_stars[j++], force, softening);
                force = pair_force(pos, cached_stars[j++], force, softening);
                force = pair_force(pos, cached_stars[j++], force, softening);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Janus eq. 4.4.10 : a_i = sgn(m_i) * G * sum_j [ m_j * (r_j - r_i) / dist^3 ]
        float sign_i = (pos.w >= 0.0f) ? 1.0f : -1.0f;

        // Lin-Shu density wave forcing
        if (pos.w > 0.0f && sp_amplitude > 0.0f)
        {
            float r_xy = native_sqrt(pos.x*pos.x + pos.y*pos.y);
            if (r_xy > 1.0f)
            {
                float theta = atan2(pos.y, pos.x);
                float phase = (float)sp_arms * (theta - sp_pitch_tan * native_log(r_xy) - sp_pattern_speed * sim_time);
                float wave  = sp_amplitude * native_sin(phase);
                float scale = wave / r_xy;
                force.x += pos.x * scale;
                force.y += pos.y * scale;
            }
        }

        // ── Leapfrog symplectique (drift-kick) ──────────────────────────────
        // vel contient v_{n+1/2} (vitesse à demi-pas)
        // 1. Drift : pos_{n+1} = pos_n + v_{n+1/2} * dt
        pos.x += vel.x * timestep;
        pos.y += vel.y * timestep;
        pos.z += vel.z * timestep;

        // 2. Kick : v_{n+3/2} = v_{n+1/2} + a_n * dt
        //            où a_n = sign_i * force * G (force calculée sur pos_n)
        vel.x += sign_i * force.x * G * timestep;
        vel.y += sign_i * force.y * G * timestep;
        vel.z += sign_i * force.z * G * timestep;

        p_end[globalid] = pos;
        v_end[globalid] = vel;
    }
}
