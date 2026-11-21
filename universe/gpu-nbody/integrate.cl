kernel void integrate(
    __global float4*       p_start,
    __global float4*       p_end,
    __global float4*       v_start,
    __global float4*       v_end,
    __global const float4* bh_forces,
    const float            G,
    const float            dt,
    const unsigned int     N,
    const float            sim_time,
    const float            sp_amplitude,
    const float            sp_pattern_speed,
    const float            sp_pitch_tan,
    const unsigned int     sp_arms
)
{
    unsigned int i = get_global_id(0);
    if (i >= N) return;

    float4 pos = p_start[i];
    float4 vel = v_start[i];
    float4 f   = bh_forces[i];

    float sign_i = (pos.w >= 0.0f) ? 1.0f : -1.0f;

    if (pos.w > 0.0f && sp_amplitude > 0.0f)
    {
        float r_xy = native_sqrt(pos.x*pos.x + pos.y*pos.y);
        if (r_xy > 1.0f)
        {
            float theta = atan2(pos.y, pos.x);
            float phase = (float)sp_arms * (theta
                          - sp_pitch_tan * native_log(r_xy)
                          - sp_pattern_speed * sim_time);
            float wave  = sp_amplitude * native_sin(phase);
            float scale = wave / r_xy;
            f.x += pos.x * scale;
            f.y += pos.y * scale;
        }
    }

    vel.x += sign_i * f.x * G * dt;
    vel.y += sign_i * f.y * G * dt;
    vel.z += sign_i * f.z * G * dt;

    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;

    p_end[i] = pos;
    v_end[i] = vel;
}
