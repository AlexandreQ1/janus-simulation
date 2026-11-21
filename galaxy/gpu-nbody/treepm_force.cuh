// treepm_force.cuh — GPU Barnes-Hut octree as short-range replacement for P³M.
//
// Approach: Linear BVH (LBVH, Karras 2012) built from sorted Morton codes.
// For Janus, builds TWO separate trees (one per species) with unsigned masses,
// then combines forces with proper Janus signs at the particle level. This
// avoids the ρ_net cancellation problem that plagues PM in mixed regions.
//
// Pipeline per step:
//   1. Compute Morton codes for all particles
//   2. Partition by species (+/-), sort each partition by Morton code
//   3. Build LBVH tree for each species (Karras 2012)
//   4. Bottom-up: compute total mass + COM per internal node
//   5. For each particle: traverse both trees, apply Janus-aware force kick

#pragma once
#include <cuda_runtime.h>

struct TreeContext {
    // Particle partitions by species (+ and -)
    int        N_pos;       // count of + particles
    int        N_neg;       // count of - particles
    int        N_total;     // = N_pos + N_neg

    // Sorted positions per species (sorted by Morton code)
    float4*    d_sorted_pos_pos;   // [N_pos]  positives sorted
    float4*    d_sorted_pos_neg;   // [N_neg]  negatives sorted
    int*       d_orig_idx_pos;     // [N_pos]  original index in global array
    int*       d_orig_idx_neg;     // [N_neg]  original index in global array
    unsigned long long* d_codes_pos;  // [N_pos] Morton codes
    unsigned long long* d_codes_neg;  // [N_neg] Morton codes

    // LBVH tree storage (per-species). Trees have N leaves + (N-1) internal nodes.
    // We use a flat array of 2N-1 nodes; nodes [0, N-2] are internal, [N-1, 2N-2] are leaves.
    int*       d_tree_parent_pos;    // [2*N_pos - 1]   parent index
    int*       d_tree_left_pos;      // [N_pos - 1]     left child of internal node
    int*       d_tree_right_pos;     // [N_pos - 1]     right child of internal node
    float4*    d_tree_com_pos;       // [2*N_pos - 1]   (com.x, com.y, com.z, mass)
    float*     d_tree_size_pos;      // [2*N_pos - 1]   bounding-box half-extent
    int*       d_tree_visit_pos;     // [N_pos - 1]     atomic visit counter for bottom-up
    float4*    d_tree_quadA_pos;     // [2*N_pos - 1]   reduced quadrupole (Qxx,Qyy,Qzz,_)
    float4*    d_tree_quadB_pos;     // [2*N_pos - 1]   reduced quadrupole (Qxy,Qxz,Qyz,_)

    int*       d_tree_parent_neg;
    int*       d_tree_left_neg;
    int*       d_tree_right_neg;
    float4*    d_tree_com_neg;
    float*     d_tree_size_neg;
    int*       d_tree_visit_neg;
    float4*    d_tree_quadA_neg;
    float4*    d_tree_quadB_neg;

    // Persistent scratch buffers (avoid cudaMalloc/Free each step)
    float*     d_bbmin;       // [3] global min  (x, y, z)
    float*     d_bbmax;       // [3] global max
    int*       d_cnt_pos;     // [1] counter for + species
    int*       d_cnt_neg;     // [1] counter for - species

    // Allocation capacities (in particles) — buffers sized for worst case
    int        capacity;

    // Bounding box of the whole particle distribution (recomputed each step)
    float3     bbox_min;
    float3     bbox_max;

    // Tree parameters
    float      theta;        // Barnes-Hut opening angle (default 0.5)
    float      softening;    // Plummer softening (Mpc)
    bool       use_quadrupole;    // order-2 multipole correction (default true)
    bool       use_warp_coherent; // warp-coherent traversal (default true)

    bool       initialized;
};

void tree_init(TreeContext& ctx, int N_capacity, float softening);
void tree_free(TreeContext& ctx);

// Drop-in replacement for pm_janus_sr_kick.
// Builds both trees from current positions, then applies tree force kick
// to all particles. Janus-aware: + and - feel correct attraction/repulsion.
void tree_sr_kick(TreeContext& ctx,
                  const float4* d_pos, float4* d_vel,
                  unsigned int N, float G, float a_inv3, float dt,
                  int physicsModel);

// Pure drift : p_out = p_in + v * dt   (no force, just position update).
// Useful for Galaxy-mode where the integrator is split kick / drift.
void tree_drift_full(const float4* d_pos_in, float4* d_pos_out,
                     const float4* d_vel, unsigned int N, float dt);
