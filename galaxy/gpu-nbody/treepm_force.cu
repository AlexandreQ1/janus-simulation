// treepm_force.cu — GPU Barnes-Hut octree (LBVH, Karras 2012) for Janus.
//
// Two-trees approach: one tree per species with unsigned masses. For each
// particle, traverse both trees and combine with Janus signs:
//   F_+(r) = Newtonian force from + particles (always attractive, |mass|)
//   F_-(r) = Newtonian force from - particles (always attractive, |mass|)
//   + particle accel = +F_+ − F_-  (attracted by +, repelled by -)
//   - particle accel = -F_+ + F_- = -(F_+ − F_-)
//   → general:  a_i = sign_i × (F_+ − F_-)   with sign_i = +1 for +, -1 for -

#include "treepm_force.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// ----------------------------------------------------------------------------
//  Atomic min/max on float (define BEFORE any kernel that uses them)
// ----------------------------------------------------------------------------
__device__ inline float atomicMinFloat(float* addr, float v) {
    int* addr_int = (int*)addr;
    int old = *addr_int, assumed;
    do {
        assumed = old;
        float curr = __int_as_float(assumed);
        if (v >= curr) break;
        old = atomicCAS(addr_int, assumed, __float_as_int(v));
    } while (old != assumed);
    return __int_as_float(old);
}

__device__ inline float atomicMaxFloat(float* addr, float v) {
    int* addr_int = (int*)addr;
    int old = *addr_int, assumed;
    do {
        assumed = old;
        float curr = __int_as_float(assumed);
        if (v <= curr) break;
        old = atomicCAS(addr_int, assumed, __float_as_int(v));
    } while (old != assumed);
    return __int_as_float(old);
}

// ----------------------------------------------------------------------------
//  Morton code utilities (64-bit, 21 bits per axis)
// ----------------------------------------------------------------------------
__device__ __host__ inline unsigned long long expandBits21(unsigned int v) {
    unsigned long long x = v & 0x1fffffull;
    x = (x | (x << 32)) & 0x1f00000000ffffull;
    x = (x | (x << 16)) & 0x1f0000ff0000ffull;
    x = (x | (x << 8))  & 0x100f00f00f00f00full;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3ull;
    x = (x | (x << 2))  & 0x1249249249249249ull;
    return x;
}

__device__ __host__ inline unsigned long long mortonCode64(float x, float y, float z) {
    x = fminf(fmaxf(x * 2097152.0f, 0.0f), 2097151.0f);
    y = fminf(fmaxf(y * 2097152.0f, 0.0f), 2097151.0f);
    z = fminf(fmaxf(z * 2097152.0f, 0.0f), 2097151.0f);
    unsigned int xi = (unsigned int)x;
    unsigned int yi = (unsigned int)y;
    unsigned int zi = (unsigned int)z;
    return (expandBits21(xi) << 2) | (expandBits21(yi) << 1) | expandBits21(zi);
}

// ----------------------------------------------------------------------------
//  Bounding box reduction (block-level then global atomic)
//
//  Each block reduces its chunk in shared memory → 1 atomic op per block per
//  coordinate (6 total per block) instead of 6 per thread. With 4096 blocks
//  for 1M particles, that's 24k atomics vs 6M → 250× less contention.
// ----------------------------------------------------------------------------
__global__ void bbox_block_reduce(
    const float4* __restrict__ pos, unsigned int N,
    float* d_bbmin, float* d_bbmax)
{
    __shared__ float s_min[3];
    __shared__ float s_max[3];

    int tid = threadIdx.x;
    if (tid < 3) { s_min[tid] = 1.0e30f; s_max[tid] = -1.0e30f; }
    __syncthreads();

    unsigned int gid = blockIdx.x * blockDim.x + tid;
    if (gid < N) {
        float4 p = pos[gid];
        atomicMinFloat(&s_min[0], p.x);
        atomicMinFloat(&s_min[1], p.y);
        atomicMinFloat(&s_min[2], p.z);
        atomicMaxFloat(&s_max[0], p.x);
        atomicMaxFloat(&s_max[1], p.y);
        atomicMaxFloat(&s_max[2], p.z);
    }
    __syncthreads();

    if (tid == 0) {
        atomicMinFloat(d_bbmin + 0, s_min[0]);
        atomicMinFloat(d_bbmin + 1, s_min[1]);
        atomicMinFloat(d_bbmin + 2, s_min[2]);
        atomicMaxFloat(d_bbmax + 0, s_max[0]);
        atomicMaxFloat(d_bbmax + 1, s_max[1]);
        atomicMaxFloat(d_bbmax + 2, s_max[2]);
    }
}

// ----------------------------------------------------------------------------
//  Compute Morton codes + partition by species (WARP-AGGREGATED ATOMICS)
//
//  Per warp: __ballot_sync to count +/- threads, one atomicAdd per warp per
//  species. Reduces atomic contention by 32× compared to per-thread atomics.
//  Each thread computes its slot within the warp using __popc on a prefix mask.
// ----------------------------------------------------------------------------
__global__ void compute_codes_split(
    const float4* __restrict__ pos, unsigned int N,
    float3 bbmin, float3 bbsize,
    float4* sorted_pos_pos, unsigned long long* codes_pos, int* orig_idx_pos,
    float4* sorted_pos_neg, unsigned long long* codes_neg, int* orig_idx_neg,
    int* counter_pos, int* counter_neg)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    bool active   = (gid < N);
    bool is_pos   = false;
    float4 p;
    unsigned long long code = 0;

    if (active) {
        p = pos[gid];
        is_pos = (p.w >= 0.0f);
        float nx = (p.x - bbmin.x) / fmaxf(bbsize.x, 1.0e-6f);
        float ny = (p.y - bbmin.y) / fmaxf(bbsize.y, 1.0e-6f);
        float nz = (p.z - bbmin.z) / fmaxf(bbsize.z, 1.0e-6f);
        code = mortonCode64(nx, ny, nz);
    }

    const unsigned FULL_MASK = 0xFFFFFFFFu;
    unsigned active_mask = __ballot_sync(FULL_MASK, active);
    unsigned pos_mask    = __ballot_sync(FULL_MASK, active && is_pos);
    unsigned neg_mask    = active_mask & ~pos_mask;

    int lane = threadIdx.x & 31;
    int warp_cnt_pos = __popc(pos_mask);
    int warp_cnt_neg = __popc(neg_mask);

    // Lane 0 of each warp performs the global atomic for its species count
    int warp_base_pos = 0;
    int warp_base_neg = 0;
    if (lane == 0) {
        if (warp_cnt_pos > 0) warp_base_pos = atomicAdd(counter_pos, warp_cnt_pos);
        if (warp_cnt_neg > 0) warp_base_neg = atomicAdd(counter_neg, warp_cnt_neg);
    }
    warp_base_pos = __shfl_sync(FULL_MASK, warp_base_pos, 0);
    warp_base_neg = __shfl_sync(FULL_MASK, warp_base_neg, 0);

    if (!active) return;

    // Compute this thread's offset within its species inside the warp
    unsigned lane_mask_below = (lane == 0) ? 0u : ((1u << lane) - 1u);
    if (is_pos) {
        int my_offset = __popc(pos_mask & lane_mask_below);
        int slot = warp_base_pos + my_offset;
        sorted_pos_pos[slot] = p;
        codes_pos[slot]      = code;
        orig_idx_pos[slot]   = (int)gid;   // map sorted slot → global index
    } else {
        int my_offset = __popc(neg_mask & lane_mask_below);
        int slot = warp_base_neg + my_offset;
        sorted_pos_neg[slot] = p;
        codes_neg[slot]      = code;
        orig_idx_neg[slot]   = (int)gid;
    }
}

// ----------------------------------------------------------------------------
//  Karras 2012 LBVH internal-node construction
// ----------------------------------------------------------------------------

// δ(i,j) = length of longest common prefix of codes[i] and codes[j].
// Returns -1 if j is out of range. Handles duplicate codes via index tie-break.
__device__ inline int deltaCP(const unsigned long long* codes, int N, int i, int j) {
    if (j < 0 || j >= N) return -1;
    unsigned long long ci = codes[i];
    unsigned long long cj = codes[j];
    if (ci == cj) return 64 + __clz((unsigned int)(i ^ j));
    return __clzll(ci ^ cj);
}

// For internal node i, determine the OTHER end of its range (besides i itself).
__device__ inline int determineRange(const unsigned long long* codes, int N, int i) {
    int dPlus  = deltaCP(codes, N, i, i + 1);
    int dMinus = deltaCP(codes, N, i, i - 1);
    int d = (dPlus > dMinus) ? 1 : -1;
    int deltaMin = (d == 1) ? dMinus : dPlus;

    int lmax = 2;
    while (deltaCP(codes, N, i, i + lmax * d) > deltaMin) lmax *= 2;

    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2) {
        if (deltaCP(codes, N, i, i + (l + t) * d) > deltaMin) l += t;
    }
    return i + l * d;
}

// Within range [first, last], find split position (last leaf of the left subtree).
__device__ inline int findSplit(const unsigned long long* codes, int first, int last) {
    if (first == last) return first;
    int commonPrefix;
    unsigned long long cf = codes[first];
    unsigned long long cl = codes[last];
    if (cf == cl) commonPrefix = 64 + __clz((unsigned int)(first ^ last));
    else          commonPrefix = __clzll(cf ^ cl);

    int split = first;
    int step  = last - first;
    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;
        if (newSplit < last) {
            int splitPrefix;
            unsigned long long cn = codes[newSplit];
            if (cf == cn) splitPrefix = 64 + __clz((unsigned int)(first ^ newSplit));
            else          splitPrefix = __clzll(cf ^ cn);
            if (splitPrefix > commonPrefix) split = newSplit;
        }
    } while (step > 1);
    return split;
}

// Build all N-1 internal nodes in parallel. Leaves are at indices [N-1, 2N-2].
__global__ void karras_build_internal(
    const unsigned long long* __restrict__ codes, int N,
    int* parent, int* leftCh, int* rightCh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N - 1) return;

    int j     = determineRange(codes, N, i);
    int first = min(i, j);
    int last  = max(i, j);
    int split = findSplit(codes, first, last);

    int L = (split     == first) ? (N - 1 + split)     : split;
    int R = (split + 1 == last)  ? (N - 1 + split + 1) : (split + 1);

    leftCh[i]  = L;
    rightCh[i] = R;
    parent[L] = i;
    parent[R] = i;
}

__global__ void mark_root_parent(int* parent) {
    if (blockIdx.x == 0 && threadIdx.x == 0) parent[0] = -1;
}

// ----------------------------------------------------------------------------
//  Multipole computation: leaves + bottom-up sweep
// ----------------------------------------------------------------------------

__global__ void init_leaf_multipoles(
    const float4* __restrict__ sorted_pos, int N,
    float4* com, float* nsize, float4* quadA, float4* quadB)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int leafIdx = (N - 1) + i;
    float4 p = sorted_pos[i];
    com[leafIdx]   = make_float4(p.x, p.y, p.z, fabsf(p.w));
    nsize[leafIdx] = 0.0f;
    // Single particle sits at its own COM → zero reduced quadrupole
    quadA[leafIdx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    quadB[leafIdx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__global__ void zero_visit_counts(int* visit, int N_internal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_internal) visit[i] = 0;
}

// Each thread starts at a leaf and walks up. Only the SECOND child to reach
// a node combines (using atomic visit counter).
__global__ void multipoles_bottom_up(
    int N,
    const int* __restrict__ parent,
    const int* __restrict__ leftCh,
    const int* __restrict__ rightCh,
    float4* com, float* nsize,
    float4* quadA, float4* quadB,
    int* visit)
{
    int leaf = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf >= N) return;
    int node = parent[(N - 1) + leaf];
    while (node >= 0) {
        int old = atomicAdd(&visit[node], 1);
        if (old == 0) return;  // first arrival → bail

        int L = leftCh[node];
        int R = rightCh[node];
        float4 cL = com[L];
        float4 cR = com[R];
        float mTot = cL.w + cR.w;
        float invM = (mTot > 1.0e-12f) ? (1.0f / mTot) : 0.0f;
        float cx = (cL.x * cL.w + cR.x * cR.w) * invM;
        float cy = (cL.y * cL.w + cR.y * cR.w) * invM;
        float cz = (cL.z * cL.w + cR.z * cR.w) * invM;
        com[node] = make_float4(cx, cy, cz, mTot);

        float dxL = cL.x - cx, dyL = cL.y - cy, dzL = cL.z - cz;
        float dxR = cR.x - cx, dyR = cR.y - cy, dzR = cR.z - cz;
        float sL = nsize[L] + sqrtf(dxL*dxL + dyL*dyL + dzL*dzL);
        float sR = nsize[R] + sqrtf(dxR*dxR + dyR*dyR + dzR*dzR);
        nsize[node] = fmaxf(sL, sR);

        // ── Reduced quadrupole via parallel-axis theorem ──────────────────
        // Q_parent = Σ_child [ Q_child + m_child (3 s_i s_j − δ_ij |s|²) ]
        // with s = child_COM − parent_COM. Stays traceless.
        float4 qAL = quadA[L], qBL = quadB[L];
        float4 qAR = quadA[R], qBR = quadB[R];
        float s2L = dxL*dxL + dyL*dyL + dzL*dzL;
        float s2R = dxR*dxR + dyR*dyR + dzR*dzR;
        float mL = cL.w, mR = cR.w;
        float Qxx = qAL.x + mL*(3.0f*dxL*dxL - s2L) + qAR.x + mR*(3.0f*dxR*dxR - s2R);
        float Qyy = qAL.y + mL*(3.0f*dyL*dyL - s2L) + qAR.y + mR*(3.0f*dyR*dyR - s2R);
        float Qzz = qAL.z + mL*(3.0f*dzL*dzL - s2L) + qAR.z + mR*(3.0f*dzR*dzR - s2R);
        float Qxy = qBL.x + mL*(3.0f*dxL*dyL)       + qBR.x + mR*(3.0f*dxR*dyR);
        float Qxz = qBL.y + mL*(3.0f*dxL*dzL)       + qBR.y + mR*(3.0f*dxR*dzR);
        float Qyz = qBL.z + mL*(3.0f*dyL*dzL)       + qBR.z + mR*(3.0f*dyR*dzR);
        quadA[node] = make_float4(Qxx, Qyy, Qzz, 0.0f);
        quadB[node] = make_float4(Qxy, Qxz, Qyz, 0.0f);

        node = parent[node];
    }
}

// ----------------------------------------------------------------------------
//  Tree traversal (Barnes-Hut) — per particle, both trees
// ----------------------------------------------------------------------------
__global__ void tree_traverse_and_kick(
    const float4* __restrict__ pos, unsigned int N_total,
    int N_pos, const int* leftP, const int* rightP,
    const float4* comP, const float* sizeP,
    const float4* quadAP, const float4* quadBP,
    int N_neg, const int* leftN, const int* rightN,
    const float4* comN, const float* sizeN,
    const float4* quadAN, const float4* quadBN,
    float theta_sq, float soft2,
    float4* vel, float kick_coeff, int physicsModel, int use_quad)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N_total) return;
    float4 pi = pos[gid];

    float3 Fpos = {0.0f, 0.0f, 0.0f};
    float3 Fneg = {0.0f, 0.0f, 0.0f};
    int stack[128];

    // --- + tree ---
    if (N_pos > 0) {
        int sp = 0;
        if (N_pos == 1) {
            stack[sp++] = 0;  // single leaf
        } else {
            stack[sp++] = 0;  // root internal node
        }
        while (sp > 0) {
            int node = stack[--sp];
            bool isLeaf = (node >= N_pos - 1);
            float4 c = comP[node];
            if (c.w < 1.0e-12f) continue;
            float dx = c.x - pi.x;
            float dy = c.y - pi.y;
            float dz = c.z - pi.z;
            float r2 = dx*dx + dy*dy + dz*dz;
            if (isLeaf) {
                if (r2 < 1.0e-10f) continue;  // skip self/coincident
                float ir = rsqrtf(r2 + soft2);
                float ir3 = ir * ir * ir;
                float fmag = c.w * ir3;
                Fpos.x += fmag * dx;
                Fpos.y += fmag * dy;
                Fpos.z += fmag * dz;
            } else {
                float s = sizeP[node];
                if (s * s < theta_sq * r2) {
                    float ir = rsqrtf(r2 + soft2);
                    float ir3 = ir * ir * ir;
                    float fmag = c.w * ir3;
                    Fpos.x += fmag * dx;
                    Fpos.y += fmag * dy;
                    Fpos.z += fmag * dz;
                    if (use_quad) {
                        float4 qa = quadAP[node], qb = quadBP[node];
                        float ir2 = ir*ir, ir5 = ir3*ir2, ir7 = ir5*ir2;
                        float QRx = qa.x*dx + qb.x*dy + qb.y*dz;
                        float QRy = qb.x*dx + qa.y*dy + qb.z*dz;
                        float QRz = qb.y*dx + qb.z*dy + qa.z*dz;
                        float RQR = dx*QRx + dy*QRy + dz*QRz;
                        float c1 = 2.5f * RQR * ir7;
                        Fpos.x += -QRx*ir5 + c1*dx;
                        Fpos.y += -QRy*ir5 + c1*dy;
                        Fpos.z += -QRz*ir5 + c1*dz;
                    }
                } else if (sp < 126) {
                    stack[sp++] = leftP[node];
                    stack[sp++] = rightP[node];
                }
            }
        }
    }

    // --- - tree ---
    if (N_neg > 0) {
        int sp = 0;
        stack[sp++] = 0;
        while (sp > 0) {
            int node = stack[--sp];
            bool isLeaf = (node >= N_neg - 1);
            float4 c = comN[node];
            if (c.w < 1.0e-12f) continue;
            float dx = c.x - pi.x;
            float dy = c.y - pi.y;
            float dz = c.z - pi.z;
            float r2 = dx*dx + dy*dy + dz*dz;
            if (isLeaf) {
                if (r2 < 1.0e-10f) continue;
                float ir = rsqrtf(r2 + soft2);
                float ir3 = ir * ir * ir;
                float fmag = c.w * ir3;
                Fneg.x += fmag * dx;
                Fneg.y += fmag * dy;
                Fneg.z += fmag * dz;
            } else {
                float s = sizeN[node];
                if (s * s < theta_sq * r2) {
                    float ir = rsqrtf(r2 + soft2);
                    float ir3 = ir * ir * ir;
                    float fmag = c.w * ir3;
                    Fneg.x += fmag * dx;
                    Fneg.y += fmag * dy;
                    Fneg.z += fmag * dz;
                    if (use_quad) {
                        float4 qa = quadAN[node], qb = quadBN[node];
                        float ir2 = ir*ir, ir5 = ir3*ir2, ir7 = ir5*ir2;
                        float QRx = qa.x*dx + qb.x*dy + qb.y*dz;
                        float QRy = qb.x*dx + qa.y*dy + qb.z*dz;
                        float QRz = qb.y*dx + qb.z*dy + qa.z*dz;
                        float RQR = dx*QRx + dy*QRy + dz*QRz;
                        float c1 = 2.5f * RQR * ir7;
                        Fneg.x += -QRx*ir5 + c1*dx;
                        Fneg.y += -QRy*ir5 + c1*dy;
                        Fneg.z += -QRz*ir5 + c1*dz;
                    }
                } else if (sp < 126) {
                    stack[sp++] = leftN[node];
                    stack[sp++] = rightN[node];
                }
            }
        }
    }

    // Janus combination: a_i = sign_i × (F_+ − F_-)
    float sign_i = (physicsModel == 1 && pi.w < 0.0f) ? -1.0f : 1.0f;
    float4 v = vel[gid];
    v.x += kick_coeff * sign_i * (Fpos.x - Fneg.x);
    v.y += kick_coeff * sign_i * (Fpos.y - Fneg.y);
    v.z += kick_coeff * sign_i * (Fpos.z - Fneg.z);
    vel[gid] = v;
}

// ============================================================================
//  Warp-coherent traversal (Burtscher & Pingali 2011, Bédorf et al. 2012)
//
//  All 32 lanes of a warp walk ONE shared traversal path. A node is read once
//  per warp (cache-coalesced) instead of 32× independently. The warp descends
//  if ANY lane fails the MAC; lanes that would have accepted then process the
//  children instead — same physics, just (slightly) more accurate for those
//  lanes, with far less memory traffic. 2-3× faster in dense regions.
// ============================================================================
#define WC_WARPS 8     // block of 256 threads / 32
#define WC_STACK 64    // per-warp DFS stack depth (tree height + margin)

__device__ inline void warp_traverse_tree(
    float3 pi, int N,
    const int* __restrict__ left, const int* __restrict__ right,
    const float4* __restrict__ com, const float* __restrict__ sizeArr,
    const float4* __restrict__ quadA, const float4* __restrict__ quadB,
    float theta_sq, float soft2, int use_quad,
    int* stack /*shared, per-warp*/, int lane, float3& F)
{
    if (N <= 0) return;
    const unsigned FULL = 0xffffffffu;

    int sp = 0;
    if (lane == 0) stack[sp++] = 0;       // root (leaf if N==1)
    sp = __shfl_sync(FULL, sp, 0);

    while (sp > 0) {
        int node;
        if (lane == 0) node = stack[--sp];
        node = __shfl_sync(FULL, node, 0);  // broadcast the shared node

        float4 c = com[node];               // one read, broadcast across lanes
        bool valid = (c.w >= 1.0e-12f);
        float dx = c.x - pi.x, dy = c.y - pi.y, dz = c.z - pi.z;
        float r2 = dx*dx + dy*dy + dz*dz;
        bool isLeaf = (node >= N - 1);

        if (isLeaf) {
            if (valid && r2 >= 1.0e-10f) {  // skip self/coincident per lane
                float ir = rsqrtf(r2 + soft2);
                float ir3 = ir * ir * ir;
                float fmag = c.w * ir3;
                F.x += fmag*dx; F.y += fmag*dy; F.z += fmag*dz;
            }
        } else {
            float s = sizeArr[node];
            bool accept      = valid && (s*s < theta_sq * r2);
            bool wantDescend = valid && !accept;
            // Descend together if ANY lane needs finer resolution.
            if (!__any_sync(FULL, wantDescend)) {
                if (valid && accept) {
                    float ir = rsqrtf(r2 + soft2);
                    float ir3 = ir * ir * ir;
                    float fmag = c.w * ir3;
                    F.x += fmag*dx; F.y += fmag*dy; F.z += fmag*dz;
                    if (use_quad) {
                        float4 qa = quadA[node], qb = quadB[node];
                        float ir2 = ir*ir, ir5 = ir3*ir2, ir7 = ir5*ir2;
                        float QRx = qa.x*dx + qb.x*dy + qb.y*dz;
                        float QRy = qb.x*dx + qa.y*dy + qb.z*dz;
                        float QRz = qb.y*dx + qb.z*dy + qa.z*dz;
                        float RQR = dx*QRx + dy*QRy + dz*QRz;
                        float c1 = 2.5f * RQR * ir7;
                        F.x += -QRx*ir5 + c1*dx;
                        F.y += -QRy*ir5 + c1*dy;
                        F.z += -QRz*ir5 + c1*dz;
                    }
                }
            } else if (valid) {
                if (lane == 0 && sp < WC_STACK - 2) {
                    stack[sp++] = left[node];
                    stack[sp++] = right[node];
                }
            }
        }
        sp = __shfl_sync(FULL, sp, 0);      // keep loop bound warp-uniform
    }
}

// Processes ONE species in Morton-sorted order so the 32 lanes of a warp hold
// spatially-adjacent particles (the prerequisite for warp coherence to pay off).
// Writes the kick back to the GLOBAL velocity array via origIdx[slot].
__global__ void tree_traverse_and_kick_warp(
    const float4* __restrict__ sortedThis, const int* __restrict__ origIdx, int N_this,
    int N_pos, const int* leftP, const int* rightP,
    const float4* comP, const float* sizeP,
    const float4* quadAP, const float4* quadBP,
    int N_neg, const int* leftN, const int* rightN,
    const float4* comN, const float* sizeN,
    const float4* quadAN, const float4* quadBN,
    float theta_sq, float soft2,
    float4* vel, float kick_coeff, int physicsModel, int use_quad)
{
    __shared__ int s_stack[WC_WARPS][WC_STACK];
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = (threadIdx.x >> 5);
    int lane   = (threadIdx.x & 31);
    int* stack = s_stack[warpId];

    // Inactive lanes (slot >= N_this) must still join warp votes → no early return.
    bool active = (slot < (unsigned)N_this);
    float4 pself = active ? sortedThis[slot] : make_float4(0,0,0,0);
    float3 pi = make_float3(pself.x, pself.y, pself.z);

    float3 Fpos = make_float3(0,0,0);
    float3 Fneg = make_float3(0,0,0);

    warp_traverse_tree(pi, N_pos, leftP, rightP, comP, sizeP,
                       quadAP, quadBP, theta_sq, soft2, use_quad, stack, lane, Fpos);
    warp_traverse_tree(pi, N_neg, leftN, rightN, comN, sizeN,
                       quadAN, quadBN, theta_sq, soft2, use_quad, stack, lane, Fneg);

    if (active) {
        int gid = origIdx[slot];
        float sign_i = (physicsModel == 1 && pself.w < 0.0f) ? -1.0f : 1.0f;
        float4 v = vel[gid];
        v.x += kick_coeff * sign_i * (Fpos.x - Fneg.x);
        v.y += kick_coeff * sign_i * (Fpos.y - Fneg.y);
        v.z += kick_coeff * sign_i * (Fpos.z - Fneg.z);
        vel[gid] = v;
    }
}

// ============================================================================
//  Host helpers
// ============================================================================

void tree_init(TreeContext& ctx, int N_capacity, float softening) {
    ctx.capacity    = N_capacity;
    ctx.softening   = softening;
    ctx.theta       = 0.5f;
    ctx.use_quadrupole = true;
    ctx.use_warp_coherent = true;
    ctx.initialized = true;
    ctx.N_pos = ctx.N_neg = ctx.N_total = 0;
    ctx.d_sorted_pos_pos = nullptr; ctx.d_sorted_pos_neg = nullptr;
    ctx.d_orig_idx_pos = nullptr;   ctx.d_orig_idx_neg = nullptr;
    ctx.d_codes_pos = nullptr;      ctx.d_codes_neg = nullptr;
    ctx.d_tree_parent_pos = nullptr; ctx.d_tree_left_pos = nullptr; ctx.d_tree_right_pos = nullptr;
    ctx.d_tree_com_pos = nullptr;    ctx.d_tree_size_pos = nullptr; ctx.d_tree_visit_pos = nullptr;
    ctx.d_tree_quadA_pos = nullptr;  ctx.d_tree_quadB_pos = nullptr;
    ctx.d_tree_parent_neg = nullptr; ctx.d_tree_left_neg = nullptr; ctx.d_tree_right_neg = nullptr;
    ctx.d_tree_com_neg = nullptr;    ctx.d_tree_size_neg = nullptr; ctx.d_tree_visit_neg = nullptr;
    ctx.d_tree_quadA_neg = nullptr;  ctx.d_tree_quadB_neg = nullptr;
    ctx.d_bbmin = nullptr; ctx.d_bbmax = nullptr;
    ctx.d_cnt_pos = nullptr; ctx.d_cnt_neg = nullptr;

    int cap       = N_capacity;
    int treeNodes = (cap > 1) ? (2 * cap - 1) : 1;
    int internal  = (cap > 1) ? (cap - 1)     : 1;

    cudaMalloc(&ctx.d_sorted_pos_pos, cap * sizeof(float4));
    cudaMalloc(&ctx.d_sorted_pos_neg, cap * sizeof(float4));
    cudaMalloc(&ctx.d_codes_pos,      cap * sizeof(unsigned long long));
    cudaMalloc(&ctx.d_codes_neg,      cap * sizeof(unsigned long long));
    cudaMalloc(&ctx.d_orig_idx_pos,   cap * sizeof(int));
    cudaMalloc(&ctx.d_orig_idx_neg,   cap * sizeof(int));

    cudaMalloc(&ctx.d_tree_parent_pos, treeNodes * sizeof(int));
    cudaMalloc(&ctx.d_tree_left_pos,   internal  * sizeof(int));
    cudaMalloc(&ctx.d_tree_right_pos,  internal  * sizeof(int));
    cudaMalloc(&ctx.d_tree_com_pos,    treeNodes * sizeof(float4));
    cudaMalloc(&ctx.d_tree_size_pos,   treeNodes * sizeof(float));
    cudaMalloc(&ctx.d_tree_visit_pos,  internal  * sizeof(int));
    cudaMalloc(&ctx.d_tree_quadA_pos,  treeNodes * sizeof(float4));
    cudaMalloc(&ctx.d_tree_quadB_pos,  treeNodes * sizeof(float4));

    cudaMalloc(&ctx.d_tree_parent_neg, treeNodes * sizeof(int));
    cudaMalloc(&ctx.d_tree_left_neg,   internal  * sizeof(int));
    cudaMalloc(&ctx.d_tree_right_neg,  internal  * sizeof(int));
    cudaMalloc(&ctx.d_tree_com_neg,    treeNodes * sizeof(float4));
    cudaMalloc(&ctx.d_tree_size_neg,   treeNodes * sizeof(float));
    cudaMalloc(&ctx.d_tree_visit_neg,  internal  * sizeof(int));
    cudaMalloc(&ctx.d_tree_quadA_neg,  treeNodes * sizeof(float4));
    cudaMalloc(&ctx.d_tree_quadB_neg,  treeNodes * sizeof(float4));

    // Persistent small scratch — avoids per-step alloc/free fragmentation
    cudaMalloc(&ctx.d_bbmin,   3 * sizeof(float));
    cudaMalloc(&ctx.d_bbmax,   3 * sizeof(float));
    cudaMalloc(&ctx.d_cnt_pos, sizeof(int));
    cudaMalloc(&ctx.d_cnt_neg, sizeof(int));

    printf("[Tree] Initialized: capacity=%d, softening=%.1f, theta=%.2f\n",
           cap, softening, ctx.theta);
}

void tree_free(TreeContext& ctx) {
    if (!ctx.initialized) return;
#define FREE(p) if (p) { cudaFree(p); p = nullptr; }
    FREE(ctx.d_sorted_pos_pos); FREE(ctx.d_sorted_pos_neg);
    FREE(ctx.d_codes_pos);      FREE(ctx.d_codes_neg);
    FREE(ctx.d_orig_idx_pos);   FREE(ctx.d_orig_idx_neg);
    FREE(ctx.d_tree_parent_pos); FREE(ctx.d_tree_left_pos); FREE(ctx.d_tree_right_pos);
    FREE(ctx.d_tree_com_pos);    FREE(ctx.d_tree_size_pos); FREE(ctx.d_tree_visit_pos);
    FREE(ctx.d_tree_quadA_pos);  FREE(ctx.d_tree_quadB_pos);
    FREE(ctx.d_tree_parent_neg); FREE(ctx.d_tree_left_neg); FREE(ctx.d_tree_right_neg);
    FREE(ctx.d_tree_com_neg);    FREE(ctx.d_tree_size_neg); FREE(ctx.d_tree_visit_neg);
    FREE(ctx.d_tree_quadA_neg);  FREE(ctx.d_tree_quadB_neg);
    FREE(ctx.d_bbmin); FREE(ctx.d_bbmax);
    FREE(ctx.d_cnt_pos); FREE(ctx.d_cnt_neg);
#undef FREE
    ctx.initialized = false;
    printf("[Tree] Freed.\n");
}

static void buildOneTree(int N,
                         const float4* d_sorted_pos,
                         const unsigned long long* d_codes,
                         int* d_parent, int* d_left, int* d_right,
                         float4* d_com, float* d_size, int* d_visit,
                         float4* d_quadA, float4* d_quadB)
{
    if (N <= 0) return;
    dim3 block(256);

    if (N == 1) {
        // Single particle = single leaf at index 0. No internal nodes.
        init_leaf_multipoles<<<1, block>>>(d_sorted_pos, N, d_com, d_size, d_quadA, d_quadB);
        return;
    }

    int internal = N - 1;
    karras_build_internal<<<(internal + 255) / 256, block>>>(
        d_codes, N, d_parent, d_left, d_right);
    mark_root_parent<<<1, 1>>>(d_parent);

    init_leaf_multipoles<<<(N + 255) / 256, block>>>(d_sorted_pos, N, d_com, d_size, d_quadA, d_quadB);
    zero_visit_counts<<<(internal + 255) / 256, block>>>(d_visit, internal);
    multipoles_bottom_up<<<(N + 255) / 256, block>>>(
        N, d_parent, d_left, d_right, d_com, d_size, d_quadA, d_quadB, d_visit);
}

void tree_sr_kick(TreeContext& ctx,
                  const float4* d_pos, float4* d_vel,
                  unsigned int N, float G, float a_inv3, float dt,
                  int physicsModel)
{
    if (!ctx.initialized || N == 0) return;
    if ((int)N > ctx.capacity) {
        printf("[Tree] N=%u exceeds capacity=%d, skipping.\n", N, ctx.capacity);
        return;
    }

    // ---- 1. Bounding box (reuse persistent scratch) ----
    float h_bbmin[3] = {  1e30f,  1e30f,  1e30f };
    float h_bbmax[3] = { -1e30f, -1e30f, -1e30f };
    cudaMemcpy(ctx.d_bbmin, h_bbmin, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.d_bbmax, h_bbmax, 3 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256), grid_N((N + 255) / 256);
    bbox_block_reduce<<<grid_N, block>>>(d_pos, N, ctx.d_bbmin, ctx.d_bbmax);

    cudaMemcpy(h_bbmin, ctx.d_bbmin, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bbmax, ctx.d_bbmax, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    float3 bbmin  = make_float3(h_bbmin[0] - 1.0f, h_bbmin[1] - 1.0f, h_bbmin[2] - 1.0f);
    float3 bbsize = make_float3(
        (h_bbmax[0] - h_bbmin[0]) + 2.0f,
        (h_bbmax[1] - h_bbmin[1]) + 2.0f,
        (h_bbmax[2] - h_bbmin[2]) + 2.0f);

    // ---- 2. Compute codes + partition (reuse persistent counters) ----
    cudaMemset(ctx.d_cnt_pos, 0, sizeof(int));
    cudaMemset(ctx.d_cnt_neg, 0, sizeof(int));

    compute_codes_split<<<grid_N, block>>>(
        d_pos, N, bbmin, bbsize,
        ctx.d_sorted_pos_pos, ctx.d_codes_pos, ctx.d_orig_idx_pos,
        ctx.d_sorted_pos_neg, ctx.d_codes_neg, ctx.d_orig_idx_neg,
        ctx.d_cnt_pos, ctx.d_cnt_neg);

    int h_npos = 0, h_nneg = 0;
    cudaMemcpy(&h_npos, ctx.d_cnt_pos, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_nneg, ctx.d_cnt_neg, sizeof(int), cudaMemcpyDeviceToHost);

    ctx.N_pos = h_npos; ctx.N_neg = h_nneg; ctx.N_total = (int)N;

    // ---- 3. Sort each species by Morton code (thrust) ----
    // Sort by Morton key, permuting BOTH the positions and the global-index map
    // together (zip) so the warp kernel can write kicks back to vel[origIdx].
    if (h_npos > 1) {
        auto vals = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_ptr<float4>(ctx.d_sorted_pos_pos),
            thrust::device_ptr<int>(ctx.d_orig_idx_pos)));
        thrust::sort_by_key(thrust::device,
            thrust::device_ptr<unsigned long long>(ctx.d_codes_pos),
            thrust::device_ptr<unsigned long long>(ctx.d_codes_pos + h_npos),
            vals);
    }
    if (h_nneg > 1) {
        auto vals = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_ptr<float4>(ctx.d_sorted_pos_neg),
            thrust::device_ptr<int>(ctx.d_orig_idx_neg)));
        thrust::sort_by_key(thrust::device,
            thrust::device_ptr<unsigned long long>(ctx.d_codes_neg),
            thrust::device_ptr<unsigned long long>(ctx.d_codes_neg + h_nneg),
            vals);
    }

    // ---- 4. Build both trees ----
    buildOneTree(h_npos,
                 ctx.d_sorted_pos_pos, ctx.d_codes_pos,
                 ctx.d_tree_parent_pos, ctx.d_tree_left_pos, ctx.d_tree_right_pos,
                 ctx.d_tree_com_pos, ctx.d_tree_size_pos, ctx.d_tree_visit_pos,
                 ctx.d_tree_quadA_pos, ctx.d_tree_quadB_pos);
    buildOneTree(h_nneg,
                 ctx.d_sorted_pos_neg, ctx.d_codes_neg,
                 ctx.d_tree_parent_neg, ctx.d_tree_left_neg, ctx.d_tree_right_neg,
                 ctx.d_tree_com_neg, ctx.d_tree_size_neg, ctx.d_tree_visit_neg,
                 ctx.d_tree_quadA_neg, ctx.d_tree_quadB_neg);

    // ---- 5. Traverse + kick ----
    float theta_sq   = ctx.theta * ctx.theta;
    float soft2      = ctx.softening * ctx.softening;
    float kick_coeff = G * a_inv3 * dt;

    if (ctx.use_warp_coherent) {
        // Warp-coherent kernel needs a fixed 256-thread block (8 warps) to match
        // its static shared-memory stack (WC_WARPS × WC_STACK). Launched once per
        // species over the MORTON-SORTED arrays so warp lanes are spatially close.
        dim3 wblock(256);
        if (h_npos > 0) {
            dim3 wgrid((h_npos + 255) / 256);
            tree_traverse_and_kick_warp<<<wgrid, wblock>>>(
                ctx.d_sorted_pos_pos, ctx.d_orig_idx_pos, h_npos,
                h_npos, ctx.d_tree_left_pos, ctx.d_tree_right_pos,
                ctx.d_tree_com_pos, ctx.d_tree_size_pos,
                ctx.d_tree_quadA_pos, ctx.d_tree_quadB_pos,
                h_nneg, ctx.d_tree_left_neg, ctx.d_tree_right_neg,
                ctx.d_tree_com_neg, ctx.d_tree_size_neg,
                ctx.d_tree_quadA_neg, ctx.d_tree_quadB_neg,
                theta_sq, soft2, d_vel, kick_coeff, physicsModel,
                ctx.use_quadrupole ? 1 : 0);
        }
        if (h_nneg > 0) {
            dim3 wgrid((h_nneg + 255) / 256);
            tree_traverse_and_kick_warp<<<wgrid, wblock>>>(
                ctx.d_sorted_pos_neg, ctx.d_orig_idx_neg, h_nneg,
                h_npos, ctx.d_tree_left_pos, ctx.d_tree_right_pos,
                ctx.d_tree_com_pos, ctx.d_tree_size_pos,
                ctx.d_tree_quadA_pos, ctx.d_tree_quadB_pos,
                h_nneg, ctx.d_tree_left_neg, ctx.d_tree_right_neg,
                ctx.d_tree_com_neg, ctx.d_tree_size_neg,
                ctx.d_tree_quadA_neg, ctx.d_tree_quadB_neg,
                theta_sq, soft2, d_vel, kick_coeff, physicsModel,
                ctx.use_quadrupole ? 1 : 0);
        }
    } else {
        tree_traverse_and_kick<<<grid_N, block>>>(
            d_pos, N,
            h_npos, ctx.d_tree_left_pos, ctx.d_tree_right_pos,
            ctx.d_tree_com_pos, ctx.d_tree_size_pos,
            ctx.d_tree_quadA_pos, ctx.d_tree_quadB_pos,
            h_nneg, ctx.d_tree_left_neg, ctx.d_tree_right_neg,
            ctx.d_tree_com_neg, ctx.d_tree_size_neg,
            ctx.d_tree_quadA_neg, ctx.d_tree_quadB_neg,
            theta_sq, soft2, d_vel, kick_coeff, physicsModel,
            ctx.use_quadrupole ? 1 : 0);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("[Tree] traverse_and_kick error: %s\n", cudaGetErrorString(err));
}

// ============================================================================
//  Pure drift kernel (no force) — used in Galaxy split kick/drift integration
// ============================================================================
__global__ void tree_drift_full_kernel(
    const float4* __restrict__ pos_in,
    float4* __restrict__ pos_out,
    const float4* __restrict__ vel,
    unsigned int N, float dt)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    float4 p = pos_in[gid];
    float4 v = vel[gid];
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;
    pos_out[gid] = p;
}

void tree_drift_full(const float4* d_pos_in, float4* d_pos_out,
                     const float4* d_vel, unsigned int N, float dt)
{
    if (N == 0) return;
    dim3 block(256), grid((N + 255) / 256);
    tree_drift_full_kernel<<<grid, block>>>(d_pos_in, d_pos_out, d_vel, N, dt);
}
