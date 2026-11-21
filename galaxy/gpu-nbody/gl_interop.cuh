#pragma once
// CUDA↔OpenGL interop : build the 6-float render vertex buffer directly into a mapped
// GL VBO from the simulation's device buffers (no CPU readback/rebuild/upload).

// Register a GL VBO with CUDA. outRes receives an opaque resource handle. Returns false
// on failure (caller falls back to the CPU upload path).
bool interopRegister(void** outRes, unsigned int vbo);

// Unregister a previously registered VBO.
void interopUnregister(void* res);

// Map the VBO, launch the vertex-build kernel from the device buffers, unmap.
//   d_pos   : device float4 positions   (required ; p.w = mass)
//   d_vel   : device float4 velocities  (or nullptr ; used iff colorByVel)
//   d_smooth: device float SPH lengths  (or nullptr ; used iff softHalo)
//   n       : particle count
// Returns false if the VBO is too small / not mapped -> caller uses the CPU path.
bool interopFillVertices(void* res, const void* d_pos, const void* d_vel,
                         const void* d_smooth, int n, int colorByVel, int softHalo);

// Device float-buffer helpers (for the per-particle SPH smoothing length).
void* interopAllocFloats(int n);
void  interopFree(void* p);
void  interopUploadFloats(void* d, const float* host, int n);
