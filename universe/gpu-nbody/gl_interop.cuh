#pragma once
// CUDA↔OpenGL interop : build the 6-float render vertex buffer directly into a mapped
// GL VBO from the simulation's device buffers (no CPU readback/rebuild/upload).
// Vertex layout = [x, y, z, value, origIdx, size] ; size is signed by species.

// Register a GL VBO with CUDA. outRes receives an opaque resource handle. Returns false
// on failure (caller falls back to the CPU upload path).
bool interopRegister(void** outRes, unsigned int vbo);

// Unregister a previously registered VBO.
void interopUnregister(void* res);

// Map the VBO, launch the vertex-build kernel, unmap.
//   d_pos   : device float4 positions (required ; p.w = mass/species)
//   d_vel   : device float4 velocities (or nullptr)
//   d_smooth: device float SPH smoothing lengths (or nullptr)
//   d_aux   : device float auxiliary colour values (or nullptr)
//   mode    : 0 = mass, 1 = |velocity|, 2 = aux value
// Returns false if the VBO is too small / not mapped -> caller uses the CPU path.
bool interopFillVertices(void* res, const void* d_pos, const void* d_vel,
                         const void* d_smooth, const void* d_aux,
                         int n, int mode, int softHalo);

// Device float-buffer helpers (SPH smoothing lengths, auxiliary colour values).
void* interopAllocFloats(int n);
void  interopFree(void* p);
void  interopUploadFloats(void* d, const float* host, int n);
