// volume_grid.cuh — SPLATTING (no more voxel grid). We just compute the
// |v| velocity of each particle (on GPU) to color the splats, and a
// reference (mean) velocity to normalize. Memory = O(N), not O(grid^3).
#pragma once

bool  volumeInit(int N);                 // allocate the buffers for N particles
void  volumeFree();
void  volumeComputeSpeeds(const void* d_vel, int N);  // |v| per particle + mean
const float* volumeSpeedData();          // N floats (velocity per particle) in host RAM
float volumeVelRef();                    // mean velocity (color normalization)
