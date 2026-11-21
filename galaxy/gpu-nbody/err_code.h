#pragma once
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

static inline void checkError(int err, const char* msg)
{
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "OpenCL error %d: %s\n", err, msg);
        exit(1);
    }
}
