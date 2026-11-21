#include "nbody.h"
#include <stdio.h>

int output_device_info(cl_device_id device_id)
{
    char   name[128];
    cl_uint compute_units;

    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(compute_units), &compute_units, NULL);
    printf("Device: %s, Compute units: %u\n", name, compute_units);
    return 0;
}
