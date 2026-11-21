#include "nbody.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *load_cl_file(char *filename)
{
    FILE   *fp;
    char   *source;
    long    size;

    fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error: cannot open kernel file: %s\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    rewind(fp);

    source = (char *)malloc(size + 1);
    if (!source)
    {
        fclose(fp);
        return NULL;
    }
    if (fread(source, 1, (size_t)size, fp) != (size_t)size)
    {
        free(source);
        fclose(fp);
        return NULL;
    }
    source[size] = '\0';
    fclose(fp);
    return source;
}

void frame_to_file(cl_float4 *frame, unsigned long n,
                   unsigned long framenumber, char *title)
{
    (void)frame; (void)n; (void)framenumber; (void)title;
}

cl_float4 *file_to_frame(char *filename, unsigned long *count)
{
    (void)filename; (void)count;
    return NULL;
}
