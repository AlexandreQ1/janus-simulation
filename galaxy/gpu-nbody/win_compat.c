#ifdef _WIN32
#include "win_compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Les fonctions open/read/write/close sont déjà dans le SDK Windows via <io.h> */
/* Ce fichier ne fournit que les stubs libft et fsync */

int fsync(int fd)
{
    (void)fd;
    return 0;
}
#endif
