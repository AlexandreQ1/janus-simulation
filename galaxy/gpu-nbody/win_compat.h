#ifndef WIN_COMPAT_H
# define WIN_COMPAT_H

# ifdef _WIN32

#  include <windows.h>
#  include <stdio.h>
#  include <stdlib.h>
#  include <string.h>

/* The O_* and I/O functions are already provided by the Windows SDK */
#  include <fcntl.h>
#  include <io.h>
#  include <sys/stat.h>

#  ifndef S_IRUSR
#    define S_IRUSR _S_IREAD
#    define S_IWUSR _S_IWRITE
#    define S_IROTH 0
#  endif

void  *ft_memalloc(size_t size);
void   ft_bzero(void *s, size_t n);
void  *ft_memcpy(void *dst, const void *src, size_t n);
char  *ft_strdup(const char *s1);
char  *ft_strjoin(const char *s1, const char *s2);
char  *ft_itoa(int n);
char **ft_strsplit(const char *s, char c);
int    get_next_line(int fd, char **line);

double wtime(void);

# else

#  include <unistd.h>
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <sys/types.h>

# endif /* _WIN32 */

#endif
