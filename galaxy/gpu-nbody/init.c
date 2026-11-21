#include "nbody.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* libft stubs */
void *ft_memalloc(size_t size)   { return calloc(1, size); }
void  ft_bzero(void *s, size_t n){ memset(s, 0, n); }
void *ft_memcpy(void *d, const void *s, size_t n){ return memcpy(d,s,n); }

char *ft_strdup(const char *s)
{
    size_t len = strlen(s) + 1;
    char  *r   = (char *)malloc(len);
    if (r) memcpy(r, s, len);
    return r;
}

char *ft_strjoin(const char *s1, const char *s2)
{
    size_t l1 = strlen(s1), l2 = strlen(s2);
    char  *r  = (char *)malloc(l1 + l2 + 1);
    if (!r) return NULL;
    memcpy(r, s1, l1);
    memcpy(r + l1, s2, l2 + 1);
    return r;
}

char *ft_itoa(int n)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", n);
    return ft_strdup(buf);
}

char **ft_strsplit(const char *s, char c)
{
    (void)s; (void)c;
    return NULL;
}

int get_next_line(int fd, char **line)
{
    (void)fd; (void)line;
    return -1;
}
