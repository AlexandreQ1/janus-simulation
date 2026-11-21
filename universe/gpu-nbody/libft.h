#ifndef LIBFT_H
# define LIBFT_H
# include <stdlib.h>
# include <string.h>

void  *ft_memalloc(size_t size);
void   ft_bzero(void *s, size_t n);
void  *ft_memcpy(void *dst, const void *src, size_t n);
char  *ft_strdup(const char *s1);
char  *ft_strjoin(const char *s1, const char *s2);
char  *ft_itoa(int n);
char **ft_strsplit(const char *s, char c);
int    get_next_line(int fd, char **line);

#endif
