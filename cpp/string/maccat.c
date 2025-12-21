
// C,C++ program demonstrate difference between
// strncat() and strcat()
#include <stdio.h>
#include <string.h>

#define PPCAT_NX(A, B, C, D, E) A##B##C##D##E
int
main ()
{
  // Take any two strings
  char src[50] = PPCAT_NX (a, 2, b, 3, c);
  printf ("src : %s\n", src);

  return 0;
}
