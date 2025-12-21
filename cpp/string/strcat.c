// C,C++ program demonstrate difference between
// strncat() and strcat()
#include <stdio.h>
#include <string.h>

int
main ()
{
  // Take any two strings
  char src[50] = "forgeeks";
  char dest1[50] = "geeks";
  char dest2[50] = "geeks";

  printf ("Before strcat() function execution, ");
  printf ("destination string : %s\n", dest1);

  // Appends the entire string of src to dest1
  strcat (dest1, src);

  // Prints the string
  printf ("After strcat() function execution, ");
  printf ("destination string : %s\n", dest1);

  printf ("Before strncat() function execution, ");
  printf ("destination string : %s\n", dest2);

  // Appends 3 characters from src to dest2
  strncat (dest2, src, 3);

  // Prints the string
  printf ("After strncat() function execution, ");
  printf ("destination string : %s\n", dest2);

  return 0;
}
