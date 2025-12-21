// C,C++ program demonstrate difference between
// strncat() and strcat()
#include <cstring>
#include <iostream>
using namespace std;

int
main ()
{
  // Take any two strings
  char src[50] = "forgeeks";
  char dest1[50] = "geeks";
  char dest2[50] = "geeks";

  cout << "Before strcat() function execution, ";
  cout << "destination string : " << dest1 << endl;

  // Appends the entire string of src to dest1
  strcat (dest1, src);

  // Prints the string
  cout << "After strcat() function execution, ";
  cout << "destination string : " << dest1 << endl;

  cout << "Before strncat() function execution, ";
  cout << "destination string : " << dest2 << endl;

  // Appends 3 characters from src to dest2
  strncat (dest2, src, 3);

  // Prints the string
  cout << "After strncat() function execution, ";
  cout << "destination string : " << dest2 << endl;

  return 0;
}

// this code is contributed by shivanisinghss2110
