#include <stdio.h>

__attribute__((noinline, noipa)) int foo(int n) {
  int sum = 0;
  for (int i = 0; i < n; i++) {
    if (i % 5 != 0)
      sum += i;
    else
      sum -= i;
  }
  return sum;
}

int main() {
  printf("result: %d", foo(1024));
  return 0;
}
