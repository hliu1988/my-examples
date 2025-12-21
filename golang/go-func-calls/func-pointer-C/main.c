#include <stdio.h>

__attribute__((noinline, noipa))
int bar() {
  return 1;
}

int (*gf)();

__attribute__((noinline, noipa))
int foo(int (*f)()) {
  int r = (*f)();
  r += (*gf)();
  return r;
}

int main() {
  gf = &bar;
  int (*lf)();
  lf = &bar;
  int r = foo(lf);
  printf("%d", r);
}
