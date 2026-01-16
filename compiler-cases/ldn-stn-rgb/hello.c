
void foo(char *RGB, int a) {
  for (int i = 0; i < 1024 * 3; i += 3) {
    RGB[i] += a;
    RGB[i + 1] -= a;
    RGB[i + 2] *= a;
  }
}
