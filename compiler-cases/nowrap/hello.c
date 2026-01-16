int A[1024 * 2];

int foo(unsigned offset, unsigned N) {
  int sum = 0;

  for (unsigned i = 0; i < N; i++)
    sum += A[i + offset];

  return sum;
}
