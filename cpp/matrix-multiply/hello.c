void MatrixMultiplication(float *M, float *N, float *P, int Width) {
  for (int i = 0; i < Width; i++) {
    for (int j = 0; j < Width; j++) {
      P[i * Width + j] = 0;
      for (int k = 0; k < Width; k++) {
        P[i * Width + j] += M[i * Width + k] * N[k * Width + j];
      }
    }
  }
}
