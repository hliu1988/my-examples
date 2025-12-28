
#include <assert.h>
#include <stdlib.h>

typedef struct {
  int height; // 高度，行数，y
  int width;  // 宽度，列数，x
  float *elements;
} Matrix;

Matrix create_matrix(int height, int width) {
  Matrix m;
  m.height = height;
  m.width = width;
  m.elements = calloc(sizeof(float) * height * width);
  assert(m.elements);
  return m;
}

Matrix gemm_before(Matrix A, Matrix B) {
  assert(A.width == B.height);
  Matrix C = create_matrix(A.height, B.width);

  for (int i = 0; i < A.height; i++) { // A's row
    float *A_i_row = &A.elements[i * A.width];
    for (int j = 0; j < B.width; j++) { // B's cols
      for (int k = 0; k < /* C.width */ A.width; k++) {
        // C[i][j] += A[i][k] * B[k][j]
        C.elements[i * C.width + j] += A_i_row[k] * B.elements[k * B.width + j];
      }
    }
  }
}
