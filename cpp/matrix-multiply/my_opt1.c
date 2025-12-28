
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
  m.elements = calloc(height * width, sizeof(float));
  assert(m.elements);
  return m;
}

Matrix gemm_before(Matrix A, Matrix B) {
  assert(A.width == B.height);
  Matrix C = create_matrix(A.height, B.width);

  // 循环交换（i->j->k→i->k->j）不影响数学结果正确性，且优化了 B
  // 矩阵的缓存访问，整体性能更优；
  for (int i = 0; i < A.height; i++) { // A's row
    float *A_i_row = &A.elements[i * A.width];
    float *C_i_row = &C.elements[i * C.width];
    for (int k = 0; k < /* C.width */ A.width; k++) {
      for (int j = 0; j < B.width; j++) { // B's cols
        // C[i][j] += A[i][k] * B[k][j]
        C_i_row[j] += A_i_row[k] * B.elements[k * B.width + j];
      }
    }
  }

  return C;
}
