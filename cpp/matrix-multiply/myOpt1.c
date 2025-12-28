
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
  m.elements = malloc(sizeof(float) * height * width);

  // 无初始数据则置0
  for (int i = 0; i < height * width; i++) {
    m.elements[i] = 0.0f;
  }
  return m;
}

Matrix gemm_before(Matrix A, Matrix B) {
  Matrix C = create_matrix(A.height, B.width);

  for (int i = 0; i < A.height; i++) { // A's row
    float *A_i_row = &A.elements[i * A.width];
    float *C_i_row = &C.elements[i * C.width];
    for (int j = 0; j < B.width; j++) {   // B's col
      for (int k = 0; k < A.width; k++) { // for A's i-th row x B's j-th col
        C_i_row[j] += A_i_row[k] * B.elements[k * B.width + j];
      }
    }
  }
  return C;
}

Matrix gemm_opt1(Matrix A, Matrix B) {
  Matrix C = create_matrix(A.height, B.width);

  for (int i = 0; i < A.height; i++) { // A's rows
    float *A_i_row = &A.elements[i * A.width];
    float *C_i_row = &C.elements[i * C.width];
    for (int k = 0; k < B.height; k++) { // B's rows
      float A_i_k = A_i_row[k];
      float *B_k_row = &B.elements[k * B.width];
      for (int j = 0; j < B.width /* A.width, C.height, K */;
           j++) { // A's i-th row, B's k-th row
        C_i_row[j] += A_i_k * B_k_row[j];
      }
    }
  }

  return C;
}
