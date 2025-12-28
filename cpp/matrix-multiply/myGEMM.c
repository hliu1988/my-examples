#include <assert.h>
#include <stdlib.h>

// 简单的 Width x Width 方形的矩阵
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

// row-major: 表示行优先

typedef struct {
  int height; // 高度，行数，y
  int width;  // 宽度，列数，x
  float *elements;
} Matrix;

// 辅助函数：初始化矩阵
Matrix create_matrix(int height, int width, const float *data) {
  Matrix m;
  m.height = height;
  m.width = width;
  m.elements = malloc(sizeof(float) * height * width);
  assert(m.elements != NULL);
  if (data != NULL) {
    for (int i = 0; i < height * width; i++) {
      m.elements[i] = data[i];
    }
  } else {
    // 无初始数据则置0
    for (int i = 0; i < height * width; i++) {
      m.elements[i] = 0.0f;
    }
  }
  return m;
}

Matrix gemm(Matrix A, Matrix B) {
  assert(A.width == B.height);
  Matrix C = create_matrix(A.height, B.width, NULL);

  for (int i = 0; i < A.height; i++) {
    for (int j = 0; j < B.width; j++) {
      float C_element = 0;
      for (int k = 0; k < A.width; k++) {
        C_element += A.elements[i * A.width + k]    // A 的第 i 行
                     * B.elements[k * B.width + j]; // B 的第 j 列
      }
      // C.elements[i * A.height + j] = C_element;
      C.elements[i * C.width + j] = C_element;
    }
  }

  return C;
}

// 优化1：调整循环顺序为 i-k-j，提升B矩阵的缓存命中率
Matrix gemm_optim1(Matrix A, Matrix B) {
  assert(A.width == B.height);
  Matrix C = create_matrix(A.height, B.width, NULL);

  // 核心改动：循环顺序从 i-j-k → i-k-j
  for (int i = 0; i < A.height; i++) { // 遍历A的行（C的行）
    // 预计算A的行起始地址，减少重复计算
    float *A_row = &A.elements[i * A.width];
    for (int k = 0; k < A.width; k++) { // k：是遍历B的列/A的行
      float A_val = A_row[k]; // 预取A的当前元素，减少内存访问
      // 预计算B的行起始地址
      float *B_row = &B.elements[k * B.width];
      // 遍历B的列（C的列）
      for (int j = 0; j < B.width; j++) { // j: 是遍历B的行/A的列
        // C[i][j] += A[i][k] * B[k][j]
        C.elements[i * C.width + j] += A_val * B_row[j];
      }
    }
  }

  return C;
}

// 优化2：循环展开（以j循环展开4次为例，适合列数为4的倍数）
Matrix gemm_optim2(Matrix A, Matrix B) {
  assert(A.width == B.height);
  Matrix C = create_matrix(A.height, B.width, NULL);

  int j;
  for (int i = 0; i < A.height; i++) {
    float *A_row = &A.elements[i * A.width];
    float *C_row = &C.elements[i * C.width];
    for (int k = 0; k < A.width; k++) {
      float A_val = A_row[k];
      float *B_row = &B.elements[k * B.width];

      // 循环展开：每次处理4个元素，减少循环次数
      for (j = 0; j <= C.width - 4; j += 4) {
        C_row[j] += A_val * B_row[j];
        C_row[j + 1] += A_val * B_row[j + 1];
        C_row[j + 2] += A_val * B_row[j + 2];
        C_row[j + 3] += A_val * B_row[j + 3];
      }
      // 处理剩余不足4个的元素
      for (; j < C.width; j++) {
        C_row[j] += A_val * B_row[j];
      }
    }
  }

  return C;
}

// 优化3：分块（Tiling）优化，推荐块大小32/64（匹配CPU L1/L2缓存）
#define BLOCK_SIZE 32

Matrix gemm_optim3(Matrix A, Matrix B) {
  assert(A.width == B.height);
  Matrix C = create_matrix(A.height, B.width, NULL);

  // 外层按块遍历
  for (int i_block = 0; i_block < A.height; i_block += BLOCK_SIZE) {
    for (int j_block = 0; j_block < B.width; j_block += BLOCK_SIZE) {
      for (int k_block = 0; k_block < A.width; k_block += BLOCK_SIZE) {
        // 内层处理块内元素（循环顺序仍为i-k-j）
        for (int i = i_block; i < i_block + BLOCK_SIZE && i < A.height; i++) {
          float *A_row = &A.elements[i * A.width];
          float *C_row = &C.elements[i * C.width];
          for (int k = k_block; k < k_block + BLOCK_SIZE && k < A.width; k++) {
            float A_val = A_row[k];
            float *B_row = &B.elements[k * B.width];
            for (int j = j_block; j < j_block + BLOCK_SIZE && j < B.width;
                 j++) {
              C_row[j] += A_val * B_row[j];
            }
          }
        }
      }
    }
  }

  return C;
}

// 可选：测试函数（验证正确性）
#include <stdio.h>
void print_matrix(Matrix m) {
  for (int i = 0; i < m.height; i++) {
    for (int j = 0; j < m.width; j++) {
      printf("%.1f ", m.elements[i * m.width + j]);
    }
    printf("\n");
  }
}

int main() {
  // 测试用例：A(2×2) = [[1,2],[3,4]], B(2×2) = [[5,6],[7,8]]
  Matrix A = {2, 2, malloc(4 * sizeof(float))};
  Matrix B = {2, 2, malloc(4 * sizeof(float))};
  float A_data[] = {1, 2, 3, 4};
  float B_data[] = {5, 6, 7, 8};
  for (int i = 0; i < 4; i++) {
    A.elements[i] = A_data[i];
    B.elements[i] = B_data[i];
  }

  Matrix C = gemm(A, B);
  printf("矩阵C = A × B 的结果：\n");
  print_matrix(C); // 正确结果应为 [[19,22],[43,50]]

  C = gemm_optim1(A, B);
  printf("矩阵C = A × B 的结果：\n");
  print_matrix(C); // 正确结果应为 [[19,22],[43,50]]

  C = gemm_optim2(A, B);
  printf("矩阵C = A × B 的结果：\n");
  print_matrix(C); // 正确结果应为 [[19,22],[43,50]]

  C = gemm_optim3(A, B);
  printf("矩阵C = A × B 的结果：\n");
  print_matrix(C); // 正确结果应为 [[19,22],[43,50]]

  // 释放内存（避免内存泄漏）
  free(A.elements);
  free(B.elements);
  free(C.elements);
  return 0;
}
