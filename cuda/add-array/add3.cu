#include <iostream>
#include <math.h>

__global__ void add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  if (index == 0){
    printf("Hello from GPU! blockDim.x = %d\n", blockDim.x);
    printf("Hello from GPU! gridDim.x = %d\n", gridDim.x);
    printf("Hello from GPU! stride = %d\n", stride);
    }
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void) {
  int N = (1 << 20) + 32;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  int blockSize = 256;                             // block dim
  // int numBlocks = (N + blockSize - 1) / blockSize; // grid dim
  int numBlocks = N / (3 * 256); // grid dim
  printf("Hello from CPU! N = %d\n", N);
  printf("Hello from CPU! numBlocks = %d\n", numBlocks);
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  return 0;
}
