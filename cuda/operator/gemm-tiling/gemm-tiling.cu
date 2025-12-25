#include <cuda_runtime.h>
#include <stdio.h>

// 定义Tiling的块大小（需根据GPU架构调整，如32适合Volta/Turing）
#define BLOCK_SIZE 32

// CUDA核函数：带Tiling的矩阵乘法
__global__ void gemmTilingKernel(const float* A, const float* B, float* C, 
                                 int M, int K, int N) {
    // 1. 线程坐标与输出矩阵C的元素坐标映射
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 共享内存：存储A和B的Tile
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    // 2. 按K维度拆分子块，循环加载并计算
    for (int k_tile = 0; k_tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; k_tile++) {
        // 3. 加载A的Tile到共享内存（线程协作）
        int a_k = k_tile * BLOCK_SIZE + threadIdx.x;
        if (row < M && a_k < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + a_k];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f; // 边界补0
        }

        // 4. 加载B的Tile到共享内存（线程协作）
        int b_k = k_tile * BLOCK_SIZE + threadIdx.y;
        if (b_k < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[b_k * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f; // 边界补0
        }

        // 等待共享内存加载完成
        __syncthreads();

        // 5. 共享内存内的子块乘法（减少全局内存访问）
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        // 等待当前子块计算完成，避免下一次加载覆盖数据
        __syncthreads();
    }

    // 6. 将结果写入全局内存的C矩阵
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host端调用接口
void gemmTiling(const float* h_A, const float* h_B, float* h_C, int M, int K, int N) {
    // Device内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // 异步拷贝（Host→Device）
    cudaMemcpyAsync(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 配置线程块和网格（Device端Tiling的块大小由Host端配置）
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 启动Kernel
    gemmTilingKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    // 拷贝结果（Device→Host）
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // 测试矩阵维度（M=1024, K=512, N=1024）
    int M = 1024, K = 512, N = 1024;
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // 初始化矩阵
    for (int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;

    // 调用带Tiling的矩阵乘法
    gemmTiling(h_A, h_B, h_C, M, K, N);

    printf("Tiling优化的GEMM计算完成！\n");

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
