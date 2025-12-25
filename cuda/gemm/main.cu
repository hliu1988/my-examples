#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA错误检查宏
#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * 无Tiling的CUDA矩阵乘法核函数
 * @param A: 输入矩阵A（M×K），全局内存
 * @param B: 输入矩阵B（K×N），全局内存
 * @param C: 输出矩阵C（M×N），全局内存
 * @param M: 矩阵A的行数 / 矩阵C的行数
 * @param K: 矩阵A的列数 / 矩阵B的行数
 * @param N: 矩阵B的列数 / 矩阵C的列数
 */
__global__ void matmulNoTilingKernel(const float* A, const float* B, float* C, 
                                     int M, int K, int N) {
    // 计算当前线程对应的C矩阵的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // C的行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // C的列索引

    // 边界检查：如果线程超出C的范围，直接返回
    if (row >= M || col >= N) return;

    // 乘累加计算C[row][col]
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // 直接从全局内存读取A和B的元素
        sum += A[row * K + k] * B[k * N + col];
    }

    // 将结果写入C矩阵的对应位置
    C[row * N + col] = sum;
}

// 主机端辅助函数：初始化矩阵（随机值）
void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;  // 随机值[0,1]
    }
}

// 主机端辅助函数：验证结果（与CPU计算对比）
void verifyResult(const float* A, const float* B, const float* C, int M, int K, int N) {
    float* cpuC = new float[M * N]();
    // CPU计算矩阵乘法
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            cpuC[row * N + col] = sum;
        }
    }
    // 对比CPU与GPU结果
    float maxErr = 0.0f;
    for (int i = 0; i < M * N; i++) {
        maxErr = fmaxf(maxErr, fabs(C[i] - cpuC[i]));
    }
    printf("最大误差: %.6f\n", maxErr);
    delete[] cpuC;
}

int main() {
    // 矩阵维度（可调整，注意：无Tiling版本在K较大时性能极差）
    const int M = 512;
    const int K = 256;
    const int N = 512;

    // 1. 主机内存分配与初始化
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N]();
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    // 2. 设备内存分配与数据拷贝（Host→Device）
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 3. 配置CUDA核函数参数（线程块大小通常选16x16或32x32）
    dim3 blockDim(16, 16);  // 每个线程块包含16x16=256个线程
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,  // 列方向的线程块数
                 (M + blockDim.y - 1) / blockDim.y);  // 行方向的线程块数

    // 4. 启动核函数
    matmulNoTilingKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CHECK(cudaGetLastError());  // 检查核函数启动错误
    CHECK(cudaDeviceSynchronize());  // 等待核函数执行完成

    // 5. 数据拷贝（Device→Host）并验证结果
    CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(h_A, h_B, h_C, M, K, N);

    // 6. 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    printf("无Tiling的矩阵乘法执行完成！\n");
    return 0;
}
