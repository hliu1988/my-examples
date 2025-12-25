#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

// 高斯分量的数量（可根据需求调整）
#define K 3
// 图像尺寸（示例）
#define WIDTH 512
#define HEIGHT 512
// 像素值范围（0-255）
#define PIXEL_MAX 255.0f

// CUDA错误检查宏
#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/**
 * @brief CUDA核函数：计算每个像素的高斯混合概率
 * @param d_image 设备端图像数据（浮点型，0-1归一化）
 * @param d_gm_prob 设备端输出的GM概率
 * @param d_weights 设备端高斯分量权重（K个）
 * @param d_means 设备端高斯分量均值（K个，0-1归一化）
 * @param d_vars 设备端高斯分量方差（K个）
 * @param width 图像宽度
 * @param height 图像高度
 */
__global__ void gmOperatorKernel(const float *d_image, float *d_gm_prob,
                                 const float *d_weights, const float *d_means,
                                 const float *d_vars, int width, int height) {
  // 计算线程对应的像素坐标
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // 边界检查
  if (x >= width || y >= height)
    return;
  int idx = y * width + x;

  float pixel = d_image[idx];
  float gm_prob = 0.0f;

  // 遍历所有高斯分量，计算加权概率和
  for (int k = 0; k < K; k++) {
    float w = d_weights[k];
    float mu = d_means[k];
    float var = d_vars[k];
    float sigma = sqrtf(var);

    // 一元高斯概率密度计算
    float diff = pixel - mu;
    float exp_term = expf(-(diff * diff) / (2.0f * var));
    float gauss_prob = (1.0f / (sqrtf(2.0f * M_PI) * sigma)) * exp_term;

    // 加权求和
    gm_prob += w * gauss_prob;
  }

  d_gm_prob[idx] = gm_prob;
}

/**
 * @brief 计算每个像素的高斯混合后验概率（责任度）
 * @param d_image 设备端图像数据
 * @param d_posterior 设备端输出的后验概率（K*width*height，按分量存储）
 * @param d_weights 高斯分量权重
 * @param d_means 高斯分量均值
 * @param d_vars 高斯分量方差
 * @param width 图像宽度
 * @param height 图像高度
 */
__global__ void gmPosteriorKernel(const float *d_image, float *d_posterior,
                                  const float *d_weights, const float *d_means,
                                  const float *d_vars, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;
  int idx = y * width + x;

  float pixel = d_image[idx];
  float total_prob = 0.0f;
  float component_probs[K];

  // 先计算每个分量的未归一化概率
  for (int k = 0; k < K; k++) {
    float w = d_weights[k];
    float mu = d_means[k];
    float var = d_vars[k];
    float sigma = sqrtf(var);

    float diff = pixel - mu;
    float exp_term = expf(-(diff * diff) / (2.0f * var));
    float gauss_prob = (1.0f / (sqrtf(2.0f * M_PI) * sigma)) * exp_term;

    component_probs[k] = w * gauss_prob;
    total_prob += component_probs[k];
  }

  // 计算后验概率（责任度）
  for (int k = 0; k < K; k++) {
    d_posterior[k * width * height + idx] = component_probs[k] / total_prob;
  }
}

/**
 * @brief 主机端GM算子接口
 * @param h_image 主机端图像数据（unsigned char，0-255）
 * @param h_gm_prob 主机端输出GM概率
 * @param h_posterior 主机端输出后验概率（可选）
 * @param h_weights 高斯分量权重
 * @param h_means 高斯分量均值（0-255）
 * @param h_vars 高斯分量方差
 */
void gmOperator(const unsigned char *h_image, float *h_gm_prob,
                float *h_posterior, const float *h_weights,
                const float *h_means, const float *h_vars) {
  // 1. 设备端内存分配
  float *d_image, *d_gm_prob, *d_posterior = nullptr;
  float *d_weights, *d_means, *d_vars;
  int image_size = WIDTH * HEIGHT * sizeof(float);
  int gm_size = K * WIDTH * HEIGHT * sizeof(float);
  int component_size = K * sizeof(float);

  CHECK(cudaMalloc(&d_image, image_size));
  CHECK(cudaMalloc(&d_gm_prob, image_size));
  if (h_posterior)
    CHECK(cudaMalloc(&d_posterior, gm_size));
  CHECK(cudaMalloc(&d_weights, component_size));
  CHECK(cudaMalloc(&d_means, component_size));
  CHECK(cudaMalloc(&d_vars, component_size));

  // 2. 数据预处理与拷贝（将像素值和均值归一化到0-1）
  float *h_image_float = (float *)malloc(image_size);
  float *h_means_float = (float *)malloc(component_size);
  for (int i = 0; i < WIDTH * HEIGHT; i++) {
    h_image_float[i] = h_image[i] / PIXEL_MAX;
  }
  for (int k = 0; k < K; k++) {
    h_means_float[k] = h_means[k] / PIXEL_MAX;
  }

  CHECK(cudaMemcpy(d_image, h_image_float, image_size, cudaMemcpyHostToDevice));
  CHECK(
      cudaMemcpy(d_weights, h_weights, component_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_means, h_means_float, component_size,
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_vars, h_vars, component_size, cudaMemcpyHostToDevice));

  // 3. 配置CUDA核函数参数
  dim3 blockDim(16, 16); // 16x16线程块（共256线程）
  dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x,
               (HEIGHT + blockDim.y - 1) / blockDim.y);

  // 4. 启动GM概率计算核函数
  gmOperatorKernel<<<gridDim, blockDim>>>(d_image, d_gm_prob, d_weights,
                                          d_means, d_vars, WIDTH, HEIGHT);
  CHECK(cudaGetLastError()); // 检查核函数启动错误

  // 5. （可选）启动后验概率计算核函数
  if (h_posterior) {
    gmPosteriorKernel<<<gridDim, blockDim>>>(d_image, d_posterior, d_weights,
                                             d_means, d_vars, WIDTH, HEIGHT);
    CHECK(cudaGetLastError());
    CHECK(
        cudaMemcpy(h_posterior, d_posterior, gm_size, cudaMemcpyDeviceToHost));
  }

  // 6. 将结果拷贝回主机
  CHECK(cudaMemcpy(h_gm_prob, d_gm_prob, image_size, cudaMemcpyDeviceToHost));

  // 7. 释放内存
  free(h_image_float);
  free(h_means_float);
  CHECK(cudaFree(d_image));
  CHECK(cudaFree(d_gm_prob));
  if (d_posterior)
    CHECK(cudaFree(d_posterior));
  CHECK(cudaFree(d_weights));
  CHECK(cudaFree(d_means));
  CHECK(cudaFree(d_vars));
}

// 测试主函数
int main() {
  // 1. 生成测试图像（随机像素值）
  unsigned char *h_image =
      (unsigned char *)malloc(WIDTH * HEIGHT * sizeof(unsigned char));
  for (int i = 0; i < WIDTH * HEIGHT; i++) {
    h_image[i] = rand() % 256; // 0-255随机像素
  }

  // 2. 定义高斯混合模型参数（3个分量）
  float h_weights[K] = {0.3f, 0.5f, 0.2f};    // 权重和为1
  float h_means[K] = {64.0f, 128.0f, 192.0f}; // 均值（0-255）
  float h_vars[K] = {256.0f, 512.0f, 128.0f}; // 方差

  // 3. 分配输出内存
  float *h_gm_prob = (float *)malloc(WIDTH * HEIGHT * sizeof(float));
  float *h_posterior = (float *)malloc(K * WIDTH * HEIGHT * sizeof(float));

  // 4. 执行GM算子
  gmOperator(h_image, h_gm_prob, h_posterior, h_weights, h_means, h_vars);

  // 5. 输出结果示例（前10个像素的GM概率）
  printf("前10个像素的GM概率：\n");
  for (int i = 0; i < 10; i++) {
    printf("像素[%d]: %.6f\n", i, h_gm_prob[i]);
  }

  // 6. 释放主机内存
  free(h_image);
  free(h_gm_prob);
  free(h_posterior);

  printf("GM算子计算完成！\n");
  return 0;
}
