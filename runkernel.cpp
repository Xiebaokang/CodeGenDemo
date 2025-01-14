#include <iostream>
#include <fstream>
#include <sstream>
#include <hip/hip_runtime.h>
// #include <rocblas.h>
#include <algorithm>
#include <vector>


template <typename T>
void display(T *host, int len) {
    // 打印
    int mid = len / 2;
    int start = (rand() % (mid - 1)) + 1;
    int end = (rand() % (mid - 1)) + mid + 1;
    std::cout << "{" << host[0] << ", ..., " << host[start] << ", ..., "  << host[mid] << ", ..., "  << host[end] << ", ..., " << host[len - 1] << "}\n";
}

template <typename T>
void hostMatmul(T *A, T *B, T *C, int M, int N, int K) {
    // host矩阵乘
    for (int i=0; i<M; i++) {
      for (int j=0; j<N; j++) {
        for (int k=0; k<K; k++) {
          C[i * N + j] += A[k * M + i] * B[k * N + j];
        }
      }
    }
}

template <typename T>
void verify(T *host, T *device, int M, int N) {
  // 验证
  bool result = true;
  int errorCount = 0;
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      int index = i * N + j;
      // printf("test: %.1f  mine: %.1f\n", host[index], device[index]);
      if (std::abs(host[index] - device[index]) >= 0.1) {
        // printf("error index: (y=%d, x=%d)\n", i, j);
        // printf("errer host: %.1f   error device: %.1f\n", host[index], device[index]);
        result = false;
        errorCount++;
      }
    }
  }
  if (result) {
    printf(" -- no error!\n");
  } else {
    printf(" -- have %d error!\n", errorCount);
  }
}


__global__ void gemm_kernel(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[i*m+row] * B[i*n+col];
        }
        C[row * n + col] = sum;
    }
}


int main() {

    int device_count;
    hipGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "没有找到可用的HIP设备" << std::endl;
        // delete[] hsaco_buffer;
        return 1;
    }

    int device_id;
    hipGetDevice(&device_id);
    hipSetDevice(device_id);

    // 加载HSACO文件作为模块
    hipModule_t module;
    hipModuleLoad(&module, "/tmp/kcg_kernel-abf9b8/kcg_kernel-abf9b8.hsaco");

    // 获取内核函数
    hipFunction_t kernel;
    hipError_t error = hipModuleGetFunction(&kernel, module, "GEMM_testKernel");
    if (error == hipSuccess) {
        std::cout << "Successfully obtained the function handle." << std::endl;
    } else {
        std::cerr << "Error in obtaining the function handle: " << hipGetErrorString(error) << std::endl;
        return 0;
    }

    int M = 1024;
    int N = 1024;
    int K = 1024;

    float *A = new float[M * K];
    float *B = new float[N * K];
    float *C = new float[N * M];
    float *D = new float[N * M];
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
        D[i] = 0.0f;
    }
    for (int i = 0; i < M * K; i++) {
        A[i] = (rand() % 50000) * 0.01f;
        // A[i] = 1.0f;
    } 
    for (int i = 0; i < N * K; i++) {
        B[i] = (rand() % 50000) * 0.01f;
        // B[i] = 1.0f;
    }
    
    float *d_A, *d_B, *d_C, *d_D;
    hipMalloc(&d_A, M * K * sizeof(float));
    hipMalloc(&d_B, K * N * sizeof(float));
    hipMalloc(&d_C, M * N * sizeof(float));
    hipMalloc(&d_D, M * N * sizeof(float));

    hipMemcpy(d_A, A, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, K * N * sizeof(float), hipMemcpyHostToDevice);

    void* args[] = {&d_A, &d_B, &d_C};
    dim3 dimBlock = {16, 16};
    dim3 dimGrid = {(N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y};

    std::vector<float> costs;
    hipEvent_t startEvent, stopEvent;
    hipEventCreate(&startEvent);
    hipEventCreate(&stopEvent);
    hipModuleLaunchKernel(kernel, 256, 1, 1, 256, 1, 1, 50000, 0, args, NULL); 
    gemm_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_D, M, N, K);

    for (int i=0; i<10; i++) {
        // 执行内核函数
        float elapsedTime = 0;
        hipEventRecord(startEvent, 0);
        hipModuleLaunchKernel(kernel, 256, 1, 1, 256, 1, 1, 50000, 0, args, NULL);  
        hipEventRecord(stopEvent, 0);
        hipEventSynchronize(stopEvent);

        hipEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        costs.push_back(elapsedTime);
        gemm_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_D, M, N, K);
    }

    hipDeviceSynchronize();

    std::sort(costs.begin(), costs.end());
    for(auto num : costs) std::cout << num << " ";
    std::cout << std::endl;
    float time = costs[costs.size()/2];
    double tflops = (double)(2 * (M / 10000.0) * (N / 10000.0) * (K / 10000.0)) / (time / 1000);

    hipMemcpy(C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);
    std::cout << "time cost: " << time << "ms\n";
    std::cout << "tflops: " << tflops << std::endl;
    hipMemcpy(D, d_D, M * N * sizeof(float), hipMemcpyDeviceToHost);
    verify(D, C, M, N);

    // 清理内存
    hipModuleUnload(module);
    // delete[] hsaco_buffer;

    return 0;
}