/**
 * @file matmul.cpp
 * @author xiebaokang
 * @brief 用于验证新的config下的gemm算法。（带localsplitU）
 * @version 0.1
 * @date 2024-12-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <iostream>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <vector>

template<int VecLen>
__device__ __forceinline__ void VecCpy(float* a, float* b);

template<>
__device__ __forceinline__ void VecCpy<8>(float* a, float* b) {
  (reinterpret_cast<float4*>(a)[0]) = (reinterpret_cast<float4*>(b)[0]);
  (reinterpret_cast<float4*>(a+4)[0]) = (reinterpret_cast<float4*>(b+4)[0]);
}

template<>
__device__ __forceinline__ void VecCpy<6>(float* a, float* b) {
  (reinterpret_cast<float4*>(a)[0]) = (reinterpret_cast<float4*>(b)[0]);
  (reinterpret_cast<float2*>(a+4)[0]) = (reinterpret_cast<float2*>(b+4)[0]);
}

template<>
__device__ __forceinline__ void VecCpy<4>(float* a, float* b) {
  (reinterpret_cast<float4*>(a)[0]) = (reinterpret_cast<float4*>(b)[0]);
}

template<>
__device__ __forceinline__ void VecCpy<2>(float* a, float* b) {
  (reinterpret_cast<float2*>(a)[0]) = (reinterpret_cast<float2*>(b)[0]);
}

template<>
__device__ __forceinline__ void VecCpy<1>(float* a, float* b) {
  (reinterpret_cast<float*>(a)[0]) = (reinterpret_cast<float*>(b)[0]);
} 


/*mine method*/
template <
  const int BM,
  const int BN,
  const int BK,
  const int TM,
  const int TN,
  const int GLOB_LOAD_WIDTH_A,   /*6*/
  const int GLOB_LOAD_WIDTH_B,
  const int BLOCK_LAYOUT_Y,   // BM / TM / WARP_LAYOUT_Y
  const int BLOCK_LAYOUT_X,    // BN / TN / WARP_LAYOUT_X
  const int WARP_LAYOUT_Y,
  const int WARP_LAYOUT_X,
  const int BLOCK_SCATTER_WIDTH_A,
  const int BLOCK_SCATTER_WIDTH_B,
  const int WARP_SCATTER_WIDTH_A,
  const int WARP_SCATTER_WIDTH_B,
  const int LOCAL_SPLIT_U,   /*2*/
  const int BLOCK_MAPPING,
  const int WARP_SIZE,
  const int GLOB_STORE_WIDTH>
__global__ void matmul(float* A, float* B, float* C, const int M, const int N, const int K) {
  const int BLOCK_Y = BM / TM;
  const int BLOCK_X = BN / TN;
  const int THREAD_NUM = BLOCK_X * BLOCK_Y * LOCAL_SPLIT_U;
  const int tid = threadIdx.x;
  const int tz = tid / (BLOCK_X * BLOCK_Y);
  const int tid_other = tid % (BLOCK_X * BLOCK_Y);

  // enhance L2 cache hit rate
  const int bid = blockIdx.x;
  const int GROUP_NUM = BLOCK_MAPPING * (N / BN);
  const int group_id = bid / GROUP_NUM;
  const int start_y = group_id * BLOCK_MAPPING;
  const int block_mapping = (M / BM) - start_y < BLOCK_MAPPING ? (M / BM) - start_y : BLOCK_MAPPING;
  const int by = start_y + (bid % block_mapping);
  const int bx = (bid % GROUP_NUM) / block_mapping;

  // thread mapping
  const int warp_id = tid_other / WARP_SIZE;
  const int lane_id = tid_other % WARP_SIZE;
  const int warp_y = warp_id / BLOCK_LAYOUT_X;
  const int warp_x = warp_id % BLOCK_LAYOUT_X;
  const int lane_y = lane_id / WARP_LAYOUT_X;
  const int lane_x = lane_id % WARP_LAYOUT_X;

  // split number
  const int BLOCK_REPEAT_A = TM / BLOCK_SCATTER_WIDTH_A;  // 4 / 2 = 2 
  const int BLOCK_REPEAT_B = TN / BLOCK_SCATTER_WIDTH_B;  // 6 / 2 = 3
  const int WARP_REPEAT_A = BLOCK_SCATTER_WIDTH_A / WARP_SCATTER_WIDTH_A;  // 2 / 2 = 1
  const int WARP_REPEAT_B = BLOCK_SCATTER_WIDTH_B / WARP_SCATTER_WIDTH_B;  // 2 / 2 = 1

  // LDSMemory size: BM * BN * 2 > BM * BK * 2 + BN * BK * 2 ? BM * BN * 2 : BM * BK * 2 + BN * BK * 2
  const int LDS_SZIE = (BM * BN * 2) > (BM + BN) * BK * 2 ? (BM * BN * 2) : (BM + BN) * BK * 2;
  __shared__ float LDSMemory[LDS_SZIE];
  float regA[2*TM];
  float regB[2*TN];
  float regC[TM*TN] = {0.0f};

  // global to shread args
  const int GLOB_LOAD_TOTAL_WIDTH_A = BM * BK / THREAD_NUM;  // 8
  const int GLOB_LOAD_TOTAL_WIDTH_B = BN * BK / THREAD_NUM;  // 6

  const int GLOB_LOAD_NUM_A = GLOB_LOAD_TOTAL_WIDTH_A / GLOB_LOAD_WIDTH_A;  // 8 / 6 = 1
  const int GLOB_LOAD_NUM_B = GLOB_LOAD_TOTAL_WIDTH_B / GLOB_LOAD_WIDTH_B;  // 6 / 2 = 3

  const int GLOB_LOAD_REAR_WIDTH_A = GLOB_LOAD_TOTAL_WIDTH_A % GLOB_LOAD_WIDTH_A;  // 8 % 6 = 2
  const int GLOB_LOAD_REAR_WIDTH_B = GLOB_LOAD_TOTAL_WIDTH_B % GLOB_LOAD_WIDTH_B;  // 6 % 2 = 0

  const int GLOB_LOAD_ROW_WIDTH_A = (THREAD_NUM / BK) * GLOB_LOAD_WIDTH_A;  // 48
  const int GLOB_LOAD_ROW_WIDTH_B = (THREAD_NUM / BK) * GLOB_LOAD_WIDTH_B;  // 16

  const int sh_load_row = tid / (THREAD_NUM / BK);
  const int sh_load_col = tid % (THREAD_NUM / BK);

  // mid temp reg
  float tempA[GLOB_LOAD_TOTAL_WIDTH_A];  // 8
  float tempB[GLOB_LOAD_TOTAL_WIDTH_B];  // 6

  // shared to registers arg

  A = &A[by * BM];
  B = &B[bx * BN];

  float* shA = (float *)(LDSMemory);
  float* shB = (float *)(LDSMemory + (BM * BK * 2));

  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
    // GLOB_LOAD_WIDTH_A = 4 + 2
    VecCpy<GLOB_LOAD_WIDTH_A>(&shA[sh_load_row * BM + 
                                   i * GLOB_LOAD_ROW_WIDTH_A + 
                                   sh_load_col * GLOB_LOAD_WIDTH_A], 
                                &A[sh_load_row * M + 
                                   i * GLOB_LOAD_ROW_WIDTH_A + 
                                   sh_load_col * GLOB_LOAD_WIDTH_A]);
  }
  // rear width = 2
  if (GLOB_LOAD_REAR_WIDTH_A) {
    VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&shA[sh_load_row * BM + 
                                        GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_A], 
                                     &A[sh_load_row * M + 
                                        GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_A]);
  }
  
  // globB to sharedB
  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
    // GLOB_LOAD_WTDTH_B = 2
    VecCpy<GLOB_LOAD_WIDTH_B>(&shB[sh_load_row * BN + 
                                   i * GLOB_LOAD_ROW_WIDTH_B + 
                                   sh_load_col * GLOB_LOAD_WIDTH_B], 
                                &B[sh_load_row * N + 
                                   i * GLOB_LOAD_ROW_WIDTH_B + 
                                   sh_load_col * GLOB_LOAD_WIDTH_B]);
  }
  if (GLOB_LOAD_REAR_WIDTH_B) {
    VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&shB[sh_load_row * BN + 
                                        GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_B], 
                                     &B[sh_load_row * N + 
                                        GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_B]);
  }
  __syncthreads();

  #pragma unroll
  for (int i=0; i<BLOCK_REPEAT_A; i++) {
    #pragma unroll
    for (int j=0; j<WARP_REPEAT_A; j++) {
      VecCpy<WARP_SCATTER_WIDTH_A>(&regA[i * BLOCK_SCATTER_WIDTH_A + 
                                         j * WARP_SCATTER_WIDTH_A], 
                                    &shA[tz * BM + 
                                         (i * BLOCK_LAYOUT_Y + warp_y) * WARP_LAYOUT_Y * BLOCK_SCATTER_WIDTH_A + 
                                         (j * WARP_LAYOUT_Y + lane_y) * WARP_SCATTER_WIDTH_A]);
    }
  }
  #pragma unroll
  for (int i=0; i<BLOCK_REPEAT_B; i++) {
    #pragma unroll
    for (int j=0; j<WARP_REPEAT_B; j++) {
      VecCpy<WARP_SCATTER_WIDTH_B>(&regB[i * BLOCK_SCATTER_WIDTH_B + 
                                         j * WARP_SCATTER_WIDTH_B], 
                                    &shB[tz * BN + (i * BLOCK_LAYOUT_X + warp_x) * WARP_LAYOUT_X * BLOCK_SCATTER_WIDTH_B + 
                                         (j * WARP_LAYOUT_X + lane_x) * WARP_SCATTER_WIDTH_B]);
    }
  }

  int write_buffer_id = 1;
  for (int k=BK; k<=K; k+=BK) {

    if (k < K) {   // 最后一轮迭代只有数据计算，没有glob -> shared的操作
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        // GLOB_LOAD_WIDTH_A = 4 + 2
        VecCpy<GLOB_LOAD_WIDTH_A>(&tempA[i * GLOB_LOAD_WIDTH_A], 
                                      &A[(sh_load_row + k) * M + 
                                         i * GLOB_LOAD_ROW_WIDTH_A + 
                                         sh_load_col * GLOB_LOAD_WIDTH_A]);
      }
      // rear width = 2
      if (GLOB_LOAD_REAR_WIDTH_A) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&tempA[GLOB_LOAD_NUM_A * GLOB_LOAD_WIDTH_A], 
                                           &A[(sh_load_row + k) * M + 
                                              GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                              sh_load_col * GLOB_LOAD_REAR_WIDTH_A]);
      }
      
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        // GLOB_LOAD_WTDTH_B = 2
        VecCpy<GLOB_LOAD_WIDTH_B>(&tempB[i * GLOB_LOAD_WIDTH_B], 
                                    &B[(sh_load_row + k) * N + 
                                       i * GLOB_LOAD_ROW_WIDTH_B + 
                                       sh_load_col * GLOB_LOAD_WIDTH_B]);
      }
      if (GLOB_LOAD_REAR_WIDTH_B) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&tempB[GLOB_LOAD_NUM_B * GLOB_LOAD_WIDTH_B], 
                                         &B[(sh_load_row + k) * N + 
                                            GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                            sh_load_col * GLOB_LOAD_REAR_WIDTH_B]);
      }
    }

    int read_buffer_id = write_buffer_id ^ 1;

    for (int bk=0; bk<BK/LOCAL_SPLIT_U-1; bk++) {   // 外积
      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_A; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_A; j++) {
          VecCpy<WARP_SCATTER_WIDTH_A>(&regA[((bk + 1) % 2) * TM +
                                            i * BLOCK_SCATTER_WIDTH_A + 
                                            j * WARP_SCATTER_WIDTH_A], 
                                        &shA[read_buffer_id * BM * BK + 
                                            ((bk + 1) * LOCAL_SPLIT_U + tz) * BM + 
                                            (i * BLOCK_LAYOUT_Y + warp_y) * WARP_LAYOUT_Y * BLOCK_SCATTER_WIDTH_A + 
                                            (j * WARP_LAYOUT_Y + lane_y) * WARP_SCATTER_WIDTH_A]);
        }
      }

      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_B; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_B; j++) {
          VecCpy<WARP_SCATTER_WIDTH_B>(&regB[((bk + 1) % 2) * TN + 
                                             i * BLOCK_SCATTER_WIDTH_B + 
                                             j * WARP_SCATTER_WIDTH_B], 
                                        &shB[read_buffer_id * BN * BK + 
                                            ((bk + 1) * LOCAL_SPLIT_U + tz) * BN + 
                                            (i * BLOCK_LAYOUT_X + warp_x) * WARP_LAYOUT_X * BLOCK_SCATTER_WIDTH_B + 
                                            (j * WARP_LAYOUT_X + lane_x) * WARP_SCATTER_WIDTH_B]);
        }
      }

      // computing result
      for (int cy=0; cy<TM; cy++) {
        for (int cx=0; cx<TN; cx++) {
          regC[cy * TN + cx] += regA[(bk % 2) * TM + cy] * regB[(bk % 2) * TN + cx];
        }
      }
    }

    if (k < K) {
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        // GLOB_LOAD_WIDTH_A = 4 + 2
        VecCpy<GLOB_LOAD_WIDTH_A>(&shA[write_buffer_id * BM * BK + 
                                       sh_load_row * BM + 
                                       i * GLOB_LOAD_ROW_WIDTH_A + 
                                       sh_load_col * GLOB_LOAD_WIDTH_A], 
                                &tempA[i * GLOB_LOAD_WIDTH_A]);
      }
      // rear width = 2
      if (GLOB_LOAD_REAR_WIDTH_A) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&shA[write_buffer_id * BM * BK + 
                                            sh_load_row * BM + 
                                            GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                            sh_load_col * GLOB_LOAD_REAR_WIDTH_A], 
                                     &tempA[GLOB_LOAD_NUM_A * GLOB_LOAD_WIDTH_A]);
      }
      
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        // GLOB_LOAD_WTDTH_B = 2
        VecCpy<GLOB_LOAD_WIDTH_B>(&shB[write_buffer_id * BN * BK + 
                                       sh_load_row * BN + 
                                       i * GLOB_LOAD_ROW_WIDTH_B + 
                                       sh_load_col * GLOB_LOAD_WIDTH_B], 
                                &tempB[i * GLOB_LOAD_WIDTH_B]);
      }
      if (GLOB_LOAD_REAR_WIDTH_B) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&shB[write_buffer_id * BN * BK + 
                                            sh_load_row * BN + 
                                            GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                            sh_load_col * GLOB_LOAD_REAR_WIDTH_B], 
                                     &tempB[GLOB_LOAD_NUM_B * GLOB_LOAD_WIDTH_B]);
      }
      __syncthreads();
      write_buffer_id ^= 1;
    }

    // last computing result
    for (int cy=0; cy<TM; cy++) {
      for (int cx=0; cx<TN; cx++) {
        regC[cy * TN + cx] += regA[TM + cy] * regB[TN + cx];
      }
    }

    // reg[0] prefatch 
    #pragma unroll
    for (int i=0; i<BLOCK_REPEAT_A; i++) {
      #pragma unroll
      for (int j=0; j<WARP_REPEAT_A; j++) {
        VecCpy<WARP_SCATTER_WIDTH_A>(&regA[i * BLOCK_SCATTER_WIDTH_A + 
                                           j * WARP_SCATTER_WIDTH_A], 
                                      &shA[(read_buffer_id^1) * BM * BK + 
                                           tz * BM + 
                                           (i * BLOCK_LAYOUT_Y + warp_y) * WARP_LAYOUT_Y * BLOCK_SCATTER_WIDTH_A + 
                                           (j * WARP_LAYOUT_Y + lane_y) * WARP_SCATTER_WIDTH_A]);
      }
    }
    #pragma unroll
    for (int i=0; i<BLOCK_REPEAT_B; i++) {
      #pragma unroll
      for (int j=0; j<WARP_REPEAT_B; j++) {
        VecCpy<WARP_SCATTER_WIDTH_B>(&regB[i * BLOCK_SCATTER_WIDTH_B + 
                                           j * WARP_SCATTER_WIDTH_B], 
                                      &shB[(read_buffer_id^1) * BN * BK + 
                                           tz * BN + 
                                           (i * BLOCK_LAYOUT_X + warp_x) * WARP_LAYOUT_X * BLOCK_SCATTER_WIDTH_B + 
                                           (j * WARP_LAYOUT_X + lane_x) * WARP_SCATTER_WIDTH_B]);
      }
    }
  }

  // regc reduce
  if (LOCAL_SPLIT_U > 1) {
    // reg_c -> shared
    const int LDS_C_STRIDE = BM * BN;
    const int GLOB_STORE_TOTAL_WIDTH = LDS_C_STRIDE / THREAD_NUM;   // 3072 / 256 = 12
    const int REG_C_STRIDE = GLOB_STORE_TOTAL_WIDTH / LOCAL_SPLIT_U;  // 12 /2 = 6
    const int GLOB_STORE_NUM = GLOB_STORE_TOTAL_WIDTH / GLOB_STORE_WIDTH;   // 12 / 6 = 2
    const int GLOB_STORE_REAR_WIDTH = GLOB_STORE_TOTAL_WIDTH % GLOB_STORE_WIDTH;   // 12 % 6 = 0
    const int GLOB_STORE_ROW_WIDTH = (THREAD_NUM / BM) * GLOB_STORE_WIDTH;   // (256 / 64) * 6 = 24

    const int sh_load_row = tid / (THREAD_NUM / BM);   // [0, 64]
    const int sh_load_col = tid % (THREAD_NUM / BM);   // [0, 4]

    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<WARP_SCATTER_WIDTH_A; k++) {
              VecCpy<WARP_SCATTER_WIDTH_B>(&LDSMemory[tz * LDS_C_STRIDE + 
                                                      ((i0 * BLOCK_LAYOUT_Y + warp_y) * WARP_LAYOUT_Y * BLOCK_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_Y + lane_y) * WARP_SCATTER_WIDTH_A + k) * BN + 
                                                      (i1 * BLOCK_LAYOUT_X + warp_x) * WARP_LAYOUT_X * BLOCK_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_X + lane_x) * WARP_SCATTER_WIDTH_B], 
                                                &regC[(i0 * BLOCK_SCATTER_WIDTH_A + j0 * WARP_SCATTER_WIDTH_A + k) * TN +
                                                      i1 * BLOCK_SCATTER_WIDTH_B + j1 * WARP_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
    __syncthreads();

    // shared ->reg
    #pragma unroll
    for (int i=0; i<LOCAL_SPLIT_U; i++) {
      #pragma unroll
      for (int j=0; j<GLOB_STORE_NUM; j++) {
        VecCpy<GLOB_STORE_WIDTH>(&regC[i * GLOB_STORE_TOTAL_WIDTH + 
                                       j * GLOB_STORE_WIDTH], 
                            &LDSMemory[i * LDS_C_STRIDE + 
                                       sh_load_row * BN + 
                                       j * GLOB_STORE_ROW_WIDTH + 
                                       sh_load_col * GLOB_STORE_WIDTH]);
      }
      if (GLOB_STORE_REAR_WIDTH) {
        VecCpy<GLOB_STORE_REAR_WIDTH>(&regC[i * GLOB_STORE_TOTAL_WIDTH + 
                                            GLOB_STORE_NUM * GLOB_STORE_WIDTH], 
                                 &LDSMemory[i * LDS_C_STRIDE + 
                                            sh_load_row * BN +
                                            GLOB_STORE_NUM * GLOB_STORE_ROW_WIDTH + 
                                            sh_load_col * GLOB_STORE_REAR_WIDTH]);
      }
      if (i > 0) {
        #pragma unroll
        for (int k=0; k<GLOB_STORE_TOTAL_WIDTH; k++) {
          regC[k] += regC[i * GLOB_STORE_TOTAL_WIDTH + k];
        }
      }
    }

    // computing result
    // #pragma unroll
    // for (int i=1; i< LOCAL_SPLIT_U; i++) {
    //   #pragma unroll
    //   for (int j=0; j<GLOB_STORE_TOTAL_WIDTH; j++) {
    //     regC[j] += regC[i * GLOB_STORE_TOTAL_WIDTH + j];
    //   }
    // }

    // reg -> global
    #pragma unroll
    for (int i=0; i<GLOB_STORE_NUM; i++) {
      VecCpy<GLOB_STORE_WIDTH>(&C[(by * BM + sh_load_row) * N + 
                                  bx * BN + i * GLOB_STORE_ROW_WIDTH + sh_load_col * GLOB_STORE_WIDTH], 
                            &regC[i * GLOB_STORE_WIDTH]);
    }
    if (GLOB_STORE_REAR_WIDTH) {
      VecCpy<GLOB_STORE_REAR_WIDTH>(&C[(by * BM + sh_load_row) * N + 
                                       bx * BN + GLOB_STORE_NUM * GLOB_STORE_ROW_WIDTH + sh_load_col * GLOB_STORE_REAR_WIDTH], 
                                 &regC[GLOB_STORE_NUM * GLOB_STORE_WIDTH]);
    }

  } else {
    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<WARP_SCATTER_WIDTH_A; k++) {
              VecCpy<WARP_SCATTER_WIDTH_B>(&C[(by * BM + (i0 * BLOCK_LAYOUT_Y + warp_y) * WARP_LAYOUT_Y * BLOCK_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_Y + lane_y) * WARP_SCATTER_WIDTH_A + k) * N + 
                                              bx * BN + (i1 * BLOCK_LAYOUT_X + warp_x) * WARP_LAYOUT_X * BLOCK_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_X + lane_x) * WARP_SCATTER_WIDTH_B], 
                                                &regC[(i0 * BLOCK_SCATTER_WIDTH_A + j0 * WARP_SCATTER_WIDTH_A + k) * TN + 
                                                      i1 * BLOCK_SCATTER_WIDTH_B + j1 * WARP_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
  }
}


template <typename T>
void display(T *host, int len) {
    // 打印
    int mid = len / 2;
    int start = (rand() % (mid - 1)) + 1;
    int end = (rand() % (mid - 1)) + mid + 1;
    std::cout << "{" << host[0] << ", ..., " << host[start] << ", ..., "  << host[mid] << ", ..., "  << host[end] << ", ..., " << host[len - 1] << "}\n";
}

int main() {
  int device_count;
  hipGetDeviceCount(&device_count);
  int device_id;
  hipGetDevice(&device_id);
  hipSetDevice(device_id);

  const int M = 1024;
  const int N = 1056;
  const int K = 1024;

  const int BM = 64;
  const int BN = 48;
  const int BK = 32;
  const int TM = 4;
  const int TN = 6;
  const int GLOB_LOAD_WIDTH_A = 6;   /*6*/
  const int GLOB_LOAD_WIDTH_B = 2;
  const int BLOCK_LAYOUT_Y = 2;   // BM / TM / WARP_LAYOUT_Y
  const int BLOCK_LAYOUT_X = 1;    // BN / TN / WARP_LAYOUT_X
  const int WARP_LAYOUT_Y = 8;
  const int WARP_LAYOUT_X = 8;
  const int BLOCK_SCATTER_WIDTH_A = 2;
  const int BLOCK_SCATTER_WIDTH_B = 2;
  const int WARP_SCATTER_WIDTH_A = 2;
  const int WARP_SCATTER_WIDTH_B = 2;
  const int LOCAL_SPLIT_U = 1;   /*2*/
  const int BLOCK_MAPPING = 8;
  const int WARP_SIZE = 64;
  const int GLOB_STORE_WIDTH = 6;

  float *A = new float[M * K];
  float *B = new float[N * K];
  float *C = new float[N * M];
  for (int i = 0; i < M * K; i++) A[i] = 1.0f;
  for (int i = 0; i < N * K; i++) B[i] = 1.0f;

  float *DA, *DB, *DC;
  hipMalloc(&DA, M * K * sizeof(float));
  hipMalloc(&DB, N * K * sizeof(float));
  hipMalloc(&DC, N * M * sizeof(float));
  hipMemcpy(DA, A, M * K * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(DB, B, N * K * sizeof(float), hipMemcpyHostToDevice);
  
  dim3 grid_size((M/BM)*(N/BN));
  dim3 block_size(((BM/TM)*(BN/TN)) * LOCAL_SPLIT_U);

  std::vector<float> costs;
  for (int i=0; i<10; i++) {
    // 执行内核函数
    hipEvent_t startEvent, stopEvent;
    hipEventCreate(&startEvent);
    hipEventCreate(&stopEvent);
    hipEventRecord(startEvent, 0);

    matmul<BM, BN, BK, TM, TN, 
      GLOB_LOAD_WIDTH_A, GLOB_LOAD_WIDTH_B,
      BLOCK_LAYOUT_Y, BLOCK_LAYOUT_X,
      WARP_LAYOUT_Y, WARP_LAYOUT_X, 
      BLOCK_SCATTER_WIDTH_A, BLOCK_SCATTER_WIDTH_B, 
      WARP_SCATTER_WIDTH_A, WARP_SCATTER_WIDTH_B,
      LOCAL_SPLIT_U, BLOCK_MAPPING, WARP_SIZE, GLOB_STORE_WIDTH><<<grid_size, block_size>>>(DA, DB, DC, M, N, K);

    hipEventRecord(stopEvent, 0);
    hipEventSynchronize(stopEvent);

    float elapsedTime;
    hipEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    costs.push_back(elapsedTime);
  }

  float time = costs[costs.size()/2];
  double tflops = (2 * static_cast<uint64_t>(M) * N * K) / (time / 1000) / 1e12;

  hipMemcpy(C, DC, M * N * sizeof(float), hipMemcpyDeviceToHost);
  std::cout << "time cost: " << time << "ms\n";
  std::cout << "tflops: " << tflops << std::endl;
  display(C, M * N);
  // int num = 0;
  // for (int i=0; i<M*N; i++) {
  //   if (C[i] == 0.0) num++;
  //   // printf("%.1f ", C[i]);
  // }
  // printf("%d\n", num);

  hipFree(DA);
  hipFree(DB);
  hipFree(DC);
  delete[] A;
  delete[] B;
  delete[] C;
  return 0;
}