int main(){
    const int M = 1024, N = 1024, K = 1024;
    const int BM = 64, BN = 64, BK = 16;
    const int TM = 4, TN = 4;
    int tx,ty;
    int bx,by,bz;
    int blockDimX, blockDimY, gridDimX, gridDimY;
    float **A,**B,**C;     
    const int SPLIT_U = 4;
    const int WARP_SIZE = 64;
    const int WARP_LAYOUT_X = 16;
    const int WARP_LAYOUT_Y = 4;
    const int BLOCK_LAYOUT_X = 1;
    const int BLOCK_LAYOUT_Y = 4;
    
    // ========= 朴素表达 ==============
    // for(int i=0;i<M;++i){
    //     for(int j=0;j<N;++j){
    //         for(int k=0;k<K;++k){
    //             C[i][j] += A[i][k] * B[k][j];
    //         }
    //     }
    // }

    // ======= split & reorder ==========
    // for(int i=0;i<M;i += BM){
    //     for(int j=0;j < N;j += BN){
    //         for(int k=0;k<K; k+= BK){
    //             for(int ii = 0;ii < BM; ii += TM){
    //                 for(int jj=0;jj < BN; jj+= TN){
    //                     // thread work
    //                     for(int kk = 0;kk < BK; kk += 1){
    //                         for(int iii = 0; iii < TM; iii += 1){
    //                             for(int jjj = 0;jjj < TN; jjj += 1){
    //                                 int indexI = i + ii + iii;
    //                                 int indexJ = j + jj + jjj;
    //                                 int indexK = k + kk;
    //                                 C[indexI][indexJ] += A[indexI][indexK] * B[indexK][indexJ];
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    
    // // ======= add splitU & reorder ==========
    // for(int i=0;i<M;i += BM){
    //     for(int j=0;j < N;j += BN){
    //         for(int kkk = 0; kkk < SPLIT_U ; kkk += 1){
    //             // block work
    //             for(int k=0;k<K; k+= BK){
    //                 for(int ii = 0;ii < BM; ii += TM){
    //                     for(int jj=0;jj < BN; jj+= TN){
    //                         // thread work
    //                         for(int kk = 0;kk < BK; kk += SPLIT_U){
    //                             for(int iii = 0; iii < TM; iii += 1){
    //                                 for(int jjj = 0;jjj < TN; jjj += 1){
    //                                     int indexI = i + ii + iii;
    //                                     int indexJ = j + jj + jjj;
    //                                     int indexK = k + kk + kkk;
    //                                     C[indexI][indexJ] += A[indexI][indexK] * B[indexK][indexJ];
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    
    // // ======= loop merge dims ==========
    // for(int i=0,j=0,kkk = 0; i<M && j<N && kkk < SPLIT_U ; i += BM, j += BN, kkk += 1){
    //     // block work
    //     for(int k=0;k<K; k+= BK){
    //         for(int ii = 0, jj=0 ;ii < BM && jj < BN ; ii += TM, jj+= TN){
    //             // thread work
    //             for(int kk = 0;kk < BK; kk += SPLIT_U){
    //                 for(int iii = 0; iii < TM; iii += 1){
    //                     for(int jjj = 0;jjj < TN; jjj += 1){
    //                         int indexI = i + ii + iii;
    //                         int indexJ = j + jj + jjj;
    //                         int indexK = k + kk + kkk;
    //                         C[indexI][indexJ] += A[indexI][indexK] * B[indexK][indexJ];
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    // // ======= 并行化 & loop invariant code motion ==========
    // int i = by * BM , j = bx * BN, kkk = bz * 1;  // [bx,by,bz] < [N/BN, M/BM, SPLIT_U] 
    // int ii = ty * TM, jj = tx * TN;  // [tx,ty] < [BN/TN, BM/TM]
    // // block work
    // for(int k=0;k<K; k+= BK){       
    //     // thread work
    //     for(int kk = 0;kk < BK; kk += SPLIT_U){
    //         for(int iii = 0; iii < TM; iii += 1){
    //             for(int jjj = 0;jjj < TN; jjj += 1){
    //                 int indexI = i + ii + iii;
    //                 int indexJ = j + jj + jjj;
    //                 int indexK = k + kk + kkk;
    //                 C[indexI][indexJ] += A[indexI][indexK] * B[indexK][indexJ];
    //             }
    //         }
    //     }
    // }

    // ======= warp 处理的区域离散化 & thread 区域离散化 ==========
#if 1
    const int WSSY = 2;  // 某warp划分单元的最小尺寸, warpscattersizey
    const int WSSX = 2;
    const int THREAD_SCATTER_SIZE_Y = 1;  // thread处理的连续的最小单元尺寸 TSSY
    const int THREAD_SCATTER_SIZE_X = 1;  // TSSX
    
    int blockRepeatY = TM / WSSY;
    int blockRepeatX = TN / WSSX;
    int warpRepeatY = TM / THREAD_SCATTER_SIZE_Y;
    int warpRepeatX = TN / THREAD_SCATTER_SIZE_X;
    int threadRepeatX = WSSY / THREAD_SCATTER_SIZE_Y;
    int threadRepeatY = WSSY / THREAD_SCATTER_SIZE_Y;
    
    int i = by * BM , j = bx * BN, kkk = bz * 1;  // [bx,by,bz] < [N/BN, M/BM, SPLIT_U] 
    int ii = ty * TM, jj = tx * TN;  // [tx,ty] < [BN/TN, BM/TM]
    int tid = ty * blockDimX + tx;

    int laneId = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int warpIdx = warpId % BLOCK_LAYOUT_X;
    int warpIdy = warpId / BLOCK_LAYOUT_X;
    int laneIdx = laneId % WARP_LAYOUT_X;
    int laneIdy = laneId / WARP_LAYOUT_X;

    // block work
    for(int k=0;k<K; k+= BK){       
        // thread work
        for(int kk = 0;kk < BK; kk += SPLIT_U){
            // thread 处理连续的 TM*TN 大小 -> 处理离散的几个小连续区域. 映射关系由warpId laneId wss tss导出
            // for(int iii = 0; iii < TM; iii += 1){
            //     for(int jjj = 0;jjj < TN; jjj += 1){
            //         int indexI = i + ii + iii;
            //         int indexJ = j + jj + jjj;
            //         int indexK = k + kk + kkk;
            //         C[indexI][indexJ] += A[indexI][indexK] * B[indexK][indexJ];
            //     }
            // }
            // warp 离散化 (重映射 i+ii, j+jj -> warpIndexX, warpIndexY)
            for(int wi = 0; wi < warpRepeatX; ++wi){
                for(int wj = 0; wj < warpRepeatY;++wj){
                    int warpIndexX = BN/warpRepeatX * wi + warpIdx*BN/(warpRepeatX * BLOCK_LAYOUT_X);  // offs + base
                    int warpIndexY = BM/warpRepeatY * wj + warpIdy*BM/(BLOCK_LAYOUT_Y * warpRepeatY);
                    // thread 离散化 （重映射 iii,jjj->x_offs,y_offs）
                    for(int p = 0;p < threadRepeatX;++p){
                        for(int q=0;q < threadRepeatY;++q){
                            int x_offs = BN / (warpRepeatX * threadRepeatX) * p + laneIdx * THREAD_SCATTER_SIZE_X; // offs + base
                            int y_offs = BM / (warpRepeatY * threadRepeatY) * q + laneIdy * THREAD_SCATTER_SIZE_Y;
                            // 重映射后的组装 (i+ii+iii, j+jj+jjj -> xx,yy)
                            int xx = warpIndexX + x_offs;
                            int yy = warpIndexY + y_offs;
                            int indexK = k + kk + kkk;
                            // 连续小区域( tssx * tssy 大小)
                            for(int m = 0;m<THREAD_SCATTER_SIZE_X;++m){
                                for(int n = 0;n<THREAD_SCATTER_SIZE_Y;++n){
                                    int _y = yy + n;
                                    int _x = xx + m;
                                    C[_y][_x] += A[_y][indexK] * B[indexK][_x];
                                }
                            }
                        }
                    }
                }
            }
            /**
             * @brief 
             * 至此，for TM*TN 被拆分为 for( warpRepeatX*warpRepeatY) * (threadRepeatX*threadRepeatY) * (TSSX*TSSY)
             * = (warpRepeatX*threadRepeatX*TSSX) * (warpRepeatY*threadRepeatY*TSSY)
             * == (TN) * (TM)
             * 所以 : C[_y][_x] += A[_y][indexK] * B[indexK][_x];
             * 
             * _y = f( iv_warpRepeatY, iv_threadRepeatY, iv_TSSY)
             * _x = g(iv_warpRepeatX, iv_threadRepeatX, iv_TSSX)
             * indexK = k + kk + kkk == h(iv_K, iv_BK, iv_SPLIT_U)
             * 
             * 可得到 ABC affinemap的所需维度：
             * affinemapC = map(_y,_x) = 
             *          map(f(iv_warpRepeatY, iv_threadRepeatY, iv_TSSY), g(iv_warpRepeatX, iv_threadRepeatX, iv_TSSX))
             *          dim = [iv_warpRepeatY, iv_threadRepeatY, iv_TSSY, iv_warpRepeatX, iv_threadRepeatX, iv_TSSX]
             *              + [ty,tx]
             * affinemapA = map(_y,indexK) = 
             *          map(f(iv_warpRepeatY, iv_threadRepeatY, iv_TSSY), h(iv_K, iv_BK, iv_SPLIT_U))
             *          dim = [iv_warpRepeatY, iv_threadRepeatY, iv_TSSY, iv_K, iv_BK, iv_SPLIT_U] + [ty,tx]
             * 
             * affinemapB = map(indexK,_x) = 
             *          map(h(iv_K, iv_BK, iv_SPLIT_U), g(iv_warpRepeatX, iv_threadRepeatX, iv_TSSX))
             *          dim = [iv_K, iv_BK, iv_SPLIT_U,iv_warpRepeatX, iv_threadRepeatX, iv_TSSX] + [ty,tx]
             */
        }
    }
#endif
    
    
    
    
    return 0;
}