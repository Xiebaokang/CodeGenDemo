#include <vector>
int vectorize_copy(float* from, float* to, int width){
    // 向量化拷贝
    return 0;
}
int nextValidWidth(int width){
    // 从列表里寻找下一个可用的validwidth 用于搬运 global->shm
    return 0;
}

void sync(){  // hip 的线程同步函数
    ;
}

int main(){
    const int M = 1024, N = 1024, K = 1024;
    const int BM = 64, BN = 64, BK = 16;
    const int TM = 4, TN = 4;
    int tx,ty,tz;
    int bx,by;
    int blockDimX, blockDimY, blockDimZ, gridDimX, gridDimY;
    float **A,**B,**C;     
    const int SPLIT_U = 4; // = blockDimZ
    const int WARP_SIZE = 64;
    const int WARP_LAYOUT_X = 16;
    const int WARP_LAYOUT_Y = 4;
    const int BLOCK_LAYOUT_X = 1;
    const int BLOCK_LAYOUT_Y = 4;
    int THREAD_COUNT_Y = BM / TM;
    int THREAD_COUNT_X = BN / TN;
    int THREAD_COUNT = THREAD_COUNT_Y * THREAD_COUNT_X;
    
    // // // ========= 朴素表达 ==============
    // for(int i=0;i<M;++i){
    //     for(int j=0;j<N;++j){
    //         for(int k=0;k<K;++k){
    //             C[i][j] += A[i][k] * B[k][j];
    //         }
    //     }
    // }
    // -------------------------------------

    // // // ======= split & reorder ==========
    // for(int i=0;i<M;i += BM){
    //     for(int j=0;j < N;j += BN){   
    //         for(int ii = 0;ii < BM; ii += TM){
    //             for(int jj=0;jj < BN; jj+= TN){
    //                 // thread work
    //                 // BK应当放在这儿 reorder
    //                 for(int k=0;k<K; k+= BK){
    //                     // gm->shm 
    //                     for(int kk = 0;kk < BK; kk += 1){ // 在此做lsu
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
    //                 // TM * TN 的结果此时统一从reg放回gm
    //             }
    //         }
    //     }
    // }
    // -------------------------------------
    
    // // // ======= add splitU  ==========
    // for(int i=0;i<M;i += BM){
    //     for(int j=0;j < N;j += BN){
    //         // block work
    //         for(int ii = 0;ii < BM; ii += TM){
    //             for(int jj=0;jj < BN; jj+= TN){
    //                 for(int k=0;k<K; k+= BK){
    //                     for(int kkk = 0; kkk < SPLIT_U ; kkk += 1){
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
    // -------------------------------------

    // // // ======= splitU reorder ==========
    // for(int i=0;i<M;i += BM){
    //     for(int j=0;j < N;j += BN){
    //         // block work
    //         for(int ii = 0;ii < BM; ii += TM){
    //             for(int jj=0;jj < BN; jj+= TN){
    //                 for(int kkk = 0; kkk < SPLIT_U ; kkk += 1){    
    //                     // thread work 
    //                     for(int k=0;k<K; k+= BK){
    //                         for(int iii = 0; iii < TM; iii += 1){
    //                             for(int jjj = 0;jjj < TN; jjj += 1){
    //                                 for(int kk = 0;kk < BK; kk += SPLIT_U){
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
    // -------------------------------------
    
    // // // // ======= loop merge dims ==========
    // for(int i=0,j=0; i<M && j<N ; i += BM, j += BN){
    //     // block work
    //     for(int ii = 0, jj=0, kkk = 0;ii < BM && jj < BN && kkk < SPLIT_U ; ii += TM, jj+= TN, kkk += 1){
    //         // thread work 
    //         for(int k=0;k<K; k+= BK){
    //             for(int iii = 0; iii < TM; iii += 1){
    //                 for(int jjj = 0;jjj < TN; jjj += 1){
    //                     for(int kk = 0;kk < BK; kk += SPLIT_U){
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
    // -------------------------------------

    // // // ======= 并行化 & loop invariant code motion ==========
    // int i = by * BM , j = bx * BN;  // (bx,by) < [N/BN, M/BM], 这里的gridid排序为: (0,0)->(1,0)
    // // block work
    // int ii = ty * TM, jj = tx * TN, kkk = tz * 1;  // (tx,ty,tz) < [BN/TN, BM/TM, SPLIT_U], 这里的blockid排序为: (0,0,0)->(1,0,0)
    //     // thread work 
    //     for(int k=0;k<K; k+= BK){
    //         //todo : gm -> shm
    //         for(int iii = 0; iii < TM; iii += 1){
    //             for(int jjj = 0;jjj < TN; jjj += 1){
    //                 for(int kk = 0;kk < BK; kk += SPLIT_U){
    //                     int indexI = i + ii + iii;
    //                     int indexJ = j + jj + jjj;
    //                     int indexK = k + kk + kkk;
    //                     C[indexI][indexJ] += A[indexI][indexK] * B[indexK][indexJ];
    //                 }
    //             }
    //         }
    //     }
    //     // reg -> shm  ----c(lsu)
    //     // shm c(lsu) -> reg, regsize:(BM * BN / ( TM * TN * SPLIT_U))  -----reduce 
    //     // reg -> gm(tile c)
    // ------------------------------------------------------



    // ======= warp 处理的区域离散化 & thread 区域离散化 ==========
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
#if 0
    int i = by * BM , j = bx * BN, kkk = tz * 1;  // [bx,by,tz] < [N/BN, M/BM, SPLIT_U] 
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
                    int warpIndexX = BN/warpRepeatX * wi + warpIdx*BN/(warpRepeatX * BLOCK_LAYOUT_X) ;  // offs + base
                    int warpIndexY = BM/warpRepeatY * wj + warpIdy*BM/(BLOCK_LAYOUT_Y * warpRepeatY) ;
                    // thread 离散化 （重映射 iii,jjj->x_offs,y_offs）
                    for(int p = 0;p < threadRepeatX;++p){
                        for(int q=0;q < threadRepeatY;++q){
                            int x_offs = BN / (warpRepeatX * threadRepeatX) * p + laneIdx * THREAD_SCATTER_SIZE_X; // offs + base
                            int y_offs = BM / (warpRepeatY * threadRepeatY) * q + laneIdy * THREAD_SCATTER_SIZE_Y;
                            // 重映射后的组装 (i+ii+iii, j+jj+jjj -> xx,yy)
                            int xx = bx * BN + warpIndexX + x_offs ;
                            int yy = by * BM + warpIndexY + y_offs ;
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
             * indexK = k + kk + kkk == h(iv_K, iv_BK, iv_SPLIT_U), iv_SPLIT_U=tz 
             * so indexK = h(iv_K, iv_BK, tz)
             * 
             * 可得到 ABC affinemap的所需维度：
             * affinemapC = map(_y,_x) = 
             *          map(f(iv_warpRepeatY, iv_threadRepeatY, iv_TSSY), g(iv_warpRepeatX, iv_threadRepeatX, iv_TSSX))
             *          dim = [iv_warpRepeatY, iv_threadRepeatY, iv_TSSY, iv_warpRepeatX, iv_threadRepeatX, iv_TSSX]
             *              + [ty,tx,bx,by,tz]
             * affinemapA = map(_y,indexK) = 
             *          map(f(iv_warpRepeatY, iv_threadRepeatY, iv_TSSY), h(iv_K, iv_BK, tz))
             *          dim = [iv_warpRepeatY, iv_threadRepeatY, iv_TSSY, iv_K, iv_BK, tz] + [ty,tx,by,bx]
             * 
             * affinemapB = map(indexK,_x) = 
             *          map(h(iv_K, iv_BK, tz), g(iv_warpRepeatX, iv_threadRepeatX, iv_TSSX))
             *          dim = [iv_K, iv_BK, tz,iv_warpRepeatX, iv_threadRepeatX, iv_TSSX] + [ty,tx,by,bx]
             */
        }
    }
#endif
    
    // ======= add global->shm & shm->reg ========
#if 1
    int i = by * BM , j = bx * BN, kkk = tz * 1;  // [bx,by,tz] < [N/BN, M/BM, SPLIT_U] 
    int ii = ty * TM, jj = tx * TN;  // [tx,ty] < [BN/TN, BM/TM]
    int tid = ty * blockDimX + tx;

    int laneId = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int warpIdx = warpId % BLOCK_LAYOUT_X;
    int warpIdy = warpId / BLOCK_LAYOUT_X;
    int laneIdx = laneId % WARP_LAYOUT_X;
    int laneIdy = laneId / WARP_LAYOUT_X;

    // global->shm
    int GLOBAL_LOAD_WIDTH_A = 8, GLOBAL_LOAD_WIDTH_B = 8;  // 设定的搬运最大宽度
    std::vector<int> validLoadWidth = {8,4,2,1};  // 可用值
    int GlobalLoadTargetWidthB = BN*BK/THREAD_COUNT;



    float** smA;  // BK * BM
    float** smB;  // BK * BN

    float* regA;  // TM * 1
    float* regB;  // TN * 1
    float** regC;  // TM * TN
    // block work
    for(int k=0;k<K; k+= BK){   
        // globalA->shmA
        int GlobalLoadTargetWidthA = BM*BK/THREAD_COUNT;  // 每个线程应该搬运多少数字
        {
            int remain = GlobalLoadTargetWidthA;
            int maxwidth = remain > GLOBAL_LOAD_WIDTH_A ? 
                GLOBAL_LOAD_WIDTH_A : nextValidWidth(GlobalLoadTargetWidthA);
            int threadNeedsPerLine = BM / maxwidth;
            int copyCount = remain / maxwidth;  // 需要搬运几次
            for(int i_=0;i_<copyCount;++i_){
                auto virtualTid = tid + i_ * THREAD_COUNT;
                int coordX = virtualTid % threadNeedsPerLine;
                int coordY = virtualTid / threadNeedsPerLine;
                vectorize_copy(&A[k+coordY][coordX] , &smA[coordY][coordX], maxwidth);
                // GlboalAToTempAMap : [k+coordY][coordX] -> reg
                // dims [ivK, ty, BLOCKDIMX ,tx , BM, _maxwidth, ivCpyCount, THREAD_COUNT]
                // [k + virtualTid % threadNeedsPerLine]
                // TempAToSMAMap : reg-> [coordY][coordX]
                // i.e. [ty, BLOCKDIMX ,tx , ivCpyCount , THREAD_COUNT, BM , _maxwidth ]
            }
            remain = remain % maxwidth;  // 更新remain
            if(remain > 0){
                // 复制以上代码（update remain, update width, update copyCount, update threadNeedsPerLine）
                // check remain > 0
                // 直到remain == 0 结束拷贝。 代码总体作为 global->shm 的代码
            }
        }
        // globalB->shmB
        int GlobalLoadTargetWidthB = BN*BK/THREAD_COUNT;  // 每个线程应该搬运多少数字
        {
            int remain = GlobalLoadTargetWidthB;
            int maxwidth = remain > GLOBAL_LOAD_WIDTH_B ? 
                GLOBAL_LOAD_WIDTH_B : nextValidWidth(GlobalLoadTargetWidthB);
            int threadNeedsPerLine = BN / maxwidth;
            int copyCount = remain / maxwidth;  // 需要搬运几次
            for(int i_=0;i_<copyCount;++i_){
                auto virtualTid = tid + i_ * THREAD_COUNT;
                int coordX = virtualTid % threadNeedsPerLine;
                int coordY = virtualTid / threadNeedsPerLine;
                vectorize_copy(&B[k+coordY][coordX] , &smB[coordY][coordX], maxwidth);
                // GlboalBToTempBMap : [k+coordY][coordX] -> reg
                // dims : [ivK, ty, tx, blockDimX ,ivCopyCount , THREAD_COUNT, BN , maxwidth]
                // TempBToSMBMap : reg-> [coordY][coordX]
            }
            remain = remain % maxwidth;  // 更新remain
            if(remain > 0){
                // 复制以上代码（update remain, update width, update copyCount, update threadNeedsPerLine）
                // check remain > 0
                // 直到remain == 0 结束拷贝。 代码总体作为 global->shm 的代码
            }
        }
        sync();  // 等待拷贝完成
        // thread caculate C
        for(int kk = 0;kk < BK; kk += SPLIT_U){
            // thread 处理连续的 TM*TN 大小 -> 处理离散的几个小连续区域. 映射关系由warpId laneId wss tss导出
            // warp 离散化 (重映射 i+ii, j+jj -> warpIndexX, warpIndexY)
            for(int wi = 0; wi < warpRepeatX; ++wi){
                for(int wj = 0; wj < warpRepeatY;++wj){
                    int warpIndexX = BN/warpRepeatX * wi + warpIdx*BN/(warpRepeatX * BLOCK_LAYOUT_X) ;  // offs + base
                    int warpIndexY = BM/warpRepeatY * wj + warpIdy*BM/(BLOCK_LAYOUT_Y * warpRepeatY) ;
                    // thread 离散化 （重映射 iii,jjj->x_offs,y_offs）
                    for(int p = 0;p < threadRepeatX;++p){
                        for(int ivThreadRepeatY = 0;ivThreadRepeatY < threadRepeatY;++ivThreadRepeatY){
                            int x_offs = BN / (warpRepeatX * threadRepeatX) * p + laneIdx * THREAD_SCATTER_SIZE_X; // offs + base
                            int y_offs = BM / (warpRepeatY * threadRepeatY) * ivThreadRepeatY + laneIdy * THREAD_SCATTER_SIZE_Y;
                            // 重映射后的组装 (i+ii+iii, j+jj+jjj -> xx,yy)
                            // int xx = bx * BN + warpIndexX + x_offs ;
                            // int yy = by * BM + warpIndexY + y_offs ;
                            int indexK = k + kk + kkk;
                            // shm -> reg
                            // 由于引入了sm，故只需要考虑 tid 在 sm的位置映射即可
                            int xx_ = warpIndexX + x_offs ;
                            int yy_ = warpIndexY + y_offs ;
                            for(int m = 0;m < THREAD_SCATTER_SIZE_X;++m){
                                int _x = xx_ + m;
                                regB[_x] = smB[indexK][_x];
                                // SMBToTempMap: [indexK][_x]
                                /**
                                 * @brief 
                                 * dims : [ivK , ivBK, tz, BN,WARPREPEATX, ivWRX , tx,ty,BLOCKDIMX, 
                                 *          WARP_SIZE, BLOCK_LAYOUT_X , THREADREPEATX, ivThreadRepeatX , 
                                 *       THREAD_SCATTER_SIZE_X , ivTHREAD_SCATTER_SIZE_X]
                                 */
                                // 
                                // TempToRegBMap : [_x] 
                            }
                            for(int ivTSSY=0;ivTSSY < THREAD_SCATTER_SIZE_Y;++ivTSSY){
                                int _y = yy_ + ivTSSY;
                                regA[_y] = smA[indexK][_y];
                                // SMAToTempMap: [indexK][_y]
                                // dims : [ivK , ivBK , tz , yy_ + ivTSSY]
                                // TempToRegAMap : [_y] 
                            }

                            // 连续小区域( tssx * tssy 大小)
                            for(int m = 0;m<THREAD_SCATTER_SIZE_X;++m){
                                for(int n = 0;n<THREAD_SCATTER_SIZE_Y;++n){
                                    int _y = yy_ + n;
                                    int _x = xx_ + m;
                                    regC[_y][_x] += regA[_y] * regB[_x];
                                    // ReduceRegCMap : [_y][_x]
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    sync();  // 等待计算C完成
    // write back C ( 向量化? ) ：根据 splitU的数目，复用sm，将旧值写回sm，然后串行化累加，最后写回globalC
    for(int i=0;i<TM;++i){
        for(int j=0;j<TN;++j){
            C[i][j] = regC[i][j];
        }
    }

#endif
    
    
    
    
    return 0;
}