#include <vector>

// 从列表里寻找下一个可用的validwidth 用于搬运 global->shm
int nextValidInstWidth(int remain){
    std::vector<int> validInstWidth = {8,4,2,1};
    for(auto instWidth : validInstWidth){
        if(remain >= instWidth){
            return instWidth;
        }
    }
    return 0;
}

void inst_vecCpy(float* from, float* to,int instwidth){
    // 向量化读取的指令 如 flaot4读, float2读 等等
}

// 使用一连串 veccpy 指令将 targetWidth 长度的数据从 from 拷贝到 to。连续 veccpy 指令的宽度根据剩余长度自适应计算
int vectorize_copy_with_sequence_inst(float* from, float* to, int targetWidth){
    int remain = targetWidth;
    int instWidth = nextValidInstWidth(remain);
    while(remain > 0){
        inst_vecCpy(from,to,instWidth);
        from += instWidth;
        to += instWidth;
        remain -= instWidth;
        instWidth = nextValidInstWidth(remain);
    }
}

#define vectorize_copy_with_sequence_inst( from,  to, targetWidth)  \
{   \
    int remain = targetWidth;   \
    int instWidth = nextValidInstWidth(remain); \
    int offs = 0;   \
    while(remain > 0){  \
        inst_vecCpy(from + offs,to + offs,instWidth); \
        offs += instWidth;  \
        remain -= instWidth;    \
        instWidth = nextValidInstWidth(remain); \
    }   \
}   \




/**
 * @brief 组拷贝：根据targetWidth和groupWidth，以及可用的instWidth进行向量化拷贝指令组合，将数据从from拷贝到to
 * 
 * @param from 从哪copy
 * @param to copy到哪里
 * @param groupWidth 用户设定。可以是任意值。但必须能够由 validInstWidth 组合成。且 targetWidth % groupWidth==0
 * @param targetWidth 单个线程应该搬运的数字个数。通过总数据量/线程数 计算得到
 * @return int 
 */
int group_copy(float* from, float* to, int groupWidth, int targetWidth, int THREAD_COUNT){
    int count = targetWidth / groupWidth;  // 单个线程需要搬运几轮
    int width = groupWidth;
    if(count < 1){
        // 轮数<1, 说明 groupWidth 超了。直接缩短为 targetWidth
        width = targetWidth;
        count = 1;
    }
    int stride = THREAD_COUNT * width;
    for(int iv=0;iv<count;++iv){
        from += iv * stride;
        to += iv * stride;
        int remain = width;  // 每轮应该拷贝的宽度
        auto instWidth = nextValidInstWidth(remain);  // 寻找可用指令宽度
        while(remain > 0 && instWidth > 0){
            inst_vecCpy(from,to,instWidth);  // 插入真实的向量化拷贝指令
            from += instWidth;  // addr ptr offset
            to += instWidth;
            remain -= instWidth;
            instWidth = nextValidInstWidth(remain);
        }
    }
}

#define group_copy(from, to, groupWidth,  targetWidth, THREAD_COUNT)   \
{   \
    int count = targetWidth / groupWidth;   \ 
    int width = groupWidth; \
    if(count < 1){  \
        width = targetWidth;    \
        count = 1;  \
    }   \
    for(int ivLoad=0;ivLoad<count;++ivLoad){    \
        from += ivLoad * THREAD_COUNT * width;  \
        to += ivLoad * THREAD_COUNT * width;    \
        int remain = width;    \
        auto instWidth = nextValidInstWidth(remain);     \
        while(remain > 0 && instWidth > 0){ \
            inst_vecCpy(from,to,instWidth);      \
            from += instWidth;    \
            to += instWidth;    \
            remain -= instWidth;    \
            instWidth = nextValidInstWidth(remain); \
        }   \
    }   \
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
    const int WARP_LAYOUT_N = 16;
    const int WARP_LAYOUT_M = 4;
    const int BLOCK_LAYOUT_N = 1;
    const int BLOCK_LAYOUT_M = 4;
    int THREAD_COUNT_Y = BM / TM;
    int THREAD_COUNT_X = BN / TN;
    int THREAD_COUNT = THREAD_COUNT_Y * THREAD_COUNT_X * SPLIT_U;
    
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
    // 这里需要考虑相邻线程处理的数据是怎样分布的，来决定 x y z 分别对应的维度
    // 将线程模型等效为一个立方体。其具有三个轴(m,n,u) 对应不同的方向
    // 按照计算区域，称处理同一个C[M*N]大小的U个线程为 同一个 batch 的线程
    // “物理上”相邻的线程为 (0,0,0),(1,0,0),(2,0,0) ... (x,0,0)(0,1,0),(1,1,0),(2,1,0)...(x,y,0)(0,0,1)(1,0,1)...(x,y,z) 即按照xyz的顺序递增
    // 关键：
    // 同一个 batch 的线程，
    // int i = by * BM , j = bx * BN;  // (bx,by) < [N/BN, M/BM], 这里的bid排序为: (0,0)->(1,0)
    // // block work
    // int ii = ty * TM, jj = tx * TN, kkk = tz * 1;  // (tx,ty,tz) < [BN/TN, BM/TM, SPLIT_U], 这里的tid排序为: (0,0,0)->(1,0,0)
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
    int tid = ty * blockDimX + tx + tz* blockDimX * blockDimY;
    // 即： (bx,by) (tx,ty,tz) 划分方式
    // block work
    int laneId = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int warpIdx = warpId % BLOCK_LAYOUT_N;
    int warpIdy = warpId / BLOCK_LAYOUT_N;
    int laneIdx = laneId % WARP_LAYOUT_N;
    int laneIdy = laneId / WARP_LAYOUT_N;

    // thread work
    for(int k=0;k<K; k+= BK){       
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
                    int warpIndexX = BN/warpRepeatX * wi + warpIdx*BN/(warpRepeatX * BLOCK_LAYOUT_N) ;  // offs + base
                    int warpIndexY = BM/warpRepeatY * wj + warpIdy*BM/(BLOCK_LAYOUT_M * warpRepeatY) ;
                    // thread 离散化 （重映射 iii,jjj->x_offs,y_offs）
                    for(int ti = 0;ti < threadRepeatX;++p){
                        for(int tj=0;tj < threadRepeatY;++q){
                            int x_offs = BN / (warpRepeatX * threadRepeatX) * ti + laneIdx * THREAD_SCATTER_SIZE_X; // offs + base
                            int y_offs = BM / (warpRepeatY * threadRepeatY) * tj + laneIdy * THREAD_SCATTER_SIZE_Y;
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
    
    // ======= add global->shm & shm->reg, 以及 reduce smC to regC, 以及 regC写回globalC ========
#if 1
    int i = by * BM , j = bx * BN;  // [bx,by,tz] < [N/BN, M/BM ] 
    int ii = ty * TM, jj = tx * TN, kkk = tz * 1;  // [tx,ty,tz] < [BN/TN, BM/TM, SPLIT_U]
    int tid = ty * blockDimX + tx + tz * blockDimX * blockDimY;


    int laneId = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int warpIdx = warpId % BLOCK_LAYOUT_N;
    int warpIdy = warpId / BLOCK_LAYOUT_N;
    int laneIdx = laneId % WARP_LAYOUT_N;
    int laneIdy = laneId / WARP_LAYOUT_N;

    // global->shm
    // 搬运算法：
    int GLOBAL_LOAD_WIDTH_A = 8, GLOBAL_LOAD_WIDTH_B = 8;  // 设定的搬运最大宽度
    std::vector<int> validLoadWidth = {8,4,2,1};  // 可用值
    int GlobalLoadTargetWidthB = BN*BK/THREAD_COUNT;  // 每个线程应该搬运多少？
    // 规定： GlobalLoadTargetWidth % GLOBAL_LOAD_WIDTH == 0
    // 之后，根据可用宽度列表 validLoadWidth， 对 GLOBAL_LOAD_WIDTH 做组合
    // 6 = 4+2
    // 9 = 4+4+1 ...

    float** smA;  // BK * BM
    float** smB;  // BK * BN
    float** smC; // BN * BM * U

    float* regA;  // TM * 1
    float* regB;  // TN * 1
    float** regC;  // TM * TN
    // block work
    for(int k=0;k<K; k+= BK){   
        // globalA->shmA
        int GlobalLoadTargetWidthA = BM*BK/THREAD_COUNT;  // 每个线程应该搬运多少数字
        int loadCountA = GlobalLoadTargetWidthA / GLOBAL_LOAD_WIDTH_A;
        int _widthA = GLOBAL_LOAD_WIDTH_A;
        if(loadCountA < 1){
            _widthA = GlobalLoadTargetWidthA;
            loadCountA = 1;
        }
        for(int ila = 0;ila < loadCountA;++ila)
        {
            int threadNeedsPerLine = BM / _widthA;
            auto virtualTid = tid + ila * THREAD_COUNT;
            int coordX = virtualTid % threadNeedsPerLine;
            int coordY = virtualTid / threadNeedsPerLine;
            int remain = _widthA;
            int instwidth = nextValidInstWidth(remain);
            int __x = coordX;
            while (remain > 0 && instwidth > 0)
            {
                inst_vecCpy(&A[k+coordY][__x] , &smA[coordY][__x], instwidth);
                remain -= instwidth;
                instwidth = nextValidInstWidth(remain);
                __x += instwidth;
            }
        }
        // globalb->shmB
        int GlobalLoadTargetWidthB = BN*BK/THREAD_COUNT;  // 每个线程应该搬运多少数字
        int loadCountB = GlobalLoadTargetWidthB / GLOBAL_LOAD_WIDTH_B;
        int widthB = GLOBAL_LOAD_WIDTH_B;
        if(loadCountB < 0){
            widthB = GlobalLoadTargetWidthB;
            loadCountB = 1;
        }
        for(int ilb = 0;ilb < loadCountB;++ilb)
        {
            int threadNeedsPerLine = BN / widthB;
            auto virtualTid = tid + ilb * THREAD_COUNT;
            int coordX = virtualTid % threadNeedsPerLine;
            int coordY = virtualTid / threadNeedsPerLine;
            int remain = widthB;
            int instwidth = nextValidInstWidth(remain);
            int __x = coordX;
            while (remain > 0 && instwidth > 0)
            {
                inst_vecCpy(&B[k+coordY][__x] , &smB[coordY][__x], instwidth);
                remain -= instwidth;
                instwidth = nextValidInstWidth(remain);
                __x += instwidth;
            }
        }
 
        sync();  // 等待拷贝完成
        // thread caculate C
        for(int kk = 0;kk < BK; kk += SPLIT_U){
            // thread 处理连续的 TM*TN 大小 -> 处理离散的几个小连续区域. 映射关系由warpId laneId wss tss导出
            // warp 离散化 (重映射 i+ii, j+jj -> warpIndexX, warpIndexY)
            for(int wi = 0; wi < warpRepeatX; ++wi){
                for(int wj = 0; wj < warpRepeatY;++wj){
                    int warpIndexX = BN/warpRepeatX * wi + warpIdx*BN/(warpRepeatX * BLOCK_LAYOUT_N) ;  // offs + base
                    int warpIndexY = BM/warpRepeatY * wj + warpIdy*BM/(BLOCK_LAYOUT_M * warpRepeatY) ;
                    // thread 离散化 （重映射 iii,jjj->x_offs,y_offs）
                    for(int ivTRX = 0;ivTRX < threadRepeatX;++ivTRX){
                        for(int ivThreadRepeatY = 0;ivThreadRepeatY < threadRepeatY;++ivThreadRepeatY){
                            int x_offs = BN / (warpRepeatX * threadRepeatX) * ivTRX + laneIdx * THREAD_SCATTER_SIZE_X; // offs + base
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
                                // (ivTRX, m, wi)->_xreg
                                int _xreg = m + ivTRX * THREAD_SCATTER_SIZE_X + wi * WSSX;
                                regB[_xreg] = smB[indexK][_x];
                                // SMBToTempMap: [indexK][_x]
                                /**
                                 * @brief 
                                 * dims : [ivK , ivBK, tz, BN,WARPREPEATX, ivWRX , tx,ty,BLOCKDIMX, 
                                 *          WARP_SIZE, BLOCK_LAYOUT_N , THREADREPEATX, ivThreadRepeatX , 
                                 *       THREAD_SCATTER_SIZE_X , ivTHREAD_SCATTER_SIZE_X]
                                 */
                                // 
                                // TempToRegBMap : [_x] 
                            }
                            for(int ivTSSY=0;ivTSSY < THREAD_SCATTER_SIZE_Y;++ivTSSY){
                                int _y = yy_ + ivTSSY;
                                int _yreg = ivTSSY + ivThreadRepeatY * THREAD_SCATTER_SIZE_Y + wj * WSSY;
                                regA[_yreg] = smA[indexK][_y];
                                // SMAToTempMap: [indexK][_y]
                                // dims : [ivK , ivBK , tz , yy_ + ivTSSY]
                                // TempToRegAMap : [_y] 
                            }

                            // 连续小区域( tssx * tssy 大小)
                            for(int m = 0;m<THREAD_SCATTER_SIZE_X;++m){
                                for(int n = 0;n<THREAD_SCATTER_SIZE_Y;++n){
                                    // int _y = yy_ + n;
                                    // int _x = xx_ + m;  // sm内的坐标
                                    int _yreg = n + ivThreadRepeatY * THREAD_SCATTER_SIZE_Y + wj * WSSY;
                                    int _xreg = m + ivTRX * THREAD_SCATTER_SIZE_X + wi * WSSX;
                                    regC[_yreg][_xreg] += regA[_yreg] * regB[_xreg];
                                    // ReduceRegCMap : [_y][_x]
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    sync();  // 需要等到 所有的regC计算完毕才可以继续（确保 smA smB 不再被使用）


    // 此时，regC里存放的为 C的部分和,还需要reduce U (tz)
    // regC to smC
    for(int wi = 0; wi < warpRepeatX; ++wi){
        for(int wj = 0; wj < warpRepeatY;++wj){
            int warpIndexX = BN/warpRepeatX * wi + warpIdx*BN/(warpRepeatX * BLOCK_LAYOUT_N) ;  // offs + base
            int warpIndexY = BM/warpRepeatY * wj + warpIdy*BM/(BLOCK_LAYOUT_M * warpRepeatY) ;
            // thread 离散化 （重映射 iii,jjj->x_offs,y_offs）
            for(int ivTRX = 0;ivTRX < threadRepeatX;++ivTRX){
                for(int ivThreadRepeatY = 0;ivThreadRepeatY < threadRepeatY;++ivThreadRepeatY){
                    int x_offs = BN / (warpRepeatX * threadRepeatX) * ivTRX + laneIdx * THREAD_SCATTER_SIZE_X; // offs + base
                    int y_offs = BM / (warpRepeatY * threadRepeatY) * ivThreadRepeatY + laneIdy * THREAD_SCATTER_SIZE_Y;

                    // 由于引入了sm，故只需要考虑 tid 在 sm的位置映射即可
                    int xx_ = warpIndexX + x_offs ;
                    int yy_ = warpIndexY + y_offs ;

                    // 连续小区域( tssx * tssy 大小)  结合向量化搬运做优化
                    // for(int m = 0;m<THREAD_SCATTER_SIZE_X;++m){
                        for(int n = 0;n<THREAD_SCATTER_SIZE_Y;++n){
                            int _y = yy_ + n;
                            // int _x = xx_ + m;
                            int _x = xx_;
                            int u = tz;
                            int _yreg = n + ivThreadRepeatY * THREAD_SCATTER_SIZE_Y + wj * WSSY;
                            int _xreg = 0 + ivTRX * THREAD_SCATTER_SIZE_X + wi * WSSX;
                            // smC[_y + u*BM][_x] = regC[_yreg][_xreg] ;
                            vectorize_copy_with_sequence_inst(&regC[_yreg][_xreg], &smC[_y + u*BM][_x],THREAD_SCATTER_SIZE_X);
                        }
                    // }
                }
            }
        }
    }
    sync();  // 等待所有partsum 写入到smC 完成
    // 至此，smC = (BM*SPLIT_U) * BN, 存满了部分和. smC的单个U对应C上整块连续的部分
    // reduce smC(线程顺着依次计算单个U的内容)

    int smCLoadTargetWidth = BM * BN / THREAD_COUNT;  // 每个线程应该计算多少 == TM*TN / SPLIT_U
    int GLOBAL_STORE_WIDTH = 4;  // 每个线程设定的搬运的最大宽度
    int smCloadCount = smCLoadTargetWidth / GLOBAL_STORE_WIDTH;  // 全部thread总共需要搬运几次
    int width = GLOBAL_STORE_WIDTH;
    if(smCloadCount < 0){
        smCloadCount = 1; width = smCLoadTargetWidth;
    }
    // 规定 smCLoadTargetWidth % smCCfgLoadWidth == 0
    float* tempC = new float[smCLoadTargetWidth];
    for(int iu=0;iu < SPLIT_U;++iu){
        int ic = 0;
        int remain = smCLoadTargetWidth;
        while(remain > 0){
            width = nextValidInstWidth(remain);
            remain -= width;
            for(int ilc = 0;ilc < smCloadCount;++ilc){
                auto vtid = tid + ilc * THREAD_COUNT;
                int y = BN / (tid / width);
                int x = BN % (tid / width);
                inst_vecCpy(&smC[y+iu*BM][x], tempC + ic, width);
                // 累加。放到这里可以起到 访存-计算 交叉排布的效果，提高并行度
                for(int i=0;i<smCLoadTargetWidth;++i){
                    regC[i+ic] += tempC[i+ic];  // 将regC 一维化
                }
                ic += width;
            }
        }
    }
    sync();  // 等待所有regC 计算完毕
    // 写回 globalC
    int y = BN / (tid / smCLoadTargetWidth) + by * BM;
    int x = BN % (tid / smCLoadTargetWidth) + bx * BN;
    
    group_copy(regC, &C[y][x], GLOBAL_STORE_WIDTH, smCLoadTargetWidth, THREAD_COUNT);
    // vectorize_copy(regC, &C[y][x], smCLoadTargetWidth);  // 按照 smCLoadTargetWidth 的宽度拷贝入C。按照指令允许的width可能分多次进行
    


#endif
    
    
    
    
    return 0;
}



// 先定下 pipeline
//（pass的变换过程）
// pipeline只能顺序开发—— optimize 。顺序开发 无法验证 中间IR 
// 朴素IR -> 