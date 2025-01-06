// ======= start point =====
// module {
//   func.func public @GEMM_testKernel(%A: memref<1024x1024xf16, 1>, %B: memref<1024x1024xf16, 1>, %C: memref<1024x1024xf16, 1>) attributes {func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
//     affine.parallel (%by, %bx) = (0, 0) to (16, 22) {
//       %0 = affine.apply affine_map<(d0) -> (d0 * 64)>(%by)
//       %1 = affine.apply affine_map<(d0) -> (d0 * 48)>(%bx)
//       affine.parallel (%tz, %ty, %tx) = (0, 0, 0) to (2, 16, 8) {
//         %2 = affine.apply affine_map<(d0) -> (d0 * 4)>(%ty)
//         %3 = affine.apply affine_map<(d0) -> (d0 * 6)>(%tx)
//         %regC = memref.alloca() {alignment = 16 : i64} : memref<4x6xf16, 5>
//         affine.for %iii = 0 to 4 {
//           affine.for %jjj = 0 to 6 {
//             %cst = arith.constant 0.000000e+00 : f16
//             affine.store %cst, %regC[%iii, %jjj] : memref<4x6xf16, 5>
//             affine.for %k = 0 to 1024 step 32 {
//               affine.for %kk = 0 to 32 step 2 {
//                 %5 = affine.load %regC[%iii, %jjj] : memref<4x6xf16, 5>
//                 %6 = affine.load %A[%k + %kk + %tz, %0 + %2 + %iii] : memref<1024x1024xf16, 1>
//                 %7 = affine.load %B[%k + %kk + %tz, %1 + %3 + %jjj] : memref<1024x1024xf16, 1>
//                 %8 = arith.mulf %6, %7 : f16
//                 %9 = arith.addf %8, %5 : f16
//                 affine.store %9, %regC[%iii, %jjj] : memref<4x6xf16, 5>
//               }
//             }
//             %4 = affine.load %regC[%iii, %jjj] : memref<4x6xf16, 5>
//             affine.store %4, %C[%0 + %2 + %iii, %1 + %3 + %jjj] : memref<1024x1024xf16, 1>
//           }
//         }
//       }
//     }
//     return
//   }
// }


// 目标IR
// BM=64, BN=48, TM=4,TN=6, SPLITU=2, BK=32
// MNK=1024, 
// M/BM = 16, N/BN = 21.33 =22
// thread_count = BM/TM * BN/TN = 16*8

module {
  func.func public @GEMM_testKernel(%A: memref<1024x1024xf16, 1>, %B: memref<1024x1024xf16, 1>, %C: memref<1024x1024xf16, 1>) attributes {func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
    affine.parallel (%by, %bx) = (0, 0) to (16, 22) {
        // block work
      %0 = affine.apply affine_map<(d0) -> (d0 * 64)>(%by)
      %1 = affine.apply affine_map<(d0) -> (d0 * 48)>(%bx)
      affine.parallel (%tz, %ty, %tx) = (0, 0, 0) to (2, 16, 8) {
        // thread work
        %2 = affine.apply affine_map<(d0) -> (d0 * 4)>(%ty)
        %3 = affine.apply affine_map<(d0) -> (d0 * 6)>(%tx)
        %regC = memref.alloca() {alignment = 16 : i64} : memref<4x6xf16, 5>
        affine.for %iii = 0 to 4 {
          affine.for %jjj = 0 to 6 {
            %cst = arith.constant 0.000000e+00 : f16
            affine.store %cst, %regC[%iii, %jjj] : memref<4x6xf16, 5>
            affine.for %k = 0 to 1024 step 32 {
              affine.for %kk = 0 to 32 step 2 {
                %5 = affine.load %regC[%iii, %jjj] : memref<4x6xf16, 5>
                %6 = affine.load %A[%k + %kk + %tz, %0 + %2 + %iii] : memref<1024x1024xf16, 1>
                %7 = affine.load %B[%k + %kk + %tz, %1 + %3 + %jjj] : memref<1024x1024xf16, 1>
                %8 = arith.mulf %6, %7 : f16
                %9 = arith.addf %8, %5 : f16
                affine.store %9, %regC[%iii, %jjj] : memref<4x6xf16, 5>
              }
            }
            %4 = affine.load %regC[%iii, %jjj] : memref<4x6xf16, 5>
            affine.store %4, %C[%0 + %2 + %iii, %1 + %3 + %jjj] : memref<1024x1024xf16, 1>
          }
        }
      }
    }
    return
  }
}


// BM=64, BN=48, TM=4,TN=6, SPLITU=2, BK=32
// MNK=1024, 
// M/BM = 16, N/BN = 21.33 =22
// thread_count = BM/TM * BN/TN = 16*8

module {
  func.func public @GEMM_testKernel(%A: memref<1024x1024xf16, 1>, %B: memref<1024x1024xf16, 1>, %C: memref<1024x1024xf16, 1>) attributes {func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {

    %i = affine.apply affine_map<(d0) -> (d0 * 64)>(%by)
    %j = affine.apply affine_map<(d0) -> (d0 * 48)>(%bx)
    %ii = affine.apply affine_map<(d0) -> (d0 * 4)>(%ty)
    %jj = affine.apply affine_map<(d0) -> (d0 * 6)>(%tx)
    %tid = affine.apply affine_map<(d0,d1,d2)->(d0 + d1 * 16 + d2 * 16*8)>(%tx,%ty,%tz)

    // thread work
    %regC = memref.alloca() {alignment = 16 : i64} : memref<4x6xf16, 5>
    // zero init regC
    affine.for %iii = 0 to 4 {
        affine.for %jjj = 0 to 6 {
        %cst = arith.constant 0.000000e+00 : f16
        affine.store %cst, %regC[%iii, %jjj] : memref<4x6xf16, 5>
        }
    }

    affine.for %k = 0 to 1024 step 32 {
    affine.for %kk = 0 to 32 step 2 {
            affine.for %iii = 0 to 4 {
            affine.for %jjj = 0 to 6 {
                %5 = affine.load %regC[%iii, %jjj] : memref<4x6xf16, 5>
                %6 = affine.load %A[%k + %kk + %tz, %i + %ii + %iii] : memref<1024x1024xf16, 1>
                %7 = affine.load %B[%k + %kk + %tz, %j + %jj + %jjj] : memref<1024x1024xf16, 1>
                %8 = arith.mulf %6, %7 : f16
                %9 = arith.addf %8, %5 : f16
                affine.store %9, %regC[%iii, %jjj] : memref<4x6xf16, 5>
                }
            }
        %4 = affine.load %regC[%iii, %jjj] : memref<4x6xf16, 5>
        affine.store %4, %C[%i + %ii + %iii, %j + %jj + %jjj] : memref<1024x1024xf16, 1>
        }
    }

    return
  }
}



// BM=64, BN=48, TM=4,TN=6, SPLITU=2, BK=32
// MNK=1024, 
// M/BM = 16, N/BN = 21.33 =22
// thread_count = BM/TM * BN/TN = 16*8
// BlockLayoutY = 2, BlockLayoutX = 1,
// ThreadLayout = 8*8
//   const int WARP_SCATTER_WIDTH_A = 2;  warpRepeatX = 3
//   const int WARP_SCATTER_WIDTH_B = 2;  warpRepeatY = 2
// THREAD_SCATTER_SIZE_X = 1
// THREAD_SCATTER_SIZE_Y = 1

module {
  func.func public @GEMM_testKernel(%A: memref<1024x1024xf16, 1>, %B: memref<1024x1024xf16, 1>, %C: memref<1024x1024xf16, 1>) attributes {func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
    %bx = gpu.block_id  x
    %by = gpu.block_id  y
    %tx = gpu.thread_id  x
    %ty = gpu.thread_id  y
    %tz = gpu.thread_id  z

    %i = affine.apply affine_map<(d0) -> (d0 * 64)>(%by)
    %j = affine.apply affine_map<(d0) -> (d0 * 48)>(%bx)
    %ii = affine.apply affine_map<(d0) -> (d0 * 4)>(%ty)
    %jj = affine.apply affine_map<(d0) -> (d0 * 6)>(%tx)
    %tid = affine.apply affine_map<(d0,d1,d2)->(d0 + d1 * 16 + d2 * 16*8)>(%tx,%ty,%tz)

    // thread work
    %regC = memref.alloca() {alignment = 16 : i64} : memref<4x6xf16, 5>
    // zero init regC
    affine.for %iii = 0 to 4 {
        affine.for %jjj = 0 to 6 {
        %cst = arith.constant 0.000000e+00 : f16
        affine.store %cst, %regC[%iii, %jjj] : memref<4x6xf16, 5>
        }
    }
    // warp lane id calculate
    %laneId = affine.apply affine_map<(d0,d1,d2)->((d0 + d1 * 16 + d2 * 16*8) mod 64)>(%tx,%ty,%tz) ; //tid % WARP_SIZE;
    %warpId = affine.apply affine_map<(d0,d1,d2)->((d0 + d1 * 16 + d2 * 16*8) floordiv 64)>(%tx,%ty,%tz) ; //tid / WARP_SIZE;
    %warpIdx = affine.apply affine_map<(d0)->(d0 mod 1)>(%warpId) ;//warpId % BLOCK_LAYOUT_X;
    %warpIdy =  affine.apply affine_map<(d0)->(d0 floordiv 1)>(%warpId); // warpId / BLOCK_LAYOUT_X;
    %laneIdx = affine.apply affine_map<(d0)->(d0 mod 8)>(%laneId); //laneId % WARP_LAYOUT_X;
    %laneIdy = affine.apply affine_map<(d0)->(d0 floordiv 8)>(%laneId);// laneId / WARP_LAYOUT_X;

    affine.for %k = 0 to 1024 step 32 {
        affine.for %kk = 0 to 32 step 2 {
            // warp 离散化
            // for(int wi = 0; wi < warpRepeatX; ++wi){
            // for(int wj = 0; wj < warpRepeatY;++wj){
            affine.for %wi = 0 to 3 step 1 {
                affine.for %wj = 0 to 2 step 1 {
                    // int warpIndexX = BN/warpRepeatX * wi + warpIdx*BN/(warpRepeatX * BLOCK_LAYOUT_X) ;  // offs + base
                    // int warpIndexY = BM/warpRepeatY * wj + warpIdy*BM/(BLOCK_LAYOUT_Y * warpRepeatY) ;
                    %warpIndexX = affine.apply affine_map<(d0,d1)->( 16 * d0 + d1 * 16 )>(%wi,%warpIdx) ;// BN/warpRepeatX * wi + warpIdx*BN/(warpRepeatX * BLOCK_LAYOUT_X) ;  // offs + base
                    %warpIndexY = affine.apply affine_map<(d0,d1)->( 32 * d0 + d1 * 16 )>(%wj,%warpIdy); // BM/warpRepeatY * wj + warpIdy*BM/(BLOCK_LAYOUT_Y * warpRepeatY) ;
                    // thread 离散化 （重映射 iii,jjj->x_offs,y_offs）
                    // for(int ti = 0;ti < threadRepeatX;++ti){
                    //     for(int tj=0;tj < threadRepeatY;++tj){
                    affine.for %ti = 0 to 2 step 1 {
                        affine.for %tj = 0 to 2 step 1 {
                            // int x_offs = BN / (warpRepeatX * threadRepeatX) * ti + laneIdx * THREAD_SCATTER_SIZE_X; // offs + base
                            // int y_offs = BM / (warpRepeatY * threadRepeatY) * tj + laneIdy * THREAD_SCATTER_SIZE_Y;
                            // // 重映射后的组装 (i+ii+iii, j+jj+jjj -> xx,yy)
                            // int xx = bx * BN + warpIndexX + x_offs ;
                            // int yy = by * BM + warpIndexY + y_offs ;
                            // int indexK = k + kk + kkk;
                            %x_offs = affine.apply affine_map<(d0,d1)->(8 * d0 + d1 * 1)>(%ti,%laneIdx)  ; // offs + base
                            %y_offs = affine.apply affine_map<(d0,d1)->(16 * d0 + d1 * 1)>(%tj, %laneIdy); 
                            %xx = affine.apply affine_map<(d0,d1,d2)->(d0 * 48 + d1 + d2)>(%bx, %warpIndexX, %x_offs);
                            %yy = affine.apply affine_map<(d0,d1,d2)->(d0 * 64 + d1 + d2)>(%by, %warpIndexY, %y_offs);
                            %indexK = affine.apply affine_map<(d0,d1,d2)->(d0+d1+d2)>(%k,%kk,%tz);
                            // // 连续小区域( tssx * tssy 大小)
                            // for(int m = 0;m<THREAD_SCATTER_SIZE_X;++m){
                            //     for(int n = 0;n<THREAD_SCATTER_SIZE_Y;++n){
                            //         int _y = yy + n;
                            //         int _x = xx + m;
                            //         C[_y][_x] += A[_y][indexK] * B[indexK][_x];
                            //     }
                            // }
                            affine.for %m=0 to 1 step 1 {
                                affine.for %n=0 to 1 step 1 {
                                    %_y = affine.apply affine_map<(d0,d1)->(d0+d1)>(%yy, %n); // yy + n;
                                    %_x = affine.apply affine_map<(d0,d1)->(d0+d1)>(%xx, %m); // xx + m;
                                    %tempC = affine.load %C[%_y, %_x] : memref<1024x1024xf16, 1>
                                    %tempA = affine.load %A[%_y, %indexK] :  memref<1024x1024xf16, 1>
                                    %tempB = affine.load %B[%indexK, %_x] :  memref<1024x1024xf16, 1>
                                    %20 = arith.mulf %tempA, %tempB : f32
                                    %21 = arith.addf %20, %tempC : f32
                                    affine.store %21, %C[%tj * %n, %ti * %m] : memref<1024x1024xf16, 1>
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return
  }
}
