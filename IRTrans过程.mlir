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
