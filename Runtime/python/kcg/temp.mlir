===== sharedPrefetch =======
module {
  func.func public @GEMM_MNK1024x1024x1024_DTabcfloat32xfloat32xfloat32_AT1_TTmn4x4_BTmnk32x32x8_BLmn1x1_WLmn8x8_GLWab4x4_GSW4_WSWab2x2_TSWab2x2_LSU1_BM16_UNROLL1_REGP0_SHMP1_LC0_RC0_(%A: memref<1024x1024xf32, 1>, %B: memref<1024x1024xf32, 1>, %C: memref<1024x1024xf32, 1>) attributes {func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
    affine.parallel (%arg3, %arg4) = (0, 0) to (32, 32) {
      %smA = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smA"} : memref<2x8x32xf32, 3>
      %smB = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smB"} : memref<2x8x32xf32, 3>
      %0 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg3)
      %1 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg4)
      affine.parallel (%tx) = (0) to (64) {
        %2 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%tx)
        %3 = affine.apply affine_map<(d0) -> ((d0 mod 64) floordiv 8)>(%tx)
        %4 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%tx)
        %tempA = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tempA"} : memref<4xf32, 5>
        %tempB = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tempB"} : memref<4xf32, 5>
        %regA = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regA"} : memref<4xf32, 5>
        %regB = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regB"} : memref<4xf32, 5>
        %5 = affine.apply affine_map<(d0) -> (d0 * 4)>(%3)
        %6 = affine.apply affine_map<(d0) -> (d0 * 4)>(%4)
        %regC = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regC"} : memref<4x4xf32, 5>
        // init regc = 0
        affine.for %arg6 = 0 to 4 {
          affine.for %arg7 = 0 to 4 {
            %cst = arith.constant 0.000000e+00 : f32
            affine.store %cst, %regC[%arg6, %arg7] : memref<4x4xf32, 5>
          }
        }
        ///////////////////////
        // a->tempa
        affine.for %arg6 = 0 to 1 {
          %7 = affine.vector_load %A[%tx floordiv 8, %arg3 * 32 + %arg6 * 32 + (%tx mod 8) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
          affine.vector_store %7, %tempA[%arg6 * 4] : memref<4xf32, 5>, vector<4xf32>
        }
        // b->tempb
        affine.for %arg6 = 0 to 1 {
          %7 = affine.vector_load %B[%tx floordiv 8, %arg4 * 32 + %arg6 * 32 + (%tx mod 8) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
          affine.vector_store %7, %tempB[%arg6 * 4] : memref<4xf32, 5>, vector<4xf32>
        }
        // tempa->sma
        affine.for %arg6 = 0 to 1 {
          %7 = affine.vector_load %tempA[%arg6 * 4] : memref<4xf32, 5>, vector<4xf32>
          affine.vector_store %7, %smA[0, %tx floordiv 8, %arg6 * 32 + (%tx mod 8) * 4] : memref<2x8x32xf32, 3>, vector<4xf32>
        }
        // tempb->smb
        affine.for %arg6 = 0 to 1 {
          %7 = affine.vector_load %tempB[%arg6 * 4] : memref<4xf32, 5>, vector<4xf32>
          affine.vector_store %7, %smB[0, %tx floordiv 8, %arg6 * 32 + (%tx mod 8) * 4] : memref<2x8x32xf32, 3>, vector<4xf32>
        }
        gpu.barrier
        //------------ outer BK ---------------------
        affine.for %iBK = 8 to 1032 step 8 {
          affine.if affine_set<(d0) : (-d0 + 1016 >= 0)>(%iBK) {
            // a->ta
            affine.for %arg7 = 0 to 1 {
              %7 = affine.vector_load %A[%tx floordiv 8 + %iBK, %arg3 * 32 + %arg7 * 32 + (%tx mod 8) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
              affine.vector_store %7, %tempA[%arg7 * 4] : memref<4xf32, 5>, vector<4xf32>
            }
            // b->tb
            affine.for %arg7 = 0 to 1 {
              %7 = affine.vector_load %B[%tx floordiv 8 + %iBK, %arg4 * 32 + %arg7 * 32 + (%tx mod 8) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
              affine.vector_store %7, %tempB[%arg7 * 4] : memref<4xf32, 5>, vector<4xf32>
            }
          }
          // inner Bk
          affine.for %arg7 = 0 to 8 {
            // sma->rega
            affine.for %arg8 = 0 to 2 {
              affine.for %arg9 = 0 to 1 {
                %7 = affine.vector_load %smA[(%iBK floordiv 8 - 1) mod 2, %arg7 + %tx floordiv 64, (%arg8 + (%tx mod 64) floordiv 64) * 16 + (%arg9 * 8 + (%tx mod 64) floordiv 8) * 2] : memref<2x8x32xf32, 3>, vector<2xf32>
                affine.vector_store %7, %regA[%arg8 * 2 + %arg9 * 2] : memref<4xf32, 5>, vector<2xf32>
              }
            }
            // smb->regb
            affine.for %arg8 = 0 to 2 {
              affine.for %arg9 = 0 to 1 {
                %7 = affine.vector_load %smB[(%iBK floordiv 8 - 1) mod 2, %arg7 + %tx floordiv 64, %arg8 * 16 + (%arg9 * 8 + %tx mod 8) * 2] : memref<2x8x32xf32, 3>, vector<2xf32>
                affine.vector_store %7, %regB[%arg8 * 2 + %arg9 * 2] : memref<4xf32, 5>, vector<2xf32>
              }
            }
            // calculate regc
            affine.for %arg8 = 0 to 4 {
              affine.for %arg9 = 0 to 4 {
                %7 = affine.load %regC[%arg8, %arg9] : memref<4x4xf32, 5>
                %8 = affine.load %regA[%arg8] : memref<4xf32, 5>
                %9 = affine.load %regB[%arg9] : memref<4xf32, 5>
                %10 = arith.mulf %8, %9 : f32
                %11 = arith.addf %10, %7 : f32
                affine.store %11, %regC[%arg8, %arg9] : memref<4x4xf32, 5>
              }
            }
          }
          affine.if affine_set<(d0) : (-d0 + 1016 >= 0)>(%iBK) {
            // tempa->sma
            affine.for %arg7 = 0 to 1 {
              %7 = affine.vector_load %tempA[%arg7 * 4] : memref<4xf32, 5>, vector<4xf32>
              affine.vector_store %7, %smA[(%iBK floordiv 8) mod 2, %tx floordiv 8, %arg7 * 32 + (%tx mod 8) * 4] : memref<2x8x32xf32, 3>, vector<4xf32>
            }
            // tempb->smb
            affine.for %arg7 = 0 to 1 {
              %7 = affine.vector_load %tempB[%arg7 * 4] : memref<4xf32, 5>, vector<4xf32>
              affine.vector_store %7, %smB[(%iBK floordiv 8) mod 2, %tx floordiv 8, %arg7 * 32 + (%tx mod 8) * 4] : memref<2x8x32xf32, 3>, vector<4xf32>
            }
            gpu.barrier
          }
        }
        // regC->C
        affine.for %arg6 = 0 to 4 step 2 {
          affine.for %arg7 = 0 to 4 step 2 {
            affine.for %arg8 = 0 to 2 step 2 {
              affine.for %arg9 = 0 to 2 step 2 {
                affine.for %arg10 = 0 to 2 {
                  affine.for %arg11 = 0 to 2 step 2 {
                    %7 = affine.vector_load %regC[%arg6 + %arg8 + %arg10, %arg7 + %arg9 + %arg11] : memref<4x4xf32, 5>, vector<2xf32>
                    affine.vector_store %7, %C[%arg3 * 32 + (%arg6 + (%tx floordiv 64) * 2) * 8 + %arg8 * 8 + ((%tx mod 64) floordiv 8) * 2 + %arg10, %arg4 * 32 + %arg7 * 8 + %arg9 * 8 + (%tx mod 8) * 2 + %arg11] : memref<1024x1024xf32, 1>, vector<2xf32>
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