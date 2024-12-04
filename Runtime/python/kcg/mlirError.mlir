//  === start mlir =====
module {
  func.func public @GEMM_mnk1024x512x128_f32f32f32_TTmn4x4_BTmnk64x64x16(%A: memref<128x1024xf32, 1>, %B: memref<128x512xf32, 1>, %C: memref<1024x512xf32, 1>) attributes {func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
    affine.parallel (%arg3, %arg4) = (0, 0) to (16, 8) {
      %alloc = memref.alloc() : memref<2x16x64xf32, 3>
      %alloc_0 = memref.alloc() : memref<2x16x64xf32, 3>
      %0 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)
      %1 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg4)
      affine.parallel (%arg5, %arg6) = (0, 0) to (16, 16) {
        %c0 = arith.constant 0 : index
        %c0_1 = arith.constant 0 : index
        %c15 = arith.constant 15 : index
        %c0_2 = arith.constant 0 : index
        %c0_3 = arith.constant 0 : index
        %c0_4 = arith.constant 0 : index
        %c0_5 = arith.constant 0 : index
        %alloca = memref.alloca() {alignment = 16 : i64} : memref<4xf32, 5>
        %alloca_6 = memref.alloca() {alignment = 16 : i64} : memref<4xf32, 5>
        %alloca_7 = memref.alloca() {alignment = 16 : i64} : memref<2x4xf32, 5>
        %alloca_8 = memref.alloca() {alignment = 16 : i64} : memref<2x4xf32, 5>
        %alloca_9 = memref.alloca() {alignment = 16 : i64} : memref<4x4xf32, 5>
        %2 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg5)
        %3 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg6)
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %cst = arith.constant 0.000000e+00 : f32
            affine.store %cst, %alloca_9[%arg7, %arg8] : memref<4x4xf32, 5>
          } {affine.loop = "unroll"}
        } {affine.loop = "unroll"}
        %4 = affine.vector_load %A[%arg5 * 4 + %arg6 floordiv 4 + %c0 * 64 + %arg3 * 64, (%arg6 mod 4) * 4 + %c0_4] : memref<128x1024xf32, 1>, vector<4xf32>
        affine.vector_store %4, %alloca[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
        %5 = affine.vector_load %B[%arg5 + %arg6 floordiv 16 + %c0 * 16 + %c0_5, (%arg6 mod 16) * 4 + %arg4 * 64] : memref<128x512xf32, 1>, vector<4xf32>
        affine.vector_store %5, %alloca_6[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
        affine.for %arg7 = 0 to 4 {
          %9 = affine.vector_load %alloca[%c0 * 4 + %arg7] : memref<4xf32, 5>, vector<1xf32>
          affine.vector_store %9, %alloc[0, (%arg6 mod 4) * 4 + %arg7, %arg5 * 4 + %arg6 floordiv 4 + %c0 * 64] : memref<2x16x64xf32, 3>, vector<1xf32>
        } {affine.loop = "unroll"}
        %6 = affine.vector_load %alloca_6[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
        affine.vector_store %6, %alloc_0[0, %arg5 + %arg6 floordiv 16 + %c0 * 16, (%arg6 mod 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        gpu.barrier
        %7 = affine.vector_load %alloc[(%c0_1 floordiv 16) mod 2, %c0_2, (((%arg5 * 16 + %arg6) mod 64) floordiv 16 + ((%arg5 * 16 + %arg6) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        affine.vector_store %7, %alloca_7[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        %8 = affine.vector_load %alloc_0[(%c0 floordiv 16) mod 2, %c0_3, (%arg6 mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        affine.vector_store %8, %alloca_8[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        affine.for %arg7 = 0 to 128 step 16 {
          affine.if affine_set<(d0) : (-d0 + 96 >= 0)>(%arg7) {
            %11 = affine.vector_load %A[%arg5 * 4 + %arg6 floordiv 4 + %c0 * 64 + %arg3 * 64, (%arg6 mod 4) * 4 + %arg7 + 16] : memref<128x1024xf32, 1>, vector<4xf32>
            affine.vector_store %11, %alloca[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
            %12 = affine.vector_load %B[%arg5 + %arg6 floordiv 16 + %c0 * 16 + %arg7 + 16, (%arg6 mod 16) * 4 + %arg4 * 64] : memref<128x512xf32, 1>, vector<4xf32>
            affine.vector_store %12, %alloca_6[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
          }
          affine.for %arg8 = 0 to 15 {
            %11 = affine.vector_load %alloc[(%arg7 floordiv 16) mod 2, %arg8 + 1, (((%arg5 * 16 + %arg6) mod 64) floordiv 16 + ((%arg5 * 16 + %arg6) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
            affine.vector_store %11, %alloca_7[(%arg8 + 1) mod 2, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
            %12 = affine.vector_load %alloc_0[(%arg7 floordiv 16) mod 2, %arg8 + 1, (%arg6 mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
            affine.vector_store %12, %alloca_8[(%arg8 + 1) mod 2, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
            affine.for %arg9 = 0 to 4 {
              affine.for %arg10 = 0 to 4 {
                %13 = affine.load %alloca_9[%arg9, %arg10] : memref<4x4xf32, 5>
                %14 = affine.load %alloca_7[%arg8 mod 2, %arg9] : memref<2x4xf32, 5>
                %15 = affine.load %alloca_8[%arg8 mod 2, %arg10] : memref<2x4xf32, 5>
                %16 = arith.mulf %14, %15 : f32
                %17 = arith.addf %16, %13 : f32
                affine.store %17, %alloca_9[%arg9, %arg10] : memref<4x4xf32, 5>
              } {affine.loop = "unroll"}
            } {affine.loop = "unroll"}
          } {affine.loop = "unroll"}
          affine.if affine_set<(d0) : (-d0 + 96 >= 0)>(%arg7) {
            affine.for %arg8 = 0 to 4 {
              %12 = affine.vector_load %alloca[%c0 * 4 + %arg8] : memref<4xf32, 5>, vector<1xf32>
              affine.vector_store %12, %alloc[(%arg7 floordiv 16 + 1) mod 2, (%arg6 mod 4) * 4 + %arg8, %arg5 * 4 + %arg6 floordiv 4 + %c0 * 64] : memref<2x16x64xf32, 3>, vector<1xf32>
            } {affine.loop = "unroll"}
            %11 = affine.vector_load %alloca_6[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
            affine.vector_store %11, %alloc_0[(%arg7 floordiv 16 + 1) mod 2, %arg5 + %arg6 floordiv 16 + %c0 * 16, (%arg6 mod 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
            gpu.barrier
          }
          affine.for %arg8 = 0 to 4 {
            affine.for %arg9 = 0 to 4 {
              %11 = affine.load %alloca_9[%arg8, %arg9] : memref<4x4xf32, 5>
              %12 = affine.load %alloca_7[%c15 mod 2, %arg8] : memref<2x4xf32, 5>
              %13 = affine.load %alloca_8[%c15 mod 2, %arg9] : memref<2x4xf32, 5>
              %14 = arith.mulf %12, %13 : f32
              %15 = arith.addf %14, %11 : f32
              affine.store %15, %alloca_9[%arg8, %arg9] : memref<4x4xf32, 5>
            } {affine.loop = "unroll"}
          } {affine.loop = "unroll"}
          %9 = affine.vector_load %alloc_0[(%arg7 floordiv 16 + 1) mod 2, %c0_3, (%arg6 mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
          affine.vector_store %9, %alloca_8[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
          %10 = affine.vector_load %alloc[(%arg7 floordiv 16 + 1) mod 2, %c0_2, (((%arg5 * 16 + %arg6) mod 64) floordiv 16 + ((%arg5 * 16 + %arg6) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
          affine.vector_store %10, %alloca_7[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        } {affine.loop = "unroll"}
        affine.for %arg7 = 0 to 4 {
          %9 = affine.vector_load %alloca_9[%c0 + %arg7, %c0 + %c0] : memref<4x4xf32, 5>, vector<4xf32>
          affine.vector_store %9, %C[%arg3 * 64 + (((%arg5 * 16 + %arg6) mod 64) floordiv 16 + ((%arg5 * 16 + %arg6) floordiv 64 + (%c0 floordiv 4) * 4) * 4) * 4 + %arg7, %arg4 * 64 + (%arg6 mod 16 + (%c0 floordiv 4) * 16) * 4 + %c0] : memref<1024x512xf32, 1>, vector<4xf32>
        } {affine.loop = "unroll"}
      }
    }
    return
  }
}