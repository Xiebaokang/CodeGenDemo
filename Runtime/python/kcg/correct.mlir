 === start mlir =====
module {
  func.func public @GEMM_mnk1024x1024x512_f32f32f32_TTmn4x4_BTmnk64x64x16(%A: memref<1024x512xf32, 1>, %B: memref<512x1024xf32, 1>, %C: memref<1024x1024xf32, 1>) attributes {func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
    affine.parallel (%BX, %BY) = (0, 0) to (16, 16) {
      %shmA = memref.alloc() : memref<2x16x64xf32, 3>
      %shmB = memref.alloc() : memref<2x16x64xf32, 3>
      %0 = affine.apply affine_map<(d0) -> (d0 * 64)>(%BX)
      %1 = affine.apply affine_map<(d0) -> (d0 * 64)>(%BY)
      affine.parallel (%TX, %TY) = (0, 0) to (16, 16) {
        %c0 = arith.constant 0 : index
        %c0_1 = arith.constant 0 : index
        %c15 = arith.constant 15 : index
        %c0_2 = arith.constant 0 : index
        %c0_3 = arith.constant 0 : index
        %c0_4 = arith.constant 0 : index
        %c0_5 = arith.constant 0 : index
        %tempA = memref.alloca() {alignment = 16 : i64} : memref<4xf32, 5>
        %tempB = memref.alloca() {alignment = 16 : i64} : memref<4xf32, 5>
        %regA = memref.alloca() {alignment = 16 : i64} : memref<2x4xf32, 5>
        %regB = memref.alloca() {alignment = 16 : i64} : memref<2x4xf32, 5>
        %rC = memref.alloca() {alignment = 16 : i64} : memref<4x4xf32, 5>
        %2 = affine.apply affine_map<(d0) -> (d0 * 4)>(%TX)
        %3 = affine.apply affine_map<(d0) -> (d0 * 4)>(%TY)
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 4 {
            %cst = arith.constant 0.000000e+00 : f32
            affine.store %cst, %rC[%arg7, %arg8] : memref<4x4xf32, 5>
          } {affine.loop = "unroll"}
        } {affine.loop = "unroll"}
        // 
        %4 = affine.vector_load %A[%TX * 4 + %TY floordiv 4 + %c0 * 64 + %BX * 64, (%TY mod 4) * 4 + %c0_4] : memref<1024x512xf32, 1>, vector<4xf32>
        affine.vector_store %4, %tempA[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
        %5 = affine.vector_load %B[%TX + %TY floordiv 16 + %c0 * 16 + %c0_5, (%TY mod 16) * 4 + %BY * 64] : memref<512x1024xf32, 1>, vector<4xf32>
        affine.vector_store %5, %tempB[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
        affine.for %arg7 = 0 to 4 {
          %9 = affine.vector_load %tempA[%c0 * 4 + %arg7] : memref<4xf32, 5>, vector<1xf32>
          affine.vector_store %9, %shmA[0, (%TY mod 4) * 4 + %arg7, %TX * 4 + %TY floordiv 4 + %c0 * 64] : memref<2x16x64xf32, 3>, vector<1xf32>
        } {affine.loop = "unroll"}
        %6 = affine.vector_load %tempB[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
        affine.vector_store %6, %shmB[0, %TX + %TY floordiv 16 + %c0 * 16, (%TY mod 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        gpu.barrier
        %7 = affine.vector_load %shmA[(%c0_1 floordiv 16) mod 2, %c0_2, (((%TX * 16 + %TY) mod 64) floordiv 16 + ((%TX * 16 + %TY) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        affine.vector_store %7, %regA[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        %8 = affine.vector_load %shmB[(%c0 floordiv 16) mod 2, %c0_3, (%TY mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        affine.vector_store %8, %regB[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        affine.for %BK = 0 to 512 step 16 {
          affine.if affine_set<(d0) : (-d0 + 480 >= 0)>(%BK) {
            %11 = affine.vector_load %A[%TX * 4 + %TY floordiv 4 + %c0 * 64 + %BX * 64, (%TY mod 4) * 4 + %BK + 16] : memref<1024x512xf32, 1>, vector<4xf32>
            affine.vector_store %11, %tempA[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
            %12 = affine.vector_load %B[%TX + %TY floordiv 16 + %c0 * 16 + %BK + 16, (%TY mod 16) * 4 + %BY * 64] : memref<512x1024xf32, 1>, vector<4xf32>
            affine.vector_store %12, %tempB[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
          }
          affine.for %arg8 = 0 to 15 {
            %11 = affine.vector_load %shmA[(%BK floordiv 16) mod 2, %arg8 + 1, (((%TX * 16 + %TY) mod 64) floordiv 16 + ((%TX * 16 + %TY) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
            affine.vector_store %11, %regA[(%arg8 + 1) mod 2, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
            %12 = affine.vector_load %shmB[(%BK floordiv 16) mod 2, %arg8 + 1, (%TY mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
            affine.vector_store %12, %regB[(%arg8 + 1) mod 2, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
            affine.for %arg9 = 0 to 4 {
              affine.for %arg10 = 0 to 4 {
                %13 = affine.load %rC[%arg9, %arg10] : memref<4x4xf32, 5>
                %14 = affine.load %regA[%arg8 mod 2, %arg9] : memref<2x4xf32, 5>
                %15 = affine.load %regB[%arg8 mod 2, %arg10] : memref<2x4xf32, 5>
                %16 = arith.mulf %14, %15 : f32
                %17 = arith.addf %16, %13 : f32
                affine.store %17, %rC[%arg9, %arg10] : memref<4x4xf32, 5>
              } {affine.loop = "unroll"}
            } {affine.loop = "unroll"}
          } {affine.loop = "unroll"}
          affine.if affine_set<(d0) : (-d0 + 480 >= 0)>(%BK) {
            affine.for %arg8 = 0 to 4 {
              %12 = affine.vector_load %tempA[%c0 * 4 + %arg8] : memref<4xf32, 5>, vector<1xf32>
              affine.vector_store %12, %shmA[(%BK floordiv 16 + 1) mod 2, (%TY mod 4) * 4 + %arg8, %TX * 4 + %TY floordiv 4 + %c0 * 64] : memref<2x16x64xf32, 3>, vector<1xf32>
            } {affine.loop = "unroll"}
            %11 = affine.vector_load %tempB[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
            affine.vector_store %11, %shmB[(%BK floordiv 16 + 1) mod 2, %TX + %TY floordiv 16 + %c0 * 16, (%TY mod 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
            gpu.barrier
          }
          affine.for %arg8 = 0 to 4 {
            affine.for %arg9 = 0 to 4 {
              %11 = affine.load %rC[%arg8, %arg9] : memref<4x4xf32, 5>
              %12 = affine.load %regA[%c15 mod 2, %arg8] : memref<2x4xf32, 5>
              %13 = affine.load %regB[%c15 mod 2, %arg9] : memref<2x4xf32, 5>
              %14 = arith.mulf %12, %13 : f32
              %15 = arith.addf %14, %11 : f32
              affine.store %15, %rC[%arg8, %arg9] : memref<4x4xf32, 5>
            } {affine.loop = "unroll"}
          } {affine.loop = "unroll"}
          %9 = affine.vector_load %shmB[(%BK floordiv 16 + 1) mod 2, %c0_3, (%TY mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
          affine.vector_store %9, %regB[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
          %10 = affine.vector_load %shmA[(%BK floordiv 16 + 1) mod 2, %c0_2, (((%TX * 16 + %TY) mod 64) floordiv 16 + ((%TX * 16 + %TY) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
          affine.vector_store %10, %regA[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        }
        affine.for %arg7 = 0 to 4 {
          %9 = affine.vector_load %rC[%c0 + %arg7, %c0 + %c0] : memref<4x4xf32, 5>, vector<4xf32>
          affine.vector_store %9, %C[%BX * 64 + (((%TX * 16 + %TY) mod 64) floordiv 16 + ((%TX * 16 + %TY) floordiv 64 + (%c0 floordiv 4) * 4) * 4) * 4 + %arg7, %BY * 64 + (%TY mod 16 + (%c0 floordiv 4) * 16) * 4 + %c0] : memref<1024x1024xf32, 1>, vector<4xf32>
        } {affine.loop = "unroll"}
      }
    }
    return
  }
}