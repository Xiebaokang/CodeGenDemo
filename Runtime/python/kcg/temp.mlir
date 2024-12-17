//  === start mlir =====
module {
  func.func public @GEMM_mnk1024x1024x1024_f32f32f32_TTmn8x4_BTmnk256x32x8_BLmn2x2_WLmn16x4(%arg0: memref<1024x1024xf32, 1>, %arg1: memref<1024x1024xf32, 1>, %C: memref<1024x1024xf32, 1>) attributes {func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
    affine.parallel (%by, %bx) = (0, 0) to (4, 32) {
      %alloc = memref.alloc() : memref<2x8x256xf32, 3>
      %alloc_0 = memref.alloc() : memref<2x8x32xf32, 3>
      %0 = affine.apply affine_map<(d0) -> (d0 * 256)>(%by)
      %1 = affine.apply affine_map<(d0) -> (d0 * 32)>(%bx)
      affine.parallel (%ty, %tx) = (0, 0) to (32, 8) {
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c0_1 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c0_2 = arith.constant 0 : index
        %c0_3 = arith.constant 0 : index
        %c0_4 = arith.constant 0 : index
        %c0_5 = arith.constant 0 : index
        %alloca = memref.alloca() {alignment = 16 : i64} : memref<8xf32, 5>
        %alloca_6 = memref.alloca() {alignment = 16 : i64} : memref<1xf32, 5>
        %regA = memref.alloca() {alignment = 16 : i64} : memref<2x8xf32, 5>
        %regB = memref.alloca() {alignment = 16 : i64} : memref<2x4xf32, 5>
        %regC = memref.alloca() {alignment = 16 : i64} : memref<8x4xf32, 5>
        %2 = affine.apply affine_map<(d0) -> (d0 * 8)>(%ty)
        %3 = affine.apply affine_map<(d0) -> (d0 * 4)>(%tx)
        affine.for %arg7 = 0 to 8 {
          affine.for %arg8 = 0 to 4 {
            %cst = arith.constant 0.000000e+00 : f32
            affine.store %cst, %regC[%arg7, %arg8] : memref<8x4xf32, 5>
          } {affine.loop = "unroll"}
        } {affine.loop = "unroll"}
        %4 = affine.vector_load %arg0[%ty * 4 + %tx floordiv 2 + %c0 * 128 + %by * 256, (%tx mod 2) * 4 + %c0_4] : memref<1024x1024xf32, 1>, vector<4xf32>
        affine.vector_store %4, %alloca[%c0 * 4] : memref<8xf32, 5>, vector<4xf32>
        %5 = affine.vector_load %arg0[%ty * 4 + %tx floordiv 2 + %c1 * 128 + %by * 256, (%tx mod 2) * 4 + %c0_4] : memref<1024x1024xf32, 1>, vector<4xf32>
        affine.vector_store %5, %alloca[%c1 * 4] : memref<8xf32, 5>, vector<4xf32>
        %6 = affine.vector_load %arg1[(%ty * 8 + %tx) floordiv 32 + %c0 * 8 + %c0_5, (%ty * 8 + %tx) mod 32 + %bx * 32] : memref<1024x1024xf32, 1>, vector<1xf32>
        affine.vector_store %6, %alloca_6[%c0] : memref<1xf32, 5>, vector<1xf32>
        affine.for %arg7 = 0 to 4 {
          %11 = affine.vector_load %alloca[%c0 * 4 + %arg7] : memref<8xf32, 5>, vector<1xf32>
          affine.vector_store %11, %alloc[0, (%tx mod 2) * 4 + %arg7, %ty * 4 + %tx floordiv 2 + %c0 * 128] : memref<2x8x256xf32, 3>, vector<1xf32>
        } {affine.loop = "unroll"}
        affine.for %arg7 = 0 to 4 {
          %11 = affine.vector_load %alloca[%c1 * 4 + %arg7] : memref<8xf32, 5>, vector<1xf32>
          affine.vector_store %11, %alloc[0, (%tx mod 2) * 4 + %arg7, %ty * 4 + %tx floordiv 2 + %c1 * 128] : memref<2x8x256xf32, 3>, vector<1xf32>
        } {affine.loop = "unroll"}
        %7 = affine.vector_load %alloca_6[%c0] : memref<1xf32, 5>, vector<1xf32>
        affine.vector_store %7, %alloc_0[0, (%ty * 8 + %tx) floordiv 32 + %c0 * 8, (%ty * 8 + %tx) mod 32] : memref<2x8x32xf32, 3>, vector<1xf32>
        gpu.barrier
        %8 = affine.vector_load %alloc[(%c0_1 floordiv 8) mod 2, %c0_2, (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + %c0 * 2) * 16) * 4] : memref<2x8x256xf32, 3>, vector<4xf32>
        affine.vector_store %8, %regA[0, %c0 * 4] : memref<2x8xf32, 5>, vector<4xf32>
        %9 = affine.vector_load %alloc[(%c0_1 floordiv 8) mod 2, %c0_2, (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + %c1 * 2) * 16) * 4] : memref<2x8x256xf32, 3>, vector<4xf32>
        affine.vector_store %9, %regA[0, %c1 * 4] : memref<2x8xf32, 5>, vector<4xf32>
        %10 = affine.vector_load %alloc_0[(%c0 floordiv 8) mod 2, %c0_3, (%tx mod 4 + (((%ty * 8 + %tx) floordiv 64) mod 2 + %c0 * 2) * 4) * 4] : memref<2x8x32xf32, 3>, vector<4xf32>
        affine.vector_store %10, %regB[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        affine.for %arg7 = 0 to 1024 step 8 {
          affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg7) {
            %14 = affine.vector_load %arg0[%ty * 4 + %tx floordiv 2 + %c0 * 128 + %by * 256, (%tx mod 2) * 4 + %arg7 + 8] : memref<1024x1024xf32, 1>, vector<4xf32>
            affine.vector_store %14, %alloca[%c0 * 4] : memref<8xf32, 5>, vector<4xf32>
            %15 = affine.vector_load %arg0[%ty * 4 + %tx floordiv 2 + %c1 * 128 + %by * 256, (%tx mod 2) * 4 + %arg7 + 8] : memref<1024x1024xf32, 1>, vector<4xf32>
            affine.vector_store %15, %alloca[%c1 * 4] : memref<8xf32, 5>, vector<4xf32>
            %16 = affine.vector_load %arg1[(%ty * 8 + %tx) floordiv 32 + %c0 * 8 + %arg7 + 8, (%ty * 8 + %tx) mod 32 + %bx * 32] : memref<1024x1024xf32, 1>, vector<1xf32>
            affine.vector_store %16, %alloca_6[%c0] : memref<1xf32, 5>, vector<1xf32>
          }
          affine.for %arg8 = 0 to 7 {
            %14 = affine.vector_load %alloc[(%arg7 floordiv 8) mod 2, %arg8 + 1, (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + %c0 * 2) * 16) * 4] : memref<2x8x256xf32, 3>, vector<4xf32>
            affine.vector_store %14, %regA[(%arg8 + 1) mod 2, %c0 * 4] : memref<2x8xf32, 5>, vector<4xf32>
            %15 = affine.vector_load %alloc[(%arg7 floordiv 8) mod 2, %arg8 + 1, (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + %c1 * 2) * 16) * 4] : memref<2x8x256xf32, 3>, vector<4xf32>
            affine.vector_store %15, %regA[(%arg8 + 1) mod 2, %c1 * 4] : memref<2x8xf32, 5>, vector<4xf32>
            %16 = affine.vector_load %alloc_0[(%arg7 floordiv 8) mod 2, %arg8 + 1, (%tx mod 4 + (((%ty * 8 + %tx) floordiv 64) mod 2 + %c0 * 2) * 4) * 4] : memref<2x8x32xf32, 3>, vector<4xf32>
            affine.vector_store %16, %regB[(%arg8 + 1) mod 2, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
            affine.for %arg9 = 0 to 8 {
              affine.for %arg10 = 0 to 4 {
                %17 = affine.load %regC[%arg9, %arg10] : memref<8x4xf32, 5>
                %18 = affine.load %regA[%arg8 mod 2, %arg9] : memref<2x8xf32, 5>
                %19 = affine.load %regB[%arg8 mod 2, %arg10] : memref<2x4xf32, 5>
                %20 = arith.mulf %18, %19 : f32
                %21 = arith.addf %20, %17 : f32
                affine.store %21, %regC[%arg9, %arg10] : memref<8x4xf32, 5>
              } {affine.loop = "unroll"}
            } {affine.loop = "unroll"}
          } {affine.loop = "unroll", kcg.desc = "k_inner"}
          affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg7) {
            affine.for %arg8 = 0 to 4 {
              %15 = affine.vector_load %alloca[%c0 * 4 + %arg8] : memref<8xf32, 5>, vector<1xf32>
              affine.vector_store %15, %alloc[(%arg7 floordiv 8 + 1) mod 2, (%tx mod 2) * 4 + %arg8, %ty * 4 + %tx floordiv 2 + %c0 * 128] : memref<2x8x256xf32, 3>, vector<1xf32>
            } {affine.loop = "unroll"}
            affine.for %arg8 = 0 to 4 {
              %15 = affine.vector_load %alloca[%c1 * 4 + %arg8] : memref<8xf32, 5>, vector<1xf32>
              affine.vector_store %15, %alloc[(%arg7 floordiv 8 + 1) mod 2, (%tx mod 2) * 4 + %arg8, %ty * 4 + %tx floordiv 2 + %c1 * 128] : memref<2x8x256xf32, 3>, vector<1xf32>
            } {affine.loop = "unroll"}
            %14 = affine.vector_load %alloca_6[%c0] : memref<1xf32, 5>, vector<1xf32>
            affine.vector_store %14, %alloc_0[(%arg7 floordiv 8 + 1) mod 2, (%ty * 8 + %tx) floordiv 32 + %c0 * 8, (%ty * 8 + %tx) mod 32] : memref<2x8x32xf32, 3>, vector<1xf32>
            gpu.barrier
          }
          affine.for %arg8 = 0 to 8 {
            affine.for %arg9 = 0 to 4 {
              %14 = affine.load %regC[%arg8, %arg9] : memref<8x4xf32, 5>
              %15 = affine.load %regA[%c7 mod 2, %arg8] : memref<2x8xf32, 5>
              %16 = affine.load %regB[%c7 mod 2, %arg9] : memref<2x4xf32, 5>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %17, %14 : f32
              affine.store %18, %regC[%arg8, %arg9] : memref<8x4xf32, 5>
            } {affine.loop = "unroll"}
          } {affine.loop = "unroll"}
          %11 = affine.vector_load %alloc_0[(%arg7 floordiv 8 + 1) mod 2, %c0_3, (%tx mod 4 + (((%ty * 8 + %tx) floordiv 64) mod 2 + %c0 * 2) * 4) * 4] : memref<2x8x32xf32, 3>, vector<4xf32>
          affine.vector_store %11, %regB[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
          %12 = affine.vector_load %alloc[(%arg7 floordiv 8 + 1) mod 2, %c0_2, (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + %c0 * 2) * 16) * 4] : memref<2x8x256xf32, 3>, vector<4xf32>
          affine.vector_store %12, %regA[0, %c0 * 4] : memref<2x8xf32, 5>, vector<4xf32>
          %13 = affine.vector_load %alloc[(%arg7 floordiv 8 + 1) mod 2, %c0_2, (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + %c1 * 2) * 16) * 4] : memref<2x8x256xf32, 3>, vector<4xf32>
          affine.vector_store %13, %regA[0, %c1 * 4] : memref<2x8xf32, 5>, vector<4xf32>
        } {kcg.desc = "k_outer"}
        affine.for %arg7 = 0 to 4 {
          %11 = affine.vector_load %regC[%c0 + %arg7, %c0 + %c0] : memref<8x4xf32, 5>, vector<4xf32>
          affine.vector_store %11, %C[%by * 256 + (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + (%c0 floordiv 4) * 2) * 16) * 4 + %arg7, %bx * 32 + (%tx mod 4 + (((%ty * 8 + %tx) floordiv 64) mod 2 + (%c0 floordiv 4) * 2) * 4) * 4 + %c0] : memref<1024x1024xf32, 1>, vector<4xf32>
        } {affine.loop = "unroll", kcg.desc = "m_inner_1"}
        affine.for %arg7 = 0 to 4 {
          %11 = affine.vector_load %regC[%c4 + %arg7, %c0 + %c0] : memref<8x4xf32, 5>, vector<4xf32>
          affine.vector_store %11, %C[%by * 256 + (%landId floordiv 4 + ( %warpid floordiv 2 + (%c4 floordiv 4) * 2) * 16) * 4 + %arg7, %bx * 32 + (%tx mod 4 + (((%ty * 8 + %tx) floordiv 64) mod 2 + (%c0 floordiv 4) * 2) * 4) * 4 + %c0] : memref<1024x1024xf32, 1>, vector<4xf32>
        } {affine.loop = "unroll", kcg.desc = "m_inner_1"}
      }
    }
    return
  }
}