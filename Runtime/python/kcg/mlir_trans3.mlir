 === after transforms =====
module {
  memref.global "public" @kcg_shm1 : memref<2x16x64xf32, 3> {alignment = 16 : i64}
  memref.global "public" @kcg_shm0 : memref<2x16x64xf32, 3> {alignment = 16 : i64}
  func.func public @GEMM_mnk1024x1024x1024_f32f32f32_TTmn4x4_BTmnk64x64x16(%arg0: memref<1024x1024xf32, 1>, %arg1: memref<1024x1024xf32, 1>, %arg2: memref<1024x1024xf32, 1>) attributes {func.block.dim = array<i32: 16, 16>, func.grid.dim = array<i32: 16, 16>, func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
    %bx = gpu.block_id  x
    %by = gpu.block_id  y
    %tx = gpu.thread_id  x
    %ty = gpu.thread_id  y
    %4 = memref.get_global @kcg_shm0 : memref<2x16x64xf32, 3>
    %5 = memref.get_global @kcg_shm1 : memref<2x16x64xf32, 3>
    %c0 = arith.constant 0 : index
    %c15 = arith.constant 15 : index
    %alloca = memref.alloca() {alignment = 16 : i64} : memref<4xf32, 5>
    %alloca_0 = memref.alloca() {alignment = 16 : i64} : memref<4xf32, 5>
    %alloca_1 = memref.alloca() {alignment = 16 : i64} : memref<2x4xf32, 5>
    %alloca_2 = memref.alloca() {alignment = 16 : i64} : memref<2x4xf32, 5>
    %alloca_3 = memref.alloca() {alignment = 16 : i64} : memref<4x4xf32, 5>
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        affine.store %cst, %alloca_3[%arg3, %arg4] : memref<4x4xf32, 5>
      } {affine.loop = "unroll"}
    } {affine.loop = "unroll"}
    %6 = affine.vector_load %arg0[%ty * 4 + %tx floordiv 4 + %c0 * 64 + %by * 64, (%tx mod 4) * 4 + %c0] : memref<1024x1024xf32, 1>, vector<4xf32>
    affine.vector_store %6, %alloca[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
    %7 = affine.vector_load %arg1[%ty + %tx floordiv 16 + %c0 * 16 + %c0, (%tx mod 16) * 4 + %bx * 64] : memref<1024x1024xf32, 1>, vector<4xf32>
    affine.vector_store %7, %alloca_0[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
    affine.for %arg3 = 0 to 4 {
      %11 = affine.vector_load %alloca[%c0 * 4 + %arg3] : memref<4xf32, 5>, vector<1xf32>
      affine.vector_store %11, %4[0, (%tx mod 4) * 4 + %arg3, %ty * 4 + %tx floordiv 4 + %c0 * 64] : memref<2x16x64xf32, 3>, vector<1xf32>
    } {affine.loop = "unroll"}
    %8 = affine.vector_load %alloca_0[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
    affine.vector_store %8, %5[0, %ty + %tx floordiv 16 + %c0 * 16, (%tx mod 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
    gpu.barrier
    %9 = affine.vector_load %4[(%c0 floordiv 16) mod 2, %c0, (((%ty * 16 + %tx) mod 64) floordiv 16 + ((%ty * 16 + %tx) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
    affine.vector_store %9, %alloca_1[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
    %10 = affine.vector_load %5[(%c0 floordiv 16) mod 2, %c0, (%tx mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
    affine.vector_store %10, %alloca_2[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
    affine.for %arg3 = 0 to 1024 step 16 {
      affine.if affine_set<(d0) : (-d0 + 992 >= 0)>(%arg3) {
        %13 = affine.vector_load %arg0[%ty * 4 + %tx floordiv 4 + %c0 * 64 + %by * 64, (%tx mod 4) * 4 + %arg3 + 16] : memref<1024x1024xf32, 1>, vector<4xf32>
        affine.vector_store %13, %alloca[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
        %14 = affine.vector_load %arg1[%ty + %tx floordiv 16 + %c0 * 16 + %arg3 + 16, (%tx mod 16) * 4 + %bx * 64] : memref<1024x1024xf32, 1>, vector<4xf32>
        affine.vector_store %14, %alloca_0[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 15 {
        %13 = affine.vector_load %4[(%arg3 floordiv 16) mod 2, %arg4 + 1, (((%ty * 16 + %tx) mod 64) floordiv 16 + ((%ty * 16 + %tx) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        affine.vector_store %13, %alloca_1[(%arg4 + 1) mod 2, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        %14 = affine.vector_load %5[(%arg3 floordiv 16) mod 2, %arg4 + 1, (%tx mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        affine.vector_store %14, %alloca_2[(%arg4 + 1) mod 2, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        affine.for %arg5 = 0 to 4 {
          %15 = affine.load %alloca_1[%arg4 mod 2, %arg5] : memref<2x4xf32, 5>
          affine.for %arg6 = 0 to 4 {
            %16 = affine.load %alloca_3[%arg5, %arg6] : memref<4x4xf32, 5>
            %17 = affine.load %alloca_2[%arg4 mod 2, %arg6] : memref<2x4xf32, 5>
            %18 = arith.mulf %15, %17 : f32
            %19 = arith.addf %18, %16 : f32
            affine.store %19, %alloca_3[%arg5, %arg6] : memref<4x4xf32, 5>
          } {affine.loop = "unroll"}
        } {affine.loop = "unroll"}
      } {affine.loop = "unroll"}
      affine.if affine_set<(d0) : (-d0 + 992 >= 0)>(%arg3) {
        affine.for %arg4 = 0 to 4 {
          %14 = affine.vector_load %alloca[%c0 * 4 + %arg4] : memref<4xf32, 5>, vector<1xf32>
          affine.vector_store %14, %4[(%arg3 floordiv 16 + 1) mod 2, (%tx mod 4) * 4 + %arg4, %ty * 4 + %tx floordiv 4 + %c0 * 64] : memref<2x16x64xf32, 3>, vector<1xf32>
        } {affine.loop = "unroll"}
        %13 = affine.vector_load %alloca_0[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
        affine.vector_store %13, %5[(%arg3 floordiv 16 + 1) mod 2, %ty + %tx floordiv 16 + %c0 * 16, (%tx mod 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        gpu.barrier
      }
      affine.for %arg4 = 0 to 4 {
        %13 = affine.load %alloca_1[%c15 mod 2, %arg4] : memref<2x4xf32, 5>
        affine.for %arg5 = 0 to 4 {
          %14 = affine.load %alloca_3[%arg4, %arg5] : memref<4x4xf32, 5>
          %15 = affine.load %alloca_2[%c15 mod 2, %arg5] : memref<2x4xf32, 5>
          %16 = arith.mulf %13, %15 : f32
          %17 = arith.addf %16, %14 : f32
          affine.store %17, %alloca_3[%arg4, %arg5] : memref<4x4xf32, 5>
        } {affine.loop = "unroll"}
      } {affine.loop = "unroll"}
      %11 = affine.vector_load %5[(%arg3 floordiv 16 + 1) mod 2, %c0, (%tx mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
      affine.vector_store %11, %alloca_2[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
      %12 = affine.vector_load %4[(%arg3 floordiv 16 + 1) mod 2, %c0, (((%ty * 16 + %tx) mod 64) floordiv 16 + ((%ty * 16 + %tx) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
      affine.vector_store %12, %alloca_1[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
    }
    affine.for %arg3 = 0 to 4 {
      %11 = affine.vector_load %alloca_3[%c0 + %arg3, %c0 + %c0] : memref<4x4xf32, 5>, vector<4xf32>
      affine.vector_store %11, %arg2[%by * 64 + (((%ty * 16 + %tx) mod 64) floordiv 16 + ((%ty * 16 + %tx) floordiv 64 + (%c0 floordiv 4) * 4) * 4) * 4 + %arg3, %bx * 64 + (%tx mod 16 + (%c0 floordiv 4) * 16) * 4 + %c0] : memref<1024x1024xf32, 1>, vector<4xf32>
    } {affine.loop = "unroll"}
    return
  }
}