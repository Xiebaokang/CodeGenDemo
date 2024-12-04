 === after transforms =====
module {
  memref.global "public" @kcg_shm1 : memref<2x16x64xf32, 3> {alignment = 16 : i64}
  memref.global "public" @kcg_shm0 : memref<2x16x64xf32, 3> {alignment = 16 : i64}
  func.func public @GEMM_mnk1024x1024x1024_f32f32f32_TTmn4x4_BTmnk64x64x16(%A: memref<1024x1024xf32, 1>, %B: memref<1024x1024xf32, 1>, %C: memref<1024x1024xf32, 1>) attributes {func.block.dim = array<i32: 16, 16>, func.grid.dim = array<i32: 16, 16>, func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
    %bx = gpu.block_id  x
    %by = gpu.block_id  y
    %tx = gpu.thread_id  x
    %ty = gpu.thread_id  y
    %smA = memref.get_global @kcg_shm0 : memref<2x16x64xf32, 3>
    %smB = memref.get_global @kcg_shm1 : memref<2x16x64xf32, 3>
    %c0 = arith.constant 0 : index
    %c15 = arith.constant 15 : index
    %tempA = memref.alloca() {alignment = 16 : i64} : memref<4xf32, 5>
    %tempB = memref.alloca() {alignment = 16 : i64} : memref<4xf32, 5>
    %regA = memref.alloca() {alignment = 16 : i64} : memref<2x4xf32, 5>
    %regB = memref.alloca() {alignment = 16 : i64} : memref<2x4xf32, 5>
    %regC = memref.alloca() {alignment = 16 : i64} : memref<4x4xf32, 5>
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        affine.store %cst, %regC[%arg3, %arg4] : memref<4x4xf32, 5>
      } {affine.loop = "unroll"}
    } {affine.loop = "unroll"}
    // ok
    %6 = affine.vector_load %A[%ty * 4 + %tx floordiv 4 + %c0 * 64 + %by * 64, (%tx mod 4) * 4 + %c0] : memref<1024x1024xf32, 1>, vector<4xf32>
    affine.vector_store %6, %tempA[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
    // ok
    %7 = affine.vector_load %B[%ty + %tx floordiv 16 + %c0 * 16 + %c0, (%tx mod 16) * 4 + %bx * 64] : memref<1024x1024xf32, 1>, vector<4xf32>
    affine.vector_store %7, %tempB[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
    // 将tempA里的数字 转置后 存入 smA  ok
    affine.for %arg3 = 0 to 4 {
      %11 = affine.vector_load %tempA[%c0 * 4 + %arg3] : memref<4xf32, 5>, vector<1xf32>
      affine.vector_store %11, %smA[0, (%tx mod 4) * 4 + %arg3, %ty * 4 + %tx floordiv 4 + %c0 * 64] : memref<2x16x64xf32, 3>, vector<1xf32>
    } {affine.loop = "unroll"}
    // tempB 原貌存入 smB ok
    %8 = affine.vector_load %tempB[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
    affine.vector_store %8, %smB[0, %ty + %tx floordiv 16 + %c0 * 16, (%tx mod 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
    gpu.barrier  // global->sm copy ok
    // ((%ty * 16 + %tx) mod 64)= 一维的tid
    // (%tid mod 64) = tidInWarp ,warp内的tid
    // %tid floordiv 64 = warpId
    // %tidInWarp floordiv 16 = thread在warp内的第几行， threadLineInWarp
    // (%threadLineInWarp + %warpId * 4) = thread 在 block内的第几行
    // %9 = affine.vector_load %smA[(%c0 floordiv 16) mod 2, %c0, (%threadLineInWarp + %warpId * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
    %9 = affine.vector_load %smA[(%c0 floordiv 16) mod 2, %c0, (((%ty * 16 + %tx) mod 64) floordiv 16 + ((%ty * 16 + %tx) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>  // ok
    // shmA->regA, ok
    affine.vector_store %9, %regA[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
    // smB->regB ,ok
    %10 = affine.vector_load %smB[(%c0 floordiv 16) mod 2, %c0, (%tx mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
    affine.vector_store %10, %regB[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
    // k axis iter
    affine.for %outK = 0 to 1024 step 16 {
        // 没到最后一个循环
      affine.if affine_set<(d0) : (-d0 + 992 >= 0)>(%outK) {
        // read next (BK*BM)  A -> temp
        %13 = affine.vector_load %A[%ty * 4 + %tx floordiv 4 + %c0 * 64 + %by * 64, (%tx mod 4) * 4 + %outK + 16] : memref<1024x1024xf32, 1>, vector<4xf32>
        affine.vector_store %13, %tempA[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
        // read next (BK*BN) B->temp
        %14 = affine.vector_load %B[%ty + %tx floordiv 16 + %c0 * 16 + %outK + 16, (%tx mod 16) * 4 + %bx * 64] : memref<1024x1024xf32, 1>, vector<4xf32>
        affine.vector_store %14, %tempB[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
      }
      affine.for %innerK = 0 to 15 {
        // smA->next regA
        %13 = affine.vector_load %smA[(%outK floordiv 16) mod 2, %innerK + 1, (((%ty * 16 + %tx) mod 64) floordiv 16 + ((%ty * 16 + %tx) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        affine.vector_store %13, %regA[(%innerK + 1) mod 2, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        %14 = affine.vector_load %smB[(%outK floordiv 16) mod 2, %innerK + 1, (%tx mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        affine.vector_store %14, %regB[(%innerK + 1) mod 2, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        // calc regC
        affine.for %arg5 = 0 to 4 {
          %15 = affine.load %regA[%innerK mod 2, %arg5] : memref<2x4xf32, 5>
          affine.for %arg6 = 0 to 4 {
            %16 = affine.load %regC[%arg5, %arg6] : memref<4x4xf32, 5>
            %17 = affine.load %regB[%innerK mod 2, %arg6] : memref<2x4xf32, 5>
            %18 = arith.mulf %15, %17 : f32
            %19 = arith.addf %18, %16 : f32
            affine.store %19, %regC[%arg5, %arg6] : memref<4x4xf32, 5>
          } {affine.loop = "unroll"}
        } {affine.loop = "unroll"}
      } {affine.loop = "unroll"}
      // 没到最后
      affine.if affine_set<(d0) : (-d0 + 992 >= 0)>(%outK) {
        affine.for %arg4 = 0 to 4 {
          %14 = affine.vector_load %tempA[%c0 * 4 + %arg4] : memref<4xf32, 5>, vector<1xf32>
          affine.vector_store %14, %smA[(%outK floordiv 16 + 1) mod 2, (%tx mod 4) * 4 + %arg4, %ty * 4 + %tx floordiv 4 + %c0 * 64] : memref<2x16x64xf32, 3>, vector<1xf32>
        } {affine.loop = "unroll"}
        %13 = affine.vector_load %tempB[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
        affine.vector_store %13, %smB[(%outK floordiv 16 + 1) mod 2, %ty + %tx floordiv 16 + %c0 * 16, (%tx mod 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        gpu.barrier  // temp->sm copy OK
      }
      affine.for %arg4 = 0 to 4 {
        %13 = affine.load %regA[%c15 mod 2, %arg4] : memref<2x4xf32, 5>
        affine.for %arg5 = 0 to 4 {
          %14 = affine.load %regC[%arg4, %arg5] : memref<4x4xf32, 5>
          %15 = affine.load %regB[%c15 mod 2, %arg5] : memref<2x4xf32, 5>
          %16 = arith.mulf %13, %15 : f32
          %17 = arith.addf %16, %14 : f32
          affine.store %17, %regC[%arg4, %arg5] : memref<4x4xf32, 5>
        } {affine.loop = "unroll"}
      } {affine.loop = "unroll"}
      %11 = affine.vector_load %smB[(%outK floordiv 16 + 1) mod 2, %c0, (%tx mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
      affine.vector_store %11, %regB[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
      %12 = affine.vector_load %smA[(%outK floordiv 16 + 1) mod 2, %c0, (((%ty * 16 + %tx) mod 64) floordiv 16 + ((%ty * 16 + %tx) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
      affine.vector_store %12, %regA[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
    }
    // write C
    affine.for %arg3 = 0 to 4 {
      %11 = affine.vector_load %regC[%c0 + %arg3, %c0 + %c0] : memref<4x4xf32, 5>, vector<4xf32>
      // 这里两个维度搞反了
    //   affine.vector_store %11, %C[%bx * 64 + %threadLineinBlock + %arg3, %by * 64 + %ty] : memref<1024x1024xf32, 1>, vector<4xf32>
      affine.vector_store %11, %C[%bx * 64 + (((%tx * 16 + %ty) mod 64) floordiv 16 + ((%tx * 16 + %ty) floordiv 64 + (%c0 floordiv 4) * 4) * 4) * 4 + %arg3, %by * 64 + (%ty mod 16 + (%c0 floordiv 4) * 16) * 4 + %c0] : memref<1024x1024xf32, 1>, vector<4xf32>
    } {affine.loop = "unroll"}
    return
  }
}



