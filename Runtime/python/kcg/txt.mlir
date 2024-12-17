  memref.global "public" @kcg_shm1 : memref<2x8x32xf32, 3> {alignment = 16 : i64}

  memref.global "public" @kcg_shm0 : memref<2x8x256xf32, 3> {alignment = 16 : i64}

  func.func public @GEMM_mnk1024x1024x1024_f32f32f32_TTmn8x4_BTmnk256x32x8_BLmn2x2_WLmn16x4(%A: memref<1024x1024xf32, 1>, %B: memref<1024x1024xf32, 1>, %C: memref<1024x1024xf32, 1>) attributes {func.block.dim = array<i32: 32, 8>, func.grid.dim = array<i32: 4, 32>, func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {

    %smA = memref.get_global @kcg_shm0 : memref<2x8x256xf32, 3>

    %smB = memref.get_global @kcg_shm1 : memref<2x8x32xf32, 3>

    %tempA = memref.alloca() {alignment = 16 : i64} : memref<8xf32, 5>

    %tempB = memref.alloca() {alignment = 16 : i64} : memref<1xf32, 5>

    %regA = memref.alloca() {alignment = 16 : i64} : memref<2x8xf32, 5>

    %regB = memref.alloca() {alignment = 16 : i64} : memref<2x4xf32, 5>

    %regC = memref.alloca() {alignment = 16 : i64} : memref<8x4xf32, 5>

        affine.store %cst, %regC[%i08, %i04] : memref<8x4xf32, 5>

    %6 = affine.vector_load %A[%ty * 4 + %tx floordiv 2 + %c0 * 128 + %by * 256, (%tx mod 2) * 4 + %c0] : memref<1024x1024xf32, 1>, vector<4xf32>

    affine.vector_store %6, %tempA[%c0 * 4] : memref<8xf32, 5>, vector<4xf32>

    %7 = affine.vector_load %A[%ty * 4 + %tx floordiv 2 + %c1 * 128 + %by * 256, (%tx mod 2) * 4 + %c0] : memref<1024x1024xf32, 1>, vector<4xf32>

    affine.vector_store %7, %tempA[%c1 * 4] : memref<8xf32, 5>, vector<4xf32>

    %8 = affine.vector_load %B[(%ty * 8 + %tx) floordiv 32 + %c0 * 8 + %c0, (%ty * 8 + %tx) mod 32 + %bx * 32] : memref<1024x1024xf32, 1>, vector<1xf32>

    affine.vector_store %8, %tempB[%c0] : memref<1xf32, 5>, vector<1xf32>

      %13 = affine.vector_load %tempA[%c0 * 4 + %i04] : memref<8xf32, 5>, vector<1xf32>

      affine.vector_store %13, %smA[0, (%tx mod 2) * 4 + %i04, %ty * 4 + %tx floordiv 2 + %c0 * 128] : memref<2x8x256xf32, 3>, vector<1xf32>

      %13 = affine.vector_load %tempA[%c1 * 4 + %i04] : memref<8xf32, 5>, vector<1xf32>

      affine.vector_store %13, %smA[0, (%tx mod 2) * 4 + %i04, %ty * 4 + %tx floordiv 2 + %c1 * 128] : memref<2x8x256xf32, 3>, vector<1xf32>

    %9 = affine.vector_load %tempB[%c0] : memref<1xf32, 5>, vector<1xf32>

    affine.vector_store %9, %smB[0, (%ty * 8 + %tx) floordiv 32 + %c0 * 8, (%ty * 8 + %tx) mod 32] : memref<2x8x32xf32, 3>, vector<1xf32>

    %10 = affine.vector_load %smA[(%c0 floordiv 8) mod 2, %c0, (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + %c0 * 2) * 16) * 4] : memref<2x8x256xf32, 3>, vector<4xf32>

    affine.vector_store %10, %regA[0, %c0 * 4] : memref<2x8xf32, 5>, vector<4xf32>

    %11 = affine.vector_load %smA[(%c0 floordiv 8) mod 2, %c0, (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + %c1 * 2) * 16) * 4] : memref<2x8x256xf32, 3>, vector<4xf32>

    affine.vector_store %11, %regA[0, %c1 * 4] : memref<2x8xf32, 5>, vector<4xf32>

    %12 = affine.vector_load %smB[(%c0 floordiv 8) mod 2, %c0, (%tx mod 4 + (((%ty * 8 + %tx) floordiv 64) mod 2 + %c0 * 2) * 4) * 4] : memref<2x8x32xf32, 3>, vector<4xf32>

    affine.vector_store %12, %regB[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>

        %16 = affine.vector_load %A[%ty * 4 + %tx floordiv 2 + %c0 * 128 + %by * 256, (%tx mod 2) * 4 + %kout_0_8_1024 + 8] : memref<1024x1024xf32, 1>, vector<4xf32>

        affine.vector_store %16, %tempA[%c0 * 4] : memref<8xf32, 5>, vector<4xf32>

        %17 = affine.vector_load %A[%ty * 4 + %tx floordiv 2 + %c1 * 128 + %by * 256, (%tx mod 2) * 4 + %kout_0_8_1024 + 8] : memref<1024x1024xf32, 1>, vector<4xf32>

        affine.vector_store %17, %tempA[%c1 * 4] : memref<8xf32, 5>, vector<4xf32>

        %18 = affine.vector_load %B[(%ty * 8 + %tx) floordiv 32 + %c0 * 8 + %kout_0_8_1024 + 8, (%ty * 8 + %tx) mod 32 + %bx * 32] : memref<1024x1024xf32, 1>, vector<1xf32>

        affine.vector_store %18, %tempB[%c0] : memref<1xf32, 5>, vector<1xf32>

        %16 = affine.vector_load %smA[(%kout_0_8_1024 floordiv 8) mod 2, %kin_07 + 1, (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + %c0 * 2) * 16) * 4] : memref<2x8x256xf32, 3>, vector<4xf32>

        affine.vector_store %16, %regA[(%kin_07 + 1) mod 2, %c0 * 4] : memref<2x8xf32, 5>, vector<4xf32>

        %17 = affine.vector_load %smA[(%kout_0_8_1024 floordiv 8) mod 2, %kin_07 + 1, (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + %c1 * 2) * 16) * 4] : memref<2x8x256xf32, 3>, vector<4xf32>

        affine.vector_store %17, %regA[(%kin_07 + 1) mod 2, %c1 * 4] : memref<2x8xf32, 5>, vector<4xf32>

        %18 = affine.vector_load %smB[(%kout_0_8_1024 floordiv 8) mod 2, %kin_07 + 1, (%tx mod 4 + (((%ty * 8 + %tx) floordiv 64) mod 2 + %c0 * 2) * 4) * 4] : memref<2x8x32xf32, 3>, vector<4xf32>

        affine.vector_store %18, %regB[(%kin_07 + 1) mod 2, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>

          %19 = affine.load %regA[%kin_07 mod 2, %i08] : memref<2x8xf32, 5>

            %20 = affine.load %regC[%i08, %i04] : memref<8x4xf32, 5>

            %21 = affine.load %regB[%kin_07 mod 2, %i04] : memref<2x4xf32, 5>

            affine.store %23, %regC[%i08, %i04] : memref<8x4xf32, 5>

          %17 = affine.vector_load %tempA[%c0 * 4 + %i04] : memref<8xf32, 5>, vector<1xf32>

          affine.vector_store %17, %smA[(%kout_0_8_1024 floordiv 8 + 1) mod 2, (%tx mod 2) * 4 + %i04, %ty * 4 + %tx floordiv 2 + %c0 * 128] : memref<2x8x256xf32, 3>, vector<1xf32>

          %17 = affine.vector_load %tempA[%c1 * 4 + %i04] : memref<8xf32, 5>, vector<1xf32>

          affine.vector_store %17, %smA[(%kout_0_8_1024 floordiv 8 + 1) mod 2, (%tx mod 2) * 4 + %i04, %ty * 4 + %tx floordiv 2 + %c1 * 128] : memref<2x8x256xf32, 3>, vector<1xf32>

        %16 = affine.vector_load %tempB[%c0] : memref<1xf32, 5>, vector<1xf32>

        affine.vector_store %16, %smB[(%kout_0_8_1024 floordiv 8 + 1) mod 2, (%ty * 8 + %tx) floordiv 32 + %c0 * 8, (%ty * 8 + %tx) mod 32] : memref<2x8x32xf32, 3>, vector<1xf32>

        %16 = affine.load %regA[%c7 mod 2, %i08] : memref<2x8xf32, 5>

          %17 = affine.load %regC[%i08, %arg5] : memref<8x4xf32, 5>

          %18 = affine.load %regB[%c7 mod 2, %arg5] : memref<2x4xf32, 5>

          affine.store %20, %regC[%i08, %arg5] : memref<8x4xf32, 5>

      %13 = affine.vector_load %smB[(%kout_0_8_1024 floordiv 8 + 1) mod 2, %c0, (%tx mod 4 + (((%ty * 8 + %tx) floordiv 64) mod 2 + %c0 * 2) * 4) * 4] : memref<2x8x32xf32, 3>, vector<4xf32>

      affine.vector_store %13, %regB[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>

      %14 = affine.vector_load %smA[(%kout_0_8_1024 floordiv 8 + 1) mod 2, %c0, (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + %c0 * 2) * 16) * 4] : memref<2x8x256xf32, 3>, vector<4xf32>

      affine.vector_store %14, %regA[0, %c0 * 4] : memref<2x8xf32, 5>, vector<4xf32>

      %15 = affine.vector_load %smA[(%kout_0_8_1024 floordiv 8 + 1) mod 2, %c0, (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + %c1 * 2) * 16) * 4] : memref<2x8x256xf32, 3>, vector<4xf32>

      affine.vector_store %15, %regA[0, %c1 * 4] : memref<2x8xf32, 5>, vector<4xf32>

      %13 = affine.vector_load %regC[%c0 + %i04, %c0 + %c0] : memref<8x4xf32, 5>, vector<4xf32>

      affine.vector_store %13, %C[%by * 256 + (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + (%c0 floordiv 4) * 2) * 16) * 4 + %i04, %bx * 32 + (%tx mod 4 + (((%ty * 8 + %tx) floordiv 64) mod 2 + (%c0 floordiv 4) * 2) * 4) * 4 + %c0] : memref<1024x1024xf32, 1>, vector<4xf32>

      %13 = affine.vector_load %regC[%c4 + %i04, %c0 + %c0] : memref<8x4xf32, 5>, vector<4xf32>

      affine.vector_store %13, %C[%by * 256 + (((%ty * 8 + %tx) mod 64) floordiv 4 + (((%ty * 8 + %tx) floordiv 64) floordiv 2 + (%c4 floordiv 4) * 2) * 16) * 4 + %i04, %bx * 32 + (%tx mod 4 + (((%ty * 8 + %tx) floordiv 64) mod 2 + (%c0 floordiv 4) * 2) * 4) * 4 + %c0] : memref<1024x1024xf32, 1>, vector<4xf32>

