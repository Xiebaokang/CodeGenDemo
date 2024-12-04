//  === start mlir =====
module {
  func.func public @GEMM_mnk1024x1024x512_f32f32f32_TTmn4x4_BTmnk64x64x16(%gA: memref<512x1024xf32, 1>, %gB: memref<512x1024xf32, 1>, %gC: memref<1024x1024xf32, 1>) attributes {func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
    affine.parallel (%BX, %BY) = (0, 0) to (16, 16) {  // bx,by
      %SHMA = memref.alloc() : memref<2x16x64xf32, 3>  // shm
      %SHMB = memref.alloc() : memref<2x16x64xf32, 3>
      %0 = affine.apply affine_map<(d0) -> (d0 * 64)>(%BX)  // (bx*64,by*64)
      %1 = affine.apply affine_map<(d0) -> (d0 * 64)>(%BY)
      affine.parallel (%TX, %TY) = (0, 0) to (16, 16) {  // tx,ty
        %c0 = arith.constant 0 : index
        %c0_1 = arith.constant 0 : index
        %c15 = arith.constant 15 : index
        %c0_2 = arith.constant 0 : index
        %c0_3 = arith.constant 0 : index
        %c0_4 = arith.constant 0 : index
        %c0_5 = arith.constant 0 : index
        %TMPA = memref.alloca() {alignment = 16 : i64} : memref<4xf32, 5>  // temp A (global->shm的中转站)
        %TMPB = memref.alloca() {alignment = 16 : i64} : memref<4xf32, 5>  // temp B
        %regA = memref.alloca() {alignment = 16 : i64} : memref<2x4xf32, 5>  // reg doublebuf
        %regB = memref.alloca() {alignment = 16 : i64} : memref<2x4xf32, 5>  // reg doublebuf
        %regC = memref.alloca() {alignment = 16 : i64} : memref<4x4xf32, 5>  // reg C
        %2 = affine.apply affine_map<(d0) -> (d0 * 4)>(%TX)
        %3 = affine.apply affine_map<(d0) -> (d0 * 4)>(%TY)
        affine.for %ii = 0 to 4 {
          affine.for %jj = 0 to 4 {
            %cst = arith.constant 0.000000e+00 : f32
            affine.store %cst, %regC[%ii, %jj] : memref<4x4xf32, 5>
          } {affine.loop = "unroll"}
        } {affine.loop = "unroll"}
        // A[tx * 4 + ty floordiv 4 + 0 * 64 + bx * 64, (ty mod 4) * 4 + 0]
        // BK=16,16/4个float=4,所以一行需要4次vecload即可。所以用 %TY floordiv 4, thread按顺序依次搬运 gA
        // 当a 采用transpose存储后，需要变下索引计算(shmA形状变为 BK行BM列。vecload ：[%TX + %TY floordiv(BM/4),(%TY mod (BM/4)) * 4]  )
        // %4 = affine.vector_load %gA[ERROR: %TX * 4 + %TY floordiv 4 + %c0 * 64 + %BX * 64, (%TY mod 4) * 4 + %c0_4] : memref<512x1024xf32, 1>, vector<4xf32>  // load A
        %4 = affine.vector_load %gA[%BX * 64 + %TX + %TY floordiv 16,(%TY mod 16) * 4] : memref<512x1024xf32, 1>, vector<4xf32>  // load A
        affine.vector_store %4, %TMPA[0] : memref<4xf32, 5>, vector<4xf32>  // A->tmpA
        // B[tx + ty floordiv 16 + 0 * 16 + 0, (ty mod 16) * 4 + by * 64]
        // BN=64,64/4=16次。访问BlockB一行需要16ci vecload，所以 %TY floordiv 16， thread按顺序依次搬运 gB
        %5 = affine.vector_load %gB[%TX + %TY floordiv 16 + %c0 * 16 + %c0_5, (%TY mod 16) * 4 + %BY * 64] : memref<512x1024xf32, 1>, vector<4xf32>  // load B
        affine.vector_store %5, %TMPB[0] : memref<4xf32, 5>, vector<4xf32>  // B->tmpB
        // modify index
        // affine.for %iBK = 0 to 4 {
        //   %9 = affine.vector_load %TMPA[%c0 * 4 + %iBK] : memref<4xf32, 5>, vector<1xf32>  // read tmpa
        //   affine.vector_store %9, %SHMA[0, (%TY mod 4) * 4 + %iBK, %TX * 4 + %TY floordiv 4 + %c0 * 64] : memref<2x16x64xf32, 3>, vector<1xf32>  // temA-> shmA[0, (ty mod 4) * 4 + i, %tx * 4 + ty floordiv 4 + 0 * 64]
        // } {affine.loop = "unroll"}
        %9 = affine.vector_load %TMPA[0] : memref<4xf32, 5>, vector<4xf32>  // read tmpa
        affine.vector_store %9, %SHMA[0, %TX + %TY floordiv 16 ,(%TY mod 16) * 4 ] : memref<2x16x64xf32, 3>, vector<4xf32>  // temA-> shmA[0, (ty mod 4) * 4 + i, %tx * 4 + ty floordiv 4 + 0 * 64]

        %6 = affine.vector_load %TMPB[0] : memref<4xf32, 5>, vector<4xf32>
        affine.vector_store %6, %SHMB[0, %TX + %TY floordiv 16 + %c0 * 16, (%TY mod 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>  // store shmB
        gpu.barrier  // global->shm copy ok
        // shm[0]->reg[0]
        %7 = affine.vector_load %SHMA[(%c0_1 floordiv 16) mod 2, %c0_2, (((%TX * 16 + %TY) mod 64) floordiv 16 + ((%TX * 16 + %TY) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        affine.vector_store %7, %regA[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        %8 = affine.vector_load %SHMB[(%c0 floordiv 16) mod 2, %c0_3, (%TY mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
        affine.vector_store %8, %regB[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        // k axis 归约
        affine.for %iBK = 0 to 512 step 16 {
          affine.if affine_set<(d0) : (-d0 + 480 >= 0)>(%iBK) {
            // A->tempA, B->tempB
            %11 = affine.vector_load %gA[%TX + %TY floordiv 16 + %BX * 64+ %iBK + 16, (%TY mod 16) * 4 ] : memref<512x1024xf32, 1>, vector<4xf32>
            affine.vector_store %11, %TMPA[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
            %12 = affine.vector_load %gB[%TX + %TY floordiv 16 + %iBK + 16, (%TY mod 16) * 4 + %BY * 64] : memref<512x1024xf32, 1>, vector<4xf32>
            affine.vector_store %12, %TMPB[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
          }
          affine.for %i_k = 0 to 15 {
            // shm[k+1] -> reg[k+1]
            %11 = affine.vector_load %SHMA[(%iBK floordiv 16) mod 2, %i_k + 1, (((%TX * 16 + %TY) mod 64) floordiv 16 + ((%TX * 16 + %TY) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
            affine.vector_store %11, %regA[(%i_k + 1) mod 2, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
            %12 = affine.vector_load %SHMB[(%iBK floordiv 16) mod 2, %i_k + 1, (%TY mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
            affine.vector_store %12, %regB[(%i_k + 1) mod 2, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
            affine.for %arg9 = 0 to 4 {
              affine.for %arg10 = 0 to 4 {
                // regC[i,j] += regA[k,i] * regB[k,j]
                %13 = affine.load %regC[%arg9, %arg10] : memref<4x4xf32, 5>
                %14 = affine.load %regA[%i_k mod 2, %arg9] : memref<2x4xf32, 5>
                %15 = affine.load %regB[%i_k mod 2, %arg10] : memref<2x4xf32, 5>
                %16 = arith.mulf %14, %15 : f32
                %17 = arith.addf %16, %13 : f32
                affine.store %17, %regC[%arg9, %arg10] : memref<4x4xf32, 5>
              } {affine.loop = "unroll"}
            } {affine.loop = "unroll"}
          } {affine.loop = "unroll"}
          affine.if affine_set<(d0) : (-d0 + 480 >= 0)>(%iBK) {
            // affine.for %arg8 = 0 to 4 {
            //   %12 = affine.vector_load %TMPA[%c0 * 4 + %arg8] : memref<4xf32, 5>, vector<1xf32>
            //   affine.vector_store %12, %SHMA[(%iBK floordiv 16 + 1) mod 2, (%TY mod 4) * 4 + %arg8, %TX * 4 + %TY floordiv 4 + %c0 * 64] : memref<2x16x64xf32, 3>, vector<1xf32>
            // } {affine.loop = "unroll"}
            // global->shm
            %12 = affine.vector_load %TMPA[%c0 * 4 ] : memref<4xf32, 5>, vector<4xf32>
            affine.vector_store %12, %SHMA[(%iBK floordiv 16 + 1) mod 2, %TX + %TY floordiv 16 + %c0 * 64,(%TY mod 16) * 4  ] : memref<2x16x64xf32, 3>, vector<4xf32>
            
            %11 = affine.vector_load %TMPB[%c0 * 4] : memref<4xf32, 5>, vector<4xf32>
            affine.vector_store %11, %SHMB[(%iBK floordiv 16 + 1) mod 2, %TX + %TY floordiv 16 + %c0 * 16, (%TY mod 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
            gpu.barrier
          }
          // regC += regA * regB
          affine.for %arg88 = 0 to 4 {
            affine.for %arg9 = 0 to 4 {
              %11 = affine.load %regC[%arg88, %arg9] : memref<4x4xf32, 5>
              %12 = affine.load %regA[%c15 mod 2, %arg88] : memref<2x4xf32, 5>
              %13 = affine.load %regB[%c15 mod 2, %arg9] : memref<2x4xf32, 5>
              %14 = arith.mulf %12, %13 : f32
              %15 = arith.addf %14, %11 : f32
              affine.store %15, %regC[%arg88, %arg9] : memref<4x4xf32, 5>
            } {affine.loop = "unroll"}
          } {affine.loop = "unroll"}
          // shm->reg
          %99 = affine.vector_load %SHMB[(%iBK floordiv 16 + 1) mod 2, %c0_3, (%TY mod 16 + %c0 * 16) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
          affine.vector_store %99, %regB[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
          %10 = affine.vector_load %SHMA[(%iBK floordiv 16 + 1) mod 2, %c0_2, (((%TX * 16 + %TY) mod 64) floordiv 16 + ((%TX * 16 + %TY) floordiv 64 + %c0 * 4) * 4) * 4] : memref<2x16x64xf32, 3>, vector<4xf32>
          affine.vector_store %10, %regA[0, %c0 * 4] : memref<2x4xf32, 5>, vector<4xf32>
        }
        // regC -> globalC
        affine.for %iBK = 0 to 4 {
          %99 = affine.vector_load %regC[%c0 + %iBK, %c0 + %c0] : memref<4x4xf32, 5>, vector<4xf32>
          affine.vector_store %99, %gC[%BX * 64 + (((%TX * 16 + %TY) mod 64) floordiv 16 + ((%TX * 16 + %TY) floordiv 64 + (%c0 floordiv 4) * 4) * 4) * 4 + %iBK, %BY * 64 + (%TY mod 16 + (%c0 floordiv 4) * 16) * 4 + %c0] : memref<1024x1024xf32, 1>, vector<4xf32>
        } {affine.loop = "unroll"}
      }
    }
    return
  }
}