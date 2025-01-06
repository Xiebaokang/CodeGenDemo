# 工具func 从 Rwriter 迁移到 Analyzer 中

```c++
mlir::OpBuilder getBuilder(mlir::affine::AffineForOp op, Position pos)
// 这个函数是将builder的移动到某个op的什么位置
// 可将affineForOp修改成Oparetion变得更加通用
// 未修改

std::vector<mlir::Value> getParallelIdx(mlir::affine::AffineParallelOp parallelLevel)
// 获取AffineParallelOp的循环变量
// 无
// 无

mlir::AffineExpr shiftAffineExprDim(mlir::MLIRContext* context, mlir::AffineExpr expr, int shift)
// 移动 expr 中的dimexpr的位置：(d0 + 3 * d1 - d2) => (d1 + 3 * d2 - d3)
// 无
// 无

mlir::AffineExpr getModifiedExpr(mlir::MLIRContext* context, mlir::AffineExpr inExpr, mlir::AffineExpr replaceExpr, int targetDim, int replaceNumberDims)
// 
// 无
// 无

mlir::affine::AffineForOp findRootLoop(mlir::Operation* op)
// 向上获取最上层的 loop op，若父op为 moduleOp，funcOp或者parallelOp则停止寻找，直接返回这个loop
// 无
// 无

mlir::Block* getClostScopeOp(mlir::Operation* op)
// 向上找到一个含有body的父op，并将其block进行返回
// 无
// 无



```

# Rewriter 中的函数

```c++
mlir::Value bufferizeLoopCarryVar(std::vector<mlir::affine::AffineForOp> &loops);
// 根据给定的loops，找到其中含有迭代变量的loop，以及判断其他loop是否包含了含迭代变量的loop；将使用其他loop的循环参数构建buffer，替换掉含有迭代变量的loop
// 将参数列表改成两个，第一个为含迭代变量的loop，第二个为包含这个loop的其他loops，并且buffer使用一维表示
// 已修改


```


```c++

==== original mlir =====
module {
  func.func public @GEMM_testKernel(%arg0: memref<1024x1024xf16, 1>, %arg1: memref<1024x1024xf16, 1>, %arg2: memref<1024x1024xf16, 1>) attributes {func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
    affine.for %arg3 = 0 to 1024 {
      affine.for %arg4 = 0 to 1024 {
        %cst = arith.constant 0.000000e+00 : f16
        %0 = affine.for %arg5 = 0 to 1024 iter_args(%arg6 = %cst) -> (f16) {
          %1 = affine.load %arg0[%arg5, %arg3] : memref<1024x1024xf16, 1>
          %2 = affine.load %arg1[%arg5, %arg4] : memref<1024x1024xf16, 1>
          %3 = arith.mulf %1, %2 : f16
          %4 = arith.addf %3, %arg6 : f16
          affine.yield %4 : f16
        }
        affine.store %0, %arg2[%arg3, %arg4] : memref<1024x1024xf16, 1>
      }
    }
    return
  }
}
===== after split & reorder =======
module {
  func.func public @GEMM_testKernel(%arg0: memref<1024x1024xf16, 1>, %arg1: memref<1024x1024xf16, 1>, %arg2: memref<1024x1024xf16, 1>) attributes {func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
    affine.for %arg3 = 0 to 1024 step 64 {
      affine.for %arg4 = 0 to 1024 step 48 {
        affine.for %arg5 = 0 to 64 step 4 {
          affine.for %arg6 = 0 to 48 step 6 {
            affine.for %arg7 = 0 to 4 {
              affine.for %arg8 = 0 to 6 {
                %cst = arith.constant 0.000000e+00 : f16
                %0 = affine.for %arg9 = 0 to 1024 iter_args(%arg10 = %cst) -> (f16) {
                  %1 = affine.load %arg0[%arg9, %arg3 + %arg5 + %arg7] : memref<1024x1024xf16, 1>
                  %2 = affine.load %arg1[%arg9, %arg4 + %arg6 + %arg8] : memref<1024x1024xf16, 1>
                  %3 = arith.mulf %1, %2 : f16
                  %4 = arith.addf %3, %arg10 : f16
                  affine.yield %4 : f16
                }
                affine.store %0, %arg2[%arg3 + %arg5 + %arg7, %arg4 + %arg6 + %arg8] : memref<1024x1024xf16, 1>
              }
            }
          }
        }
      }
    }
    return
  }
}
===== after parallel =======
module {
  func.func public @GEMM_testKernel(%arg0: memref<1024x1024xf16, 1>, %arg1: memref<1024x1024xf16, 1>, %arg2: memref<1024x1024xf16, 1>) attributes {func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
    affine.parallel (%arg3, %arg4) = (0, 0) to (16, 22) {
      %0 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)
      %1 = affine.apply affine_map<(d0) -> (d0 * 48)>(%arg4)
      affine.parallel (%arg5, %arg6) = (0, 0) to (16, 8) {
        %2 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg5)
        %3 = affine.apply affine_map<(d0) -> (d0 * 6)>(%arg6)
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 6 {
            %cst = arith.constant 0.000000e+00 : f16
            %4 = affine.for %arg9 = 0 to 1024 iter_args(%arg10 = %cst) -> (f16) {
              %5 = affine.load %arg0[%arg9, %0 + %2 + %arg7] : memref<1024x1024xf16, 1>
              %6 = affine.load %arg1[%arg9, %1 + %3 + %arg8] : memref<1024x1024xf16, 1>
              %7 = arith.mulf %5, %6 : f16
              %8 = arith.addf %7, %arg10 : f16
              affine.yield %8 : f16
            }
            affine.store %4, %arg2[%0 + %2 + %arg7, %1 + %3 + %arg8] : memref<1024x1024xf16, 1>
          }
        }
      }
    }
    return
  }
}
===== after bufferizeLoopCarryVar =======
module {
  func.func public @GEMM_testKernel(%arg0: memref<1024x1024xf16, 1>, %arg1: memref<1024x1024xf16, 1>, %arg2: memref<1024x1024xf16, 1>) attributes {func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32} {
    affine.parallel (%arg3, %arg4) = (0, 0) to (16, 22) {
      %0 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)
      %1 = affine.apply affine_map<(d0) -> (d0 * 48)>(%arg4)
      affine.parallel (%arg5, %arg6) = (0, 0) to (16, 8) {
        %2 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg5)
        %3 = affine.apply affine_map<(d0) -> (d0 * 6)>(%arg6)
        %alloca = memref.alloca() {alignment = 16 : i64} : memref<4x6xf16, 5>
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 6 {
            %cst = arith.constant 0.000000e+00 : f16
            affine.store %cst, %alloca[%arg7, %arg8] : memref<4x6xf16, 5>
            affine.for %arg9 = 0 to 1024 {
              %5 = affine.load %alloca[%arg7, %arg8] : memref<4x6xf16, 5>
              %6 = affine.load %arg0[%arg9, %0 + %2 + %arg7] : memref<1024x1024xf16, 1>
              %7 = affine.load %arg1[%arg9, %1 + %3 + %arg8] : memref<1024x1024xf16, 1>
              %8 = arith.mulf %6, %7 : f16
              %9 = arith.addf %8, %5 : f16
              affine.store %9, %alloca[%arg7, %arg8] : memref<4x6xf16, 5>
            }
            %4 = affine.load %alloca[%arg7, %arg8] : memref<4x6xf16, 5>
            affine.store %4, %arg2[%0 + %2 + %arg7, %1 + %3 + %arg8] : memref<1024x1024xf16, 1>
          }
        }
      }
    }
    return
  }
}
==== optimize status: SUCCESS
```