#! /bin/bash
#  // ./mlir-translate /home/xie/CodeGenDemo/build/llvm-dialect.mlir -mlir-to-llvmir > /home/xie/CodeGenDemo/build/llvm-ir.ll
mlirInput=$1
outputLLIR=$2
~/llvm-install/bin/mlir-translate $mlirInput -mlir-to-llvmir > output.ll
