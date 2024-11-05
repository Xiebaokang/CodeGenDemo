#pragma once

#include "utils.h"

namespace KernelCodeGen {

std::string randName(int length=20);

mlir::func::FuncOp buildFunction(mlir::ModuleOp module, 
                                  mlir::OpBuilder& builder, 
                                  const std::string& funcName, 
                                  const std::string& OpName, 
                                  const std::vector<mlir::Type>& inputsTypes);
 
mlir::Type getDType(mlir::OpBuilder& builder, const std::string& dtype);

template <typename T>
struct Operator {
  template <typename... Args>
  static void build(mlir::ModuleOp module, mlir::OpBuilder& builder, Args &&...args) {
    return T::build(module, builder, std::forward<Args>(args)...);
  }
};

struct Matmul : Operator<Matmul> {
  static void build(mlir::ModuleOp module, 
                    mlir::OpBuilder& builder, 
                    std::vector<int64_t> shape, 
                    const std::string& dtype={"float32"}, 
                    std::vector<MemorySpace> mss={MemorySpace::global, MemorySpace::global, MemorySpace::global});

  static std::pair<bool, std::string> verify(mlir::OpBuilder& builder, 
                                              std::vector<int64_t> shape, 
                                              const std::string& dtype={"float32"});
  
  static mlir::func::FuncOp createFunc(mlir::ModuleOp module, 
                                        mlir::OpBuilder& builder, 
                                        std::vector<int64_t> shape, 
                                        mlir::Type dtype,   
                                        std::vector<MemorySpace> mss);
};
}