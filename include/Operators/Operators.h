#pragma once

#include "utils.h"

namespace KernelCodeGen {

mlir::func::FuncOp buildFunction(mlir::ModuleOp module, const std::string& funcName, const std::string& OpName, 
                                 const std::vector<mlir::Type>& inputsTypes, mlir::Type dtype);
 
mlir::Type getDType(mlir::OpBuilder& builder, const std::string& dtype);

template <typename T>
struct Operator {
  template <typename... Args>
  static void build(mlir::ModuleOp module, Args &&...args) {
    T::build(module, std::forward<Args>(args)...);
  }
};

struct Matmul : Operator<Matmul> {
  static void build(mlir::ModuleOp module, std::vector<int64_t> shape, const std::string& dtype={"float32"});

  static std::pair<bool, std::string> verify(mlir::OpBuilder builder, std::vector<int64_t> shape, const std::string& dtype);
  
  static mlir::func::FuncOp createFunc(mlir::ModuleOp module, std::vector<int64_t> shape, mlir::Type dtype);
};
}