#ifndef __Operators_h__
#define __Operators_h__

#include "utils.h"

namespace KernelCodeGen {

mlir::func::FuncOp buildFunction(mlir::ModuleOp module, const std::string& funcName, const std::string& OpName, 
                                 const std::vector<mlir::Type>& inputsTypes);
 
mlir::Type getDType(mlir::OpBuilder& builder, const std::string& dtype);

template <typename T>
struct Operator {
  template <typename... Args>
  static void build(mlir::ModuleOp module, Args &&...args) {
    T::build(module, std::forward<Args>(args)...);
  }

  static std::string getKernelName(){
    return T::getKernelName();
  }
};

struct Matmul : Operator<Matmul> {
  static void build(mlir::ModuleOp module, std::vector<int64_t> shape, const std::vector<std::string>& dtypes);

  static std::optional<std::string> verify(mlir::OpBuilder builder, std::vector<int64_t> shape, const std::vector<std::string>& dtype);
  
  static mlir::func::FuncOp createFunc(mlir::ModuleOp module, std::vector<int64_t> shape, const std::vector<mlir::Type>& dtype);

  static std::string s_function;

  static std::string getKernelName(){
    return s_function;
  }
};
}


#endif