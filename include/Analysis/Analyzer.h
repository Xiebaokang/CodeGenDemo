#ifndef _Analyzer_h_
#define _Analyzer_h_

#include "Common/Utils.h"
#include <vector>

namespace KernelCodeGen {

struct CompareLoop {
  int operator()(const mlir::affine::AffineForOp& x, const mlir::affine::AffineForOp& y) const {
    mlir::Operation* x_ptr = x;
    mlir::Operation* y_ptr = y;
    auto x_hashCode = reinterpret_cast<size_t>(x_ptr);
    auto y_hashCode = reinterpret_cast<size_t>(y_ptr);
    if (x_hashCode >= y_hashCode) return 0;
    else return 1;
  }
};

struct CompareFunc {
  int operator()(const mlir::func::FuncOp& x, const mlir::func::FuncOp& y) const {
    mlir::Operation* x_ptr = x;
    mlir::Operation* y_ptr = y;
    auto x_hashCode = reinterpret_cast<size_t>(x_ptr);
    auto y_hashCode = reinterpret_cast<size_t>(y_ptr);
    if (x_hashCode >= y_hashCode) return 0;
    else return 1;
  }
};


struct Analyzer {
  Analyzer() = default;

  static std::vector<int64_t> getParallelNumber(mlir::affine::AffineParallelOp parallelLevel, int64_t& totalNumber);
  static std::vector<mlir::func::FuncOp> collectFunctions(mlir::ModuleOp& module, const std::string& targetFuncName = {""});
  static std::vector<mlir::affine::AffineForOp> collectFuncLoops(mlir::func::FuncOp funcOp);
  static std::set<std::string> collectFuncNames(mlir::ModuleOp& module);
  
  static int getThreadsPerCTA(mlir::ModuleOp module); 
};

}  // KernelCodeGen

#endif  // _Analyzer_h_