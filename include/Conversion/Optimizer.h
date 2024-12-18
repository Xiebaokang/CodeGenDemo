#pragma once

#include "Analysis/Analyzer.h"
#include "Conversion/Rewriter.h"
// #include "Frontend/Operators.h"

#include "utils.h"

#include <unordered_map>

namespace KernelCodeGen {

struct Optimizer {
  virtual bool applicable(mlir::ModuleOp& module) = 0;
  virtual void applyOptimzer(mlir::ModuleOp& module, std::map<std::string, int> config) = 0;

  bool operator==(const Optimizer& other) {
    return name == other.name;
  }
  std::string name;
};


struct MatmulOptimizer : Optimizer {

  MatmulOptimizer() {
    this->name = std::move(std::string("Matmul"));
  }

  virtual bool applicable(mlir::ModuleOp& module) override;
  virtual void applyOptimzer(mlir::ModuleOp& module, std::map<std::string, int> config) override;

  mlir::AffineMap getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder, std::map<std::string, int> config);

  void clear() {
    matmuls.clear();
    matmulLoops.clear();
    matmulBuffers.clear();
  }

  // using the outermost loop represent a matmul.
  std::set<mlir::func::FuncOp, CompareFunc> matmuls;

  // Map: from outermost loop to all loops in the matmul(loopM->[loopM, loopN, loopK]).
  std::map<mlir::func::FuncOp, std::vector<mlir::affine::AffineForOp>, CompareFunc> matmulLoops;


  // Memory: A, B, C
  struct MemoryBuffer {
    mlir::Value A;
    mlir::Value B;
    mlir::Value C;
  };

  // loopM->[A, B, C]
  std::map<mlir::func::FuncOp, MemoryBuffer, CompareFunc> matmulBuffers;
  
  // private method
  void _opSetDescription(mlir::Operation* op, const std::string& attrValue);
};

}