#include "Analysis/Analyzer.h"


namespace KernelCodeGen {

std::vector<mlir::func::FuncOp> Analyzer::collectFunctions(mlir::ModuleOp& module, const std::string& targetFuncName) {
  std::vector<mlir::func::FuncOp> result;
  module.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp funcOp) {
    auto state = funcOp->getAttr(std::string("func.state"));
    auto stateAttr = state.dyn_cast<mlir::StringAttr>();

    if (stateAttr.getValue().str() == "cpu") {
      auto opName = funcOp->getAttr(std::string("func.op.name"));
      auto opNameAttr = opName.dyn_cast<mlir::StringAttr>();

      if (opNameAttr.getValue().str() == targetFuncName) 
        result.push_back(funcOp);
    }   
  });
  return std::move(result);
}

std::vector<int64_t> Analyzer::getParallelNumber(mlir::affine::AffineParallelOp parallelLevel, int64_t& totalNumber) {
  auto dim = parallelLevel.getNumDims();
  totalNumber = 1;
  std::vector<int64_t> result;
  for (int i = 0; i < dim; i++) {
    auto map = parallelLevel.getUpperBoundMap(i);
    auto exprs = map.getResults();
    assert(exprs.size() == 1);
    auto constExpr = exprs[0].dyn_cast<mlir::AffineConstantExpr>();
    assert(constExpr);
    totalNumber *= constExpr.getValue();
    result.push_back(constExpr.getValue());
  }
  return result;
}

std::vector<mlir::affine::AffineForOp> Analyzer::collectFuncLoops(mlir::func::FuncOp funcOp) {
  std::vector<mlir::affine::AffineForOp> res;
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
    res.push_back(forOp);
  });
  return std::move(res);
}

std::set<std::string> Analyzer::collectFuncNames(mlir::ModuleOp& module) {
  std::set<std::string> result;
  module.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp funcOp) {
    auto state = funcOp->getAttr(std::string("func.state"));
    auto stateAttr = state.dyn_cast<mlir::StringAttr>();

    if (stateAttr.getValue().str() == "cpu") {
      auto opName = funcOp->getAttr(std::string("func.op.name"));
      auto opNameAttr = opName.dyn_cast<mlir::StringAttr>();
      result.insert(opNameAttr.getValue().str());
    } 
  });
  return result;
}


}