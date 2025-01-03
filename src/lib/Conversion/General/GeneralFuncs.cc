#include "Conversion/General/GeneralFuncs.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Pass/PassManager.h"
#include <algorithm>
#include <map>
#include <cmath>

namespace KernelCodeGen {

mlir::OpBuilder getBuilder(mlir::Operation* op, Position pos) {
  // 按照位置和op创建builder
  switch (pos){
  case Position::after
  {
    mlir::OpBuilder builder(op->getContext());
    builder.setInsertionPointAfter(op);
    return builder;
  }
  case Position::before:
  {
    mlir::OpBuilder builder(op);
    return builder;
  }
  case Position::begin:
  {
    return mlir::OpBuilder::atBlockBegin(&op->getRegion(0).front());
  }
  case Position::end:
  {
    return mlir::OpBuilder::atBlockEnd(&op->getRegion(0).front());
  }
  default:
    assert(false);
  }
}

std::tuple<int64_t, int64_t, int64_t> getLoopBoundAndStep(mlir::affine::AffineForOp loop) {
  // 获取forOp的上界、下界以及步长
  int64_t ub = loop.getConstantUpperBound();
  int64_t lb = loop.getConstantLowerBound();
  int64_t step = loop.getStep().getLimitedValue();
  return std::make_tuple(lb, ub, step);
}

void replaceAndErase(mlir::Operation* newOp, mlir::Operation* oldOp) {
  // 替换后面op使用到oldOp的值，且删除oldOp
  auto oldResult = oldOp->getResult(0);
  oldResult.replaceAllUsesWith(newOp->getResult(0));
  oldOp->erase();
}

// void spliceHaveBlockOp(mlir::Operation* newOp, mlir::Operation* oldOp, int index, bool isBegin) {
//   // 将 oldOp 中的 ops 转到 newOp 中，index决定转移newOp的位置
//     auto& newOpOperations = newOp->getRegion(0).front().getOperations();
//     auto& oldOpOperations = oldOp->getRegion(0).front().getOperations();
//     llvm::iplist<mlir::Operation>::iterator it;
//     if (isBegin) {
//       it = newOpOperations.begin();
//     } else {
//       it = newOpOperations.end();
//     }
//     std::advance(it, index);
//     newOpOperations.splice(it, oldOpOperations);
// }

void spliceHaveBlockOp(mlir::Operation* newOp, mlir::Operation* oldOp, int index) {
  // 将 oldOp 中的 ops 转到 newOp 中，index决定转移newOp的位置
    auto& newOpOperations = newOp->getRegion(0).front().getOperations();
    auto& oldOpOperations = oldOp->getRegion(0).front().getOperations();
    llvm::iplist<mlir::Operation>::iterator it = newOpOperations.begin();
    std::advance(it, index);
    newOpOperations.splice(it, oldOpOperations);
}

int getOpIndex(mlir::Operation* haveBlockOp, mlir::Operation* targetOp) {
  // 找到op在block中的index
  auto& ops = haveBlockOp->getRegion(0).front().getOperations();
  int index = -1;
  for (auto& op : ops) {
    index++;
    if (&op == targetOp) return index;
  }
  return -1;
}


std::set<mlir::Operation*> getValueUsers(mlir::Value var) {
  // 获取value的使用者
  std::set<mlir::Operation*> users;
  for (auto user: var.getUsers()) {
    users.insert(user);
  }
  return users;
}

mlir::AffineExpr getOrderExpr(mlir::OpBuilder builder, int dimCount) {
  // 获取一个有序的连续累加的affine表达式
  mlir::AffineExpr sumExpr = builder.getAffineConstantExpr(0);
  for (int i=0; i<dimCount; i++) {
    sumExpr = sumExpr + builder.getAffineDimExpr(i);
  }
  return sumExpr;
}

int getExprDimNum(mlir::AffineExpr expr) {
  // 获取expr中exprDim的数量
  if (expr.dyn_cast<mlir::AffineDimExpr>()) {
    return 1;
  } else if (auto binaryExpr = expr.dyn_cast<mlir::AffineBinaryOpExpr>()) {
    return getExprDimNum(binaryExpr.getLHS()) + getExprDimNum(binaryExpr.getRHS());
  } else {
    return 0;
  }
}

mlir::AffineExpr shiftAffineExprDim(mlir::MLIRContext* context, mlir::AffineExpr expr, int shift) {
  // d0 + d1 + d2  =>  shift==1  =>  d1 + d2 + d3
  if (auto dimExpr_ = expr.dyn_cast<mlir::AffineDimExpr>()) {
    return mlir::getAffineDimExpr(dimExpr_.getPosition() + shift, context);
  } else if (auto binaryExpr_ = expr.dyn_cast<mlir::AffineBinaryOpExpr>()){
    auto LHS = shiftAffineExprDim(context, binaryExpr_.getLHS(), shift);
    auto RHS = shiftAffineExprDim(context, binaryExpr_.getRHS(), shift);
    return mlir::getAffineBinaryOpExpr(binaryExpr_.getKind(), LHS, RHS);
  } else {
    // allowed dim, constant, binaryOp
    auto constExpr_ = expr.dyn_cast<mlir::AffineConstantExpr>();
    assert(constExpr_);
    return constExpr_;
  }
}

mlir::AffineExpr getModifiedExpr(mlir::MLIRContext* context, mlir::AffineExpr inExpr, mlir::AffineExpr replaceExpr, int targetDim, int replaceNumberDims) {
  // d0 + d1 + d2  =>  target==1 & replace==[d1 + d2 + d3] =>  d0 + [d1 + d2 + d3] + d4
  if (auto dimExpr_ = inExpr.dyn_cast<mlir::AffineDimExpr>()) {
    if (dimExpr_.getPosition() == targetDim) {
      return replaceExpr;
    } else if (dimExpr_.getPosition() > targetDim) {
      return mlir::getAffineDimExpr(dimExpr_.getPosition() + replaceNumberDims - 1, context);
    } else {
      return dimExpr_;
    }
  } else if (auto binaryExpr_ = inExpr.dyn_cast<mlir::AffineBinaryOpExpr>()){
    auto LHS = getModifiedExpr(context, binaryExpr_.getLHS(), replaceExpr, targetDim, replaceNumberDims);
    auto RHS = getModifiedExpr(context, binaryExpr_.getRHS(), replaceExpr, targetDim, replaceNumberDims);
    return mlir::getAffineBinaryOpExpr(binaryExpr_.getKind(), LHS, RHS);
  } else {
    // allowed dim, constant, binaryOp
    auto constExpr_ = inExpr.dyn_cast<mlir::AffineConstantExpr>();
    assert(constExpr_);
    return constExpr_;
  }
}

std::vector<int64_t> getOptVectorizeGroup(int64_t width) {
  // 计算最优的向量化组合根据width
  std::vector<int64_t> group;
  while (width != 0) {
    if (width - 4 >= 0) {
      width = width - 4;
      group.push_back(4);
    } else if (width - 2 >= 0) {
      width = width - 2;
      group.push_back(2);
    } else {
      width = width -1;
      group.push_back(1);
    }
  }
  return group;
}



}