#ifndef _General_Funcs_h_
#define _General_Funcs_h_

#include "Common/Utils.h"
#include "Analysis/Analyzer.h"
#include "mlir/Support/LLVM.h"
#include <vector>
#include <tuple>

namespace KernelCodeGen {

mlir::OpBuilder getBuilder(mlir::Operation* op, Position pos);

std::tuple<int64_t, int64_t, int64_t> getLoopBoundAndStep(mlir::affine::AffineForOp loop);

template <typename memrefAllocOp>
memrefAllocOp createAllocOp(mlir::OpBuilder builder, llvm::SmallVector<int64_t> shape, mlir::Type dtype, MemorySpace space, int alignment) {
  // 创建allocaOp
  auto bufferType = mlir::MemRefType::get(shape, dtype, {}, static_cast<int>(space));
  auto allocOp = builder.create<memrefAllocOp>(builder.getUnknownLoc(), bufferType);
  allocOp.setAlignment(alignment);
  return allocOp;
}

void replaceAndErase(mlir::Operation* newOp, mlir::Operation* oldOp);


void spliceHaveBlockOp(mlir::Operation* newOp, mlir::Operation* oldOp, int index=0);

template <typename memrefAllocOp>
void eraseForOpIterVar(mlir::OpBuilder builder, mlir::affine::AffineForOp &forOp, memrefAllocOp allocOp, llvm::SmallVector<mlir::Value> bufVars) {
  // 只是将含有迭代变量的forop转换成不含迭代遍历的for，迭代变量使用storeOp代替
  // 创建新的forop
  builder.setInsertionPointAfter(forOp);
  mlir::Value replaceValue;
  auto loopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv, mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard nestedGuard(builder);
    auto loadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), allocOp.getResult(), bufVars);
    replaceValue = loadOp.getResult();
    builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
  };
  auto loopBAS = getLoopBoundAndStep(forOp);
  auto newLoop = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), std::get<0>(loopBAS), 
                  std::get<1>(loopBAS), std::get<2>(loopBAS), mlir::ValueRange({}), loopBody);
  auto& oldYieldOp = forOp.getBody()->getOperations().back();

  // 将旧forop的body转移到新的forop中，且body中使用了迭代变量的使用 loadop 的结果代替
  spliceHaveBlockOp(newLoop, forOp, /*index*/1);
  forOp.getInductionVar().replaceAllUsesWith(newLoop.getInductionVar());
  forOp.getRegionIterArgs()[0].replaceAllUsesWith(replaceValue);

  // 由于上面的操作将oldYieldOp也移动到了新的forop中，所以最后需要删除，且迭代过程使用storeOp代替
  auto yieldResult = oldYieldOp.getOperand(0);
  builder.setInsertionPointAfter(&oldYieldOp);
  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), yieldResult, allocOp.getResult(), bufVars); 
  oldYieldOp.erase();

  // 最后将forop之外使用了其迭代变量的地方全部替换成 loadOp 加载的数据
  builder.setInsertionPointAfter(newLoop);
  auto loadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), allocOp.getResult(), bufVars);
  replaceAndErase(loadOp, forOp);
  forOp = newLoop;
}

template <typename collectOp>
std::vector<collectOp> collectInnerOps(mlir::Operation* haveBlockOp) {
  // 收集含有block的op的下面指定的op
  std::vector<collectOp> ops;
  haveBlockOp->walk<mlir::WalkOrder::PreOrder>([&](collectOp op) {
    ops.push_back(op);
  });
  return ops;
}

template <typename innermostOp>
innermostOp getInnerMostOp(innermostOp imop) {
  // 找个最内层的那个op，这些算子都是含有block的op
  while (true) {
    bool isfind = true;
    for (auto &op : imop.getBody()->getOperations()) {
      if (auto childOp = mlir::dyn_cast<innermostOp>(op)) {
        imop = childOp;
        isfind = false;
        break;
      }
    }
    if (isfind) {
      return imop;
    }
  }
}

int getOpIndex(mlir::Operation* haveBlockOp, mlir::Operation* targetOp);

std::set<mlir::Operation*> getValueUsers(mlir::Value var);

mlir::AffineExpr getOrderExpr(mlir::OpBuilder builder, int dimCount);

mlir::AffineExpr shiftAffineExprDim(mlir::MLIRContext* context, mlir::AffineExpr expr, int shift);

mlir::AffineExpr getModifiedExpr(mlir::MLIRContext* context, mlir::AffineExpr inExpr, mlir::AffineExpr replaceExpr, int targetDim, int replaceNumberDims);

template <typename AffineMemoryOp>
int replaceIndexWithExpr(mlir::Value oldIv, std::vector<mlir::Value>& newIvs, AffineMemoryOp memOp, mlir::AffineExpr replaceExpr,
                         llvm::SmallVector<mlir::AffineExpr>& exprs, llvm::SmallVector<mlir::Value>& operands) {
  // d0 + d1 + d2 + [d3] + d4  => old:d3 & new: [d0 + d1 + d2]  =>  d0 + d1 + d2 + [d3 + d4 + d5] + d6
  mlir::OpBuilder builder(memOp);
  llvm::SmallVector<mlir::Value> operands_(memOp.getMapOperands());
  // 找到oldIv在Operands中的index，对应找到dx的x
  int targetDim = -1;
  bool found = false;
  for (auto item : operands_) {
    if (!found) targetDim += 1;
    if (item == oldIv) {
      found = true;
      for (auto iv : newIvs) { operands.push_back(iv); }
    } else {
      operands.push_back(item);
    }
  }

  // [d0 + d1 + d2]  =>  target==3  =>  [d3 + d4 + d5]
  replaceExpr = shiftAffineExprDim(builder.getContext(), replaceExpr, targetDim);

  // d0 + d1 + d2 + [d3] + d4  =>  d0 + d1 + d2 + [d3 + d4 + d5] + d6
  auto exprs_ = memOp.getAffineMap().getResults();
  for (auto expr_ : exprs_) {
    auto expr = getModifiedExpr(builder.getContext(), expr_, replaceExpr, targetDim, newIvs.size());
    exprs.push_back(expr);
  }  
  return operands.size();
}

std::vector<int64_t> getOptVectorizeGroup(int64_t width);


mlir::affine::AffineForOp load(mlir::OpBuilder builder, mlir::Value src, mlir::Value dst, mlir::AffineMap srcMap, mlir::AffineMap dstMap, 
                               llvm::SmallVector<mlir::Value> srcOperands, llvm::SmallVector<mlir::Value> dstOperands, 
                               int64_t loadWidth, int loadTimes);

}

#endif