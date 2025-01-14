#include "Conversion/General/Rewriter.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Pass/PassManager.h"
#include <algorithm>
#include <map>
#include <cmath>

namespace KernelCodeGen {
namespace Rewriter {

mlir::Value searchDescOp(mlir::ModuleOp module, std::string desc){
  // 遍历模块中的所有操作
  mlir::Value result;
  module.walk([&](mlir::Operation *op) {
    // 检查操作是否有 "kcg.desc" 属性
    if (auto descAttr = op->getAttrOfType<mlir::StringAttr>("kcg.desc")) {
      // 如果属性值匹配
      if (descAttr.getValue() == desc) {
        // 检查操作的第一个结果值
        if (!op->getResults().empty()) {
          result = op->getResult(0);
        }
        // 停止遍历
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  return result;
}

mlir::Value _inner_alloc_buffer(mlir::OpBuilder &builder, mlir::MemRefType &type) {
  if (type.getMemorySpaceAsInt() == int(KernelCodeGen::MemorySpace::local)){
    return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), type)->getResult(0);
  }
  return builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), type);
}

void OpSetDesc(mlir::Operation* op, const std::string& attrValue){
  mlir::OpBuilder b(op->getContext());
  op->setAttr("kcg.desc",b.getStringAttr(attrValue));
}

std::vector<mlir::affine::AffineForOp> split(mlir::affine::AffineForOp forOp, uint64_t num_output, std::vector<int64_t>&& factors) {
<<<<<<< HEAD
  // 创建一个空的切开的嵌套循环
  std::sort(factors.begin(), factors.end(), std::greater<uint64_t>());
  auto loopBAS = getLoopBoundAndStep(forOp);
  factors.insert(factors.begin(), std::get<1>(loopBAS));
  factors.push_back(1);
  mlir::SmallVector<int64_t, 16> upperBounds(factors.begin(), --(factors.end()));
  mlir::SmallVector<int64_t, 16> steps(++(factors.begin()), factors.end());
=======
  auto upperBoundsVector = factors;
  factors.insert(factors.begin(), 1);
  assert(factors.size() == num_output);
  std::reverse(factors.begin(), factors.end());
  auto lowerbound = forOp.getLowerBoundMap();
  auto upperbound = forOp.getUpperBoundMap();
  int step = forOp.getStep().getLimitedValue();
  assert(lowerbound.isConstant() == true);
  assert(upperbound.isConstant() == true);
  int64_t lb = lowerbound.getSingleConstantResult();
  int64_t ub = upperbound.getSingleConstantResult();
  assert(step == 1 && lb == 0);
  upperBoundsVector.push_back(ub);
  std::reverse(upperBoundsVector.begin(), upperBoundsVector.end());
>>>>>>> dev_bizefeng
  mlir::SmallVector<int64_t, 16> lowerBounds(num_output, /*Value=*/0);

  std::vector<mlir::Value> ivsVector;
  mlir::OpBuilder builder(forOp.getOperation());
  mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      //empty nested loops.
      for (auto iv : ivs) { ivsVector.push_back(iv); }
    }
  );

<<<<<<< HEAD
  // 将旧的forop内部的op转移到新的嵌套forOp中
  auto loops = collectInnerOps<mlir::affine::AffineForOp>(forOp->getPrevNode());
=======
  // build AffineMap: (i) -> (i1 + i2 + i3)

  auto prevNode = forOp->getPrevNode();
  std::vector<mlir::affine::AffineForOp> loops;
  mlir::affine::AffineForOp outermostForOp = mlir::dyn_cast<mlir::affine::AffineForOp>(prevNode);
  outermostForOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp newLoop) {
    loops.push_back(newLoop);
  });
  mlir::affine::AffineForOp innermostForOp = loops.back();
>>>>>>> dev_bizefeng
  // erase the yield op, as the forOp will bring the AffineYieldOp
  mlir::affine::AffineForOp innermostForOp = loops.back();
  innermostForOp.getBody()->back().erase();
<<<<<<< HEAD
  spliceHaveBlockOp(innermostForOp, forOp, -1);
=======
  innermostForOp.getBody()->getOperations().splice(innermostForOp.getBody()->end(), forOp.getBody()->getOperations());
  /* Method 1: Replace old iv with new iv attached affineMapAttr */
>>>>>>> dev_bizefeng

  // 需要修改affineMap
  auto oldIv = forOp.getInductionVar();
<<<<<<< HEAD
  std::set<mlir::Operation*> users = getValueUsers(oldIv);
  mlir::AffineExpr sumExpr = getOrderExpr(builder, ivsVector.size());
=======
  auto users_ = oldIv.getUsers();
  std::set<mlir::Operation*> users;
  for (auto user : users_) {users.insert(user);}

  int dimCount = 0;
  ///TODO: can be passed as a functional<>, so free the build style of expr.
  std::vector<mlir::AffineExpr> dims;
  mlir::AffineExpr sumExpr;
  for (int i = 0; i < ivsVector.size(); i += 1 ) {
    dims.push_back(std::move(builder.getAffineDimExpr(dimCount++)));
    if (i == 0) {
      sumExpr = dims[0];
    } else {
      sumExpr = sumExpr + dims.back();
    }
  }
>>>>>>> dev_bizefeng

  // 替换load/store/apply的map
  for (auto user : users) {
    mlir::OpBuilder builder(user);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    llvm::SmallVector<mlir::Value> operands;
    if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(user)) {
      auto mem = loadOp.getMemref();
      int dimCount = replaceIndexWithExpr(oldIv, ivsVector, loadOp, sumExpr, exprs, operands);
      mlir::AffineMap map = mlir::AffineMap::get(/*dimCount*/dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
      auto newLoadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), mem, map, llvm::ArrayRef<mlir::Value>(operands));
      replaceAndErase(newLoadOp, loadOp);
    } else if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(user)) {
      auto valueToStore = storeOp.getValue();
      auto mem = storeOp.getMemref();
      int dimCount = replaceIndexWithExpr(oldIv, ivsVector, storeOp, sumExpr, exprs, operands);
      mlir::AffineMap map = mlir::AffineMap::get(/*dimCount*/dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
      builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), valueToStore, mem, map, llvm::ArrayRef<mlir::Value>(operands));
      storeOp.erase();
    } else if (auto applyOp = mlir::dyn_cast<mlir::affine::AffineApplyOp>(user)) {
      int dimCount = replaceIndexWithExpr(oldIv, ivsVector, applyOp, sumExpr, exprs, operands);
      mlir::AffineMap map = mlir::AffineMap::get(/*dimCount*/dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
      auto newApplyOp = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), map, mlir::ValueRange(operands));
      replaceAndErase(newApplyOp, applyOp);
    } else {
      assert(false);
    }
  }

  forOp.erase();
  return loops;
}

<<<<<<< HEAD
mlir::Value bufferizeLoopCarryVar(mlir::affine::AffineForOp &carryVarLoop, std::vector<mlir::affine::AffineForOp> &loops) {
  // 将迭代遍历变成buffer，loops为buffer提供索引值
  llvm::SmallVector<int64_t> bufferShape;
=======
std::vector<mlir::affine::AffineForOp> localSplitU(mlir::affine::AffineForOp forOp, uint64_t num_output) {
  auto lowerbound = forOp.getLowerBoundMap();
  auto upperbound = forOp.getUpperBoundMap();
  int64_t lb = lowerbound.getSingleConstantResult();
  int64_t ub = upperbound.getSingleConstantResult();
  auto step = forOp.getStep().getLimitedValue();;

  std::vector<int64_t> upperBoundsVector, lowerBoundsVector, stepVector;
  upperBoundsVector.push_back(num_output);
  upperBoundsVector.push_back(ub / num_output);
  lowerBoundsVector.push_back(lb);
  lowerBoundsVector.push_back(lb);
  stepVector.push_back(step);
  stepVector.push_back(step);
  assert(upperBoundsVector.size() == 2);
  assert(lowerBoundsVector.size() == 2);
  assert(stepVector.size() == 2);
  

  assert(lowerbound.isConstant() == true);
  assert(upperbound.isConstant() == true);
  assert(step == 1 && lb == 0);

  mlir::SmallVector<int64_t, 16> lowerBounds(lowerBoundsVector.begin(), lowerBoundsVector.end());
  mlir::SmallVector<int64_t, 16> steps(stepVector.begin(), stepVector.end());
  mlir::SmallVector<int64_t, 16> upperBounds(upperBoundsVector.begin(), upperBoundsVector.end());

  std::vector<mlir::Value> ivsVector;

  mlir::OpBuilder builder(forOp.getOperation());
  mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      //empty nested loops.
      for (auto iv : ivs) {
        ivsVector.push_back(iv);
      }
    }
  );

  // build AffineMap

  auto prevNode = forOp->getPrevNode();
  std::vector<mlir::affine::AffineForOp> loops;
  mlir::affine::AffineForOp outermostForOp = mlir::dyn_cast<mlir::affine::AffineForOp>(prevNode);
  outermostForOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp newLoop) {
    loops.push_back(newLoop);
  });

  mlir::affine::AffineForOp innermostForOp = loops.back();
  // erase the yield op, as the forOp will bring the AffineYieldOp
  innermostForOp.getBody()->back().erase();
  innermostForOp.getBody()->getOperations().splice(innermostForOp.getBody()->end(), forOp.getBody()->getOperations());

  /* Method 1: Replace old iv with new iv attached affineMapAttr */

  auto oldIv = forOp.getInductionVar();
  auto users_ = oldIv.getUsers();

  std::set<mlir::Operation*> users;
  for (auto user : users_) {users.insert(user);}

  int dimCount = 0;
  ///TODO: can be passed as a functional<>, so free the build style of expr.
  std::vector<mlir::AffineExpr> dims;
  mlir::AffineExpr sumExpr;
  for (int i = 0; i < ivsVector.size(); i += 1 ) {
    dims.push_back(std::move(builder.getAffineDimExpr(dimCount++)));
    if (i == 0) {
      sumExpr = dims[0];
    } else {
      sumExpr = sumExpr + num_output * dims.back();
    }
    // llvm::outs() << "==== dims =====" << dims[i] << "\n";llvm::outs().flush();
  }

  for (auto user : users) {
    mlir::OpBuilder builder(user);
    if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(user)) {
      llvm::SmallVector<mlir::AffineExpr> exprs;
      llvm::SmallVector<mlir::Value> operands;
      int dimCount = replaceIndexWithExpr<mlir::affine::AffineLoadOp>(oldIv, ivsVector, loadOp, sumExpr, exprs, operands);
      mlir::AffineMap map = mlir::AffineMap::get(/*dimCount*/dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
      auto mem = loadOp.getMemref();
      auto newLoadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), mem, map, llvm::ArrayRef<mlir::Value>(operands));
      loadOp.getResult().replaceAllUsesWith(newLoadOp.getResult());
      loadOp.erase();
    } else if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(user)) {
      llvm::SmallVector<mlir::AffineExpr> exprs;
      llvm::SmallVector<mlir::Value> operands;
      auto valueToStore = storeOp.getValue();
      auto mem = storeOp.getMemref();
      int dimCount = replaceIndexWithExpr<mlir::affine::AffineStoreOp>(oldIv, ivsVector, storeOp, sumExpr, exprs, operands);
      // llvm::errs() << exprs.size() << "滑天下之大稽\n";
      mlir::AffineMap map = mlir::AffineMap::get(/*dimCount*/dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
      builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), valueToStore, mem, map, llvm::ArrayRef<mlir::Value>(operands));
      storeOp.erase();
    } else if (auto applyOp = mlir::dyn_cast<mlir::affine::AffineApplyOp>(user)) {
      llvm::SmallVector<mlir::AffineExpr> exprs;
      llvm::SmallVector<mlir::Value> operands;
      int dimCount = replaceIndexWithExpr<mlir::affine::AffineApplyOp>(oldIv, ivsVector, applyOp, sumExpr, exprs, operands);
      mlir::AffineMap map = mlir::AffineMap::get(/*dimCount*/dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
      auto newApplyOp = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), map, mlir::ValueRange(operands));
      applyOp.getResult().replaceAllUsesWith(newApplyOp.getResult());
      applyOp.erase();
    } else {
      assert(false);
    }
  }

  /* Method 2: Replace old iv with AffineApplyOp's result */


  // std::vector<mlir::AffineExpr> dims;
  // mlir::AffineExpr sumExpr;
  // for (int i = 0; i < num_output; i += 1 ) {
  //   dims.push_back(std::move(builder.getAffineDimExpr(i)));
  //   if (i == 0) {
  //     sumExpr = dims[0];
  //   } else {
  //     sumExpr = sumExpr + dims.back();
  //   }
  // }
  // auto sumMap = mlir::AffineMap::get(/*dimCount*/num_output, 0, sumExpr);
  // auto attribute = mlir::AffineMapAttr::get(sumMap);
  // builder.setInsertionPointToStart(innermostForOp.getBody());
  // mlir::IRRewriter rewriter(builder);
  // // rewriter.setInsertionPointToStart();
  // auto results = mlir::getAsOpFoldResult(llvm::SmallVector<mlir::Value>(ivsVector.begin(), ivsVector.end()));
  // auto operands = llvm::ArrayRef<mlir::OpFoldResult>(llvm::SmallVector<mlir::OpFoldResult>(results));
  // auto ivReplacement = mlir::makeComposedFoldedAffineApply(rewriter, rewriter.getUnknownLoc(), sumMap, operands);
  // auto oldIv = forOp.getInductionVar();
  // oldIv.replaceAllUsesWith(ivReplacement.get<mlir::Value>());

  forOp.erase();

  return loops;
}

mlir::Block* getClostScopeOp(mlir::Operation* op) {
  while (true) {
    auto parentOp = op->getParentOp();
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(parentOp)) {
      return module.getBody();
    } else if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(parentOp)){
      return &(func.getBlocks().front());
    } else if (auto parallelOp = mlir::dyn_cast<mlir::affine::AffineParallelOp>(parentOp)) {
      return parallelOp.getBody();
    }
    op = parentOp;
  }
}

//改成获取当前作用域
mlir::Block* getClostScopeOpNew(mlir::Operation* op) {
  while (true) {
    auto parentOp = op->getParentOp();
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(parentOp)) {
      return module.getBody();
    } else if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(parentOp)){
      return &(func.getBlocks().front());
    } else if (auto parallelOp = mlir::dyn_cast<mlir::affine::AffineParallelOp>(parentOp)) {
      return parallelOp.getBody();
    }
    op = parentOp;
  }
}

mlir::Value bufferizeLoopCarryVar(std::vector<mlir::affine::AffineForOp>& loops) {
  auto contain = [&](mlir::affine::AffineForOp A, mlir::affine::AffineForOp B) {  // A 包括 B
    if (A == B) return false;
    bool result = false;
    A.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
      if (forOp == B) {
        result = true;
      }
    });
    return result;
  };

  bool hasLoopCarryVar = false;
  mlir::affine::AffineForOp carryVarLoop;
  std::vector<int64_t> bufferShape;
>>>>>>> dev_bizefeng
  llvm::SmallVector<mlir::Value> bufferAdrressOperand;
  for (auto loop : loops) {
    auto loopBAS = getLoopBoundAndStep(loop);
    bufferShape.push_back((std::get<1>(loopBAS) - std::get<0>(loopBAS)) / std::get<2>(loopBAS));
    bufferAdrressOperand.push_back(loop.getInductionVar());
  }
  
  auto builder = getBuilder(loops[0], Position::before);
  auto carryVar = carryVarLoop.getRegionIterArgs()[0];
<<<<<<< HEAD
  auto allocOp = createAllocOp<mlir::memref::AllocaOp>(builder, bufferShape, carryVar.getType(), MemorySpace::local, KCG_ALIGNBYTE);
  // step1: 将buffer初始化值
=======
  auto dtype = carryVar.getType();
  auto bufferType = mlir::MemRefType::get(
    bufferShape, dtype, {}, static_cast<int>(MemorySpace::local));
  builder.setInsertionPoint(loops[1]);
  auto allocOp = builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), bufferType);
  allocOp.setAlignment(KCG_ALIGNBYTE);
  // step1: init the buffer
  // the last operand of AffineForOp.
>>>>>>> dev_bizefeng
  auto initValue = carryVarLoop.getOperands().back();
  auto defineOp = initValue.getDefiningOp();
  builder.setInsertionPointAfter(defineOp);
  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), initValue, allocOp.getResult(), bufferAdrressOperand);
  // step2: 替换含迭代变量的循环
  eraseForOpIterVar(builder, carryVarLoop, allocOp, bufferAdrressOperand);
  return allocOp.getResult();
}

void loopToParallelZ(mlir::affine::AffineForOp loop, mlir::affine::AffineParallelOp &parallelOp) {
  // loop add to parallel

  llvm::SmallVector<mlir::AffineMap> lbMaps;
  llvm::SmallVector<mlir::AffineMap> upMaps;
  llvm::SmallVector<mlir::Value> lbOperands;
  llvm::SmallVector<mlir::Value> upOperands;
  llvm::SmallVector<int64_t> steps;

  // loop operand
  lbMaps.push_back(loop.getLowerBoundMap());
  upMaps.push_back(loop.getUpperBoundMap());
  lbOperands.append(loop.getLowerBoundOperands().begin(), loop.getLowerBoundOperands().end());
  upOperands.append(loop.getUpperBoundOperands().begin(), loop.getUpperBoundOperands().end());
  steps.push_back(loop.getStep().getLimitedValue());

  // parallel operands
  auto ss = parallelOp.getSteps();
  for (unsigned i=0; i<parallelOp.getNumDims(); i++) {
    lbMaps.push_back(parallelOp.getLowerBoundMap(i));
    upMaps.push_back(parallelOp.getUpperBoundMap(i));
    steps.push_back(ss[i]);
  }
  lbOperands.append(parallelOp.getLowerBoundsOperands().begin(), parallelOp.getLowerBoundsOperands().end());
  upOperands.append(parallelOp.getUpperBoundsOperands().begin(), parallelOp.getUpperBoundsOperands().end());

  // create new parallelOp
  mlir::OpBuilder builder = getBuilder(parallelOp, Position::before);
  mlir::affine::AffineParallelOp newParallelOp = builder.create<mlir::affine::AffineParallelOp>(
    builder.getUnknownLoc(), mlir::TypeRange(), llvm::ArrayRef<mlir::arith::AtomicRMWKind>(),
    llvm::ArrayRef<mlir::AffineMap>(lbMaps), lbOperands,
    llvm::ArrayRef<mlir::AffineMap>(upMaps), upOperands,
    llvm::ArrayRef<int64_t>(steps));
  
  // replace old parallelOp
  parallelOp.getBody()->back().erase();
  spliceHaveBlockOp(newParallelOp, parallelOp);
  auto oldIVs = parallelOp.getIVs();
  auto newIVs = newParallelOp.getIVs();
  for (int i=0; i<oldIVs.size(); i++) {
    oldIVs[i].replaceAllUsesWith(newIVs[i+1]);
  }
  parallelOp.erase();

  // erase loop
  auto parentOp = loop.getOperation()->getParentOp();
  int index = getOpIndex(parentOp, loop);
  loop.getBody()->back().erase();
  spliceHaveBlockOp(parentOp, loop, index);
  loop.getInductionVar().replaceAllUsesWith(newIVs[0]);
  loop.erase();

  parallelOp = newParallelOp;
}
 
// Swap two nested loops.
// if outer loop contains multiple Operations, clone the outer loop to maintain correctness.
void swap(mlir::affine::AffineForOp outer, mlir::affine::AffineForOp inner) {
  auto& ops = outer.getBody()->getOperations();
  auto opNumber = ops.size();
  int position = 0;
  mlir::Operation* innerOp = inner;
  for (auto& op : ops) {
    if (&op == innerOp) {
      break;
    }
    position += 1;
  }
  // must found.
  assert(position < opNumber);

  bool existOpBeforeLoop = position != 0;
  // considering the affine.yield
  bool existOpAfterLoop = position != opNumber - 2;

  if (existOpBeforeLoop) {
    mlir::OpBuilder b(outer->getBlock(), mlir::Block::iterator(outer));
    mlir::IRMapping mapper;
    // auto cloned = b.clone(*outer, mapper);
    b.clone(*outer, mapper);
    auto cloned = (--mlir::Block::iterator(outer));

    auto clonedFor = mlir::dyn_cast<mlir::affine::AffineForOp>(cloned);
    assert(clonedFor);
    auto& ops_ = clonedFor.getBody()->getOperations();
  
    int count = 0;
    auto iter = --(--(ops_.end()));
    int number = ops_.size();
    for (int i = 0; i < number - position - 1; i++) {
      ++count;
      // it seems that iter->erase() will cause segment false.
      (iter--)->erase();
    }
  }
  if (existOpAfterLoop) {
    mlir::OpBuilder b(outer->getBlock(), ++mlir::Block::iterator(outer));
    mlir::IRMapping mapper;
    auto cloned = b.clone(*outer, mapper);
    auto& ops_ = mlir::dyn_cast<mlir::affine::AffineForOp>(cloned).getBody()->getOperations();
    auto iter = ops_.end();
    int number = ops_.size();
    for (int i = 0; i < number - position; i++) --iter;
    for(int i = 0; i <= position; i++) {
      (iter--)->erase();
    }
  }
  // clear current outer loop
  if (existOpBeforeLoop || existOpAfterLoop) {
    auto iter = --(ops.end());
    int number = ops.size();
    for (int i = 0; i < number; i++) {
      if (i == number - 1 - position || i == 0) {
        --iter;
      } else {
        (iter--)->erase();
      }
    }

  }
  // int count = 0;
  // for (auto& op : ops) {
  //   if (count != position && count !=  opNumber-1) {
  //     op.erase();
  //   }
  //   count += 1;
  // }


  /// step1: move the body of inner to outer
  // erase the yield op
  inner.getBody()->back().erase();
  // this block contain the inner Op
  inner->getBlock()->getOperations().splice( // this block is belong to outer
    mlir::Block::iterator(inner),
    inner.getBody()->getOperations());

  /// step2: move inner before outer.
  inner->moveBefore(outer);

  /// step3: make the outer as the body of inner
  inner.getBody()->getOperations().splice(inner.getBody()->end(),
                  outer->getBlock()->getOperations(), mlir::Block::iterator(outer));//only the outer.

  mlir::OpBuilder builder(inner.getContext());
  builder.setInsertionPointToEnd(inner.getBody());
  builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
}

void reorder(const std::vector<mlir::affine::AffineForOp>& loops) {

  std::map<mlir::affine::AffineForOp, int, CompareLoop> loopPriority;
  int priority = loops.size();
  for (auto loop : loops) {
    loopPriority[loop] = priority--;
  }

  auto findFirstTargetLoop = [&](mlir::affine::AffineForOp root) {
    if (loopPriority.count(root) != 0) return root;
    mlir::affine::AffineForOp result;
    bool found = false;
    root.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
      if ((!found) && loopPriority.count(forOp) != 0) {
        result = forOp;
        found = true;
      }
    });
    assert(found);
    return result;
  };

  auto containTargetLoop = [&](mlir::affine::AffineForOp root) {

    auto& ops = root.getBody()->getOperations();
    mlir::affine::AffineForOp sonLoop;
    bool result = false;

    for (auto& op : ops) {
      if (auto sonOp = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
        if (loopPriority.count(sonOp)) {
          result = true;
          sonLoop = sonOp;
          break;
        }
      }
    }
    return result ? sonLoop : nullptr;
  };

  bool swapped;

  mlir::affine::AffineForOp rootForOp = Analyzer::findRootLoop(loops[0]);

  auto parentLoop_ = findFirstTargetLoop(rootForOp);

  // bubble sort.
  do {
    swapped = false;
    mlir::affine::AffineForOp parentLoop = parentLoop_;
    while (auto sonLoop = containTargetLoop(parentLoop)) {
      if (loopPriority[parentLoop] < loopPriority[sonLoop]) {
        swap(parentLoop, sonLoop);
        swapped = true;
      } else {
        parentLoop = sonLoop;
      }
    }
  } while (swapped);
}

mlir::affine::AffineParallelOp parallel(const std::vector<mlir::affine::AffineForOp>& forOps) {
  // X, Y, Z
  assert(forOps.size() <= 3);
  llvm::SmallVector<mlir::AffineMap> lbMaps;
  llvm::SmallVector<mlir::AffineMap> upMaps;
  llvm::SmallVector<int64_t> steps;
  llvm::SmallVector<int64_t> applySteps;

  mlir::OpBuilder builder(forOps[0]);
  for (auto forOp : forOps) {
    int64_t step = forOp.getStep().getLimitedValue();
    int64_t lowerBound = forOp.getLowerBoundMap().getConstantResults()[0];
    int64_t upperBound = forOp.getUpperBoundMap().getConstantResults()[0] / step;
    lbMaps.push_back(mlir::AffineMap::get(0, 0, llvm::ArrayRef<mlir::AffineExpr>(builder.getAffineConstantExpr(lowerBound)), builder.getContext()));
    upMaps.push_back(mlir::AffineMap::get(0, 0, llvm::ArrayRef<mlir::AffineExpr>(builder.getAffineConstantExpr(upperBound)), builder.getContext()));
    steps.push_back(1);
    applySteps.push_back(step);
  }

  mlir::affine::AffineParallelOp parallelOp = builder.create<mlir::affine::AffineParallelOp>(
    builder.getUnknownLoc(), mlir::TypeRange(), llvm::ArrayRef<mlir::arith::AtomicRMWKind>(),
    llvm::ArrayRef<mlir::AffineMap>(lbMaps), mlir::ValueRange({}),
    llvm::ArrayRef<mlir::AffineMap>(upMaps), mlir::ValueRange({}),
    llvm::ArrayRef<int64_t>(steps));

  // erase the yield op of innermost loop
  auto innermost = forOps.back();
  innermost.getBody()->back().erase();
  spliceHaveBlockOp(parallelOp, innermost);

  llvm::SmallVector<mlir::Value> applyResults;
  mlir::AffineExpr dim = builder.getAffineDimExpr(0);
  auto ivs = parallelOp.getIVs();
  builder.setInsertionPointToStart(parallelOp.getBody());
  for (int i=0; i<ivs.size(); i++) {
    auto map = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(dim * applySteps[i]), builder.getContext());
    auto applyOp = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), map, mlir::ValueRange({ivs[i]}));
    applyResults.push_back(applyOp.getResult());
  }

  int count = applyResults.size() - 1;
  for (auto iter = forOps.rbegin(); iter != forOps.rend(); ++iter) {
    auto forOp = *iter;
    forOp.getInductionVar().replaceAllUsesWith(applyResults[count--]);
    forOp.erase();
  }
  return parallelOp;
}

llvm::SmallVector<mlir::Value> parallelToOneDim(mlir::affine::AffineParallelOp &parallelOp, int* outUpperBound) {
  // 将parallelOp转成一维表示
  std::vector<int64_t> uppers;
  auto builder = getBuilder(parallelOp, Position::before);
  int64_t upperBound = 1;
  for (auto i : parallelOp.getUpperBoundsMap().getConstantResults()) {
    upperBound *= i;
    uppers.push_back(i);
  }
  if(outUpperBound != nullptr){
    *outUpperBound = upperBound;
  }
  // create new parallelOp
  auto lowerMap = mlir::AffineMap::get(0, 0, llvm::ArrayRef<mlir::AffineExpr>(builder.getAffineConstantExpr(0)), builder.getContext());
  auto upperMap = mlir::AffineMap::get(0, 0, llvm::ArrayRef<mlir::AffineExpr>(builder.getAffineConstantExpr(upperBound)), builder.getContext());
  mlir::affine::AffineParallelOp newOp = builder.create<mlir::affine::AffineParallelOp>(
    builder.getUnknownLoc(), mlir::TypeRange(), llvm::ArrayRef<mlir::arith::AtomicRMWKind>(),
    llvm::ArrayRef<mlir::AffineMap>(lowerMap), mlir::ValueRange({}),
    llvm::ArrayRef<mlir::AffineMap>(upperMap), mlir::ValueRange({}),
    llvm::ArrayRef<int64_t>({1}));

  // create new maps
  llvm::SmallVector<mlir::AffineMap> maps;
  builder.setInsertionPointToStart(newOp.getBody());
  mlir::AffineExpr tid = builder.getAffineDimExpr(0);
  int64_t front = 1;
  for (int i=1; i<uppers.size(); i++) {
    int64_t sum = 1;
    for (int j=i; j<uppers.size(); j++) { 
      sum *= uppers[j]; 
    }
    maps.push_back(mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(tid.floorDiv(sum)), builder.getContext()));
    tid = tid % sum;
    if (i == uppers.size() - 1) {
      maps.push_back(mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(tid), builder.getContext()));
    }
  }

  // create affineApplyOp
  llvm::SmallVector<mlir::Value> newIVs;
  for (auto map : maps) {
    auto axesApplyOp = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), map, mlir::ValueRange(newOp.getIVs()));
    newIVs.push_back(axesApplyOp.getResult());
  }

  // move
  parallelOp.getBody()->back().erase();
  spliceHaveBlockOp(newOp, parallelOp, maps.size());
  auto oldIVs = parallelOp.getIVs();
  for (int i=0; i<oldIVs.size(); i++) {
    oldIVs[i].replaceAllUsesWith(newIVs[i]);
  }
  parallelOp.erase();
  parallelOp = newOp;

  return newIVs;
}

// dst is register.
mlir::affine::AffineForOp read(mlir::Value src, mlir::Value dst, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, 
                              std::vector<int64_t> widths, mlir::affine::AffineForOp compute_at, Position pos) {
  auto dimsNum = map.getNumDims();
  auto builder = getBuilder(compute_at, pos);
  auto dstType = dst.getType().dyn_cast<mlir::MemRefType>();
  int64_t totalWidth = dstType.getShape()[0];

  std::vector<int> times;
  mlir::AffineExpr expr = builder.getAffineConstantExpr(0);
  for (int i=0; i<widths.size(); i++) {
    auto dim = builder.getAffineDimExpr(i);
    expr = expr + dim * widths[i];
    times.push_back(totalWidth / widths[i]);
    totalWidth = widths[i];
  }
  mlir::AffineMap dstMap;
  if (dimsNum - (operands.size() + times.size()) == 1) {
    expr = expr + builder.getAffineDimExpr(widths.size());
    dstMap = mlir::AffineMap::get(/*dimCount*/widths.size() + 1, 0, llvm::ArrayRef<mlir::AffineExpr>(expr), builder.getContext());
  } else {
    dstMap = mlir::AffineMap::get(/*dimCount*/widths.size(), 0, llvm::ArrayRef<mlir::AffineExpr>(expr), builder.getContext());
  }
  
  llvm::SmallVector<mlir::Value> dstOperands;
  auto load = shiftBufferDatas(builder, src, dst, map, dstMap, operands, dstOperands, widths.back(), times);
  return load;
}

// src is register
mlir::affine::AffineForOp write(mlir::Value src, mlir::Value dst, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, 
                                std::vector<int64_t> widths, mlir::affine::AffineForOp compute_at, Position pos) {
  auto dimsNum = map.getNumDims();
  auto builder = getBuilder(compute_at, pos);
  auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
  int64_t totalWidth = srcType.getShape()[0];

  std::vector<int> times;
  mlir::AffineExpr expr = builder.getAffineConstantExpr(0);
  for (int i=0; i<widths.size(); i++) {
    auto dim = builder.getAffineDimExpr(i);
    expr = expr + dim * widths[i];
    times.push_back(totalWidth / widths[i]);
    totalWidth = widths[i];
  }
  mlir::AffineMap srcMap;
  if (dimsNum - (operands.size() + times.size()) == 1) {
    expr = expr + builder.getAffineDimExpr(widths.size());
    srcMap = mlir::AffineMap::get(/*dimCount*/widths.size() + 1, 0, llvm::ArrayRef<mlir::AffineExpr>(expr), builder.getContext());
  } else {
    srcMap = mlir::AffineMap::get(/*dimCount*/widths.size(), 0, llvm::ArrayRef<mlir::AffineExpr>(expr), builder.getContext());
  }
  
  llvm::SmallVector<mlir::Value> srcOperands;
  auto store = shiftBufferDatas(builder, src, dst, srcMap, map, srcOperands, operands, widths.back(), times);
  return store;
}

// src is register
mlir::affine::AffineForOp write(mlir::OpBuilder& builder, mlir::Value src, mlir::Value dst, 
    mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, int64_t width) {
  auto dimsNum = map.getNumDims();
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  bool twoLoop = abs(dimsNum - operands.size()) == 2;
  auto srcMap = !twoLoop ? 
                mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(dim0 * width), builder.getContext()) :
                mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(dim0 * width + dim1), builder.getContext());
  auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
  // registers is always 1 dim.
  auto storeTimes = srcType.getShape()[0] / width;
  auto storeBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
                      mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard nestedGuard(builder);
    // loop iterator is the last operand.
    operands.push_back(iv);
    if (twoLoop) {
      auto innerBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv_inner,
                        mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder);
        // loop iterator is the last operand.
        operands.push_back(iv_inner);
        auto vectorType = mlir::VectorType::get(1, srcType.getElementType());
        auto ld = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, src, srcMap, mlir::ValueRange({iv, iv_inner}));
        auto st = builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), dst, map, operands);
        builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
      };
      auto storeInner = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), 
          0, width, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({}), innerBody);
      builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
    } else { 
      auto vectorType = mlir::VectorType::get(width, srcType.getElementType());
      auto ld = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, src, srcMap, mlir::ValueRange({iv}));
      auto st = builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), dst, map, operands);
      builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
    }
  };
  auto store = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), 
     0, storeTimes, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({}), storeBody);
  return store;
}

mlir::gpu::BarrierOp barrier(mlir::affine::AffineForOp compute_at, Position pos) {
  auto builder = getBuilder(compute_at, pos);
  return builder.create<mlir::gpu::BarrierOp>(builder.getUnknownLoc());
}

void cache_read(mlir::affine::AffineForOp scope, mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands) {
  scope.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineLoadOp load) {
    if (load.getMemref() != src) return;
    mlir::OpBuilder builder(load);
    auto newLoad = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), cached, map, operands);
    load.getResult().replaceAllUsesWith(newLoad.getResult());
    load.erase();
  });
}

void cache_write(mlir::affine::AffineForOp scope, mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands) {
  scope.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineStoreOp store) {
    if (store.getMemref() != src) return;
    mlir::OpBuilder builder(store);
    auto newStore = builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), store.getValue(), cached, map, operands);
    store.erase();
  });
}

///TODO: two level vector.
std::vector<std::vector<mlir::affine::AffineForOp>> get_write(mlir::affine::AffineParallelOp parallelLevel, mlir::Value dst) {
  std::vector<std::vector<mlir::affine::AffineForOp>> results;
  std::vector<mlir::affine::AffineStoreOp> stores;
  parallelLevel.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineStoreOp store) {
    if (store.getMemref() != dst) return;
    stores.push_back(store);
  });
  for (auto store : stores) {
    std::vector<mlir::affine::AffineForOp> result;
    mlir::affine::AffineForOp parent;
    mlir::Operation* cur = store;
    while (parent = mlir::dyn_cast<mlir::affine::AffineForOp>(cur->getParentOp())) {
      result.push_back(parent);
      cur = parent;
    }
    std::reverse(result.begin(), result.end());
    results.push_back(result);
  }
  return results;
}

mlir::affine::AffineForOp vectorize(mlir::affine::AffineForOp readOrWrite, int64_t width) {
  int64_t step = readOrWrite.getStep().getLimitedValue();
  int64_t ub = readOrWrite.getConstantUpperBound();
  int64_t lb = readOrWrite.getConstantLowerBound();
  assert(step = 1 && lb == 0 && ub % width == 0);
  readOrWrite.setStep(width);
  readOrWrite.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineLoadOp load) {
    mlir::OpBuilder builder(load);
    auto type = load.getMemRef().getType().dyn_cast<mlir::MemRefType>();
    auto vectorType = mlir::VectorType::get(width, type.getElementType());
    auto vectorLoad = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, load.getMemRef(), load.getAffineMap(), load.getMapOperands());
    load.getResult().replaceAllUsesWith(vectorLoad.getResult());
    load.erase();
  });
  readOrWrite.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineStoreOp store) {
    mlir::OpBuilder builder(store);
     auto type = store.getMemRef().getType().dyn_cast<mlir::MemRefType>();
    auto vectorType = mlir::VectorType::get(width, type.getElementType());
    auto vectorStore = builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), store.getValue(), store.getMemRef(), store.getAffineMap(), store.getMapOperands());
    store.erase();
  });
  return readOrWrite;
}

std::pair<mlir::affine::AffineForOp, mlir::affine::AffineForOp> 
splitUReduce(mlir::Value src, mlir::Value dst, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands,
               int localSplitU, int64_t globStoreWidth, mlir::affine::AffineForOp compute_at, Position pos) {
  // splitU!=1时，插入将多层结果进行累加求和的结构
  auto builder = getBuilder(compute_at, pos);
  auto dstType = dst.getType().dyn_cast<mlir::MemRefType>();
  int64_t regCTotalWidth = dstType.getShape()[0];   // 16
  int64_t globStoreTotalWidth = regCTotalWidth / localSplitU;  // 8
  int64_t globStoreNum = globStoreTotalWidth / globStoreWidth;  // 4

  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  auto dstExpr = dim0 * globStoreWidth;
  auto reduceExpr = dim0 * globStoreWidth + dim1;
  auto dstMap = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(dstExpr), builder.getContext());
  auto reduceMap = mlir::AffineMap::get(2, 0, llvm::ArrayRef<mlir::AffineExpr>(reduceExpr), builder.getContext());

  auto oneMap = mapDimToConstant(builder, map, /*target*/1, /*constant*/0);
  llvm::SmallVector<mlir::Value> oneLoopSrcOperands(operands.begin(), operands.end());
  auto oneLoopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value iv, mlir::ValueRange iterArgs) {
    oneLoopSrcOperands.push_back(iv);
    llvm::SmallVector<mlir::Value> oneLoopDstOperands{iv};
    auto vectorType = mlir::VectorType::get(globStoreWidth, dstType.getElementType());
    auto ld = b.create<mlir::affine::AffineVectorLoadOp>(b.getUnknownLoc(), vectorType, src, oneMap, oneLoopSrcOperands);
    b.create<mlir::affine::AffineVectorStoreOp>(b.getUnknownLoc(), ld, dst, dstMap, oneLoopDstOperands);
    b.create<mlir::affine::AffineYieldOp>(b.getUnknownLoc());
  };
  auto oneLoop = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), 0, globStoreNum, 1, mlir::ValueRange({}), oneLoopBody);

  auto newMap = addDimToMap(builder, map);
  llvm::SmallVector<mlir::Value> twoLoopSrcOperands(operands.begin(), operands.end());
  mlir::SmallVector<int64_t, 2> upperBounds{localSplitU, globStoreNum, globStoreWidth};
  mlir::SmallVector<int64_t, 2> steps(3, /*Value=*/1);
  mlir::SmallVector<int64_t, 2> lowerBounds{1, 0, 0};
  mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange ivs) {
      for (auto iv : ivs) {
        twoLoopSrcOperands.push_back(iv);
      }
      llvm::SmallVector<mlir::Value> reduceOperands{ivs[1], ivs[2]};
      auto loadRegC = b.create<mlir::affine::AffineLoadOp>(b.getUnknownLoc(), dst, reduceMap, reduceOperands);
      auto loadShC = b.create<mlir::affine::AffineLoadOp>(b.getUnknownLoc(), src, newMap, twoLoopSrcOperands);
      auto addOp = b.create<mlir::arith::AddFOp>(b.getUnknownLoc(), loadRegC, loadShC);
      b.create<mlir::affine::AffineStoreOp>(b.getUnknownLoc(), addOp, dst, reduceMap, reduceOperands);
    });
  
  return std::make_pair(oneLoop, mlir::dyn_cast<mlir::affine::AffineForOp>(oneLoop->getNextNode()));
}


mlir::affine::AffineForOp splitUWrite(mlir::Value src, mlir::Value dst, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, 
                                      int localSplitU, int64_t globStoreWidth, mlir::affine::AffineForOp compute_at, Position pos) {
  // 将结果累加完成后，再将结果写回到C矩阵
  auto builder = getBuilder(compute_at, pos);
  auto dim0 = builder.getAffineDimExpr(0);
  auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
  int64_t regTotalWidth = srcType.getShape()[0];
  int globStoreNum = regTotalWidth / localSplitU / globStoreWidth;
  mlir::AffineMap srcMap = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(dim0 * globStoreWidth), builder.getContext());
  llvm::SmallVector<mlir::Value> srcOperands;
  auto store = shiftBufferDatas(builder, src, dst, srcMap, map, srcOperands, operands, globStoreWidth, {globStoreNum});
  return store;
}

mlir::Value bufferCombine(std::vector<std::vector<mlir::Value>> buffers) {
  // 将buffer合并到一个，“{{smA, smB}, {smC}}”，smA+smB的大小比较smC的大小，取最大的size创建一维的buffer
  // {smA, smB}与{smC}可复用
  std::vector<std::pair<mlir::Value, int64_t>> bufAndOffsets;
  int64_t maxBufSize = 0;
  for (auto buffer : buffers) {
    int64_t bufSize = 0;
    for (auto buf : buffer) {
      auto bufType = buf.getType().dyn_cast<mlir::MemRefType>();
      int64_t size = 1;
      for (auto shape : bufType.getShape()) { size *= shape; }
      bufSize += size;
      bufAndOffsets.push_back(std::make_pair(buf, bufSize - size));
    }
    if (maxBufSize < bufSize) { maxBufSize = bufSize; }
  }

  mlir::OpBuilder builder = getBuilder(buffers[0][0].getDefiningOp(), Position::before);
  auto bufType = buffers[0][0].getType().dyn_cast<mlir::MemRefType>();
  auto memSpace = static_cast<MemorySpace>(bufType.getMemorySpaceAsInt());
  auto elementType = bufType.getElementType();
  mlir::Value newBuffer;
  if (memSpace == MemorySpace::shared) {
    auto sharedAlloc = createAllocOp<mlir::memref::AllocOp>(builder, {maxBufSize}, elementType, memSpace, KCG_ALIGNBYTE);
    newBuffer = sharedAlloc.getResult();
  } else {
    auto localAlloca = createAllocOp<mlir::memref::AllocaOp>(builder, {maxBufSize}, elementType, memSpace, KCG_ALIGNBYTE);
    newBuffer = localAlloca.getResult();
  }

  for (auto bufAndOffset : bufAndOffsets) {
    auto users = getValueUsers(bufAndOffset.first);
    int64_t offset = bufAndOffset.second;
    for (auto user : users) {
      builder.setInsertionPointAfter(user);
      if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(user)) {
        auto newMap = getOneDimMap(loadOp, offset);
        auto newLoadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), newBuffer, newMap, loadOp.getMapOperands());
        replaceAndErase(newLoadOp, loadOp);
      } else if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(user)) {
        auto newMap = getOneDimMap(storeOp, offset);
        builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), storeOp.getValue(), newBuffer, newMap, storeOp.getMapOperands());
        storeOp.erase();
      } else if (auto vectorLoadOp = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(user)) {
        auto newMap = getOneDimMap(vectorLoadOp, offset);
        auto newVectorLoadOp = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorLoadOp.getVectorType(), 
                                                                                newBuffer, newMap, vectorLoadOp.getMapOperands());
        replaceAndErase(newVectorLoadOp, vectorLoadOp);
      } else if (auto vectorStoreOp = mlir::dyn_cast<mlir::affine::AffineVectorStoreOp>(user)) {
        auto newMap = getOneDimMap(vectorStoreOp, offset);
        builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), vectorStoreOp.getValue(), 
                                                          newBuffer, newMap, vectorStoreOp.getMapOperands());
        vectorStoreOp.erase();
      }
    }
    mlir::Operation* defOp = bufAndOffset.first.getDefiningOp();
    defOp->erase();
  }
  return newBuffer;
}

void BlockMapping(mlir::affine::AffineParallelOp gridLevel, int64_t groupWidth, bool isCol) {
  // 重映射block的位置，提高L2 cache命中率
  std::vector<int64_t> uppers;
  for (auto i : gridLevel.getUpperBoundsMap().getConstantResults()) { uppers.push_back(i); }
  auto applyResults = parallelToOneDim(gridLevel);
  
  auto ivs = gridLevel.getIVs();
  mlir::OpBuilder builder = getBuilder(gridLevel, Position::begin);
  mlir::AffineExpr dim = builder.getAffineDimExpr(0);

  int64_t groupHeight, otherWidth;
  if (isCol) {
    groupHeight = uppers[uppers.size()-1];
    otherWidth = uppers[uppers.size()-2];
  } else {
    groupHeight = uppers[uppers.size()-2];
    otherWidth = uppers[uppers.size()-1];
  }

  int64_t groupNum = groupWidth * groupHeight;
  auto start = dim.floorDiv(groupNum) * groupWidth;
  mlir::Value exasVal0, exasVal1;
  if (otherWidth % groupWidth == 0) {  // 可以整除
    auto exas0Map = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(start + dim % groupWidth), builder.getContext());
    auto exas0 = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), exas0Map, mlir::ValueRange({ivs[0]}));
    auto exas1Map = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>((dim % groupNum).floorDiv(groupWidth)), builder.getContext());
    auto exas1 = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), exas1Map, mlir::ValueRange({ivs[0]}));
    exasVal0 = exas0.getResult();
    exasVal1 = exas1.getResult();
  } else {
    auto startMap = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(start), builder.getContext());
    auto cstGroupWidth = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), groupWidth);
    auto cstgroupNum = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), groupNum);

    auto applyStart = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), startMap, mlir::ValueRange({ivs[0]}));
    auto cstGroupWidth_ = builder.create<mlir::arith::MinUIOp>(builder.getUnknownLoc(), applyStart.getResult(), cstGroupWidth.getResult());
    auto modOp0 = builder.create<mlir::arith::RemUIOp>(builder.getUnknownLoc(), ivs[0], cstGroupWidth_.getResult());
    auto exas0 = builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), modOp0.getResult(), applyStart.getResult());

    auto modOp1 = builder.create<mlir::arith::RemUIOp>(builder.getUnknownLoc(), ivs[0], cstgroupNum.getResult());
    auto exas1 = builder.create<mlir::arith::DivUIOp>(builder.getUnknownLoc(), modOp1.getResult(), cstGroupWidth_.getResult());
    exasVal0 = exas0.getResult();
    exasVal1 = exas1.getResult();
  }

  if (isCol) {
    applyResults[0].replaceAllUsesWith(exasVal0);
    applyResults[1].replaceAllUsesWith(exasVal1);
  } else {
    applyResults[0].replaceAllUsesWith(exasVal1);
    applyResults[1].replaceAllUsesWith(exasVal0);
  }
}

std::vector<std::vector<mlir::affine::AffineForOp>> pipeline(std::vector<mlir::affine::AffineForOp> readBodys, mlir::Value& buffer, mlir::affine::AffineForOp compute_at, std::string bufferName) {

  std::vector<std::vector<mlir::affine::AffineForOp>> results;

  // llvm::outs() << "test1\n";llvm::outs().flush();
  /* step1: double buffer.*/
  // 获取 buffer 的类型，并尝试转为 mlir::MemRefType。
<<<<<<< HEAD
  // llvm::outs() << buffer.getType();llvm::outs().flush();
  auto bufferType = buffer.getType().dyn_cast<mlir::MemRefType>();
  // llvm::outs() << "test2\n";llvm::outs().flush();
=======
  auto bufferType = buffer.getType().dyn_cast<mlir::MemRefType>();
>>>>>>> dev_bizefeng
  // 创建一个小向量 shape 来保存新缓冲区的形状。
  mlir::SmallVector<int64_t> shape;
  /// double size on top dim.
  // 先插入一个 2，意味着我们在最高维度多插入了一维，大小为 2，用于实现双缓冲。
  shape.push_back(2);
<<<<<<< HEAD
  // llvm::outs() << "test3\n";llvm::outs().flush();
=======
>>>>>>> dev_bizefeng
  // 遍历原有的 bufferType 的维度，并将其依次推入 shape，让新缓冲区的其余维度和原来的保持一致。
  for (auto dim : bufferType.getShape()) {
    shape.push_back(dim);
  }
<<<<<<< HEAD
  // llvm::outs() << "test31\n";llvm::outs().flush();
=======
>>>>>>> dev_bizefeng
  // 基于新的 shape 和原先相同的元素类型（ElementType）、地址空间（MemorySpace）等，创建一个新的 MemRefType，这就是我们的「双缓冲」类型。
  auto newBufferType = mlir::MemRefType::get(
    shape, bufferType.getElementType(), {}, bufferType.getMemorySpaceAsInt());
  // 用于保存如果是 alloca 的结果。
<<<<<<< HEAD
  // llvm::outs() << "test4\n";llvm::outs().flush();
=======
>>>>>>> dev_bizefeng
  mlir::memref::AllocaOp allocRegistOp;
  // 用于保存 alloc 的结果
  mlir::memref::AllocOp allocOp;
  // isAllocRegist 用于标识到底是 alloca 还是 alloc
  bool isAllocRegist = false;
  // defineBufferOp 用来记录原始的 buffer 定义操作
  mlir::Operation* defineBufferOp = nullptr;
  // mlir::OpBuilder* builder = nullptr;
  // builder 是一个 OpBuilder 的智能指针，用来在 IR 中插入新的操作
  std::shared_ptr<mlir::OpBuilder> builder = nullptr;

  /*
  这里分三种情况处理 buffer 的定义操作：
  如果 buffer 来自 memref.alloc，就用 OpBuilder 再创建一个同样的 allocOp，但类型是 newBufferType（即带有双缓冲维度的类型）。
  如果 buffer 来自 memref.alloca，就用 OpBuilder 创建一个 AllocaOp 来分配一个更大的栈上缓冲区，并设置对齐方式。这里设置了 KCG_ALIGNBYTE。
  否则就打印一下操作名，说明并不支持或没有匹配到，代表报错。
  */
<<<<<<< HEAD
  //llvm::outs() << "test5\n";llvm::outs().flush();
=======
>>>>>>> dev_bizefeng
  if(mlir::dyn_cast<mlir::memref::AllocOp>(buffer.getDefiningOp()) != nullptr){
    defineBufferOp = mlir::dyn_cast<mlir::memref::AllocOp>(buffer.getDefiningOp());
    // mlir::OpBuilder builder(defineBufferOp);
    builder = std::make_shared<mlir::OpBuilder>(defineBufferOp);
    allocOp = builder->create<mlir::memref::AllocOp>(builder->getUnknownLoc(), newBufferType);
    allocOp.setAlignment(KCG_ALIGNBYTE);
    tools::_opSetDescription(allocOp->getResult(0).getDefiningOp(), bufferName);
  }
  else if(mlir::dyn_cast<mlir::memref::AllocaOp>(buffer.getDefiningOp()) != nullptr){
    defineBufferOp = mlir::dyn_cast<mlir::memref::AllocaOp>(buffer.getDefiningOp());
    builder = std::make_shared<mlir::OpBuilder>(defineBufferOp);
    allocRegistOp = builder->create<mlir::memref::AllocaOp>(builder->getUnknownLoc(), newBufferType);
    allocRegistOp.setAlignment(KCG_ALIGNBYTE);
    isAllocRegist = true;
    tools::_opSetDescription(allocRegistOp->getResult(0).getDefiningOp(), bufferName);
  }
  else{
    llvm::outs() << "[D] OpName = "<< buffer.getDefiningOp()->getName().getStringRef();
    llvm::outs().flush();
  }
<<<<<<< HEAD
  //llvm::outs() << "test6\n";llvm::outs().flush();
=======
>>>>>>> dev_bizefeng
  // 确保 builder 不为空，否则报错
  assert(builder != nullptr && "PipeLineNullptrError");
  // auto doubleBuffer = allocOp.getResult();
  /*
  根据前面判断出来的 isAllocRegist 标识，来获取最终的双缓冲 Value：
  如果是 alloca，则从 allocRegistOp 拿结果；
  否则就从 allocOp 拿结果。
  */
  mlir::TypedValue<mlir::MemRefType> doubleBuffer;
  if(isAllocRegist){
    doubleBuffer = allocRegistOp.getResult();
  }
  else{
    doubleBuffer = allocOp.getResult();
  }
  //llvm::outs() << "test7\n";llvm::outs().flush();

  /* step2: prefetch before the loop.*/
  //1. replace every use of compute_at's inductionvar with compute_at'lb.
  // 用来遍历 AffineForOp body 内部所有的 AffineVectorLoadOp 和 AffineVectorStoreOp，如果它们的索引用到了 src，就替换成 dst
  auto replaceOperand = [&](mlir::affine::AffineForOp body, mlir::Value src, mlir::Value dst) {
    // 处理 load
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineVectorLoadOp load) {
      auto oldOperands = load.getMapOperands();
      mlir::SmallVector<mlir::Value> operands;
      bool needReplace = false;
      for (auto operand : oldOperands) {
        if (operand == src) {
          needReplace = true;
          operands.push_back(dst);
        } else {
          operands.push_back(operand);
        }
      }
      if (!needReplace) return;
      mlir::OpBuilder builder(load);
      auto newVectorLoadOp = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), load.getVectorType(), load.getMemref(), load.getAffineMap(), operands);
      load.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
      load.erase();
    });
    // 处理 store
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineVectorStoreOp store) {
      auto oldOperands = store.getMapOperands();
      mlir::SmallVector<mlir::Value> operands;
      bool needReplace = false;
      for (auto operand : oldOperands) {
        if (operand == src) {
          needReplace = true;
          operands.push_back(dst);
        } else {
          operands.push_back(operand);
        }
      }
      if (!needReplace) return;
      mlir::OpBuilder builder(store);
      auto newVectorStoreOp = builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), store.getValue(), store.getMemref(), store.getAffineMap(), operands);
      store.erase();
    });
  };
  //llvm::outs() << "test7\n";llvm::outs().flush();
  //2. replace every reference to buffer with doubleBuffer, and select doubleBuffer[0];
  // 用于将对 bufferSrc 的访问替换为对 bufferDst 的访问，并且在索引最前面加一个常量维度 0，来访问 doubleBuffer[0, ...]
  auto replaceBufferRef = [&](mlir::affine::AffineForOp body, mlir::Value bufferSrc, mlir::Value bufferDst) {
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineVectorLoadOp load) {
      auto oldMemref = load.getMemref();
      if (oldMemref != bufferSrc) return;

      auto oldAffineMap = load.getAffineMap();
      auto oldExprs = oldAffineMap.getResults();
      mlir::SmallVector<mlir::AffineExpr> exprs;
      exprs.push_back(mlir::getAffineConstantExpr(0, body->getContext()));
      for (auto expr : oldExprs) exprs.push_back(expr);
      auto map = mlir::AffineMap::get(/*dimCount*/oldAffineMap.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), body->getContext());

      mlir::OpBuilder builder(load);
      auto newVectorLoadOp = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), load.getVectorType(), bufferDst, map, load.getMapOperands());
      load.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
      load.erase();
    });
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineVectorStoreOp store) {
      auto oldMemref = store.getMemref();
      if (oldMemref != bufferSrc) return;

      auto oldAffineMap = store.getAffineMap();
      auto oldExprs = oldAffineMap.getResults();
      mlir::SmallVector<mlir::AffineExpr> exprs;
      exprs.push_back(mlir::getAffineConstantExpr(0, body->getContext()));
      for (auto expr : oldExprs) exprs.push_back(expr);
      auto map = mlir::AffineMap::get(/*dimCount*/oldAffineMap.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), body->getContext());

      mlir::OpBuilder builder(store);
      auto newVectorStoreOp = builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), store.getValue(), bufferDst, map, store.getMapOperands());
      store.erase();
    });
  };
<<<<<<< HEAD
  //llvm::outs() << "test8\n";llvm::outs().flush();
=======
>>>>>>> dev_bizefeng
  // 用来保存克隆出来的新循环。
  std::vector<mlir::affine::AffineForOp> result;
  // 把 builder 的插入点设置到 compute_at 这个循环的地方
  builder->setInsertionPoint(compute_at);
  // 创建一个 ConstantIndexOp，值是 compute_at 的下界，用来替代原先循环里对迭代变量的引用，使得克隆出来的循环在主循环开始之前就固定在下界处执行一次
  auto lbOp = builder->create<mlir::arith::ConstantIndexOp>(builder->getUnknownLoc(), compute_at.getConstantLowerBound());
  // 找到 compute_at 所在的 最外层循环
<<<<<<< HEAD
  auto rootLoop = Analyzer::findRootLoop(compute_at);
  // 把 lbOp 这个操作移动到 rootLoop 最开始的位置，保证这个常量在后续插入的操作之前就定义好。
  lbOp->moveBefore(&(rootLoop->getBlock()->getOperations().front()));
 // llvm::outs() << "test9\n";llvm::outs().flush();
=======
  auto rootLoop = findRootLoop(compute_at);
  // 把 lbOp 这个操作移动到 rootLoop 最开始的位置，保证这个常量在后续插入的操作之前就定义好。
  lbOp->moveBefore(&(rootLoop->getBlock()->getOperations().front()));
>>>>>>> dev_bizefeng
  /*
  1 用 builder->clone 拷贝一份新的循环 newBody。
  2 把 newBody 转为 AffineForOp 类型 loopBody。
  3 调用 replaceOperand：将所有对 compute_at.getInductionVar() 的引用替换为 lbOp.getResult()，即固定这个循环在“下界”位置。
  4 调用 replaceBufferRef：将所有对原 buffer 的读写替换为 doubleBuffer[0]。
  5 把新的 loopBody 放进 result。
  */
  for (auto readBody : readBodys) {
    mlir::IRMapping mapper;
    auto newBody = builder->clone(*readBody, mapper);
    auto loopBody = mlir::dyn_cast<mlir::affine::AffineForOp>(newBody);
    replaceOperand(loopBody, compute_at.getInductionVar(), lbOp.getResult());
    replaceBufferRef(loopBody, buffer, doubleBuffer);
    result.push_back(loopBody);
  }
<<<<<<< HEAD
  //llvm::outs() << "test10\n";llvm::outs().flush();
=======
>>>>>>> dev_bizefeng
  // 新生成的循环，以及原先的循环，都放进 results 中
  results.push_back(result);
  results.push_back(readBodys);


  // 循环内部的预取
  /* step3: prefetch in the main loop*/
  //1. create the affine.if to check if we can prefetch
  // 获取两个 AffineDimExpr，在后面构造 AffineMap 或 IntegerSet 时会用到
  auto dim0 = builder->getAffineDimExpr(0);
  auto dim1 = builder->getAffineDimExpr(1);
  // 提取主循环的步长 step、上界 ub、下界 lb，方便后面做边界判断。
  int64_t step = compute_at.getStep().getLimitedValue();
  int64_t ub = compute_at.getConstantUpperBound();
  int64_t lb = compute_at.getConstantLowerBound();

  /*
  /// Array of affine constraints: a constraint is either an equality
  /// (affine_expr == 0) or an inequality (affine_expr >= 0).
  ArrayRef<AffineExpr> constraints;

  // Bits to check whether a constraint is an equality or an inequality.
  ArrayRef<bool> eqFlags;
  */
  // 构造一个整数集合（IntegerSet），表示 ub - 2*step - iv >= 0，即“如果当前迭代变量 + 2 * step 还没越过上界，那就可以预取下一批数据”
  llvm::SmallVector<mlir::AffineExpr> exprs;
  llvm::SmallVector<bool> eqFlags;
  // iv + 2 * step <= ub
  //-> ub - 2 * step - iv >= 0
  exprs.push_back(ub - 2 * step - dim0);
  eqFlags.push_back(false);
  //llvm::outs() << "test11\n";llvm::outs().flush();
  auto cst = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), llvm::ArrayRef<bool>(eqFlags));

  // 把插入点放到 compute_at 主循环的开头
  builder->setInsertionPointToStart(compute_at.getBody());
  // 创建一个 AffineIfOp，其条件是上面的 cst，并把 compute_at.getInductionVar() 当作它的实参（用于判断 iv + 2*step <= ub）
  auto ifOp = builder->create<mlir::affine::AffineIfOp>(builder->getUnknownLoc(), cst, mlir::ValueRange{compute_at.getInductionVar()}, 
                                               /*withElseRegion=*/false);// withElseRegion=false 表示只生成 then 块，没有 else 块
  // 把插入点移动到 ifOp 的 then 块开头，后续创建操作就会插入到这个区域
  builder->setInsertionPointToStart(ifOp.getThenBlock());
<<<<<<< HEAD
  //llvm::outs() << "test12\n";llvm::outs().flush();
=======

>>>>>>> dev_bizefeng
  // 拷贝一份 readBodys 并反转顺序，然后把每个 readBody 的操作节点 splice 到 ifOp 的 then 区块里。
  auto reverseReadBodys = readBodys;
  std::reverse(reverseReadBodys.begin(), reverseReadBodys.end());
  for (auto readBody : reverseReadBodys) {
    ifOp.getBody()->getOperations().splice(ifOp.getBody()->begin(),
                    readBody->getBlock()->getOperations(), mlir::Block::iterator(readBody));//only the readBody.
  }// 
<<<<<<< HEAD
  //llvm::outs() << "test13\n";llvm::outs().flush();
=======
>>>>>>> dev_bizefeng
  // 2. replace 
  // 如果索引中出现了 src，就把它替换为 dstExpr。常见做法是 (iv -> iv + step) 或类似变换
  auto replaceAffineExprInLoop = [&](mlir::affine::AffineForOp body, mlir::Value src, mlir::AffineExpr dstExpr, int dimCount) {
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineVectorLoadOp load) {
      auto operands = load.getMapOperands();
      bool needReplace = false;
      int targetDim = -1;
      for (auto operand : operands) {
        if (!needReplace) targetDim += 1;
        if (operand == src) {
          needReplace = true;
          break;
        }
      }
      if (!needReplace) return;
      auto shiftedDstExpr = shiftAffineExprDim(body->getContext(), dstExpr, targetDim);
      llvm::SmallVector<mlir::AffineExpr> exprs;
      auto oldExprs = load.getAffineMap().getResults();
      for (auto oldExpr : oldExprs) {
        auto expr = getModifiedExpr(body->getContext(), oldExpr, shiftedDstExpr, targetDim, dimCount);
        exprs.push_back(expr);
      }
      auto map = mlir::AffineMap::get(/*dimCount*/load.getAffineMap().getNumDims() + dimCount - 1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), body->getContext());
      mlir::OpBuilder builder(load);
      auto newVectorLoadOp = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), load.getVectorType(), load.getMemref(), map, operands);
      load.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
      load.erase();
    });
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineVectorStoreOp store) {
      auto operands = store.getMapOperands();
      bool needReplace = false;
      int targetDim = -1;
      for (auto operand : operands) {
        if (!needReplace) targetDim += 1;
        if (operand == src) {
          needReplace = true;
          break;
        }
      }
      if (!needReplace) return;
      auto shiftedDstExpr = shiftAffineExprDim(body->getContext(), dstExpr, targetDim);
      llvm::SmallVector<mlir::AffineExpr> exprs;
      auto oldExprs = store.getAffineMap().getResults();
      for (auto oldExpr : oldExprs) {
        auto expr = getModifiedExpr(body->getContext(), oldExpr, shiftedDstExpr, targetDim, dimCount);
        exprs.push_back(expr);
      }
      auto map = mlir::AffineMap::get(/*dimCount*/store.getAffineMap().getNumDims() + dimCount - 1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), body->getContext());
      mlir::OpBuilder builder(store);
      auto newVectorStoreOp = builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), store.getValue(), store.getMemref(), map, operands);
      store.erase();
    });
  };
  //llvm::outs() << "test14\n";llvm::outs().flush();
  // 3.replace every reference to buffer with doubleBuffer, and select doubleBuffer[0];
  // 用于将所有对 bufferSrc 的读写替换为对 bufferDst（双缓冲）的访问，并且根据主循环迭代变量 iv 来决定是 [0] 还是 [1] 片
  auto replaceBufferRefInLoop = [&](mlir::affine::AffineForOp body, mlir::Value bufferSrc, mlir::Value bufferDst, mlir::affine::AffineForOp compute_at) {
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineVectorLoadOp load) {
      auto oldMemref = load.getMemref();
      if (oldMemref != bufferSrc) return;

      int targetDim = -1;
      int additionDim = 0;
      bool existInductionVar = false;
      for (auto operand : load.getMapOperands()) {
        targetDim += 1;
        if (operand == compute_at.getInductionVar()) {
          existInductionVar = true;
          break;
        }
      }
      // reference to double buffer must depend on the iteration var.
      llvm::SmallVector<mlir::Value> operands;
      for (auto operand : load.getMapOperands()) operands.push_back(operand);
      if (!existInductionVar) {
        operands.push_back(compute_at.getInductionVar());
        targetDim += 1;
        additionDim += 1;
      }
      auto dim = mlir::getAffineDimExpr(targetDim, body->getContext());
      mlir::SmallVector<mlir::AffineExpr> exprs;
      exprs.push_back((dim.floorDiv(compute_at.getStep().getLimitedValue()) + 1) % 2);
      auto oldAffineMap = load.getAffineMap();
      auto oldExprs = oldAffineMap.getResults();
      for (auto expr : oldExprs) exprs.push_back(expr);
      auto map = mlir::AffineMap::get(/*dimCount*/oldAffineMap.getNumDims() + additionDim, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), body->getContext());

      mlir::OpBuilder builder(load);
      auto newVectorLoadOp = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), load.getVectorType(), bufferDst, map, operands);
      load.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
      load.erase();
    });
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineVectorStoreOp store) {
      auto oldMemref = store.getMemref();
      if (oldMemref != bufferSrc) return;

      int targetDim = -1;
      int additionDim = 0;
      bool existInductionVar = false;
      for (auto operand : store.getMapOperands()) {
        targetDim += 1;
        if (operand == compute_at.getInductionVar()) {
          existInductionVar = true;
          break;
        }
      }
      // reference to double buffer must depend on the iteration var.
      llvm::SmallVector<mlir::Value> operands;
      for (auto operand : store.getMapOperands()) operands.push_back(operand);
      if (!existInductionVar) {
        operands.push_back(compute_at.getInductionVar());
        targetDim += 1;
        additionDim += 1;
      }
      auto dim = mlir::getAffineDimExpr(targetDim, body->getContext());
      mlir::SmallVector<mlir::AffineExpr> exprs;
      exprs.push_back((dim.floorDiv(compute_at.getStep().getLimitedValue()) + 1) % 2);
      auto oldAffineMap = store.getAffineMap();
      auto oldExprs = oldAffineMap.getResults();
      for (auto expr : oldExprs) exprs.push_back(expr);
      auto map = mlir::AffineMap::get(/*dimCount*/oldAffineMap.getNumDims() + additionDim, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), body->getContext());

      mlir::OpBuilder builder(store);
      auto newVectorStoreOp = builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), store.getValue(), bufferDst, map, operands);
      store.erase();
    });
  };
<<<<<<< HEAD
  //llvm::outs() << "test15\n";llvm::outs().flush();
=======
>>>>>>> dev_bizefeng
  /*
  对 readBodys 中每个循环：
  1 调整循环里的访存索引，让它们在主循环内读取“下一批”（iv + step）。
  2 替换对原 buffer 的引用，让它们改为写/读 doubleBuffer[(iv + 1) % 2]
  */
  for (auto readBody : readBodys) {
    auto dim0 = builder->getAffineDimExpr(0);
    replaceAffineExprInLoop(readBody, compute_at.getInductionVar(), dim0 + compute_at.getStep().getLimitedValue(), 1);
    replaceBufferRefInLoop(readBody, buffer, doubleBuffer, compute_at);
  }
  //4. replace load
  // 找到 buffer 的所有使用者，遍历检查是否是 AffineVectorLoadOp 或 AffineLoadOp; 根据 compute_at.getInductionVar() 把访问改为访问 doubleBuffer 的 [i % 2] 或 [ (i.floorDiv(step) ) % 2 ]
  auto users = buffer.getUsers();
  for (auto user : users) {
    // must be load Op
    // 判断是否是一个 AffineVectorLoadOp（即向量加载操作）
    if (auto load = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(user)) {
      // 检查 load.getMemref() 是否就是我们的 buffer
      auto oldMemref = load.getMemref();
      if (oldMemref != buffer) assert(false);

      int targetDim = -1;
      int additionDim = 0;
      bool existInductionVar = false;
      // 遍历 load.getMapOperands()（即加载操作使用的索引操作数），如果在其中找到 compute_at.getInductionVar()，则说明这次加载已经和主循环迭代变量关联
      for (auto operand : load.getMapOperands()) {
        targetDim += 1; // targetDim 记录“在哪个维度”上找到了迭代变量
        if (operand == compute_at.getInductionVar()) {
          existInductionVar = true;
          break;
        }
      }
      // reference to double buffer must depend on the iteration var.
      llvm::SmallVector<mlir::Value> operands;
      for (auto operand : load.getMapOperands()) operands.push_back(operand);
      // 如果原本没有迭代变量（existInductionVar == false），就把 compute_at.getInductionVar() 手动加进去，targetDim 和 additionDim 也随之各加 1, 保证“对双缓冲的访问一定包含主循环的迭代变量”，从而能在运行时根据 i % 2 来选 [0] 或 [1]
      if (!existInductionVar) {
        operands.push_back(compute_at.getInductionVar());
        targetDim += 1;
        additionDim += 1;
      }
      auto dim = mlir::getAffineDimExpr(targetDim, load->getContext());
      // 构造一个新的表达式向量 exprs, 新增选择双缓冲哪一片
      mlir::SmallVector<mlir::AffineExpr> exprs;
      exprs.push_back(dim.floorDiv(compute_at.getStep().getLimitedValue()) % 2);
      auto oldAffineMap = load.getAffineMap();
      auto oldExprs = oldAffineMap.getResults();
      // 家如旧的表达式作map第二维三维参数
      for (auto expr : oldExprs) exprs.push_back(expr);
      // 调用 mlir::AffineMap::get(...) 创建一个新的 AffineMap，其中 dimCount 等于 oldAffineMap.getNumDims() + additionDim
      auto map = mlir::AffineMap::get(/*dimCount*/oldAffineMap.getNumDims() + additionDim, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), load->getContext());

      // 创建新的AffineVectorLoadOp
      mlir::OpBuilder builder(load);
      auto newVectorLoadOp = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), load.getVectorType(), doubleBuffer, map, operands);
      load.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
      load.erase();
    } else if (auto load = mlir::dyn_cast<mlir::affine::AffineLoadOp>(user)) { //如果不是 AffineVectorLoadOp，则判断是不是 AffineLoadOp, 逻辑和上面相同
      auto oldMemref = load.getMemref();
      if (oldMemref != buffer) assert(false);

      int targetDim = -1;
      int additionDim = 0;
      bool existInductionVar = false;
      for (auto operand : load.getMapOperands()) {
        targetDim += 1;
        if (operand == compute_at.getInductionVar()) {
          existInductionVar = true;
          break;
        }
      }
      // reference to double buffer must depend on the iteration var.
      llvm::SmallVector<mlir::Value> operands;
      for (auto operand : load.getMapOperands()) operands.push_back(operand);
      if (!existInductionVar) {
        operands.push_back(compute_at.getInductionVar());
        targetDim += 1;
        additionDim += 1;
      }
      auto dim = mlir::getAffineDimExpr(targetDim, load->getContext());
      mlir::SmallVector<mlir::AffineExpr> exprs;
      exprs.push_back(dim.floorDiv(compute_at.getStep().getLimitedValue()) % 2);
      auto oldAffineMap = load.getAffineMap();
      auto oldExprs = oldAffineMap.getResults();
      for (auto expr : oldExprs) exprs.push_back(expr);
      auto map = mlir::AffineMap::get(/*dimCount*/oldAffineMap.getNumDims() + additionDim, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), load->getContext());

      mlir::OpBuilder builder(load);
      auto newVectorLoadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), doubleBuffer, map, operands);
      load.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
      load.erase();
    } else {
      assert(false);
    }
  }
  //llvm::outs() << "test16\n";llvm::outs().flush();

  /* step4: clear work*/
  // 删除原先那个 memref.alloc 或 memref.alloca 操作
  defineBufferOp->erase();
  // 把外部传进来的 buffer 更新为新创建的双缓冲 doubleBuffer
  buffer = doubleBuffer;
  //llvm::outs() << "test17\n";llvm::outs().flush();
  return results;
}


void detach_last_loop(mlir::affine::AffineForOp forOp) {
  auto step = forOp.getStep().getLimitedValue();
  auto ub = forOp.getConstantUpperBound();
  forOp.setConstantUpperBound(ub - step);

  auto builder = getBuilder(forOp, Position::after);
  auto replaceInducetionVar = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), ub - step);
  auto rootLoop = Analyzer::findRootLoop(forOp);
  replaceInducetionVar->moveBefore(&(rootLoop->getBlock()->getOperations().front()));
  mlir::IRMapping mapper;
  auto newBody = builder.clone(*forOp, mapper);
  auto loopBody = mlir::dyn_cast<mlir::affine::AffineForOp>(newBody);
  loopBody.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    auto oldOperands = op->getOperands();
    llvm::SmallVector<mlir::Value> operands;
    for (auto operand : oldOperands) {
      if (operand == loopBody.getInductionVar()) {
        operands.push_back(replaceInducetionVar.getResult());
      } else {
        operands.push_back(operand);
      }
    }
    op->setOperands(operands);
  });

  loopBody.getBody()->getOperations().back().erase();
  loopBody->getBlock()->getOperations().splice( 
    mlir::Block::iterator(loopBody),
    loopBody.getBody()->getOperations());
  loopBody.erase();

}

void schedule(mlir::Operation* srcOp, mlir::Operation* dstOp, Position pos) {
  mlir::OpBuilder builder(dstOp->getContext());
  switch (pos) {
    case Position::after: {
      builder.setInsertionPointAfter(dstOp);
      srcOp->moveAfter(dstOp);
      break;
    }
    case Position::before: {
      builder.setInsertionPoint(dstOp);
      srcOp->moveBefore(dstOp);
      break;
    }
    case Position::end: {
      if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(dstOp)) {
        srcOp->moveBefore(&(forOp.getBody()->getOperations().back()));
      } else {
        assert(false);
      }
      break;
    }
    case Position::begin: {
      if (auto forOp = mlir::dyn_cast<mlir::affine::AffineParallelOp>(dstOp)) {
        srcOp->moveBefore(&(forOp.getBody()->getOperations().front()));
      } else if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(dstOp)) {
        srcOp->moveBefore(&(forOp.getBody()->getOperations().front()));
      } else {
        assert(false);
      }
      break;
    }
    default:
      assert(false);
  }

}

void replaceOperands(mlir::Operation* op, mlir::Value src, mlir::Value dst) {
  auto oldOperands = op->getOperands();
  llvm::SmallVector<mlir::Value> operands;
  for (auto operand : oldOperands) {
    if (operand == src) {
      operands.push_back(dst);
    } else {
      operands.push_back(operand);
    }
  }
  op->setOperands(operands);

  if (op->getRegions().size() != 0) {
    auto& blocks = op->getRegions().front().getBlocks();
    for (auto& block : blocks) {
      auto& ops = block.getOperations();
      for (auto& op : ops) {
        replaceOperands(&op, src, dst);
      }
    }
  }
}

void extract_loop(mlir::Operation* srcOp, mlir::affine::AffineForOp forOp, int64_t iteration) {
  mlir::OpBuilder builder(forOp->getContext());
  builder.setInsertionPoint(forOp);
  mlir::IRMapping mapper;
  auto clonedOp = builder.clone(*srcOp, mapper);

  int64_t step = forOp.getStep().getLimitedValue();
  int64_t ub = forOp.getConstantUpperBound();
  int64_t lb = forOp.getConstantLowerBound();

  auto index = lb + iteration * step;
  auto replaceVar = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), index);

  auto rootLoop = Analyzer::findRootLoop(forOp);
  replaceVar->moveBefore(&(rootLoop->getBlock()->getOperations().front()));

  replaceOperands(clonedOp, forOp.getInductionVar(), replaceVar.getResult());

}

std::pair<bool, int64_t> getMaxValue(mlir::Value value) {
  mlir::Operation* op;
  if (auto blockArgument = value.dyn_cast<mlir::BlockArgument>()) {
    op = blockArgument.getOwner()->getParentOp();
  } else {
    op = value.getDefiningOp();
  }
  auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(op);
  std::pair<bool, int64_t> result;
  if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(op)) {
    result.first = true;
    result.second = constOp.value();
    // result.second = constOp.getValue().cast<mlir::IntegerAttr>().getInt();
  } else if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
    if (!forOp.hasConstantBounds()) {
      result.first = false;
    } else {
      result.first = true;
      result.second = forOp.getConstantUpperBound() - 1;
    }
  } else {
    llvm::errs() << "Append new op type here.";
    assert(false);
  }
  return result;
}

std::pair<bool, int64_t> getMinValue(mlir::Value value) {
  mlir::Operation* op;
  if (auto blockArgument = value.dyn_cast<mlir::BlockArgument>()) {
    op = blockArgument.getOwner()->getParentOp();
  } else {
    op = value.getDefiningOp();
  }
  std::pair<bool, int64_t> result;
  if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(op)) {
    result.first = true;
    result.second = constOp.value();
  } else if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
    if (!forOp.hasConstantBounds()) {
      result.first = false;
    } else {
      result.first = true;
      result.second = forOp.getConstantLowerBound();
    }
  } else {
    llvm::errs() << "Append new op type here.";
    assert(false);
  }
  return result;
}

int64_t eval(mlir::AffineExpr expr, std::vector<int64_t> values) {
  if (auto dimExpr = expr.dyn_cast<mlir::AffineDimExpr>()) {
    return values[dimExpr.getPosition()];
  }
  if (auto constExpr = expr.dyn_cast<mlir::AffineConstantExpr>()) {
    return constExpr.getValue();
  }
  auto binaryExpr = expr.dyn_cast<mlir::AffineBinaryOpExpr>();
  assert(binaryExpr);
  auto lhs = eval(binaryExpr.getLHS(), values);
  auto rhs = eval(binaryExpr.getRHS(), values);
  switch (binaryExpr.getKind()) {
    case mlir::AffineExprKind::Add: return lhs + rhs;
    case mlir::AffineExprKind::CeilDiv: return (lhs + rhs - 1) / rhs;
    case mlir::AffineExprKind::FloorDiv: return lhs / rhs;
    case mlir::AffineExprKind::Mod: return lhs % rhs;
    case mlir::AffineExprKind::Mul: return lhs * rhs;
    default: assert(false);
  }
}

struct TakeOffTrueIf : public mlir::PassWrapper<TakeOffTrueIf, mlir::OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TakeOffTrueIf)
   TakeOffTrueIf() = default;
   void runOnOperation() override {
     auto module = getOperation();
     module.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineIfOp ifOp) {
      bool result = true;
      auto iset = ifOp.getIntegerSet();
      auto operands = ifOp->getOperands();

      int constraintNum = iset.getNumConstraints();
      std::vector<int64_t> maxValues;
      std::vector<int64_t> minValues;

      for (auto operand : operands) {
        auto maxValue = getMaxValue(operand);
        if (maxValue.first) {
          maxValues.push_back(maxValue.second);
        } else {
          //can't deduction
          return;
        }
        auto minValue = getMinValue(operand);
        if (minValue.first) {
          minValues.push_back(minValue.second);
        } else {
          //can't deduction
          return;
        }
      }
      for (int i = 0; i < constraintNum; i++) {
        auto expr = iset.getConstraint(i);
        auto isEq = iset.isEq(i);
        if (isEq) {
          if (eval(expr, maxValues) != 0 | eval(expr, minValues) != 0) {
            result = false;
            break;
          }
        } else {
          if (eval(expr, maxValues) < 0 | eval(expr, minValues) < 0) {
            result = false;
            break;
          }
        }
      }
      if (result) {
        ifOp.getBody()->getOperations().back().erase();
        ifOp->getBlock()->getOperations().splice(
          mlir::Block::iterator(ifOp),
          ifOp.getBody()->getOperations());
        ifOp.erase();
      }
     });
   }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> TakeOffTrueIfPass() {
  return std::make_unique<TakeOffTrueIf>();
}

void take_off_true_if(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  pm.addPass(TakeOffTrueIfPass());
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Take off the true if failed.";
  }
  return;
}

struct DeleteFalseIf : public mlir::PassWrapper<DeleteFalseIf, mlir::OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeleteFalseIf)
   DeleteFalseIf() = default;
   void runOnOperation() override {
     auto module = getOperation();
     module.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineIfOp ifOp) {
      auto iset = ifOp.getIntegerSet();
      auto operands = ifOp->getOperands();

      int constraintNum = iset.getNumConstraints();
      std::vector<int64_t> maxValues;
      std::vector<int64_t> minValues;

      for (auto operand : operands) {
        auto maxValue = getMaxValue(operand);
        if (maxValue.first) {
          maxValues.push_back(maxValue.second);
        } else {
          //can't deduction
          return;
        }
        auto minValue = getMinValue(operand);
        if (minValue.first) {
          minValues.push_back(minValue.second);
        } else {
          //can't deduction
          return;
        }
      }
      int64_t count = 0;
      for (int i = 0; i < constraintNum; i++) {
        auto expr = iset.getConstraint(i);
        auto isEq = iset.isEq(i);
        ///TODO:need to verify all the case of all inputs.
        if (isEq) {
          if (eval(expr, maxValues) != 0 && eval(expr, minValues) != 0) {
            count += 1;
          }
        } else {
          if (eval(expr, maxValues) < 0 && eval(expr, minValues) < 0) {
            count += 1;
          }
        }
      }
      if (count == constraintNum) {
        // delete the entile body of if operaiton.
        ifOp.erase();
      }
     });
   }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> DeleteFalseIfPass() {
  return std::make_unique<DeleteFalseIf>();
}

void delete_false_if(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  pm.addPass(DeleteFalseIfPass());
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Delete false if failed.";
  }
  return;
}


struct UnrollAffineFor : public mlir::PassWrapper<UnrollAffineFor, mlir::OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnrollAffineFor)
   UnrollAffineFor(mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn_ = nullptr) : unrollCheckFn(unrollCheckFn_) {};
   void runOnOperation() override {

     auto module = getOperation();
     module.walk<mlir::WalkOrder::PostOrder>([&](mlir::affine::AffineForOp forOp) {
      if (!unrollCheckFn(forOp)) return;

      auto rootLoop = Analyzer::findRootLoop(forOp);
      auto& allOps = rootLoop->getBlock()->getOperations();

      auto findConstValue = [&](int64_t value)->mlir::Value {
        auto curIter = allOps.begin();
        while (true) {
          auto constOp = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(*curIter);
          if (!constOp) break;
          if (value == constOp.value()) {
            return constOp.getResult();
          }
          curIter++;
        }
        return nullptr;
      };

      mlir::OpBuilder builder(forOp);

      for (auto index = forOp.getConstantLowerBound(); index < forOp.getConstantUpperBound(); index += forOp.getStep().getLimitedValue()) {
        auto iterVarReplace = findConstValue(index);
        if (!iterVarReplace) {
          auto constOp = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), index);
          constOp->moveBefore(&(rootLoop->getBlock()->getOperations().front()));
          iterVarReplace = constOp.getResult();
        }
        mlir::IRMapping mapper;
        auto cloned = builder.clone(*forOp, mapper);
        auto clonedForOp = mlir::dyn_cast<mlir::affine::AffineForOp>(cloned);
        clonedForOp.getBody()->getOperations().back().erase();
        clonedForOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
          auto oldOperands = op->getOperands();
          llvm::SmallVector<mlir::Value> operands;
          for (auto operand : oldOperands) {
            if (operand == clonedForOp.getInductionVar()) {
              operands.push_back(iterVarReplace);
            } else {
              operands.push_back(operand);
            }
          }
          op->setOperands(operands);
        });
        clonedForOp->getBlock()->getOperations().splice(
          mlir::Block::iterator(clonedForOp),
          clonedForOp.getBody()->getOperations());
        clonedForOp.erase();
      }
      forOp.erase();
     });
   }
  mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn;
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> UnrollAffineForPass(mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn = nullptr) {
  return std::make_unique<UnrollAffineFor>(unrollCheckFn);
}

void unroll(mlir::ModuleOp module, mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn) {

  mlir::PassManager pm(module.getContext());
  pm.addPass(UnrollAffineForPass(unrollCheckFn));
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Unroll affine for failed.";
  }
  return;
}

struct UnrollAttribute : public mlir::PassWrapper<UnrollAttribute, mlir::OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnrollAttribute)
   UnrollAttribute(mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn_ = nullptr) : unrollCheckFn(unrollCheckFn_) {};
   void runOnOperation() override {

     auto module = getOperation();
     module.walk<mlir::WalkOrder::PostOrder>([&](mlir::affine::AffineForOp forOp) {
      if (!unrollCheckFn(forOp)) return;
      mlir::OpBuilder builder(forOp->getContext());
      forOp->setAttr(std::string("affine.loop"), builder.getStringAttr("unroll"));
     });
   }
  mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn;
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> UnrollAttributePass(mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn = nullptr) {
  return std::make_unique<UnrollAttribute>(unrollCheckFn);
}

void unrollAttribute(mlir::ModuleOp module, mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn) {

  mlir::PassManager pm(module.getContext());
  pm.addPass(UnrollAttributePass(unrollCheckFn));
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Unroll affine for failed.";
  }
  return;
}

void change_double_buffer(mlir::affine::AffineForOp scope, mlir::Value buffer) {
  scope.walk<mlir::WalkOrder::PostOrder>([&](mlir::affine::AffineVectorLoadOp load) {
    auto mem = load.getMemref();
    if (mem == buffer) {
      auto builder = mlir::OpBuilder(load);
      auto vecT = load.getVectorType();
      auto oldMap = load.getAffineMap();
      auto operands = load.getMapOperands();
      auto oldExprs = oldMap.getResults();
      llvm::SmallVector<mlir::AffineExpr> exprs;
      for (int i = 0; i < oldExprs.size(); i++) {
        if (i == 0) {
          auto binaryExpr = oldExprs[i].dyn_cast<mlir::AffineBinaryOpExpr>();
          assert(binaryExpr && binaryExpr.getKind() == mlir::AffineExprKind::Mod);
          auto constExpr = binaryExpr.getRHS().dyn_cast<mlir::AffineConstantExpr>();
          assert(constExpr && constExpr.getValue() == 2);
          exprs.push_back((binaryExpr.getLHS() + 1) % 2);
        } else {
          exprs.push_back(oldExprs[i]);
        }
      }
      auto map = mlir::AffineMap::get(/*dimCount*/oldMap.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), load->getContext());
      auto ld = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vecT, buffer, map, operands);
      load.getResult().replaceAllUsesWith(ld.getResult());
      load.erase();
    }
  });
  ///TODO: support more operations for change double buffer.
  
}

} // rewriter
} // kernelcodegen