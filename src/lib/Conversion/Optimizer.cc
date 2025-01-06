#include "Conversion/Optimizer.h"
#include <cfloat>
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include <filesystem>

namespace KernelCodeGen {

bool MatmulOptimizer::applicable(mlir::ModuleOp& module) {
  clear();
  auto&& matmulFuncs = Analyzer::collectFunctions(module, "Matmul");
  if (!matmulFuncs.size()) {
    llvm::errs() << "Optimization failed: No find Matmul funcOp.\n";
    return false;
  }

  for (auto& matmulFunc : matmulFuncs) {
    if (matmuls.count(matmulFunc) != 0 || matmulLoops.count(matmulFunc) != 0
      || matmulBuffers.count(matmulFunc) != 0) {
      llvm::errs() << "Optimization failed: Duplicated Matmul in module\n";
      return false;
    }

    matmuls.insert(matmulFunc);
    auto&& loops = Analyzer::collectFuncLoops(matmulFunc);
    matmulLoops[matmulFunc] = std::move(loops);
    auto funcArgs = matmulFunc.front().getArguments();

    MemoryBuffer ABC;
    ABC.A = funcArgs[0];
    ABC.B = funcArgs[1];
    ABC.C = funcArgs[2];
    matmulBuffers[matmulFunc] = ABC;
  }
  return true;
}


mlir::AffineMap MatmulOptimizer::getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder, std::map<std::string, int> config) {
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  auto dim2 = builder.getAffineDimExpr(2);
  auto dim3 = builder.getAffineDimExpr(3);
  auto dim4 = builder.getAffineDimExpr(4);
  auto dim5 = builder.getAffineDimExpr(5);
  auto dim6 = builder.getAffineDimExpr(6);
  auto dim7 = builder.getAffineDimExpr(7);

  const int& BM = config[KEY_BLOCK_SIZE_M]; 
  const int& BN = config[KEY_BLOCK_SIZE_N]; 
  const int& BK = config[KEY_BLOCK_SIZE_K];
  const int& TM = config[KEY_THREAD_SIZE_M];
  const int& TN = config[KEY_THREAD_SIZE_N];


  if (mapIdentifier == "loadTileA") {
    
  } else if (mapIdentifier == "loadTileB") {
    
  } else if (mapIdentifier == "storeTileA") {
    
  } else if (mapIdentifier == "storeTileB") {
    
  } else if (mapIdentifier == "loadFragA") {
    
  } else if (mapIdentifier == "loadFragB") {
    
  } else if (mapIdentifier == "cacheReadA" || mapIdentifier == "cacheReadB") {
    
  } else if (mapIdentifier == "cacheWriteC") {
    
  } else {
    assert(false);
  }
}


void MatmulOptimizer::applyOptimzer(mlir::ModuleOp& module, std::map<std::string, int> config) {
  mlir::OpBuilder builder(module);
  for (auto& matmul : matmuls) {
    matmul->setAttr(std::string("func.state"), builder.getStringAttr("gpu"));
    auto loops = matmulLoops[matmul];
    auto loopM = loops[0], loopN = loops[1], loopK = loops[2];
    auto buffers = matmulBuffers[matmul];
    auto A = buffers.A, B = buffers.B, C = buffers.C;
    llvm::outs() << "==== original mlir =====\n";llvm::outs().flush();module.dump();
    auto m_axes = Rewriter::split(loopM, 3, {config["THREAD_SIZE_M"], config["BLOCK_SIZE_M"]});
    auto n_axes = Rewriter::split(loopN, 3, {config["THREAD_SIZE_N"], config["BLOCK_SIZE_N"]});

    auto m_outer = m_axes[0], m_mider = m_axes[1], m_inner = m_axes[2];
    auto n_outer = n_axes[0], n_mider = n_axes[1], n_inner = n_axes[2];

    Rewriter::reorder({m_outer, n_outer, m_mider, n_mider, m_inner, n_inner});
    llvm::outs() << "===== after split & reorder =======\n";llvm::outs().flush(); module.dump();

    auto gridLevel = Rewriter::parallel({m_outer, n_outer});
    auto blockLevel = Rewriter::parallel({m_mider, n_mider});
    LOG_DEBUG("===== after parallel =======\n",module);

    std::vector<mlir::affine::AffineForOp> tileCLoops{m_inner, n_inner};
    auto tileC = Rewriter::bufferizeLoopCarryVar(loopK, tileCLoops);
    LOG_DEBUG("===== after bufferizeLoopCarryVar =======\n",module);

    auto k_axes = Rewriter::split(loopK, 3, {config["LOCAL_SPLIT_U"], config["BLOCK_SIZE_K"]});
    auto k_outer = k_axes[0], k_mider = k_axes[1], k_inner = k_axes[2];
    LOG_DEBUG("===== after split =======\n",module);

    Rewriter::loopToParallelZ(k_inner, blockLevel);
    LOG_DEBUG("===== after loopToParallelZ =======\n",module);
    Rewriter::reorder({k_outer, k_mider, m_inner, n_inner});
    LOG_DEBUG("===== after reorder =======\n",module);

    int64_t blockThreads;
    auto blockDim = Analyzer::getParallelNumber(blockLevel, blockThreads);
    
    // // size of loading from glob to reg
    auto glob_load_total_width_a = config["BLOCK_SIZE_K"] * config["BLOCK_SIZE_M"] / blockThreads;
    auto glob_load_total_width_b = config["BLOCK_SIZE_K"] * config["BLOCK_SIZE_N"] / blockThreads;
    auto elementA = A.getType().dyn_cast<mlir::MemRefType>().getElementType();
    auto elementB = B.getType().dyn_cast<mlir::MemRefType>().getElementType();
    auto regB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {config["THREAD_SIZE_N"]}, elementB);
    auto regA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {config["THREAD_SIZE_M"]}, elementA);

    auto tempB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {glob_load_total_width_b}, elementB);
    auto tempA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {glob_load_total_width_a}, elementA);
    auto smB = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared, {config["BLOCK_SIZE_K"], config["BLOCK_SIZE_N"]}, elementB);
    auto smA = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared, {config["BLOCK_SIZE_K"], config["BLOCK_SIZE_M"]}, elementA);
    LOG_DEBUG("===== before alloc_buffer =======\n",module);

    Rewriter::parallelToOneDim(gridLevel);
    Rewriter::parallelToOneDim(blockLevel);
    LOG_DEBUG("===== before parallelToOneDim =======\n",module);
    
    auto blockIdx = Analyzer::getParallelIdx(gridLevel);
    auto threadIdx = Analyzer::getParallelIdx(blockLevel);
    
    // auto loadTileAMap = getAffineMap("loadTileA", builder, config);
    // auto loadTileA = Rewriter::read(A, tileA, loadTileAMap, {threadIdx[0], threadIdx[1], blockIdx[0], k_outer.getInductionVar()}, 
    //                   (ldgASize < config["VECTORIZE_WIDTH"] ? ldgASize : config["VECTORIZE_WIDTH"]), 
    //                   k_outer, Position::begin);
    // auto loadTileBMap = getAffineMap("loadTileB", builder, config);
    // auto loadTileB = Rewriter::read(B, tileB, loadTileBMap, 
    //                   {threadIdx[0], threadIdx[1], k_outer.getInductionVar(), blockIdx[1]}, 
    //                   (ldgBSize < config["VECTORIZE_WIDTH"] ? ldgBSize : config["VECTORIZE_WIDTH"]), 
    //                   loadTileA, Position::after);
    // // module.dump();
    // LOG_DEBUG("===== shm->temp =======\n",module);

    // auto storeTileAMap = getAffineMap("storeTileA", builder, config);
    // // auto storeTileA = Rewriter::write(tileA, smA, storeTileAMap, {threadIdx[1], threadIdx[0]}, 
    // auto storeTileA = Rewriter::write(tileA, smA, storeTileAMap, {threadIdx[0], threadIdx[1]}, 
    //                     (ldgASize < config["VECTORIZE_WIDTH"] ? ldgASize : config["VECTORIZE_WIDTH"]), 
    //                     loadTileB, Position::after);
    // auto storeTileBMap = getAffineMap("storeTileB", builder, config);
    // auto storeTileB = Rewriter::write(tileB, smB, storeTileBMap, {threadIdx[0], threadIdx[1]}, 
    //                     (ldgBSize < config["VECTORIZE_WIDTH"] ? ldgBSize : config["VECTORIZE_WIDTH"]), 
    //                     storeTileA, Position::after);
    // auto gpuBarrierPrefix = Rewriter::barrier(loadTileA, Position::before);
    // auto gpuBarrierSuffix = Rewriter::barrier(storeTileB, Position::after);

    // LOG_DEBUG("===== storeTileAB =======\n",module);

    // auto loadFragAMap = getAffineMap("loadFragA", builder, config);
    // auto loadFragA = Rewriter::read(smA, fragA, loadFragAMap, {threadIdx[0], threadIdx[1], k_inner.getInductionVar()}, 
    //                   config["VECTORIZE_WIDTH"], k_inner, Position::begin);
    // tools::_opSetDescription(loadFragA,"loadFragA");
    // auto loadFragBMap = getAffineMap("loadFragB", builder, config);
    // auto loadFragB = Rewriter::read(smB, fragB, loadFragBMap, {threadIdx[0], threadIdx[1], k_inner.getInductionVar()}, 
    //                   config["VECTORIZE_WIDTH"], loadFragA, Position::after);
    // tools::_opSetDescription(loadFragB,"loadFragB");

    // Rewriter::cache_read(k_inner, A, fragA, getAffineMap("cacheReadA", builder, config), {m_inner.getInductionVar()});
    // Rewriter::cache_read(k_inner, B, fragB, getAffineMap("cacheReadB", builder, config), {n_inner.getInductionVar()});
    // LOG_DEBUG("===== load frag & cache_read =======\n",module);

    // auto writeCbody = Rewriter::get_write(blockLevel, C);
    // assert(writeCbody.size() == 1);
    // auto m_inner_axes = Rewriter::split(writeCbody[0][0], 2, {config["VECTORIZE_WIDTH"]});
    // auto n_inner_axes = Rewriter::split(writeCbody[0][1], 2, {config["VECTORIZE_WIDTH"]});
    // auto m_inner_0 = m_inner_axes[0], m_inner_1 = m_inner_axes[1];
    // auto n_inner_0 = n_inner_axes[0], n_inner_1 = n_inner_axes[1];
    // Rewriter::reorder({m_inner_0, n_inner_0, m_inner_1, n_inner_1});
    // // module.dump();
    // tools::_opSetDescription(m_inner_0,"m_inner_0");
    // tools::_opSetDescription(m_inner_1,"m_inner_1");
    // tools::_opSetDescription(n_inner_0,"n_inner_0");
    // tools::_opSetDescription(n_inner_1,"n_inner_1");
    // Rewriter::cache_write(m_inner_0, C, C, getAffineMap("cacheWriteC", builder, config), 
    //                       {threadIdx[0], threadIdx[1], blockIdx[0], blockIdx[1],
    //                        m_inner_0.getInductionVar(),n_inner_0.getInductionVar(),
    //                        m_inner_1.getInductionVar(),n_inner_1.getInductionVar()
    //                       });

    // Rewriter::vectorize(n_inner_1, config["VECTORIZE_WIDTH"]);
    // // module.dump();
    
    // auto doubleLoadTileB = Rewriter::pipeline({loadTileB, storeTileB}, smB, k_outer);
    // auto doubleLoadTileA = Rewriter::pipeline({loadTileA, storeTileA}, smA, k_outer);
    // auto doubleLoadFragB = Rewriter::pipeline({loadFragB}, fragB, k_inner);
    // auto doubleLoadFragA = Rewriter::pipeline({loadFragA}, fragA, k_inner);
    // // module.dump();

    // Rewriter::detach_last_loop(k_inner);

    // Rewriter::schedule(doubleLoadTileA[0][0], doubleLoadTileB[0][0], Position::before);
    // Rewriter::schedule(doubleLoadTileA[0][1], doubleLoadTileB[0][1], Position::before); 
    // Rewriter::schedule(gpuBarrierPrefix, doubleLoadTileB[0][1], Position::after);
    // Rewriter::schedule(doubleLoadTileB[1][0], doubleLoadTileA[1][0], Position::after);
    // Rewriter::schedule(doubleLoadTileA[1][1], doubleLoadTileB[1][1], Position::before);
    // Rewriter::schedule(gpuBarrierSuffix, doubleLoadTileB[1][1], Position::after);
    // auto ifOp = doubleLoadTileA[1][1]->getParentOp();
    // Rewriter::schedule(ifOp, k_inner, Position::after); 
    // Rewriter::extract_loop(doubleLoadFragA[0][0], k_outer, /*iteration*/0);
    // Rewriter::extract_loop(doubleLoadFragB[0][0], k_outer, /*iteration*/0);
    // Rewriter::schedule(doubleLoadFragB[0][0], k_outer, Position::end);
    // Rewriter::schedule(doubleLoadFragA[0][0], k_outer, Position::end);
    // // module.dump();

    // Rewriter::change_double_buffer(doubleLoadFragA[0][0], smA);
    // Rewriter::change_double_buffer(doubleLoadFragB[0][0], smB);;

    // Rewriter::take_off_true_if(module);
    // Rewriter::delete_false_if(module);
    // // module.dump();

    // int64_t threshold = std::max(config["BLOCK_SIZE_K"], std::max(config["THREAD_SIZE_M"], config["THREAD_SIZE_N"]));
    // Rewriter::unroll(module, [&](mlir::affine::AffineForOp forOp)->bool {
    //   if (!forOp.hasConstantBounds()) return false;
    //   auto step = forOp.getStep().getLimitedValue();
    //   auto ub = forOp.getConstantUpperBound();
    //   auto lb = forOp.getConstantLowerBound();
    //   auto times = (ub - lb) / step;
    //   if (times >= std::min<int64_t>(threshold, config["VECTORIZE_WIDTH"])) return false;
    //   return true;
    // });
    // // module.dump();

    // Rewriter::unrollAttribute(module, [&](mlir::affine::AffineForOp forOp)->bool {
    //   if (!forOp.hasConstantBounds()) return false;
    //   auto step = forOp.getStep().getLimitedValue();
    //   auto ub = forOp.getConstantUpperBound();
    //   auto lb = forOp.getConstantLowerBound();
    //   auto times = (ub - lb) / step;
    //   if (times > threshold) return false;
    //   return true;
    // });
    // LOG_DEBUG("===== after applyOptimizer =======\n",module);
  }
}

}

