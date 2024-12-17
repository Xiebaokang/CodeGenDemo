#include "Conversion/Optimizer.h"
#include <cfloat>
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include <filesystem>

namespace KernelCodeGen {

void _opSetDbgPrintAttr(mlir::Operation* op, const std::string& attrValue){
  mlir::OpBuilder b(op->getContext());
  // auto log = b.getStringAttr(attrValue);
  // b.setInsertionPointAfter(op);
  // auto zero = b.create<mlir::arith::ConstantIntOp>(b.getUnknownLoc(),0,32);
  // b.create<mlir::gpu::PrintfOp>(b.getUnknownLoc(),log,mlir::ValueRange(zero));
  op->setAttr("kcg.debug",b.getStringAttr(attrValue));
}

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


int64_t smAReadSride(int64_t blockDim, int64_t warpSize, int64_t blockLayoutN, int64_t warpLayoutM) {
  int64_t warpNum = blockDim / warpSize;
  return (warpNum / blockLayoutN) * warpLayoutM;
}


int64_t smBReadSride(int64_t blockDim, int64_t warpSize, int64_t blockLayoutM, int64_t warpLayoutN) {
  int64_t warpNum = blockDim / warpSize;
  return (warpNum / blockLayoutM) * warpLayoutN;
}

// mlir::AffineMap getAffineMap_GlobalToShmA(mlir::OpBuilder& builder,std::map<std::string, int> config){
//   auto dim_tx = builder.getAffineDimExpr(0);
//   auto dim_ty = builder.getAffineDimExpr(1);
//   auto dim_by = builder.getAffineDimExpr(2);
//   auto dim_k_outer = builder.getAffineDimExpr(3);
//   const int& BM = config[KEY_BLOCK_SIZE_M]; 
//   const int& BN = config[KEY_BLOCK_SIZE_N]; 
//   const int& BK = config[KEY_BLOCK_SIZE_K];
//   const int& TM = config[KEY_THREAD_SIZE_M];
//   const int& TN = config[KEY_THREAD_SIZE_N];
//   int64_t blockDimY = BM / TM;  // y轴 thread个数
//   int64_t blockDimX = BN / TN;  // x轴 thread个数
//   int64_t nThreadsInBlock =  blockDimY * blockDimX;
//   bool vectorize = config.count("VECTORIZE_WIDTH") != 0;
//   int WIDTH = vectorize ? config[KEY_VECTORIZE_WIDTH] : 1;
//   int nElementsPerThread = BM*BK/nThreadsInBlock;  // Atile 需要每个thread 搬运多少数据
//   int nReadLoopCount = nElementsPerThread / WIDTH;  // 几次能搬运ok
//   if(nElementsPerThread < WIDTH){
//     WIDTH = nElementsPerThread;
//   }
//   // tid = ty * blockDimx + tx
//   auto tid = dim_ty * blockDimX + dim_tx;
  

// }

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
  int64_t blockDimY = BM/TM;
  int64_t blockDimX = BN/TN;
  int nThreads = blockDimX * blockDimY;

  bool vectorize = config.count("VECTORIZE_WIDTH") != 0;
  int width = vectorize ? config[KEY_VECTORIZE_WIDTH] : 1;

  int nLoopsLoadTileA = BM * BK / nThreads / width;
  int nLoopsLoadTileB = BN * BK / nThreads / width;
  if (mapIdentifier == "loadTileA" 
  || mapIdentifier == "storeTileA" 
  // || mapIdentifier == "loadFragA"
  )
  {
    width = nLoopsLoadTileA < 1 ? (BM * BK / nThreads) : width;
  }
  if (mapIdentifier == "loadTileB" 
  || mapIdentifier == "storeTileB" 
  // || mapIdentifier == "loadFragB"
  )
  {
    width = nLoopsLoadTileB < 1 ? (BN * BK / nThreads) : width;
  }

  std::vector<int64_t> warpOrg {config["BLOCK_LAYOUT_M"], config["BLOCK_LAYOUT_N"]};  
  std::vector<int64_t> threadOrg {config["WARP_LAYOUT_M"], config["WARP_LAYOUT_N"]};

  if (mapIdentifier == "loadTileA") {
    // dims are:[dim0, dim1, dim2, dim3, dim4]
    // operands are: [threadIdx.y, threadIdx.x, blockIdx.y, k_outer, iv]
    // iv represent a block copy for iv times. 
    auto& _ty = dim0;
    auto& _tx = dim1;
    auto& _by = dim2;
    auto& _kouter = dim3;
    auto& _iv = dim4;
    auto threadIdExpr = _ty * blockDimX + _tx;  // ty * blockDimx + tx

#if 1
    auto virtaulThreadIxExpr = threadIdExpr + _iv * blockDimY * blockDimX;

    auto M_Offset = virtaulThreadIxExpr.floorDiv((uint64_t)(BK / width));
    auto K_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(BK) / width); 
    auto M_Base = _by * BM;
    auto K_Base = _kouter;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(M_Offset + M_Base);
    exprs.push_back(K_Offset * width + K_Base);
    return mlir::AffineMap::get(/*dimCount*/5, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
#else
    int nLoopUbs = BK * BM / nThreads / width;  // 搬运几次
    if(nLoopUbs < 1){
      width = BK * BM / nThreads;
    }
    auto iM = (threadIdExpr * width).floorDiv(BM);
    auto iK = threadIdExpr * width % BM;
    auto K_Base = _kouter;
    auto M_Base = _by * BM;

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(iM + M_Base);
    exprs.push_back(iK + K_Base);
    return mlir::AffineMap::get(/*dimCount*/5, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
#endif
  } else if (mapIdentifier == "loadTileB") {
    // dims are:[dim0, dim1, dim2, dim3, dim4]
    // operands are: [threadIdx.y, threadIdx.x, k_outer, blockIdx.x, iv]
    auto threadIdExpr = dim0 * blockDimX + dim1;  // ty * blockDimx + tx
#if 1
    auto virtaulThreadIxExpr = threadIdExpr + dim4 * blockDimY * blockDimX;
    auto K_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(config["BLOCK_SIZE_N"]) / width);
    auto N_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(config["BLOCK_SIZE_N"]) / width); 
    auto K_Base = dim2;
    auto N_Base = dim3 * config["BLOCK_SIZE_N"];
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_Offset + K_Base);
    exprs.push_back(N_Offset * width + N_Base);
#else
    auto& _ty = dim0;
    auto& _tx = dim1;
    auto& _kouter = dim2;
    auto& _bx = dim3;
    auto tid = _ty * blockDimX + _tx;  // ty * blockDimx + tx
    int nLoopUbs = BK * BN / nThreads / width;  // 搬运几次
    if(nLoopUbs < 1){
      width = BK * BN / nThreads;
    }
    auto iK = (tid * width).floorDiv(BN);
    auto iN = tid * width % BN;
    auto K_Base = _kouter;
    auto N_Base = _bx * BN;

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(iK + K_Base);
    exprs.push_back(iN + N_Base);
#endif

    return mlir::AffineMap::get(/*dimCount*/5, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "storeTileA") {
    // dims are:[dim0, dim1, dim2, dim3]
    // operands are: [threadIdx.y, threadIdx.x, iv, ivInVector]
    auto& ty = dim0;
    auto& tx = dim1;
    auto& iv = dim2;
    auto& ivInVector = dim3;

    auto threadIdExpr = ty * blockDimX + tx;
    auto virtaulThreadIxExpr = threadIdExpr + iv * nThreads;
    auto lineThreadDeal = static_cast<uint64_t>(config["BLOCK_SIZE_K"]) / width;
    auto M_Offset = virtaulThreadIxExpr.floorDiv(lineThreadDeal);
    auto K_Offset = virtaulThreadIxExpr % lineThreadDeal;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_Offset * width + ivInVector);
    exprs.push_back(M_Offset);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "storeTileB") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.y, threadIdx.x, iv]
    auto threadIdExpr = dim0 * blockDimX + dim1;  // tid = ty*Blockdimx + tx
    auto virtaulThreadIxExpr = threadIdExpr + dim2 * blockDimY * blockDimX;
    auto K_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(config["BLOCK_SIZE_N"]) / width);
    auto N_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(config["BLOCK_SIZE_N"]) / width); 
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_Offset);
    exprs.push_back(N_Offset * width);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadFragA") {
    // dims are:[dim0, dim1, dim2, dim3]
    // operands are: [threadIdx.y, threadIdx.x, k_inner, iv]
    auto& ty = dim0;
    auto& tx = dim1;

    auto threadIdExpr = ty * blockDimX + tx;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(config["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(config["WARP_SIZE"]);

    auto M_offset = laneId.floorDiv(threadOrg[1]) + threadOrg[0] * (warpId.floorDiv(warpOrg[1]) + dim3 * warpOrg[0]);
    auto K_offset = dim2;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_offset);
    exprs.push_back(M_offset * width);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadFragB") {
    // dims are:[dim0, dim1, dim2, dim3]
    // operands are: [threadIdx.y, threadIdx.x, k_inner, iv]
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(config["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(config["WARP_SIZE"]);

    auto N_offset = laneId % threadOrg[1] + threadOrg[1] * (warpId % warpOrg[1] + dim3 * warpOrg[1]);
    auto K_offset = dim2;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_offset);
    exprs.push_back(N_offset * width);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "cacheReadA" || mapIdentifier == "cacheReadB") {
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    return mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "cacheWriteC") {
    // dims are:[dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7]
    // operands are: [threadIdx.y, threadIdx.x, blockIdx.y, blockIdx.x, iv0, iv1, iv2, iv3]
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(config["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(config["WARP_SIZE"]);

    auto M_offset = laneId.floorDiv(threadOrg[1]) + threadOrg[0] * (warpId.floorDiv(warpOrg[1]) + dim4.floorDiv(width) * warpOrg[0]);
    auto N_offset = laneId % threadOrg[1] + threadOrg[1] * (warpId % warpOrg[1] + dim5.floorDiv(width) * warpOrg[1]);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim2 * config["BLOCK_SIZE_M"] + M_offset * width + dim6);
    exprs.push_back(dim3 * config["BLOCK_SIZE_N"] + N_offset * width + dim7);
    return mlir::AffineMap::get(/*dimCount*/8, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
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

    std::vector<mlir::affine::AffineForOp> kmn_axes{loopK, m_inner, n_inner};
    auto tileC = Rewriter::bufferizeLoopCarryVar(kmn_axes);
    loopK = kmn_axes[0], m_inner = kmn_axes[1], n_inner = kmn_axes[2];

    Rewriter::reorder({loopK, m_inner, n_inner});
    // module.dump();

    auto k_axes = Rewriter::split(loopK, 2, {config["BLOCK_SIZE_K"]});
    auto k_outer = k_axes[0], k_inner = k_axes[1];
    tools::_opSetDescription(k_inner,"k_inner");
    tools::_opSetDescription(k_outer,"k_outer");
    int64_t blockThreads;
    auto blockDim = Analyzer::getParallelNumber(blockLevel, blockThreads);

    auto ldgASize = config["BLOCK_SIZE_K"] * config["BLOCK_SIZE_M"] / blockThreads;
    auto ldgBSize = config["BLOCK_SIZE_K"] * config["BLOCK_SIZE_N"] / blockThreads;
    auto fragASize = config["BLOCK_SIZE_M"] / smAReadSride(blockThreads, config["WARP_SIZE"], 
                                                          config["BLOCK_LAYOUT_N"], config["WARP_LAYOUT_M"]);
    auto fragBSize = config["BLOCK_SIZE_N"] / smBReadSride(blockThreads, config["WARP_SIZE"], 
                                                          config["BLOCK_LAYOUT_M"], config["WARP_LAYOUT_N"]);
    auto elementA = A.getType().dyn_cast<mlir::MemRefType>().getElementType();
    auto elementB = B.getType().dyn_cast<mlir::MemRefType>().getElementType();

    auto fragB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {fragBSize}, elementB);
    auto fragA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {fragASize}, elementA);

    auto tileB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {ldgBSize}, elementB);
    auto tileA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {ldgASize}, elementA);
    auto smB = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared,
            {config["BLOCK_SIZE_K"], config["BLOCK_SIZE_N"]}, elementB);
    auto smA = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared,
            {config["BLOCK_SIZE_K"], config["BLOCK_SIZE_M"]}, elementA);
    // module.dump();
    
    auto blockIdx = Rewriter::getParallelIdx(gridLevel);
    auto threadIdx = Rewriter::getParallelIdx(blockLevel);
    
    auto loadTileAMap = getAffineMap("loadTileA", builder, config);
    // threadIdx[0], threadIdx[1] 两者先后交换，即可改变 tx ty 的方向
    // shm->temp
    auto loadTileA = Rewriter::read(A, tileA, loadTileAMap, {threadIdx[0], threadIdx[1], blockIdx[0], k_outer.getInductionVar()}, 
                      (ldgASize < config["VECTORIZE_WIDTH"] ? ldgASize : config["VECTORIZE_WIDTH"]), 
                      k_outer, Position::begin);
    auto loadTileBMap = getAffineMap("loadTileB", builder, config);
    auto loadTileB = Rewriter::read(B, tileB, loadTileBMap, 
                      {threadIdx[0], threadIdx[1], k_outer.getInductionVar(), blockIdx[1]}, 
                      (ldgBSize < config["VECTORIZE_WIDTH"] ? ldgBSize : config["VECTORIZE_WIDTH"]), 
                      loadTileA, Position::after);
    // module.dump();
    LOG_DEBUG("===== shm->temp =======\n",module);

    auto storeTileAMap = getAffineMap("storeTileA", builder, config);
    // auto storeTileA = Rewriter::write(tileA, smA, storeTileAMap, {threadIdx[1], threadIdx[0]}, 
    auto storeTileA = Rewriter::write(tileA, smA, storeTileAMap, {threadIdx[0], threadIdx[1]}, 
                        (ldgASize < config["VECTORIZE_WIDTH"] ? ldgASize : config["VECTORIZE_WIDTH"]), 
                        loadTileB, Position::after);
    auto storeTileBMap = getAffineMap("storeTileB", builder, config);
    auto storeTileB = Rewriter::write(tileB, smB, storeTileBMap, {threadIdx[0], threadIdx[1]}, 
                        (ldgBSize < config["VECTORIZE_WIDTH"] ? ldgBSize : config["VECTORIZE_WIDTH"]), 
                        storeTileA, Position::after);
    auto gpuBarrierPrefix = Rewriter::barrier(loadTileA, Position::before);
    auto gpuBarrierSuffix = Rewriter::barrier(storeTileB, Position::after);

    LOG_DEBUG("===== storeTileAB =======\n",module);

    auto loadFragAMap = getAffineMap("loadFragA", builder, config);
    auto loadFragA = Rewriter::read(smA, fragA, loadFragAMap, {threadIdx[0], threadIdx[1], k_inner.getInductionVar()}, 
                      config["VECTORIZE_WIDTH"], k_inner, Position::begin);
    tools::_opSetDescription(loadFragA,"loadFragA");
    auto loadFragBMap = getAffineMap("loadFragB", builder, config);
    auto loadFragB = Rewriter::read(smB, fragB, loadFragBMap, {threadIdx[0], threadIdx[1], k_inner.getInductionVar()}, 
                      config["VECTORIZE_WIDTH"], loadFragA, Position::after);
    tools::_opSetDescription(loadFragB,"loadFragB");

    Rewriter::cache_read(k_inner, A, fragA, getAffineMap("cacheReadA", builder, config), {m_inner.getInductionVar()});
    Rewriter::cache_read(k_inner, B, fragB, getAffineMap("cacheReadB", builder, config), {n_inner.getInductionVar()});
    LOG_DEBUG("===== load frag & cache_read =======\n",module);

    auto writeCbody = Rewriter::get_write(blockLevel, C);
    assert(writeCbody.size() == 1);
    auto m_inner_axes = Rewriter::split(writeCbody[0][0], 2, {config["VECTORIZE_WIDTH"]});
    auto n_inner_axes = Rewriter::split(writeCbody[0][1], 2, {config["VECTORIZE_WIDTH"]});
    auto m_inner_0 = m_inner_axes[0], m_inner_1 = m_inner_axes[1];
    auto n_inner_0 = n_inner_axes[0], n_inner_1 = n_inner_axes[1];
    Rewriter::reorder({m_inner_0, n_inner_0, m_inner_1, n_inner_1});
    // module.dump();
    tools::_opSetDescription(m_inner_0,"m_inner_0");
    tools::_opSetDescription(m_inner_1,"m_inner_1");
    tools::_opSetDescription(n_inner_0,"n_inner_0");
    tools::_opSetDescription(n_inner_1,"n_inner_1");
    Rewriter::cache_write(m_inner_0, C, C, getAffineMap("cacheWriteC", builder, config), 
                          {threadIdx[0], threadIdx[1], blockIdx[0], blockIdx[1],
                           m_inner_0.getInductionVar(),n_inner_0.getInductionVar(),
                           m_inner_1.getInductionVar(),n_inner_1.getInductionVar()
                          });

    Rewriter::vectorize(n_inner_1, config["VECTORIZE_WIDTH"]);
    // module.dump();
    
    auto doubleLoadTileB = Rewriter::pipeline({loadTileB, storeTileB}, smB, k_outer);
    auto doubleLoadTileA = Rewriter::pipeline({loadTileA, storeTileA}, smA, k_outer);
    auto doubleLoadFragB = Rewriter::pipeline({loadFragB}, fragB, k_inner);
    auto doubleLoadFragA = Rewriter::pipeline({loadFragA}, fragA, k_inner);
    // module.dump();

    Rewriter::detach_last_loop(k_inner);

    Rewriter::schedule(doubleLoadTileA[0][0], doubleLoadTileB[0][0], Position::before);
    Rewriter::schedule(doubleLoadTileA[0][1], doubleLoadTileB[0][1], Position::before); 
    Rewriter::schedule(gpuBarrierPrefix, doubleLoadTileB[0][1], Position::after);
    Rewriter::schedule(doubleLoadTileB[1][0], doubleLoadTileA[1][0], Position::after);
    Rewriter::schedule(doubleLoadTileA[1][1], doubleLoadTileB[1][1], Position::before);
    Rewriter::schedule(gpuBarrierSuffix, doubleLoadTileB[1][1], Position::after);
    auto ifOp = doubleLoadTileA[1][1]->getParentOp();
    Rewriter::schedule(ifOp, k_inner, Position::after); 
    Rewriter::extract_loop(doubleLoadFragA[0][0], k_outer, /*iteration*/0);
    Rewriter::extract_loop(doubleLoadFragB[0][0], k_outer, /*iteration*/0);
    Rewriter::schedule(doubleLoadFragB[0][0], k_outer, Position::end);
    Rewriter::schedule(doubleLoadFragA[0][0], k_outer, Position::end);
    // module.dump();

    Rewriter::change_double_buffer(doubleLoadFragA[0][0], smA);
    Rewriter::change_double_buffer(doubleLoadFragB[0][0], smB);;

    Rewriter::take_off_true_if(module);
    Rewriter::delete_false_if(module);
    // module.dump();

    int64_t threshold = std::max(config["BLOCK_SIZE_K"], std::max(config["THREAD_SIZE_M"], config["THREAD_SIZE_N"]));
    Rewriter::unroll(module, [&](mlir::affine::AffineForOp forOp)->bool {
      if (!forOp.hasConstantBounds()) return false;
      auto step = forOp.getStep().getLimitedValue();
      auto ub = forOp.getConstantUpperBound();
      auto lb = forOp.getConstantLowerBound();
      auto times = (ub - lb) / step;
      if (times >= std::min<int64_t>(threshold, config["VECTORIZE_WIDTH"])) return false;
      return true;
    });
    // module.dump();

    Rewriter::unrollAttribute(module, [&](mlir::affine::AffineForOp forOp)->bool {
      if (!forOp.hasConstantBounds()) return false;
      auto step = forOp.getStep().getLimitedValue();
      auto ub = forOp.getConstantUpperBound();
      auto lb = forOp.getConstantLowerBound();
      auto times = (ub - lb) / step;
      if (times > threshold) return false;
      return true;
    });
    LOG_DEBUG("===== after applyOptimizer =======\n",module);
  }
}

}

