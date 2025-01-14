#include "Conversion/Optimizer.h"
#include <cfloat>
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include <filesystem>
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"

namespace KernelCodeGen {

MatmulConfigUtils::MatmulConfigUtils(const std::map<std::string,int>& config){
  this->BM = config.at(KEY_BLOCK_SIZE_M); 
  this->BN = config.at(KEY_BLOCK_SIZE_N); 
  this->BK = config.at(KEY_BLOCK_SIZE_K);
  this->TM = config.at(KEY_THREAD_SIZE_M);
  this->TN = config.at(KEY_THREAD_SIZE_N);
  this->BLOCK_Y = BM / TM;
  this->BLOCK_X = BN / TN;
  this->THREAD_NUM = BLOCK_X * BLOCK_Y * config.at(KEY_LOCAL_SPLIT_U);
  this->SHARED_SIZE_A = BM * BK;
  this->SHARED_SIZE_B = BN * BK;
  this->GLOB_LOAD_ROW_WIDTH_A = THREAD_NUM / BK * config.at(KEY_GLOB_LOAD_WIDTH_A);
  this->GLOB_LOAD_ROW_WIDTH_B = THREAD_NUM / BK * config.at(KEY_GLOB_LOAD_WIDTH_B);
  this->BLOCK_REPEAT_A = TM / config.at(KEY_WARP_SCATTER_WIDTH_A);
  this->WARP_REPEAT_A = config.at(KEY_WARP_SCATTER_WIDTH_A) / config.at(KEY_THREAD_SCATTER_WIDTH_A);
  this->BLOCK_REPEAT_B = TN / config.at(KEY_WARP_SCATTER_WIDTH_B);
  this->WARP_REPEAT_B = config.at(KEY_WARP_SCATTER_WIDTH_B) / config.at(KEY_THREAD_SCATTER_WIDTH_B);
  this->GLOB_STORE_ROW_WIDTH = THREAD_NUM / BM * config.at(KEY_GLOB_STORE_WIDTH);
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

void test(mlir::AffineMap map) {
  LOG_DEBUG("",map);
}

mlir::AffineMap MatmulOptimizer::getGlobToTempMapA(mlir::OpBuilder& builder, const std::map<std::string, int>& config, bool isContinuous) {
  int dimCount = 0;
  auto by = builder.getAffineDimExpr(dimCount++);
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto k = builder.getAffineDimExpr(dimCount++);
  auto iter = builder.getAffineDimExpr(dimCount++);
  
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  // ***** load glob to temp reg *****
  auto sh_load = tools::mapUtils::reshapeBlock(tid, {cfg.BK, cfg.THREAD_NUM/cfg.BK});
  exprs.push_back(sh_load[0] + k);
  exprs.push_back(by * cfg.BM + iter * cfg.GLOB_LOAD_ROW_WIDTH_A + sh_load[1] * config.at(KEY_GLOB_LOAD_WIDTH_A));
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap MatmulOptimizer::getGlobToTempMapB(mlir::OpBuilder& builder, const std::map<std::string, int>& config, bool isContinuous) {
  int dimCount = 0;
  auto bx = builder.getAffineDimExpr(dimCount++);
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto k = builder.getAffineDimExpr(dimCount++);
  auto iter = builder.getAffineDimExpr(dimCount++);
  
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  // ***** load glob to temp reg *****
  auto sh_load = tools::mapUtils::reshapeBlock(tid, {cfg.BK, cfg.THREAD_NUM/cfg.BK});
  exprs.push_back(sh_load[0] + k);
  exprs.push_back(bx * cfg.BN + iter * cfg.GLOB_LOAD_ROW_WIDTH_B + sh_load[1] * config.at(KEY_GLOB_LOAD_WIDTH_B));

  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap MatmulOptimizer::getTempToSharedMapSMA(mlir::OpBuilder& builder, const std::map<std::string, int>& config, bool isContinuous) {
  int dimCount = 0;
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto iter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  // ***** load temp reg to shared *****
  auto sh_load = tools::mapUtils::reshapeBlock(tid, {cfg.BK, cfg.THREAD_NUM/cfg.BK});
  exprs.push_back(sh_load[0]);
  exprs.push_back(iter * cfg.GLOB_LOAD_ROW_WIDTH_A + sh_load[1] * config.at(KEY_GLOB_LOAD_WIDTH_A));
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getTempToSharedMapSMB(mlir::OpBuilder& builder, const std::map<std::string, int>& config, bool isContinuous) {
  int dimCount = 0;
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto iter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  // ***** load temp reg to shared *****
  auto sh_load = tools::mapUtils::reshapeBlock(tid, {cfg.BK, cfg.THREAD_NUM/cfg.BK});
  exprs.push_back(sh_load[0]);
  exprs.push_back(iter * cfg.GLOB_LOAD_ROW_WIDTH_B + sh_load[1] * config.at(KEY_GLOB_LOAD_WIDTH_B));
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getSharedToRegMapSMA(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto bk = builder.getAffineDimExpr(dimCount++);
  auto blockRepIter = builder.getAffineDimExpr(dimCount++);
  auto warpRepIter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  auto reTh = tools::mapUtils::reshapeBlock(tid, {config.at(KEY_LOCAL_SPLIT_U), cfg.THREAD_NUM/config.at(KEY_LOCAL_SPLIT_U)});
  auto warp_y = tools::mapUtils::wapr_y(reTh[1], config);
  auto lane_y = tools::mapUtils::lane_y(reTh[1], config);
  exprs.push_back(bk + reTh[0]);
  exprs.push_back((blockRepIter * config.at(KEY_BLOCK_LAYOUT_M) + warp_y) * config.at(KEY_WARP_LAYOUT_M) * config.at(KEY_WARP_SCATTER_WIDTH_A) + 
                  (warpRepIter * config.at(KEY_WARP_LAYOUT_M) + lane_y) * config.at(KEY_THREAD_SCATTER_WIDTH_A));
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getSharedToRegMapSMB(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto bk = builder.getAffineDimExpr(dimCount++);
  auto blockRepIter = builder.getAffineDimExpr(dimCount++);
  auto warpRepIter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  auto reTh = tools::mapUtils::reshapeBlock(tid, {config.at(KEY_LOCAL_SPLIT_U), cfg.THREAD_NUM/config.at(KEY_LOCAL_SPLIT_U)});
  auto warp_x = tools::mapUtils::wapr_x(reTh[1], config);
  auto lane_x = tools::mapUtils::lane_x(reTh[1], config);
  exprs.push_back(bk + reTh[0]);
  exprs.push_back((blockRepIter * config.at(KEY_BLOCK_LAYOUT_N) + warp_x) * config.at(KEY_WARP_LAYOUT_N) * config.at(KEY_WARP_SCATTER_WIDTH_B) + 
                  (warpRepIter * config.at(KEY_WARP_LAYOUT_N) + lane_x) * config.at(KEY_THREAD_SCATTER_WIDTH_B));
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getCalculateMapReg(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto iter = builder.getAffineDimExpr(dimCount++);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back(iter);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getRegToGlobMapC(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto by = builder.getAffineDimExpr(dimCount++);
  auto bx = builder.getAffineDimExpr(dimCount++);
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto blockRepIterA = builder.getAffineDimExpr(dimCount++);
  auto blockRepIterB = builder.getAffineDimExpr(dimCount++);
  auto warpRepIterA = builder.getAffineDimExpr(dimCount++);
  auto warpRepIterB = builder.getAffineDimExpr(dimCount++);
  auto iterA = builder.getAffineDimExpr(dimCount++);
  auto iterB = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  auto warp_y = tools::mapUtils::wapr_y(tid, config);
  auto warp_x = tools::mapUtils::wapr_x(tid, config);
  auto lane_y = tools::mapUtils::lane_y(tid, config);
  auto lane_x = tools::mapUtils::lane_x(tid, config);
  exprs.push_back(by * cfg.BM + (blockRepIterA * config.at(KEY_BLOCK_LAYOUT_M) + warp_y * config.at(KEY_WARP_SCATTER_WIDTH_A)) * config.at(KEY_WARP_LAYOUT_M) + 
                                 warpRepIterA * config.at(KEY_WARP_LAYOUT_M) + lane_y * config.at(KEY_THREAD_SCATTER_WIDTH_A) + iterA);
  exprs.push_back(bx * cfg.BN + (blockRepIterB * config.at(KEY_BLOCK_LAYOUT_N) + warp_x * config.at(KEY_WARP_SCATTER_WIDTH_B)) * config.at(KEY_WARP_LAYOUT_N) + 
                                 warpRepIterB * config.at(KEY_WARP_LAYOUT_N) + lane_x * config.at(KEY_THREAD_SCATTER_WIDTH_B) + iterB);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getRegToSharedMapSMC(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto blockRepIterA = builder.getAffineDimExpr(dimCount++);
  auto blockRepIterB = builder.getAffineDimExpr(dimCount++);
  auto warpRepIterA = builder.getAffineDimExpr(dimCount++);
  auto warpRepIterB = builder.getAffineDimExpr(dimCount++);
  auto iterA = builder.getAffineDimExpr(dimCount++);
  auto iterB = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  auto reTh = tools::mapUtils::reshapeBlock(tid, {config.at(KEY_LOCAL_SPLIT_U), cfg.THREAD_NUM/config.at(KEY_LOCAL_SPLIT_U)});
  auto warp_y = tools::mapUtils::wapr_y(reTh[1], config);
  auto warp_x = tools::mapUtils::wapr_x(reTh[1], config);
  auto lane_y = tools::mapUtils::lane_y(reTh[1], config);
  auto lane_x = tools::mapUtils::lane_x(reTh[1], config);
  exprs.push_back(reTh[0]);
  exprs.push_back((blockRepIterA * config.at(KEY_BLOCK_LAYOUT_M) + warp_y * config.at(KEY_WARP_SCATTER_WIDTH_A)) * config.at(KEY_WARP_LAYOUT_M) + 
                   warpRepIterA * config.at(KEY_WARP_LAYOUT_M) + lane_y * config.at(KEY_THREAD_SCATTER_WIDTH_A) + iterA);
  exprs.push_back((blockRepIterB * config.at(KEY_BLOCK_LAYOUT_N) + warp_x * config.at(KEY_WARP_SCATTER_WIDTH_B)) * config.at(KEY_WARP_LAYOUT_N) + 
                   warpRepIterB * config.at(KEY_WARP_LAYOUT_N) + lane_x * config.at(KEY_THREAD_SCATTER_WIDTH_B) + iterB);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getReduceMapSMC(mlir::OpBuilder& builder, const std::map<std::string, int>& config, bool isContinuous) {
  int dimCount = 0;
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto iterSplitU = builder.getAffineDimExpr(dimCount++);
  auto iter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  auto sh_store = tools::mapUtils::reshapeBlock(tid, {cfg.BM, cfg.THREAD_NUM/cfg.BM});
  exprs.push_back(iterSplitU);
  exprs.push_back(sh_store[0]);
  exprs.push_back(iter * cfg.GLOB_STORE_ROW_WIDTH + sh_store[1] * config.at(KEY_GLOB_STORE_WIDTH));
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getReduceMapRegC(mlir::OpBuilder& builder, const std::map<std::string, int>& config, bool isContinuous) {
  int dimCount = 0;
  auto by = builder.getAffineDimExpr(dimCount++);
  auto bx = builder.getAffineDimExpr(dimCount++);
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto iter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  auto sh_store = tools::mapUtils::reshapeBlock(tid, {cfg.BM, cfg.THREAD_NUM/cfg.BM});
  exprs.push_back(by * cfg.BM + sh_store[0]);
  exprs.push_back(bx * cfg.BN + iter * cfg.GLOB_STORE_ROW_WIDTH + sh_store[1] * config.at(KEY_GLOB_STORE_WIDTH));
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}


void MatmulOptimizer::applyOptimzer(mlir::ModuleOp& module, std::map<std::string, int> config) {
  mlir::OpBuilder builder(module);
  for (auto& matmul : matmuls) {
    matmul->setAttr(std::string("func.state"), builder.getStringAttr("gpu"));
    auto loops = matmulLoops[matmul];
    auto loopM = loops[0], loopN = loops[1], loopK = loops[2];
    auto buffers = matmulBuffers[matmul];
    auto A = buffers.A, B = buffers.B, C = buffers.C;
    LOG_DEBUG("===== original mlir =======\n",module);

    auto m_axes = Rewriter::split(loopM, 3, {config["THREAD_SIZE_M"], config["BLOCK_SIZE_M"]});
    auto n_axes = Rewriter::split(loopN, 3, {config["THREAD_SIZE_N"], config["BLOCK_SIZE_N"]});
<<<<<<< HEAD
=======
    llvm::outs() << "===== after split m/n =======\n";llvm::outs().flush(); module.dump();

>>>>>>> dev_bizefeng
    auto m_outer = m_axes[0], m_mider = m_axes[1], m_inner = m_axes[2];
    auto n_outer = n_axes[0], n_mider = n_axes[1], n_inner = n_axes[2];
    Rewriter::reorder({m_outer, n_outer, m_mider, n_mider, m_inner, n_inner});
<<<<<<< HEAD
    LOG_DEBUG("===== after split & reorder =======\n",module);

    auto gridLevel = Rewriter::parallel({m_outer, n_outer});
    auto blockLevel = Rewriter::parallel({m_mider, n_mider});
    LOG_DEBUG("===== after parallel =======\n",module);

    std::vector<mlir::affine::AffineForOp> tileCLoops{m_inner, n_inner};
    auto regC = Rewriter::bufferizeLoopCarryVar(loopK, tileCLoops);
    LOG_DEBUG("===== after bufferizeLoopCarryVar =======\n",module);

    auto k_axes = Rewriter::split(loopK, 3, {config["LOCAL_SPLIT_U"], config["BLOCK_SIZE_K"]});
    auto k_outer = k_axes[0], k_mider = k_axes[1], k_inner = k_axes[2];
    LOG_DEBUG("===== after split =======\n",module);

    // LOG_DEBUG("===== k_outer =======\n",k_outer);
    // LOG_DEBUG("===== k_mider =======\n",k_mider);
    // LOG_DEBUG("===== k_inner =======\n",k_inner);
    
=======
    llvm::outs() << "===== m/n reorder =======\n";llvm::outs().flush(); module.dump();

    std::vector<mlir::affine::AffineForOp> kmn_axes{loopK, m_inner, n_inner};
    LOG_DEBUG("===== m_inner =======\n",m_inner);
    LOG_DEBUG("===== n_inner =======\n",n_inner);
    LOG_DEBUG("===== loopk =======\n",loopK);
    // loopk寄存器实例化，迭代变量外提
    auto tileC = Rewriter::bufferizeLoopCarryVar(kmn_axes);
    LOG_DEBUG("===== after bufferizeLoopCarryVar =======\n",module);
    loopK = kmn_axes[0], m_inner = kmn_axes[1], n_inner = kmn_axes[2];

    Rewriter::reorder({loopK, m_inner, n_inner});
    // module.dump();
    LOG_DEBUG("===== after reorderK =======\n",module);

    auto k_axes = Rewriter::split(loopK, 2, {config["BLOCK_SIZE_K"]});
    LOG_DEBUG("===== after splitK =======\n",module);
    auto k_outer = k_axes[0], k_inner = k_axes[1];
    tools::_opSetDescription(k_inner,"k_inner");
    tools::_opSetDescription(k_outer,"k_outer");
    LOG_DEBUG("===== after splitKopSetDescription =======\n",module);
    
    // int localsplitu = 2;
    // auto lsu_axes = Rewriter::localSplitU(k_inner, localsplitu);
    // tools::_opSetDescription(lsu_axes[0],"local_split_u");
    // LOG_DEBUG("===== after localSplitU =======\n",module);
    
    auto gridLevel = Rewriter::parallel({m_outer, n_outer});
    auto blockLevel = Rewriter::parallel({m_mider, n_mider});
    
    LOG_DEBUG("===== after parallel =======\n",module);
    
    int64_t blockThreads;
    auto blockDim = Analyzer::getParallelNumber(blockLevel, blockThreads);
>>>>>>> dev_bizefeng

    Rewriter::loopToParallelZ(k_inner, blockLevel);
    LOG_DEBUG("===== after loopToParallelZ =======\n",module);

    Rewriter::reorder({k_outer, k_mider, m_inner, n_inner});
    LOG_DEBUG("===== after reorder =======\n",module);

    int64_t blockThreads = Analyzer::getThreadPerBlock(blockLevel);
    // // size of loading from glob to reg
    auto glob_load_total_width_a = config["BLOCK_SIZE_K"] * config["BLOCK_SIZE_M"] / blockThreads;
    auto glob_load_total_width_b = config["BLOCK_SIZE_K"] * config["BLOCK_SIZE_N"] / blockThreads;
    auto elementA = A.getType().dyn_cast<mlir::MemRefType>().getElementType();
    auto elementB = B.getType().dyn_cast<mlir::MemRefType>().getElementType();
<<<<<<< HEAD
    auto regB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {config["THREAD_SIZE_N"]}, elementB);
    auto regA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {config["THREAD_SIZE_M"]}, elementA);
    auto tempB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {glob_load_total_width_b}, elementB);
    auto tempA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {glob_load_total_width_a}, elementA);
    auto smB = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared, {config["BLOCK_SIZE_K"], config["BLOCK_SIZE_N"]}, elementB);
    auto smA = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared, {config["BLOCK_SIZE_K"], config["BLOCK_SIZE_M"]}, elementA);
    tools::_opSetDescription(smA.getDefiningOp(),"smA");
    tools::_opSetDescription(smB.getDefiningOp(),"smB");
    LOG_DEBUG("===== before alloc_buffer =======\n",module);
    int blockDimX = 0;
    int gridDimX = 0;
    Rewriter::parallelToOneDim(blockLevel,&blockDimX);
    tools::opSetAttr(module,AttrGridDim, gridDimX);
    tools::opSetAttr(module,AttrBlockDim, blockDimX);
    LOG_DEBUG("===== before parallelToOneDim =======\n",module);
    
    auto blockIdx = Analyzer::getParallelIdx(gridLevel);
    auto threadIdx = Analyzer::getParallelIdx(blockLevel);
    auto loadTileAMap = getGlobToTempMapA(builder, config);
    // test(loadTileAMap);
    auto loadTileA = Rewriter::read(A, tempA, loadTileAMap, {blockIdx[0], threadIdx[0], k_outer.getInductionVar()}, 
                                    {config["GLOB_LOAD_WIDTH_A"]}, k_outer, Position::begin);
    auto loadTileBMap = getGlobToTempMapB(builder, config);
    // test(loadTileBMap);
    auto loadTileB = Rewriter::read(B, tempB, loadTileBMap, {blockIdx[1], threadIdx[0], k_outer.getInductionVar()}, 
                                    {config["GLOB_LOAD_WIDTH_B"]}, loadTileA, Position::after);
    LOG_DEBUG("===== before read A/B =======\n",module);

    auto storeTileAMap = getTempToSharedMapSMA(builder, config);
    // test(storeTileAMap);
    auto storeTileA = Rewriter::write(tempA, smA, storeTileAMap, {threadIdx[0]}, {config["GLOB_LOAD_WIDTH_A"]}, loadTileB, Position::after);
    auto storeTileBMap = getTempToSharedMapSMB(builder, config);
    // test(storeTileBMap);
    auto storeTileB = Rewriter::write(tempB, smB, storeTileBMap, {threadIdx[0]}, {config["GLOB_LOAD_WIDTH_B"]}, storeTileA, Position::after);
    auto gpuBarrierPrefix = Rewriter::barrier(loadTileA, Position::before);
    auto gpuBarrierSuffix = Rewriter::barrier(storeTileB, Position::after);
    LOG_DEBUG("===== write A/B =======\n",module);

    auto loadFragAMap = getSharedToRegMapSMA(builder, config);
    // test(loadFragAMap);
    auto loadFragA = Rewriter::read(smA, regA, loadFragAMap, {threadIdx[0], k_mider.getInductionVar()}, 
                                    {config["WARP_SCATTER_WIDTH_A"], config["THREAD_SCATTER_WIDTH_A"]}, k_mider, Position::begin);
    auto loadFragBMap = getSharedToRegMapSMB(builder, config);
    // test(loadFragBMap);
    auto loadFragB = Rewriter::read(smB, regB, loadFragBMap, {threadIdx[0], k_mider.getInductionVar()}, 
                                    {config["WARP_SCATTER_WIDTH_B"], config["THREAD_SCATTER_WIDTH_B"]}, loadFragA, Position::after);
    LOG_DEBUG("===== read sh_A/B =======\n",module);
=======

    auto fragB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {fragBSize}, elementB);
    auto fragA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {fragASize}, elementA);

    auto tileB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {ldgBSize}, elementB);
    auto tileA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {ldgASize}, elementA);
    auto smB = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared,
            {config["BLOCK_SIZE_K"], config["BLOCK_SIZE_N"]}, elementB);
    auto smA = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared,
            {config["BLOCK_SIZE_K"], config["BLOCK_SIZE_M"]}, elementA);
    // module.dump();
    LOG_DEBUG("===== after alloc_buffer =======\n",module);
    
    auto blockIdx = Rewriter::getParallelIdx(gridLevel);
    auto threadIdx = Rewriter::getParallelIdx(blockLevel);
    
    auto loadTileAMap = getAffineMap("loadTileA", builder, config);
    // threadIdx[0], threadIdx[1] 两者先后交换，即可改变 tx ty 的方向
    // shm->temp
    llvm::outs() << "===== loadTileAMap =======\n" << loadTileAMap << "\n";llvm::outs().flush();
    // auto loadTileA = Rewriter::read(A, tileA, loadTileAMap, {threadIdx[0], threadIdx[1], blockIdx[0], k_outer.getInductionVar()}, 
    //                   (ldgASize < config["VECTORIZE_WIDTH"] ? ldgASize : config["VECTORIZE_WIDTH"]), 
    //                   k_outer, Position::begin);
    auto loadTileA = Rewriter::read(A, tileA, loadTileAMap, {threadIdx[0], threadIdx[1], blockIdx[0], k_outer.getInductionVar()}, 
                      (ldgASize < config["VECTORIZE_WIDTH"] ? ldgASize : config["VECTORIZE_WIDTH"]), 
                      k_outer, Position::begin);
    LOG_DEBUG("===== gm->temp loadTileA =======\n",module);

    auto loadTileBMap = getAffineMap("loadTileB", builder, config);
    llvm::outs() << "===== loadTileB =======\n" << loadTileBMap << "\n";llvm::outs().flush();
    auto loadTileB = Rewriter::read(B, tileB, loadTileBMap, 
                      {threadIdx[0], threadIdx[1], k_outer.getInductionVar(), blockIdx[1]}, 
                      (ldgBSize < config["VECTORIZE_WIDTH"] ? ldgBSize : config["VECTORIZE_WIDTH"]), 
                      loadTileA, Position::after);
    // module.dump();
    LOG_DEBUG("===== gm->temp loadTileB =======\n",module);

    auto storeTileAMap = getAffineMap("storeTileA", builder, config);
    // auto storeTileA = Rewriter::write(tileA, smA, storeTileAMap, {threadIdx[1], threadIdx[0]}, 
    auto storeTileA = Rewriter::write(tileA, smA, storeTileAMap, {threadIdx[0], threadIdx[1]}, 
                        (ldgASize < config["VECTORIZE_WIDTH"] ? ldgASize : config["VECTORIZE_WIDTH"]), 
                        loadTileB, Position::after);
    LOG_DEBUG("===== temp->shm storeTileA =======\n",module);

    auto storeTileBMap = getAffineMap("storeTileB", builder, config);
    auto storeTileB = Rewriter::write(tileB, smB, storeTileBMap, {threadIdx[0], threadIdx[1]}, 
                        (ldgBSize < config["VECTORIZE_WIDTH"] ? ldgBSize : config["VECTORIZE_WIDTH"]), 
                        storeTileA, Position::after);
    LOG_DEBUG("===== temp->shm storeTileB =======\n",module);

    auto gpuBarrierPrefix = Rewriter::barrier(loadTileA, Position::before);
    auto gpuBarrierSuffix = Rewriter::barrier(storeTileB, Position::after);
    LOG_DEBUG("===== barrierTileATileB =======\n",module);
>>>>>>> dev_bizefeng

    auto cacheRead = getCalculateMapReg(builder, config);
    // test(cacheRead);
    Rewriter::cache_read(n_inner, A, regA, cacheRead, {m_inner.getInductionVar()});
    Rewriter::cache_read(n_inner, B, regB, cacheRead, {n_inner.getInductionVar()});
    LOG_DEBUG("===== load regA & cache_read =======\n",module);

    auto writeCbody = Rewriter::get_write(blockLevel, C);
    assert(writeCbody.size() == 1);
<<<<<<< HEAD
    auto m_inner_axes = Rewriter::split(writeCbody[0][0], 3, {config["WARP_SCATTER_WIDTH_A"], config["THREAD_SCATTER_WIDTH_A"]});
    auto n_inner_axes = Rewriter::split(writeCbody[0][1], 3, {config["WARP_SCATTER_WIDTH_B"], config["THREAD_SCATTER_WIDTH_B"]});
    auto m_inner_0 = m_inner_axes[0], m_inner_1 = m_inner_axes[1], m_inner_2 = m_inner_axes[2];
    auto n_inner_0 = n_inner_axes[0], n_inner_1 = n_inner_axes[1], n_inner_2 = n_inner_axes[2];
    Rewriter::reorder({m_inner_0, n_inner_0, m_inner_1, n_inner_1, m_inner_2, n_inner_2});
    LOG_DEBUG("===== load split & reorder regC to C =======\n",module);

    if (config["LOCAL_SPLIT_U"] > 1) {
      auto elementC = C.getType().dyn_cast<mlir::MemRefType>().getElementType();
      auto regC_ = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {config["THREAD_SIZE_M"] * config["THREAD_SIZE_N"]}, elementC);
      auto smC = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared, {config["LOCAL_SPLIT_U"], config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"]}, elementC);
      tools::_opSetDescription(regC_.getDefiningOp(),"regC_");
      tools::_opSetDescription(smC.getDefiningOp(),"smC");

      auto cacheWriteShCMap = getRegToSharedMapSMC(builder, config);
      test(cacheWriteShCMap);
      Rewriter::cache_write(m_inner_0, C, smC, cacheWriteShCMap, 
                            {threadIdx[0], m_inner_0.getInductionVar(),n_inner_0.getInductionVar(), m_inner_1.getInductionVar(), 
                            n_inner_1.getInductionVar(), m_inner_2.getInductionVar(), n_inner_2.getInductionVar()});
      LOG_DEBUG("===== load cache_write regC to C =======\n",module);

      auto reduceCMap = getReduceMapSMC(builder, config);
      test(reduceCMap);
      auto reduceCLoop = Rewriter::splitUReduce(smC, regC_, reduceCMap, {threadIdx[0]}, config["LOCAL_SPLIT_U"], config["GLOB_STORE_WIDTH"], m_inner_0, Position::after);
      auto reduceLoop_0 = reduceCLoop.first, reduceLoop_1 = reduceCLoop.second;
      LOG_DEBUG("===== load splitUReduce =======\n",module);

      auto writeCMap = getReduceMapRegC(builder, config);
      test(writeCMap);
      Rewriter::splitUWrite(regC_, C, writeCMap, {blockIdx[0], blockIdx[1], threadIdx[0]}, config["LOCAL_SPLIT_U"], config["GLOB_STORE_WIDTH"], reduceLoop_1, Position::after);
      auto StoreBarrier = Rewriter::barrier(m_inner_0, Position::after);
      LOG_DEBUG("===== load write to C =======\n",module);

      // Rewriter::bufferCombine({{smA, smB}, {smC}});
      // Rewriter::bufferCombine({{regC}, {regC_}});
      // LOG_DEBUG("===== bufferCombine =======\n",module);

    } else {
      auto cacheWriteCMap = getRegToGlobMapC(builder, config);
      test(cacheWriteCMap);
      Rewriter::cache_write(m_inner_0, C, C, cacheWriteCMap, 
                            {blockIdx[0], blockIdx[1], threadIdx[0], 
                            m_inner_0.getInductionVar(),n_inner_0.getInductionVar(), m_inner_1.getInductionVar(), 
                            n_inner_1.getInductionVar(),m_inner_2.getInductionVar(), n_inner_2.getInductionVar()});
      LOG_DEBUG("===== load cache_write regC to C =======\n",module);
    }

    Rewriter::vectorize(n_inner_2, config["THREAD_SCATTER_WIDTH_B"]);
    LOG_DEBUG("===== vectorize =======\n",module);
    
    int gridDims = 0;
    Rewriter::parallelToOneDim(gridLevel, &gridDims);
    tools::opSetAttr(module,AttrGridDim,gridDims);
    // Rewriter::BlockMapping(gridLevel, config["BLOCK_MAPPING"]);
    LOG_DEBUG("===== parallelToOneDim gridLevel =======\n",module);
    
=======
    auto m_inner_axes = Rewriter::split(writeCbody[0][0], 2, {config["VECTORIZE_WIDTH"]});
    auto n_inner_axes = Rewriter::split(writeCbody[0][1], 2, {config["VECTORIZE_WIDTH"]});
    LOG_DEBUG("===== VECTORIZE_WIDTH =======\n",module);
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
    LOG_DEBUG("===== cache_write =======\n",module);
    // module.dump();
    
    auto doubleLoadTileB = Rewriter::pipeline({loadTileB, storeTileB}, smB, k_outer);
    auto doubleLoadTileA = Rewriter::pipeline({loadTileA, storeTileA}, smA, k_outer);
    LOG_DEBUG("===== pipeline 1 =======\n",module);
    auto doubleLoadFragB = Rewriter::pipeline({loadFragB}, fragB, k_inner);
    auto doubleLoadFragA = Rewriter::pipeline({loadFragA}, fragA, k_inner);
    LOG_DEBUG("===== k_inner =======\n",k_inner);
    LOG_DEBUG("===== pipeline 2 =======\n",module);
>>>>>>> dev_bizefeng
    // module.dump();
    // mlir::PassManager pm { module.getContext() };
    // pm.addPass(mlir::createSymbolDCEPass());
    // pm.addPass(mlir::createCSEPass());
    // pm.addPass(mlir::createCanonicalizerPass());
    // pm.run(module);
    // LOG_DEBUG("===== after DCE =======\n",module);

<<<<<<< HEAD
    
    LOG_DEBUG("===== k_outer =======\n",k_outer);
    auto doubleLoadTileB = Rewriter::pipeline({loadTileB, storeTileB}, smB, k_outer, "smB");
    auto doubleLoadTileA = Rewriter::pipeline({loadTileA, storeTileA}, smA, k_outer, "smA");
    LOG_DEBUG("===== after sm pipeline =======\n",module);
    LOG_DEBUG("===== k_inner =======\n",k_inner);
    auto doubleLoadFragB = Rewriter::pipeline({loadFragB}, regB, k_mider, "regB");
    auto doubleLoadFragA = Rewriter::pipeline({loadFragA}, regA, k_mider, "regA");
    LOG_DEBUG("===== after reg pipeline =======\n",module);
    // // module.dump();


    Rewriter::detach_last_loop(k_mider);

=======
    Rewriter::detach_last_loop(k_inner);
    LOG_DEBUG("===== detach_last_loop =======\n",module);
    // gma->regmovea 放在 gmb->regmoveb 位置的before
>>>>>>> dev_bizefeng
    Rewriter::schedule(doubleLoadTileA[0][0], doubleLoadTileB[0][0], Position::before);
    LOG_DEBUG("===== schedule_0 =======\n",module);
    // regmovea->shma 放在 regmoveb->shmb 位置的before
    Rewriter::schedule(doubleLoadTileA[0][1], doubleLoadTileB[0][1], Position::before); 
    LOG_DEBUG("===== schedule_1 =======\n",module);
    // gpuBarrierPrefix->shma 放在 regmoveb->shmb 位置的after
    Rewriter::schedule(gpuBarrierPrefix, doubleLoadTileB[0][1], Position::after);
    LOG_DEBUG("===== schedule_2 =======\n",module);
    // gma->regmovea 放在 gmb->regmoveb 位置的before(小k循环内)
    Rewriter::schedule(doubleLoadTileA[1][0], doubleLoadTileB[1][0], Position::before);
    LOG_DEBUG("===== schedule_3 =======\n",module);
    // regmovea->shma 放在 regmoveb->shmb 位置的before(小k循环内)
    Rewriter::schedule(doubleLoadTileA[1][1], doubleLoadTileB[1][1], Position::before);
    LOG_DEBUG("===== schedule_4 =======\n",module);
    // gpuBarrierPrefix->shma 放在 regmoveb->shmb 位置的after(小k循环内)
    // gpuBarrierSuffix.erase(); ppl0107_1
    Rewriter::schedule(gpuBarrierSuffix, k_inner, Position::after); // ppl 0107_2
    // Rewriter::schedule(gpuBarrierSuffix, doubleLoadTileB[1][1], Position::after);
    LOG_DEBUG("===== schedule_5 =======\n",module);
    auto ifOp = doubleLoadTileA[1][1]->getParentOp();
<<<<<<< HEAD
    Rewriter::schedule(ifOp, k_mider, Position::after); 
=======
    Rewriter::schedule(ifOp, k_inner, Position::after); 
    LOG_DEBUG("===== first schedule =======\n",module);
>>>>>>> dev_bizefeng
    Rewriter::extract_loop(doubleLoadFragA[0][0], k_outer, /*iteration*/0);
    Rewriter::extract_loop(doubleLoadFragB[0][0], k_outer, /*iteration*/0);
    LOG_DEBUG("===== extract_loop =======\n",module);
    Rewriter::schedule(doubleLoadFragB[0][0], k_outer, Position::end);
    Rewriter::schedule(doubleLoadFragA[0][0], k_outer, Position::end);
<<<<<<< HEAD
    // // module.dump();
=======
    LOG_DEBUG("===== second schedule =======\n",module);
    // module.dump();
>>>>>>> dev_bizefeng

    Rewriter::change_double_buffer(doubleLoadFragA[0][0], smA);
    Rewriter::change_double_buffer(doubleLoadFragB[0][0], smB);

    if (config["LOCAL_SPLIT_U"] > 1) {
      auto smC = Rewriter::searchDescOp(module, "smC");
      auto regC_ = Rewriter::searchDescOp(module, "regC_");
      auto smCombined = Rewriter::bufferCombine({{smA, smB}, {smC}});
      tools::_opSetDescription(smCombined.getDefiningOp(),"smCombined");
      auto regCombined = Rewriter::bufferCombine({{regC}, {regC_}});
      tools::_opSetDescription(regCombined.getDefiningOp(),"regCombined");
      LOG_DEBUG("===== bufferCombine =======\n",module);
    }

    Rewriter::take_off_true_if(module);
    Rewriter::delete_false_if(module);
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

    Rewriter::unrollAttribute(module, [&](mlir::affine::AffineForOp forOp)->bool {
      if (!forOp.hasConstantBounds()) return false;
      auto step = forOp.getStep().getLimitedValue();
      auto ub = forOp.getConstantUpperBound();
      auto lb = forOp.getConstantLowerBound();
      auto times = (ub - lb) / step;
      if (times > 16) return false;
      return true;
    });
    LOG_DEBUG("===== unrollAttribute =======\n",module);
  }
}

}

