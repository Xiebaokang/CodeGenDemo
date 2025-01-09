#include "Conversion/Optimizer.h"
#include <cfloat>
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include <filesystem>
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"

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

mlir::AffineMap MatmulOptimizer::affineMap_loadtileA(mlir::OpBuilder& builder, const std::map<std::string, int>& config)
{
  int dimCount = 0;
  auto dim_k = builder.getAffineDimExpr(dimCount++);
  auto dim_by = builder.getAffineDimExpr(dimCount++);
  auto dim_iter = builder.getAffineDimExpr(dimCount++);
  auto dim_tid = builder.getAffineDimExpr(dimCount++);
  
  const int& BM = config.at(KEY_BLOCK_SIZE_M); 
  const int& BN = config.at(KEY_BLOCK_SIZE_N); 
  const int& BK = config.at(KEY_BLOCK_SIZE_K);
  const int& TM = config.at(KEY_THREAD_SIZE_M);
  const int& TN = config.at(KEY_THREAD_SIZE_N);
  const int BLOCK_Y = BM / TM;
  const int BLOCK_X = BN / TN;
  const int THREAD_NUM = BLOCK_X * BLOCK_Y * config.at(KEY_LOCAL_SPLIT_U);
  const int SHARED_SIZE_A = BM * BK;
  const int SHARED_SIZE_B = BN * BK;
  // glob -> reg -> shared 
  const int GLOB_LOAD_ROW_WIDTH_A = THREAD_NUM / BK * config.at(KEY_GLOB_LOAD_WIDTH_A);
  const int GLOB_LOAD_ROW_WIDTH_B = THREAD_NUM / BK * config.at(KEY_GLOB_LOAD_WIDTH_B);
  // shared -> reg
  const int BLOCK_REPEAT_A = TM / config.at(KEY_WARP_SCATTER_WIDTH_A);
  const int WARP_REPEAT_A = config.at(KEY_WARP_SCATTER_WIDTH_A) / config.at(KEY_THREAD_SCATTER_WIDTH_A);
  const int BLOCK_REPEAT_B = TN / config.at(KEY_WARP_SCATTER_WIDTH_B);
  const int WARP_REPEAT_B = config.at(KEY_WARP_SCATTER_WIDTH_B) / config.at(KEY_THREAD_SCATTER_WIDTH_B);
  // reduce C (sharedC to regC)
  const int GLOB_STORE_ROW_WIDTH = THREAD_NUM / BM * config.at(KEY_GLOB_STORE_WIDTH);

  llvm::SmallVector<mlir::AffineExpr> exprs;
    // ***** load glob to temp reg *****
  auto sh_load_row = dim_tid.floorDiv((THREAD_NUM / BK));
  auto sh_load_col = dim_tid % (THREAD_NUM / BK);

  // A[sh_load_row + k][(by * BM) + (iter * GLOB_LOAD_ROW_WIDTH_A) + (sh_load_col * GLOB_LOAD_WIDTH_A)]
  exprs.push_back(sh_load_row + dim_k);
  exprs.push_back(dim_by * BM + dim_iter * GLOB_LOAD_ROW_WIDTH_A + sh_load_col * config.at(KEY_GLOB_LOAD_WIDTH_A));
 
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());

}

mlir::AffineMap MatmulOptimizer::affineMap_loadtileB(mlir::OpBuilder& builder, const std::map<std::string, int>& config)
{
  int dimCount = 0;
  auto dim_k = builder.getAffineDimExpr(dimCount++);
  auto dim_bx = builder.getAffineDimExpr(dimCount++);
  auto dim_iter = builder.getAffineDimExpr(dimCount++);
  auto dim_tid = builder.getAffineDimExpr(dimCount++);
  
  const int& BM = config.at(KEY_BLOCK_SIZE_M); 
  const int& BN = config.at(KEY_BLOCK_SIZE_N); 
  const int& BK = config.at(KEY_BLOCK_SIZE_K);
  const int& TM = config.at(KEY_THREAD_SIZE_M);
  const int& TN = config.at(KEY_THREAD_SIZE_N);
  const int BLOCK_Y = BM / TM;
  const int BLOCK_X = BN / TN;
  const int THREAD_NUM = BLOCK_X * BLOCK_Y * config.at(KEY_LOCAL_SPLIT_U);
  const int SHARED_SIZE_A = BM * BK;
  const int SHARED_SIZE_B = BN * BK;
  // glob -> reg -> shared 
  const int GLOB_LOAD_ROW_WIDTH_A = THREAD_NUM / BK * config.at(KEY_GLOB_LOAD_WIDTH_A);
  const int GLOB_LOAD_ROW_WIDTH_B = THREAD_NUM / BK * config.at(KEY_GLOB_LOAD_WIDTH_B);
  // shared -> reg
  const int BLOCK_REPEAT_A = TM / config.at(KEY_WARP_SCATTER_WIDTH_A);
  const int WARP_REPEAT_A = config.at(KEY_WARP_SCATTER_WIDTH_A) / config.at(KEY_THREAD_SCATTER_WIDTH_A);
  const int BLOCK_REPEAT_B = TN / config.at(KEY_WARP_SCATTER_WIDTH_B);
  const int WARP_REPEAT_B = config.at(KEY_WARP_SCATTER_WIDTH_B) / config.at(KEY_THREAD_SCATTER_WIDTH_B);
  // reduce C (sharedC to regC)
  const int GLOB_STORE_ROW_WIDTH = THREAD_NUM / BM * config.at(KEY_GLOB_STORE_WIDTH);

  llvm::SmallVector<mlir::AffineExpr> exprs;
      // ***** load glob to temp reg *****
  auto sh_load_row = dim_tid.floorDiv((THREAD_NUM / BK));
  auto sh_load_col = dim_tid % (THREAD_NUM / BK);

  // B[sh_load_row + k][(bx * BN) + (iter * GLOB_LOAD_ROW_WIDTH_B) + (sh_load_col * GLOB_LOAD_WIDTH_B)]
  exprs.push_back(sh_load_row + dim_k);
  exprs.push_back(dim_bx * BN + dim_iter * GLOB_LOAD_ROW_WIDTH_B + sh_load_col * config.at(KEY_GLOB_LOAD_WIDTH_B));

  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap MatmulOptimizer::affineMap_storetileA(mlir::OpBuilder& builder, const std::map<std::string, int>& config)
{
  int dimCount = 0;
  auto dim_tid = builder.getAffineDimExpr(dimCount++);
  auto dim_iter = builder.getAffineDimExpr(dimCount++);
  
  const int& BM = config.at(KEY_BLOCK_SIZE_M); 
  const int& BN = config.at(KEY_BLOCK_SIZE_N); 
  const int& BK = config.at(KEY_BLOCK_SIZE_K);
  const int& TM = config.at(KEY_THREAD_SIZE_M);
  const int& TN = config.at(KEY_THREAD_SIZE_N);
  const int BLOCK_Y = BM / TM;
  const int BLOCK_X = BN / TN;
  const int THREAD_NUM = BLOCK_X * BLOCK_Y * config.at(KEY_LOCAL_SPLIT_U);
  const int SHARED_SIZE_A = BM * BK;
  const int SHARED_SIZE_B = BN * BK;
  // glob -> reg -> shared 
  const int GLOB_LOAD_ROW_WIDTH_A = THREAD_NUM / BK * config.at(KEY_GLOB_LOAD_WIDTH_A);
  const int GLOB_LOAD_ROW_WIDTH_B = THREAD_NUM / BK * config.at(KEY_GLOB_LOAD_WIDTH_B);
  // shared -> reg
  const int BLOCK_REPEAT_A = TM / config.at(KEY_WARP_SCATTER_WIDTH_A);
  const int WARP_REPEAT_A = config.at(KEY_WARP_SCATTER_WIDTH_A) / config.at(KEY_THREAD_SCATTER_WIDTH_A);
  const int BLOCK_REPEAT_B = TN / config.at(KEY_WARP_SCATTER_WIDTH_B);
  const int WARP_REPEAT_B = config.at(KEY_WARP_SCATTER_WIDTH_B) / config.at(KEY_THREAD_SCATTER_WIDTH_B);
  // reduce C (sharedC to regC)
  const int GLOB_STORE_ROW_WIDTH = THREAD_NUM / BM * config.at(KEY_GLOB_STORE_WIDTH);

  llvm::SmallVector<mlir::AffineExpr> exprs;
    // ***** load temp reg to shared *****
  auto sh_load_row = dim_tid.floorDiv((THREAD_NUM / BK));
  auto sh_load_col = dim_tid % (THREAD_NUM / BK);
    // sh_A[sh_load_row][(iter * GLOB_LOAD_ROW_WIDTH_A) + (sh_load_col * GLOB_LOAD_WIDTH_A)]
  exprs.push_back(sh_load_row);
  exprs.push_back(dim_iter * GLOB_LOAD_ROW_WIDTH_A + sh_load_col * config.at(KEY_GLOB_LOAD_WIDTH_A));
  
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 

}
mlir::AffineMap MatmulOptimizer::affineMap_storetileB(mlir::OpBuilder& builder, const std::map<std::string, int>& config)
{
  // todo
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
  auto dim8 = builder.getAffineDimExpr(8);

  const int& BM = config[KEY_BLOCK_SIZE_M]; 
  const int& BN = config[KEY_BLOCK_SIZE_N]; 
  const int& BK = config[KEY_BLOCK_SIZE_K];
  const int& TM = config[KEY_THREAD_SIZE_M];
  const int& TN = config[KEY_THREAD_SIZE_N];

  const int BLOCK_Y = BM / TM;
  const int BLOCK_X = BN / TN;
  const int THREAD_NUM = BLOCK_X * BLOCK_Y * config["LOCAL_SPLIT_U"];
  const int SHARED_SIZE_A = BM * BK;
  const int SHARED_SIZE_B = BN * BK;

  // glob -> reg -> shared 
  // const int GLOB_LOAD_NUM_A = SHARED_SIZE_A / THREAD_NUM / config["GLOB_LOAD_WIDTH_A"];
  // const int GLOB_LOAD_NUM_B = SHARED_SIZE_B / THREAD_NUM / config["GLOB_LOAD_WIDTH_B"];
  const int GLOB_LOAD_ROW_WIDTH_A = THREAD_NUM / BK * config["GLOB_LOAD_WIDTH_A"];
  const int GLOB_LOAD_ROW_WIDTH_B = THREAD_NUM / BK * config["GLOB_LOAD_WIDTH_B"];

  // shared -> reg
  const int BLOCK_REPEAT_A = TM / config["WARP_SCATTER_WIDTH_A"];
  const int WARP_REPEAT_A = config["WARP_SCATTER_WIDTH_A"] / config["THREAD_SCATTER_WIDTH_A"];
  const int BLOCK_REPEAT_B = TN / config["WARP_SCATTER_WIDTH_B"];
  const int WARP_REPEAT_B = config["WARP_SCATTER_WIDTH_B"] / config["THREAD_SCATTER_WIDTH_B"];

  // reduce C (sharedC to regC)
  const int GLOB_STORE_ROW_WIDTH = THREAD_NUM / BM * config["GLOB_STORE_WIDTH"];


  llvm::SmallVector<mlir::AffineExpr> exprs;
  if (mapIdentifier == "loadTileA" || mapIdentifier == "loadTileB") {
    // ***** load glob to temp reg *****
    auto sh_load_row = dim0.floorDiv((THREAD_NUM / BK));
    auto sh_load_col = dim0 % (THREAD_NUM / BK);
    if (mapIdentifier == "loadTileA") {
      // A[sh_load_row + k][(by * BM) + (iter * GLOB_LOAD_ROW_WIDTH_A) + (sh_load_col * GLOB_LOAD_WIDTH_A)]
      exprs.push_back(sh_load_row + dim1);
      exprs.push_back(dim2 * BM + dim3 * GLOB_LOAD_ROW_WIDTH_A + sh_load_col * config["GLOB_LOAD_WIDTH_A"]);
    } else {
      // B[sh_load_row + k][(bx * BN) + (iter * GLOB_LOAD_ROW_WIDTH_B) + (sh_load_col * GLOB_LOAD_WIDTH_B)]
      exprs.push_back(sh_load_row + dim1);
      exprs.push_back(dim2 * BN + dim3 * GLOB_LOAD_ROW_WIDTH_B + sh_load_col * config["GLOB_LOAD_WIDTH_B"]);
    }
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "storeTileA" || mapIdentifier == "storeTileB") {
    // ***** load temp reg to shared *****
    auto sh_load_row = dim0.floorDiv((THREAD_NUM / BK));
    auto sh_load_col = dim0 % (THREAD_NUM / BK);
    if (mapIdentifier == "storeTileA") {
      // sh_A[sh_load_row][(iter * GLOB_LOAD_ROW_WIDTH_A) + (sh_load_col * GLOB_LOAD_WIDTH_A)]
      exprs.push_back(sh_load_row);
      exprs.push_back(dim1 * GLOB_LOAD_ROW_WIDTH_A + sh_load_col * config["GLOB_LOAD_WIDTH_A"]);
    } else {
      // sh_B[sh_load_row][(iter * GLOB_LOAD_ROW_WIDTH_B) + (sh_load_col * GLOB_LOAD_WIDTH_B)]
      exprs.push_back(sh_load_row);
      exprs.push_back(dim1 * GLOB_LOAD_ROW_WIDTH_B + sh_load_col * config["GLOB_LOAD_WIDTH_B"]);
    }
    return mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadFragA" || mapIdentifier == "loadFragB") {
    // ***** load shared to reg *****
    // thread idx
    auto tz = dim1.floorDiv(BLOCK_X * BLOCK_Y);
    auto tid_other = dim1 % (BLOCK_X * BLOCK_Y);
    // thread mapping
    auto warp_id = tid_other.floorDiv(config["WARP_SIZE"]);
    auto lane_id = tid_other % config["WARP_SIZE"];
    auto warp_y = warp_id.floorDiv(config["BLOCK_LAYOUT_N"]);
    auto warp_x = warp_id % config["BLOCK_LAYOUT_N"];
    auto lane_y = lane_id.floorDiv(config["WARP_LAYOUT_N"]);
    auto lane_x = lane_id % config["WARP_LAYOUT_N"];
    if (mapIdentifier == "loadFragA") {
      // sh_A[bk + tz][(i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]
      exprs.push_back(dim0 + tz);
      exprs.push_back((dim2 * config["BLOCK_LAYOUT_M"] + warp_y) * config["WARP_LAYOUT_M"] * config["WARP_SCATTER_WIDTH_A"] + 
                      (dim3 * config["WARP_LAYOUT_M"] + lane_y) * config["THREAD_SCATTER_WIDTH_A"]);
    } else {
      // sh_B[bk + tz][(i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]
      exprs.push_back(dim0 + tz);
      exprs.push_back((dim2 * config["BLOCK_LAYOUT_N"] + warp_x) * config["WARP_LAYOUT_N"] * config["WARP_SCATTER_WIDTH_B"] + 
                      (dim3 * config["WARP_LAYOUT_N"] + lane_x) * config["THREAD_SCATTER_WIDTH_B"]);
    }
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "cacheReadA" || mapIdentifier == "cacheReadB") {
    // ***** count result & store to regC *****
    exprs.push_back(dim0);
    return mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "cacheWriteC") {
    // ***** load regC to globC *****
    // thread mapping
    auto warp_id = dim2.floorDiv(config["WARP_SIZE"]);
    auto lane_id = dim2 % config["WARP_SIZE"];
    auto warp_y = warp_id.floorDiv(config["BLOCK_LAYOUT_N"]);
    auto warp_x = warp_id % config["BLOCK_LAYOUT_N"];
    auto lane_y = lane_id.floorDiv(config["WARP_LAYOUT_N"]);
    auto lane_x = lane_id % config["WARP_LAYOUT_N"];
    // C[(by * BM + (i0 * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A + k)]
    //  [bx * BN + (i1 * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]]
    exprs.push_back(dim0 * BM + (dim3 * config["BLOCK_LAYOUT_M"] * config["WARP_LAYOUT_M"]) + (warp_y * config["WARP_SCATTER_WIDTH_A"]) + 
                                (dim4 * config["WARP_LAYOUT_M"]) + (lane_y * config["THREAD_SCATTER_WIDTH_A"]) + dim5);
    exprs.push_back(dim1 * BN + (dim6 * config["BLOCK_LAYOUT_N"] * config["WARP_LAYOUT_N"]) + (warp_x * config["WARP_SCATTER_WIDTH_B"]) + 
                                (dim7 * config["WARP_LAYOUT_N"]) + (lane_x * config["THREAD_SCATTER_WIDTH_B"]) + dim8);
    return mlir::AffineMap::get(/*dimCount*/9, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "cacheWriteShC") {
    // load regC to sharedC
    // thread idx
    auto tz = dim0.floorDiv(BLOCK_X * BLOCK_Y);
    auto tid_other = dim0 % (BLOCK_X * BLOCK_Y);
    // thread mapping
    auto warp_id = tid_other.floorDiv(config["WARP_SIZE"]);
    auto lane_id = tid_other % config["WARP_SIZE"];
    auto warp_y = warp_id.floorDiv(config["BLOCK_LAYOUT_N"]);
    auto warp_x = warp_id % config["BLOCK_LAYOUT_N"];
    auto lane_y = lane_id.floorDiv(config["WARP_LAYOUT_N"]);
    auto lane_x = lane_id % config["WARP_LAYOUT_N"];
    exprs.push_back(tz);
    exprs.push_back((dim1 * config["BLOCK_LAYOUT_M"] * config["WARP_LAYOUT_M"]) + (warp_y * config["WARP_SCATTER_WIDTH_A"]) + 
                    (dim2 * config["WARP_LAYOUT_M"]) + (lane_y * config["THREAD_SCATTER_WIDTH_A"]) + dim3);
    exprs.push_back((dim4 * config["BLOCK_LAYOUT_N"] * config["WARP_LAYOUT_N"]) + (warp_x * config["WARP_SCATTER_WIDTH_B"]) + 
                    (dim5 * config["WARP_LAYOUT_N"]) + (lane_x * config["THREAD_SCATTER_WIDTH_B"]) + dim6);
    return mlir::AffineMap::get(/*dimCount*/7, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "reduceC") {
    // reduce C (sharedC add to regC)
    auto sh_store_row = dim0.floorDiv((THREAD_NUM / BM));
    auto sh_store_col = dim0 % ((THREAD_NUM / BM));
    exprs.push_back(dim1);
    exprs.push_back(sh_store_row);
    exprs.push_back(dim2 * GLOB_STORE_ROW_WIDTH + sh_store_col * config["GLOB_STORE_WIDTH"]);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "writeC") {
    // regC_ to GlobC
    auto sh_store_row = dim2.floorDiv((THREAD_NUM / BM));
    auto sh_store_col = dim2 % ((THREAD_NUM / BM));
    exprs.push_back(dim0 * BM + sh_store_row);
    exprs.push_back(dim1 * BN + dim3 * GLOB_STORE_ROW_WIDTH + sh_store_col * config["GLOB_STORE_WIDTH"]);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
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
    auto regC = Rewriter::bufferizeLoopCarryVar(loopK, tileCLoops);
    LOG_DEBUG("===== after bufferizeLoopCarryVar =======\n",module);

    auto k_axes = Rewriter::split(loopK, 3, {config["LOCAL_SPLIT_U"], config["BLOCK_SIZE_K"]});
    auto k_outer = k_axes[0], k_mider = k_axes[1], k_inner = k_axes[2];
    LOG_DEBUG("===== after split =======\n",module);

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
    auto regB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {config["THREAD_SIZE_N"]}, elementB);
    auto regA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {config["THREAD_SIZE_M"]}, elementA);

    auto tempB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {glob_load_total_width_b}, elementB);
    auto tempA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {glob_load_total_width_a}, elementA);
    auto smB = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared, {config["BLOCK_SIZE_K"], config["BLOCK_SIZE_N"]}, elementB);
    auto smA = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared, {config["BLOCK_SIZE_K"], config["BLOCK_SIZE_M"]}, elementA);
    tools::_opSetDescription(smA.getDefiningOp(),"smA");
    tools::_opSetDescription(smB.getDefiningOp(),"smB");
    LOG_DEBUG("===== before alloc_buffer =======\n",module);

    Rewriter::parallelToOneDim(blockLevel);
    LOG_DEBUG("===== before parallelToOneDim =======\n",module);
    
    auto blockIdx = Analyzer::getParallelIdx(gridLevel);
    auto threadIdx = Analyzer::getParallelIdx(blockLevel);
    
    auto loadTileAMap = getAffineMap("loadTileA", builder, config);
    auto loadTileA = Rewriter::read(A, tempA, loadTileAMap, {threadIdx[0], k_outer.getInductionVar(), blockIdx[0]}, 
                                    {config["GLOB_LOAD_WIDTH_A"]}, k_outer, Position::begin);
    auto loadTileBMap = getAffineMap("loadTileB", builder, config);
    auto loadTileB = Rewriter::read(B, tempB, loadTileBMap, {threadIdx[0], k_outer.getInductionVar(), blockIdx[1]}, 
                                    {config["GLOB_LOAD_WIDTH_B"]}, loadTileA, Position::after);
    LOG_DEBUG("===== before read A/B =======\n",module);

    auto storeTileAMap = getAffineMap("storeTileA", builder, config);
    auto storeTileA = Rewriter::write(tempA, smA, storeTileAMap, {threadIdx[0]}, {config["GLOB_LOAD_WIDTH_A"]}, loadTileB, Position::after);
    auto storeTileBMap = getAffineMap("storeTileB", builder, config);
    auto storeTileB = Rewriter::write(tempB, smB, storeTileBMap, {threadIdx[0]}, {config["GLOB_LOAD_WIDTH_B"]}, storeTileA, Position::after);
    auto gpuBarrierPrefix = Rewriter::barrier(loadTileA, Position::before);
    auto gpuBarrierSuffix = Rewriter::barrier(storeTileB, Position::after);
    LOG_DEBUG("===== write A/B =======\n",module);

    auto loadFragAMap = getAffineMap("loadFragA", builder, config);
    auto loadFragA = Rewriter::read(smA, regA, loadFragAMap, {k_mider.getInductionVar(), threadIdx[0]}, 
                                    {config["WARP_SCATTER_WIDTH_A"], config["THREAD_SCATTER_WIDTH_A"]}, k_mider, Position::begin);
    auto loadFragBMap = getAffineMap("loadFragB", builder, config);
    auto loadFragB = Rewriter::read(smB, regB, loadFragBMap, {k_mider.getInductionVar(), threadIdx[0]}, 
                                    {config["WARP_SCATTER_WIDTH_B"], config["THREAD_SCATTER_WIDTH_B"]}, loadFragA, Position::after);
    LOG_DEBUG("===== read sh_A/B =======\n",module);

    Rewriter::cache_read(n_inner, A, regA, getAffineMap("cacheReadA", builder, config), {m_inner.getInductionVar()});
    Rewriter::cache_read(n_inner, B, regB, getAffineMap("cacheReadB", builder, config), {n_inner.getInductionVar()});
    LOG_DEBUG("===== load regA & cache_read =======\n",module);

    auto writeCbody = Rewriter::get_write(blockLevel, C);
    assert(writeCbody.size() == 1);
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
      tools::_opSetDescription(smC.getDefiningOp(),"smC");
      auto cacheWriteShCMap = getAffineMap("cacheWriteShC", builder, config);
      Rewriter::cache_write(m_inner_0, C, smC, cacheWriteShCMap, 
                            {threadIdx[0], m_inner_0.getInductionVar(),m_inner_1.getInductionVar(), m_inner_2.getInductionVar(), 
                            n_inner_0.getInductionVar(),n_inner_1.getInductionVar(), n_inner_2.getInductionVar()});
      LOG_DEBUG("===== load cache_write regC to C =======\n",module);

      auto reduceCMap = getAffineMap("reduceC", builder, config);
      auto reduceCLoop = Rewriter::splitUReduce(smC, regC_, reduceCMap, {threadIdx[0]}, config["LOCAL_SPLIT_U"], config["GLOB_STORE_WIDTH"], m_inner_0, Position::after);
      LOG_DEBUG("===== load splitUReduce =======\n",module);

      auto writeCMap = getAffineMap("writeC", builder, config);
      Rewriter::splitUWrite(regC_, C, writeCMap, {blockIdx[0], blockIdx[1], threadIdx[0]}, config["LOCAL_SPLIT_U"], config["GLOB_STORE_WIDTH"], reduceCLoop, Position::after);
      auto StoreBarrier = Rewriter::barrier(m_inner_0, Position::after);
      LOG_DEBUG("===== load write to C =======\n",module);

      Rewriter::bufferCombine({{smA, smB}, {smC}});
      Rewriter::bufferCombine({{regC}, {regC_}});
      LOG_DEBUG("===== bufferCombine =======\n",module);

    } else {
      auto cacheWriteCMap = getAffineMap("cacheWriteC", builder, config);
      Rewriter::cache_write(m_inner_0, C, C, cacheWriteCMap, 
                            {blockIdx[0], blockIdx[1], threadIdx[0], 
                            m_inner_0.getInductionVar(),m_inner_1.getInductionVar(), m_inner_2.getInductionVar(), 
                            n_inner_0.getInductionVar(),n_inner_1.getInductionVar(), n_inner_2.getInductionVar()});
      LOG_DEBUG("===== load cache_write regC to C =======\n",module);
    }

    Rewriter::vectorize(n_inner_2, config["THREAD_SCATTER_WIDTH_B"]);
    LOG_DEBUG("===== vectorize =======\n",module);

    Rewriter::BlockMapping(gridLevel, config["BLOCK_MAPPING"]);
    LOG_DEBUG("===== BlockMapping gridLevel =======\n",module);
    
    // module.dump();
    mlir::PassManager pm { module.getContext() };
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.run(module);
    LOG_DEBUG("===== after DCE =======\n",module);

    

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

