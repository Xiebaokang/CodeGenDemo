#ifndef _Rewriter_h_
#define _Rewriter_h_

#include "Common/Utils.h"
#include "Analysis/Analyzer.h"
#include "mlir/Support/LLVM.h"
#include <vector>

namespace KernelCodeGen {
namespace Rewriter {

mlir::OpBuilder getBuilder(mlir::affine::AffineForOp op, Position pos);

std::vector<mlir::Value> getParallelIdx(mlir::affine::AffineParallelOp parallelLevel);

std::vector<mlir::affine::AffineForOp> split(
  mlir::affine::AffineForOp forOp, uint64_t num_output, std::vector<int64_t> &&factors);

std::vector<mlir::affine::AffineForOp> localSplitU(mlir::affine::AffineForOp forOp, uint64_t num_output);

mlir::Value bufferizeLoopCarryVar(std::vector<mlir::affine::AffineForOp> &loops);

void reorder(const std::vector<mlir::affine::AffineForOp> &forOp);

mlir::affine::AffineParallelOp parallel(const std::vector<mlir::affine::AffineForOp> &forOp);

template <typename ParentOpType>
mlir::Value alloc_buffer(
  ParentOpType father, MemorySpace ms, const std::vector<int64_t> shape_, mlir::Type dtype)
{
  llvm::ArrayRef<int64_t> shape(shape_);
  int64_t flatSize = 1;
  for (auto dim : shape){
    flatSize *= dim;
  }
  llvm::ArrayRef<int64_t> flatShape{1, flatSize};
  mlir::MemRefType tensorShape = mlir::MemRefType::get(
      shape, dtype, {}, static_cast<int>(ms));

  mlir::OpBuilder builder(father->getContext());
  builder.setInsertionPointToStart(father.getBody());
  if (ms == MemorySpace::local)
  {
    // register alloc
    auto op = builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), tensorShape);
    op.setAlignment(KCG_ALIGNBYTE);
    return op->getResult(0);
  }
  else
  {
    // shm alloc
    auto op = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), tensorShape);
    op.setAlignment(KCG_ALIGNBYTE);
    return op->getResult(0);
  }
}

mlir::Value _inner_alloc_buffer(mlir::OpBuilder &builder, mlir::MemRefType &type);

template <typename ContextOp>
mlir::Value alloc_buffer(ContextOp contextOp, Position pos, MemorySpace ms,
                          const std::vector<int64_t> shape_, mlir::Type dtype)
{
  llvm::ArrayRef<int64_t> shape(shape_);
  mlir::MemRefType tensorShape = mlir::MemRefType::get(
      shape, dtype, {}, static_cast<int>(ms));

  switch (pos){
  case Position::before:
  {
    mlir::OpBuilder builder(contextOp);
    return _inner_alloc_buffer(builder, tensorShape);
    // return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), tensorShape)->getResult(0);
  }
  case Position::after:
  {
    mlir::OpBuilder builder(contextOp->getContext());
    builder.setInsertionPointAfter(contextOp);
    return _inner_alloc_buffer(builder, tensorShape);
    // return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), tensorShape)->getResult(0);
  }
  default:
  {
    assert(false && "alloc_buffer error!");
  }
  }
}

mlir::affine::AffineForOp read(mlir::Value src, mlir::Value dst, mlir::AffineMap map,
                                llvm::SmallVector<mlir::Value> operands, int64_t width,
                                mlir::affine::AffineForOp compute_at, Position pos);

mlir::affine::AffineForOp read(mlir::OpBuilder &builder, mlir::Value src, mlir::Value dst,
                                mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, int64_t width);

mlir::affine::AffineForOp write(mlir::Value src, mlir::Value dst, mlir::AffineMap map,
                                llvm::SmallVector<mlir::Value> operands, int64_t width,
                                mlir::affine::AffineForOp compute_at, Position pos);

mlir::affine::AffineForOp write(mlir::OpBuilder &builder, mlir::Value src, mlir::Value dst,
                                mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, int64_t width);

mlir::gpu::BarrierOp barrier(mlir::affine::AffineForOp compute_at, Position pos);

mlir::affine::AffineForOp vectorize(mlir::affine::AffineForOp readOrWrite, int64_t width);

void cache_read(mlir::affine::AffineForOp scope, 
    mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands);

void cache_write(mlir::affine::AffineForOp scope, 
    mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands);

std::vector<std::vector<mlir::affine::AffineForOp>> get_write(
  mlir::affine::AffineParallelOp parallelLevel, mlir::Value dst);

std::vector<std::vector<mlir::affine::AffineForOp>> pipeline(
  std::vector<mlir::affine::AffineForOp> readBodys, mlir::Value &buffer, mlir::affine::AffineForOp compute_at);

void change_double_buffer(mlir::affine::AffineForOp, mlir::Value buffer);

void detach_last_loop(mlir::affine::AffineForOp forOp);

void schedule(mlir::Operation *srcOp, mlir::Operation *dstOp, Position pos);

void extract_loop(mlir::Operation *srcOp, mlir::affine::AffineForOp forOp, int64_t iteration);

void take_off_true_if(mlir::ModuleOp module);

void delete_false_if(mlir::ModuleOp module);

void unroll(mlir::ModuleOp module, mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn);

void unrollAttribute(mlir::ModuleOp module, mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn);

}  // rewriter

}  // KernelCodeGen

#endif // _Rewriter_h_