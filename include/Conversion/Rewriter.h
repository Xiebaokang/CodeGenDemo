#pragma once

#include "utils.h"
#include "Analysis/Analyzer.h"

#include "mlir/Support/LLVM.h"

#include <vector>

namespace KernelCodeGen {

struct Rewriter {
  Rewriter() = default;

  static mlir::OpBuilder getBuilder(mlir::affine::AffineForOp op, Position pos) {
    switch (pos) {
      case Position::after: {
        mlir::OpBuilder builder(op->getContext());
        builder.setInsertionPointAfter(op);
        return builder;
      }
      case Position::before: {
        mlir::OpBuilder builder(op);
        return builder;
      }
      case Position::begin: {
        return mlir::OpBuilder::atBlockBegin(op.getBody());
      }
      case Position::end: {
        return mlir::OpBuilder::atBlockEnd(op.getBody());
      }
      default:
        assert(false);
    } 
  }

  static std::vector<mlir::Value> getParallelIdx(mlir::affine::AffineParallelOp parallelLevel) {
    auto dim = parallelLevel.getNumDims();
    std::vector<mlir::Value> idxes;
    auto ivs = parallelLevel.getIVs();
    for (auto iv : ivs) {
      idxes.push_back(iv);
    }
    return idxes;
  }

  static std::vector<mlir::affine::AffineForOp> split(mlir::affine::AffineForOp forOp, uint64_t num_output, std::vector<int64_t>&& factors);
  
  static mlir::Value bufferizeLoopCarryVar(std::vector<mlir::affine::AffineForOp>& loops);

  static void reorder(const std::vector<mlir::affine::AffineForOp>& forOp);

  static mlir::affine::AffineParallelOp parallel(const std::vector<mlir::affine::AffineForOp>& forOp);

  template<typename ParentOpType>
  static mlir::Value alloc_buffer(ParentOpType father, MemorySpace ms, 
                          const std::vector<int64_t> shape_, mlir::Type dtype) {
    llvm::ArrayRef<int64_t> shape (shape_);
    mlir::MemRefType tensorShape = mlir::MemRefType::get(
      shape, dtype, {}, static_cast<int>(ms));
    
    mlir::OpBuilder builder(father->getContext());
    builder.setInsertionPointToStart(father.getBody());
    if(ms == MemorySpace::local){
      return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), tensorShape)->getResult(0);
    }
    else{
      return builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), tensorShape)->getResult(0);
    }
  }

  static mlir::Value _inner_alloc_buffer(mlir::OpBuilder& builder,mlir::MemRefType& type){
    if(type.getMemorySpaceAsInt() == int(KernelCodeGen::MemorySpace::local)){
      return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), type)->getResult(0);
    }
    return builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(),type);
    // return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), type)->getResult(0);
  }

  template<typename ContextOp>
  static mlir::Value alloc_buffer(ContextOp contextOp, Position pos, MemorySpace ms, 
                          const std::vector<int64_t> shape_, mlir::Type dtype) {
    llvm::ArrayRef<int64_t> shape (shape_);
    mlir::MemRefType tensorShape = mlir::MemRefType::get(
      shape, dtype, {}, static_cast<int>(ms));
    
    switch (pos) {
      case Position::before: {
        mlir::OpBuilder builder(contextOp);
        return _inner_alloc_buffer(builder,tensorShape);
        // return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), tensorShape)->getResult(0);
      }
      case Position::after: {
        mlir::OpBuilder builder(contextOp->getContext());
        builder.setInsertionPointAfter(contextOp);
        return _inner_alloc_buffer(builder,tensorShape);
        // return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), tensorShape)->getResult(0);
      }
      default: {
        assert(false);
      }
    }
  }


  static mlir::affine::AffineForOp read(mlir::Value src, mlir::Value dst, mlir::AffineMap map, 
                                   llvm::SmallVector<mlir::Value> operands, int64_t width,
                                   mlir::affine::AffineForOp compute_at, Position pos);

  static mlir::affine::AffineForOp read(mlir::OpBuilder& builder, mlir::Value src, mlir::Value dst, 
    mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, int64_t width);

  static mlir::affine::AffineForOp write(mlir::Value src, mlir::Value dst, mlir::AffineMap map, 
                                   llvm::SmallVector<mlir::Value> operands, int64_t width,
                                   mlir::affine::AffineForOp compute_at, Position pos);

  static mlir::affine::AffineForOp write(mlir::OpBuilder& builder, mlir::Value src, mlir::Value dst, 
    mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, int64_t width);

  static mlir::gpu::BarrierOp barrier(mlir::affine::AffineForOp compute_at, Position pos);

  static mlir::affine::AffineForOp vectorize(mlir::affine::AffineForOp readOrWrite, int64_t width);

  static void cache_read(mlir::affine::AffineForOp scope, mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands);

  static void cache_write(mlir::affine::AffineForOp scope, mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands);

  static std::vector<std::vector<mlir::affine::AffineForOp>> get_write(mlir::affine::AffineParallelOp parallelLevel, mlir::Value dst);

  static std::vector<std::vector<mlir::affine::AffineForOp>> pipeline(std::vector<mlir::affine::AffineForOp> readBodys, mlir::Value& buffer, mlir::affine::AffineForOp compute_at);

  static void change_double_buffer(mlir::affine::AffineForOp, mlir::Value buffer);

  static void detach_last_loop(mlir::affine::AffineForOp forOp);

  static void schedule(mlir::Operation* srcOp, mlir::Operation* dstOp, Position pos);

  static void extract_loop(mlir::Operation* srcOp, mlir::affine::AffineForOp forOp, int64_t iteration);

  static void take_off_true_if(mlir::ModuleOp module);

  static void delete_false_if(mlir::ModuleOp module);

  static void replace_alloc_shm(mlir::ModuleOp module) ;

  static void unroll(mlir::ModuleOp module, mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn);

  static void unrollAttribute(mlir::ModuleOp module, mlir::function_ref<bool(mlir::affine::AffineForOp)> unrollCheckFn);

};

}