#ifndef _utils_h_
#define _utils_h_
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Tools/ParseUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "llvm/IR/Module.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"


#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"


// reused Dialects
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"

// IR Infrastructure
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"

// LLVM ADT
#include "llvm/ADT/ScopedHashTable.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h.inc"
#include "mlir/Dialect/Affine/Utils.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/GPU/IR/GPUOpsAttributes.h.inc"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "PassDetail.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/DialectConversion.h"

// lowering
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Conversion/Passes.h.inc"
#include "config.h"

// conversion
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

#include <iostream>
#include <memory>
#include <functional>
#include <deque>
#include <string>
#include <random>
#include <cstdio>
#include <sstream>

namespace KernelCodeGen {

enum class Target {
  CUDA = 0,
  ROCm = 1,
};

enum class MemorySpace {
  global = 1,
  shared = 3,
  // local = 0,
  local = 5,
  constant = 4,
  unallocated = 7,
  inplace = 6,
};

enum class Position {
  before = 0,
  after = 1,
  begin = 2,
  end = 3,
};

enum class Layout {
  rowMajor = 0,
  colMajor = 1,
};

struct NVVMMetadata {
  llvm::SmallVector<int, 3> maxntid;
  bool isKernel{};
  // Free to extend with other information.
};


enum class KcgDtype : int {
  float8 = 1,
  float16 = 2,
  float32 = 4,
  float64 = 8,
  float128 = 16,
  int8 = 11,
  int16 = 12,
  int32 = 14,
  int64 = 18
};

static std::ostream& operator<<(std::ostream& s, KcgDtype ty){
  switch(ty){
    case KcgDtype::float8: s << "f8" ; break;
    case KcgDtype::float16: s << "f16" ; break;
    case KcgDtype::float32: s << "f32" ; break;
    case KcgDtype::float64: s << "f64" ; break;
    case KcgDtype::float128: s << "f128" ; break;
    case KcgDtype::int8: s << "i8" ; break;
    case KcgDtype::int16: s << "i16" ; break;
    case KcgDtype::int32: s << "i32" ; break;
    case KcgDtype::int64: s << "i64" ; break;
    default : break;
  }
  return s;
}

#define AttrKernelFunc "nvvm.kernel"
#define AttrVisibility "sym_visibility" 
#define AttrExternLib "kcg.externLibs"

#define SHM_VAR_NAME(i) (std::string("kcg_shm")+std::to_string(i))

#define KCG_ALIGNBYTE 16

std::string getenv(const char *name);

mlir::Type getDType(mlir::OpBuilder& builder, const std::string& dtype);

std::string typeToStr(mlir::Type type);

std::string KcgDtypeToStr(KcgDtype type);

#define  KEY_BLOCK_SIZE_M         "BLOCK_SIZE_M"
#define  KEY_BLOCK_SIZE_N         "BLOCK_SIZE_N"
#define  KEY_BLOCK_SIZE_K         "BLOCK_SIZE_K"
#define  KEY_THREAD_SIZE_M        "THREAD_SIZE_M"
#define  KEY_THREAD_SIZE_N        "THREAD_SIZE_N"
#define  KEY_VECTORIZE_WIDTH      "VECTORIZE_WIDTH"
#define  KEY_WARP_SIZE            "WARP_SIZE"
#define  KEY_BLOCK_LAYOUT_M       "BLOCK_LAYOUT_M"
#define  KEY_BLOCK_LAYOUT_N       "BLOCK_LAYOUT_N"
#define  KEY_WARP_LAYOUT_M        "WARP_LAYOUT_M"
#define  KEY_WARP_LAYOUT_N        "WARP_LAYOUT_N"
#define  KEY_DTYPE_A              "DATATYPE_A"
#define  KEY_DTYPE_B              "DATATYPE_B"
#define  KEY_DTYPE_C              "DATATYPE_C"
#define  KEY_M                    "M_SIZE"
#define  KEY_N                    "N_SIZE"
#define  KEY_K                    "K_SIZE"
#define  KEY_IS_A_TRANSPOSE       "IS_ATRANS"

#define LOG_DEBUG(message,module)  \
{\
  llvm::outs() << message;llvm::outs().flush(); module.dump();\
}


}  // namespace KernelCodeGen end

#endif