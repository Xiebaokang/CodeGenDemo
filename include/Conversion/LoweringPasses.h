#pragma once

#include "utils.h"

using namespace mlir;

namespace KernelCodeGen {

static const int INDEX_BIT_WIDTH = 32;

std::unique_ptr<OperationPass<ModuleOp>> createParallelToROCDLPass();

std::unique_ptr<OperationPass<ModuleOp>> createROCDLIdOpModifyPass();  // no use

std::unique_ptr<OperationPass<ModuleOp>> createEraseRedundantUnCCastPass();

std::unique_ptr<OperationPass<ModuleOp>> createConvertArithIndexToI64Pass();

std::unique_ptr<OperationPass<ModuleOp>> createAffineFullUnrollPass();

std::unique_ptr<OperationPass<ModuleOp>> createVectorToLLVMPass(unsigned indexBitWidth);

std::unique_ptr<OperationPass<ModuleOp>> createModifyMallocFuncAndCallPass();

}