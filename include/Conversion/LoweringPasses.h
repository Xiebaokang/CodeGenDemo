#pragma once

#include "utils.h"

using namespace mlir;

namespace KernelCodeGen {

std::unique_ptr<OperationPass<ModuleOp>> createParallelToROCDLPass();

std::unique_ptr<OperationPass<ModuleOp>> createROCDLIdOpModifyPass();  // no use

std::unique_ptr<OperationPass<ModuleOp>> createEraseRedundantUnCCastPass();

std::unique_ptr<OperationPass<ModuleOp>> createConvertArithIndexToI64Pass();

std::unique_ptr<OperationPass<ModuleOp>> createAffineFullUnrollPass();

}