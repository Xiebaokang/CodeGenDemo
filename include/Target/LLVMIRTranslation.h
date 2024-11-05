#pragma once

#include "utils.h"

namespace KernelCodeGen {

std::unique_ptr<llvm::Module> translateModuleToLLVMIR(mlir::ModuleOp module);

}
