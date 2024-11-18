#ifndef _LLVMIRTranslation_h_
#define _LLVMIRTranslation_h_

#include "utils.h"

namespace KernelCodeGen {

std::string translateMLIRToLLVMIR(mlir::ModuleOp module, Target target, const int wavesPerEU);

std::string tranlateAndSaveLLVMIR(mlir::ModuleOp module);
}
#endif