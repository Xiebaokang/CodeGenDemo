#ifndef _LLVMIRTranslation_h_
#define _LLVMIRTranslation_h_

#include "utils.h"

namespace KernelCodeGen {

std::string translateMLIRToLLVMIR(mlir::ModuleOp module);

void optimizeLLVMIRModule(
    llvm::Module* llvmModule,
    llvm::DenseMap<llvm::StringRef, NVVMMetadata>* nvvmMetadata,
    KernelCodeGen::Target target
);

void getNVVMMetaData(mlir::ModuleOp& module,llvm::DenseMap<llvm::StringRef, NVVMMetadata>* meta);


std::string tranlateAndSaveLLVMIR(mlir::ModuleOp module);
}
#endif