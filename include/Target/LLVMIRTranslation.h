#ifndef _LLVMIRTranslation_h_
#define _LLVMIRTranslation_h_

#include "utils.h"

namespace KernelCodeGen {

std::unique_ptr<llvm::Module> translateModuleToLLVMIR(mlir::ModuleOp module);

void optimizeLLVMIRModule(
    llvm::Module* llvmModule,
    llvm::DenseMap<llvm::StringRef, NVVMMetadata>* nvvmMetadata,
    KernelCodeGen::Target target
);

void getNVVMMetaData(mlir::ModuleOp& module,llvm::DenseMap<llvm::StringRef, NVVMMetadata>* meta);

}
#endif