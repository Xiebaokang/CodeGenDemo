#include "Target/LLVMIRTranslation.h"

namespace KernelCodeGen {

std::unique_ptr<llvm::Module> translateModuleToLLVMIR(mlir::ModuleOp module) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);   // 注册
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);

  module->getContext()->appendDialectRegistry(registry);

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return nullptr;
  }
  return llvmModule;
}

}