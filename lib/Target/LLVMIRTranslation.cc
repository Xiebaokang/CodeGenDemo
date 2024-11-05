#include "Target/LLVMIRTranslation.h"

namespace KernelCodeGen {

std::unique_ptr<llvm::Module> translateModuleToLLVMIR(mlir::ModuleOp module) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);   // 注册
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);

  module->getContext()->appendDialectRegistry(registry);

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return nullptr;
  }
  // // Initialize LLVM targets.
  // llvm::InitializeNativeTarget();
  // llvm::InitializeNativeTargetAsmPrinter();
  // ExecutionEngine::setupTargetTriple(llvmModule.get());

  // /// Optionally run an optimization pipeline over the llvm module.
  // auto optPipeline = makeOptimizingTransformer(/*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  // if (auto err = optPipeline(llvmModule.get())) {
  //   llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
  //   return nullptr;
  // }
  // llvm::errs() << *llvmModule << "\n";
  return llvmModule;
}

}