#include "KernelCodeGen.h"

namespace KernelCodeGen {

std::unique_ptr<Optimizer> createOptimizer(const std::string& opName) {
  if (opName == "Matmul") {
    return std::make_unique<MatmulOptimizer>();
  }
  return nullptr;
}

std::vector<mlir::ModuleOp> KernelCodeGenerator::optimize(std::map<std::string, std::vector<std::map<std::string, int>>> configs) {
  auto tempMod = mlir::dyn_cast<mlir::ModuleOp>(module->clone());
  auto opNames = Analyzer::collectFuncNames(tempMod);
  std::vector<mlir::ModuleOp> results;

  for (auto opName: opNames) {
    if (configs.count(opName) == 0) continue;
    auto opt = createOptimizer(opName);
    if (opt == nullptr) continue;

    std::vector<mlir::ModuleOp> mods;
    for (auto config: configs[opName]) {
      auto mod = mlir::dyn_cast<mlir::ModuleOp>(tempMod->clone());
      mlir::OpBuilder builder(mod);

      if (!opt->applicable(mod)) break;   // collect matmul datas
      opt->applyOptimzer(mod, builder, config);
      mods.push_back(mod);
    }

    if (mods.size() != 0) tempMod = mods[0];   // 取个最好的，再优化下一个算子
    results = mods;
  }

  return results;
}

bool firstLowering(mlir::ModuleOp& mod, mlir::MLIRContext& context) {
  mlir::PassManager pm(&context);
  pm.addPass(mlir::createLowerAffinePass());                     // affine -> scf/vector
  pm.addPass(mlir::createParallelLoopToGpuPass());               // scf.parallelOp -> gpu...
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  if (mlir::failed(pm.run(mod)))
    return false;
  return true;  
}

bool secondLowering(mlir::ModuleOp& mod, mlir::MLIRContext& context) {
  mlir::PassManager pm(&context);
  pm.addPass(createParallelToROCDLPass());                      // 自定义 scf.parallelOp -> gpu.workitem/workgroup.id.x/y
  // pm.addPass(createROCDLIdOpModifyPass());                      // 自定义 rocdl idop加attr (弃用)
  pm.addPass(mlir::createConvertSCFToCFPass());                  // scf -> cf
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());        // cf -> llvm
  pm.addPass(mlir::createArithToLLVMConversionPass());           // arith -> llvm
  pm.addPass(mlir::createConvertVectorToLLVMPass());             // vector -> llvm
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());  // memref -> llvm
  pm.addPass(mlir::createConvertFuncToLLVMPass());               // func -> llvm

  // pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addPass(createMoveUnCCPass());
  if (mlir::failed(pm.run(mod)))
    return false;
  return true;  
}

bool KernelCodeGenerator::lowering(mlir::ModuleOp& mod) {
  auto res_1 = firstLowering(mod, context);
  if (res_1) {
    auto res_2 = secondLowering(mod, context);
    if (res_2) {
      mod.dump();
      if (auto llvm_mod = translateModuleToLLVMIR(mod)){
        llvm_mod->print(llvm::outs(), nullptr);
        return true;
      }
      return true;
    }
    return false;
  }
  return false;
}

}