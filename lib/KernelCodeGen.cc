#include "KernelCodeGen.h"
#include "Target/HSACOTranslation.h"
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

bool transforms(mlir::ModuleOp& mod, mlir::MLIRContext& context) {
  mlir::PassManager pm(&context);
  // pm.addPass(createAffineFullUnrollPass());
  if (mlir::failed(pm.run(mod)))
    return false;
  return true;  
}

bool firstLowering(mlir::ModuleOp& mod, mlir::MLIRContext& context) {
  mlir::PassManager pm(&context);
  pm.addPass(mlir::createCSEPass());
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
  int indexBitWidth = INDEX_BIT_WIDTH;
  mlir::PassManager pm(&context);
  pm.addPass(createParallelToROCDLPass());                      // 自定义 gpu.parallelOp -> rocdl.workitem/workgroup.id.x/y
  // pm.addPass(createROCDLIdOpModifyPass());                      // 自定义 rocdl idop加attr (弃用)
  pm.addPass(mlir::createConvertSCFToCFPass());                  // scf -> cf

  ConvertControlFlowToLLVMPassOptions cfOptions;
  cfOptions.indexBitwidth = 32;
  
  pm.addPass(mlir::createConvertControlFlowToLLVMPass(cfOptions));        // cf -> llvm

  // pm.addPass(createMemrefToLLVMPtrPass());
  // pm.addPass(createConvertArithIndexToI64Pass());  

  // -----------------------------------------------------------------------------------

  // pm.addPass(mlir::createArithToLLVMConversionPass());           // arith -> llvm
  // pm.addPass(mlir::createConvertVectorToLLVMPass());             // vector -> llvm
  // pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());  // memref -> llvm
  // ------------------------------------------------------------------------------------

  ArithToLLVMConversionPassOptions arithOptions;
  arithOptions.indexBitwidth = 32;
  pm.addPass(mlir::createArithToLLVMConversionPass(arithOptions));           // arith -> llvm

  pm.addPass(createVectorToLLVMPass(32)); 
  // pm.addPass(mlir::createConvertVectorToLLVMPass());             // vector -> llvm

  FinalizeMemRefToLLVMConversionPassOptions memrefOptions;
  memrefOptions.indexBitwidth = 32;
  memrefOptions.useAlignedAlloc = true;
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(memrefOptions));  // memref -> llvm
  // --------------------------------------------------------------------------------------

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());

  ConvertFuncToLLVMPassOptions funcOptions;
  funcOptions.indexBitwidth = 32;
  funcOptions.useBarePtrCallConv = true;
  pm.addPass(mlir::createConvertFuncToLLVMPass(funcOptions));               // func -> llvm
  // pm.addPass(createEraseRedundantUnCCastPass());
  // pm.addPass(mlir::createReconcileUnrealizedCastsPass());       // 内置去除多余cast的pass

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // pm.addPass(createModifyMallocFuncAndCallPass());
  // ./mlir-translate /home/pangyunfei/xie/CodeGenDemo/build/llvm-dialect.mlir -mlir-to-llvmir > /home/pangyunfei/xie/CodeGenDemo/build/llvm-ir.ll

  if (mlir::failed(pm.run(mod)))
    return false;
  return true;  
}


bool KernelCodeGenerator::lowering(mlir::ModuleOp& mod) {
  // mod.dump();
  
  // transforms(mod, context);
  // mod.dump();

  firstLowering(mod, context);
  mod.dump();

  secondLowering(mod, context);
  mod.dump();
#if 0
  auto llvm_mod = translateModuleToLLVMIR(mod);
  llvm_mod->print(llvm::outs(), nullptr);
#endif
  
  return true;
}

}