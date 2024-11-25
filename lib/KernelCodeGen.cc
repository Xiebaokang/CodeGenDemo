#include "KernelCodeGen.h"
#include "Target/HSACOTranslation.h"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>

namespace KernelCodeGen {

std::unique_ptr<Optimizer> createOptimizer(const std::string& opName) {
  if (opName == "Matmul") {
    return std::make_unique<MatmulOptimizer>();
  }
  return nullptr;
}


bool KernelCodeGenerator::optimize(mlir::ModuleOp &mod, std::map<std::string, int> config) {
  auto opNames = Analyzer::collectFuncNames(mod);
  for (auto opName : opNames) {
    auto opt = createOptimizer(opName);
    if (opt == nullptr) {
      llvm::errs() << "Optimization failed: Create Optimizer Failed.\n";
      return false;
    }
    if (!opt->applicable(mod)) return false;   // collect matmul datas
    opt->applyOptimzer(mod, config);
    return true;
  }
}


bool transforms(mlir::ModuleOp& mod, mlir::MLIRContext& context, const std::string& libsPath, const std::string& gfx_arch) {
  mlir::PassManager pm(&context);
  pm.addPass(createCombineMemrefPass());
  pm.addPass(ReplaceAllocToGetglobalPass());
  // pm.addPass(createAddExternalLibPass(libsPath, gfx_arch));      // 给mlir module添加lib属性
  pm.addPass(createAffineFullUnrollPass());                      // 对打了unroll属性的affine loop进行循环展开
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
  mlir::PassManager pm(&context);
  pm.addPass(createParallelToROCDLPass());                         // 自定义 gpu.parallelOp -> rocdl.workitem/workgroup.id.x/y
  // pm.addPass(createROCDLIdOpModifyPass());                      // 自定义 rocdl idop加attr (弃用)
  pm.addPass(mlir::createConvertSCFToCFPass());                    // scf -> cf

  ConvertControlFlowToLLVMPassOptions cfOptions;
  cfOptions.indexBitwidth = INDEX_BIT_WIDTH;
  
  pm.addPass(mlir::createConvertControlFlowToLLVMPass(cfOptions));        // cf -> llvm
  // pm.addPass(createConvertArithIndexToI64Pass());                      // 自定义 将arith中的constantOp的result为index类型的Op全部转成result为i64的op

  ArithToLLVMConversionPassOptions arithOptions;
  arithOptions.indexBitwidth = INDEX_BIT_WIDTH;
  pm.addPass(mlir::createArithToLLVMConversionPass(arithOptions));            // arith -> llvm

  pm.addPass(createVectorToLLVMPass(/*indexBitwidth*/INDEX_BIT_WIDTH));                    // 自定义 vector to llvm pass
  // pm.addPass(mlir::createConvertVectorToLLVMPass());                       // vector -> llvm

  FinalizeMemRefToLLVMConversionPassOptions memrefOptions;
  memrefOptions.indexBitwidth = INDEX_BIT_WIDTH;                                           // 这个32会将malloc func的参数也定义为i32，以及将ptrtointOp的返回也是i32，llvm malloc func不支持i32
  // memrefOptions.useAlignedAlloc = true;                                    // 这个如果不开启的话，且上为i32，则llir转换失败，解决使用pass - createMallocFuncOpArgTypeI32ToI64Pass
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(memrefOptions));  // memref -> llvm

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());

  ConvertFuncToLLVMPassOptions funcOptions;                                 // passes.h.inc文件中有通过tablegen生成的pass base类型 以及createxxx()
  funcOptions.indexBitwidth = INDEX_BIT_WIDTH;                                           // func loewring 到 llvm 时，其index转到llvm上是使用i32类型
  funcOptions.useBarePtrCallConv = true;                                    // 使用裸指针，而不使用结构体指针表示memref类型
  pm.addPass(mlir::createConvertFuncToLLVMPass(funcOptions));               // func -> llvm
  // pm.addPass(createEraseRedundantUnCCastPass());                         // 手动写的去除多余UnrealizedCast
  // pm.addPass(mlir::createReconcileUnrealizedCastsPass());                // 内置去除多余cast的pass

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // pm.addPass(createMallocFuncOpArgTypeI32ToI64Pass());                      // 将malloc 的 func 的函数签名换成 i64，ptrtointOp/callOp跟着换（因为如果强制使用malloci32，后续llvmtranslation报错，llvm malloc只支持i64）
  pm.addPass(createGlobalShmSetZeroPass());
  if (mlir::failed(pm.run(mod))){
    return false;
  }  
  return true;  
}


bool KernelCodeGenerator::lowering(mlir::ModuleOp& mod) {
  // mod.dump();
  std::string libsPath{"/home/pangyunfei/xushilong/CodeGenDemo/third_party/hip/lib/bitcode"};
  transforms(mod, context, libsPath, arch);
  llvm::outs() << " === start mlir =====\n";llvm::outs().flush();
  mod.dump();

  firstLowering(mod, context);
  llvm::outs() << " === after firstLowering =====\n";llvm::outs().flush();
  mod.dump();

  secondLowering(mod, context);
  llvm::outs() << " === after secondLowering =====\n";llvm::outs().flush();
  mod.dump();
  
  return true;
}


std::string KernelCodeGenerator::translate(mlir::ModuleOp& mod) {
  const int wavesPerEU = 0;
  const std::string gfx_triple{"amdgcn-amd-amdhsa"};
  const std::string gfx_features{""};
#if 1
  std::string llvmIR = std::move(translateMLIRToLLVMIR(mod, target, wavesPerEU));
  // std::tuple<std::string, std::string> result = translateLLVMIRToHSACO(llvmIR, "gfx" + arch, gfx_triple, gfx_features);
  // return std::get<1>(result);
  std::cout << "\n====llvmIR\n" << llvmIR << std::endl;
#else

  // test insert 
  std::ifstream ifs("/home/pangyunfei/xushilong/CodeGenDemo/ceshiData/kcg/testLLVMIR.mlir");
  std::stringstream buffer;
  if(ifs.is_open()){
    buffer << ifs.rdbuf();
    ifs.close();
  }
  auto llvmIR = buffer.str();
#endif
  return generateAmdgcnAndHsacoFromLLIRFile(llvmIR, "gfx" + arch, gfx_triple, gfx_features);
}

}