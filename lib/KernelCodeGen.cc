#include "KernelCodeGen.h"
#include "Target/HSACOTranslation.h"
#include <dlfcn.h>
#include <filesystem>

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
  int indexBitWidth = INDEX_BIT_WIDTH;
  mlir::PassManager pm(&context);
  pm.addPass(createParallelToROCDLPass());                         // 自定义 gpu.parallelOp -> rocdl.workitem/workgroup.id.x/y
  // pm.addPass(createROCDLIdOpModifyPass());                      // 自定义 rocdl idop加attr (弃用)
  pm.addPass(mlir::createConvertSCFToCFPass());                    // scf -> cf

  ConvertControlFlowToLLVMPassOptions cfOptions;
  cfOptions.indexBitwidth = 32;
  
  pm.addPass(mlir::createConvertControlFlowToLLVMPass(cfOptions));        // cf -> llvm
  // pm.addPass(createConvertArithIndexToI64Pass());                      // 自定义 将arith中的constantOp的result为index类型的Op全部转成result为i64的op

  ArithToLLVMConversionPassOptions arithOptions;
  arithOptions.indexBitwidth = 32;
  pm.addPass(mlir::createArithToLLVMConversionPass(arithOptions));            // arith -> llvm

  pm.addPass(createVectorToLLVMPass(/*indexBitwidth*/32));                    // 自定义 vector to llvm pass
  // pm.addPass(mlir::createConvertVectorToLLVMPass());                       // vector -> llvm

  FinalizeMemRefToLLVMConversionPassOptions memrefOptions;
  memrefOptions.indexBitwidth = 32;                                           // 这个32会将malloc func的参数也定义为i32，以及将ptrtointOp的返回也是i32，llvm malloc func不支持i32
  // memrefOptions.useAlignedAlloc = true;                                    // 这个如果不开启的话，且上为i32，则llir转换失败，解决使用pass - createMallocFuncOpArgTypeI32ToI64Pass
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(memrefOptions));  // memref -> llvm

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());

  ConvertFuncToLLVMPassOptions funcOptions;                                 // passes.h.inc文件中有通过tablegen生成的pass base类型 以及createxxx()
  funcOptions.indexBitwidth = 32;                                           // func loewring 到 llvm 时，其index转到llvm上是使用i32类型
  funcOptions.useBarePtrCallConv = true;                                    // 使用裸指针，而不使用结构体指针表示memref类型
  pm.addPass(mlir::createConvertFuncToLLVMPass(funcOptions));               // func -> llvm
  // pm.addPass(createEraseRedundantUnCCastPass());                         // 手动写的去除多余UnrealizedCast
  // pm.addPass(mlir::createReconcileUnrealizedCastsPass());                // 内置去除多余cast的pass

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(createMallocFuncOpArgTypeI32ToI64Pass());                      // 将malloc 的 func 的函数签名换成 i64，ptrtointOp/callOp跟着换（因为如果强制使用malloci32，后续llvmtranslation报错，llvm malloc只支持i64）

  // ./mlir-translate /home/pangyunfei/xie/CodeGenDemo/build/llvm-dialect.mlir -mlir-to-llvmir > /home/pangyunfei/xie/CodeGenDemo/build/llvm-ir.ll

  if (mlir::failed(pm.run(mod)))
    return false;
  return true;  
}


bool isBCFile(const std::string& extname,const std::string& fileName, const std::string& gfxArchDigits){
    bool is_oclcVersion = false;
    bool is_gfxArchMatch = false;
    static std::string nameList[] = {
            "opencl.bc",
            "ocml.bc",
            "ockl.bc",
            "oclc_finite_only_off.bc",
            "oclc_daz_opt_on.bc",
            "oclc_correctly_rounded_sqrt_on.bc",
            "oclc_unsafe_math_off.bc",
            "oclc_wavefrontsize64_on.bc",
            "oclc_abi_version_400.bc"
    };

    if(extname != ".bc"){
        return false;
    }
    for(const auto& name : nameList){
        if(fileName == name){
            return true;
        }
    }
    if(fileName.find("oclc_isa_version") != fileName.npos && fileName.find(gfxArchDigits) != fileName.npos){
        return true;
    }
    return false;
}


std::vector<std::pair<std::string,std::string>> getROCMBitcodefiles(
    const std::string& path, const std::string& gfx_arch) 
{
    std::vector<std::pair<std::string,std::string>> files;
    int index = 0;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            const auto& fileName = entry.path().filename().string();
            auto extname = entry.path().extension().string();
            if(isBCFile(extname, fileName,"906")){
                auto pair = std::make_pair("library_"+std::to_string(index++),path+"/"+fileName);
                files.push_back(std::move(pair));
            }
        }
    }
    assert(files.size() == 10);
    return files;
}


void addExternalLibs(mlir::ModuleOp &module) 
{
  using namespace mlir;
  const auto& bcfiles = getROCMBitcodefiles(
    "/home/pangyunfei/xushilong/CodeGenDemo/third_party/hip/lib/bitcode","906");

  llvm::SmallVector<NamedAttribute, 2> attrs;

  for (size_t i = 0; i < bcfiles.size(); ++i) {
    auto name = StringAttr::get(module->getContext(), bcfiles[i].first);
    auto path = StringAttr::get(module->getContext(), bcfiles[i].second);
    NamedAttribute attr(name, path);
    attrs.push_back(attr);
  }

  DictionaryAttr dict = DictionaryAttr::get(module->getContext(), attrs);
  module.getOperation()->setAttr(AttrExternLib, dict);
}


bool KernelCodeGenerator::lowering(mlir::ModuleOp& mod) {
  addExternalLibs(mod);
  mod.dump();

  // transforms(mod, context);
  llvm::outs() << " === start mlir =====\n";llvm::outs().flush();
  mod.dump();

  firstLowering(mod, context);
  llvm::outs() << " === after firstLowering =====\n";llvm::outs().flush();
  mod.dump();

  secondLowering(mod, context);
  llvm::outs() << " === after secondLowering =====\n";llvm::outs().flush();
  mod.dump();
#if 0
  auto llvm_mod = translateModuleToLLVMIR(mod);
  llvm_mod->print(llvm::outs(), nullptr);
#endif
  
  return true;
}

}