
#include "Target/LLVMIRTranslation.h"
#include <dlfcn.h>
#include <filesystem>
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include <optional>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "utils.h"

namespace {
using namespace llvm;

static std::optional<OptimizationLevel> mapToLevel(unsigned optLevel, unsigned sizeLevel) {
  switch (optLevel) {
  case 0:
    return OptimizationLevel::O0;
  case 1:
    return OptimizationLevel::O1;
  case 2:
    switch (sizeLevel) {
    case 0:
      return OptimizationLevel::O2;
    case 1:
      return OptimizationLevel::Os;
    case 2:
      return OptimizationLevel::Oz;
    }
    break;
  case 3:
    return OptimizationLevel::O3;
  }
  return std::nullopt;
}

static std::function<Error(Module *)> makeOptimizingPipeline(unsigned optLevel, unsigned sizeLevel, TargetMachine *targetMachine) {
  return [optLevel, sizeLevel, targetMachine](Module *m) -> Error {
    std::optional<OptimizationLevel> ol = mapToLevel(optLevel, sizeLevel);
    if (!ol) {
      return make_error<StringError>(
          formatv("invalid optimization/size level {0}/{1}", optLevel, sizeLevel).str(),
          inconvertibleErrorCode());
    }
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;

    PipelineTuningOptions tuningOptions;
    tuningOptions.LoopUnrolling = true;
    tuningOptions.LoopInterleaving = true;
    tuningOptions.LoopVectorization = true;
    tuningOptions.SLPVectorization = true;

    PassBuilder pb(targetMachine, tuningOptions);

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm;
    pb.registerVectorizerStartEPCallback(
        [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
          fpm.addPass(InstCombinePass());
        });
    mpm.addPass(pb.buildPerModuleDefaultPipeline(*ol));
    mpm.run(*m, mam);
    return Error::success();
  };
}
}


using namespace llvm;

namespace KernelCodeGen {

// Add the nvvm related metadata to LLVM IR.
static void amendLLVMFunc(llvm::Function *func, const NVVMMetadata &metadata,
                          Target target, const int threadsPerCTA,
                          const int wavesPerEU) {
  auto *module = func->getParent();
  auto &ctx = func->getContext();

  if (!metadata.maxntid.empty()) {
    auto maxntid =
        llvm::to_vector(llvm::map_range(metadata.maxntid, [&](int value) {
          return llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32),
                                        llvm::APInt(32, value));
        }));

    SmallVector<llvm::Metadata *> md_args = {llvm::ValueAsMetadata::get(func)};
    if (maxntid.size() > 0) {
      md_args.push_back(llvm::MDString::get(ctx, "maxntidx"));
      md_args.push_back(llvm::ValueAsMetadata::get(maxntid[0]));
    }
    if (maxntid.size() > 1) {
      md_args.push_back(llvm::MDString::get(ctx, "maxntidy"));
      md_args.push_back(llvm::ValueAsMetadata::get(maxntid[1]));
    }
    if (maxntid.size() > 2) {
      md_args.push_back(llvm::MDString::get(ctx, "maxntidz"));
      md_args.push_back(llvm::ValueAsMetadata::get(maxntid[2]));
    }

    module->getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(llvm::MDNode::get(ctx, md_args));
  }

  if (metadata.isKernel) {
    switch (target) {
    case Target::CUDA: {
      llvm::Metadata *mdArgs[] = {
          llvm::ValueAsMetadata::get(func), llvm::MDString::get(ctx, "kernel"),
          llvm::ValueAsMetadata::get(
              llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))};
      module->getOrInsertNamedMetadata("nvvm.annotations")
          ->addOperand(llvm::MDNode::get(ctx, mdArgs));
    } break;
    case Target::ROCm: {
      func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      func->addFnAttr("amdgpu-flat-work-group-size",
                      "1, " + std::to_string(threadsPerCTA));
      if (wavesPerEU > 0)
        func->addFnAttr("amdgpu-waves-per-eu", std::to_string(wavesPerEU));
      func->addFnAttr("denormal-fp-math-f32", "preserve-sign");
      func->addFnAttr("amdgpu-unsafe-fp-atomics", "true");
      for (unsigned I = 0; I < func->arg_size(); ++I) {
        Argument &Arg = *func->getArg(I);
        // Check for incompatible attributes.
        if (Arg.hasByRefAttr() || Arg.hasNestAttr())
          break;

        Arg.addAttr(llvm::Attribute::InReg);
      }
    } break;
    }
  }
}


static void
extractNVVMMetadata(mlir::ModuleOp module,
                    llvm::DenseMap<llvm::StringRef, NVVMMetadata> *dic) {
  llvm::outs() << "extractNVVMMetadata[1]\n";
  llvm::outs().flush();
  for (auto op : module.getOps<mlir::LLVM::LLVMFuncOp>()) {
    NVVMMetadata meta;

    bool hasMetadata{};

    // maxntid
    if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("nvvm.maxntid")) {
      llvm::transform(attr.getAsValueRange<mlir::IntegerAttr>(),
                      std::back_inserter(meta.maxntid),
                      [](llvm::APInt value) { return value.getZExtValue(); });
      hasMetadata = true;
    }

    // kernel
    if (op->hasAttr("nvvm.kernel")) {
      llvm::outs() << "[D] is Kernel";
      llvm::outs().flush();
      meta.isKernel = true;
      hasMetadata = true;
    }

    if (hasMetadata){
      dic->try_emplace(op.getNameAttr().strref(), std::move(meta));
      llvm::outs() << "add meta: " << op.getNameAttr().strref() << "\n"; llvm::outs().flush();
    }
  }
}



void optimizeLLVMIRModule(
    llvm::Module* llvmModule,
    llvm::DenseMap<llvm::StringRef, NVVMMetadata>* nvvmMetadata,
    KernelCodeGen::Target target
  )
{
  // const int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(module);
  const int numWarps = 4;
  // const int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(module);
  const int warpSize = 64;
  const int threadsPerCTA = numWarps * warpSize;



  for (auto &func : llvmModule->functions()) {
    auto it = nvvmMetadata->find(func.getName());
    if (it != nvvmMetadata->end())
      // amendLLVMFunc(&func, it->second, target, threadsPerCTA, wavesPerEU);
      amendLLVMFunc(&func, it->second, target, threadsPerCTA, 2);
  }



}


void getNVVMMetaData(mlir::ModuleOp& module,llvm::DenseMap<llvm::StringRef, NVVMMetadata>* nvvmMetadata){
  extractNVVMMetadata(module, nvvmMetadata);
}


std::string translateMLIRToLLVMIR(mlir::ModuleOp module) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::DLTIDialect, mlir::func::FuncDialect>();
  registerAllToLLVMIRTranslations(registry);
  module.getContext()->appendDialectRegistry(registry);
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);

  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return "";
  }
  auto optPipeline = makeOptimizingPipeline(/*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return "";
  }  
  std::string str;
  llvm::raw_string_ostream os(str);
  llvmModule->print(os, nullptr);
  os.flush();
  return str;
}

// 弃用
std::string tranlateAndSaveLLVMIR(mlir::ModuleOp module) {
  std::string llvmPath{"/home/pangyunfei/xie/CodeGenDemo/build/llvmir.ll"};
  std::string mlirPtah{"/home/pangyunfei/xie/CodeGenDemo/build/llvm-dialect.mlir"};
  // 存储mlir
  std::error_code ec;
  llvm::raw_fd_ostream outputFile(mlirPtah, ec);
  if (ec) {
    llvm::errs() << "Error: " << ec.message() << "\n";
    return "";
  }
  module->print(outputFile);
  outputFile.close();
  // 运行指令
  std::stringstream command;
  command << "/home/pangyunfei/xie/CodeGenDemo/bin/mlir-translate ";
  command << mlirPtah << " -mlir-to-llvmir > " << llvmPath;

  int result = system(command.str().c_str());

  return llvmPath;
}

}