#ifndef _KernelCodeGen_h_
#define _KernelCodeGen_h_

#include "Operators/Operators.h"
#include "Conversion/Optimizer.h"
#include "Conversion/LoweringPasses.h"
#include "Target/LLVMIRTranslation.h"
#include "Target/HSACOTranslation.h"

#include <string>
#include <sstream>
#include <fstream>
#include <initializer_list>
#include <climits>
#include <cfloat>
#include <filesystem>
#include <dlfcn.h>
#include "llvm/ADT/StringRef.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"

namespace KernelCodeGen
{
  class KernelCodeGenerator
  {
  public:
    KernelCodeGenerator(const std::string &platform_ = {"CUDA"}) : builder(&context), platform(std::move(platform_))
    {
      initMLIRContext();
      createModule();
    }

    KernelCodeGenerator() = delete;

    void initMLIRContext()
    {
      context.getOrLoadDialect<mlir::affine::AffineDialect>();
      context.getOrLoadDialect<mlir::memref::MemRefDialect>();
      context.getOrLoadDialect<mlir::func::FuncDialect>();
      context.getOrLoadDialect<mlir::arith::ArithDialect>();
      context.getOrLoadDialect<mlir::gpu::GPUDialect>();
      context.getOrLoadDialect<mlir::vector::VectorDialect>();
      context.getOrLoadDialect<mlir::scf::SCFDialect>();
      context.getOrLoadDialect<mlir::math::MathDialect>();
      context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
      context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
      mlir::registerAllPasses();
    }

    void createModule()
    {
      module = mlir::ModuleOp::create(builder.getUnknownLoc());
      builder.setInsertionPointToEnd(module.getBody());
    }

    template <typename OperatorType, typename... Args>
    void create(Args &&...args)
    {
      OperatorType::build(module, builder, std::forward<Args>(args)...);
    }

    void dump(const std::string &info = "")
    {
      llvm::errs() << "----------------------------------------------------------\n";
      llvm::errs() << "           " << info << "\n";
      llvm::errs() << "----------------------------------------------------------\n";
      module->dump();
      if (mlir::failed(mlir::verify(module)))
      {
        module->emitError("module verification error");
        assert(false);
      }
    }

    std::vector<mlir::ModuleOp> optimize(std::map<std::string, std::vector<std::map<std::string, int>>> configs);

    bool lowering(mlir::ModuleOp &mod);
    std::string translate(mlir::ModuleOp& mod, llvm::DenseMap<llvm::StringRef, NVVMMetadata>* meta);
    static std::map<std::string, std::string> getExternLibs(mlir::ModuleOp module)
    {
      // std::cout << "getExternLibs" << std::endl;
      std::map<std::string, std::string> externLibs;
      SmallVector<mlir::LLVM::LLVMFuncOp> funcs;
      module.walk([&](mlir::LLVM::LLVMFuncOp func){
        if (func.isExternal())
          funcs.push_back(func); 
      });

      for (auto &func : funcs)
      {
        if (func.getOperation()->hasAttr("libname"))
        {
          auto name =
              func.getOperation()->getAttr("libname").dyn_cast<mlir::StringAttr>();
          auto path =
              func.getOperation()->getAttr("libpath").dyn_cast<mlir::StringAttr>();
          if (name)
          {
            std::string libName = name.str();
            externLibs[libName] = path.str();
          }
        }
      }

      if (module.getOperation()->hasAttr(AttrExternLib))
      {
        auto dict = module.getOperation()
                        ->getAttr(AttrExternLib)
                        .dyn_cast<mlir::DictionaryAttr>();
        for (auto &attr : dict)
        {
          externLibs[attr.getName().strref().trim().str()] =
              attr.getValue().dyn_cast<mlir::StringAttr>().strref().trim().str();
        }
      }

      if (!funcs.empty())
      {
        static const std::string libdevice = "libdevice";
        // first search for environmental path
        std::string env_path = KernelCodeGen::getenv("TRITON_LIBDEVICE_PATH");
        if (!env_path.empty())
        {
          externLibs.try_emplace(libdevice, env_path);
          return externLibs;
        }
        namespace fs = std::filesystem;
        // Search for libdevice relative to its library path if used from Python
        // Then native code is in `triton/_C/libtriton.so` and libdevice in
        // `triton/third_party/cuda/lib/libdevice.10.bc`
        static const auto this_library_path = []
        {
          Dl_info fileinfo;
          if (dladdr(reinterpret_cast<void *>(&getExternLibs), &fileinfo) == 0)
          {
            return std::filesystem::path();
          }
          return std::filesystem::path(fileinfo.dli_fname);
        }();
        // static const auto runtime_path =
        //     this_library_path.parent_path().parent_path() / "third_party" / "cuda" /
        //     "lib" / "libdevice.10.bc";
        const std::string runtime_path = "/home/pangyunfei/xushilong/CodeGenDemo/third_party/cuda/lib/libdevice.10.bc" ;
        if (fs::exists(runtime_path))
        {
          // externLibs.try_emplace(libdevice, runtime_path.string());
          externLibs.try_emplace(libdevice, runtime_path);
        }
        else
        {
          // When using the Math Dialect, it is possible that some ops (e.g., log)
          // are lowered to a function call. In this case, we need to link libdevice
          // using its default path:
          // [triton root dir]/python/triton/language/libdevice.10.bc
          // TODO(Keren): handle external linkage other than libdevice?
          static const auto this_file_path = std::filesystem::path(__FILE__);
          // static const auto compiletime_path = this_file_path.parent_path()
          //                                          .parent_path()
          //                                          .parent_path()
          //                                          .parent_path() /
          //                                      "python" / "triton" / "third_party" /
          //                                      "cuda" / "lib" / "libdevice.10.bc";
          static std::string compiletime_path = "/home/pangyunfei/xushilong/CodeGenDemo/third_party/cuda/lib/libdevice.10.bc";
          if (!fs::exists(compiletime_path))
          {
            std::string error_msg = "Can't find libdevice at neither " +
                                    runtime_path + " nor " + compiletime_path;
            llvm::report_fatal_error(error_msg.c_str());
          }
          externLibs.try_emplace(libdevice, compiletime_path);
        }
      }

      return externLibs;
    }

    static void linkLibdevice(llvm::Module &module)
    {
      // std::cout << "linkLibdevice" << std::endl;
      // please check https://llvm.org/docs/NVPTXUsage.html#reflection-parameters
      // this will enable fast math path in libdevice
      // for example, when enable nvvm-reflect-ftz, sqrt.approx.f32 will change to
      // sqrt.approx.ftz.f32
      auto &ctx = module.getContext();
      llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
      llvm::Metadata *mdFour =
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(i32, 4));
      llvm::Metadata *mdName = llvm::MDString::get(ctx, "nvvm-reflect-ftz");
      llvm::Metadata *mdOne =
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(i32, 1));
      llvm::MDNode *reflect = llvm::MDNode::get(ctx, {mdFour, mdName, mdOne});
      module.addModuleFlag(reflect);
    }

    static bool linkExternLib(llvm::Module &module, llvm::StringRef name,
                              llvm::StringRef path, bool isROCM)
    {
      // std::cout << "linkExternLib" << std::endl;
      llvm::SMDiagnostic err;
      auto &ctx = module.getContext();

      auto extMod = llvm::parseIRFile(path, err, ctx);
      if (!extMod)
      {
        llvm::errs() << "Failed to load " << path;
        return true;
      }

      extMod->setTargetTriple(module.getTargetTriple());
      extMod->setDataLayout(module.getDataLayout());

      if (llvm::Linker::linkModules(module, std::move(extMod),
                                    llvm::Linker::Flags::LinkOnlyNeeded))
      {
        llvm::errs() << "Failed to link " << path;
        return true;
      }

      // check if ROCM
      if (!isROCM)
      {
        if (name == "libdevice")
        {
          linkLibdevice(module);
        }
        // else {
        //   assert(false && "unknown extern lib: ");
        // }
      }

      return false;
    }

  private:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    std::string platform;
  };

}
#endif