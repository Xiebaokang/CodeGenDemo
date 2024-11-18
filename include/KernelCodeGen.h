#ifndef _KernelCodeGen_h_
#define _KernelCodeGen_h_

#include "Operators/Operators.h"
#include "Conversion/Optimizer.h"
#include "Conversion/LoweringPasses.h"
#include "Target/LLVMIRTranslation.h"
#include "Target/HSACOTranslation.h"

namespace KernelCodeGen
{
  class KernelCodeGenerator
  {
  public:
    KernelCodeGenerator(Target target_, const std::string& arch_) : builder(&context), target(target_), arch(arch_) {
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

    std::string translate(mlir::ModuleOp& mod);

  private:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    // const std::string platform;
    Target target;
    const std::string arch;
  };

}
#endif