#ifndef _KernelCodeGen_h_
#define _KernelCodeGen_h_

#include "Operators/Operators.h"
#include "Conversion/Optimizer.h"
#include "Conversion/LoweringPasses.h"
#include "Target/LLVMIRTranslation.h"
#include "Target/HSACOTranslation.h"

namespace KernelCodeGen
{
  class KernelCodeGenerator {
    using Config = std::map<std::string, std::vector<std::map<std::string, int>>>;
  public:
    KernelCodeGenerator(Target target_, const std::string& arch_) : target(target_), arch(arch_) {
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

    KernelCodeGenerator() = delete;

    template <typename OperatorType, typename... Args> 
    mlir::ModuleOp create(Args &&...args) {
      mlir::OpBuilder builder(&context);
      mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
      OperatorType::build(module, std::forward<Args>(args)...);
      return module;
    }

    bool optimize(mlir::ModuleOp &mod, std::map<std::string, int> config);

    bool lowering(mlir::ModuleOp &mod);

    std::string translate(mlir::ModuleOp& mod);

  private:
    mlir::MLIRContext context;
    Target target;
    const std::string arch;
  };

}
#endif