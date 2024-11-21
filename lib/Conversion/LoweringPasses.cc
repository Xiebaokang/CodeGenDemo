#include "Conversion/LoweringPasses.h"
#include <dlfcn.h>
#include <filesystem>

using namespace mlir;

namespace KernelCodeGen {

// 将scf的parallelOp 转成Gpu的block/threadIdOp表示，func添加grid/block size作为属性
struct SCFParallelToGPULowering : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp outerParallelOp, PatternRewriter &rewriter) const final {
    constexpr gpu::Dimension dims[] = {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
    auto &ops = outerParallelOp.getBody()->getOperations();
    if (ops.empty())
      return failure();

    scf::ParallelOp innerParallelOp = nullptr;
    for (Operation &op : ops) {
      innerParallelOp = dyn_cast<scf::ParallelOp>(&op);
      if (innerParallelOp)
        break;
    }
    if (!innerParallelOp)
      return failure();

    auto outerUpperBounds = outerParallelOp.getUpperBound();
    auto innerUpperBounds = innerParallelOp.getUpperBound();

    std::vector<int32_t> blockUpperBounds;
    std::vector<int32_t> threadUpperBounds;

    // 替换外层 parallelOp 为 gpu::BlockIdOp
    Location loc = outerParallelOp.getLoc();
    SmallVector<Value, 3> blockIds;
    for (unsigned i = 0; i < outerParallelOp.getNumLoops(); ++i) {
      auto blockId = rewriter.create<gpu::BlockIdOp>(loc, dims[i]);
      blockIds.push_back(blockId);

      auto constOp = outerUpperBounds[i].getDefiningOp<arith::ConstantOp>();
      auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>();
      blockUpperBounds.push_back(intAttr.getInt());
    }

    // 替换内层 parallelOp 为 gpu::ThreadIdOp
    rewriter.setInsertionPoint(innerParallelOp);
    SmallVector<Value, 3> threadIds;
    for (unsigned i = 0; i < innerParallelOp.getNumLoops(); ++i) {
      auto threadId = rewriter.create<gpu::ThreadIdOp>(loc, dims[i]);
      threadIds.push_back(threadId);

      auto constOp = innerUpperBounds[i].getDefiningOp<arith::ConstantOp>();
      auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>();
      threadUpperBounds.push_back(intAttr.getInt());
    }

    // 将func设置block和thread的上界属性
    auto parentOp = outerParallelOp->getParentOp();
    auto funcOp = mlir::dyn_cast<func::FuncOp>(parentOp);
    if (funcOp == nullptr) {
      llvm::errs() << "The ParentOp of scf::ParallelOp must is FuncOp!\n";
      assert(false);
    }
    funcOp->setAttr("func.grid.dim", rewriter.getDenseI32ArrayAttr(llvm::ArrayRef<int32_t>(blockUpperBounds)));
    funcOp->setAttr("func.block.dim", rewriter.getDenseI32ArrayAttr(llvm::ArrayRef<int32_t>(threadUpperBounds)));

    // 替换使用外层和内层循环变量的操作
    auto outerInductionVars = outerParallelOp.getInductionVars();
    for (unsigned i = 0; i < outerInductionVars.size(); ++i) {
      outerInductionVars[i].replaceAllUsesWith(blockIds[i]);
    }

    auto innerInductionVars = innerParallelOp.getInductionVars();
    for (unsigned i = 0; i < innerInductionVars.size(); ++i) {
      innerInductionVars[i].replaceAllUsesWith(threadIds[i]);
    }

    // 内层操作移出内层 p  collect op
    SmallVector<Operation *, 4> innerOpsToMove;
    for (Operation &op : innerParallelOp.getBody()->getOperations()) {
      if (!dyn_cast<scf::YieldOp>(op)) {
        innerOpsToMove.push_back(&op);
      }
    }
    // 内层操作移出内层 p 
    Operation *innerTempOp = threadIds.back().getDefiningOp();
    for (Operation *op : innerOpsToMove) {
      op->moveAfter(innerTempOp);
      innerTempOp = op;
    }
    rewriter.eraseOp(innerParallelOp);

    // 外 collect op
    SmallVector<Operation *, 4> outerOpsToMove;
    for (Operation &op : outerParallelOp.getBody()->getOperations()) {
      if (!dyn_cast<scf::YieldOp>(op)) {
        outerOpsToMove.push_back(&op);
      }
    }
    // move
    Operation *outerTempOp = blockIds.back().getDefiningOp();
    for (Operation *op : outerOpsToMove) {
      op->moveAfter(outerTempOp);
      outerTempOp = op;
    }
    rewriter.eraseOp(outerParallelOp);

    return success();
  }
};

// 将 GUP 的IdOp转成 rocdl的IdOp，读取func的attr加到新的IdOp上
template <typename IdOp, typename XOp, typename YOp, typename ZOp>
struct IdOpGPUToROCDLLowering : public OpRewritePattern<IdOp> {
  using OpRewritePattern<IdOp>::OpRewritePattern;

  private:
    StringRef boundsAttrName;

  public:
    explicit IdOpGPUToROCDLLowering(MLIRContext *context) 
            : OpRewritePattern<IdOp>(context), boundsAttrName("") {}

    explicit IdOpGPUToROCDLLowering(MLIRContext *context, StringRef boundsAttrName) 
            : OpRewritePattern<IdOp>(context), boundsAttrName(boundsAttrName) {}

  LogicalResult matchAndRewrite(IdOp idOp, PatternRewriter &rewriter) const final {
    auto loc = idOp->getLoc();
    MLIRContext *context = rewriter.getContext();
    Value newOp;
    uint32_t bitWidth = INDEX_BIT_WIDTH;
    switch (idOp.getDimension()) {
    case gpu::Dimension::x:
      newOp = rewriter.create<XOp>(loc, IntegerType::get(context, bitWidth));
      break;
    case gpu::Dimension::y:
      newOp = rewriter.create<YOp>(loc, IntegerType::get(context, bitWidth));
      break;
    case gpu::Dimension::z:
      newOp = rewriter.create<ZOp>(loc, IntegerType::get(context, bitWidth));
      break;
    }

    auto parentOp = idOp->getParentOp();
    auto funcOp = mlir::dyn_cast<func::FuncOp>(parentOp);
    if (!boundsAttrName.empty() && funcOp) {
      if (auto attr = mlir::dyn_cast<DenseI32ArrayAttr>(funcOp->getAttr(boundsAttrName))) {
        int32_t maximum = attr[static_cast<uint32_t>(idOp.getDimension())];
        newOp.getDefiningOp()->setAttr("range", rewriter.getDenseI32ArrayAttr({0, maximum}));
      }
    }
    Value indexVal = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), newOp);
    // auto castOp = rewriter.create<arith::BitcastOp>(loc, IndexType::get(context), newOp);
    rewriter.replaceOp(idOp, indexVal);
    return success();
  }
};

// 将gpu barrier转成rocdl的barrier
struct GPUBarrierToROCDLLowering : public OpRewritePattern<gpu::BarrierOp> {
  using OpRewritePattern<gpu::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::BarrierOp brOp, PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ROCDL::BarrierOp>(brOp);
    return success();
  }
};

// 将上述三个重写加到这个pass中
struct ParallelToROCDLPass : public PassWrapper<ParallelToROCDLPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelToROCDLPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ROCDL::ROCDLDialect>();
  }
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());
    LLVMTypeConverter typeConverter(&getContext());
    target.addIllegalOp<gpu::BlockIdOp, gpu::ThreadIdOp>();
    target.addLegalDialect<ROCDL::ROCDLDialect, arith::ArithDialect>();

    patterns.add<SCFParallelToGPULowering>(&getContext());
    // mlir::populateGpuToROCDLConversionPatterns(typeConverter, patterns, gpu::amd::HIP);
    patterns.add<IdOpGPUToROCDLLowering<gpu::BlockIdOp, ROCDL::BlockIdXOp, 
                                       ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>>(&getContext(), StringRef{"func.grid.dim"});
    patterns.add<IdOpGPUToROCDLLowering<gpu::ThreadIdOp, ROCDL::ThreadIdXOp,
                                       ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>>(&getContext(), StringRef{"func.block.dim"});

    patterns.add<GPUBarrierToROCDLLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      return signalPassFailure();
  }
};


// ***弃用***，自己写了一个从gpu到rocdl的pass，转了idop和BarrierOp，主要是添加属性range
struct ROCDLIdOpModifyPass : public PassWrapper<ROCDLIdOpModifyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ROCDLIdOpModifyPass)
  
  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    for (Operation &op : module.getBody()->getOperations()) {
      auto funcOp = mlir::dyn_cast<func::FuncOp>(&op);
      if (funcOp == nullptr) {
        llvm::errs() << "there is other operations which is not funcOp in the module!\n";
        assert(false);
      }
      auto blockDims = mlir::dyn_cast<DenseI32ArrayAttr>(funcOp->getAttr("func.grid.dim"));
      auto threadDims = mlir::dyn_cast<DenseI32ArrayAttr>(funcOp->getAttr("func.block.dim"));

      funcOp.walk([&](Operation *op) {
        if (auto blockIdXOp = mlir::dyn_cast<ROCDL::BlockIdXOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, blockDims[0]}));
        } else if (auto blockIdYOp = mlir::dyn_cast<ROCDL::BlockIdYOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, blockDims[1]}));
        } else if (auto blockIdZOp = mlir::dyn_cast<ROCDL::BlockIdZOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, blockDims[2]}));
        } else if (auto threadIdXOp = mlir::dyn_cast<ROCDL::ThreadIdXOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, threadDims[0]}));
        } else if (auto threadIdYOp = mlir::dyn_cast<ROCDL::ThreadIdYOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, threadDims[1]}));
        } else if (auto threadIdZOp = mlir::dyn_cast<ROCDL::ThreadIdZOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, threadDims[2]}));
        }
      });
    }
  }
};

// ***弃用***，去除多余的unrealized_conversion_cast操作/因为内置有去除函数
struct EraseRedundantUnCCastPass : public PassWrapper<EraseRedundantUnCCastPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EraseRedundantUnCCastPass)
  
  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    SmallVector<std::pair<Operation*, Operation*>> pairOps;
    SmallVector<Operation*> noChOps;
    module.walk([&](Operation *op){
      if (auto uccOp = mlir::dyn_cast<UnrealizedConversionCastOp>(op)) {
        for (auto &use: uccOp.getResult(0).getUses()) {
          Operation *nextOp = use.getOwner();
          if (isa<UnrealizedConversionCastOp>(nextOp))
            pairOps.push_back(std::make_pair(op, nextOp));
          break;
        }
        if (uccOp.use_empty()) {
          noChOps.push_back(op);
        }
      }
    });
    for (auto pairOp: pairOps) {
      auto firstOp = mlir::dyn_cast<UnrealizedConversionCastOp>(pairOp.first);
      auto secondOp = mlir::dyn_cast<UnrealizedConversionCastOp>(pairOp.second);
      if (firstOp.getOperand(0).getType() == secondOp.getResult(0).getType()) {
        secondOp.getResult(0).replaceAllUsesWith(firstOp.getOperand(0));
      }
      pairOp.second->erase();
      pairOp.first->erase();
    }
    for (auto noChOp: noChOps) {
      // llvm::outs() << *noChOp << "\n";
      noChOp->erase();
    }
  }
};

// ***弃用***，将arith的constantOp（index）转成i64
struct ConvertArithIndexToI64Pass : public PassWrapper<ConvertArithIndexToI64Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertArithIndexToI64Pass)

  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    module.walk([&](Operation *op) {
      if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
        Type constantType = constantOp.getValue().getType();
        if (constantType.isIndex()) {
          auto indexValue = constantOp.getValue().cast<IntegerAttr>().getInt();
          OpBuilder builder(op);
          auto i64Op = builder.create<arith::ConstantOp>(constantOp.getLoc(), builder.getI64IntegerAttr(indexValue));
          auto indexVal = builder.create<arith::IndexCastOp>(i64Op.getLoc(), builder.getIndexType(), i64Op);
          constantOp.getResult().replaceAllUsesWith(indexVal.getResult());
          constantOp.erase();
        }
      }
    });
  }
};

// ***弃用*** 没有使用，和上面一样的功能
struct ConvertArithConstantIndexToI64 : public OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constOp, PatternRewriter &rewriter) const final {
    Type constType = constOp.getValue().getType();
    if (constType.isIndex()) {
      auto indexValue = constOp.getValue().cast<IntegerAttr>().getInt();
      auto i64Op = rewriter.create<arith::ConstantOp>(constOp.getLoc(), rewriter.getI64IntegerAttr(indexValue));
      auto indexVal = rewriter.create<arith::IndexCastOp>(i64Op.getLoc(), rewriter.getIndexType(), i64Op);
      rewriter.replaceOp(constOp, indexVal);
      return success();
    } else {
      return failure();
    }
  }
};
// ***弃用***
struct ConvertIndexToI64Pass : public PassWrapper<ConvertIndexToI64Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertIndexToI64Pass)
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    patterns.add<ConvertArithConstantIndexToI64>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      return signalPassFailure();
  }
};

// affine 循环展开
struct AffineFullUnrollPass : public PassWrapper<AffineFullUnrollPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineFullUnrollPass)

  void runOnOperation() override {
    getOperation().walk([&] (affine::AffineForOp forOp){
      if (auto unrollName = forOp->getAttr("affine.loop")) {
        auto unrollAttr = mlir::dyn_cast<mlir::StringAttr>(unrollName);
        // llvm::outs() << unrollAttr.getValue().str() << "\n";
        if (unrollAttr.getValue().str() == "unroll") {
          if (failed(affine::loopUnrollFull(forOp))) {
            return signalPassFailure();
          }
        }
      }
    });
  }
};

// 将memref lowering到llvm上，因为 passes.h.inc中的base类没有提供可以选择indexBitWidth的options，所以自己写了一个
struct VectorToLLVMPass : public PassWrapper<VectorToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorToLLVMPass)

  VectorToLLVMPass(unsigned indexBitWidth_=32) : indexBitWidth(indexBitWidth_) {};

  unsigned indexBitWidth;

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    LowerToLLVMOptions options(&getContext());
    options.overrideIndexBitwidth(indexBitWidth);

    LLVMTypeConverter converter(&getContext(), options);
    mlir::populateVectorToLLVMConversionPatterns(converter, patterns, false, true);
    // mlir::populateVectorToLLVMMatrixConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

// ***弃用*** 使用函数代替了没有写模式匹配了，因为一直报错
struct ReplacePtrtointAndCallOp : public OpRewritePattern<LLVM::PtrToIntOp> {
  using OpRewritePattern<LLVM::PtrToIntOp>::OpRewritePattern;

  explicit ReplacePtrtointAndCallOp(MLIRContext *context, LLVM::LLVMFuncOp newFuncOp_) : 
            OpRewritePattern<LLVM::PtrToIntOp>(context), newFuncOp(newFuncOp_) {};

  LogicalResult matchAndRewrite(LLVM::PtrToIntOp ptrOp, PatternRewriter &rewriter) const final {
    // get ptrop & new ptrop
    Value argPtr = ptrOp.getArg();
    auto newPtrOp = rewriter.create<LLVM::PtrToIntOp>(ptrOp.getLoc(), rewriter.getIntegerType(64), argPtr);
    auto users = ptrOp.getResult().getUsers();
    for (auto user : users) {
      if (auto callOp = mlir::dyn_cast<LLVM::CallOp>(user)) {
        rewriter.setInsertionPointAfter(callOp);
        auto newCallOp = rewriter.create<LLVM::CallOp>(callOp->getLoc(), newFuncOp, ValueRange({newPtrOp.getResult()}));
        // callOp.getResult().replaceAllUsesWith(newCallOp.getResult());
        // callOp.erase();
        rewriter.replaceOp(callOp, newCallOp);
        // rewriter.eraseOp(callOp);
      }
    }
    // ptrOp.erase();
    rewriter.replaceOp(ptrOp, newPtrOp);
    return success();
  }

  private:
  LLVM::LLVMFuncOp newFuncOp;
};

// 替换malloc(i32) -> malloc(i64) / ptrtointOp & callOp 也替换成符合i64的操作
struct MallocFuncOpArgTypeI32ToI64Pass : public PassWrapper<MallocFuncOpArgTypeI32ToI64Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MallocFuncOpArgTypeI32ToI64Pass)

  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    auto reuslt = createI64MallocFuncOp(module);
    replacePtrtointAndCallOp(module, reuslt.first);   // create i64 malloc
    deleteI32MallocFuncOp(reuslt.second);   // erase i32 malloc
  }

  private:

  void replacePtrtointAndCallOp(ModuleOp &module, LLVM::LLVMFuncOp &mallocFuncOp) {
    if (!mallocFuncOp) return;
    SmallVector<LLVM::PtrToIntOp> ptrOps;
    module.walk([&](LLVM::PtrToIntOp ptrOp) {
      ptrOps.push_back(ptrOp);
    });
    
    for (auto ptrOp: ptrOps) {
      Value argPtr = ptrOp.getArg();
      OpBuilder buidler(ptrOp);
      auto newPtrOp = buidler.create<LLVM::PtrToIntOp>(ptrOp.getLoc(), buidler.getIntegerType(64), argPtr);

      auto users = ptrOp.getResult().getUsers();
      for (auto user : users) {
        if (auto callOp = mlir::dyn_cast<LLVM::CallOp>(user)) {
          buidler.setInsertionPointAfter(callOp);
          auto newCallOp = buidler.create<LLVM::CallOp>(callOp->getLoc(), mallocFuncOp, ValueRange({newPtrOp.getResult()}));
          callOp.getResult().replaceAllUsesWith(newCallOp.getResult());
          callOp.erase();
          // rewriter.replaceOp(callOp, newCallOp);
          // rewriter.eraseOp(callOp);
        }
      }
      ptrOp.erase();
      // rewriter.replaceOp(ptrOp, newPtrOp);
      // llvm::outs() <<newPtrOp << "\n";
    }
  }

  std::pair<LLVM::LLVMFuncOp, LLVM::LLVMFuncOp> createI64MallocFuncOp(ModuleOp &module) {
    for (Operation &op : module.getBody()->getOperations()) {
      if (auto funcOp = mlir::dyn_cast<LLVM::LLVMFuncOp>(&op)) {
        if (funcOp.getName() != "malloc") {
          continue;
        } else {
          LLVM::LLVMFunctionType funcType = funcOp.getFunctionType();
          if (funcType.getNumParams() == 1 && funcType.getParams()[0].isInteger(64)) {
            return std::make_pair(funcOp, nullptr);   // malloc本身就是i64的func

          } else if (funcType.getNumParams() == 1 && funcType.getParams()[0].isInteger(32)) {
            OpBuilder b(module.getBodyRegion());
            MLIRContext *context = module->getContext();
            auto newFuncType = LLVM::LLVMFunctionType::get(LLVM::LLVMPointerType::get(context), 
                                      {IntegerType::get(context, 64)}, /*opaquePointers*/false);
            auto newFuncOp = b.create<LLVM::LLVMFuncOp>(module->getLoc(), "malloc", newFuncType);
            return std::make_pair(newFuncOp, funcOp);   // malloc是i32的func，创建新的true
          }
        }
      }
    }
    return std::make_pair(nullptr, nullptr);
  }

  void deleteI32MallocFuncOp(LLVM::LLVMFuncOp &funcOp) {
    if (funcOp){
      funcOp.erase();
    }
  }

};


struct AddExternalLibPass : public PassWrapper<AddExternalLibPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddExternalLibPass)

  AddExternalLibPass(const std::string& libsPath_, const std::string& gfx_arch_)
               : libsPath(libsPath_), gfx_arch(gfx_arch_) {};

  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    const auto& bcfiles = getROCMBitcodefiles();
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

  private:

  const std::string libsPath;
  const std::string gfx_arch;

  bool isBCFile(const std::string& extname,const std::string& fileName){
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
    if(fileName.find("oclc_isa_version") != fileName.npos && fileName.find(gfx_arch) != fileName.npos){
      return true;
    }
    return false;
  }

  std::vector<std::pair<std::string,std::string>> getROCMBitcodefiles() {
    std::vector<std::pair<std::string,std::string>> files;
    int index = 0;
    for (const auto& entry : std::filesystem::directory_iterator(libsPath)) {
      if (entry.is_regular_file()) {
        const auto& fileName = entry.path().filename().string();
        auto extname = entry.path().extension().string();
        if(isBCFile(extname, fileName)){
          auto pair = std::make_pair("library_"+std::to_string(index++), libsPath+"/"+fileName);
          files.push_back(std::move(pair));
        }
      }
    }
    assert(files.size() == 10);
    return files;
  }

};


// replace alloc<shared> to getGlobalOp

struct ReplaceAllocOpToGetGlobalOp : public PassWrapper<ReplaceAllocOpToGetGlobalOp, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceAllocOpToGetGlobalOp)
   ReplaceAllocOpToGetGlobalOp() = default;
   void runOnOperation() override {
     auto module = getOperation();
    int i = 0;
     std::vector<MemRefType> memTypesToAdd {};
     module.walk<WalkOrder::PreOrder>([&](memref::AllocOp allocOp) {
      auto memspace = allocOp.getResult().getType().getMemorySpaceAsInt();
      if(memspace == (int)KernelCodeGen::MemorySpace::shared){
        OpBuilder builder(allocOp);
        OpBuilder b(module);
        b.setInsertionPointToStart(module.getBody());
        b.create<memref::GlobalOp>(
          b.getUnknownLoc(),
          SHM_VAR_NAME(i),
          b.getStringAttr("public"),
          allocOp.getResult().getType(),
          Attribute(),
          false,
          IntegerAttr()
          );
        auto newop = builder.create<memref::GetGlobalOp>(
          builder.getUnknownLoc(),allocOp.getResult().getType(),SHM_VAR_NAME(i));
        allocOp.getResult().replaceAllUsesWith(newop);
        allocOp.erase();
        ++i;
      }
     });
   }
}; 

// 将alloc和alloca操作合并，生成一个memref
struct CombineMemrefPass : public PassWrapper<CombineMemrefPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CombineMemrefPass)

  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    for (Operation &op : module.getBody()->getOperations()) {
      if (auto funcOp = mlir::dyn_cast<func::FuncOp>(&op)) {
        combineAllocOrAllocaOp<memref::AllocOp>(funcOp);
        combineAllocOrAllocaOp<memref::AllocaOp>(funcOp);
      }
    }
  }

  template <typename LoadOrStoreOp>
  AffineMap moreDimToOneDimMap(LoadOrStoreOp op, int64_t startIndex, llvm::ArrayRef<int64_t> shapes, MLIRContext* context) {
    auto oldAffineMap = op.getAffineMap();
    auto oldExprs = oldAffineMap.getResults();
    OpBuilder b(context);
    AffineExpr expr = b.getAffineConstantExpr(0);
    for (size_t i=0; i<oldExprs.size(); i++) {
      int num = 1;
      for (size_t j=i+1; j<shapes.size(); j++) {
        num *= shapes[j];
      }
      expr = expr + oldExprs[i] * num;
    }
    expr = startIndex + expr;
    auto map = AffineMap::get(oldAffineMap.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>({expr}), context);
    return map;
  }

  
  template<typename AllocOrAllocaOp>
  void combineAllocOrAllocaOp(func::FuncOp &funcOp) {
    int64_t memSize = 0;
    llvm::DenseMap<AllocOrAllocaOp, int64_t> indexMap;
    AllocOrAllocaOp firstOp = nullptr;
    MemRefType type;

    funcOp.walk<WalkOrder::PreOrder>([&](AllocOrAllocaOp allocOp) {
      if (memSize == 0) firstOp = allocOp;
      indexMap.try_emplace(allocOp, memSize);
      type = mlir::dyn_cast<MemRefType>(allocOp.getResult().getType());
      int64_t temp = 1;
      for (auto shape : type.getShape()) {
        temp *= shape;
      }
      memSize += temp;
    });
    if (!memSize) return;

    OpBuilder b(firstOp);
    auto newType = MemRefType::get({memSize}, type.getElementType(), {}, type.getMemorySpaceAsInt());
    auto newAllocOp = b.create<AllocOrAllocaOp>(firstOp.getLoc(), newType);

    for (const auto& pair : indexMap) {
      Value result = pair.first->getResult(0);
      auto t = mlir::dyn_cast<MemRefType>(result.getType());
      auto shapes = t.getShape();

      SmallVector<Operation *> users;

      for (auto user : result.getUsers()) {
        users.push_back(user);
      }
      // auto users = result.getUsers();

      for (auto user : users) {
        if (auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(user)) {
          auto map = moreDimToOneDimMap(loadOp, pair.second, shapes, loadOp->getContext());
          b.setInsertionPointAfter(loadOp);
          auto newLoadOp = b.create<affine::AffineLoadOp>(loadOp.getLoc(), newAllocOp.getResult(), map, loadOp.getMapOperands());
          loadOp.getResult().replaceAllUsesWith(newLoadOp.getResult());
          loadOp.erase();

        } else if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(user)) {
          auto map = moreDimToOneDimMap(storeOp, pair.second, shapes, storeOp->getContext());
          b.setInsertionPointAfter(storeOp);
          b.create<affine::AffineStoreOp>(storeOp.getLoc(), storeOp.getValue(), newAllocOp.getResult(), map, storeOp.getMapOperands());
          storeOp.erase();

        } else if (auto vectorLoadOp = mlir::dyn_cast<affine::AffineVectorLoadOp>(user)) {
          auto map = moreDimToOneDimMap(vectorLoadOp, pair.second, shapes, vectorLoadOp->getContext());
          b.setInsertionPointAfter(vectorLoadOp);
          auto newVectorLoadOp = b.create<affine::AffineVectorLoadOp>(vectorLoadOp.getLoc(), vectorLoadOp.getVectorType(), 
                                                              newAllocOp.getResult(), map, vectorLoadOp.getMapOperands());
          vectorLoadOp.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
          vectorLoadOp.erase();

        } else if (auto vectorStoreOp = mlir::dyn_cast<affine::AffineVectorStoreOp>(user)) {
          auto map = moreDimToOneDimMap(vectorStoreOp, pair.second, shapes, vectorStoreOp->getContext());
          b.setInsertionPointAfter(vectorStoreOp);
          b.create<affine::AffineVectorStoreOp>(vectorStoreOp.getLoc(), vectorStoreOp.getValue(), 
                                            newAllocOp.getResult(), map, vectorStoreOp.getMapOperands());
          vectorStoreOp.erase();
        }
      }
      pair.first->erase();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createParallelToROCDLPass() {
  return std::make_unique<ParallelToROCDLPass>();
}

// ***弃用***，自己写了一个从gpu到rocdl的pass，转了idop和BarrierOp
std::unique_ptr<OperationPass<ModuleOp>> createROCDLIdOpModifyPass() {
  return std::make_unique<ROCDLIdOpModifyPass>();
}
// ***弃用***
std::unique_ptr<OperationPass<ModuleOp>> createEraseRedundantUnCCastPass() {
  return std::make_unique<EraseRedundantUnCCastPass>();
}
// ***弃用***
std::unique_ptr<OperationPass<ModuleOp>> createConvertArithIndexToI64Pass() {
  return std::make_unique<ConvertArithIndexToI64Pass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createAffineFullUnrollPass() {
  return std::make_unique<AffineFullUnrollPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createVectorToLLVMPass(unsigned indexBitWidth) {
  return std::make_unique<VectorToLLVMPass>(indexBitWidth);
}

std::unique_ptr<OperationPass<ModuleOp>> createMallocFuncOpArgTypeI32ToI64Pass() {
  return std::make_unique<MallocFuncOpArgTypeI32ToI64Pass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createAddExternalLibPass(const std::string& libsPath, const std::string& gfx_arch) {
  return std::make_unique<AddExternalLibPass>(libsPath, gfx_arch);
}

std::unique_ptr<OperationPass<ModuleOp>> ReplaceAllocToGetglobalPass() {
  return std::make_unique<ReplaceAllocOpToGetGlobalOp>();
}

std::unique_ptr<OperationPass<ModuleOp>> createCombineMemrefPass() {
  return std::make_unique<CombineMemrefPass>();
}

}