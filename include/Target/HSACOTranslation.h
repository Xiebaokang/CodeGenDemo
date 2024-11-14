#pragma once

#include "utils.h"

using namespace mlir;

namespace KernelCodeGen {
std::string generateAmdgcnAndHsacoFromLLIRFile(
        const char* filePath,
        const std::string& gfx_arch,
        const std::string& gfx_triple,
        const std::string& gfx_features,
        llvm::DenseMap<llvm::StringRef, KernelCodeGen::NVVMMetadata>* metadata = nullptr
);

}