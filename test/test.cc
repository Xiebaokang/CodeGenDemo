#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
#include "Target/HSACOTranslation.h"

using namespace KernelCodeGen;
using Config = std::map<std::string, std::vector<std::map<std::string, int>>>;

std::string llirFilePath = "";

void test(const char* llirPath , bool amendLLIR = true) {
  #if 1
  KernelCodeGenerator generator("CUDA");

  Config configs = {
    {"Matmul", {
        {{"BLOCK_SIZE_M", 128}, {"BLOCK_SIZE_N", 128}, {"BLOCK_SIZE_K", 8}, {"THREAD_SIZE_M", 8}, {"THREAD_SIZE_N", 8}, {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 32}}
        // {{"BLOCK_SIZE_M", 128}, {"BLOCK_SIZE_N", 128}, {"BLOCK_SIZE_K", 16}, {"THREAD_SIZE_M", 8}, {"THREAD_SIZE_N", 8}, {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 32}}
      }}
  };

  generator.create<Matmul>(std::vector<int64_t>{1024, 1024, 1024});

  auto mods = generator.optimize(configs);
  llvm::DenseMap<llvm::StringRef, NVVMMetadata> metadata ;
  for (auto mod: mods) {
    // mod.dump();
    auto res = generator.lowering(mod);
    getNVVMMetaData(mod,&metadata);

    std::cout << res << "\n";
  }
  #endif
  if(amendLLIR){
    generateAmdgcnAndHsacoFromLLIRFile(llirPath,"gfx906","amdgcn-amd-amdhsa","",&metadata);
  }
  else{
    generateAmdgcnAndHsacoFromLLIRFile(llirPath,"gfx906","amdgcn-amd-amdhsa","",nullptr);
  }
}


int main(int argc, char* argv[]) {
  if(argc < 3){
    std::cout << "error. correct usage: codegen ${llirPath} ${amendLLIR}[1|0]" << std::endl;
    return 1;
  }
  const char *llirPath = argv[1];
  int amendLLIR = std::stoi((char *)argv[2]);
  test(llirPath, (amendLLIR > 0));
  return 0;
}