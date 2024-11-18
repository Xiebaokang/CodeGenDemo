#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
#include "Target/HSACOTranslation.h"

using namespace KernelCodeGen;
using Config = std::map<std::string, std::vector<std::map<std::string, int>>>;

void test() {
  #if 1
  KernelCodeGenerator generator(Target::ROCm, "906");

  Config configs = {
    {"Matmul", {
        {{"BLOCK_SIZE_M", 128}, {"BLOCK_SIZE_N", 128}, {"BLOCK_SIZE_K", 8}, {"THREAD_SIZE_M", 8}, {"THREAD_SIZE_N", 8}, {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 32}}
        // {{"BLOCK_SIZE_M", 128}, {"BLOCK_SIZE_N", 128}, {"BLOCK_SIZE_K", 16}, {"THREAD_SIZE_M", 8}, {"THREAD_SIZE_N", 8}, {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 32}}
      }}
  };

  generator.create<Matmul>(std::vector<int64_t>{1024, 1024, 1024});
  auto mods = generator.optimize(configs);
  for (auto mod: mods) {

    auto res = generator.lowering(mod);
    std::cout << "lowering status: " << res << "\n";
    auto result = generator.translate(mod);
    // std::cout << result << "\n";
  }
  #endif
  
}

int main(int argc, char* argv[]) {
  test();
  return 0;
}