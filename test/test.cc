#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"

using namespace KernelCodeGen;
using Config = std::map<std::string, std::vector<std::map<std::string, int>>>;

void test() {
  KernelCodeGenerator generator("CUDA");

  Config configs = {
    {"Matmul", {
        {{"BLOCK_SIZE_M", 128}, {"BLOCK_SIZE_N", 128}, {"BLOCK_SIZE_K", 8}, {"THREAD_SIZE_M", 8}, {"THREAD_SIZE_N", 8}, {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 32}}
        // {{"BLOCK_SIZE_M", 128}, {"BLOCK_SIZE_N", 128}, {"BLOCK_SIZE_K", 16}, {"THREAD_SIZE_M", 8}, {"THREAD_SIZE_N", 8}, {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 32}}
      }}
  };

  generator.create<Matmul>(std::vector<int64_t>{1024, 1024, 1024});

  auto mods = generator.optimize(configs);
  for (auto mod: mods) {
    // mod.dump();
    auto res = generator.lowering(mod);
    std::cout << res << "\n";
  }


}


int main(int argc, char* argv[]) {

  test();

}