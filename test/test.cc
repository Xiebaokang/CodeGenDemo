#define PY_SSIZE_T_CLEAN
#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
#include "Target/HSACOTranslation.h"
#include "Python.h"
#include <stdio.h>
#include <stdlib.h>


using namespace KernelCodeGen;

enum class MdlOperatorType : int{
  Matmul = 1,
  Convolution = 2,
  Pool = 3
};

struct UserTilingInputs {
  MdlOperatorType m_type;
  int m_BLOCK_SIZE_M;
  int m_BLOCK_SIZE_N;
  int m_BLOCK_SIZE_K;
  int m_THREAD_SIZE_M;
  int m_THREAD_SIZE_N;
  int m_VECTORIZE_WIDTH;
  int m_WARP_SIZE;
  int m_BLOCK_LAYOUT_M;
  int m_BLOCK_LAYOUT_N;
  int m_WARP_LAYOUT_M;
  int m_WARP_LAYOUT_N;
};

std::ostream& operator<<(std::ostream& os, UserTilingInputs arg){
  os << "==UserTilingInputs:\n";
  os << "- m_BLOCK_SIZE_M : " <<arg.m_BLOCK_SIZE_M << std::endl;
  os << "- m_BLOCK_SIZE_N : " <<arg.m_BLOCK_SIZE_N << std::endl;
  os << "- m_BLOCK_SIZE_K : " <<arg.m_BLOCK_SIZE_K << std::endl;
  os << "- m_THREAD_SIZE_M : " <<arg.m_THREAD_SIZE_M << std::endl;
  os << "- m_THREAD_SIZE_N : " <<arg.m_THREAD_SIZE_N << std::endl;
  os << "- m_VECTORIZE_WIDTH : " <<arg.m_VECTORIZE_WIDTH << std::endl;
  os << "- m_WARP_SIZE : " <<arg.m_WARP_SIZE << std::endl;
  os << "- m_BLOCK_LAYOUT_M : " <<arg.m_BLOCK_LAYOUT_M << std::endl;
  os << "- m_BLOCK_LAYOUT_N : " <<arg.m_BLOCK_LAYOUT_N << std::endl;
  os << "- m_WARP_LAYOUT_M : " <<arg.m_WARP_LAYOUT_M << std::endl;
  os << "- m_WARP_LAYOUT_N : " <<arg.m_WARP_LAYOUT_N << std::endl;
  return os;
}

std::string _compile(const UserTilingInputs& cfg ) {
  using Config = std::map<std::string, std::vector<std::map<std::string, int>>>;
  KernelCodeGenerator generator(Target::ROCm, "906");

  Config configs = {
    // {"Matmul", {
    //     {{"BLOCK_SIZE_M", 64}, {"BLOCK_SIZE_N", 64}, {"BLOCK_SIZE_K", 16}, {"THREAD_SIZE_M", 4}, {"THREAD_SIZE_N", 4}, {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 64}, 
    //     {"BLOCK_LAYOUT_M", 4}, {"BLOCK_LAYOUT_N", 1}, {"WARP_LAYOUT_M", 4}, {"WARP_LAYOUT_N", 16}}
    //     // {{"BLOCK_SIZE_M", 64}, {"BLOCK_SIZE_N", 64}, {"BLOCK_SIZE_K", 16}, {"THREAD_SIZE_M", 4}, {"THREAD_SIZE_N", 4}, {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 64}}
    //   }}
    {"Matmul", {
        {
          {"BLOCK_SIZE_M", cfg.m_BLOCK_SIZE_M}, {"BLOCK_SIZE_N", cfg.m_BLOCK_SIZE_N},
          {"BLOCK_SIZE_K", cfg.m_BLOCK_SIZE_K}, 
          {"THREAD_SIZE_M", cfg.m_THREAD_SIZE_M}, {"THREAD_SIZE_N", cfg.m_THREAD_SIZE_N}, 
          {"VECTORIZE_WIDTH", cfg.m_VECTORIZE_WIDTH}, {"WARP_SIZE", cfg.m_WARP_SIZE}, 
          {"BLOCK_LAYOUT_M", cfg.m_BLOCK_LAYOUT_M}, {"BLOCK_LAYOUT_N", cfg.m_BLOCK_LAYOUT_N}, 
          {"WARP_LAYOUT_M", cfg.m_WARP_LAYOUT_M}, {"WARP_LAYOUT_N", cfg.m_WARP_LAYOUT_N}
        }
        // {{"BLOCK_SIZE_M", 64}, {"BLOCK_SIZE_N", 64}, {"BLOCK_SIZE_K", 16}, {"THREAD_SIZE_M", 4}, {"THREAD_SIZE_N", 4}, {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 64}}
      }}
  };

  generator.create<Matmul>(std::vector<int64_t>{1024, 1024, 1024});
  auto mods = generator.optimize(configs);
  std::string hsacoPath = "";
  for (mlir::ModuleOp& mod: mods) {
    auto res = generator.lowering(mod);
    std::cout << "==== lowering status: " << (res?"SUCCESS":"FAILED") << "\n";
    hsacoPath = generator.translate(mod);
    std::cout << "==== translate res :" << "\n";
    std::cout << hsacoPath << "\n";
  }
  return hsacoPath;
}

static PyObject* compile_kernel(PyObject* self, PyObject* args) {
  UserTilingInputs cfg;
  cfg.m_type = MdlOperatorType::Matmul;
  if(!PyArg_ParseTuple(args, "iiiiiiiiiii",
    &cfg.m_BLOCK_SIZE_M,
    &cfg.m_BLOCK_SIZE_N,
    &cfg.m_BLOCK_SIZE_K,
    &cfg.m_THREAD_SIZE_M,
    &cfg.m_THREAD_SIZE_N,
    &cfg.m_VECTORIZE_WIDTH,
    &cfg.m_WARP_SIZE,
    &cfg.m_BLOCK_LAYOUT_M,
    &cfg.m_BLOCK_LAYOUT_N,
    &cfg.m_WARP_LAYOUT_M,
    &cfg.m_WARP_LAYOUT_N  )) {
    return NULL;
  }
  std::cout << cfg << std::endl;
  std::string hsacoPath;
  Py_BEGIN_ALLOW_THREADS;
  hsacoPath = _compile(cfg);
  Py_END_ALLOW_THREADS;
  Py_INCREF(Py_None);
  // return Py_None;
  return Py_BuildValue("s",hsacoPath.c_str());
}

static PyMethodDef ModuleMethods[] = {
    {"compile_kernel", compile_kernel, METH_VARARGS,
     "compile kernel according to user input tiling and type"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "KCGCompiler",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_KCGCompiler(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
