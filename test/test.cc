#define PY_SSIZE_T_CLEAN
#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
#include "Target/HSACOTranslation.h"
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#ifdef COMPILE_AS_PYMODULE
#include "Python.h"
#endif


using namespace KernelCodeGen;

using Config = std::map<std::string, int>;



enum class MdlOperatorType : int{
  Matmul = 1,
  Convolution = 2,
  Pool = 3
};


class MatmulParams {
public:
  KcgDtype m_dtypeA, m_dtypeB, m_dtypeC;
  int m_size,n_size,k_size;
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

  bool parse(PyObject* args){
    if(PyArg_ParseTuple(args, std::string(17,'i').c_str(),
      &m_BLOCK_SIZE_M,
      &m_BLOCK_SIZE_N,
      &m_BLOCK_SIZE_K,
      &m_THREAD_SIZE_M,
      &m_THREAD_SIZE_N,
      &m_VECTORIZE_WIDTH,
      &m_WARP_SIZE,
      &m_BLOCK_LAYOUT_M,
      &m_BLOCK_LAYOUT_N,
      &m_WARP_LAYOUT_M,
      &m_WARP_LAYOUT_N,
      &m_dtypeA, &m_dtypeB, &m_dtypeC,
      &m_size,&n_size,&k_size
    )){
      return true;
    }
    assert(false && "PyArg_ParseTuple Error");
    return false;
  }

};

std::ostream& operator<<(std::ostream& os, MatmulParams arg){
  os << "== UserKernelCfg :\n";
  os << "- M : " << arg.m_size << std::endl;
  os << "- N : " << arg.n_size << std::endl;
  os << "- K : " << arg.k_size << std::endl;
  os << "- m_dtypeA : " << KcgDtypeToStr(arg.m_dtypeA) << std::endl;
  os << "- m_dtypeB : " << KcgDtypeToStr(arg.m_dtypeB) << std::endl;
  os << "- m_dtypeC : " << KcgDtypeToStr(arg.m_dtypeC) << std::endl;
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

struct KernelInfo {
  std::string m_hsacoPath;
  std::string m_kernelName;
};

std::map<Config, KernelInfo> testConfigs(std::vector<Config> configs) {
  std::map<Config, KernelInfo> result;
  KernelCodeGenerator generator(Target::ROCm, "906");
  for (auto config : configs) {
    KernelInfo info;
    auto dtypeA = KcgDtypeToStr((KcgDtype)config[KEY_DTYPE_A]);
    auto dtypeB = KcgDtypeToStr((KcgDtype)config[KEY_DTYPE_B]);
    auto dtypeC = KcgDtypeToStr((KcgDtype)config[KEY_DTYPE_C]);
    auto M = config[KEY_M];
    auto N = config[KEY_N];
    auto K = config[KEY_K];
    auto kernel = generator.create<Matmul>(
      std::vector<int64_t>{M, N, K},
      std::vector<std::string>{dtypeA,dtypeB,dtypeC}
    );
    auto res1 = generator.optimize(kernel, config);
    std::cout << "==== optimize status: " << (res1?"SUCCESS":"FAILED") << "\n";
    auto res2 = generator.lowering(kernel);
    std::cout << "==== lowering status: " << (res2?"SUCCESS":"FAILED") << "\n";
    std::string hsacoPath = generator.translate(kernel);
    std::cout << "==== translate res :" << "\n";
    std::cout << hsacoPath << "\n";
    info.m_hsacoPath = hsacoPath;
    info.m_kernelName = generator.kernelFuncName<Matmul>();
    result[config] = info;
    std::cout << "==== kernel name : " << info.m_kernelName << "\n";
  }
  return result;
}

KernelInfo _compile(const MatmulParams& cfg) {
  std::vector<Config> configs = {
    {
      {KEY_BLOCK_SIZE_M, cfg.m_BLOCK_SIZE_M}, {KEY_BLOCK_SIZE_N, cfg.m_BLOCK_SIZE_N}, {KEY_BLOCK_SIZE_K, cfg.m_BLOCK_SIZE_K}, 
      {KEY_THREAD_SIZE_M, cfg.m_THREAD_SIZE_M}, {KEY_THREAD_SIZE_N, cfg.m_THREAD_SIZE_N}, 
      {KEY_VECTORIZE_WIDTH, cfg.m_VECTORIZE_WIDTH}, {KEY_WARP_SIZE, cfg.m_WARP_SIZE}, 
      {KEY_BLOCK_LAYOUT_M, cfg.m_BLOCK_LAYOUT_M}, {KEY_BLOCK_LAYOUT_N, cfg.m_BLOCK_LAYOUT_N}, 
      {KEY_WARP_LAYOUT_M, cfg.m_WARP_LAYOUT_M}, {KEY_WARP_LAYOUT_N, cfg.m_WARP_LAYOUT_N},
      {KEY_DTYPE_A, (int)cfg.m_dtypeA},{KEY_DTYPE_B, (int)cfg.m_dtypeB},{KEY_DTYPE_C, (int)cfg.m_dtypeC},
      {KEY_M, (int)cfg.m_size},{KEY_N, (int)cfg.n_size},{KEY_K, (int)cfg.k_size},
    }
  };
  auto result = testConfigs(configs);
  return result[configs[0]];
}

#ifdef COMPILE_AS_PYMODULE
static PyObject* compile_kernel_matmul(PyObject* self, PyObject* args) {
  MatmulParams cfg;
  if(!cfg.parse(args)){
    return NULL;
  }
  std::cout << cfg << std::endl;
  KernelInfo kernel;
  std::string hsacoPath;
  Py_BEGIN_ALLOW_THREADS;
  kernel = _compile(cfg);
  Py_END_ALLOW_THREADS;
  Py_INCREF(Py_None);
  // return Py_None;
  return Py_BuildValue("(ss)",kernel.m_hsacoPath.c_str(),kernel.m_kernelName.c_str());
}

static PyMethodDef ModuleMethods[] = {
    {"compile_kernel_matmul", compile_kernel_matmul, METH_VARARGS,
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

#else

int main(){

  std::vector<Config> configs = {
    {
      {KEY_BLOCK_SIZE_M, 64}, {KEY_BLOCK_SIZE_N, 64}, {KEY_BLOCK_SIZE_K, 16}, 
      {KEY_THREAD_SIZE_M, 4}, {KEY_THREAD_SIZE_N, 4}, {KEY_VECTORIZE_WIDTH, 4}, {KEY_WARP_SIZE, 64}, 
      {KEY_BLOCK_LAYOUT_M, 4}, {KEY_BLOCK_LAYOUT_N, 1}, {KEY_WARP_LAYOUT_M, 4}, {KEY_WARP_LAYOUT_N, 16},
      {KEY_DTYPE, (int)KcgDtype::float32}
    },
    // {
    //   {"BLOCK_SIZE_M", 64}, {"BLOCK_SIZE_N", 64}, {"BLOCK_SIZE_K", 32}, 
    //   {"THREAD_SIZE_M", 4}, {"THREAD_SIZE_N", 4}, {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 64}, 
    //   {"BLOCK_LAYOUT_M", 4}, {"BLOCK_LAYOUT_N", 1}, {"WARP_LAYOUT_M", 4}, {"WARP_LAYOUT_N", 16}
    // }
  };
  auto result = testConfigs(configs);
  return 0;
}

#endif