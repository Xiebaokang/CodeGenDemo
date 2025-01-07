#define PY_SSIZE_T_CLEAN
#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
#include "Target/HSACOTranslation.h"
#include <stdio.h>
#include <stdlib.h>
#include "Common/Utils.h"
#include "Operators/Matmul.h"
#ifdef COMPILE_AS_PYMODULE
#include "Python.h"
#endif

#define DBG_USE_EXTERN_MLIR 0

using namespace KernelCodeGen;

using Config = std::map<std::string, int>;



enum class MdlOperatorType : int{
  Matmul = 1,
  Convolution = 2,
  Pool = 3
};


class MatmulParams {
public:
  KcgDtype m_dtypeA, m_dtypeB, m_dtypeC; // 3
  int m_size,n_size,k_size;  // 3
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
  int m_isATranspose = 0;  // 18
  // recently-added-params
  int m_GLOB_LOAD_WIDTH_A;
  int m_GLOB_LOAD_WIDTH_B;
  int m_WARP_SCATTER_WIDTH_A;
  int m_WARP_SCATTER_WIDTH_B;
  int m_THREAD_SCATTER_WIDTH_A;
  int m_THREAD_SCATTER_WIDTH_B;
  int m_LOCAL_SPLIT_U;
  int m_BLOCK_MAPPING;
  int m_GLOB_STORE_WIDTH;  // 27

#ifdef COMPILE_AS_PYMODULE
  bool parse(PyObject* args){
    if(PyArg_ParseTuple(args, std::string(27,'i').c_str(),
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
    // recent-added
      &m_GLOB_LOAD_WIDTH_A,
      &m_GLOB_LOAD_WIDTH_B,
      &m_WARP_SCATTER_WIDTH_A,
      &m_WARP_SCATTER_WIDTH_B,
      &m_THREAD_SCATTER_WIDTH_A,
      &m_THREAD_SCATTER_WIDTH_B,
      &m_LOCAL_SPLIT_U,
      &m_BLOCK_MAPPING,
      &m_GLOB_STORE_WIDTH,

      &m_dtypeA, &m_dtypeB, &m_dtypeC,
      &m_size,&n_size,&k_size,
      &m_isATranspose

    )){
      return true;
    }
    assert(false && "PyArg_ParseTuple Error");
    return false;
  }
#endif
  std::string getKernelName() const {
    std::stringstream ss;
    ss << "GEMM_mnk";
    ss << m_size << "x" << n_size << "x" << k_size << "_";
    ss << m_dtypeA << m_dtypeB << m_dtypeC << "_";
    ss << "TTmn" << m_THREAD_SIZE_M <<"x"<< m_THREAD_SIZE_N << "_";
    ss << "BTmnk" << m_BLOCK_SIZE_M << "x" << m_BLOCK_SIZE_N << "x" << m_BLOCK_SIZE_K << "_" ;
    ss << "BLmn" << m_BLOCK_LAYOUT_M << "x" << m_BLOCK_LAYOUT_N << "_";
    ss << "WLmn" << m_WARP_LAYOUT_M << "x" << m_WARP_LAYOUT_N;
    return ss.str();
  }
};

std::ostream& operator<<(std::ostream& os, MatmulParams arg){
  os << "== UserKernelCfg :\n";
  os << "- M : " << arg.m_size << std::endl;
  os << "- N : " << arg.n_size << std::endl;
  os << "- K : " << arg.k_size << std::endl;
  os << "- m_dtypeA : " << tools::KcgDtypeToStr(arg.m_dtypeA) << std::endl;
  os << "- m_dtypeB : " << tools::KcgDtypeToStr(arg.m_dtypeB) << std::endl;
  os << "- m_dtypeC : " << tools::KcgDtypeToStr(arg.m_dtypeC) << std::endl;
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
  os << "- m_isATranspose : " <<arg.m_isATranspose << std::endl;
  return os;
}

struct KernelInfo {
  std::string m_hsacoPath;
  std::string m_kernelName;
};

std::map<Config, KernelInfo> testConfigs(
  std::vector<Config>& configs,
  const std::vector<std::string>& kernelNames) 
{
  assert(configs.size() == kernelNames.size() && "configs & kernelNames size match");
  std::map<Config, KernelInfo> result;
  KernelCodeGenerator generator(Target::ROCm, "906");
  for (int i=0;i<configs.size();++i) {
    auto config = configs[i];
    const auto& name = kernelNames[i];
    KernelInfo info;
    auto dtypeA = tools::KcgDtypeToStr((KcgDtype)config[KEY_DTYPE_A]);
    auto dtypeB = tools::KcgDtypeToStr((KcgDtype)config[KEY_DTYPE_B]);
    auto dtypeC = tools::KcgDtypeToStr((KcgDtype)config[KEY_DTYPE_C]);
    auto M = config[KEY_M];
    auto N = config[KEY_N];
    auto K = config[KEY_K];
    bool isATranspose = config[KEY_IS_A_TRANSPOSE] > 0;
#if DBG_USE_EXTERN_MLIR
    MLIRContext ctx;
    ctx.loadDialect<mlir::affine::AffineDialect, 
    mlir::memref::MemRefDialect,
    mlir::arith::ArithDialect, 
    mlir::gpu::GPUDialect,
    mlir::func::FuncDialect,
    mlir::scf::SCFDialect,
    mlir::index::IndexDialect,
    mlir::vector::VectorDialect,
    mlir::cf::ControlFlowDialect,
    mlir::ROCDL::ROCDLDialect,
    mlir::LLVM::LLVMDialect
    >();
    llvm::outs() << " ---- Loading outMLIR\n" ;llvm::outs().flush();
    auto parsed = parseSourceFile<ModuleOp>("/home/pangyunfei/xushilong/CodeGenDemo/Runtime/python/kcg/error.mlir", &ctx);
    auto kernel = parsed.get();
    llvm::outs() << "=== outer MLIR = \n" ;llvm::outs().flush();kernel.dump();
#else

    auto kernel = generator.create<Operators::Matmul>(
      std::vector<int64_t>{M, N, K},
      std::vector<std::string>{dtypeA,dtypeB,dtypeC},
      name,isATranspose
    );

    auto res1 = generator.optimize(kernel, config);
    std::cout << "==== optimize status: " << (res1?"SUCCESS":"FAILED") << "\n";
#endif
    // auto res2 = generator.lowering(kernel);
    // std::cout << "==== lowering status: " << (res2?"SUCCESS":"FAILED") << "\n";
    // std::string hsacoPath = generator.translate(kernel);
    // std::cout << "==== translate res :" << "\n";
    // std::cout << hsacoPath << "\n";
    // info.m_hsacoPath = hsacoPath;
    // info.m_kernelName = generator.kernelFuncName<Operators::Matmul>();
    // result[config] = info;
    // std::cout << "==== kernel name : " << info.m_kernelName << "\n";
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
      {KEY_IS_A_TRANSPOSE,(int)cfg.m_isATranspose}
    }
  };
  std::vector<std::string> names = {cfg.getKernelName()};
  auto result = testConfigs(configs,names);
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
      {"BLOCK_SIZE_M", 64}, {"BLOCK_SIZE_N", 48}, {"BLOCK_SIZE_K", 32}, {"THREAD_SIZE_M", 4}, {"THREAD_SIZE_N", 6}, 
      {"GLOB_LOAD_WIDTH_A", 4}, {"GLOB_LOAD_WIDTH_B", 2}, 
      {"BLOCK_LAYOUT_Y", 2}, {"BLOCK_LAYOUT_X", 1}, {"WARP_LAYOUT_Y", 8}, {"WARP_LAYOUT_X", 8},
      {"WARP_SCATTER_WIDTH_A", 2}, {"WARP_SCATTER_WIDTH_B", 2}, {"THREAD_SCATTER_WIDTH_A", 2}, {"THREAD_SCATTER_WIDTH_B", 2}, 
      {"LOCAL_SPLIT_U", 2}, {"BLOCK_MAPPING", 8}, {"WARP_SIZE", 64}, {"GLOB_STORE_WIDTH", 2}, 
      {KEY_DTYPE_A, (int)KcgDtype::float16},
      {KEY_DTYPE_B, (int)KcgDtype::float16},
      {KEY_DTYPE_C, (int)KcgDtype::float16},
      {KEY_M, 1024},{KEY_N, 1024},{KEY_K, 1024}, 
      {KEY_IS_A_TRANSPOSE, 1}
    },
    // {
    //   {"BLOCK_SIZE_M", 64}, {"BLOCK_SIZE_N", 64}, {"BLOCK_SIZE_K", 32}, 
    //   {"THREAD_SIZE_M", 4}, {"THREAD_SIZE_N", 4}, {"VECTORIZE_WIDTH", 4}, {"WARP_SIZE", 64}, 
    //   {"BLOCK_LAYOUT_M", 4}, {"BLOCK_LAYOUT_N", 1}, {"WARP_LAYOUT_M", 4}, {"WARP_LAYOUT_N", 16}
    // }
  };
  std::vector<std::string> names = {"GEMM_testKernel"};
  auto result = testConfigs(configs, names);
  return 0;
}

#endif