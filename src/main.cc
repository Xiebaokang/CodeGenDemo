#define PY_SSIZE_T_CLEAN
#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
#include "Target/HSACOTranslation.h"
#include <stdio.h>
#include <stdlib.h>
#include "Common/Utils.h"
#include "Common/ThreadPool.h"
#include "Operators/Matmul.h"

#ifdef COMPILE_AS_PYMODULE
#include "Python.h"
#endif

#define DBG_USE_EXTERN_MLIR 0

using namespace KernelCodeGen;

std::string global_json_path{};


enum class MdlOperatorType : int{
  Matmul = 1,
  Convolution = 2,
  Pool = 3
};


template<typename T>
void paramCombine(std::stringstream& ss, const char* title, T p1, T p2, T p3)  {
  ss << title << p1 << "x" << p2 << "x"<< p3 << "_";
}

template<typename T>
void paramCombine(std::stringstream& ss, const char* title, T p1, T p2)  {
  ss << title << p1 << "x" << p2 << "_";
}
template<typename T>
void paramCombine(std::stringstream& ss, const char* title, T p1)  {
  ss << title << p1 << "_";
}



std::string getKernelName(const Config& cfg)  {
  std::stringstream ss;
  ss << "GEMM_";
  paramCombine(ss, "MNK", cfg.at(KEY_M),cfg.at(KEY_N),cfg.at(KEY_K));
  paramCombine(ss, "DTabc", tools::KcgDtypeToStr((KcgDtype)cfg.at(KEY_DTYPE_A)),
          tools::KcgDtypeToStr((KcgDtype)cfg.at(KEY_DTYPE_B)),
          tools::KcgDtypeToStr((KcgDtype)cfg.at(KEY_DTYPE_C)));
  paramCombine(ss, "AT", cfg.at(KEY_IS_A_TRANSPOSE));
  paramCombine(ss, "TTmn", cfg.at(KEY_THREAD_SIZE_M), cfg.at(KEY_THREAD_SIZE_N));
  paramCombine(ss, "BTmnk", cfg.at(KEY_BLOCK_SIZE_M), cfg.at(KEY_BLOCK_SIZE_N), cfg.at(KEY_BLOCK_SIZE_K));
  paramCombine(ss, "BLmn", cfg.at(KEY_BLOCK_LAYOUT_M), cfg.at(KEY_BLOCK_LAYOUT_N));
  paramCombine(ss, "WLmn", cfg.at(KEY_WARP_LAYOUT_M),  cfg.at(KEY_WARP_LAYOUT_N));
  paramCombine(ss, "GLWab",cfg.at(KEY_GLOB_LOAD_WIDTH_A),cfg.at(KEY_GLOB_LOAD_WIDTH_B));
  paramCombine(ss, "GSW",cfg.at(KEY_GLOB_STORE_WIDTH));
  paramCombine(ss, "WSWab",cfg.at(KEY_WARP_SCATTER_WIDTH_A),cfg.at(KEY_WARP_SCATTER_WIDTH_B));
  paramCombine(ss, "TSWab",cfg.at(KEY_THREAD_SCATTER_WIDTH_A),cfg.at(KEY_THREAD_SCATTER_WIDTH_B));
  paramCombine(ss, "LSU",cfg.at(KEY_LOCAL_SPLIT_U));
  paramCombine(ss, "BM",cfg.at(KEY_BLOCK_MAPPING));

  return ss.str();
}


std::vector<KernelInfo> generateKernels(
  std::vector<Config>& configs,
  const std::vector<std::string>& kernelNames) 
{
  assert(configs.size() == kernelNames.size() && "configs & kernelNames size match");
  std::map<Config, KernelInfo> result;
  ThreadPool pool(10);
  pool.init();
  // const int cfgszie = configs.size();
  const int cfgszie = 20;
  KernelCodeGenerator gen(Target::ROCm, "906");
  mlir::registerAllPasses();
  for (int i=0;i< cfgszie;++i) {
    std::function<KernelInfo(Config)> task = [=](Config cfg)->KernelInfo
    {
      KernelCodeGenerator generator = gen;
      const auto config = cfg;
      const auto name = kernelNames[i];
      KernelInfo info;
      auto dtypeA = tools::KcgDtypeToStr((KcgDtype)config.at(KEY_DTYPE_A));
      auto dtypeB = tools::KcgDtypeToStr((KcgDtype)config.at(KEY_DTYPE_B));
      auto dtypeC = tools::KcgDtypeToStr((KcgDtype)config.at(KEY_DTYPE_C));
      auto M = config.at(KEY_M);
      auto N = config.at(KEY_N);
      auto K = config.at(KEY_K);
      bool isATranspose = config.at(KEY_IS_A_TRANSPOSE)> 0;
      auto kernel = generator.create<Operators::Matmul>(
        std::vector<int64_t>{M, N, K},
        std::vector<std::string>{dtypeA,dtypeB,dtypeC},
        name,isATranspose
      );

      auto res1 = generator.optimize(kernel, config);
      std::cout << "==== optimize status: " << (res1?"SUCCESS":"FAILED") << "\n";
      auto res2 = generator.lowering(kernel);
      std::cout << "==== lowering status: " << (res2?"SUCCESS":"FAILED") << "\n";
      std::string hsacoPath = generator.translate(kernel);
      std::cout << "==== translate res :" << "\n";
      std::cout << hsacoPath << "\n";
      info.m_hsacoPath = hsacoPath;
      info.m_kernelName = generator.kernelFuncName<Operators::Matmul>();
      auto gridDim = tools::getIntArrayAttr(kernel,AttrGridDim);
      auto blockDim = tools::getIntArrayAttr(kernel,AttrBlockDim);
      info.m_gridDims = gridDim;
      info.m_blockDims = blockDim;
      // result[config] = info;
      std::cout << "==== kernel name : " << info.m_kernelName << "\n";
      return info;
    };  // end std::function<>
    pool.push_task(std::move(task),configs[i]);
  }
  pool.wait_finish(cfgszie);
  return pool.get_result();
}

std::vector< KernelInfo> _compile() {
  std::vector<Config> cfgs{};
  std::vector<std::string> names = {};
  if(tools::parseJsonToConfigs(global_json_path,cfgs)){
    for(auto config : cfgs){
      names.push_back(getKernelName(config)) ;
    }
  }
  return generateKernels(cfgs,names);
}




#ifdef COMPILE_AS_PYMODULE
static PyObject* compile_kernel_matmul(PyObject* self, PyObject* args) {
  const char* json;
  if(PyArg_ParseTuple(args, "s", &json)){
    global_json_path = json;
    std::cout << "json_path=" << global_json_path << std::endl; 
  }
  else{
    std::cout << "json_path parse error" << std::endl; 
  }
  std::vector<KernelInfo> kernels;
  std::string hsacoPath;
  Py_BEGIN_ALLOW_THREADS;
  kernels = _compile();
  Py_END_ALLOW_THREADS;
  Py_INCREF(Py_None);
  // return Py_None;
  PyObject *retArr;
  retArr = PyTuple_New(kernels.size());
    // 填充元组
  for (int i=0;i<kernels.size();++i) {
    // 假设元素数组元素是以字符串和整数对的方式存在
    // 这里你需要将每一对 (ss, i...) 插入
    const auto& kernel = kernels[i];
    PyObject* item = Py_BuildValue("(ssiiiiii)",
      kernel.m_hsacoPath.c_str(),
      kernel.m_kernelName.c_str(),
      kernel.m_gridDims[0],kernel.m_gridDims[1],kernel.m_gridDims[2],
      kernel.m_blockDims[0],kernel.m_blockDims[1],kernel.m_blockDims[2]
    );
    if (item == NULL) {
      Py_DECREF(retArr);
      return NULL;  // 如果构建某个元素失败，释放资源并返回NULL
    }
    PyTuple_SetItem(retArr, i, item);  // 将每个元素插入元组
  }
  return retArr;
}



static PyMethodDef ModuleMethods[] = {
    {"compile_kernel_matmul", compile_kernel_matmul, METH_VARARGS,"compile kernel according to user input tiling and type"},
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

int testCfg( std::vector<Config> configs){

  std::vector<std::string> names = {10,"GEMM_testKernel"};
  auto result = generateKernels(configs, names);
  return 0;
}


int main(){

  // using namespace KernelCodeGen;
  // ThreadPool p(10);
  // p.init();
  // std::function<KernelInfo(void)> f = [](void)->KernelInfo
  // {
  //   std::this_thread::sleep_for(std::chrono::seconds(2));
  //   return KernelInfo();
  // };
  // for(int i=0;i<15;++i){
  //   p.push_task(f);
  // }
  // p.wait_finish(15);

  std::vector<Config> cfgs;
  global_json_path = "/home/xushilong/CodeGenDemo/cfg_cominations.json";
  auto ret = _compile();
  return 0;
}


#endif

