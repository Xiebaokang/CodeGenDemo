# KCG 项目介绍
## 1 项目结构

python前端 + MLIR后端
目前只简单地封装了调用核函数的逻辑，可实现用户在python端传入config，之后进行kenrel生成并执行
缓存系统还没做。因此目前会生成很多文件

MLIR后端入口点：test/test.cc  提供了lib和exe两种编译选项。在CMakeLists.txt中设置ON或OFF，或者：
```sh
cmake .. -DCOMPILE_AS_PYMODULE=ON
```

## 2.构建&运行方法
### 2.1 构建
新建build文件夹并进入。之后：
```sh
cmake .. -DCOMPILE_AS_PYMODULE=ON
make -j4
# 直接cmake .. 也可，会使用CMakeCache.txt中的配置
```

当需要切换构建选项时，需要删除build 目录下的CMakeCache.txt 否则会不生效

### 2.2 参数配置&运行
1. exe模式   
参数配置：debug用，只能用固定参数配置，在main函数中修改。只用于测试MLIR后端的lowering过程，不进行gemm的正确性测试
运行：
```sh
${project_folder}/bin/kcg_compiler > log.txt 2>&1
```

2. lib模式   
参数配置：
参见test.py中的该部分：

```python
ts = TilingScheme()
ts.BLOCK_SIZE_M= 64
ts.BLOCK_SIZE_N=64
ts.BLOCK_SIZE_K=16
ts.THREAD_SIZE_M= 4
ts.THREAD_SIZE_N= 4
ts.VECTORIZE_WIDTH= 4
ts.WARP_SIZE= 64 
ts.BLOCK_LAYOUT_M= 4
ts.BLOCK_LAYOUT_N= 1
ts.WARP_LAYOUT_M= 4
ts.WARP_LAYOUT_N= 16
```
问题规模目前固定为1024，位置：
```python
# Runtime/python/kcg/CompiledKernelFactory.py:
# 用户输入：hsacopath，kernel名字(通过amdgcn获取)，
class CompiledKernelFactory :
    @staticmethod
    def getKernel(info : UserInputs) -> CompiledKernel:
        if info.operatorKind==EnumOperator.Matmul :
            signature = getMatmulSignature(info.dtype_0,info.dtype_1)
            m,n,k = 1024,1024,1024
            return CompiledKernel(
                info.hsacoPath,
                info.kernelFuncName,
                info.sharedMem(m,n,k),
                signature,
                info.gridDims(m,n,k),
                info.blockDims()
            )
        if info.operatorKind==EnumOperator.Convolution :
            return None
        if info.operatorKind==EnumOperator.Poll:
            return None
```
以及test.py中：
```python
dim = 1024
a = torch.rand(dim,dim,dtype=inConfig.dtype_0,device='cuda')
b = torch.rand(dim,dim,dtype=inConfig.dtype_1,device='cuda')
c = torch.empty(dim,dim,dtype=inConfig.dtype_0,device='cuda')
d = torch.empty(dim,dim,dtype=inConfig.dtype_0,device='cuda')
M, K = a.shape
K, N = b.shape
o.run(a,b,c)
```

运行：
```sh
conda activate triton_rocm
cd ${project_folder}/Runtime/python 
export PYTHONPATH=`pwd`
cd ./kcg
python test.py > log.log 2>&1
# 测试性能： hipprof --pmc python test.py > log.log 2>&1
```

