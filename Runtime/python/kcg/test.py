# from Runtime.python.kcg.Loader import driver
# if __name__ == "__main__":
#     print("hello")
#     print(driver)
#     print(driver.loader)

from typing import List,Type
from kcg.Utils import *
from kcg.Kernel import *
from kcg.CompiledKernelFactory import *
from kcg.Operators import matmul
import sys


class kcgData:
    hsacoPath='/home/pangyunfei/xushilong/KernelCodeGen/kernel.hsaco'
    funName = 'Matmul_m1024n1024k1024_n9hNXH23O6jBoXmTORYQ'

class tritonData_finalLL_opt: # OK
    funName = "matmul_kernel_0d1d2d3de4de5de6de7c8de9c10de11c"
    hsacoPath = "/tmp/kcg_kernel-1dffff/kcg_kernel-1dffff.hsaco"
    res='''
        hsaco OK. amdgcn 有符号表
    '''

class tritonData_finalLL_nopt: # OK
    funName = "matmul_kernel_0d1d2d3de4de5de6de7c8de9c10de11c"
    hsacoPath='/tmp/kcg_kernel-037a94/kcg_kernel-037a94.hsaco'
    res='''
        hsaco OK. amdgcn 有符号表
    '''

class tritonData_midLL_opt:
    funName = "matmul_kernel_0d1d2d3de4de5de6de7c8de9c10de11c"
    hsacoPath = '/tmp/kcg_kernel-bee8d3/kcg_kernel-bee8d3.hsaco'
    res = '''
====1
====2
"Cannot find Symbol"
Aborted (core dumped)

amdgcn: amdhsa.kernels:  []
tritonData_finalLL_nopt amdgcn: 正常.p2align = 8, 这个=2 
    '''

class tritonData_midLL_nopt:
    funName = "matmul_kernel_0d1d2d3de4de5de6de7c8de9c10de11c"
    hsacoPath = '/tmp/kcg_kernel-86e6bc/kcg_kernel-86e6bc.hsaco'
    res = '''
parsed info: data_size = 46
parsed info: shared = 50000
parsed info: device = 0
====1
====2
"Cannot find Symbol"
Aborted (core dumped)
产出amdgcn 没有符号表，正常.p2align = 8, 这个=2 
    '''

    
# '''
# 结论：搬运过来的amendFUnc逻辑缺少数据，不能直接作用在llIR上，还需要之前lowering过程里 mlir 的信息
#    基于mlirmodule的数据做amendFUnction ： 影响符号表生成
# '''




dataCollection = tritonData_midLL_nopt

inConfig = UserInputs(dataCollection.hsacoPath,dataCollection.funName)
inConfig.operatorKind = EnumOperator.Matmul
inConfig.dtype_0 = torch.float32
inConfig.dtype_1 = torch.float32

o = CompiledKernelFactory.getKernel(inConfig)
# 需要根据前端调用形式，对run的参数进行确定.并修改operator的参数模板
    # M, K = a.shape
    # K, N = b.shape
    # # Allocates output.
    # c = torch.empty((M, N), device=a.device, dtype=dataType)dataType 1D launch kernel where each block gets its own program.
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    # matmul_kernel[grid](
    #     a, b, c,  #
    #     M, N, K,  #
    #     a.stride(0), a.stride(1),  #
    #     b.stride(0), b.stride(1),  #
    #     c.stride(0), c.stride(1),  #
    #     ACTIVATION=activation  #
    # )
dim = 1024
a = torch.rand(dim,dim,dtype=inConfig.dtype_0,device='cuda')
b = torch.rand(dim,dim,dtype=inConfig.dtype_1,device='cuda')
c = torch.empty(dim,dim,dtype=inConfig.dtype_0,device='cuda')
d = torch.empty(dim,dim,dtype=inConfig.dtype_0,device='cuda')
M, K = a.shape
K, N = b.shape
# o.run(a,b,c)

o.run(a,b,c,
      M,N,K, a.stride(0), a.stride(1),  
        b.stride(0), b.stride(1),  
        c.stride(0), c.stride(1),  
    )
print(c)
d = torch.matmul(a,b)
if torch.allclose(c,d,atol=1e-2,rtol=1e-2):
    print('test correct!')
else:
    print('test fail')