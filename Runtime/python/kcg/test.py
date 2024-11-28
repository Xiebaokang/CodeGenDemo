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
from kcg.KCGCompiler import KCGCompiler

############### User config ###############
m_len=64
n_len=64
k_len=64

kp_matmul = KernelArgMatmul(m_len,n_len,k_len,
    EnumKernelDType.float32 ,
    EnumKernelDType.float32 ,
    EnumKernelDType.float32
    )

kp_matmul.BLOCK_SIZE_M= 64
kp_matmul.BLOCK_SIZE_N= 64
kp_matmul.BLOCK_SIZE_K= 16
kp_matmul.THREAD_SIZE_M= 4
kp_matmul.THREAD_SIZE_N= 4
kp_matmul.VECTORIZE_WIDTH= 4
kp_matmul.BLOCK_LAYOUT_M= 4
kp_matmul.BLOCK_LAYOUT_N= 1
kp_matmul.WARP_LAYOUT_M= 4
kp_matmul.WARP_LAYOUT_N= 16
kp_matmul.WARP_SIZE= 64

def compare_with_error(tensor1, tensor2, abs_error=1e-2, rel_error=1e-2):
    abs_diff = torch.abs(tensor1 - tensor2)
    rel_diff = abs_diff / (torch.abs(tensor1) + 1e-12)  # 避免除以零的情况

    # 比较绝对误差和相对误差
    error_mask = (abs_diff > abs_error) & (rel_diff > rel_error)
    diff_elements = torch.sum(error_mask).item()
    max_error = torch.max(torch.abs(tensor1 - tensor2))
    return diff_elements, max_error

def test_correctness(kpm : KernelArgMatmul):
    kernelCompiler = KCGCompiler()
    hsacoPath,kernelName = kernelCompiler.compileKernel(kpm)

    print("========= hsacoPath = ",hsacoPath)
    print("========= kernelName = ",kernelName)
    # funName = 'Matmul_m1024n1024k1024'

    inConfig = UserInputs(hsacoPath,kernelName,kpm)
    inConfig.operatorKind = EnumOperator.Matmul
    packedKernel = CompiledKernelFactory.getKernel(inConfig)

    a = torch.rand(kpm.M,kpm.K,dtype=inConfig.kernelParam.dtypeTorch('A'),device='cuda')
    b = torch.rand(kpm.K,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('B'),device='cuda')
    c = torch.empty(kpm.M,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('C'),device='cuda')
    d = torch.empty(kpm.M,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('C'),device='cuda')
    M, K = a.shape
    K, N = b.shape
    packedKernel.run(a,b,c)
    # o.run(a,b,c,
    #       M,N,K, a.stride(0), a.stride(1),  
    #         b.stride(0), b.stride(1),  
    #         c.stride(0), c.stride(1),  
    #     )
    print(c)
    d = torch.matmul(a,b)
    if torch.allclose(c,d,atol=1e-2,rtol=1e-2):
        print('test correct!')
    else:
        diff,max_error= compare_with_error(d,c)
        print('test fail! maxerror = ',max_error, '; diff=',diff)
        # hipprof --pmc python ./test.py 

test_correctness(kp_matmul)