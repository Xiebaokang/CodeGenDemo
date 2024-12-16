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
# import pymlir

############### User config ###############
m_len=256  # 16 blocks
n_len=32  # 16 blocks
k_len=8   # 8 blocks
normal_case = False

def case_normal_0(kp_matmul: KernelArgMatmul) :
    kp_matmul.BLOCK_SIZE_M= 64  # 256
    kp_matmul.BLOCK_SIZE_N= 64  # 32
    kp_matmul.BLOCK_SIZE_K= 16  # 8
    kp_matmul.THREAD_SIZE_M= 4
    kp_matmul.THREAD_SIZE_N= 4
    kp_matmul.VECTORIZE_WIDTH= 4  # 数字的个数
    kp_matmul.BLOCK_LAYOUT_M= 4
    kp_matmul.BLOCK_LAYOUT_N= 1
    kp_matmul.WARP_LAYOUT_M= 4
    kp_matmul.WARP_LAYOUT_N= 16
    kp_matmul.WARP_SIZE= 64
    kp_matmul.isATranspose = 0

def case_bad_0(kp_matmul: KernelArgMatmul) :
    # Cijk_Ailk_Bljk_SB_MT256x32x8_SN_APM1_AF0EM1_AF1EM1_AMAS3_ASAE01_ASCE01_ASEM1_BL1_DTL0_ETSP_EPS1_FL0_GRVW4_GSU1_GSUAMB_ISA906_IU1_K1_KLA_LPA0_LPB0_LDL1_LRVW4_MAC_MDA2_NLCA1_NLCB1_ONLL1_PK0_PGR1_PLR1_RK0_SU32_SUM0_SUS256_SVW4_SNLL0_TT8_4_USFGROn1_VAW1_VSn1_VW4_WG32_8_1_WGM1
    kp_matmul.BLOCK_SIZE_M= 256  # 256
    kp_matmul.BLOCK_SIZE_N= 32 # 32
    kp_matmul.BLOCK_SIZE_K= 8  # 8
    kp_matmul.THREAD_SIZE_M= 8
    kp_matmul.THREAD_SIZE_N= 4  # th=32*8
    kp_matmul.VECTORIZE_WIDTH= 4  # 数字的个数
    kp_matmul.BLOCK_LAYOUT_M= 2
    kp_matmul.BLOCK_LAYOUT_N= 2
    kp_matmul.WARP_LAYOUT_M= 16
    kp_matmul.WARP_LAYOUT_N= 4
    kp_matmul.WARP_SIZE= 64
    kp_matmul.isATranspose = 0

kp_matmul = KernelArgMatmul(m_len,n_len,k_len,
    EnumKernelDType.float32 ,
    EnumKernelDType.float32 ,
    EnumKernelDType.float32
    )

# case_normal_0(kp_matmul)
case_bad_0(kp_matmul)

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
    ###  ====  DBG：使用外部mlir调试   
    # funName = 'Matmul_m1024n1024k1024'
    # kernelName = 'GEMM_mnk1024x1024x512_f32f32f32_TTmn4x4_BTmnk64x64x16'

    inConfig = UserInputs(hsacoPath,kernelName,kpm)
    inConfig.operatorKind = EnumOperator.Matmul
    packedKernel = CompiledKernelFactory.getKernel(inConfig)
    warmupCount = 3
    a = torch.empty(kpm.M,kpm.K,dtype=inConfig.kernelParam.dtypeTorch('A'),device='cuda')
    b = torch.empty(kpm.K,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('B'),device='cuda')
    c = torch.empty(kpm.M,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('C'),device='cuda')
    for i in range(0,warmupCount) : # warmup
        packedKernel.run(a,b,c)
        torch.matmul(a,b)
        
    a = torch.rand(kpm.M,kpm.K,dtype=inConfig.kernelParam.dtypeTorch('A'),device='cuda')
    b = torch.rand(kpm.K,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('B'),device='cuda')
    c = torch.empty(kpm.M,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('C'),device='cuda')
    d = torch.empty(kpm.M,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('C'),device='cuda')
    M, K = a.shape
    K, N = b.shape
    atrans = torch.transpose(a,0,1)
    if kpm.isATranspose :
        packedKernel.run(atrans,b,c)
    else:
        packedKernel.run(a,b,c)
        
    # o.run(a,b,c,
    #       M,N,K, a.stride(0), a.stride(1),  
    #         b.stride(0), b.stride(1),  
    #         c.stride(0), c.stride(1),  
    #     )
    d = torch.matmul(a,b)
    print(c)
    if torch.allclose(c,d,atol=1e-2,rtol=1e-2):
        print('test correct!')
    else:
        diff,max_error= compare_with_error(d,c)
        print('test fail! maxerror = ',max_error, '; diff=',diff)
        # hipprof --pmc python ./test.py 

kp_matmul.check()
test_correctness(kp_matmul)
