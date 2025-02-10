import torch
from kcg.Kernel import kcg_kernel
from kcg.Utils import *

# # 核函数stub. 用于提供 Kernel 形参列表
# @kcg_kernel
# def _matmul_kernel_triton(
#         # Pointers to matrices
#         a_ptr, b_ptr, c_ptr,
#         # # Matrix dimensions
#         M, N, K,
#         # The stride variables represent how much to increase the ptr by when moving by 1
#         # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
#         # by to get the element one row down (A has M rows).
#         stride_am, stride_ak,
#         stride_bk, stride_bn,
#         stride_cm, stride_cn,
#         # Meta-parameters
#         # BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
#         # EVEN_K: tl.constexpr,
#         # GROUP_SIZE_M: tl.constexpr,
#         # ACTIVATION: tl.constexpr,
# ):
#     pass

@kcg_kernel
def _matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr
):
    '''
    Dump code here
    '''
    pass

# Call hook. 在这里带入实参并调用

def _matmul(a, b, c):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"


    # 1D launch kernel where each block gets its own program.
    
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    return _matmul_kernel(
        a, b, c
    )
    # return _matmul_kernel_triton(
    #     a, b, c,  #
    #     M, N, K,  #
    #     a.stride(0), a.stride(1),  #
    #     b.stride(0), b.stride(1),  #
    #     c.stride(0), c.stride(1),  #
    # )


# public interface:
def getMatmulSignature(dtypeA: torch.dtype, dtypeB : torch.dtype, dtypeC : torch.dtype) -> dict:
    # signature只和输入的dtype有关，尺寸无关
    a = torch.randn((1024, 1024), device='cuda', dtype=dtypeA)
    b = torch.randn((1024, 1024), device='cuda', dtype=dtypeB)
    # Allocates output.
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device='cuda', dtype=dtypeC)
    # get function signature
    outSignature = _matmul(a, b, c)
    # print(f"[D] mm signature = {outSignature}, type =  {type(outSignature.values())}",)
    return outSignature


class KernelArgMatmul :
    def __init__(self,m,n,k,typeA : EnumKernelDType,typeB : EnumKernelDType,typeC : EnumKernelDType):
        self.BLOCK_SIZE_M : int = 64
        self.BLOCK_SIZE_N : int = 64
        self.BLOCK_SIZE_K : int = 16
        self.THREAD_SIZE_M : int = 4
        self.THREAD_SIZE_N : int = 4
        self.WARP_SIZE : int = 64 
        self.BLOCK_LAYOUT_M : int = 4
        self.BLOCK_LAYOUT_N : int = 1
        self.WARP_LAYOUT_M : int = 16
        self.WARP_LAYOUT_N : int = 4
        self.__dataType_A : EnumKernelDType = typeA
        self.__dataType_B : EnumKernelDType = typeB
        self.__dataType_C : EnumKernelDType = typeC
        self.M : int = m
        self.N : int = n
        self.K : int = k
        self.isATranspose : int = 1
        self.GLOB_LOAD_WIDTH_A : int = 0
        self.GLOB_LOAD_WIDTH_B : int = 0
        self.WARP_SCATTER_WIDTH_A : int = 0
        self.WARP_SCATTER_WIDTH_B : int = 0
        self.THREAD_SCATTER_WIDTH_A : int = 0
        self.THREAD_SCATTER_WIDTH_B : int = 0
        self.LOCAL_SPLIT_U : int = 0
        self.BLOCK_MAPPING : int = 0
        self.GLOB_STORE_WIDTH : int = 0
        
        self.UNROLL_NUM : int = 1
        self.REG_PREFETCH : int = 0
        self.SHARED_PREFETCH : int = 0
        self.LOAD_CONTINUOUS : int = 0
        self.REDUCE_C_CONTINUOUS : int = 0
        
    def check(self) :
        # problem size check
        assert self.M % self.BLOCK_SIZE_M == 0 
        assert self.N % self.BLOCK_SIZE_N == 0 
        assert self.K % self.BLOCK_SIZE_K == 0 
        # warp-block validation check
        assert self.BLOCK_SIZE_M % self.THREAD_SIZE_M == 0
        assert self.BLOCK_SIZE_N % self.THREAD_SIZE_N == 0
        assert (self.BLOCK_LAYOUT_M * self.WARP_LAYOUT_M) == (self.BLOCK_SIZE_M / self.THREAD_SIZE_M)
        assert (self.BLOCK_LAYOUT_N * self.WARP_LAYOUT_N) == (self.BLOCK_SIZE_N / self.THREAD_SIZE_N)
        assert self.WARP_LAYOUT_N * self.WARP_LAYOUT_M == self.WARP_SIZE
        # shm size check
        assert 2*(self.BLOCK_SIZE_M + self.BLOCK_SIZE_N) * self.BLOCK_SIZE_K <= 65536
        
        print("===== config check ok!")
    
    def dtype(self,index:str)->EnumKernelDType :
        if index=='A':
            return self.__dataType_A
        if index=='B':
            return self.__dataType_B
        if index=='C':
            return self.__dataType_C
    
    def dtypeTorch(self,index:str)->torch.dtype:
        if index=='A':
            return ToTorchType(self.__dataType_A)
        if index=='B':
            return ToTorchType(self.__dataType_B)
        if index=='C':
            return ToTorchType(self.__dataType_C)
    
    def __str__(self):
        retstr = '{\n'
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_SIZE_M)}\" :  {str(self.BLOCK_SIZE_M)} , \n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_SIZE_N)}\"  :  {str(self.BLOCK_SIZE_N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_SIZE_K)}\"  :  {str(self.BLOCK_SIZE_K)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_THREAD_SIZE_M)}\"  :  {str(self.THREAD_SIZE_M)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_THREAD_SIZE_N)}\"  :  {str(self.THREAD_SIZE_N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_SIZE)}\"  :  {str(self.WARP_SIZE)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_LAYOUT_M)}\"  :  {str(self.BLOCK_LAYOUT_M)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_LAYOUT_N)}\"  :  {str(self.BLOCK_LAYOUT_N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_LAYOUT_M)}\"  :  {str(self.WARP_LAYOUT_M)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_LAYOUT_N)}\"  :  {str(self.WARP_LAYOUT_N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_DTYPE_A)}\"  :  {str(self.__dataType_A)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_DTYPE_B)}\"  :  {str(self.__dataType_B)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_DTYPE_C)}\"  :  {str(self.__dataType_C)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_M)}\"  :  {str(self.M)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_N)}\"  :  {str(self.N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_K)}\"  :  {str(self.K)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_IS_A_TRANSPOSE)}\"  :  {str(self.isATranspose)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_GLOB_LOAD_WIDTH_A)}\"  :  {str(self.GLOB_LOAD_WIDTH_A)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_GLOB_LOAD_WIDTH_B)}\"  :  {str(self.GLOB_LOAD_WIDTH_B)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_A)}\"  :  {str(self.WARP_SCATTER_WIDTH_A)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_B)}\"  :  {str(self.WARP_SCATTER_WIDTH_B)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_A)}\"  :  {str(self.THREAD_SCATTER_WIDTH_A)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_B)}\"  :  {str(self.THREAD_SCATTER_WIDTH_B)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_LOCAL_SPLIT_U)}\"  :  {str(self.LOCAL_SPLIT_U)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_MAPPING)}\"  :  {str(self.BLOCK_MAPPING)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_GLOB_STORE_WIDTH)}\"  :  {str(self.GLOB_STORE_WIDTH )} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_UNROLL_NUM)}\"  :  {str(self.UNROLL_NUM)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_REG_PREFETCH)}\"  :  {str(self.REG_PREFETCH)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_SHARED_PREFETCH)}\"  :  {str(self.SHARED_PREFETCH)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_LOAD_CONTINUOUS)}\"  :  {str(self.LOAD_CONTINUOUS)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_REDUCE_C_CONTINUOUS)}\"  :  {str(self.REDUCE_C_CONTINUOUS)} \n"
        retstr += '}'
        return retstr