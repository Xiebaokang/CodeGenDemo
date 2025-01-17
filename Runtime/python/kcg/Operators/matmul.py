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
def getMatmulSignature(dtypeA: torch.dtype, dtypeB : torch.dtype, dtypeC : torch.dtype):
    # signature只和输入的dtype有关，尺寸无关
    a = torch.randn((1024, 1024), device='cuda', dtype=dtypeA)
    b = torch.randn((1024, 1024), device='cuda', dtype=dtypeB)
    # Allocates output.
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device='cuda', dtype=dtypeC)
    # get function signature
    outSignature = _matmul(a, b, c)
    print("[D] mm signatrue = ",outSignature)
    return outSignature


class KernelArgMatmul :
    def __init__(self,m,n,k,typeA : EnumKernelDType,typeB : EnumKernelDType,typeC : EnumKernelDType):
        self.BLOCK_SIZE_M : int = 64
        self.BLOCK_SIZE_N : int = 64
        self.BLOCK_SIZE_K : int = 16
        self.THREAD_SIZE_M : int = 4
        self.THREAD_SIZE_N : int = 4
        self.VECTORIZE_WIDTH : int = 4
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
    
