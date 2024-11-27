import torch
from kcg.Kernel import kcg_kernel

# 核函数stub. 用于提供 Kernel 形参列表
@kcg_kernel
def _matmul_kernel_triton(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        # BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        # EVEN_K: tl.constexpr,
        # GROUP_SIZE_M: tl.constexpr,
        # ACTIVATION: tl.constexpr,
):
    pass

@kcg_kernel
def _matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr
):
    pass

# Call hook. 在这里带入实参并调用

def _matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
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
def getMatmulSignature(dtypeA: torch.dtype, dtypeB : torch.dtype):
    # signature只和输入的dtype有关，尺寸无关
    a = torch.randn((1024, 1024), device='cuda', dtype=dtypeA)
    b = torch.randn((1024, 1024), device='cuda', dtype=dtypeB)
    # get function signature
    outSignature = _matmul(a, b)
    print("[D] signatrue = ",outSignature)
    return outSignature
