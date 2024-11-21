#####  测试用。接口不完善，只是提供了相对便利的构造 CompiledKernel 的方式

from kcg.CompiledKernel import *
from kcg.Operators.matmul import *

class EnumOperator(Enum):
    Invalid = 0
    Matmul = 1
    Convolution = 2
    Poll = 3
    def __str__(self):
        return f'{self.name}'

class TilingScheme :
    def __init__(self):
        self.BLOCK_SIZE_M= 64
        self.BLOCK_SIZE_N=64
        self.BLOCK_SIZE_K=16
        self.THREAD_SIZE_M= 4
        self.THREAD_SIZE_N= 4
        self.VECTORIZE_WIDTH= 4
        self.WARP_SIZE= 64 
        self.BLOCK_LAYOUT_M= 4
        self.BLOCK_LAYOUT_N= 1
        self.WARP_LAYOUT_M= 4
        self.WARP_LAYOUT_N= 16
        
        # 1024/ self.BLOCK_SIZE_M, 1024/ self.BLOCK_SIZE_N 

class UserInputs:
    def __init__(self,hsaco_path:str,kernel_func_name:str,tileScheme : TilingScheme):
        # 自定义dtype的代表含义。如：matmul使用0、1分别代表A和B的元素类型
        self.dtype_0 = torch.float32
        self.dtype_1 = torch.float32
        self.dtype_2 = torch.float32
        self.dtype_3 = torch.float32
        self.dtype_4 = torch.float32
        self.operatorKind = EnumOperator.Invalid
        self.hsacoPath = hsaco_path
        self.kernelFuncName = kernel_func_name
        self.tiling = tileScheme

    def gridDims(self,m,n,k):
        return [ m // self.tiling.BLOCK_SIZE_M,n // self.tiling.BLOCK_SIZE_N,1 ]
    
    def blockDims(self):
        return [self.tiling.BLOCK_LAYOUT_M * self.tiling.WARP_LAYOUT_M,
                self.tiling.BLOCK_LAYOUT_N * self.tiling.WARP_LAYOUT_N, 1 ]
    
# 用户输入：hsacopath，kernel名字(通过amdgcn获取)，
class CompiledKernelFactory :
    @staticmethod
    def getKernel(info : UserInputs) -> CompiledKernel:
        if info.operatorKind==EnumOperator.Matmul :
            signature = getMatmulSignature(info.dtype_0,info.dtype_1)
            return CompiledKernel(
                info.hsacoPath,
                info.kernelFuncName,
                16800,
                signature,
                info.gridDims(1024,1024,1024),
                info.blockDims()
            )
        if info.operatorKind==EnumOperator.Convolution :
            return None
        if info.operatorKind==EnumOperator.Poll:
            return None