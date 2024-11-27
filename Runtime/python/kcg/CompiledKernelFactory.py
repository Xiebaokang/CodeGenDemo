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


        
        # 1024/ self.BLOCK_SIZE_M, 1024/ self.BLOCK_SIZE_N 

class UserInputs:
    def __init__(self,hsaco_path:str,kernel_func_name:str,kernelParam : KernelArgMatmul):
        self.operatorKind = EnumOperator.Matmul
        self.hsacoPath = hsaco_path
        self.kernelFuncName = kernel_func_name
        self.kernelParam = kernelParam

    def gridDims(self):
        m = self.kernelParam.M
        n = self.kernelParam.N
        return [ m // self.kernelParam.BLOCK_SIZE_M,n // self.kernelParam.BLOCK_SIZE_N,1 ]
    
    def blockDims(self):
        return [self.kernelParam.BLOCK_LAYOUT_M * self.kernelParam.WARP_LAYOUT_M,
                self.kernelParam.BLOCK_LAYOUT_N * self.kernelParam.WARP_LAYOUT_N, 1 ]
        
    def sharedMem(self):
        # 假设 ABC类型相同
        kp = self.kernelParam
        sizeA = 2*kp.BLOCK_SIZE_M*kp.BLOCK_SIZE_K*sizeof(kp.dtype('A'))
        sizeB = 2*kp.BLOCK_SIZE_N*kp.BLOCK_SIZE_K*sizeof(kp.dtype('B'))
        return sizeA + sizeB
    
    def numCTA(self) : 
        ret = 1
        m = self.kernelParam.M
        n = self.kernelParam.N
        k = self.kernelParam.K
        for dim in self.gridDims(m,n,k):
            ret *= dim
        return ret
    
# 用户输入：hsacopath，kernel名字(通过amdgcn获取)，
class CompiledKernelFactory :
    @staticmethod
    def getKernel(info : UserInputs) -> CompiledKernel:
        if info.operatorKind==EnumOperator.Matmul :
            signature = getMatmulSignature(info.kernelParam.dtypeTorch('A'),info.kernelParam.dtypeTorch('B'))
            return CompiledKernel(
                info.hsacoPath,
                info.kernelFuncName,
                info.sharedMem(),
                signature,
                info.gridDims(),
                info.blockDims()
            )
        if info.operatorKind==EnumOperator.Convolution :
            return None
        if info.operatorKind==EnumOperator.Poll:
            return None