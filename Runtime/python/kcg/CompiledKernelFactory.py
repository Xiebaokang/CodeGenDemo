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

class UserInputs:
    def __init__(self,hsaco_path,kernel_func_name):
        # 自定义dtype的代表含义。如：matmul使用0、1分别代表A和B的元素类型
        self.dtype_0 = torch.float32
        self.dtype_1 = torch.float32
        self.dtype_2 = torch.float32
        self.dtype_3 = torch.float32
        self.dtype_4 = torch.float32
        self.operatorKind = EnumOperator.Invalid
        self.hsacoPath = hsaco_path
        self.kernelFuncName = kernel_func_name


# 用户输入：hsacopath，kernel名字(通过amdgcn获取)，
class CompiledKernelFactory :
    @staticmethod
    def getKernel(info : UserInputs) -> CompiledKernel:
        if info.operatorKind==EnumOperator.Matmul :
            signature = getMatmulSignature(info.dtype_0,info.dtype_1)
            return CompiledKernel(
                info.hsacoPath,
                info.kernelFuncName,
                50000,
                signature
            )
        if info.operatorKind==EnumOperator.Convolution :
            return None
        if info.operatorKind==EnumOperator.Poll:
            return None