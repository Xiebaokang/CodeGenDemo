
from kcg.Utils import PathManager
from Operators.matmul import *


class KCGCompiler :
    def __init__(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("KCGCompiler", PathManager.kcg_compiler_path())
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.__compile_kernel_matmul = mod.compile_kernel_matmul
    
    def compileKernel(self, json_path) :
        return self.__compile_kernel_matmul(json_path)
    