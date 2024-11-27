
from kcg.Utils import PathManager,KernelArgMatmul


class KCGCompiler :
    def __init__(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("KCGCompiler", PathManager.kcg_compiler_path())
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.__compile_kernel_matmul = mod.compile_kernel_matmul
    
    def compileKernel(self, param : KernelArgMatmul) :
        return self.__compile_kernel_matmul(
            param.BLOCK_SIZE_M,
            param.BLOCK_SIZE_N,
            param.BLOCK_SIZE_K,
            param.THREAD_SIZE_M,
            param.THREAD_SIZE_N,
            param.VECTORIZE_WIDTH,
            param.WARP_SIZE,
            param.BLOCK_LAYOUT_M,
            param.BLOCK_LAYOUT_N,
            param.WARP_LAYOUT_M,
            param.WARP_LAYOUT_N,
            param.dtype('A'),
            param.dtype('B'),
            param.dtype('C'),
            param.M,param.N,param.K
        )
    