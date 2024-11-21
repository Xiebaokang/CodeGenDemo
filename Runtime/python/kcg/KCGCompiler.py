
class KCGCompiler :
    def __init__(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("KCGCompiler", "/home/pangyunfei/xushilong/CodeGenDemo/bin/libkcg_compiler.so")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.compile_kernel = mod.compile_kernel
    
    