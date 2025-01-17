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
import numpy as np
from kcg.KCGCompiler import KCGCompiler
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
# import pymlir




class KernelTestResult :
    def __init__(self,kpm : KernelArgMatmul):
        self.kpm = kpm
        self.isCorrect = False
        self.acc = 0.0
        self.kcg_elapseTimeMs = 0.0
        self.torch_elapseTimeMs = 0.0
        self.diffRate = 0.0
        self.maxError = 0.0
    def __str__(self):
        return f"[{self.isCorrect},{self.acc}]"


class KernelTaskGroup :
    def __init__(self,json_path):
        self.cfg_json_path = json_path
        self.perf_results = []
    def _compare_with_error(self,tensor1, tensor2, abs_error=1e-2, rel_error=1e-2):
        abs_diff = torch.abs(tensor1 - tensor2)
        rel_diff = abs_diff / (torch.abs(tensor1) + 1e-5)  # 避免除以零的情况
        # 比较绝对误差和相对误差
        error_mask = (abs_diff > abs_error) & (rel_diff > rel_error)
        diff_elements = torch.sum(error_mask).item()
        max_error = torch.max(torch.abs(tensor1 - tensor2))
        return diff_elements, max_error
    
    def _task_compile_kernel(self,kpm : KernelArgMatmul) : 
        # compile kernel
        kernelCompiler = KCGCompiler()
        hsacoPath,kernelName,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ = kernelCompiler.compileKernel(kpm)[0]
        print("===== test info ========")
        print("hsacoPath = ", hsacoPath)
        print("kernelName = ", kernelName)
        print("gridDimX = ", gridDimX)
        print("gridDimY = ", gridDimY)
        print("gridDimZ = ", gridDimZ)
        print("blockDimX = ", blockDimX)
        print("blockDimY = ", blockDimY)
        print("blockDimZ = ", blockDimZ)    
        print("========= hsacoPath = ",hsacoPath)
        print("========= kernelName = ",kernelName)
        inConfig = UserInputs(hsacoPath,kernelName,kpm)
        inConfig.m_gridDims = [gridDimX,gridDimY,gridDimZ]
        inConfig.m_blockDims = [blockDimX,blockDimY,blockDimZ]
        inConfig.operatorKind = EnumOperator.Matmul
        packedKernel = CompiledKernelFactory.getKernel(inConfig)
        return (kpm,inConfig,packedKernel)
        
    def test_perf(self, kpm, inConfig : UserInputs, packedKernel : CompiledKernel):
        result = KernelTestResult(kpm)
        a = torch.randn(kpm.M,kpm.K,dtype=inConfig.kernelParam.dtypeTorch('A'),device='cuda:0')
        b = torch.randn(kpm.K,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('B'),device='cuda:0')
        c = torch.empty(kpm.M,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('C'),device='cuda:0')
        d = torch.empty(kpm.M,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('C'),device='cuda:0')
        atrans = torch.transpose(a,1,0).contiguous()  # 转置会令底层存储不连续，导致失败。必须使其连续
        assert(a.is_contiguous())
        assert(b.is_contiguous())
        assert(atrans.is_contiguous())
        print(inConfig.kernelParam.dtypeTorch('A'))
        print(inConfig.kernelParam.dtypeTorch('B'))
        print(inConfig.kernelParam.dtypeTorch('C'))
        M, K = a.shape
        K, N = b.shape
        print(f"python: M,N,K = {M},{N},{K}")
        print("conti: a",a.is_contiguous())
        print("conti: b",b.is_contiguous())
        print("conti: atrans",atrans.is_contiguous())
        res = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        benchmarkCount = 5
        warmupCount = 1

        if kpm.isATranspose :
            print('transpose test')
            for i in range(warmupCount):
                packedKernel.run(atrans,b,c) # warm up
            for i in range(0,benchmarkCount) : # benchmark
                start_event.record()
                packedKernel.run(atrans,b,c)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
                res.append(elapsed_time)
        else:
            print('normal test')
            # packedKernel.run(a,b,c) # warm up
            for i in range(0,benchmarkCount) : # benchmark
                start_event.record()
                packedKernel.run(a,b,c)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
                res.append(elapsed_time)

        result.kcg_elapseTimeMs = np.median(res)
        
        print(f"codegenDemo median time: {result.kcg_elapseTimeMs} ms")
        # o.run(a,b,c,
        #       M,N,K, a.stride(0), a.stride(1),  
        #         b.stride(0), b.stride(1),  
        #         c.stride(0), c.stride(1),  
        #     )
        res1 = []
        # torch.matmul(a,b) # warm up
        for i in range(0,benchmarkCount) : # benchmark
            start_event.record()
            d = torch.matmul(a,b)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            res1.append(elapsed_time)
        result.torch_elapseTimeMs = np.median(res1)
        print(f"rocblas median time: {result.torch_elapseTimeMs} ms")
        result.acc = result.torch_elapseTimeMs/result.kcg_elapseTimeMs
        print(f"speed up: {result.acc}")
        print("c=",c)
        print("d=",d)
        if torch.allclose(c,d,atol=1e-2,rtol=1e-2):
            print('test correct!')
            result.isCorrect = True
        else:
            result.isCorrect = False
            diff,max_error= self._compare_with_error(d,c)
            result.maxError = max_error
            result.diffRate = diff/(M*N)
            print(f'test fail! maxerror={max_error}, diff=[{diff} / {M*N}], diffrate={result.diffRate}')
            # hipprof --pmc python ./test.py 
        return result
    
    def _get_kernelargMatmul(self) :
        import json
        with open(self.cfg_json_path, 'r') as file:
            json_data = json.load(file)
        cfgs = json_data['cfgs']
        kernelArgs = []
        kw = ConfigKeywords
        for config in cfgs :
            arg = KernelArgMatmul(config[kw.KEY_M],config[kw.KEY_N],config[kw.KEY_K], 
                                EnumKernelDType(config[kw.KEY_DTYPE_A]), 
                                EnumKernelDType(config[kw.KEY_DTYPE_B]),
                                EnumKernelDType(config[kw.KEY_DTYPE_C]))
            arg.BLOCK_SIZE_M = config[kw.KEY_BLOCK_SIZE_M]
            arg.BLOCK_SIZE_N = config[kw.KEY_BLOCK_SIZE_N]
            arg.BLOCK_SIZE_K = config[kw.KEY_BLOCK_SIZE_K]
            arg.THREAD_SIZE_M = config[kw.KEY_THREAD_SIZE_M]
            arg.THREAD_SIZE_N = config[kw.KEY_THREAD_SIZE_N]
            arg.WARP_SIZE = config[kw.KEY_WARP_SIZE]
            arg.BLOCK_LAYOUT_M = config[kw.KEY_BLOCK_LAYOUT_M]
            arg.BLOCK_LAYOUT_N = config[kw.KEY_BLOCK_LAYOUT_N]
            arg.WARP_LAYOUT_M = config[kw.KEY_WARP_LAYOUT_M]
            arg.WARP_LAYOUT_N = config[kw.KEY_WARP_LAYOUT_N]
            arg.isATranspose = config[kw.KEY_IS_A_TRANSPOSE]
            arg.GLOB_LOAD_WIDTH_A = config[kw.KEY_GLOB_LOAD_WIDTH_A]
            arg.GLOB_LOAD_WIDTH_B = config[kw.KEY_GLOB_LOAD_WIDTH_B]
            arg.WARP_SCATTER_WIDTH_A = config[kw.KEY_WARP_SCATTER_WIDTH_A]
            arg.WARP_SCATTER_WIDTH_B = config[kw.KEY_WARP_SCATTER_WIDTH_B]
            arg.THREAD_SCATTER_WIDTH_A = config[kw.KEY_THREAD_SCATTER_WIDTH_A]
            arg.THREAD_SCATTER_WIDTH_B = config[kw.KEY_THREAD_SCATTER_WIDTH_B]
            arg.LOCAL_SPLIT_U = config[kw.KEY_LOCAL_SPLIT_U]
            arg.BLOCK_MAPPING = config[kw.KEY_BLOCK_MAPPING]
            arg.GLOB_STORE_WIDTH = config[kw.KEY_GLOB_STORE_WIDTH]
            kernelArgs.append(arg)
        return kernelArgs
    
    def _getBestPerf(self, perfData : List[KernelTestResult]) -> KernelTestResult:
        best = None
        for d in perfData:
            if best is None:
                best = d
            else:
                if d.isCorrect and d.acc > best.acc:
                    best = d
        return best
    
    def run(self, limit = 0):
        # 读取 JSON 文件
        kernelArgs = self._get_kernelargMatmul()
        processResult = [] 
        valid_kernels = []
        perf_data = []
        _gProcessPool = ProcessPoolExecutor(10)
        if limit <= 0:
            limit = len(kernelArgs)
            
        print("====== compiled task start =========")
        for i in range(limit) :
            kernelCfg = kernelArgs[i]
            # r = _gProcessPool.submit(self._task_compile_kernel,kernelCfg)   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去    
            r = self._task_compile_kernel(kernelCfg)   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去    
            processResult.append(r)
            # (kpm,inConfig,packedKernel)
        # _gProcessPool.shutdown()   # 如果手动shutdown， attexit 会报错

        for i in range(limit) :
            try:
                valid_kernels.append(processResult[i].result(timeout=20))
            except Exception as e:
                pass
        print("====== compiled task done =========")
        # serialize test perf
        for (kpm,inConfig,packedKernel) in valid_kernels :
            perf_data.append(self.test_perf(kpm,inConfig,packedKernel))             
        best_perf = self._getBestPerf(perf_data)
        if best_perf is not None:
            print(best_perf)
        else:
            print('all process result is invalid. Please retry with new json')
