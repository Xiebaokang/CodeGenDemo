# from Runtime.python.kcg.Loader import driver
# if __name__ == "__main__":
#     print("hello")
#     print(driver)
#     print(driver.loader)

import logging
from typing import List,Type,Tuple
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
from logging import *
from typing import List, Tuple
import glob
import ctypes

# import pymlir

best_perf = []

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
        return f"[correct={self.isCorrect}, acc={self.acc}, torchMs={self.torch_elapseTimeMs}], config=\n{self.kpm}"

class PerfTester :
    _a = None
    _b = None
    torch_eps = 0
    D = None

    @staticmethod
    def _compare_with_error(tensor1, tensor2, abs_error=1e-2, rel_error=1e-2):
        abs_diff = torch.abs(tensor1 - tensor2)
        rel_diff = abs_diff / (torch.abs(tensor1) + 1e-5)  # 避免除以零的情况
        # 比较绝对误差和相对误差
        error_mask = (abs_diff > abs_error) & (rel_diff > rel_error)
        diff_elements = torch.sum(error_mask).item()
        max_error = torch.max(torch.abs(tensor1 - tensor2))
        return diff_elements, max_error
    
    @staticmethod
    def _init_AB(kpm:KernelArgMatmul, inConfig : UserInputs,deviceId) :
        PerfTester._a = torch.randn(kpm.M,kpm.K,dtype=inConfig.kernelParam.dtypeTorch('A'),device=f'cuda:{deviceId}')
        PerfTester._b = torch.randn(kpm.K,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('B'),device=f'cuda:{deviceId}')
    
    @staticmethod
    def _inner_test_torch(a,b,ev_start : torch.cuda.Event, ev_end : torch.cuda.Event) :
        ev_start.record()
        d = torch.matmul(a,b)
        ev_end.record()
        torch.cuda.synchronize()
        eps = ev_start.elapsed_time(ev_end)
        return d,eps

    @staticmethod
    def _inner_test_kcg(a : torch.tensor, b : torch.tensor, c : torch.tensor, packedKernel : CompiledKernel,start_event : torch.cuda.Event, end_event : torch.cuda.Event) :
        start_event.record()
        packedKernel.run(a,b,c)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        return c,elapsed_time
    
    @staticmethod
    def _test_perf(kpm:KernelArgMatmul, inConfig : UserInputs, packedKernel : CompiledKernel, benchmarkCount = 5, warmupCount = 1, deviceId = '0') -> KernelTestResult:
        if PerfTester._a is None or PerfTester._b is None :
            PerfTester._init_AB(kpm,inConfig,deviceId)
        result = KernelTestResult(kpm)
        a = PerfTester._a
        b = PerfTester._b
        c = torch.empty(kpm.M,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('C'),device=f'cuda:{deviceId}')
        d = torch.empty(kpm.M,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('C'),device=f'cuda:{deviceId}')
        atrans = torch.transpose(a,1,0).contiguous()  # 转置会令底层存储不连续，导致失败。必须使其连续
        assert(a.is_contiguous())
        assert(b.is_contiguous())
        assert(atrans.is_contiguous())
        # print(inConfig.kernelParam.dtypeTorch('A'))
        # print(inConfig.kernelParam.dtypeTorch('B'))
        # print(inConfig.kernelParam.dtypeTorch('C'))
        M, K = a.shape
        K, N = b.shape
        print(f"is_contiguous: a {a.is_contiguous()}, b {b.is_contiguous()}, aT {atrans.is_contiguous()}")
        res = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event_torch = torch.cuda.Event(enable_timing=True)
        end_event_torch = torch.cuda.Event(enable_timing=True)
        a_ = None
        
        if kpm.isATranspose :
            print('transpose test')
            a_ = atrans
        else:
            print('normal test')
            a_ = a
        for i in range(0,warmupCount) : # warmup
            torch.matmul(a,b)
            packedKernel.run(a_,b,c)
            
        # 计算torch的eps
        if PerfTester.torch_eps <= 0:
            eps_torch_list = []
            for i in range(0,51) :
                d,eps_torch = PerfTester._inner_test_torch(a,b,start_event_torch,end_event_torch)
                eps_torch_list.append(eps_torch)
            PerfTester.torch_eps = np.median(eps_torch_list)
            PerfTester.D = d
        
        # 测试
        for i in range(0,benchmarkCount) : 
            c,eps = PerfTester._inner_test_kcg(a_,b,c,packedKernel,start_event,end_event)
            res.append(eps)
            time.sleep(0.01)
        print("c=",c)

        if torch.allclose(c,PerfTester.D, atol=1e-1, rtol=1e-1):
            print('test correct!')
            result.isCorrect = True
            result.torch_elapseTimeMs = PerfTester.torch_eps
            result.kcg_elapseTimeMs = np.median(res)
            print(f"codegenDemo median time: {result.kcg_elapseTimeMs} ms")
            print(f"rocblas median time: {result.torch_elapseTimeMs} ms")
            result.acc = result.torch_elapseTimeMs/result.kcg_elapseTimeMs
            print(f"speed up: {result.acc}")
        else:
            result.isCorrect = False
            diff,max_error= PerfTester._compare_with_error(d,c)
            result.maxError = max_error
            result.diffRate = diff/(M*N)
            print(f'test fail! maxerror={max_error}, diff=[{diff} / {M*N}], diffrate={result.diffRate}')
            # hipprof --pmc python ./test.py 
        return result

    
    @staticmethod
    def _getBestPerf(perfData : List[KernelTestResult], topNum = 1) -> KernelTestResult:
        global best_perf
        for d in perfData:
            if d.isCorrect :
                best_perf.append(d)
                best_perf.sort(key=lambda x: x.acc, reverse=True)
                if len(best_perf) > topNum :
                    best_perf = best_perf[0:topNum]
        return best_perf
      
    @staticmethod
    def runPerfTests(lock, endsignal ,outputPAth = None, benchmarkCount = 5, warmupCount = 1, deviceId = '0', topNum = 6) : 
        # collect kernels from pkl         
        valid_kernels = [] # List[Tuple[KernelArgMatmul,UserInputs,CompiledKernel]]
        total_kernel_count = 0
        while True:
            lock.acquire()
            pklFiles = glob.glob(PathManager().pikle_dir() + '/*.pkl')
            lock.release()
            if len(pklFiles) <= 0 :
                if endsignal.value > 0:
                    break
                else:
                    time.sleep(5)
            # 输出所有找到的文件路径
            for pkl in pklFiles:
                arr = deserialize_from_file(pkl)
                valid_kernels += arr
            total_kernel_count += len(valid_kernels)
            print(f"====== Glob .pkl files : {len(pklFiles)}, Valid Kernels : {len(valid_kernels)} ========")
            for pkl in pklFiles:
                try:
                    os.remove(pkl)
                    print(f"deleted: {pkl}")
                except Exception as e:
                    print(f"exception occur when delete {pkl}: {e}")

            for t in valid_kernels :
                print(f"{t[1].hsacoPath},  {t[1].kernelFuncName}")
            perf_data = []
            # serialize test perf
            for (kpm,inConfig,packedKernel) in valid_kernels :
                perf_data.append(PerfTester._test_perf(kpm, inConfig, packedKernel, benchmarkCount, warmupCount , deviceId))        
            valid_kernels.clear()
            bests = PerfTester._getBestPerf(perf_data, topNum)
            if bests is not None and outputPAth is not None :
                with open(outputPAth,mode='w') as f:
                    for i in range(0,len(bests)) :
                        f.write(f"[{i}] {str(bests[i])} \n-------\n")
        print(f"=====[ PerfTest Finished ] =======\n   - Best : {best_perf} ")
    

class SerialCompileTask :
    def _task_compile_kernel(self,kpm : KernelArgMatmul, index:int, device:int) -> Tuple[KernelArgMatmul,UserInputs,CompiledKernel] :
        Print = print
        # compile kernel
        # Print("===== KCGCompiler ctor ========")
        kernelCompiler = KCGCompiler()
        # Print("===== call compileKernel(kpm)[0] ========")
        hsacoPath,kernelName,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ = kernelCompiler.compileKernel(kpm)[0]
        # Print("===== test info ========")
        # Print("hsacoPath = ", hsacoPath)
        # Print("kernelName = ", kernelName)
        # Print("gridDimX = ", gridDimX)
        # Print("gridDimY = ", gridDimY)
        # Print("gridDimZ = ", gridDimZ)
        # Print("blockDimX = ", blockDimX)
        # Print("blockDimY = ", blockDimY)
        # Print("blockDimZ = ", blockDimZ)    
        Print("========= hsacoPath = ",hsacoPath)
        Print("========= kernelName = ",kernelName)
        inConfig = UserInputs(hsacoPath,kernelName,kpm)
        inConfig.m_gridDims = [gridDimX,gridDimY,gridDimZ]
        inConfig.m_blockDims = [blockDimX,blockDimY,blockDimZ]
        inConfig.operatorKind = EnumOperator.Matmul
        packedKernel = CompiledKernelFactory.getKernel(inConfig,device)
        return (kpm,inConfig,packedKernel)  # 
  
    def setup_logger(logfile) -> logging.Logger:
        logging.basicConfig(filename=logfile, filemode='w+', level=logging.INFO)
        logger = logging.getLogger(logfile)
        return logger
    
    def compile_kernels(self, lock, kernelArgs : List[KernelArgMatmul],lbs=0,ubs=-1,namePrefix='',device=0) -> List:
        # 读取 JSON 文件
        output_path = PathManager.pikle_dir() + f'/valid_kernels_{namePrefix}_{lbs}_{ubs}.pkl'
        valid_kernels = [] 
        if ubs < 0:
            lbs = 0; ubs = len(kernelArgs) 
        for i in range(lbs,ubs) :
            kernelCfg = kernelArgs[i]
            r = self._task_compile_kernel(kernelCfg,i,device)   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去    
            valid_kernels.append(r)
        lock.acquire()
        serialize_to_file(output_path,valid_kernels)
        lock.release()
        return valid_kernels


class ParallelTaskManager :
    def __init__(self, total_cfg_count ,json_path_arr : List[str], perf_out_path : str, benchmarkcnt = 5, warmupcnt = 1, devId=0, keepTopNum = 1 ):
        ctx = multiprocessing.get_context('spawn')
        self.Process = ctx.Process
        self.lock = ctx.Lock()
        self.subProcList = []
        self.cfg_json_path_arr = json_path_arr
        self.CFG_COUNT = total_cfg_count
        self.task_groups = []
        self.m_totalKernels = []
        self.endSignal = ctx.Manager().Value(ctypes.c_int,0)
        self.perfTestFinalId = ctx.Manager().Value(ctypes.c_int,0)
        self.perf_out_path = perf_out_path
        self.perfProcMonitor = self.Process(target=self._perfMonitorFunc,args=())  # 创建perfTest守护进程。当perftest进程意外挂掉，由守护进程重启之
        self.nBenchMark = benchmarkcnt
        self.nWarmup = warmupcnt
        self.devId = devId
        self.topNum = keepTopNum
    
    def _perfMonitorFunc(self) :
        id = 0
        self.perfProc = self.Process(target=PerfTester.runPerfTests, args=(self.lock,self.endSignal,self.perf_out_path + str(id),self.nBenchMark, self.nWarmup, str(self.devId), self.topNum))
        self.perfProc.start()
        while True:
            self.perfProc.join()
            if self.endSignal.value == 1 :  # 进程收到结束信号正常结束
                return
            else:
                id += 1
                self.perfProc = self.Process(target=PerfTester.runPerfTests, args=(self.lock,self.endSignal, f"{self.perf_out_path}_{str(id)}.log"))
                self.perfProc.start()
    
    def _createSubProc(self,func,*params) :
        p = self.Process(target = func, args = (*params,))
        p.start()
        self.subProcList.append(p)
    
    def _waitAll(self) :
        for s in self.subProcList :
            s.join()
        self.subProcList.clear()
    
    ## 从json文件里读取 cfgs，转化为 List[KernelArgMatmul] 
    def _get_kernelargMatmul(self, json_path : str) -> List[KernelArgMatmul] : 
        import json
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        cfgs = json_data['cfgs']
        kernelArgs = []
        kw = ConfigKeywords
        for i in range(0,len(cfgs)) :
            config = cfgs[i]
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
            arg.UNROLL_NUM = config[kw.KEY_UNROLL_NUM]
            arg.REG_PREFETCH = config[kw.KEY_REG_PREFETCH]
            arg.SHARED_PREFETCH = config[kw.KEY_SHARED_PREFETCH]
            arg.LOAD_CONTINUOUS = config[kw.KEY_LOAD_CONTINUOUS]
            arg.REDUCE_C_CONTINUOUS = config[kw.KEY_REDUCE_C_CONTINUOUS]
            
            kernelArgs.append(arg)
            
        return kernelArgs
    
    def run(self, maxProcess = 10, startFromSubjson = '', needCompile = True, needPerfTest = True) :
        isStartAtSubjson = False
        if len(startFromSubjson) <= 0:
            print(f"====== start from cfg at beggining =========")
        else :
            print(f"====== start from subjson {startFromSubjson} =========")
            isStartAtSubjson = True
            
        procCount = 0
        dealed = 0
        if needPerfTest:
            self.perfProcMonitor.start()
        if needCompile :
            for jsonPath in self.cfg_json_path_arr :
                if isStartAtSubjson :
                    if jsonPath != startFromSubjson :
                        continue
                print(f"=========== Dealing json : {jsonPath} ================")
                kernelConfigs = self._get_kernelargMatmul(jsonPath)
                words = jsonPath.split('/')
                namePrefix = words[-1].split('.')[0]
                for i in range(0,len(kernelConfigs)) :
                    sct = SerialCompileTask()
                    self._createSubProc(sct.compile_kernels,self.lock,kernelConfigs,i,i+1,namePrefix,self.devId)
                    procCount += 1; dealed += 1
                    if procCount >= maxProcess or i == self.CFG_COUNT-1:
                        print(f"========= Wating for Compile tasks [{dealed}/{self.CFG_COUNT}]  ============")
                        self._waitAll()
                        procCount = 0
            print(f"========= All Compile tasks Finished [{self.CFG_COUNT}] ! ============")
        self.endSignal.value = 1
        
        if needPerfTest :
            self.perfProcMonitor.join()
