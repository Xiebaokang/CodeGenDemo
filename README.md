# KCG 项目介绍
## 1 项目结构

python前端 + MLIR后端
目前只简单地封装了调用核函数的逻辑，可实现用户在python端传入config，之后进行kenrel生成并执行
缓存系统还没做。因此目前会生成很多文件

MLIR后端入口点：test/test.cc  提供了lib和exe两种编译选项。在CMakeLists.txt中设置ON或OFF，或者：
```sh
cmake .. -DCOMPILE_AS_PYMODULE=ON
```

## 2.构建&运行方法
### 2.1 构建
新建build文件夹并进入。之后：
```sh
cmake .. -DCOMPILE_AS_PYMODULE=ON
make -j4
# 直接cmake .. 也可，会使用CMakeCache.txt中的配置
```

当需要切换构建选项时，需要删除build 目录下的CMakeCache.txt 否则会不生效

### 2.2 参数配置&运行
1. exe模式   
参数配置：debug用，只能用固定参数配置，在main函数中修改。只用于测试MLIR后端的lowering过程，不进行gemm的正确性测试
运行：
```sh
${project_folder}/bin/kcg_compiler > log.txt 2>&1
```
   
调试：f5进入调式模式。配置文件在 .vscode/launch.json 注意配置选择   
<p align = 'center'>
<img src="./doc/image.png" width=50%>
</p>

2. lib模式   
入口点为 Runtime/python/kcg/testGetKernels.py
config参数由用户输入的范围自动生成

testGetKernels.py 使用说明：   
```python
if __name__ == '__main__' :
    from KCGTask import *
    import multiprocessing 
    from ConfigGenerator import *
    PathManager.init()  # 路径管理器初始化。会在项目目录下建立 _cache,_dump,_pkls,_override 文件夹存放中间结果
    json_path = '/home/xushilong/CodeGenDemo/cfg_cominations.json'  # 调优空间config json文件 （可以不存在，此时通过config_gen函数生成；也能指定一个已有的，此时不应再调用 config_gen）
    perfPAth = '/home/xushilong/CodeGenDemo/perfRecord_2.log'  # perf结果的存放位置，会自动创建
    config_gen(json_path)   # 生成调优空间config json文件。
    tm =  ParallelTaskManager(json_path,perfPAth)   # 创建并行任务
    tm.run(maxProcess=30,st = 780, json_cfgs_limit=-1,needCompile=True,needPerfTest=True)  # 启动。
    #  maxProcess ： 用于compile的最大进程数
    #  st ：从index为几的config开始处理。默认0
    #  json_cfgs_limit ：最大处理数量，-1代表处理st之后的全部config
    #  needCompile ：是否运行kernel生成   needPerfTest ： 是否运行perf测试   这两个参数可实现只执行perftest或只生成kernel
    #  

```

运行：
```sh
conda activate triton_rocm
cd ${project_folder}/runtime/python 
export PYTHONPATH=`pwd`
cd ./kcg
python testGetKernels.py > log.log 2>&1

```

