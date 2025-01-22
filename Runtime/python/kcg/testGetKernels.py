if __name__ == '__main__' :
    from KCGTask import *
    import multiprocessing 
    from ConfigGenerator import *

    json_path = '/home/xushilong/CodeGenDemo/cfg_cominations.json'
    config_gen(json_path)
    tm =  ParallelCompileTaskManager(json_path)
    tm.run(maxProcess=10, json_cfgs_limit=30)
    res = tm.getResults()
    print(type(res))
    PerfTester.runPerfTests(res)
