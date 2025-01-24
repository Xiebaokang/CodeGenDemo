if __name__ == '__main__' :
    from KCGTask import *
    import multiprocessing 
    from ConfigGenerator import *
    PathManager.init()
    json_path = '/home/xushilong/CodeGenDemo/cfg_cominations.json'
    perfPAth = '/home/xushilong/CodeGenDemo/perfRecord_2.log'
    config_gen(json_path)
    tm =  ParallelTaskManager(json_path,perfPAth)
    tm.run(maxProcess=30,st = 780, json_cfgs_limit=-1,needCompile=True,needPerfTest=True)

