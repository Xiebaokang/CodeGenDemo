if __name__ == '__main__' :
    from KCGTask import *
    import multiprocessing 
    from ConfigGenerator import *
    PathManager.init()
    tuning_cfg_file = '/home/xushilong/CodeGenDemo/TuningConfigs/GEMM_configs.json'
    out_json_path = '/home/xushilong/CodeGenDemo/cfg_cominations.json'
    perfPAth = '/home/xushilong/CodeGenDemo/perfRecordlog_1'
    print('==== waiting for config_gen ==== ')
    config_gen(tuning_cfg_file, out_json_path)
    tm =  ParallelTaskManager(out_json_path,perfPAth,benchmarkcnt=7,warmupcnt=3,devId='0')
    tm.run(maxProcess=40,st = 585, json_cfgs_limit=-1,needCompile=True,needPerfTest=True)

