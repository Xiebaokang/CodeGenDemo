if __name__ == '__main__' :
    from KCGTask import *
    import multiprocessing 
    from ConfigGenerator import *
    PathManager.init()
    tuning_param_file = '/home/xushilong/CodeGenDemo/TuningConfigs/GEMM_configs.json'
    temp_json_dir = '/home/xushilong/CodeGenDemo/_tmp/'
    preGeneratedCombinations = '/home/xushilong/CodeGenDemo/cfg_cominations_1.json'

    perfPAth = '/home/xushilong/CodeGenDemo/perfRecordlog_3'
    print('==== waiting for config_gen ==== ')
    tempfileNames,totalLen = config_gen(tuning_param_file, temp_json_dir, preGeneratedCombinations)
    print('==== config_gen Done! Start deal ==== ')
    tm =  ParallelTaskManager(totalLen, tempfileNames, perfPAth, benchmarkcnt=7, warmupcnt=5, devId='0', keepTopNum=5)
    tm.run(maxProcess=40,st = 0,needCompile=True,needPerfTest=True)

