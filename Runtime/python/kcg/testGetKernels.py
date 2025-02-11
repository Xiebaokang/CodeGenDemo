if __name__ == '__main__' :
    from KCGTask import *
    import multiprocessing 
    from ConfigGenerator import *
    PathManager.init(clearPkl=True, clearTmp=True, clearCache=True)
    # Tuning 参数空间配置文件
    # tuning_param_file = '/home/xushilong/CodeGenDemo/TuningConfigs/GEMM_configs.json'
    tuning_param_file = '/home/xushilong/CodeGenDemo/TuningConfigs/GEMM_cfgs_1.json'
    # 是否使用已生成好的 combination 文件
    preGeneratedCombinations = '/home/xushilong/CodeGenDemo/cfg_cominations_2.json'
    # [for debug use] 是否从已生成的 subjson开始处理。
    startFromSubJson = '/home/xushilong/CodeGenDemo/_tmp/tmp_json_33400_33600.json'
    # perf文件路径
    perfPAth = '/home/xushilong/CodeGenDemo/perfRecordlog_4'
    
    '''
        文件生成关系：
        tuning_param_file -> cfg_combinations.json -> 拆分为 subjson, 存到tmp/中
        tuning_param_file 定义调优空间
        cfg_combinations.json 是所有参数的组合
        subjson 是 cfg_combinations.json 拆成的一堆小文件
    '''
    # DeviceInfo.set_current_device(7)
    print(f'===== Set current device to {DeviceInfo.get_current_device()} =======')
    print('==== waiting for config_gen ==== ')
    tempfileNames,totalLen = config_gen(tuning_param_file, preGeneratedJson= '')
    print('==== config_gen Done! Start deal ==== ')
    tm =  ParallelTaskManager(totalLen, tempfileNames, perfPAth, benchmarkcnt=7, warmupcnt=2, devId=DeviceInfo.get_current_device(), keepTopNum=5)
    tm.run(maxProcess=40, startFromSubjson = '', needCompile=True, needPerfTest=True)

