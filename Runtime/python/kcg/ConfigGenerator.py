import itertools
import json
from Utils import *

def read_params(userInputJsonPath : str) :
    with open(userInputJsonPath, 'r') as file:
        json_data = json.load(file)
        return json_data
    return None
    
def check_shm_size(config : Dict) :
    # 计算约束条件
    dtypeC = config[ ConfigKeywords.KEY_DTYPE_C]
    dtypeBytes = sizeof(get_dtype_from_int(dtypeC))
    value1 = (config[ ConfigKeywords.KEY_BLOCK_SIZE_M] + config[ ConfigKeywords.KEY_BLOCK_SIZE_N]) * config[ ConfigKeywords.KEY_BLOCK_SIZE_K] * dtypeBytes * config[ ConfigKeywords.KEY_LOCAL_SPLIT_U]
    value2 = config[ ConfigKeywords.KEY_BLOCK_SIZE_M] * config[ ConfigKeywords.KEY_BLOCK_SIZE_N] * dtypeBytes * config[ ConfigKeywords.KEY_LOCAL_SPLIT_U]

    if max(value1, value2) < 16384 :
        # 展开 KEY_DTYPE 到 KEY_DTYPE_A, KEY_DTYPE_B, KEY_DTYPE_C
        dtype_value = config.pop( ConfigKeywords.KEY_DTYPE_C)
        config[ ConfigKeywords.KEY_DTYPE_A] = dtype_value
        config[ ConfigKeywords.KEY_DTYPE_B] = dtype_value
        config[ ConfigKeywords.KEY_DTYPE_C] = dtype_value
        return True
    return False

def check_warp(config : Dict) :
    wlm = config[ConfigKeywords.KEY_WARP_LAYOUT_M]
    wln = config[ConfigKeywords.KEY_WARP_LAYOUT_N]
    warpsz = config[ConfigKeywords.KEY_WARP_SIZE]
    if wlm * wln == warpsz :
        return True
    return False

def check_size(config : Dict) :
    bm = config[ConfigKeywords.KEY_BLOCK_SIZE_M]
    bn = config[ConfigKeywords.KEY_BLOCK_SIZE_N]
    bk = config[ConfigKeywords.KEY_BLOCK_SIZE_K]
    tm = config[ConfigKeywords.KEY_THREAD_SIZE_M]
    tn = config[ConfigKeywords.KEY_THREAD_SIZE_N]
    m = config[ConfigKeywords.KEY_M]
    n = config[ConfigKeywords.KEY_N]
    k = config[ConfigKeywords.KEY_K]
    
    blm = config[ConfigKeywords.KEY_BLOCK_LAYOUT_M]
    bln = config[ConfigKeywords.KEY_BLOCK_LAYOUT_N]
    
    wlm = config[ConfigKeywords.KEY_WARP_LAYOUT_M]
    wln = config[ConfigKeywords.KEY_WARP_LAYOUT_N]
    
    blockDim_m = blm * wlm
    blockDim_n = bln * wln
    if blockDim_m * tm != bm or blockDim_n * tn != bn :
        return False
    
    wswa = config[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_A]
    wswb = config[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_B]
    tswa = config[ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_A]
    tswb = config[ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_B]
    if wswa < tswa or wswb < tswb :
        return False
    if wswa % tswa != 0 or wswb % tswb != 0 :
        return False
    return True
    
def get_cfgs(userTuningCfgPath = None) :
    # 输入参数及其可选值
    # param_options = {
    #     ConfigKeywords.KEY_BLOCK_SIZE_M : [32, 48, 64, 128],
    #     ConfigKeywords.KEY_BLOCK_SIZE_N : [32, 48, 64, 128],
    #     ConfigKeywords.KEY_BLOCK_SIZE_K : [16, 32, 64],
    #     ConfigKeywords.KEY_THREAD_SIZE_M : [4, 8],
    #     ConfigKeywords.KEY_THREAD_SIZE_N : [4, 8],
    #     ConfigKeywords.KEY_GLOB_LOAD_WIDTH_A : [2],
    #     ConfigKeywords.KEY_GLOB_LOAD_WIDTH_B : [2],
    #     ConfigKeywords.KEY_BLOCK_LAYOUT_M : [2],
    #     ConfigKeywords.KEY_BLOCK_LAYOUT_N : [2],
    #     ConfigKeywords.KEY_WARP_LAYOUT_M : [8],
    #     ConfigKeywords.KEY_WARP_LAYOUT_N : [8],
    #     ConfigKeywords.KEY_WARP_SCATTER_WIDTH_A : [2, 4],
    #     ConfigKeywords.KEY_WARP_SCATTER_WIDTH_B : [2, 4],
    #     ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_A : [1, 2, 4],
    #     ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_B : [1, 2, 4],
    #     ConfigKeywords.KEY_LOCAL_SPLIT_U : [1, 2],
    #     ConfigKeywords.KEY_BLOCK_MAPPING : [8, 16],
    #     ConfigKeywords.KEY_WARP_SIZE : [64],
    #     ConfigKeywords.KEY_GLOB_STORE_WIDTH : [2, 4],
    #     ConfigKeywords.KEY_DTYPE_A : [int(EnumKernelDType.float32)],
    #     ConfigKeywords.KEY_DTYPE_B : [int(EnumKernelDType.float32)],
    #     ConfigKeywords.KEY_DTYPE_C : [int(EnumKernelDType.float32)],
    #     ConfigKeywords.KEY_M : [1024],
    #     ConfigKeywords.KEY_N : [1024],
    #     ConfigKeywords.KEY_K : [1024],
    #     ConfigKeywords.KEY_IS_A_TRANSPOSE : [1],
    #     ConfigKeywords.KEY_UNROLL_NUM : [4,8,16],
    #     ConfigKeywords.KEY_REG_PREFETCH :         [1],
    #     ConfigKeywords.KEY_SHARED_PREFETCH :      [1],
    #     ConfigKeywords.KEY_LOAD_CONTINUOUS :      [1],
    #     ConfigKeywords.KEY_REDUCE_C_CONTINUOUS :  [1]
    # }
    
    param_options = {
        ConfigKeywords.KEY_BLOCK_SIZE_M : [32,64],
        ConfigKeywords.KEY_BLOCK_SIZE_N : [32,64],
        ConfigKeywords.KEY_BLOCK_SIZE_K : [16,8],
        ConfigKeywords.KEY_THREAD_SIZE_M : [4,8],
        ConfigKeywords.KEY_THREAD_SIZE_N : [4,8],
        ConfigKeywords.KEY_GLOB_LOAD_WIDTH_A : [4],
        ConfigKeywords.KEY_GLOB_LOAD_WIDTH_B : [4],
        ConfigKeywords.KEY_BLOCK_LAYOUT_M : [1,2,4],
        ConfigKeywords.KEY_BLOCK_LAYOUT_N : [1,2,4],
        ConfigKeywords.KEY_WARP_LAYOUT_M : [8,4,16],
        ConfigKeywords.KEY_WARP_LAYOUT_N : [8,4,16],
        ConfigKeywords.KEY_WARP_SCATTER_WIDTH_A : [2, 4],
        ConfigKeywords.KEY_WARP_SCATTER_WIDTH_B : [2, 4],
        ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_A : [1, 2, 4],
        ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_B : [1, 2, 4],
        ConfigKeywords.KEY_LOCAL_SPLIT_U : [1, 2],
        ConfigKeywords.KEY_BLOCK_MAPPING : [8, 16],
        ConfigKeywords.KEY_WARP_SIZE : [64],
        ConfigKeywords.KEY_GLOB_STORE_WIDTH : [2, 4],
        ConfigKeywords.KEY_DTYPE_A : [int(EnumKernelDType.float32)],
        ConfigKeywords.KEY_DTYPE_B : [int(EnumKernelDType.float32)],
        ConfigKeywords.KEY_DTYPE_C : [int(EnumKernelDType.float32)],
        ConfigKeywords.KEY_M : [1024],
        ConfigKeywords.KEY_N : [1024],
        ConfigKeywords.KEY_K : [1024],
        ConfigKeywords.KEY_IS_A_TRANSPOSE : [1],
        ConfigKeywords.KEY_UNROLL_NUM : [4],
        ConfigKeywords.KEY_REG_PREFETCH :         [0],
        ConfigKeywords.KEY_SHARED_PREFETCH :      [0],  # =1 生成不了valid kernel
        ConfigKeywords.KEY_LOAD_CONTINUOUS :      [0],
        ConfigKeywords.KEY_REDUCE_C_CONTINUOUS :  [0]
    }
    
    if userTuningCfgPath is not None :
        print(f"======= Read user tuning configs from {userTuningCfgPath} ==========")
        param_options = read_params(userTuningCfgPath)

    # 获取所有参数名和对应的可选值
    keys = list(param_options.keys())
    values = list(param_options.values())

    # 生成所有可能的参数组合
    combinations = {"cfgs" : [] }
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        if check_warp(config) and check_shm_size(config) and check_size(config):
            combinations['cfgs'].append(config)
    return combinations
    
# 输出为 JSON 文件
def save_to_json(combinations, output_file="config_combinations.json"):
    with open(output_file, "w") as f:
        json.dump(combinations, f, indent=4)

def getTempJsonFileName(startIndex : int, endIndex : int, outDir:str) :
    return f'{outDir}/tmp_json_{str(startIndex)}_{str(endIndex)}.json'

def split_bigjson_to_temp(bigcfgs : Dict, startIndex : int, endIndex : int, outDir:str) -> str:
    if outDir[-1] == '/':
        outDir = outDir[:-1]
    tmpjson = {'cfgs' : []}
    maxLen = len(bigcfgs['cfgs'])
    for i in range(startIndex,endIndex) :
        if i >= maxLen : 
            break
        else:
            tmpjson['cfgs'].append(bigcfgs['cfgs'][i])
    filename = getTempJsonFileName(startIndex,endIndex,outDir)
    save_to_json(tmpjson, filename)
    return filename

def config_gen(tuning_cfg_file :str, preGeneratedJson : str, singleLength : int) :
    cfgs = None
    tempFileNames = []
    # if len(startSubjson) > 0:
    #     if os.path.exists(PathManager.tmp_dir()) and os.path.isdir(PathManager.tmp_dir()):
    #     # 遍历目录中的所有文件和子目录
    #         for filename in os.listdir(PathManager.tmp_dir()):
    #             file_path = os.path.join(PathManager.tmp_dir(), filename)
    #             # 如果是文件，删除它
    #             if os.path.isfile(file_path):
    #                 tempFileNames.append(file_path)
    #     return (tempFileNames,len(items))
    if len(preGeneratedJson) > 0 :
        print(f'======== Use pre-generated json combinations {preGeneratedJson} ==========')
        cfgs = read_params(preGeneratedJson)
    else:
        print(f'======== Generate combinations by {tuning_cfg_file} ==========')
        cfgs = get_cfgs(tuning_cfg_file)
    items = cfgs['cfgs']
    for i in range(0,len(items), singleLength) :
        fname = split_bigjson_to_temp(cfgs,i,i+singleLength,PathManager.tmp_dir())
        tempFileNames.append(fname)
        
    print(f"Generated {len(items)} configurations and saved subfiles to {PathManager.tmp_dir()}")
    return (tempFileNames,len(items))


# # 主函数
# if __name__ == "__main__":
#     save_to_json(combinations)
#     print(f"Generated {len(combinations)} configurations and saved to 'config_combinations.json'.")
    