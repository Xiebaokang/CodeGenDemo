import itertools
import json
from Utils import *

# 输入参数及其可选值
param_options = {
    ConfigKeywords.KEY_BLOCK_SIZE_M : [32, 48, 64, 128],
    ConfigKeywords.KEY_BLOCK_SIZE_N : [32, 48, 64, 128],
    ConfigKeywords.KEY_BLOCK_SIZE_K : [16, 32, 64],
    ConfigKeywords.KEY_THREAD_SIZE_M : [4, 8],
    ConfigKeywords.KEY_THREAD_SIZE_N : [4, 8],
    ConfigKeywords.KEY_GLOB_LOAD_WIDTH_A : [2],
    ConfigKeywords.KEY_GLOB_LOAD_WIDTH_B : [2],
    ConfigKeywords.KEY_BLOCK_LAYOUT_M : [2],
    ConfigKeywords.KEY_BLOCK_LAYOUT_N : [2],
    ConfigKeywords.KEY_WARP_LAYOUT_M : [8],
    ConfigKeywords.KEY_WARP_LAYOUT_N : [8],
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
    ConfigKeywords.KEY_IS_A_TRANSPOSE : [1]
}

# 获取所有参数名和对应的可选值
keys = list(param_options.keys())
values = list(param_options.values())

# 生成所有可能的参数组合
combinations = {"cfgs" : [] }
for combination in itertools.product(*values):
    config = dict(zip(keys, combination))

    # 计算约束条件
    dtype_factor = 4 if config[ ConfigKeywords.KEY_DTYPE_C] == int(EnumKernelDType.float32) else 2
    value1 = (config[ ConfigKeywords.KEY_BLOCK_SIZE_M] + config[ ConfigKeywords.KEY_BLOCK_SIZE_N]) * config[ ConfigKeywords.KEY_BLOCK_SIZE_K] * dtype_factor * config[ ConfigKeywords.KEY_LOCAL_SPLIT_U]
    value2 = config[ ConfigKeywords.KEY_BLOCK_SIZE_M] * config[ ConfigKeywords.KEY_BLOCK_SIZE_N] * dtype_factor * config[ ConfigKeywords.KEY_LOCAL_SPLIT_U]

    if max(value1, value2) < 16384:
        # 展开 KEY_DTYPE 到 KEY_DTYPE_A, KEY_DTYPE_B, KEY_DTYPE_C
        dtype_value = config.pop( ConfigKeywords.KEY_DTYPE_C)
        config[ ConfigKeywords.KEY_DTYPE_A] = dtype_value
        config[ ConfigKeywords.KEY_DTYPE_B] = dtype_value
        config[ ConfigKeywords.KEY_DTYPE_C] = dtype_value

        combinations['cfgs'].append(config)

# 输出为 JSON 文件
def save_to_json(combinations, output_file="config_combinations.json"):
    with open(output_file, "w") as f:
        json.dump(combinations, f, indent=4)

def config_gen(output_file) :
    save_to_json(combinations,output_file)
    print(f"Generated {len(combinations)} configurations and saved to 'config_combinations.json'.")

# # 主函数
# if __name__ == "__main__":
#     save_to_json(combinations)
#     print(f"Generated {len(combinations)} configurations and saved to 'config_combinations.json'.")
    