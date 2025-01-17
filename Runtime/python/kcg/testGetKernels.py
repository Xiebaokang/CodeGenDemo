from KCGTask import *
from ConfigGenerator import *

json_path = '/home/xushilong/CodeGenDemo/cfg_cominations.json'
config_gen(json_path)
tg = KernelTaskGroup(json_path=json_path)
try:
    tg.run(limit=30)
except Exception as e:
    print(e)
print("OK")