
from time import sleep
from typing import List

import torch
import torch.nn as nn
import torch_mlir
from torch.export import Dim
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import (
    make_boxed_compiler,
    get_aot_graph_name,
    set_model_name,
)

import gc
import sys

# from torch_mlir import torchscript
from torch_mlir import fx

from torch_mlir.compiler_utils import run_pipeline_with_repro_report
import torch_mlir.compiler_utils
from VGG19 import VGG19
from resnet50 import ResNet,resnet50
import importlib.util

from internal import import_model


class ModelManager :
    def __init__(self):
        pass
        
    def convert_model_to_torchIR(self,model,inputs):
        # model definition
        # model = self.Model
        # input args
        # inputs = self.ModelInputs
        # convert to torch IR
        m = fx.export_and_import(model,inputs)
        m.dump()
        from torch_mlir.compiler_utils import OutputType
        # mm = torch_mlir.compiler_utils.lower_mlir_module(module=m,output_type=OutputType.LINALG_ON_TENSORS,verbose=False)
    
    def codegen(self,model) :
        # generate kernel hsaco
        # 生成 matmul 、conv和pool的IR，以及hsaco (从预生产的位置拷贝)
        # Runmodel : 调用模型的 RunModel方法即可（dcu、mlu、npu通用）
        pass
    
    def analysis(self,file_path) :
        print("collecting IR operators ...")
        lines = []
        with open(file_path,'r') as f:
            lines = f.readlines()
        conv2dCount = 0
        mmCount = 0
        poolCount = 0
        for line in lines :
            if line.find('torch.aten.conv2d') > 0 :
                conv2dCount+=1
            if line.find('torch.aten.linear') > 0 :
                mmCount+=1
            if line.find('torch.aten.max_pool2d') > 0 :
                poolCount+=1
        sleep(1)
        print("collecting IR operators OK!")
        print("========= Key Ops Statistics ==========")
        print("     conv2d count = ", conv2dCount)
        print("     gemm count = ", mmCount)
        print("     pool2d count = ", poolCount)



if __name__ == "__main__":
    model_path = sys.argv[1]
    mm = ModelManager()
    Model,ModelInputs,RunModel = import_model(model_path)
    mm.convert_model_to_torchIR(Model,ModelInputs)
