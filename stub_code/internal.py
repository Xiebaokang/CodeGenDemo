
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
from unittest.mock import patch
# from torch_mlir import torchscript
from torch_mlir import fx

from torch_mlir.compiler_utils import run_pipeline_with_repro_report
import torch_mlir.compiler_utils
from VGG19 import VGG19
from resnet50 import ResNet,resnet50
import importlib.util



def import_model(file_path : str) :
    # 导入模块
    spec = importlib.util.spec_from_file_location("user_model", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Model = module.GetModel()
    ModelInputs = module.GetInputs()
    RunModel = module.RunModel
    return (Model,ModelInputs,RunModel)


class UserInputInfo :
    def __init__(self, modelPath:str, platName : str):
        self.m_modelFilePath = modelPath
        self.m_platName = platName
    
    def set_platform(platName : str) : 
        if platName == 'dcu':
            print("current platform = dcu")
        if platName == 'npu':
            print("current platform = npu")
        if platName == 'mlu':
            print("current platform = mlu")
    

class Operator :
    def __init__(self):
        pass
    def runOperator():
        torch.conv2d()
