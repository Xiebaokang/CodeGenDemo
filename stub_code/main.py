# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import subprocess
from internal import *
from model_manager import *
    
def main_process(info : UserInputInfo) :
    # model_path = '/home/xushilong/tiaoyouqi/stub_code/VGG19.py'
    # logPath = '/home/xushilong/tiaoyouqi/stderr_output.txt'
    model_path = info.m_modelFilePath
    logPath = '/home/xushilong/tiaoyouqi/stderr_output.txt'
    print("reading model & convert into IR ...")
    cmd = ["python","model_manager.py",model_path]

    Model,ModelInputs,RunModel = import_model(model_path) 
    with open(logPath,'w') as f :
        subprocess.call(cmd,stdout=f,stderr=f)
        print("convert model OK !")
    # analysis model operators
    mm = ModelManager()
    mm.analysis(logPath)
    # run model
    print("======== start run model =========")
    RunModel()
    


if __name__ == "__main__":
    main_process(UserInputInfo('/home/xushilong/tiaoyouqi/stub_code/VGG19.py','dcu'))

