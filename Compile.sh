#! /bin/bash
option=$1
cd build && cmake .. -DCOMPILE_AS_PYMODULE=$option && make -j4