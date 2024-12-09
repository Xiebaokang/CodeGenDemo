#! /bin/bash
is_as_pymodule='ON'
is_user_dbg='OFF'
cd build  
cmake .. -DCOMPILE_AS_PYMODULE=$is_as_pymodule -DUSER_DEBUG=$is_user_dbg
make -j4
