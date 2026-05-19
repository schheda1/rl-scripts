#!/bin/bash

build_dir=../build/llvm-build/
cd ${build_dir}

cwd=$(pwd)

export PATH=${cwd}/bin:$PATH
export LD_LIBRARY_PATH=${cwd}/lib/:$LD_LIBRARY_PATH

cd ../../scripts/