#!/bin/bash

build_dir=${BUILD_DIR:-../build/llvm-build/}     # e.g. BUILD_DIR=../build/llvm-release
cd ${build_dir}

cwd=$(pwd)

export PATH=${cwd}/bin:$PATH
export LD_LIBRARY_PATH=${cwd}/lib/:$LD_LIBRARY_PATH

cd ../../scripts/