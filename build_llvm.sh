#!/bin/bash

build_dir=../build/llvm-build/
mkdir -p ${build_dir}
cd ${build_dir}

BUILD_TYPE=RelWithDebInfo
CMAKEDIR=../../llvm-project/llvm
ASSERTIONS=on

cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DLLVM_ENABLE_PROJECTS='clang;lld' \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DLLVM_OPTIMIZED_TABLEGEN=on \
    -DBUILD_SHARED_LIBS=on \
    -DLLVM_ENABLE_ASSERTIONS=${ASSERTIONS} \
    -DLLVM_USE_LINKER=gold \
    -DLLVM_CCACHE_BUILD=off \
    ${CMAKEDIR}

ninja

cwd=$(pwd)

export PATH=${cwd}/bin:$PATH
export LD_LIBRARY_PATH=${cwd}/lib/:$LD_LIBRARY_PATH

cd ../../scripts/