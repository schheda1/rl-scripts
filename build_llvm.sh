#!/bin/bash

# Defaults are the development build (RelWithDebInfo, assertions, shared libs,
# all targets).  Collection build (no assertions, static, 3 targets; ~1.6x
# faster per cell compile on this code, same codegen):
#   BUILD_DIR=../build/llvm-release BUILD_TYPE=Release ASSERTIONS=off \
#   SHARED_LIBS=off TARGETS='X86;NVPTX;AMDGPU' ./build_llvm.sh
# then `BUILD_DIR=../build/llvm-release source export_llvm.sh`.
build_dir=${BUILD_DIR:-../build/llvm-build/}
mkdir -p ${build_dir}
cd ${build_dir}

BUILD_TYPE=${BUILD_TYPE:-Release}
CMAKEDIR=../../llvm-project/llvm      # BUILD_DIR must be ../build/<name>
ASSERTIONS=${ASSERTIONS:-off}
SHARED_LIBS=${SHARED_LIBS:-off}
TARGETS=${TARGETS:-X86;NVPTX;AMDGPU}

cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DLLVM_ENABLE_PROJECTS='clang;lld' \
    -DLLVM_TARGETS_TO_BUILD="${TARGETS}" \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DLLVM_OPTIMIZED_TABLEGEN=on \
    -DBUILD_SHARED_LIBS=${SHARED_LIBS} \
    -DLLVM_ENABLE_ASSERTIONS=${ASSERTIONS} \
    -DLLVM_USE_LINKER=gold \
    -DLLVM_CCACHE_BUILD=off \
    ${CMAKEDIR}

ninja -j24

cwd=$(pwd)

export PATH=${cwd}/bin:$PATH
export LD_LIBRARY_PATH=${cwd}/lib/:$LD_LIBRARY_PATH

cd ../../scripts/
