#!/bin/bash

set -e

CC="g++"
CFLAGS="-Wall -Wextra -O2 -march=native"
INCLUDEFLAGS="-Icommon -Iscalar_cpp_kernels -Isimd_cpp_kernels"
# $PWD/scalar_cpp_kernels/multihead_attention.cpp 
SRC="common/tensor.cpp \
     scalar_cpp_kernels/matmul.cpp \
     simd_cpp_kernels/matmul_simd.cpp \
     ../tests/test_matmul.cpp \
     ../bench/bench_matmul.cpp \
     ../main.cpp"
TARGET="build/test"

$CC $CFLAGS $INCLUDEFLAGS $SRC -o $TARGET