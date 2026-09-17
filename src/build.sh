#!/bin/bash

set -e

CC="g++"
CFLAGS="-shared -fPIC -O2 -march=native $(python3 -m pybind11 --includes)"
INCLUDEFLAGS="-Icommon -Iscalar_cpp_kernels -Isimd_cpp_kernels"
# $PWD/scalar_cpp_kernels/multihead_attention.cpp 
SRC="common/core_tensor.cpp \
     scalar_cpp_kernels/matmul.cpp \
     simd_cpp_kernels/matmul_simd.cpp \
     scalar_cpp_kernels/softmax.cpp \
     simd_cpp_kernels/softmax_simd.cpp \
     scalar_cpp_kernels/division.cpp \
     scalar_cpp_kernels/multihead_attention.cpp \
     ../tests/test_matmul.cpp \
     ../tests/test_softmax.cpp \
     ../bench/bench_matmul.cpp \
     ../bench/bench_softmax.cpp"
     # ../main.cpp
TARGET="build/attention_cpp$(python3-config --extension-suffix)"

$CC $CFLAGS $INCLUDEFLAGS $SRC -o $TARGET