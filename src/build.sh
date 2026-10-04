#!/bin/bash

set -e

CC="g++"
CFLAGS="-shared -fPIC -O2 -march=native $(python -m pybind11 --includes)"
INCLUDEFLAGS="-Icommon -Iscalar_cpp_kernels -Isimd_cpp_kernels -Ithread_pool"
# $PWD/scalar_cpp_kernels/multihead_attention.cpp 
SRC="common/core_tensor.cpp \
     scalar_cpp_kernels/matmul.cpp \
     simd_cpp_kernels/matmul_simd.cpp \
     scalar_cpp_kernels/softmax.cpp \
     simd_cpp_kernels/softmax_simd.cpp \
     scalar_cpp_kernels/division.cpp \
     simd_cpp_kernels/division_simd.cpp \
     scalar_cpp_kernels/multihead_attention.cpp \
     simd_cpp_kernels/multihead_attention_simd.cpp \
     pybindings/multihead_attention_pybinding.cpp \
     pybindings/multihead_attention_simd_pybinding.cpp \
     pybindings/register_pybindings.cpp \
     thread_pool/thread_pool.cpp \
     ../tests/test_matmul.cpp \
     ../tests/test_softmax.cpp \
     ../bench/bench_matmul.cpp \
     ../bench/bench_softmax.cpp"
     # ../main.cpp
TARGET="../build/attention_cpp$(python3-config --extension-suffix)"

$CC $CFLAGS $INCLUDEFLAGS $SRC -o $TARGET