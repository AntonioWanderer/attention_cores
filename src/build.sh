#!/bin/bash

set -e

CC="g++"
CFLAGS="-Wall -Wextra -O2"
SRC="$PWD/scalar_cpp_kernels/multihead_attention.cpp"
TARGET="build/attention_cores.so"

$CC $CFLAGS $SRC -o $TARGET