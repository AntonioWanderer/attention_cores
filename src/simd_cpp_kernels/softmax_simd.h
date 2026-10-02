#ifndef SOFTMAX_SIMD_H
#define SOFTMAX_SIMD_H

#include "core_tensor.h"

Tensor1D SoftmaxSimd(Tensor1D data_line);
Tensor3D SoftmaxHeadsSIMD(Tensor3D input, size_t num_heads);

#endif