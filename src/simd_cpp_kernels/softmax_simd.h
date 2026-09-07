#ifndef SOFTMAX_SIMD_H
#define SOFTMAX_SIMD_H

#include "tensor.h"

Tensor1D SoftmaxSimd(Tensor1D data_line);
// Tensor2D softmax_simd(Tensor2D data_matrix);

#endif