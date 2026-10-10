#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "core_tensor.h"

Tensor1D Softmax(Tensor1D data_line);
Tensor3D SoftmaxHeads(Tensor3D input, size_t num_heads);

#endif