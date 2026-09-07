#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "tensor.h"

Tensor1D Softmax(Tensor1D data_line);

Tensor2D Softmax(Tensor2D data_matrix);

#endif