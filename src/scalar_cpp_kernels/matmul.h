#ifndef MATMUL_H
#define MATMUL_H

#include "tensor.h"

Tensor2D MatMul(const Tensor2D& mat1, const Tensor2D& mat2);

Tensor3D BatchMatMul(const Tensor3D& mat1, const Tensor2D& mat2);
Tensor3D BatchMatMul(const Tensor3D& mat1, const Tensor3D& mat2);
Tensor3D BatchMatMulHeads(const Tensor3D& mat1, const Tensor3D& mat2, size_t num_heads, bool last_require_transpose = false);

#endif