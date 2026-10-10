#ifndef MATMUL_SIMD_H
#define MATMUL_SIMD_H

#include "core_tensor.h"

Tensor2D MatMulSIMD(const Tensor2D& mat1, const Tensor2D& mat2);
Tensor3D BatchMatMulSIMD(const Tensor3D& mat1, const Tensor2D& mat2);
Tensor3D BatchMatMulHeadsSIMD(const Tensor3D& mat1, const Tensor3D& mat2, size_t num_heads, bool last_require_transpose = false);

#endif