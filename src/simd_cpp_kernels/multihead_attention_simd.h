#ifndef MULTIHEAD_ATTENTION_SIMD
#define MULTIHEAD_ATTENTION_SIMD

#include "core_tensor.h"

Tensor3D multihead_attention_simd(Tensor3D input_tensor, Tensor2D W_Q, Tensor2D W_K, Tensor2D W_V, size_t num_heads);

#endif