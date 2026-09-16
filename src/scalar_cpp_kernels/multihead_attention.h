#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include "tensor.h"

Tensor3D multihead_attention(Tensor3D input_tensor, Tensor2D W_Q, Tensor2D W_K, Tensor2D W_V, size_t num_heads);

#endif