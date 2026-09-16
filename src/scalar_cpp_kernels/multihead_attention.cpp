#include <cassert>
#include <cmath>
#include "tensor.h"
#include "matmul.h"
#include "softmax.h"
#include "division.h"
#include "multihead_attention.h"

Tensor3D multihead_attention(Tensor3D input_tensor, Tensor2D W_Q, Tensor2D W_K, Tensor2D W_V, size_t num_heads){
    // input tensor shape [B, S, E]; weights are [E, E]
    size_t E = input_tensor.E;

    assert(E % num_heads == 0 && "Embedding dim must be divisible by num_heads");
    assert(E == W_Q.B && "input tensor last dimension (embedding) have to == W_Q both dimensions");
    assert(E == W_Q.S && "input tensor last dimension (embedding) have to == W_Q both dimensions");
    assert(E == W_K.B && "input tensor last dimension (embedding) have to == W_K both dimensions");
    assert(E == W_K.S && "input tensor last dimension (embedding) have to == W_K both dimensions");
    assert(E == W_V.B && "input tensor last dimension (embedding) have to == W_V both dimensions");
    assert(E == W_V.S && "input tensor last dimension (embedding) have to == W_V both dimensions");

    Tensor3D Q = BatchMatMul(input_tensor, W_Q); // still [B, S, E] after matmul
    Tensor3D K = BatchMatMul(input_tensor, W_K);
    Tensor3D V = BatchMatMul(input_tensor, W_V);

    Tensor3D scores = BatchMatMulHeads(Q, K, num_heads, true);

    Tensor3D divided_scores = Divide(scores, sqrt(E / num_heads));

    Tensor3D attention = SoftmaxHeads(divided_scores, num_heads);

    Tensor3D out = BatchMatMulHeads(attention, V, num_heads, false);

    return out;

}