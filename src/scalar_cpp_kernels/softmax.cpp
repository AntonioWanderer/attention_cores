#include <cmath>
#include "softmax.h"

Tensor1D Softmax(Tensor1D data_line) {
    size_t B = data_line.B;

    float softmax_bias = data_line.at(0);
    for (size_t item = 0; item < B; item++) {
        if (data_line.at(item) > softmax_bias){
            softmax_bias = data_line.at(item);
        }
    }

    float log2e = log2(exp(1.0f));

    Tensor1D result = Tensor1D(B, false);
    float exp_accumulator = 0.0f;
    for (size_t item = 0; item < B; item++) {
        result.at(item) = std::exp2f((data_line.at(item) - softmax_bias) * log2e);
        exp_accumulator += result.at(item);
    }
    float reversed_sum = 1 / exp_accumulator;
    for (size_t item = 0; item < B; item++) {
        result.at(item) = result.at(item) * reversed_sum;
    }

    return result;
}


Tensor3D SoftmaxHeads(Tensor3D input, size_t num_heads) {
    size_t B = input.B;
    size_t S = input.S;
    size_t E = input.E;

    float softmax_bias = input.at(0, 0, 0);
    for (size_t b = 0; b < B; b++) {
        for (size_t s = 0; s < S; s++) {
            for (size_t e = 0; e < E; e++) {
                if (input.at(b, s, e) > softmax_bias) {
                    softmax_bias = input.at(b, s, e);
                }
            }
        }
    }

    float log2e = log2(exp(1.0f));
    size_t D = E / num_heads;

    Tensor3D result = Tensor3D(B, S, E, false);
    for (size_t b = 0; b < B; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t s = 0; s < S; s++) {
                float exp_accumulator = 0.0f;
                for (size_t s2 = 0; s2 < D; s2++) {
                    double exp_value = exp2((input.at(b, s, (D * h) + s2) - softmax_bias) * log2e);
                    result.at(b, s, (D * h) + s2) = exp_value;
                    exp_accumulator += exp_value;
                }
                for (size_t s2 = 0; s2 < D; s2++) {
                    result.at(b, s, (D * h) + s2) = result.at(b, s, (D * h) + s2) / exp_accumulator;
                }
            }
        }
    }
    return result;

}