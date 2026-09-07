#include <cmath>
#include "softmax.h"

Tensor1D Softmax(Tensor1D data_line) {
    size_t B = data_line.B;
    Tensor1D result = Tensor1D(B, false);
    float exp_accumulator = 0.0f;
    for (size_t item = 0; item < B; item++) {
        result.at(item) = std::exp2f(data_line.at(item));
        exp_accumulator += result.at(item);
    }
    float reversed_sum = 1 / exp_accumulator;
    for (size_t item = 0; item < B; item++) {
        result.at(item) = result.at(item) * reversed_sum;
    }

    return result;
}


Tensor2D Softmax(Tensor2D data_matrix) {
    size_t B = data_matrix.B;
    size_t S = data_matrix.S;
    Tensor2D result = Tensor2D(B, S, false);
    float exp_accumulator = 0.0f;
    for (size_t ib = 0; ib < B; ib++) {
        for (size_t is = 0; is < S; is++) {
            result.at(ib, is) = std::exp2f(data_matrix.at(ib, is));
            exp_accumulator += result.at(ib, is);
        }
    }
    float reversed_sum = 1 / exp_accumulator;
    for (size_t ib = 0; ib < B; ib++){
        for (size_t is = 0; is < S; is++){
            result.at(ib, is) = result.at(ib, is) * reversed_sum;
        }
    }

    return result;
}