#include <cmath>
#include "softmax_simd.h"
#include "immintrin.h"


Tensor1D SoftmaxSimd(Tensor1D data_line){
    size_t B = data_line.B;
    Tensor1D result = Tensor1D(B, false);
    float exp_accumulator = 0.0f;
    for (size_t item = 0; item < B; item++) {
        result.at(item) = std::exp2f(data_line.at(item));
        exp_accumulator += result.at(item);
    }
    float reversed_sum = 1 / exp_accumulator;
    __m256 reversed_sum_vector = _mm256_set1_ps(reversed_sum);
    size_t tail_item = B;
    for (size_t item = 0; item < B; item+=8) {
        if (item + 8 > B) {
            tail_item = item;
            break;
        }
        __m256 loaded_items = _mm256_loadu_ps(result.addr(item));
        __m256 result_items = _mm256_mul_ps(reversed_sum_vector, loaded_items);
        _mm256_storeu_ps(result.addr(item), result_items);
    }
    for (size_t item = tail_item; item < B; item++) {
        result.at(item) = result.at(item) * reversed_sum;
    }

    return result;
}


// Tensor2D softmax_simd(Tensor2D data_matrix){

// }