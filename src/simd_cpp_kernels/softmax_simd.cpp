#include <cmath>
#include "softmax_simd.h"
#include <immintrin.h>


Tensor1D SoftmaxSimd(Tensor1D data_line){
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


Tensor3D SoftmaxHeadsSIMD(Tensor3D input, size_t num_heads) {
    size_t B = input.B;
    size_t S = input.S;
    size_t E = input.E;
    // TODO: vectorisation second part

    float log2e = log2(exp(1.0f));
    size_t D = E / num_heads;

    Tensor3D result = Tensor3D(B, S, E, false);

    Tensor1D catpture_softmax_bias_vector = Tensor1D(8, false);

    for (size_t b = 0; b < B; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t s = 0; s < S; s++) {
                float exp_accumulator = 0.0f;

                size_t last_s2_max_search_index = D;
                float scalar_softmax_bias = input.at(b, s, D * h);
                if (D > 8) {
                    __m256 softmax_bias = _mm256_set1_ps(input.at(b, s, D * h));
                    for (size_t s2 = 0; s2 < D; s2+=8) {
                        if (s2 + 8 > D) {
                            last_s2_max_search_index = s2;
                            break;
                        }
                        __m256 new_data_vector = _mm256_loadu_ps(input.addr(b, s, (D * h) + s2));
                        softmax_bias = _mm256_max_ps(softmax_bias, new_data_vector);
                    }
                    _mm256_storeu_ps(catpture_softmax_bias_vector.addr(0), softmax_bias);
                    scalar_softmax_bias = catpture_softmax_bias_vector.at(0);
                    for (size_t into_capture_index = 0; into_capture_index < 8; into_capture_index++) {
                        if (catpture_softmax_bias_vector.at(into_capture_index) > scalar_softmax_bias) {
                            scalar_softmax_bias = catpture_softmax_bias_vector.at(into_capture_index);
                        }
                    }

                } else {
                    last_s2_max_search_index = 0;
                }
                
                for (size_t s2 = last_s2_max_search_index; s2 < D; s2++) {
                    if (input.at(b, s, (D * h) + s2) > scalar_softmax_bias) {
                        scalar_softmax_bias = input.at(b, s, (D * h) + s2);
                    }
                }

                for (size_t s2 = 0; s2 < D; s2++) {
                    double exp_value = exp2((input.at(b, s, (D * h) + s2) - scalar_softmax_bias) * log2e);
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