#include <immintrin.h>
#include "division_simd.h"

Tensor3D DivideSIMD(Tensor3D input, double divisor){
    size_t B = input.B;
    size_t S = input.S;
    size_t E = input.E;

    Tensor3D result = Tensor3D(B, S, E, false);

    __m256 reversed_divisor = _mm256_set1_ps(1 / divisor);

    for (size_t b = 0; b < B; b++) {
        for (size_t s = 0; s < S; s++) {
            size_t last_e = E;
            for (size_t e = 0; e < E; e+=8) {
                if (e + 8 > E) {
                    last_e = e;
                    break;
                }
                __m256 loaded_line = _mm256_loadu_ps(input.addr(b, s, e));
                __m256 divided_line = _mm256_mul_ps(loaded_line, reversed_divisor);
                _mm256_storeu_ps(result.addr(b, s, e), divided_line);
            }
            for (size_t e_residual = last_e; e_residual < E; e_residual++) {
                result.at(b, s, e_residual) = input.at(b, s, e_residual) / divisor;
            }
        }
    }

    return result;
}