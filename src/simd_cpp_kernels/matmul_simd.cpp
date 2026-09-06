#include <cassert>
#include "matmul_simd.h"
#include <immintrin.h>


Tensor2D MatMulSIMD(const Tensor2D& mat1, const Tensor2D& mat2) {
    size_t M = mat1.B;
    size_t N1 = mat1.S;
    size_t N2 = mat2.B;
    size_t K = mat2.S;

    assert(N1 == N2 && "Last dimension of 1 matrix must be == first dimension of 2 matrix");

    size_t N = N1;
    Tensor2D result(M, K, false);

    for (size_t im=0; im < M; im++) {
        size_t ik_tail = K;
        for (size_t ik=0; ik < K; ik+=8) {
            if (ik + 8 > K) {
                ik_tail = ik;
                break;
            }
            __m256 mk_value = _mm256_set1_ps(0.0f);
            for (size_t jn=0; jn < N; jn++) {
                __m256 first_copied = _mm256_set1_ps(mat1.at(im, jn));
                __m256 second_loaded = _mm256_loadu_ps(mat2.addr(jn, ik));
                mk_value = _mm256_fmadd_ps(first_copied, second_loaded, mk_value);
            }
            _mm256_storeu_ps(result.addr(im, ik), mk_value);
        }
        for (size_t ik_scalar = ik_tail; ik_scalar < K; ik_scalar++) {
            float mk_value_scalar = 0.0f;
            for (size_t jn=0; jn < N; jn++) {
                mk_value_scalar += mat1.at(im, jn) * mat2.at(jn, ik_scalar);
            }
            result.at(im, ik_scalar) = mk_value_scalar;
        }
    }

    return result;
}
