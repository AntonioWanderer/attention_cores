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


Tensor3D BatchMatMulSIMD(const Tensor3D& mat1, const Tensor2D& mat2) {
    size_t B = mat1.B;
    size_t M = mat1.S;
    size_t N1 = mat1.E;
    size_t N2 = mat2.B;
    size_t K = mat2.S;

    assert(N1 == N2 && "Last dimension of 1 matrix must be == first dimension of 2 matrix");

    size_t N = N1;
    Tensor3D result(B, M, K, false);

    for (size_t bs = 0; bs < B; bs++){
        for (size_t im=0; im < M; im++) {
            size_t ik_tail = K;
            for (size_t ik=0; ik < K; ik+=8) {
                if (ik + 8 > K) {
                    ik_tail = ik;
                    break;
                }
                __m256 mk_value = _mm256_set1_ps(0.0f);
                for (size_t jn=0; jn < N; jn++) {
                    __m256 first_copied = _mm256_set1_ps(mat1.at(bs, im, jn));
                    __m256 second_loaded = _mm256_loadu_ps(mat2.addr(jn, ik));
                    mk_value = _mm256_fmadd_ps(first_copied, second_loaded, mk_value);
                }
                _mm256_storeu_ps(result.addr(bs, im, ik), mk_value);
            }
            for (size_t ik_scalar = ik_tail; ik_scalar < K; ik_scalar++) {
                float mk_value_scalar = 0.0f;
                for (size_t jn=0; jn < N; jn++) {
                    mk_value_scalar += mat1.at(bs, im, jn) * mat2.at(jn, ik_scalar);
                }
                result.at(bs, im, ik_scalar) = mk_value_scalar;
            }
        }
    }

    return result;
}


Tensor3D BatchMatMulHeadsSIMD(const Tensor3D& mat1, const Tensor3D& mat2, size_t num_heads, bool last_require_transpose) {
    size_t B1 = mat1.B;
    size_t B2 = mat2.B;
    size_t S1 = mat1.S;
    size_t E1 = mat1.E;
    size_t E2 = mat2.E;
    size_t S2 = mat2.S;

    // mat1 [B, S, E1] <--> [B, H, S, dh1]
    // mat2 [B, s, E2] <--> [B, H, S, dh2]

    assert(B1==B2 && "First dim of both tensors must be equal");
    // assert(E1 == E2 && "Last dim of both tensors must be equal");
    assert(S1 == S2 && "Middle dim of both tensors must be equal");

    assert(E1 % num_heads == 0 && "Embedding dim must be divisible by num_heads");
    assert(E2 % num_heads == 0 && "Embedding dim must be divisible by num_heads");

    size_t B = B1;
    size_t S = S1;
    size_t dh1 = E1 / num_heads;
    size_t dh2 = E2 / num_heads;

    size_t final_dim_1;
    size_t final_dim_2;
    size_t internal_dimension;
    Tensor3D result;

    if (last_require_transpose) {
        result = Tensor3D(B, S, S * num_heads, false);
        final_dim_1 = S;
        final_dim_2 = S;
        assert(dh1 == dh2 && "In this case embedding dims must be equal");
        internal_dimension = dh1;
    } else {
        result = Tensor3D(B, S, E2, false);
        final_dim_1 = S;
        final_dim_2 = dh2;
        assert(dh1 == S && "In this case must be E1 == S * num_heads");
        internal_dimension = dh1;
    }

    if (last_require_transpose) {
        for (size_t b = 0; b < B; b++){
            for (size_t h=0; h < num_heads; h++){
                for (size_t s1 = 0; s1 < final_dim_1; s1++){
                    for (size_t s2 = 0; s2 < final_dim_2; s2++){
                        size_t last_d = internal_dimension;
                        __m256 mk_value = _mm256_set1_ps(0.0f);
                        for (size_t d=0; d < internal_dimension; d+=8) {
                            if (d + 8 > internal_dimension) {
                                last_d = d;
                                break;
                            }
                            __m256 first_loaded = _mm256_loadu_ps(mat1.addr(b, s1, (dh1 * h + d)));
                            __m256 second_loaded = _mm256_loadu_ps(mat2.addr(b, s2, (dh1 * h + d)));
                            mk_value = _mm256_fmadd_ps(first_loaded, second_loaded, mk_value);
                        }
                        float final_mk_value = 0.0f;
                        Tensor1D mk_vector_receiver = Tensor1D(8, false);
                        _mm256_storeu_ps(mk_vector_receiver.addr(0), mk_value);
                        for (size_t recv_i = 0; recv_i < 8; recv_i++){
                            final_mk_value += mk_vector_receiver.at(recv_i);
                        }
                        for (size_t d_residual = last_d; d_residual < internal_dimension; d_residual++){
                            final_mk_value += mat1.at(b, s1, (dh1 * h + d_residual)) * mat2.at(b, s2, (dh1 * h + d_residual));
                        }

                        result.at(b, s1, (h * S) + s2) = final_mk_value;
                    }
                }
            }
        }
    } else {
        for (size_t b = 0; b < B; b++){
            for (size_t h=0; h < num_heads; h++){
                for (size_t s1 = 0; s1 < final_dim_1; s1++){
                    size_t last_s2 = final_dim_2;
                    for (size_t s2 = 0; s2 < final_dim_2; s2+=8){
                        if (s2 + 8 > final_dim_2) {
                            last_s2 = s2;
                            break;
                        }
                        __m256 mk_value = _mm256_set1_ps(0.0f);
                        for (size_t d=0; d < internal_dimension; d++) {
                            __m256 first_diplicated = _mm256_set1_ps(mat1.at(b, s1, (dh1 * h + d)));
                            __m256 second_loaded = _mm256_loadu_ps(mat2.addr(b, d, (h * dh2) + s2));
                            mk_value = _mm256_fmadd_ps(first_diplicated, second_loaded, mk_value);
                        }
                        _mm256_storeu_ps(result.addr(b, s1, (h * dh2) + s2), mk_value);
                    }
                    for (size_t s2_residual = last_s2; s2_residual < final_dim_2; s2_residual++) {
                        float mk_value_scalar = 0.0f;
                        for (size_t d=0; d < internal_dimension; d++) {
                            mk_value_scalar += mat1.at(b, s1, (dh1 * h + d)) * mat2.at(b, d, (h * dh2) + s2_residual);
                        }
                        result.at(b, s1, (h * dh2) + s2_residual) = mk_value_scalar;
                    }
                }
            }
        }
    }
    
    return result;
}