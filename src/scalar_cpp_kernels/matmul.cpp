#include <cassert>
#include "matmul.h"

Tensor2D MatMul(const Tensor2D& mat1, const Tensor2D& mat2) {
    size_t M = mat1.B;
    size_t N1 = mat1.S;
    size_t N2 = mat2.B;
    size_t K = mat2.S;

    assert(N1 == N2 && "Last dimension of 1 matrix must be == first dimension of 2 matrix");

    size_t N = N1;
    Tensor2D result(M, K, false);

    for (size_t im=0; im < M; im++) {
        for (size_t ik=0; ik < K; ik++) {
            float mk_value = 0.0f;
            for (size_t jn=0; jn < N; jn++) {
                mk_value += mat1.at(im, jn) * mat2.at(jn, ik);
            }
            result.at(im, ik) = mk_value;
        }
    }

    return result;
}


Tensor3D BatchMatMul(const Tensor3D& mat1, const Tensor2D& mat2) {
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
            for (size_t ik=0; ik < K; ik++) {
                float mk_value = 0.0f;
                for (size_t jn=0; jn < N; jn++) {
                    mk_value += mat1.at(bs, im, jn) * mat2.at(jn, ik);
                }
                result.at(bs, im, ik) = mk_value;
            }
        }
    }

    return result;
}


Tensor3D BatchMatMul(const Tensor3D& mat1, const Tensor3D& mat2) {
    size_t B1 = mat1.B;
    size_t B2 = mat2.B;
    size_t M = mat1.S;
    size_t N1 = mat1.E;
    size_t N2 = mat2.S;
    size_t K = mat2.E;

    assert(B1==B2 && "First dim of both tensors must be equal");
    assert(N1 == N2 && "Last dimension of 1 matrix must be == first dimension of 2 matrix");

    size_t N = N1;
    size_t B = B1;
    Tensor3D result(B, M, K, false);

    for (size_t bs = 0; bs < B; bs++){
        for (size_t im=0; im < M; im++) {
            for (size_t ik=0; ik < K; ik++) {
                float mk_value = 0.0f;
                for (size_t jn=0; jn < N; jn++) {
                    mk_value += mat1.at(bs, im, jn) * mat2.at(bs, jn, ik);
                }
                result.at(bs, im, ik) = mk_value;
            }
        }
    }

    return result;
}


Tensor3D BatchMatMulHeads(const Tensor3D& mat1, const Tensor3D& mat2, size_t num_heads, bool last_require_transpose) {
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

    for (size_t b = 0; b < B; b++){
        for (size_t h=0; h < num_heads; h++){
            for (size_t s1 = 0; s1 < final_dim_1; s1++){
                for (size_t s2 = 0; s2 < final_dim_2; s2++){
                    float mk_value = 0.0f;
                    for (size_t d=0; d < internal_dimension; d++) {
                        if (last_require_transpose) {
                            mk_value += mat1.at(b, s1, (dh1 * h + d)) * mat2.at(b, s2, (dh1 * h + d));
                        } else {
                            mk_value += mat1.at(b, s1, (dh1 * h + d)) * mat2.at(b, d, (h * dh2) + s2);
                        }
                    if (last_require_transpose) {
                        result.at(b, s1, (h * S) + s2) = mk_value;
                    } else {
                        result.at(b, s1, (h * dh2) + s2) = mk_value;
                    }
                    
                    }
                }
            }
        }
    }

    return result;
}