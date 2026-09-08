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
