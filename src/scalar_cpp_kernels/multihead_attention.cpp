#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

struct Tensor3D {
    size_t B, S, E; // Batch, Sequence, Embedding dim
    std::vector<float> data;

    Tensor3D(size_t b, size_t s, size_t e, bool init_random = false) : B(b), S(s), E(e), data(b * s * e, 0.0f) {
        if (init_random) {
            std::default_random_engine generator(42);
            std::uniform_real_distribution distribution(-1.0f, 1.0f);
            for (auto& val : data) {
                val = distribution(generator);
            }
        }
    }

    float& at(size_t b, size_t s, size_t e) {
        return data[b * (S * E) + s * E + e];
    }

    const float& at(size_t b, size_t s, size_t e) const {
        return data[b * (S * E) + s * E + e];
    }
};


struct Tensor2D {
    size_t B, S;
    std::vector<float> data;

    Tensor2D(size_t b, size_t s, bool init_random = false) : B(b), S(s), data(b * s, 0.0f) {
        if (init_random) {
            std::default_random_engine generator(42);
            std::uniform_real_distribution distribution(-1.0f, 1.0f);
            for (auto& val : data) {
                val = distribution(generator);
            }
        }
    }

    float& at(size_t b, size_t s) {
        return data[b * S + s];
    }

    const float& at(size_t b, size_t s) const {
        return data[b * S + s];
    }
};


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

int main()
{
    size_t M = 8;
    size_t N = 4;
    size_t K = 16;

    Tensor2D mat1_rand(M, N, true);
    Tensor2D mat2_rand(N, K, true);

    Tensor2D matmul_res = MatMul(mat1_rand, mat2_rand);
    
    return 0;
}
