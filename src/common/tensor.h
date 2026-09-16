#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>

struct Tensor1D {
    size_t B;
    std::vector<float> data;

    Tensor1D(size_t b, bool init_random = false);
    float& at(size_t b);
    const float& at(size_t b) const;
    float* addr(size_t b);
    const float* addr(size_t b) const;
};


struct Tensor2D {
    size_t B, S;
    std::vector<float> data;

    Tensor2D(size_t b, size_t s, bool init_random = false);
    float& at(size_t b, size_t s);
    const float& at(size_t b, size_t s) const;
    float* addr(size_t b, size_t s);
    const float* addr(size_t b, size_t s) const;
};


struct Tensor3D {
    size_t B, S, E; // Batch, Sequence, Embedding dim
    std::vector<float> data;

    Tensor3D();
    Tensor3D(size_t b, size_t s, size_t e, bool init_random = false);
    float& at(size_t b, size_t s, size_t e);
    const float& at(size_t b, size_t s, size_t e) const;
    float* addr(size_t b, size_t s, size_t e);
    const float* addr(size_t b, size_t s, size_t e) const;
};


struct Tensor4D {
    size_t B, S, E, N;
    std::vector<float> data;

    Tensor4D(size_t b, size_t s, size_t e, size_t n, bool init_random = false);
    float& at(size_t b, size_t s, size_t e, size_t n);
    const float& at(size_t b, size_t s, size_t e, size_t n) const;
    float* addr(size_t b, size_t s, size_t e, size_t n);
    const float* addr(size_t b, size_t s, size_t e, size_t n) const;
};


#endif