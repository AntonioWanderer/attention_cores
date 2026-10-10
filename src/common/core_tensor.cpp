#include "core_tensor.h"
#include <random>


Tensor1D::Tensor1D(size_t b, bool init_random) : B(b), data(b, 0.0f) {
    if (init_random) {
        std::default_random_engine generator(42);
        std::uniform_real_distribution distribution(-1.0f, 1.0f);
        for (auto& val : data) {
            val = distribution(generator);
        }
    }
}

float& Tensor1D::at(size_t b) {
    return data[b];
}

const float& Tensor1D::at(size_t b) const {
    return data[b];
}

float* Tensor1D::addr(size_t b) {
    return &data[b];
}

const float* Tensor1D::addr(size_t b) const {
    return &data[b];
}


Tensor2D::Tensor2D(size_t b, size_t s, bool init_random) : B(b), S(s), data(b * s, 0.0f) {
    if (init_random) {
        std::default_random_engine generator(42);
        std::uniform_real_distribution distribution(-1.0f, 1.0f);
        for (auto& val : data) {
            val = distribution(generator);
        }
    }
}

float& Tensor2D::at(size_t b, size_t s) {
    return data[b * S + s];
}

const float& Tensor2D::at(size_t b, size_t s) const {
    return data[b * S + s];
}

float* Tensor2D::addr(size_t b, size_t s) {
    return &data[b * S + s];
}

const float* Tensor2D::addr(size_t b, size_t s) const {
    return &data[b * S + s];
}

Tensor3D::Tensor3D(){

}

Tensor3D::Tensor3D(size_t b, size_t s, size_t e, bool init_random) : B(b), S(s), E(e), data(b * s * e, 0.0f) {
    if (init_random) {
        std::default_random_engine generator(42);
        std::uniform_real_distribution distribution(-1.0f, 1.0f);
        for (auto& val : data) {
            val = distribution(generator);
        }
    }
}

float& Tensor3D::at(size_t b, size_t s, size_t e) {
    return data[b * (S * E) + s * E + e];
}

const float& Tensor3D::at(size_t b, size_t s, size_t e) const {
    return data[b * (S * E) + s * E + e];
}

float* Tensor3D::addr(size_t b, size_t s, size_t e) {
    return &data[b * (S * E) + s * E + e];
}

const float* Tensor3D::addr(size_t b, size_t s, size_t e) const {
    return &data[b * (S * E) + s * E + e];
}


Tensor4D::Tensor4D(size_t b, size_t s, size_t e, size_t n, bool init_random) : B(b), S(s), E(e), N(n), data(b * s * e * n, 0.0f) {
    if (init_random) {
        std::default_random_engine generator(42);
        std::uniform_real_distribution distribution(-1.0f, 1.0f);
        for (auto& val : data) {
            val = distribution(generator);
        }
    }
}

float& Tensor4D::at(size_t b, size_t s, size_t e, size_t n) {
    return data[b * (S * E * N) + s * (E * N) + e * N + n];
}

const float& Tensor4D::at(size_t b, size_t s, size_t e, size_t n) const {
    return data[b * (S * E * N) + s * (E * N) + e * N + n];
}

float* Tensor4D::addr(size_t b, size_t s, size_t e, size_t n) {
    return &data[b * (S * E * N) + s * (E * N) + e * N + n];
}

const float* Tensor4D::addr(size_t b, size_t s, size_t e, size_t n) const {
    return &data[b * (S * E * N) + s * (E * N) + e * N + n];
}