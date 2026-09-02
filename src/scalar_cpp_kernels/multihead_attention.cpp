#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

struct Tensor3D {
    std::vector<float> data;
    size_t B, S, E; // Batch, Sequence, Embedding dim

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
    std::vector<float> data;
    size_t B, S;

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