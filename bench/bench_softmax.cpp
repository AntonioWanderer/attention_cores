#include <cmath>
#include <iostream>
#include <chrono>
#include "softmax.h"
#include "softmax_simd.h"
#include "bench_softmax.h"


void bench_softmax()
{
    size_t M = 1081;

    Tensor1D warmup_accumulator_scalar = Tensor1D(M, false);
    Tensor1D warmup_accumulator_simd = Tensor1D(M, false);

    for (int warmup_iteration=0; warmup_iteration < 3; warmup_iteration++) {
        Tensor1D line_rand(M, true);
        Tensor1D softmax_res_scalar = Softmax(line_rand);
        Tensor1D softmax_res_simd = SoftmaxSimd(line_rand);
        warmup_accumulator_scalar.at(0) += softmax_res_scalar.at(0);
        warmup_accumulator_simd.at(0) += softmax_res_simd.at(0);
    }

    Tensor1D accumulator_scalar = Tensor1D(M, false);
    Tensor1D accumulator_simd = Tensor1D(M, false);
    double scalar_time_accumulator = 0.0d;
    double simd_time_accumulator = 0.0d;

    for (int iteration=0; iteration < 10; iteration++) {
        Tensor1D line_rand(M, true);
        
        auto scalar_start = std::chrono::steady_clock::now();
        Tensor1D softmax_res_scalar = Softmax(line_rand);
        auto scalar_stop = std::chrono::steady_clock::now();

        auto simd_start = std::chrono::steady_clock::now();
        Tensor1D softmax_res_simd = SoftmaxSimd(line_rand);
        auto simd_stop = std::chrono::steady_clock::now();

        accumulator_scalar.at(0) += softmax_res_scalar.at(0);
        accumulator_simd.at(0) += softmax_res_simd.at(0);

        double scalar_interval = std::chrono::duration<double, std::milli>(scalar_stop - scalar_start).count();
        double simd_interval = std::chrono::duration<double, std::milli>(simd_stop - simd_start).count();

        scalar_time_accumulator += scalar_interval;
        simd_time_accumulator += simd_interval;
    }

    std::cout << "Scalar softmax, ms: " << scalar_time_accumulator << " simd " <<simd_time_accumulator << std::endl;
    std::cout << "Speed up " << scalar_time_accumulator/simd_time_accumulator << "x" << std::endl;
}
