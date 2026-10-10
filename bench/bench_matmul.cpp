#include <cmath>
#include <iostream>
#include <chrono>
#include "matmul.h"
#include "matmul_simd.h"
#include "bench_matmul.h"


void bench_matmul()
{
    size_t M = 1081;
    size_t N = 153;
    size_t K = 329;

    Tensor2D warmup_accumulator_scalar = Tensor2D(M, K, false);
    Tensor2D warmup_accumulator_simd = Tensor2D(M, K, false);

    for (int warmup_iteration=0; warmup_iteration < 3; warmup_iteration++) {
        Tensor2D mat1_rand(M, N, true);
        Tensor2D mat2_rand(N, K, true);
        Tensor2D matmul_res_scalar = MatMul(mat1_rand, mat2_rand);
        Tensor2D matmul_res_simd = MatMulSIMD(mat1_rand, mat2_rand);
        warmup_accumulator_scalar.at(0, 0) += matmul_res_scalar.at(0, 0);
        warmup_accumulator_simd.at(0, 0) += matmul_res_simd.at(0, 0);
    }

    Tensor2D accumulator_scalar = Tensor2D(M, K, false);
    Tensor2D accumulator_simd = Tensor2D(M, K, false);
    double scalar_time_accumulator = 0.0d;
    double simd_time_accumulator = 0.0d;

    for (int iteration=0; iteration < 10; iteration++) {
        Tensor2D mat1_rand(M, N, true);
        Tensor2D mat2_rand(N, K, true);
        
        auto scalar_start = std::chrono::steady_clock::now();
        Tensor2D matmul_res_scalar = MatMul(mat1_rand, mat2_rand);
        auto scalar_stop = std::chrono::steady_clock::now();

        auto simd_start = std::chrono::steady_clock::now();
        Tensor2D matmul_res_simd = MatMulSIMD(mat1_rand, mat2_rand);
        auto simd_stop = std::chrono::steady_clock::now();

        accumulator_scalar.at(0, 0) += matmul_res_scalar.at(0, 0);
        accumulator_simd.at(0, 0) += matmul_res_simd.at(0, 0);

        double scalar_interval = std::chrono::duration<double, std::milli>(scalar_stop - scalar_start).count();
        double simd_interval = std::chrono::duration<double, std::milli>(simd_stop - simd_start).count();

        scalar_time_accumulator += scalar_interval;
        simd_time_accumulator += simd_interval;
    }

    std::cout << "Scalar matmul, ms: " << scalar_time_accumulator << " simd " <<simd_time_accumulator << std::endl;
    std::cout << "Speed up " << scalar_time_accumulator/simd_time_accumulator << "x" << std::endl;
}
