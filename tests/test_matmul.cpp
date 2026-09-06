#include <cmath>
#include <iostream>
#include "matmul.h"
#include "matmul_simd.h"


int main()
{
    size_t M = 8;
    size_t N = 4;
    size_t K = 16;

    Tensor2D mat1_rand(M, N, true);
    Tensor2D mat2_rand(N, K, true);

    Tensor2D matmul_res_scalar = MatMul(mat1_rand, mat2_rand);
    Tensor2D matmul_res_simd = MatMulSIMD(mat1_rand, mat2_rand);

    std::cout << "Start test\n";

    if (matmul_res_scalar.B != matmul_res_simd.B) {
        std::cout << "Not equal num rows\n";
        return 1;
    }

    if (matmul_res_scalar.B != M) {
        std::cout << "Rows num != M\n";
        return 1;
    }

    if (matmul_res_scalar.S != matmul_res_simd.S) {
        std::cout << "Not equal num cols\n";
        return 1;
    }

    if (matmul_res_scalar.S != K) {
        std::cout << "Cols num != K\n";
        return 1;
    }

    for (size_t row = 0; row < M; row++) {
        for (size_t col = 0; col < K; col++) {
            if (std::abs(matmul_res_scalar.at(row, col) - matmul_res_simd.at(row, col)) > 0.00001) {
                std::cout << "values not equal at " << row << col << "\n";
                return 1;
            }
        }
    }
    
    std::cout << "MatMul correct\n";

    return 0;
}
