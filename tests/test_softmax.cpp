#include <iostream>
#include "softmax.h"
#include "softmax_simd.h"
#include "test_softmax.h"


int test_softmax(){
    size_t M = 87;

    Tensor1D line_rand(M, true);
    Tensor1D softmax_res_scalar = Softmax(line_rand);
    Tensor1D softmax_res_simd = SoftmaxSimd(line_rand);

    std::cout << "Start test softmax scalar==simd version\n";

    if (softmax_res_scalar.B != softmax_res_simd.B) {
        std::cout << "Not equal lengths\n";
        return 1;
    }

    if (softmax_res_scalar.B != M) {
        std::cout << "Length != M\n";
        return 1;
    }
    for (size_t row = 0; row < M; row++) {
        if (std::abs(softmax_res_scalar.at(row) - softmax_res_simd.at(row)) > 0.00001) {
            std::cout << "values not equal at " << row << "\n";
            return 1;
        }
    }
    
    std::cout << "both Softmax are correct\n";
    return 0;
};