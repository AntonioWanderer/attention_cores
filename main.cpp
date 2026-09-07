#include <iostream>
#include "bench/bench_matmul.h"
#include "bench/bench_softmax.h"
#include "tests/test_matmul.h"
#include "tests/test_softmax.h"

int main() {

    int matmul_result = test_matmul();
    if (matmul_result != 0) {
        std::cout << "!!! Matmul incorrect\n";
    }

    int softmax_result = test_softmax();
    if (softmax_result != 0) {
        std::cout << "!!! Softmax incorrect\n";
    }

    bench_matmul();
    bench_softmax();

    return 0;

}