#include <iostream>
#include "bench/bench_matmul.h"
#include "tests/test_matmul.h"

int main() {

    int matmul_result = test_matmul();
    if (matmul_result == 0) {
        std::cout << "Matmul correct\n";
    }
    bench_matmul();

    return 0;

}