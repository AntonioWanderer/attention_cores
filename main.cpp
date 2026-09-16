#include <iostream>
#include "bench/bench_matmul.h"
#include "bench/bench_softmax.h"
#include "tests/test_matmul.h"
#include "tests/test_softmax.h"
#include "src/scalar_cpp_kernels/multihead_attention.h"

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
    Tensor3D inp = Tensor3D(15, 23, 64, true);
    Tensor2D wq = Tensor2D(64, 64, true);
    Tensor2D wk = Tensor2D(64, 64, true);
    Tensor2D wv = Tensor2D(64, 64, true);
    Tensor3D attention = multihead_attention(inp, wq, wk, wv, 4);

    return 0;

}