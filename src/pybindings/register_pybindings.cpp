#include "multihead_attention_pybinding.h"
#include "multihead_attention_simd_pybinding.h"

PYBIND11_MODULE(attention_cpp, m) {
    m.def("attention", &multihead_attention_py);
    m.def("attention_simd", &multihead_attention_simd_py);
}