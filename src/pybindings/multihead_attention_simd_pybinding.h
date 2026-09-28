#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<float> multihead_attention_simd_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> input_tensor, 
    py::array_t<float, py::array::c_style | py::array::forcecast> W_Q, 
    py::array_t<float, py::array::c_style | py::array::forcecast> W_K, 
    py::array_t<float, py::array::c_style | py::array::forcecast> W_V, 
    size_t num_heads);