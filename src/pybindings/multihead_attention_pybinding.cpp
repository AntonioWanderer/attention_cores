#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cassert>
#include "multihead_attention.h"


namespace py = pybind11;

py::array_t<float> multihead_attention_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> input_tensor, 
    py::array_t<float, py::array::c_style | py::array::forcecast> W_Q, 
    py::array_t<float, py::array::c_style | py::array::forcecast> W_K, 
    py::array_t<float, py::array::c_style | py::array::forcecast> W_V, 
    size_t num_heads) 
{
    assert(input_tensor.ndim() == 3 && "input_tensor must be 3D");
    assert(W_Q.ndim() == 2 && "W_Q must be 2D");
    assert(W_K.ndim() == 2 && "W_K must be 2D");
    assert(W_V.ndim() == 2 && "W_V must be 2D");

    assert(input_tensor.shape(2) == W_Q.shape(0) && "LAst dimension of input_tensor must be == both dimensions of W_Q");
    assert(input_tensor.shape(2) == W_K.shape(0) && "LAst dimension of input_tensor must be == both dimensions of W_K");
    assert(input_tensor.shape(2) == W_V.shape(0) && "LAst dimension of input_tensor must be == both dimensions of W_V");

    assert(W_Q.shape(0) == W_Q.shape(1) && "Both dimensions of W_Q must be equal");
    assert(W_K.shape(0) == W_K.shape(1) && "Both dimensions of W_K must be equal");
    assert(W_V.shape(0) == W_V.shape(1) && "Both dimensions of W_V must be equal");

    ssize_t B = input_tensor.shape(0);
    ssize_t S = input_tensor.shape(1);
    ssize_t E = input_tensor.shape(2);

    Tensor3D scalar_input_tensor = Tensor3D(B, S, E, false);
    Tensor2D scalar_W_Q = Tensor2D(E, E, false);
    Tensor2D scalar_W_K = Tensor2D(E, E, false);
    Tensor2D scalar_W_V = Tensor2D(E, E, false);

    auto* src_input_data = input_tensor.data();
    scalar_input_tensor.data.assign(src_input_data, src_input_data + B * S * E);
    
    auto* src_W_Q = W_Q.data();
    scalar_W_Q.data.assign(src_W_Q, src_W_Q + E * E);

    auto* src_W_K = W_K.data();
    scalar_W_K.data.assign(src_W_K, src_W_K + E * E);

    auto* src_W_V = W_V.data();
    scalar_W_V.data.assign(src_W_V, src_W_V + E * E);

    Tensor3D attention_out = multihead_attention(scalar_input_tensor, scalar_W_Q, scalar_W_K, scalar_W_V, num_heads);

    std::vector<py::ssize_t> out_shape{B, S, E};
    py::array_t<float_t> py_attention_out(out_shape);
    std::memcpy(py_attention_out.mutable_data(), attention_out.data.data(), B * S * E * sizeof(float));

    return py_attention_out;
}

PYBIND11_MODULE(attention_cpp, m) {
    m.def("attention", &multihead_attention_py);
}