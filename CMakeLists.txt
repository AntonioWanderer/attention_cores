cmake_minimum_required(VERSION 3.15)
project(attention_cores CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR "${pybind11_cmake_dir}")
find_package(pybind11 CONFIG REQUIRED)

find_package(Threads REQUIRED)

set(CORE_SOURCES
    src/common/core_tensor.cpp
    src/scalar_cpp_kernels/matmul.cpp
    src/scalar_cpp_kernels/softmax.cpp
    src/scalar_cpp_kernels/division.cpp
    src/scalar_cpp_kernels/multihead_attention.cpp
    src/simd_cpp_kernels/matmul_simd.cpp
    src/simd_cpp_kernels/softmax_simd.cpp
    src/simd_cpp_kernels/division_simd.cpp
    src/simd_cpp_kernels/multihead_attention_simd.cpp
    src/thread_pool/thread_pool.cpp
)

set(CORE_INCLUDE_DIRS
    src/common
    src/scalar_cpp_kernels
    src/simd_cpp_kernels
    src/thread_pool
)

pybind11_add_module(attention_cpp
    ${CORE_SOURCES}
    src/pybindings/multihead_attention_pybinding.cpp
    src/pybindings/multihead_attention_simd_pybinding.cpp
    src/pybindings/register_pybindings.cpp
)

target_include_directories(attention_cpp PRIVATE
    ${CORE_INCLUDE_DIRS}
    src/pybindings
)

target_link_libraries(attention_cpp PRIVATE Threads::Threads)
target_compile_options(attention_cpp PRIVATE -march=native)
