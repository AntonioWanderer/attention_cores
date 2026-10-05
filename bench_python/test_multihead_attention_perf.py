import time
import torch
import numpy
import matplotlib.pyplot as plt
from src_python import attention_refference
from build import attention_cpp

def test_multihead_attention_perf():

    B, S, E, H = 21, 123, 54, 3

    refference_times = []
    scalar_cpp_times = []
    vector_cpp_times = []
    for iteration in range(100):
        input_tensor = torch.randn(B, S, E)
        Wq = torch.randn(E, E)
        Wk = torch.randn(E, E)
        Wv = torch.randn(E, E)

        refference_start = time.perf_counter()
        refference_result = attention_refference.attention_refference(input_tensor=input_tensor,
                                                                    Wq=Wq,
                                                                    Wk=Wk,
                                                                    Wv=Wv, 
                                                                    num_heads=H)
        refference_stop = time.perf_counter()

        scalar_cpp_start = time.perf_counter()
        scalar_cpp_result = attention_cpp.attention(input_tensor.numpy(), 
                                                    Wq.numpy(), 
                                                    Wk.numpy(), 
                                                    Wv.numpy(), 
                                                    H)
        scalar_cpp_stop = time.perf_counter()

        vector_cpp_start = time.perf_counter()
        vector_cpp_result = attention_cpp.attention_simd(input_tensor.numpy(), 
                                                    Wq.numpy(), 
                                                    Wk.numpy(), 
                                                    Wv.numpy(), 
                                                    H)
        vector_cpp_stop = time.perf_counter()

        refference_diff = refference_stop - refference_start
        scalar_cpp_diff = scalar_cpp_stop - scalar_cpp_start
        vector_cpp_diff = vector_cpp_stop - vector_cpp_start

        refference_times.append(refference_diff)
        scalar_cpp_times.append(scalar_cpp_diff)
        vector_cpp_times.append(vector_cpp_diff)

    plt.subplot(1, 3, 1)
    plt.hist(refference_times)
    plt.subplot(1, 3, 2)
    plt.hist(scalar_cpp_times)
    plt.subplot(1, 3, 3)
    plt.hist(vector_cpp_times)
    plt.show()
    print(min(refference_times), min(scalar_cpp_times), min(vector_cpp_times))