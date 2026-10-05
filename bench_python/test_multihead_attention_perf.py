import time
import torch
from torch.utils.benchmark import Timer
import numpy
import matplotlib.pyplot as plt
from src_python import attention_refference
from build import attention_cpp

def test_multihead_attention_perf():

    B, S, E, H = 21, 123, 54, 3
    repeat_times = 10

    torch_time = Timer(stmt="attention_refference.attention_refference(input_tensor=input_tensor,Wq=Wq,Wk=Wk,Wv=Wv,num_heads=H)",
                       setup = f"B, S, E, H = {B}, {S}, {E}, {H}; input_tensor = torch.randn(B, S, E); Wq = torch.randn(E, E); Wk = torch.randn(E, E); Wv = torch.randn(E, E)", 
                       globals=globals()).timeit(repeat_times)

    cpp_scalar_time = Timer(stmt="attention_cpp.attention(input_tensor, Wq, Wk, Wv, H)",
                       setup = f"B, S, E, H = {B}, {S}, {E}, {H}; input_tensor = torch.randn(B, S, E).numpy(); Wq = torch.randn(E, E).numpy(); Wk = torch.randn(E, E).numpy(); Wv = torch.randn(E, E).numpy()", 
                       globals=globals()).timeit(repeat_times)

    cpp_simd_time = Timer(stmt="attention_cpp.attention_simd(input_tensor, Wq, Wk, Wv, H)",
                           setup = f"B, S, E, H = {B}, {S}, {E}, {H}; input_tensor = torch.randn(B, S, E).numpy(); Wq = torch.randn(E, E).numpy(); Wk = torch.randn(E, E).numpy(); Wv = torch.randn(E, E).numpy()", 
                           globals=globals()).timeit(repeat_times)

    print(f"{(torch_time.raw_times[0] * 1000 / repeat_times):.2f}, {(cpp_scalar_time.raw_times[0] * 1000 / repeat_times):.2f}, {(cpp_simd_time.raw_times[0] * 1000 / repeat_times):.2f}")