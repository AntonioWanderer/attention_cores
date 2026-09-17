import torch
import numpy
from src_python import attention_refference
from build import attention_cpp

def test_multihead_attention_scalar():

    B, S, E, H = 21, 123, 8, 4

    input_tensor = torch.randn(B, S, E)
    Wq = torch.randn(E, E)
    Wk = torch.randn(E, E)
    Wv = torch.randn(E, E)

    refference_result = attention_refference.attention_refference(input_tensor=input_tensor,
                                                                  Wq=Wq,
                                                                  Wk=Wk,
                                                                  Wv=Wv, 
                                                                  num_heads=H)
    scalar_cpp_result = attention_cpp.attention(input_tensor.numpy(), 
                                                Wq.numpy(), 
                                                Wk.numpy(), 
                                                Wv.numpy(), 
                                                H)
    scalar_cpp_result = torch.tensor(scalar_cpp_result).view(B, S, E)

    print(f"Max abs diff {(refference_result - scalar_cpp_result).abs().max()}")

    assert torch.allclose(input=refference_result, other=scalar_cpp_result, rtol=1e-6)
    