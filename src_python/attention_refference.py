import torch
import math
from torch import Tensor

def attention_refference(input_tensor: Tensor, 
                         Wq: Tensor, 
                         Wk: Tensor, 
                         Wv: Tensor, 
                         num_heads: int):
    assert len(input_tensor.shape) == 3
    assert len(Wq.shape) == 2
    assert len(Wk.shape) == 2
    assert len(Wv.shape) == 2

    B, S, E = input_tensor.shape

    assert Wq.shape[0] == Wq.shape[1] == E
    assert Wk.shape[0] == Wk.shape[1] == E
    assert Wv.shape[0] == Wv.shape[1] == E

    assert E % num_heads == 0

    dh = E // num_heads

    Q = input_tensor @ Wq
    K = input_tensor @ Wk
    V = input_tensor @ Wv

    Q = Q.view(B, S, num_heads, dh).transpose(1, 2)
    K = K.view(B, S, num_heads, dh).transpose(1, 2)
    V = V.view(B, S, num_heads, dh).transpose(1, 2)

    scores = Q @ K.transpose(-2 , -1) / math.sqrt(dh)

    attention = torch.softmax(scores, dim=-1)
    output = attention @ V

    return output.transpose(1, 2).reshape(B, S, E)