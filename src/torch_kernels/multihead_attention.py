from torch import nn, Tensor, rand, float64, float32, float16, bfloat16
from torch.nn import Module, Conv2d, MultiheadAttention, Linear, Embedding, Transformer
import torch.nn.functional as F

def get_multihead_attention(embed_dim=8, num_heads=4):
    return MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

if __name__ == "__main__":
    attention = get_multihead_attention(embed_dim=8, num_heads=4)

    B = 16
    S = 100
    E = 8
    Q = rand(size=(B, S, E), dtype=float32)
    K = rand(size=(B, S, E), dtype=float32)
    V = rand(size=(B, S, E), dtype=float32)
    random_tensor = Tensor(16, 1000, 8)
    result = attention.forward(query=Q, key=K, value=V, need_weights=False)
    print(result[0].shape)