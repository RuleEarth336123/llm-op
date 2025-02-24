import numpy as np 
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80) -> None:
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.h = heads
        self.d_k = d_model // heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)
        
        # 将掩码为0的位置的分数设置为一个很大的值（1e9），以便在softmax中被忽略
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, 1e9) 
        # 如果提供了dropout层，则在分数上应用dropout，以防止过拟合
        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

def test_positional_encoder():
    d_model = 512
    max_seq_len = 80
    encoder = PositionalEncoder(d_model, max_seq_len)

    input_tensor = torch.rand(1, max_seq_len, d_model)
    output_tensor = encoder(input_tensor)
    
    assert output_tensor.shape == input_tensor.shape, "输出形状不匹配"
    print("测试通过，输出形状匹配。")

def test_multi_head_attention():
    d_model = 512  
    heads = 8     
    seq_len = 10   
    batch_size = 2 # 批次大小

    mha = MultiHeadAttention(heads, d_model)

    q = torch.rand(batch_size, seq_len, d_model)  
    k = torch.rand(batch_size, seq_len, d_model)  
    v = torch.rand(batch_size, seq_len, d_model)  
    mask = torch.ones(batch_size, 1, seq_len)     

    # 运行前向传播
    output = mha(q, k, v, mask)

    # 验证输出形状
    assert output.shape == (batch_size, seq_len, d_model), "输出形状不匹配"
    print("测试通过，输出形状匹配。")

# test_positional_encoder()
test_multi_head_attention()