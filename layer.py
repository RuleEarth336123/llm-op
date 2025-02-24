import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  

    def forward(self, x):
        return self.embedding(x)  


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
        # scores 的形状: (batch_size, heads, seq_len, seq_len)
        # 将掩码为0的位置的分数设置为一个很大的值（1e9），以便在softmax中被忽略
        if mask is not None:
            mask = mask.unsqueeze(1)  # mask 的形状: (batch_size, 1, seq_len)
            scores = scores.masked_fill(mask == 0, 1e9) 
            # scores 的形状保持不变: (batch_size, heads, seq_len, seq_len)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        # output 的形状: (batch_size, heads, seq_len, d_k)
        return output

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k)
        # k 的形状: (batch_size, seq_len, heads, d_k)

        q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k)

        k = k.transpose(1, 2)  # k 的形状: (batch_size, heads, seq_len, d_k)
        q = q.transpose(1, 2)  # q 的形状: (batch_size, heads, seq_len, d_k)
        v = v.transpose(1, 2)  # v 的形状: (batch_size, heads, seq_len, d_k)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        # scores 的形状: (batch_size, heads, seq_len, d_k)

        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # concat 的形状: (batch_size, seq_len, d_model)

        output = self.out(concat)
        # output 的形状: (batch_size, seq_len, d_model)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
'''
    layernorm(x) = alpha * (x - 均值) / 方差 + bias
'''
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / \
               (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

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

def test_feed_forward():
    d_model = 512  
    d_ff = 2048    
    batch_size = 2  
    seq_len = 10    

    ff = FeedForward(d_model, d_ff)

    input_tensor = torch.rand(batch_size, seq_len, d_model)  # 输入形状: (batch_size, seq_len, d_model)

    output_tensor = ff(input_tensor)

    assert output_tensor.shape == (batch_size, seq_len, d_model), "输出形状不匹配"
    print("测试通过，输出形状匹配。")

def test_norm():
    d_model = 512 
    batch_size = 2  
    seq_len = 10    

    norm_layer = Norm(d_model)

    input_tensor = torch.rand(batch_size, seq_len, d_model)  # 输入形状: (batch_size, seq_len, d_model)

    output_tensor = norm_layer(input_tensor)

    assert output_tensor.shape == (batch_size, seq_len, d_model), "输出形状不匹配"
    print("测试通过，输出形状匹配。")

# test_positional_encoder()
# test_multi_head_attention()
# 运行测试
# test_feed_forward()
# test_norm()