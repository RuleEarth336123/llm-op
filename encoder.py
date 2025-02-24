from layer import Norm,MultiHeadAttention,FeedForward,PositionalEncoder,Embedder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import get_clones

class EncoderLayer(nn.Module):

    def __init__(self, d_model, heads, dropout=0.1) -> None:
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.attn(x, x, x, mask)
        attn_output = self.dropout_1(attn_output)
        x = x + attn_output
        ff__output = self.ff(x)
        ff__output = self.dropout_2(ff__output)
        x = x + ff__output
        x = self.norm_2(x)
        return x
    
class Encoder(nn.Module):

    def __init__(self,vocab_size, d_model, N, heads, dropout) -> None:
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout),N)
        self.norm = Norm(d_model)
    
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x,mask)
        return self.norm(x)

def test_encoder():
    vocab_size = 10000  # 词汇表大小
    d_model = 512       # 嵌入维度
    N = 6               # 编码器层的数量
    heads = 8           # 注意力头的数量
    dropout = 0.1       # dropout 概率
    seq_len = 10        # 输入序列长度
    batch_size = 2      # 批次大小

    # 创建 Encoder 实例
    encoder = Encoder(vocab_size, d_model, N, heads, dropout)

    # 创建随机输入张量（词汇索引）
    src = torch.randint(0, vocab_size, (batch_size, seq_len))  # 输入形状: (batch_size, seq_len)
    mask = torch.ones(batch_size, 1, seq_len)                   # 创建掩码

    # 运行前向传播
    output_tensor = encoder(src, mask)

    # 验证输出形状
    assert output_tensor.shape == (batch_size, d_model), "输出形状不匹配"
    print("测试通过，输出形状匹配。")

# 运行测试
test_encoder()




