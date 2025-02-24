from layer import Norm,MultiHeadAttention,FeedForward,PositionalEncoder,Embedder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import get_clones

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1) -> None:
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)

        self.ff = FeedForward(d_model, dropout=dropout)


    def forward(self, x, e_outputs, src_mask, trg_mask):
        #掩码自注意力层 (attn_1)：处理目标序列的时序依赖，通过trg_mask屏蔽未来位置信息
        '''
        trg_mask = [
            [1, 0, 0, 0],  # 第1个词只能看到自己
            [1, 1, 0, 0],  # 第2个词能看到第1、2个词
            [1, 1, 1, 0],  # 第3个词能看到第1、2、3个词
            [1, 1, 1, 1],  # 第4个词能看到所有词
        ]
        '''
        attn_output_1 = self.attn_1(x, x, x, trg_mask)
        attn_output_1 = self.dropout_1(attn_output_1)

        x = x + attn_output_1
        x = self.norm_1(x)
        #交叉注意力层 (attn_2)：将编码器输出(e_outputs)作为键值对，与解码器当前状态进行交互
        '''
        src_mask： 用于处理源序列中的填充部分，防止模型在计算注意力时关注无效的填充位置。
        trg_mask： 用于解码器的自注意力机制，确保模型在生成当前词时，只能关注当前词及其之前的词，防止模型提前获取未来的信息。

        主要用于解码器的自注意力机制，
        确保模型在生成当前词时，
        只能关注当前词及其之前的词，而无法看到未来的词。
        这种机制被称为“因果掩码”或“未来信息屏蔽”，用于防止模型在训练时提前获取未来的信息，从而保持自回归生成的特性。
        '''
        attn_output_2 = self.attn_2(x, e_outputs, e_outputs, src_mask)
        attn_output_2 = self.dropout_2(attn_output_2)

        x = x + attn_output_2
        x = self.norm_2(x)
        #前馈网络 (ff)：通过非线性变换增强特征表示
        ff__output = self.ff(x)
        ff__output = self.dropout_3(ff__output)
        #残差连接+层归一化：每个子层后均有x = x + attn_output和norm操作
        x = x + ff__output
        x = self.norm_3(x)
        return x
    
    class Decoder(nn.Module):

        def __init__(self,vocab_size, d_model, N, heads, dropout) -> None:
            super().__init__()
            self.N = N
            self.embed = Embedder(vocab_size, d_model)
            self.pe = PositionalEncoder(d_model)
            self.layers = get_clones(DecoderLayer(d_model, heads, dropout),N)
            self.norm = Norm(d_model)
        
        def forward(self, trg, e_outputs, src_mask, trg_mask):
            x = self.embed(trg)
            x = self.pe(x)
            for i in range(self.N):
                x = self.layers[i](x, e_outputs, src_mask, trg_mask)
            return self.norm(x)
