import torch.nn as nn

def get_clones(module, N):
    return nn.ModuleList([module for _ in range(N)])  # 创建 N 个相同的模块实例