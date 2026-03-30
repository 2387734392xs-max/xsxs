import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class MultimodalRepresentationAlignment(nn.Module):
    """
    使用匈牙利算法实现多模态表示对齐 (MRA) 模块。
    这个模块将音视频特征序列进行显式的时序对齐。
    """
    def __init__(self):
        super(MultimodalRepresentationAlignment, self).__init__()

    def forward(self, f_v, f_a):
        """
        前向传播函数。

        Args:
            f_v (torch.Tensor): 视觉特征序列, shape: (B, T, D)
            f_a (torch.Tensor): 音频特征序列, shape: (B, T, D)
                                B: batch_size, T: sequence_length, D: feature_dimension

        Returns:
            torch.Tensor: 保持不变的视觉特征序列, shape: (B, T, D)
            torch.Tensor: 经过对齐重排后的音频特征序列, shape: (B, T, D)
        """
        device = f_v.device
        batch_size = f_v.size(0)

        # 1. 计算相似度矩阵 S
        # 为了计算余弦相似度，首先对特征进行 L2 归一化
        f_v_norm = F.normalize(f_v, p=2, dim=-1)
        f_a_norm = F.normalize(f_a, p=2, dim=-1)

        # 使用批处理矩阵乘法计算整个批次的相似度矩阵
        # S 的 shape 为 (B, T_v, T_a)，这里 T_v = T_a = T
        similarity_matrix = torch.bmm(f_v_norm, f_a_norm.transpose(1, 2))
        # 返回对齐后的特征对（保持视觉特征顺序不变，重排音频特征）
        return f_v, f_a,similarity_matrix
