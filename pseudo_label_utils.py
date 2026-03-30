import torch
import torch.nn as nn
import numpy as np


# ----------------- 核心函数：高斯核挖掘 (与GlanceVAD相同) -----------------
def  gaussian_kernel_mining(scores, point_labels, alpha=0.9):
    """
    以point_labels为种子，根据scores向外挖掘伪异常点。

    Args:
        scores (Tensor): 模型预测的片段级异常分数, shape [B, T].
        point_labels (Tensor): 种子点标签 (0/1向量), shape [B, T].
        alpha (float): 挖掘阈值的比例.

    Returns:
        Tensor: 扩展后的密集二进制伪标签, shape [B, T].
    """
    batch_size, seq_len = scores.shape
    device = scores.device

    # .clone() 是必须的，否则会修改原始的point_labels
    abn_snippets = point_labels.clone().detach().to(device)

    for b in range(batch_size):
        # 找到当前样本的种子点
        abn_indices = torch.nonzero(point_labels[b]).squeeze(1)
        if len(abn_indices) == 0:
            continue

        # 对每个种子点独立进行双向挖掘
        for seed_idx in abn_indices:
            # 计算动态阈值
            abn_thresh = alpha * scores[b, seed_idx]

            # ---> 向左挖掘
            for j in range(seed_idx - 1, -1, -1):
                if scores[b, j] >= abn_thresh:
                    abn_snippets[b, j] = 1
                else:
                    break  # 停止向左挖掘

            # ---> 向右挖掘
            for j in range(seed_idx + 1, seq_len):
                if scores[b, j] >= abn_thresh:
                    abn_snippets[b, j] = 1
                else:
                    break  # 停止向右挖掘

    return abn_snippets
def asymmetric_kernel_splatting(point_labels, seq_len, b_left=0.1, b_right=0.1):
    """
    使用非对称拉普拉斯核在归一化坐标系 [-1, 1] 中生成平滑伪标签。
    这是推荐使用的高效、鲁棒的版本。
    Args:
        point_labels (Tensor): 密集的二进制伪标签, shape [B, T].
        seq_len (int): 序列长度 T.
        b_left (float): 在归一化坐标系中，种子点左侧的衰减尺度。
                        值越小，衰减越快。
        b_right (float): 在归一化坐标系中，种子点右侧的衰减尺度。
                         值越小，衰减越快。

    Returns:
        Tensor: 平滑的软伪标签, shape [B, T], 值域在 [0, 1]。
    """
    batch_size = point_labels.shape[0]
    device = point_labels.device
    rendered_scores = torch.zeros(batch_size, seq_len, device=device)
    # 1. 创建归一化坐标系：将 [0, T-1] 映射到 [-1, 1]
    t_coords_abs = torch.arange(seq_len, device=device, dtype=torch.float32)
    t_coords_norm = 2 * t_coords_abs / (seq_len - 1) - 1
    for b in range(batch_size):
        # 获取当前样本的异常点绝对索引
        seed_indices_abs = torch.nonzero(point_labels[b]).squeeze(1)
        if len(seed_indices_abs) == 0:
            continue
        # 2. 将异常点的索引也归一化到 [-1, 1]
        seed_indices_norm = 2 * seed_indices_abs.float() / (seq_len - 1) - 1
        # 3. 在归一化坐标系中高效地计算距离
        # 使用unsqueeze在不同维度广播，避免循环
        # norm_time_diff shape: [num_seeds, T]
        norm_time_diff = t_coords_norm.unsqueeze(0) - seed_indices_norm.unsqueeze(1)
        # 4. 在归一化距离上应用非对称指数衰减核
        # 左侧权重 (norm_time_diff < 0)
        left_weights = torch.exp(norm_time_diff / b_left)
        # 右侧权重 (norm_time_diff > 0)
        right_weights = torch.exp(-norm_time_diff / b_right)
        # 5. 合并左右权重
        weights = torch.where(norm_time_diff < 0, left_weights, torch.zeros_like(norm_time_diff))
        weights = torch.where(norm_time_diff > 0, right_weights, weights)
        weights = torch.where(norm_time_diff == 0, torch.ones_like(norm_time_diff), weights) # 种子点处为1
        # 6. 融合所有曲线：取最大值
        fused_weights, _ = torch.max(weights, dim=0)
        rendered_scores[b, :] = fused_weights
    return rendered_scores
# 非对称核溅射
# def asymmetric_kernel_splatting(point_labels, seq_len, b_left=2.0, b_right=5.0):
#     """
#     使用非对称的拉普拉斯核 (指数衰减) 生成平滑伪标签。
#     b_left/b_right 控制衰减速度，值越小衰减越快。
#
#     Args:
#         point_labels (Tensor): 密集的二进制伪标签, shape [B, T].
#         seq_len (int): 序列长度 T.
#         b_left (float): 种子点左侧的衰减因子.
#         b_right (float): 种子点右侧的衰减因子.
#
#     Returns:
#         Tensor: 平滑的软伪标签, shape [B, T].
#     """
#     batch_size = point_labels.shape[0]
#     device = point_labels.device
#
#     rendered_scores = torch.zeros(batch_size, seq_len, device=device)
#     t_coords = torch.arange(seq_len, device=device, dtype=torch.float32)
#
#     for b in range(batch_size):
#         seed_indices = torch.nonzero(point_labels[b]).squeeze(1)
#         if len(seed_indices) == 0:
#             continue
#
#         # 为每个种子点生成一条非对称曲线
#         # 使用unsqueeze在不同维度广播，避免循环
#         time_diff = t_coords.unsqueeze(0) - seed_indices.unsqueeze(1)  # shape: [num_seeds, T]
#
#         # 左侧权重 (time_diff < 0)
#         left_weights = torch.exp(time_diff / b_left)
#         # 右侧权重 (time_diff > 0)
#         right_weights = torch.exp(-time_diff / b_right)
#
#         # 合并左右权重
#         weights = torch.where(time_diff < 0, left_weights, torch.zeros_like(time_diff))
#         weights = torch.where(time_diff > 0, right_weights, weights)
#         weights = torch.where(time_diff == 0, torch.ones_like(time_diff), weights)  # 种子点处为1
#
#         # 融合所有曲线：取最大值
#         fused_weights, _ = torch.max(weights, dim=0)
#         rendered_scores[b, :] = fused_weights
#
#     return rendered_scores

