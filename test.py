# test.py (最终修复版 - 请完整替换)

import os
import option
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, average_precision_score
import numpy as np
import torch
import math
from tSNE import batch_tsne


def avce_test(dataloader, model_av, model_v, gt, e, device):
    args = option.parser.parse_args()
    with torch.no_grad():
        model_av.eval()
        pred = torch.zeros(0).to(device)
        if model_v is not None:
            model_v.eval()
            pred3 = torch.zeros(0).to(device)

        cur_index = 0
        for i, (f_v, f_a) in enumerate(dataloader):

            save_dir = "/home/stu2023/xs/project/MACIL_SD-main/fig/anomaly_maps"
            os.makedirs(save_dir, exist_ok=True)

            f_v, f_a = f_v.to(device), f_a.to(device)
            _, _, _, av_logits, audio_rep, visual_rep = model_av(f_a, f_v, seq_len=None)
            av_logits = torch.squeeze(av_logits)
            av_logits = torch.sigmoid(av_logits)
            av_logits = torch.mean(av_logits, 0)
            pred = torch.cat((pred, av_logits))

            # --- 这部分画图和tSNE的代码可以保持原样 ---
            if i == 10000:
                # ... (省略未改变的代码)
                pass

            if model_v is not None:
                v_logits = model_v(f_v, seq_len=None)
                v_logits = torch.squeeze(v_logits)
                v_logits = torch.sigmoid(v_logits)
                v_logits = torch.mean(v_logits, 0)
                pred3 = torch.cat((pred3, v_logits))

        # --- 【修复逻辑从这里开始】 ---

        # 1. 处理主模型 (model_av) 的预测
        pred = pred.cpu().detach().numpy()
        pred_frames = np.repeat(pred, 16)

        # 2. 根据预测长度，切片真值标签 (只做一次)
        num_pred_frames = len(pred_frames)
        gt_sliced = list(gt)[:num_pred_frames]

        # 3. 计算主模型的指标
        precision, recall, th = precision_recall_curve(gt_sliced, pred_frames)
        av_auc = auc(recall, precision)
        av_ap = average_precision_score(gt_sliced, pred_frames)

        # 4. 如果视觉模型存在，处理它的预测
        if model_v is not None:
            # 4.1 处理视觉模型的预测
            pred3 = pred3.cpu().detach().numpy()
            pred3_frames = np.repeat(pred3, 16)

            # 4.2 【核心修复】使用上面已经切片好的 gt_sliced 来计算指标
            precision, recall, th = precision_recall_curve(gt_sliced, pred3_frames)
            v_auc = auc(recall, precision)
            v_ap = average_precision_score(gt_sliced, pred3_frames)

            return av_auc, v_auc, av_ap, v_ap
        else:
            return av_auc, _, av_ap, _
