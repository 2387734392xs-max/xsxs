import torch
# 从CMA_MIL模块中导入CMAL函数，该函数可能用于计算某种特定的损失
from CMA_MIL import CMAL
import math
import pseudo_label_utils as plu
import torch.nn.functional as F
# 定义avce_train函数，用于训练音频 - 视觉融合模型（model_av）和视觉模型（model_v）
# 参数说明：
# dataloader: 数据加载器，用于加载训练数据
# model_av: 音频 - 视觉融合模型
# model_v: 视觉模型
# optimizer_av: 音频 - 视觉融合模型的优化器
# optimizer_v: 视觉模型的优化器
# criterion: 损失函数，用于计算损失
# lamda_a2b: 超参数，用于加权某些损失项
# lamda_a2n: 超参数，用于加权某些损失项
# logger: 日志记录器，用于记录训练过程中的信息
def avce_train(dataloader, model_av, model_v, optimizer_av, optimizer_v, criterion,lamda_a2b, lamda_a2n, lamda_v, lamda_a,lamda_pseudo, current_epoch, args,logger, device):
    # 启用梯度计算，确保在训练过程中可以进行反向传播
    with torch.set_grad_enabled(True):
        # 将音频 - 视觉融合模型设置为训练模式，开启一些在训练时需要的特殊层（如Dropout和BatchNorm）
        model_av.train()
        # 将视觉模型设置为训练模式
        model_v.train()
        # 遍历数据加载器中的每个批次
        total_pseudo_loss = 0.0
        for i, (f_v, f_a, label) in enumerate(dataloader):
            # 计算每个样本的有效序列长度
            # 首先计算f_v在维度2上的绝对值的最大值，然后判断哪些值大于0，最后在维度1上求和得到每个样本的有效长度
            seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)
            # 根据计算得到的有效序列长度，截取f_v的有效部分
            f_v = f_v[:, :torch.max(seq_len), :]
            # 根据计算得到的有效序列长度，截取f_a的有效部分
            f_a = f_a[:, :torch.max(seq_len), :]
            # 将视觉特征f_v、音频特征f_a和标签label转换为浮点数类型，并移动到GPU上
            f_v, f_a, label = f_v.float().to(device), f_a.float().to(device), label.float().to(device)
            # 调用音频 - 视觉融合模型进行前向传播，获取输出结果
            # mmil_logits: 某种多实例学习的逻辑值
            # audio_logits: 音频的逻辑值
            # visual_logits: 视觉的逻辑值
            # 忽略第四个输出
            # audio_rep: 音频表示
            # visual_rep: 视觉表示
            mmil_logits, audio_logits, visual_logits, _, audio_rep, visual_rep = model_av(f_a, f_v, seq_len)
            # 去除audio_logits中维度为1的维度
            audio_logits = audio_logits.squeeze()
            # 去除visual_logits中维度为1的维度
            visual_logits = visual_logits.squeeze()
            # 去除mmil_logits中维度为1的维度
            mmil_logits = mmil_logits.squeeze()
            # 计算分类损失，使用传入的损失函数criterion
            clsloss = criterion(mmil_logits, label)
            # 调用CMAL函数计算不同的CMA损失
            #cmaloss_a2v_a2b, cmaloss_a2v_a2n, cmaloss_v2a_a2b, cmaloss_v2a_a2n = CMAL(mmil_logits, audio_logits,
            #                                                                          visual_logits, seq_len, audio_rep,
            #                                                                          visual_rep)
            cmaloss_a2v_a2b, cmaloss_a2v_a2n, cmaloss_v2a_a2b, cmaloss_v2a_a2n, cmaloss_v, cmaloss_a = CMAL(mmil_logits,
                                                                                                            audio_logits,
                                                                                                            visual_logits,
                                                                                                            seq_len,
                                                                                                            audio_rep,
                                                                                                            visual_rep)
            # 计算总的损失，将分类损失和各种CMA损失加权求和
            #total_loss = clsloss + lamda_a2b * cmaloss_a2v_a2b + lamda_a2b * cmaloss_v2a_a2b + lamda_a2n * cmaloss_a2v_a2n + lamda_a2n * cmaloss_v2a_a2n
            total_loss = clsloss + lamda_a2b * (cmaloss_a2v_a2b + cmaloss_v2a_a2b) + lamda_a2n * (cmaloss_a2v_a2n + cmaloss_v2a_a2n) + lamda_v * cmaloss_v + lamda_a * cmaloss_a
            if current_epoch >= args.pseudo_warmup_epochs:
                # 1. 筛选出异常样本 (label=1)
                anomaly_mask = (label == 1.0)
                if anomaly_mask.sum() > 0:
                    # 获取异常样本的视觉和音频分数
                    # 我们主要使用更可靠的视觉分数来生成伪一瞥点
                    v_scores_abn = visual_logits[anomaly_mask]

                    # 2. 识别“伪一瞥”点 (Pseudo-Glance)
                    # detach()非常重要，避免梯度回流到这个选择过程
                    idx_max = torch.argmax(v_scores_abn.detach(), dim=-1)
                    pseudo_point_label = torch.zeros_like(v_scores_abn).scatter_(1, idx_max.unsqueeze(1), 1)

                    # 3. 高斯核挖掘
                    # 同样使用 detach() 后的分数进行挖掘
                    dense_pseudo_labels = plu.gaussian_kernel_mining(
                        v_scores_abn.detach(),
                        pseudo_point_label,
                        alpha=args.pseudo_alpha
                    )

                    # 4. 非对称核溅射 (创新点)
                    seq_len_abn = v_scores_abn.shape[1]
                    rendered_scores = plu.asymmetric_kernel_splatting(
                        dense_pseudo_labels,
                        seq_len_abn,
                        b_left=args.pseudo_b_left,
                        b_right=args.pseudo_b_right
                    )
                    # 5. 计算伪标签损失 (Refinement Loss)
                    # 让视觉和音频分支都去学习这个平滑的伪标签
                    loss_pseudo_v = F.binary_cross_entropy(v_scores_abn, rendered_scores)
                    loss_pseudo_a = F.binary_cross_entropy(audio_logits[anomaly_mask], rendered_scores)

                    loss_pseudo = loss_pseudo_v + loss_pseudo_a

                    # 6. 将伪标签损失加入总损失
                    total_loss += lamda_pseudo * loss_pseudo
                    total_pseudo_loss += loss_pseudo.item()
            # 计算数据加载器长度的一半，用于记录日志的间隔
            unit = dataloader.__len__() // 2
            # 每隔unit个批次记录一次日志
            if i % unit == 0:
                # 记录当前的超参数lambda_a2b和lambda_a2n的值
                logger.info(f"Current Lambda_a2b: {lamda_a2b:.2f}, Current Lambda_a2n: {lamda_a2n:.2f}")
                # 记录当前的各种损失值
                #logger.info(
                #    f"{int(i // unit)}/{2} MIL Loss: {clsloss:.4f}, CMA Loss A2V_A2B: {cmaloss_a2v_a2b:.4f}, CMA Loss A2V_A2N: {cmaloss_a2v_a2n:.4f},"
                #    f"CMA Loss V2A_A2B: {cmaloss_v2a_a2b:.4f},  CMA Loss V2A_A2N: {cmaloss_v2a_a2n:.4f}")
                logger.info(
                    f"{int(i // unit)}/{2} MIL Loss: {clsloss:.4f}, "
                    f"CMA_A2V_A2B: {cmaloss_a2v_a2b:.4f}, CMA_A2V_A2N: {cmaloss_a2v_a2n:.4f}, "
                    f"CMA_V2A_A2B: {cmaloss_v2a_a2b:.4f}, CMA_V2A_A2N: {cmaloss_v2a_a2n:.4f}, "
                    f"CMA_V: {cmaloss_v:.4f}, CMA_A: {cmaloss_a:.4f}")
            # 调用视觉模型进行前向传播，获取输出结果
            v_logits = model_v(f_v, seq_len)
            # 计算视觉模型的损失
            loss_v = criterion(v_logits, label)

            # 清空音频 - 视觉融合模型的梯度
            optimizer_av.zero_grad()
            # 清空视觉模型的梯度
            optimizer_v.zero_grad()
            # 允许音频 - 视觉融合模型计算梯度
            model_av.requires_grad = True
            # 禁止视觉模型计算梯度
            model_v.requires_grad = False
            # 对总的损失进行反向传播，计算梯度
            total_loss.backward()
            # 更新音频 - 视觉融合模型的参数
            optimizer_av.step()

            # 再次清空音频 - 视觉融合模型的梯度
            optimizer_av.zero_grad()
            # 再次清空视觉模型的梯度
            optimizer_v.zero_grad()
            # 禁止音频 - 视觉融合模型计算梯度
            model_av.requires_grad = False
            # 允许视觉模型计算梯度
            model_v.requires_grad = True
            # 对视觉模型的损失进行反向传播，计算梯度
            loss_v.backward()
            # 更新视觉模型的参数
            optimizer_v.step()
            avg_pseudo_loss = total_pseudo_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        # 返回总的损失和视觉模型的损失
        return total_loss, loss_v,avg_pseudo_loss