import torch.utils.data as data
import numpy as np
from utils0 import process_feat, process_test_feat
import os
import math  # <-- 确保已导入


# 定义一个自定义数据集类，继承自 PyTorch 的 data.Dataset 类
class Dataset(data.Dataset):
    def __init__(self, args, transform=None, test_mode=False):
        """
        类的初始化方法...
        """
        self.args = args
        self.modality = args.modality

        if test_mode:
            self.rgb_list_file = args.test_rgb_list
            self.audio_list_file = args.test_audio_list
        else:
            self.rgb_list_file = args.rgb_list
            self.audio_list_file = args.audio_list

        self.max_seqlen = args.max_seqlen
        self.transform = transform
        self.test_mode = test_mode
        self.normal_flag = '_label_A'
        self._parse_list()

    def _parse_list(self):
        """
        解析 RGB 和音频数据列表文件，并根据模式(训练/测试)和比例参数进行缩减。
        """
        # 1. 首先，总是将完整的数据列表读入临时变量
        full_rgb_list = list(open(self.rgb_list_file))
        full_audio_list = list(open(self.audio_list_file))

        # 2. 根据是训练模式还是测试模式，执行不同的缩减逻辑
        if self.test_mode:
            # --- 【这是为测试集新增的逻辑】 ---
            # 检查是否需要缩减测试集
            if hasattr(self.args, 'test_data_ratio') and self.args.test_data_ratio < 1.0:
                ratio = self.args.test_data_ratio

                # 计算并截取测试集的 RGB 和 Audio 列表
                num_rgb_samples = math.ceil(len(full_rgb_list) * ratio)
                self.list = full_rgb_list[:num_rgb_samples]

                num_audio_samples = math.ceil(len(full_audio_list) * ratio)
                self.audio_list = full_audio_list[:num_audio_samples]

                print(f"\n--- Using only the first {ratio * 100:.0f}% of TEST data ---")
                print(f"Full test set size: RGB={len(full_rgb_list)}, Audio={len(full_audio_list)}")
                print(f"Using {len(self.list)} RGB samples and {len(self.audio_list)} Audio samples for testing.\n")
            else:
                # 否则，使用完整的测试集
                self.list = full_rgb_list
                self.audio_list = full_audio_list
        else:
            # --- 【这是之前为训练集写的逻辑，保持不变】 ---
            # 检查是否需要缩减训练集
            if hasattr(self.args, 'train_data_ratio') and self.args.train_data_ratio < 1.0:
                ratio = self.args.train_data_ratio

                # 计算并截取训练集的 RGB 和 Audio 列表
                num_rgb_samples = math.ceil(len(full_rgb_list) * ratio)
                self.list = full_rgb_list[:num_rgb_samples]

                num_audio_samples = math.ceil(len(full_audio_list) * ratio)
                self.audio_list = full_audio_list[:num_audio_samples]

                print(f"\n--- Using only the first {ratio * 100:.0f}% of TRAINING data ---")
                print(f"Full training set size: RGB={len(full_rgb_list)}, Audio={len(full_audio_list)}")
                print(f"Using {len(self.list)} RGB samples and {len(self.audio_list)} Audio samples for training.\n")
            else:
                # 否则，使用完整的训练集
                self.list = full_rgb_list
                self.audio_list = full_audio_list

    def __getitem__(self, index):
        """
        根据给定的索引获取数据集中的一个样本。 (无需修改)
        """
        if self.normal_flag in self.list[index]:
            label = 0.0
        else:
            label = 1.0

        f_v = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
        audio_index = index // 5
        f_a = np.array(np.load(self.audio_list[audio_index].strip('\n')), dtype=np.float32)

        if self.transform is not None:
            f_v = self.transform(f_v)
            f_a = self.transform(f_a)

        if self.test_mode:
            return f_v, f_a
        else:
            f_v = process_feat(f_v, self.max_seqlen, is_random=False)
            f_a = process_feat(f_a, self.max_seqlen, is_random=False)
            return f_v, f_a, label

    def __len__(self):
        """
        返回数据集的样本数量。(无需修改)
        """
        return len(self.list)

