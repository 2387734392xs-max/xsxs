import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import json

# 必须在导入 pyplot 之前设置后端，以在无图形界面的服务器上运行
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入项目中的模块
import option
from avce_network import AVCE_Model as Model
from avce_dataset import Dataset
from sklearn.metrics import average_precision_score


def anomap(predict_dict, dataset, save_root):
    plot_dir = os.path.join(save_root, 'plot', dataset)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"图像将保存到: {os.path.abspath(plot_dir)}")

    json_root = None
    if 'XD-Violence' in dataset:
        json_root = 'xd_test_gt.json'

    if json_root is None:
        print(f"[错误] 未知的 dataset 名称: '{dataset}'。脚本终止。")
        return

    json_path = os.path.join(save_root, json_root)
    if not os.path.exists(json_path):
        print(f"[致命错误] JSON GT 文件未找到! 请确保 '{json_root}' 就在你运行脚本的目录下。脚本终止。")
        return

    with open(file=json_path, mode='r', encoding='utf-8') as f:
        label_dict = json.load(f)

    print(f"成功加载 JSON 文件, 包含 {len(label_dict)} 个视频的标签。\n")

    plotted_count = 0
    skipped_count = 0
    for i, item in enumerate(predict_dict):
        k, v = item['file_name'], item['pre_dict']

        if k not in label_dict:
            if skipped_count < 5:
                print(f"   [警告] 视频键 '{k}' 在JSON文件中未找到，跳过此视频。")
            elif skipped_count == 5:
                print("   [警告] 发现更多不匹配的键，后续警告将不再显示...")
            skipped_count += 1
            continue

        predict_np = np.repeat(v.squeeze(-1).cpu().numpy(), 16)
        label_np = np.array(label_dict[k]['labels'])

        min_len = min(len(predict_np), len(label_np))
        x = np.arange(min_len)

        plt.figure(figsize=(12, 4))
        plt.plot(x, predict_np[:min_len], color='blue', linewidth=1.5, label='Anomaly Score')
        plt.fill_between(x, 0, label_np[:min_len], where=label_np[:min_len] > 0.5, facecolor="red", alpha=0.3,
                         label='Ground Truth')

        plt.title(k, fontsize=8)
        plt.xlabel('Frames')
        plt.ylabel('Anomaly Score')
        plt.ylim(-0.05, 1.05)
        plt.yticks(np.arange(0, 1.1, step=0.2))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # --- 【核心修改】 ---
        # 1. 创建一个对文件名安全的字符串
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        safe_filename = k
        for char in invalid_chars:
            safe_filename = safe_filename.replace(char, '_')

        # 2. 使用 索引 + 安全的视频名 来保存图像
        # 加上索引号可以确保文件按处理顺序排序
        plt.savefig(os.path.join(plot_dir, f"{i:04d}_{safe_filename}.png"))
        # --- 【修改结束】 ---

        plotted_count += 1
        plt.close()

    print(f"\n--- 绘图完成 ---")
    print(f"成功生成 {plotted_count} 张图像。")
    if skipped_count > 0:
        print(f"因在JSON文件中找不到键而跳过了 {skipped_count} 个视频。")


def run_inference_and_generate_plots(args):
    """
    一个完整的脚本：加载模型，执行推理，计算AP，并为每个视频生成异常分数图。
    """
    print("--- 启动主流程 ---")
    device = torch.device(args.device)
    print(f"1. 使用设备: {device}")

    try:
        gt = np.load(args.gt)
        print(f"2. 成功加载 Ground Truth (gt.npy) from: {args.gt}")
    except FileNotFoundError:
        print(f"[错误] gt.npy 文件未找到: '{args.gt}'. 无法继续。")
        return

    model = Model(args).to(device)
    try:
        model.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(args.model_dir, map_location=device).items()}
        )
        print(f"3. 成功加载模型权重 from: {args.model_dir}")
    except Exception as e:
        print(f"[错误] 加载模型权重失败: {e}")
        return
    model.eval()

    test_dataset = Dataset(args, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"4. 成功创建数据加载器，共 {len(test_loader)} 个批次。")

    all_snippet_scores_for_ap = []
    predictions_for_plotting = []

    print("\n--- 开始推理循环 ---")
    start_time = time.time()

    with torch.no_grad():
        for i, (f_v, f_a) in enumerate(test_loader):
            video_path = test_dataset.list[i * 5].strip()

            base_name = os.path.basename(video_path)
            name_without_ext = os.path.splitext(base_name)[0]
            video_name_key = name_without_ext.rsplit('__', 1)[0]

            f_v, f_a = f_v.to(device), f_a.to(device)
            _, _, _, av_logits, _, _ = model(f_a, f_v, seq_len=None)

            av_logits = torch.squeeze(av_logits)
            av_logits = torch.sigmoid(av_logits)
            av_scores_snippet = torch.mean(av_logits, 0)

            all_snippet_scores_for_ap.extend(av_scores_snippet.cpu().numpy())

            predictions_for_plotting.append({
                'file_name': video_name_key,
                'pre_dict': av_scores_snippet
            })

    print(f"--- 推理循环结束, 耗时 {time.time() - start_time:.2f} 秒 ---")
    print(f"共处理了 {len(predictions_for_plotting)} 个视频片段。")

    print("\n--- 计算性能指标 ---")
    all_frame_scores = np.repeat(np.array(all_snippet_scores_for_ap), 16)
    min_len = min(len(all_frame_scores), len(gt))
    ap_score = average_precision_score(gt[:min_len], all_frame_scores[:min_len])

    print('--- Overall Performance ---')
    print(f'Average Precision (AP): {ap_score:.4f}')
    print('---------------------------\n')

    print("--- 开始调用绘图函数 anomap ---")
    anomap(predictions_for_plotting, args.dataset_name, '.')
    print("\n--- 脚本运行完毕 ---")


if __name__ == '__main__':
    args = option.parser.parse_args()
    run_inference_and_generate_plots(args)
