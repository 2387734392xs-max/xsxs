import os
import json
import numpy as np
import matplotlib.pyplot as plt


def anomap(predict_dict, dataset, save_root):
    print("--- 进入 anomap 函数 ---")
    plot_dir = os.path.join(save_root, 'plot', dataset)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"1. 绘图目录准备就绪: {plot_dir}")

    json_root = None
    if 'XD-Violence' in dataset:
        json_root = 'xd_test_gt.json'

    if json_root is None:
        print(f"[错误] 未知的 dataset 名称: '{dataset}'。无法确定要加载哪个JSON文件。脚本终止。")
        return

    json_path = os.path.join(save_root, json_root)
    print(f"2. 准备加载 JSON GT 文件，完整路径: {os.path.abspath(json_path)}")

    if not os.path.exists(json_path):
        print(f"[错误] JSON GT 文件未找到! 路径: {json_path}。脚本终止。")
        return

    with open(file=json_path, mode='r', encoding='utf-8') as f:
        label_dict = json.load(f)

    print(f"3. 成功加载 JSON 文件。共找到 {len(label_dict)} 个键。")
    if label_dict:
        first_key = list(label_dict.keys())[0]
        print(f"   JSON中的第一个键示例: '{first_key}'")

    print("\n--- 开始循环绘图 ---")
    plotted_count = 0
    skipped_count = 0
    for i, item in enumerate(predict_dict):
        k, v = item['file_name'], item['pre_dict']

        if k not in label_dict:
            if skipped_count < 5:
                print(f"   [警告] 视频名 '{k}' 在JSON文件中未找到，跳过此视频。")
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

        plt.title(k, fontsize=10)
        plt.xlabel('Frames')
        plt.ylabel('Anomaly Score')
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(0, 1.1, step=0.2))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        plt.savefig(os.path.join(plot_dir, f"{i:04d}_{k}.png"))
        plotted_count += 1
        plt.close()

    print(f"\n--- 绘图循环结束 ---")
    print(f"成功生成 {plotted_count} 张图像。")
    if skipped_count > 0:
        print(f"因在JSON文件中找不到键而跳过了 {skipped_count} 个视频。")
