
import os
import subprocess
import itertools
import datetime
import csv
import re
import sys # <-- 已包含环境问题修复
# 新增: 导入 multiprocessing 模块
from multiprocessing import Pool, Manager

# --- 1. 定义你要测试的超参数网格 (无需修改) ---
param_grid = {
    # 同时测试 0.5 和 0.6 这两个最有希望的 a2b 值
    'lamda_a2b': [1.4],
    'lamda_a2n': [0.5],
    # 测试所有出现过好成绩的 a2n 值
    'lamda_v': [0.1],
    'lamda_a': [0.1],
    'lamda_cof': [0.05],
    'warmup-epochs': [35],  # 没
    # --- 变量2: 探索 LR 在 Warmup 影响下的最佳点 ---
    'lr': [0.0002],
    'seed': [2026],
    'pseudo-warmup-epochs':[15],
    'lamda_pseudo':[0.05],
    'pseudo-alpha':[0.8],
    'pseudo-b_left':[1,0.1,0.01,0.001,0.0001],#0.005, 0.01, 0.02
    'pseudo-b_right':[1,0.1,0.01,0.001,0.0001],
    'pseudo_ramp_up_epochs':[15],
    'dropoutGCN':[0.45],
    'dropout0':[0.1],
    'dim':[32]
}

# --- 2. 定义其他固定的训练参数 ---
# 从你的 option.py 文件中提取的固定参数
fixed_args = [
    # '--gpus' 将由并行逻辑动态分配
    '--modality', 'MIX2',
    '--model', 'HyboNet',
    '--manifold', 'Lorentz',
    #'--lr', '0.000374',
    '--batch-size', '128',
    '--workers', '8',
    '--max-epoch', '50', # 你可以根据需要调整总的epoch数
    # '--model-name' 将被动态生成，所以从这里移除
    '--rgb-list', 'list/rgb.list',
    '--audio-list', 'list/audio.list',
    '--test-rgb-list', 'list/rgb_test.list',
    '--test-audio-list', 'list/audio_test.list',
    '--gt', 'list/gt.npy',
    '--hid_dim', '128',
    '--ffn_dim', '128',
    '--num_stages', '3',
    '--nhead', '4',
    '--dim', '32',
    # 其他固定参数...
]
# ==============================================================================
# --- 3. 并行设置 (这是你需要配置的地方) ---
# ==============================================================================

# 【需要你设置】设置你想同时进行的进程个数
# 例如，如果你有4块GPU，并且希望每个实验占用一块GPU，可以设置为 4
# 如果你只有一块GPU，但想看看能否同时跑两个（不推荐，会抢占资源），可以设置为 2
# 注意：这个数字不应超过你可用GPU的数量，否则进程会等待GPU资源
num_parallel_processes =4 # <--- 在这里设置并行进程数

# 【需要你设置】列出你可用的 GPU ID
# 例如: ['0'], ['0', '1'], ['0', '1', '2', '3']
available_gpus = ['0','0','0','0']  # <--- 在这里列出所有可用的GPU ID

# ==============================================================================

# --- 4. 实验准备 (这部分无需修改) ---
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
base_run_dir = os.path.join('tuning_runs', timestamp)
os.makedirs(base_run_dir, exist_ok=True)

results_csv_path = os.path.join(base_run_dir, 'summary_results.csv')
param_keys = list(param_grid.keys())
csv_header = ['trial_id', 'gpu_id'] + param_keys + ['best_auc', 'best_ap', 'best_epoch']

with open(results_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)


# --- 5. 定义单个实验的执行函数 (这是并行化的核心) ---
def run_trial(args_bundle):
    # 解包传入的参数
    i, params, base_run_dir, gpu_queue = args_bundle

    # 从队列中获取一个可用的 GPU ID
    gpu_id = gpu_queue.get()

    trial_id = f"trial_{i + 1:03d}"
    print(f"\n{'=' * 20}\nStarting {trial_id} on GPU {gpu_id}: {params}\n{'=' * 20}")

    trial_dir = os.path.join(base_run_dir, trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    # 动态构建命令
    python_executable = sys.executable  # 获取当前Python解释器的完整路径
    command = [python_executable, 'main.py'] + fixed_args
    # 为当前进程分配GPU
    command.extend(['--device', f'cuda:{gpu_id}'])

    for key, value in params.items():
        command.extend([f'--{key}', str(value)])

    log_path = os.path.join(trial_dir, 'training.log')
    checkpoint_path = os.path.join(trial_dir, 'ckpt')
    command.extend(['--log_path', log_path])
    command.extend(['--checkpoint_path', checkpoint_path])
    param_values = list(params.values())
    # 默认的结果行，以防实验失败
    result_row = [trial_id, gpu_id] + param_values + ['FAILED', 'FAILED', 'FAILED']

    try:
        # 运行子进程并捕获输出
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8')

        # 将输出实时写入日志文件
        with open(log_path, 'w') as log_file:
            for line in process.stdout:
                print(f"[{trial_id} on GPU {gpu_id}] {line.strip()}")
                log_file.write(line)

        process.wait()  # 等待进程结束
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        # 实验成功后，从日志文件中提取最终结果
        with open(log_path, 'r') as log_file:
            full_log = log_file.read()

        best_auc, best_ap, best_epoch = 'N/A', 'N/A', 'N/A'
        #match = re.search(r"Best Performance in Epoch (\d+): av_auc:([\d.]+)\|av_ap:([\d.]+)", full_log)
        # tune.py, line ~304
        match = re.search(r"Best Performance in Epoch (\d+): av_auc:([\d.]+).*av_ap:([\d.]+)", full_log)

        if match:
            best_epoch = match.group(1)
            best_av_auc = f"{float(match.group(2)):.6f}"
            best_av_ap = f"{float(match.group(3)):.6f}"  # 新：av_ap 现在是第3个捕获组
            result_row = [trial_id, gpu_id] + param_values + [best_av_auc, best_av_ap, best_epoch]
        else:
            print(f"!!!!!! {trial_id} on GPU {gpu_id} finished, but couldn't parse the result line! !!!!!!")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"!!!!!! {trial_id} on GPU {gpu_id} FAILED !!!!!!\nError: {e}")
    finally:
        # 无论成功还是失败，都必须将 GPU ID 放回队列，以便其他进程使用
        gpu_queue.put(gpu_id)
        print(f"--- Finished {trial_id} on GPU {gpu_id}. GPU {gpu_id} is now free. ---")
        return result_row


# --- 6. 生成并运行所有实验 (并行) ---
if __name__ == '__main__':
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"将要进行 {len(param_combinations)} 次实验...")
    print(f"使用 {num_parallel_processes} 个并行进程.")
    print(f"可用 GPUs: {available_gpus}")
    print(f"所有结果将保存在: {base_run_dir}")

    # 创建一个管理器和GPU队列，用于在进程间安全地共享GPU ID
    manager = Manager()
    gpu_queue = manager.Queue()
    for gpu in available_gpus:
        gpu_queue.put(gpu)

    # 准备传递给 run_trial 函数的参数列表
    tasks = [(i, params, base_run_dir, gpu_queue) for i, params in enumerate(param_combinations)]

    # 创建进程池并开始执行
    with Pool(processes=num_parallel_processes) as pool:
        # pool.map 会阻塞，直到所有任务完成
        results = pool.map(run_trial, tasks)

    # 所有实验完成后，将结果写入CSV
    with open(results_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # 对结果按 trial_id 排序，使其看起来更整洁
        results.sort(key=lambda x: int(x[0].split('_')[1]))
        writer.writerows(results)

    print(f"\n所有实验完成！总结报告见: {results_csv_path}")


