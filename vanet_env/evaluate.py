import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 尝试从训练脚本中导入模型类和数据集类
# 确保 dynamic_predictor.py 在同一目录下
try:
    from predict_new import DynamicTransformerPredictor, LocalTrajectoryDataset
    # 如果您在 dynamic_predictor.py 中定义了 env_config，也可以尝试导入
    # from dynamic_predictor import env_config
except ImportError:
    print("错误：无法导入 dynamic_predictor.py。请确保该文件存在且与本脚本在同一目录下。")
    sys.exit(1)


def evaluate_model():
    # --- 1. 配置参数 (必须与训练时完全一致) ---
    # 如果您在训练时修改了这些参数，这里也必须修改
    K_VEHICLES = 5  # 必须与训练时一致
    SEQ_LEN = 10  # 必须与训练时一致
    EMBED_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2

    # 地图和数据参数
    RSU_POSITIONS = np.random.rand(20, 2) * 1000  # 这里仅作占位，评估时如果不重新计算距离，这个参数其实不影响Dataset加载现有CSV
    # 但是 LocalTrajectoryDataset 需要 RSU 位置来找最近的车。
    # ⚠️ 为了准确评估，您必须使用与训练时完全相同的 RSU 位置！
    # 如果您之前的脚本使用的是 env_config.RSU_POSITIONS，请确保这里也能获取到。
    # 这里我们尝试从 vanet_env 导入，如果失败则警告。
    try:
        from vanet_env import env_config
        RSU_POSITIONS = env_config.RSU_POSITIONS
    except ImportError:
        print("⚠️ 警告：无法导入 env_config。将使用随机 RSU 位置，这会导致评估数据与训练数据不一致（找不到同一组最近车辆）。")
        print("请手动在此处设置正确的 RSU_POSITIONS！")
        # RSU_POSITIONS = ...

    X_MAX = 1000.0
    Y_MAX = 1000.0
    CSV_FILE = os.path.join(os.path.dirname(__file__), "data", "trajectory_log.csv")
    MODEL_PATH = "dynamic_predictor.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 2. 加载数据 ---
    if not os.path.exists(CSV_FILE):
        print(f"错误：找不到数据文件 {CSV_FILE}")
        return

    # 注意：这里我们加载整个数据集进行评估。
    # 在严格的学术研究中，你应该将数据集分为训练集和测试集。
    dataset = LocalTrajectoryDataset(
        csv_file=CSV_FILE,
        seq_len=SEQ_LEN,
        k_vehicles=K_VEHICLES,
        rsu_positions=RSU_POSITIONS,
        x_max=X_MAX,
        y_max=Y_MAX
    )
    # batch_size=1 方便我们进行可视化和逐个分析，也可以设大一点
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # --- 3. 加载模型 ---
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}")
        return

    model = DynamicTransformerPredictor(
        k_vehicles=K_VEHICLES,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("模型加载成功。")

    # --- 4. 计算指标 ---
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    # 用于可视化的列表
    vis_history = []
    vis_target = []
    vis_pred = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # history: (B, K, L, 2)
            # target: (B, K, 2)
            # mask: (B, K)
            history, target, mask = batch
            history = history.to(device)
            target = target.to(device)
            mask = mask.to(device)

            # 前向传播
            # (B, K, 2)
            predictions = model(history)

            # --- 计算误差 (反归一化) ---
            # 我们在真实尺度（米）上计算误差，这样更有意义

            pred_real = predictions.clone()
            pred_real[:, :, 0] *= X_MAX
            pred_real[:, :, 1] *= Y_MAX

            target_real = target.clone()
            target_real[:, :, 0] *= X_MAX
            target_real[:, :, 1] *= Y_MAX

            # 欧几里得距离: (B, K)
            distances = torch.norm(pred_real - target_real, dim=2)

            # 只统计 mask 为 True (有效车辆) 的数据
            valid_distances = distances[mask]

            if valid_distances.numel() > 0:
                # 对于单步预测（只预测下一个时间点），ADE 和 FDE 是同一个值
                # 如果预测未来多步，ADE是平均，FDE是最后一步
                batch_error = valid_distances.sum().item()
                batch_count = valid_distances.numel()

                total_ade += batch_error
                total_fde += batch_error  # 单步预测下 ADE = FDE
                total_samples += batch_count

            # 收集前几个样本用于可视化
            if len(vis_history) < 5 and mask.any():
                # 找到第一个有效的样本
                for i in range(history.shape[0]):
                    if mask[i].any():  # 如果这个 batch 里第 i 个样本（某个RSU）有有效车辆
                        vis_history.append(history[i].cpu().numpy())
                        vis_target.append(target[i].cpu().numpy())
                        vis_pred.append(predictions[i].cpu().numpy())
                        break

    # --- 5. 打印结果 ---
    if total_samples > 0:
        avg_ade = total_ade / total_samples
        avg_fde = total_fde / total_samples
        print("\n=== 评估结果 ===")
        print(f"总样本数 (车辆实例): {total_samples}")
        print(f"ADE (平均位移误差): {avg_ade:.4f} 米")
        print(f"FDE (最终位移误差): {avg_fde:.4f} 米")
        print("注意：ADE/FDE 越低越好。如果地图是 1000x1000米，误差在几米内通常是可以接受的。")
    else:
        print("未找到有效样本。")

    # --- 6. 可视化 ---
    if len(vis_history) > 0:
        print("\n正在生成可视化图表 (visualization.png)...")
        plot_results(vis_history[0], vis_target[0], vis_pred[0], X_MAX, Y_MAX)


def plot_results(history, target, pred, x_max, y_max):
    """
    绘制一个 RSU 视野内的 K 辆车
    history: (K, L, 2)
    target: (K, 2)
    pred: (K, 2)
    """
    plt.figure(figsize=(10, 10))

    # 颜色映射
    colors = plt.cm.get_cmap('tab10', len(history))

    for k in range(len(history)):
        # 检查是否全是0（无效车辆）
        if np.all(history[k] == 0) and np.all(target[k] == 0):
            continue

        # 历史轨迹
        hist_x = history[k, :, 0] * x_max
        hist_y = history[k, :, 1] * y_max

        # 真实目标
        tgt_x = target[k, 0] * x_max
        tgt_y = target[k, 1] * y_max

        # 预测目标
        pred_x = pred[k, 0] * x_max
        pred_y = pred[k, 1] * y_max

        c = colors(k)

        # 画线
        plt.plot(hist_x, hist_y, linestyle='-', color=c, alpha=0.5, label=f'Veh {k} Hist' if k == 0 else "")
        plt.scatter(hist_x, hist_y, s=10, color=c, alpha=0.5)

        # 连接历史最后一点和真实点/预测点
        plt.plot([hist_x[-1], tgt_x], [hist_y[-1], tgt_y], linestyle=':', color='green', alpha=0.3)
        plt.plot([hist_x[-1], pred_x], [hist_y[-1], pred_y], linestyle=':', color='red', alpha=0.3)

        # 画点
        plt.scatter(tgt_x, tgt_y, marker='o', s=100, color='green', label='Ground Truth' if k == 0 else "",
                    edgecolors='k')
        plt.scatter(pred_x, pred_y, marker='x', s=100, color='red', label='Prediction' if k == 0 else "", linewidths=3)

    plt.title(f"Trajectory Prediction Sample (RSU View)\nSolid: History | Green O: Truth | Red X: Prediction")
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)

    plt.savefig("visualization.png")
    print("图表已保存为 visualization.png")
    # plt.show() # 如果在 Jupyter 中可以取消注释


if __name__ == "__main__":
    evaluate_model()