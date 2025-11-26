import torch
import torch.nn as nn
from torch.nn import GRU, Linear, Sequential
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import pandas as pd
import os
import sys
from collections import deque
from vanet_env import env_config



# -----------------------------------------------------------------------------
# 1. 位置编码 (Transformer 的标准组件)
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    为 Transformer 输入添加位置编码。
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        """
        参数:
        d_model: 嵌入维度 (embed_dim)
        dropout: Dropout 概率
        max_len: 预先计算的位置编码的最大长度 (应 >= K_VEHICLES)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        输入 x 形状: (BatchSize, SequenceLength, EmbedDim)
        (在我们的例子中 SequenceLength 是 K)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# -----------------------------------------------------------------------------
# 2. 动态 Transformer 预测器 (主模型)
# -----------------------------------------------------------------------------
class DynamicTransformerPredictor(nn.Module):
    """
    使用 Transformer Encoder 来动态处理 K 辆车 (K=OBS_VEHICLES)

    架构:
    1. GRU (时序编码): (B, K, L, 2) -> (B, K, D_embed)
    2. Transformer (交互): (B, K, D_embed) -> (B, K, D_embed)
    3. MLP (预测): (B, K, D_embed) -> (B, K, 2)
    """

    def __init__(self,
                 k_vehicles: int,  # RSU 观测的固定车辆数 (K)
                 seq_len: int,  # 历史轨迹长度 (L)
                 input_dim: int = 2,  # (x, y)
                 embed_dim: int = 64,  # 嵌入维度 (D_embed)
                 num_heads: int = 4,  # Transformer 多头注意力
                 num_layers: int = 2,  # Transformer Encoder 层数
                 dropout: float = 0.1
                 ):
        super(DynamicTransformerPredictor, self).__init__()

        self.k_vehicles = k_vehicles
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # --- 1. 时序编码器 (RNN/GRU) ---
        self.temporal_encoder = GRU(
            input_size=input_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True  # (N, L, H_in) -> (B*K, L, 2)
        )

        # --- 2. 位置编码 ---
        self.pos_encoder = PositionalEncoding(
            d_model=embed_dim,
            dropout=dropout,
            max_len=k_vehicles + 5
        )

        # --- 3. Transformer 编码器 (交互模块) ---
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True  # (B, K, D_embed)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers
        )

        # --- 4. 预测头 (MLP) ---
        self.prediction_head = Sequential(
            Linear(embed_dim, embed_dim // 2),
            nn.LeakyReLU(),
            Linear(embed_dim // 2, 2)  # (pred_x, pred_y)
        )

    def forward(self, x_history):
        """
        前向传播
        参数:
        x_history (Tensor): (B, K, L, D)
                           B=BatchSize, K=k_vehicles, L=seq_len, D=input_dim(2)
        返回:
        predictions (Tensor): (B, K, 2)
        """
        B, K, L, D = x_history.shape

        # --- 1. 时序编码 ---
        # (B, K, L, D) -> (B*K, L, D)
        x_temporal_input = x_history.reshape(B * K, L, D)

        # hidden_state 形状: (1, B*K, D_embed)
        _, hidden_state = self.temporal_encoder(x_temporal_input)

        # (B*K, D_embed)
        vehicle_embeddings = hidden_state.squeeze(0)

        # (B*K, D_embed) -> (B, K, D_embed)
        x_transformer_input = vehicle_embeddings.reshape(B, K, self.embed_dim)

        # --- 2. 添加位置编码 ---
        x_transformer_input = self.pos_encoder(x_transformer_input)

        # --- 3. Transformer 交互 ---
        # (B, K, D_embed) -> (B, K, D_embed)
        transformer_output = self.transformer_encoder(x_transformer_input)

        # --- 4. 预测 ---
        # (B, K, D_embed) -> (B*K, D_embed)
        transformer_output_flat = transformer_output.reshape(B * K, self.embed_dim)

        # (B*K, D_embed) -> (B*K, 2)
        predictions = self.prediction_head(transformer_output_flat)

        # (B*K, 2) -> (B, K, 2)
        predictions = predictions.reshape(B, K, 2)

        return predictions


# -----------------------------------------------------------------------------
# 3. 新的 Dataset (用于 K 辆车的局部数据)
# -----------------------------------------------------------------------------
class LocalTrajectoryDataset(Dataset):
    """
    这个 Dataset 从全局的 trajectory.csv 中为每个 (rsu, timestep) 组合
    动态地提取 K 辆最近的车辆，并创建局部历史和目标。
    """

    def __init__(self, csv_file, seq_len, k_vehicles, rsu_positions, x_max, y_max):
        """
        参数:
        csv_file: trajectory.csv 文件路径
        seq_len: 历史序列长度 (L)
        k_vehicles: RSU 观测的车辆数 (K)
        rsu_positions: RSU 坐标 (N_rsu, 2)
        x_max, y_max: 归一化最大值
        """
        print("正在加载和预处理全局轨迹数据...")
        self.seq_len = seq_len
        self.k_vehicles = k_vehicles
        self.rsu_positions = np.array(rsu_positions)
        self.x_max = x_max
        self.y_max = y_max

        # --- 1. 加载数据 ---
        df = pd.read_csv(csv_file)
        # 归一化坐标
        df["x"] = df["x"] / self.x_max
        df["y"] = df["y"] / self.y_max

        # --- 2. 按时间步重组数据 ---
        # {timestep: {veh_id: (x, y)}}
        self.data_by_timestep = {}
        for row in df.itertuples():
            t = row.real_time
            if t not in self.data_by_timestep:
                self.data_by_timestep[t] = {}
            self.data_by_timestep[t][row.vehicle_id] = (row.x, row.y)

        self.timesteps = sorted(self.data_by_timestep.keys())
        self.num_rsus = len(rsu_positions)

        # --- 3. 创建 (timestep, rsu_idx) 索引 ---
        # 我们将为每个 RSU 在每个有效的时间点创建一个样本
        self.samples = []
        # 我们需要 seq_len 的历史 + 1 的目标，所以从 seq_len 开始
        for t_idx in range(self.seq_len, len(self.timesteps)):
            timestep = self.timesteps[t_idx]  # 这是目标时间
            # 检查这个时间点是否有车辆数据
            if timestep in self.data_by_timestep and len(self.data_by_timestep[timestep]) > 0:
                for rsu_idx in range(self.num_rsus):
                    self.samples.append((timestep, rsu_idx))

        print(f"数据加载完成。总时间步: {len(self.timesteps)}。总样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本: (history, target, mask)
        history: (K, L, 2)
        target: (K, 2)
        mask: (K,) - 标记车辆是否有效
        """
        # 1. 获取样本信息
        target_timestep, rsu_idx = self.samples[idx]
        rsu_pos = self.rsu_positions[rsu_idx]

        # 2. 找到目标时刻 K 辆最近的车
        target_vehicles = self.data_by_timestep.get(target_timestep, {})
        if not target_vehicles:
            # 如果目标时刻没有车，返回空数据
            return self._get_empty_sample()

        vehicle_distances = []
        for veh_id, (x, y) in target_vehicles.items():
            # (注意：我们使用归一化后的坐标计算距离)
            dist = np.linalg.norm(np.array([x, y]) - (rsu_pos / np.array([self.x_max, self.y_max])))
            vehicle_distances.append((dist, veh_id))

        vehicle_distances.sort(key=lambda x: x[0])

        k_nearest_ids = [veh_id for _, veh_id in vehicle_distances[:self.k_vehicles]]

        # 3. 准备数据容器
        history_data = np.zeros((self.k_vehicles, self.seq_len, 2), dtype=np.float32)
        target_data = np.zeros((self.k_vehicles, 2), dtype=np.float32)
        mask = np.zeros(self.k_vehicles, dtype=bool)

        # 4. 填充历史和目标
        target_t_idx = self.timesteps.index(target_timestep)

        for i, veh_id in enumerate(k_nearest_ids):
            # 填充目标
            target_data[i] = target_vehicles[veh_id]
            mask[i] = True  # 标记该车有效

            # 填充历史 (从 target_t_idx-L 到 target_t_idx-1)
            for j in range(self.seq_len):
                hist_t_idx = target_t_idx - self.seq_len + j
                hist_timestep = self.timesteps[hist_t_idx]

                # 获取该历史时刻的车辆位置
                hist_pos = self.data_by_timestep.get(hist_timestep, {}).get(veh_id)

                if hist_pos:
                    history_data[i, j] = hist_pos
                else:
                    # 如果该车在历史某刻不存在，保持为 0
                    # (也可以使用插值或前向填充，这里为简单起见用 0)
                    pass

                    # 5. 如果车辆数不足 K，用 0 填充 (已经由 np.zeros 完成)

        return torch.tensor(history_data), torch.tensor(target_data), torch.tensor(mask)

    def _get_empty_sample(self):
        history_data = np.zeros((self.k_vehicles, self.seq_len, 2), dtype=np.float32)
        target_data = np.zeros((self.k_vehicles, 2), dtype=np.float32)
        mask = np.zeros(self.k_vehicles, dtype=bool)
        return torch.tensor(history_data), torch.tensor(target_data), torch.tensor(mask)


# -----------------------------------------------------------------------------
# 4. 训练脚本
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # --- 1. 定义超参数 ---

    K_VEHICLES = 5
    RSU_POSITIONS = env_config.RSU_POSITIONS
    X_MAX = 1000.0
    Y_MAX = 1000.0

    # (训练参数)
    SEQ_LEN = 10  # 历史长度 (L)
    EMBED_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    BATCH_SIZE = 64
    # (文件路径)
    CSV_FILE = os.path.join(os.path.dirname(__file__), "data", "trajectory_log.csv")
    MODEL_SAVE_PATH = "dynamic_predictor.pth"

    print(f"--- 动态 Transformer 轨迹预测器训练 ---")
    print(f"K (观测车辆数): {K_VEHICLES}")
    print(f"L (历史长度): {SEQ_LEN}")
    print(f"RSU 数量: {len(RSU_POSITIONS)}")
    print(f"地图尺寸: ({X_MAX}, {Y_MAX})")

    # 检查 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 2. 初始化 Dataset 和 DataLoader ---
    if not os.path.exists(CSV_FILE):
        print(f"错误: 未找到 '{CSV_FILE}'。")
        sys.exit(1)

    dataset = LocalTrajectoryDataset(
        csv_file=CSV_FILE,
        seq_len=SEQ_LEN,
        k_vehicles=K_VEHICLES,
        rsu_positions=RSU_POSITIONS,
        x_max=X_MAX,
        y_max=Y_MAX
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # --- 3. 初始化模型、损失函数和优化器 ---
    model = DynamicTransformerPredictor(
        k_vehicles=K_VEHICLES,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)

    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. 训练循环 ---
    print("\n开始训练...")
    model.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # history: (B, K, L, 2), target: (B, K, 2), mask: (B, K)
            history, target, mask = batch

            # 将数据移动到设备
            history = history.to(device)
            target = target.to(device)
            mask = mask.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            # (B, K, L, 2) -> (B, K, 2)
            predictions = model(history)

            # --- 计算损失 (关键) ---
            # 我们只计算有效车辆 (mask=True) 的损失

            # (B, K, 2) -> (B*K, 2)
            pred_flat = predictions.reshape(-1, 2)
            target_flat = target.reshape(-1, 2)

            # (B, K) -> (B*K)
            mask_flat = mask.reshape(-1)

            # 仅选择有效的部分
            pred_masked = pred_flat[mask_flat]
            target_masked = target_flat[mask_flat]

            if pred_masked.shape[0] > 0:  # 确保这个批次里有有效车辆
                loss = criterion(pred_masked, target_masked)

                # 反向传播
                loss.backward()

                # 更新权重
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, 平均损失 (MSE): {avg_loss:.6f}")
        else:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, 没有有效数据，跳过。")

    # --- 5. 保存模型 ---
    print("训练完成。")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"模型已保存到: {MODEL_SAVE_PATH}")