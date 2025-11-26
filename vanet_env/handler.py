from abc import ABC, abstractmethod
import itertools
import random
from gymnasium import spaces
import sys
import os
from collections import deque
import torch

import numpy as np
import pandas as pd
from shapely import Point

from vanet_env import env_config, utility
from vanet_env.entites import Vehicle, Rsu

from predict_new import DynamicTransformerPredictor


sys.path.append("./")


# env handler, imp or extend to use
class Handler(ABC):
    def __init__(self, env):
        self.env = env
        pass

    @abstractmethod
    def _update_global_history(self):
        pass

    # step func here
    @abstractmethod
    def step(self, actions):
        pass

    # reward func here, return reward
    @abstractmethod
    def reward(self):
        pass

    # init action and obs spaces
    @abstractmethod
    def spaces_init(self):
        pass

    # take action logic here
    @abstractmethod
    def take_action(self):
        pass

    @abstractmethod
    def update_observation(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class TrajectoryHandler(Handler):
    def __init__(self, env):
        self.env = env
        self.log_trajectory = False  # 控制是否记录轨迹数据到CSV
        if self.log_trajectory:
            self.trajectory_records = []

        # 轨迹预测配置
        self.use_trajectory_predictor = True
        self.k_vehicles = 5  # 必须与训练时 K_VEHICLES 一致
        self.seq_len = 10  # 必须与训练时 SEQ_LEN 一致
        self.x_max = 1000.0  # 必须与训练时 X_MAX 一致 (归一化参数)
        self.y_max = 1000.0  # 必须与训练时 Y_MAX 一致

        # 历史缓冲区: 字典 {vehicle_id: deque(maxlen=seq_len)}
        self.full_history_buffer = {}
        # 用于存储上一帧对当前帧的预测结果: {veh_id: (pred_x, pred_y)}
        self.last_step_predictions = {}
        # 强制使用 CPU 以获得最佳的小批量推理性能
        self.device = torch.device("cpu")

        # 加载模型
        if self.use_trajectory_predictor and DynamicTransformerPredictor is not None:
            try:
                # 初始化模型结构 (参数需与训练时一致)
                self.predictor_model = DynamicTransformerPredictor(
                    k_vehicles=self.k_vehicles,
                    seq_len=self.seq_len,
                    embed_dim=64, num_heads=4, num_layers=2
                )

                # 加载权重
                model_path = "dynamic_predictor.pth"
                # 如果不在当前目录，尝试在 vanet_env 下查找
                if not os.path.exists(model_path):
                    model_path = os.path.join("vanet_env", "dynamic_predictor.pth")

                if os.path.exists(model_path):
                    self.predictor_model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.predictor_model.to(self.device)
                    self.predictor_model.eval()  # 切换到评估模式
                    print(f"Trajectory Prediction Model loaded from {model_path}")
                else:
                    print(f"Warning: Model file {model_path} not found. Trajectory prediction disabled.")
                    self.use_trajectory_predictor = False
            except Exception as e:
                print(f"Error loading trajectory model: {e}")
                self.use_trajectory_predictor = False
        else:
            self.use_trajectory_predictor = False

    # 更新全局历史
    def _update_global_history(self):
        """
        更新所有在场车辆的轨迹历史缓冲区
        """
        if not self.use_trajectory_predictor:
            return

        # 遍历环境中所有车辆
        for veh_id, veh in self.env.vehicles.items():
            # 如果是新车，初始化队列
            if veh_id not in self.full_history_buffer:
                self.full_history_buffer[veh_id] = deque(maxlen=self.seq_len)

            # 归一化坐标并存入
            norm_x = veh.position.x / self.x_max
            norm_y = veh.position.y / self.y_max
            self.full_history_buffer[veh_id].append([norm_x, norm_y])

    def step(self, actions):
        """
        执行一个完整的时间步逻辑。

        初始化检查
        动作执行
        奖励计算
        SUMO 物理仿真(含数据记录)
        观测更新(含轨迹预测)
        终止判定

        Args:
            actions: 智能体输出的动作（可能是离散的索引或连续的数值）。

        Returns:
            observations: 包含预测数据的最新观测值。
            rewards: 本步奖励。
            terminations: 终止信号 (True/False)。
            truncations: 截断信号 (True/False, 通常用于超时)。
            infos: 额外调试信息。
        """
        # 1. 自动重置机制：如果 SUMO 仿真器意外关闭或未初始化，先执行重置
        if not self.env.sumo_has_init:
            observations, terminations, _ = self.env.reset()

        # 2. 设置随机种子：确保实验的可复现性，种子随时间步递增
        random.seed(self.env.seed + self.env.timestep)

        # 3. 执行动作 (Take Action)
        # 根据环境配置调用对应的动作处理逻辑
        if self.env.is_discrete:
            self.take_action(actions)  # 处理离散动作 (如任务卸载决策)
        else:
            self.env._beta_take_box_actions(actions)  # 处理连续动作 (如资源分配比例)

        # 4. 计算奖励 (Calculate Reward)
        # 基于当前网络状态（延时、能耗、QoE等）计算每个智能体的奖励
        rewards = self.reward()

        # 5. SUMO 物理仿真步进 (Simulation Step)
        # self.env.fps 是决策频率参数。
        # 例如 fps=10 表示：每进行 10 次 SUMO 微观仿真步，才进行 1 次 RL 智能体决策。
        # 这样可以模拟真实世界中决策间隔大于物理模拟间隔的情况。
        if self.env.timestep % self.env.fps == 0:
            self.env.sumo.simulationStep()  # 调用 SUMO API 推演物理世界

            # [状态同步] 将 SUMO 内部的车辆最新状态（位置、速度、任务）同步到 Python 对象
            self._update_vehicles()

            # [功能模块：数据记录] 如果开启了 log_trajectory，记录车辆真实位置用于生成训练集
            if self.log_trajectory:
                for vehicle in self.env.vehicles.values():
                    record = {
                        "real_time": self.env.timestep // self.env.fps,  # 记录逻辑时间步
                        "vehicle_id": vehicle.vehicle_id,
                        "x": vehicle.position.x,
                        "y": vehicle.position.y,
                    }
                    self.trajectory_records.append(record)

            # [拓扑更新] 重新计算车辆与 RSU 的距离，更新连接队列
            self._update_connections_queue()
            # [任务维护] 清理已完成、失败或离开通信范围的任务
            self._update_job_handlings()

            # [渲染] 如果开启了 GUI 模式，更新可视化画面
            self.env.render()

        # 6. 更新观测 (Update Observation)
        # 这是一个关键点：在此方法内部，会调用 Transformer 模型对车辆未来位置进行【批量预测】，
        # 并将预测结果拼接到 Observation 中返回给智能体。
        observations = self.update_observation()

        # 7. 检查结束条件和生成 Info
        truncations = {a: False for a in self.env.agents}
        infos = {}

        # 更新所有 RSU 的空闲状态 (用于统计和全局观测)
        self._update_all_rsus_idle()

        # 检查是否达到最大时间步 (Max Steps)
        if self.env.timestep >= self.env.max_step:
            # 达到最大步数，标记为终止 (Terminate)
            terminations = {a: True for a in self.env.agents}

            # 记录结束时的状态信息
            infos = {
                a: {"bad_transition": True, "idle": self.env.rsus[idx].idle}
                for idx, a in enumerate(self.env.agents)
            }

            # 打印本回合的统计数据
            print(f"cloud_count:{self.env.cloud_times}")  # 云端处理次数
            print(f"full_count:{self.env.full_count}")  # 满载拒绝次数

            # 关闭 SUMO 仿真器，释放资源
            self.env.sumo.close()
            self.env.sumo_has_init = False

            # [功能模块：保存数据] 如果开启了记录，仿真结束时将所有轨迹保存为 CSV
            if self.log_trajectory:
                df = pd.DataFrame(self.trajectory_records)
                df.to_csv("trajectory_log.csv", index=False)
                print(">>> 轨迹数据已保存到 trajectory_log.csv")
        else:
            # 未结束，正常返回状态
            infos = {
                a: {"bad_transition": False, "idle": self.env.rsus[idx].idle}
                for idx, a in enumerate(self.env.agents)
            }
            terminations = {a: False for idx, a in enumerate(self.env.agents)}

        # 递增全局时间步
        self.env.timestep += 1

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed):
        """
        环境重置方法 (Reset)

        在每个 Episode (回合) 开始时被调用。
        清理上一回合的历史数据 (对轨迹预测至关重要)。
        重置 SUMO 仿真器到初始时刻。
        重新初始化所有 RSU 和车辆对象。
        生成初始的观测值 (Observation)。
        """
        # 1. 重置全局时间步计数器
        self.env.timestep = 0

        # 清空历史轨迹缓冲区
        self.full_history_buffer.clear()

        # 2. 重置所有RSU的状态
        # 根据配置文件 (env_config) 中的位置重新创建Rsu对象
        # 这会重置它们的连接数、资源使用率、处理队列等所有状态
        self.env.rsus = [
            Rsu(
                id=i,
                position=Point(env_config.RSU_POSITIONS[i]),
                max_connections=self.env.max_connections,
                max_cores=self.env.num_cores,
            )
            for i in range(self.env.num_rsus)
        ]

        # 3. 初始化内容缓存 (Content Caching) 模块
        # 如果尚未加载内容数据，则执行初始化，确保任务类型数据可用
        if not self.env.content_loaded:
            self.env._content_init()
            self.env.content_loaded = True

        # 4. 设置随机种子
        # 确保实验的可重复性。同时设置 numpy 和 python random 的种子。
        np.random.seed(seed)
        random.seed(seed)
        self.env.seed = seed

        # 5. 重置 SUMO 仿真器
        # 如果 SUMO 尚未启动或意外关闭，重新启动 SUMO 实例
        if not self.env.sumo_has_init:
            self.env._sumo_init()
            self.env.sumo_has_init = True

        # 6. 获取初始车辆并初始化 Vehicle 对象
        # 从 SUMO 获取当前（初始时刻）所有车辆的 ID 列表
        self.env.vehicle_ids = self.env.sumo.vehicle.getIDList()

        # 为每辆车创建一个 Python Vehicle 对象
        # 用于在 Python 端跟踪车辆的状态、位置、分配的任务类型等
        self.env.vehicles = {
            vehicle_id: Vehicle(
                vehicle_id,
                self.env.sumo,
                self.env.timestep,
                self.env.cache.get_content(
                    min(self.env.timestep // self.env.caching_step, 9)
                ),
            )
            for vehicle_id in self.env.vehicle_ids
        }

        # 7. 更新初始连接状态
        # 计算车辆与 RSU 的距离，建立初始的连接关系
        self._update_connections_queue()

        # 8. 重置智能体列表
        # 将当前活动的智能体列表重置为所有可能的智能体
        self.env.agents = list.copy(self.env.possible_agents)

        # 9. 获取初始观测值
        # 这里会立即调用一次 update_observation。
        # 即使是第 0 步，如果开启了预测功能，这里也会尝试进行预测
        # (虽然历史数据刚清空，代码中的 padding 逻辑会处理这种情况)，
        # 这一步是为了确保返回给 RL 智能体的第一个观测向量维度是正确的。
        observations = self.update_observation()

        # 10. 更新 RSU 的空闲状态
        # 检查每个 RSU 是否有任务，用于 global_obs 的构建
        self._update_all_rsus_idle()

        # 11. 构建辅助信息
        # bad_transition: 标记是否因异常状态结束 (此处刚开始，为 False)
        # idle: 记录每个 RSU 是否处于空闲状态
        infos = {
            a: {"bad_transition": False, "idle": self.env.rsus[idx].idle}
            for idx, a in enumerate(self.env.agents)
        }

        # 12. 初始化终止状态
        # 刚开始，所有智能体都处于活跃状态 (False)
        terminations = {a: False for idx, a in enumerate(self.env.agents)}

        return observations, terminations, infos

    def reward(self):
        """
        计算奖励函数

        调用 utility 模块计算当前时刻所有车辆任务的 QoE (体验质量) 和 RSU 的缓存命中率。
        将计算结果聚合，得出每个智能体 (RSU) 的标量奖励值。
        统计全系统的平均奖励 (ava_rewards)，用于监控训练效果。

        Returns:
            rewards (dict): 键为 agent_id (如 'rsu_0'), 值为 float 类型的奖励值。
        """
        rewards = {}  # 用于存储每个智能体的最终奖励
        reward_per_agent = []  # 临时列表，用于计算全局平均奖励 (只包含工作的 RSU)

        # 1. 调用外部工具函数进行核心计算
        # utility.fixed_calculate_utility 会遍历所有车辆和 RSU，
        # 根据任务是否完成、延迟大小、能耗等计算具体的数值指标。
        # rsu_qoe_dict: {rsu_id: [qoe_val1, qoe_val2, ...]} 存储每个 RSU 处理的所有任务的 QoE 值
        # caching_ratio_dict: {rsu_id: ratio} 存储每个 RSU 的缓存命中率
        rsu_qoe_dict, caching_ratio_dict = utility.fixed_calculate_utility(
            vehs=self.env.vehicles,  # 当前所有车辆对象
            rsus=self.env.rsus,  # 当前所有 RSU 对象
            rsu_network=self.env.rsu_network,  # RSU 邻居拓扑图
            time_step=self.env.timestep,  # 当前时间步
            fps=self.env.fps,  # 仿真帧率
            weight=self.env.max_weight,  # 权重参数 (用于加权计算)
        )

        # 将计算出的 QoE 字典存回 env，供其他模块使用
        self.env.rsu_qoe_dict = rsu_qoe_dict

        # 2. 更新缓存统计信息
        # 将本步的缓存命中率追加到 RSU 的历史记录中，用于后续评估
        for rid, ratio in caching_ratio_dict.items():
            self.env.rsus[rid].hit_ratios.append(ratio)

        # 3. 为每个智能体计算最终奖励
        for idx, agent in enumerate(self.env.agents):
            # 获取该 RSU (idx) 本步处理的所有任务的 QoE 列表
            a = rsu_qoe_dict[idx]

            if len(a) > 0:
                # 数据清洗与展平
                # 有时 utility 返回的可能是嵌套列表 [[qoe1], [qoe2]]，需要展平为 [qoe1, qoe2]
                try:
                    flattened = list(itertools.chain.from_iterable(a))
                except TypeError:
                    # 如果已经是扁平列表，直接使用
                    flattened = list(a)

                # 奖励定义：取该 RSU 所有处理任务 QoE 的平均值
                # 这鼓励 RSU 尽可能让它处理的所有任务都获得高质量体验
                rewards[agent] = np.mean(flattened)
            else:
                # 如果该 RSU 本步没有处理任何任务，给予 0 奖励
                # (也可以根据需求改为负惩罚，取决于具体的 RL 设计)
                rewards[agent] = 0.0

        # 4. 计算系统级平均奖励 (用于 Log 监控)
        for idx, agent in enumerate(self.env.agents):
            # 只统计那些 "非空闲" (即正在处理任务) 的 RSU
            # 避免大量空闲 RSU 的 0 奖励拉低平均分，影响对算法性能的判断
            if not self.env.rsus[idx].idle:
                reward_per_agent.append(rewards[agent])

        # 将本步的有效平均奖励存入 env.ava_rewards 列表
        self.env.ava_rewards.append(np.mean(reward_per_agent))

        return rewards

    # return obs space and action space
    def spaces_init(self):
        """
        初始化观测空间和动作空间

        定义 Observation Space (输入给神经网络的数据维度)。
        包含基础特征 (如负载、连接数)。
        包含轨迹预测数据 (未来坐标 + 掩码)。
        定义 Action Space (神经网络输出的动作维度)。
        定义了每个动作维度的含义 (如任务分配、资源分配、缓存策略等)。
        """
        # 获取邻居 RSU 的数量 (通常为 2，左右各一个)
        neighbor_num = len(self.env.rsu_network[0])

        # 1. 定义 Local Observation Space (局部观测空间)
        # 基础特征维度 = 5
        # [0] norm_self_handling (自身处理任务数/容量)
        # [1] norm_self_handling_ratio (自身处理任务权重比/容量)
        # [2] norm_num_conn_queue (排队连接数/容量)
        # [3] norm_num_connected (已连接数/容量)
        # [4] norm_nb_h (邻居平均负载)
        original_local_dim = 5

        # [动态计算总维度]
        # 如果开启了轨迹预测，观测空间需要扩容以容纳预测数据
        if self.use_trajectory_predictor:
            # 新增维度 = K 辆车 * (x坐标 + y坐标 + 有效掩码mask)
            # 例如: 5辆车 * 3 = 15 维
            trajectory_dim = self.k_vehicles * 3
            total_dim = original_local_dim + trajectory_dim
        else:
            # 如果没开启预测，保持基础维度
            total_dim = original_local_dim

        # 定义 Box 空间 (连续数值，范围 0.0 到 1.0)
        self.env.local_neighbor_obs_space = spaces.Box(
            0.0, 1.0, shape=(total_dim,)
        )

        # 2. 定义 Global Observation Space (全局观测空间)
        # 用于 Critic 网络 (在 MAPPO 等中心化训练算法中)
        # 包含自身和所有邻居的详细状态
        # 维度 = (邻居数 + 1个自己) * 2 (每人贡献2个特征)
        self.env.global_neighbor_obs_space = spaces.Box(
            0.0, 1.0, shape=((neighbor_num + 1) * 2,)
        )

        # 初始化车辆位置列表 (辅助变量)
        self.env.veh_locations = []

        # 3. 定义 Action Space (动作空间)
        # 定义连续动作空间 (Box)，用于底层处理或连续控制算法 (DDPG/SAC等)
        # 形状基于各种资源分配的组合 (连接数、核心数、缓存容量等)
        self.env.box_neighbor_action_space = spaces.Box(
            0.0,
            1.0,
            shape=(
                (neighbor_num + 1) * self.env.max_connections  # 任务迁移比例
                + self.env.max_connections  # 任务大小限制
                + self.env.num_cores  # 计算资源分配
                + self.env.max_connections  # 通信资源分配
                + 1  # 总算力控制
                + self.env.max_caching,  # 缓存内容决策
            ),
        )

        # 离散化区间数 (例如将 0-1 分为 5 档)
        bins = self.env.bins

        # 定义每个动作维度的离散大小 (MultiDiscrete)
        # 这是一个多维离散动作空间，每一维代表一种决策类型
        self.env.action_space_dims = [
            self.env.max_connections,  # 动作1: 自身任务分配比例 (针对每个连接)
            self.env.max_connections,  # 动作2: 邻居任务分配比例
            self.env.max_connections,  # 动作3: 每个连接的最大分配大小
            self.env.num_cores,  # 动作4: 算力资源分配 (针对每个核心)
            self.env.max_connections,  # 动作5: 通信资源分配
            1,  # 动作6: 总算力分配调节 (1个标量)
            self.env.max_caching,  # 动作7: 缓存策略 (针对每个缓存槽位)
        ]

        action_space_dims = self.env.action_space_dims

        # 构建最终的 MultiDiscrete 动作空间
        # 它将上述所有维度的动作拼接在一起。
        # 例如: [bins, bins, ..., max_content, max_content]
        self.env.md_discrete_action_space = spaces.MultiDiscrete(
            [bins] * action_space_dims[0]
            + [bins] * action_space_dims[1]
            + [bins] * action_space_dims[2]
            + [bins] * action_space_dims[3]
            + [bins] * action_space_dims[4]
            + [bins] * action_space_dims[5]
            + [self.env.max_content] * action_space_dims[6]  # 缓存动作的大小由内容总数决定
        )

    def take_action(self, actions):
        """
        执行动作
        将智能体（Actor网络）输出的动作索引/数值转换为具体的环境操作。
        主要包含三个阶段：
        解析动作：将扁平化的动作向量切分为不同的控制部分（迁移、分配、缓存）。
        任务迁移 (Migration): 决定是否将排队的任务迁移给邻居 RSU 处理。
        资源分配 (Resource Allocation): 分配计算资源 (CPU)、带宽资源 (Bandwidth) 和更新缓存内容。

        Args:
            actions: 包含所有智能体动作的列表或数组。
        """

        # 处理单个 RSU 的任务迁移逻辑
        def _mig_job(rsu: Rsu, action, idx):
            """
            处理 RSU 连接队列中的任务，决定是自己处理、迁移给邻居还是通过云端处理。
            """
            # 如果连接队列为空，没有任务需要处理，直接返回
            if rsu.connections_queue.is_empty():
                return

            # 动作切片
            # 根据 action_space_dims 定义的维度，将动作向量切分成不同部分
            # m_actions_self: 分配给自己处理的权重 (对应 max_connections 个槽位)
            m_actions_self = action[: self.env.action_space_dims[0]]
            pre = self.env.action_space_dims[0]

            # m_actions_nb: 分配给邻居处理的权重
            m_actions_nb = action[pre: pre + self.env.action_space_dims[1]]
            pre = pre + self.env.action_space_dims[1]

            # m_actions_job_ratio: 任务处理比例系数 (决定分配多少计算量)
            m_actions_job_ratio = action[pre: pre + self.env.action_space_dims[2]]

            # 获取当前 RSU 的两个邻居对象 (拓扑结构定义在 env.rsu_network 中)
            nb_rsu1: Rsu = self.env.rsus[self.env.rsu_network[rsu.id][0]]
            nb_rsu2: Rsu = self.env.rsus[self.env.rsu_network[rsu.id][1]]

            # 遍历连接队列
            # 遍历当前 RSU 连接队列中的每一个请求
            for m_idx, ma in enumerate(rsu.connections_queue):

                # 从队列中取出一辆车 (Vehicle ID)
                # 注意：remove 操作通常是按索引移除
                veh_id: Vehicle = rsu.connections_queue.remove(index=m_idx)

                if veh_id is None:
                    continue

                veh = self.env.vehicles[veh_id]

                # 获取该任务对应的动作值 (取模防止越界，虽然通常一一对应)
                real_m_idx = m_idx % self.env.max_connections

                # 计算分配权重
                # 将离散动作值归一化 (除以 bins)，并加微小量防止除零
                self_ratio = m_actions_self[real_m_idx] / self.env.bins + 1e-6
                nb_ratio = m_actions_nb[real_m_idx] / self.env.bins + 1e-6
                job_ratio = m_actions_job_ratio[real_m_idx] / self.env.bins + 1e-6

                sum_ratio = self_ratio + nb_ratio

                # 如果总权重 > 0，说明智能体决定处理该任务（而不是丢弃）
                if sum_ratio > 0:

                    # 检查自己和邻居的计算队列是否已满 (is_full)
                    is_full = [
                        rsu.handling_jobs.is_full(),
                        nb_rsu1.handling_jobs.is_full(),
                        nb_rsu2.handling_jobs.is_full(),
                    ]

                    # 检查该车辆是否已经在自己或邻居的处理列表中 (防止重复添加)
                    index_in_rsu = rsu.handling_jobs.index((veh, 0))
                    index_in_nb_rsu1 = nb_rsu1.handling_jobs.index((veh, 0))
                    index_in_nb_rsu2 = nb_rsu2.handling_jobs.index((veh, 0))
                    idxs_in_rsus = [index_in_rsu, index_in_nb_rsu1, index_in_nb_rsu2]
                    in_rsu = [elem is not None for elem in idxs_in_rsus]

                    # [异常处理] 如果所有节点都满了，且车辆不在任何节点中 -> 转入云端处理
                    if all(is_full):
                        if not any(in_rsu):
                            veh.is_cloud = True
                            veh.job.is_cloud = True
                            self.env.cloud_times += 1  # 记录云端处理次数
                            # 清理车辆之前的处理记录
                            if not veh.job.processing_rsus.is_empty():
                                for rsu_item in veh.job.processing_rsus:
                                    if rsu_item is not None:
                                        rsu_item.remove_job(veh)
                                veh.job.processing_rsus.clear()
                            continue

                    # 归一化分配比例
                    self_ratio = self_ratio / sum_ratio
                    nb_ratio = nb_ratio / sum_ratio

                    # 定义迁移候选列表 (自己, 邻居1, 邻居2)
                    mig_rsus = np.array([rsu, nb_rsu1, nb_rsu2])

                    # 定义迁移比例分配：自己分 self_ratio，两个邻居平分 nb_ratio
                    mig_ratio = np.array(
                        [
                            self_ratio,
                            nb_ratio / 2,
                            nb_ratio / 2,
                        ]
                    )

                    # 将状态列表转换为 numpy 数组以便进行布尔运算
                    is_full = np.array(is_full)
                    in_rsu = np.array(in_rsu)

                    # [关键掩码]
                    # update_mask: 车辆已在该 RSU 中，需要更新任务比例
                    update_mask = in_rsu
                    # store_mask: 车辆不在该 RSU 中，且该 RSU 未满，可以存入新任务
                    store_mask = (~is_full & ~in_rsu)

                    # 合并有效掩码
                    valid_mask = update_mask | store_mask

                    # 重新归一化比例 (只分配给有效的 RSU)
                    if np.any(valid_mask):
                        valid_ratios = mig_ratio[valid_mask]
                        total_ratio = np.sum(valid_ratios)
                        if total_ratio > 0:
                            mig_ratio[valid_mask] = valid_ratios / total_ratio
                    else:
                        # 如果所有 RSU 都不可用 (满了且不在其中)，记录一次 Full 异常
                        self.env.full_count += 1
                        continue

                    # 执行状态更新
                    # (A) 更新已存在的任务
                    for u_idx, u_rsu in enumerate(mig_rsus):
                        if update_mask[u_idx]:
                            u_rsu: Rsu
                            # 更新任务元组: (车辆对象, 新的计算资源比例)
                            u_rsu.handling_jobs[idxs_in_rsus[u_idx]] = (
                                veh,
                                float(mig_ratio[u_idx] * job_ratio),
                            )

                    # (B) 存入新任务
                    for s_idx, s_rsu in enumerate(mig_rsus):
                        if store_mask[s_idx]:
                            s_rsu: Rsu
                            veh_disconnect: Vehicle = None

                            # 如果车辆尚未与此 RSU 建立底层连接逻辑
                            if veh not in rsu.connections:
                                # 尝试加入连接列表，如果满了会挤掉最旧的连接 (append_and_out)
                                veh_disconnect = rsu.connections.append_and_out(veh)

                            # 加入处理队列
                            s_rsu.handling_jobs.append(
                                (veh, float(mig_ratio[s_idx] * job_ratio))
                            )
                            # 车辆状态标记为正在被 s_rsu 处理
                            veh.job_process(s_idx, s_rsu)

                            # 如果有车辆因为被挤掉而断开连接，执行去处理逻辑
                            if veh_disconnect is not None:
                                veh_disconnect: Vehicle
                                veh_disconnect.job_deprocess(
                                    self.env.rsus, self.env.rsu_network
                                )
                else:
                    # 如果总权重 <= 0 (智能体放弃处理)，直接转入云端
                    self.env.cloud_times += 1
                    veh.is_cloud = True
                    veh.job.is_cloud = True
                    if not veh.job.processing_rsus.is_empty():
                        for rsu_item in veh.job.processing_rsus:
                            if rsu_item is not None:
                                rsu_item.remove_job(veh)
                        veh.job.processing_rsus.clear()

                pass

        # 遍历所有 RSU 并执行动作
        # actions 通常是 (1, num_agents, ...) 的形式，取第一个维度
        actions = actions[0]

        # 随机打乱动作执行顺序。如果不打乱，索引靠前的 RSU 总是先抢占资源或填满队列，导致不公平
        indexed_actions = list(enumerate(actions))
        random.shuffle(indexed_actions)
        num_nb = len(self.env.rsu_network[0])

        # 处理任务迁移
        # 优先处理任务的接收和分发
        for idx, action in indexed_actions:
            rsu: Rsu = self.env.rsus[idx]

            # 如果 RSU 空闲 (idle)，跳过处理以节省计算
            if rsu.idle:
                continue

            _mig_job(rsu=rsu, action=action, idx=idx)

        # 处理资源分配
        # 在任务队列确定后，分配具体的计算和带宽资源
        for idx, action in indexed_actions:
            rsu: Rsu = self.env.rsus[idx]

            if rsu.idle:
                continue

            # 1. 解析动作索引偏移量
            dims = self.env.action_space_dims
            # 跳过前三个与迁移相关的动作维度
            pre = sum(dims[:3])

            # 2. 解析计算资源分配动作 (cp_alloc)
            cp_alloc_actions = np.array(action[pre: pre + dims[3]]) / self.env.bins
            pre = pre + dims[3]

            # 3. 解析带宽资源分配动作 (bw_alloc)
            bw_alloc_actions = np.array(action[pre: pre + dims[4]]) / self.env.bins
            pre = pre + dims[4]

            # 4. 解析总计算量调节动作 (cp_usage)
            cp_usage = np.array(action[pre: pre + dims[5]]) / self.env.bins
            pre = pre + dims[5]

            # 5. 执行底层分配逻辑 (调用 Rsu 类的方法)
            # box_alloc_cp: 根据权重分配 CPU 核心给队列中的任务
            rsu.box_alloc_cp(alloc_cp_list=cp_alloc_actions, cp_usage=cp_usage)
            # box_alloc_bw: 根据权重分配通信带宽
            rsu.box_alloc_bw(
                alloc_bw_list=bw_alloc_actions, veh_ids=self.env.vehicle_ids
            )

            # 6. 处理缓存动作 (Caching)
            # 决定缓存哪些内容 (Content ID)
            a = action[pre:]
            rsu.frame_cache_content(a, self.env.max_content)

        pass

    def update_observation(self):
        """
        更新观测

        数据同步：调用 _update_global_history 记录当前帧所有车辆的位置。
        批量预测 (Batch Inference)： 一次性计算所有 RSU 的车辆轨迹预测结果。
        观测构建：为每个 RSU 智能体生成 Observation 向量。
        Local Obs: 自身状态 + 预测数据 (K*3 维)
        Global Obs: 自身 + 邻居的负载状态

        Returns:
            observations (dict): 包含所有智能体观测数据的字典。
        """
        # if self.use_trajectory_predictor and len(self.last_step_predictions) > 0:
        #     total_error = 0.0
        #     count = 0
        #
        #     # 遍历上一帧我们预测过的所有车辆
        #     for veh_id, predicted_pos in self.last_step_predictions.items():
        #         # 如果这辆车现在还在地图上
        #         if veh_id in self.env.vehicles:
        #             current_veh = self.env.vehicles[veh_id]
        #
        #             # 获取车辆当前的真实位置 (归一化)
        #             real_x = current_veh.position.x / self.x_max
        #             real_y = current_veh.position.y / self.y_max
        #
        #             # 获取上一帧预测它应该在的位置
        #             pred_x, pred_y = predicted_pos
        #
        #             # 计算欧几里得距离误差
        #             dist = np.sqrt((real_x - pred_x) ** 2 + (real_y - pred_y) ** 2)
        #
        #             real_meter_error = dist * self.x_max
        #
        #             total_error += real_meter_error
        #             count += 1
        #
        #     if count > 0:
        #         avg_error = total_error / count
        #         print(f">>> [预测监控] 当前帧对比验证: 跟踪了 {count} 辆车, 平均位置误差: {avg_error:.2f} 米")
        #
        #     # 清空旧预测，准备存新的
        # self.last_step_predictions = {}

        # 将本帧所有车辆的最新坐标存入历史缓冲区
        # 这是预测模型的数据源，必须在预测前执行
        self._update_global_history()

        observations = {}

        # 预测结果缓存表: {rsu_id: 扁平化的预测数组}
        # 用于稍后在遍历 Agent 时快速查找，避免重复计算
        prediction_map = {}

        # 批量预测准备与执行
        if self.use_trajectory_predictor:
            batch_histories = []  # 存放所有 RSU 的输入 Tensor (作为 Batch 的一部分)
            batch_rsu_ids = []  # 记录 Tensor 对应的 RSU ID，用于后续结果分发
            # 记录每个 Tensor 对应是哪个 RSU 的哪辆车 (用于后续获取 job_type)
            # 格式: [(rsu_id, vehicle_id), ...]
            batch_meta_info = []

            # 为每个 RSU 准备输入数据
            for rsu in self.env.rsus:
                rsu_pos = np.array([rsu.position.x, rsu.position.y])

                # 找到 RSU 范围内有历史数据的车辆
                dist_list = []
                for veh_id, veh in self.env.vehicles.items():
                    # 只考虑缓冲区中有数据的车 (新车可能还没数据)
                    if veh_id in self.full_history_buffer:
                        # 计算车辆到 RSU 的欧几里得距离
                        d = np.linalg.norm(np.array([veh.position.x, veh.position.y]) - rsu_pos)
                        dist_list.append((d, veh_id))

                # 按距离从小到大排序，取最近的 K 辆车 (K=5)
                dist_list.sort(key=lambda x: x[0])
                top_k_vehs = dist_list[:self.k_vehicles]

                # 构建 Tensor: 形状为 (K, seq_len, 2)
                # 初始化全 0 张量
                rsu_hist_tensor = torch.zeros(self.k_vehicles, self.seq_len, 2, dtype=torch.float32)
                # 初始化掩码 (Mask): 1.0 表示该位置有真实车辆，0.0 表示填充数据
                valid_mask = np.zeros(self.k_vehicles, dtype=np.float32)
                # 临时列表，记录当前 RSU 这 5 辆车的 ID
                current_rsu_veh_ids = []

                for k, (dist, vid) in enumerate(top_k_vehs):
                    current_rsu_veh_ids.append(vid)  # 记录 ID
                    # 获取该车辆的历史轨迹 (deque 转为 list)
                    hist = list(self.full_history_buffer[vid])

                    # Padding (填充): 如果历史不足 seq_len (10帧)
                    # 策略：在序列前面复制第一帧数据补齐，保持长度一致
                    while len(hist) < self.seq_len:
                        hist.insert(0, hist[0] if hist else [0.0, 0.0])

                    # 填入 Tensor 并标记 Mask 为有效
                    rsu_hist_tensor[k] = torch.tensor(hist, dtype=torch.float32)
                    valid_mask[k] = 1.0

                    # 如果不足 5 辆，补 None
                while len(current_rsu_veh_ids) < self.k_vehicles:
                    current_rsu_veh_ids.append(None)

                # 将构建好的单个 RSU 数据加入 Batch 列表
                batch_histories.append(rsu_hist_tensor)
                batch_rsu_ids.append(rsu.id)
                # 将这 5 辆车的 ID 保存，稍后用于缓存决策
                batch_meta_info.append(current_rsu_veh_ids)

                # 暂存 mask，预测完成后需要把它和坐标拼在一起
                prediction_map[rsu.id] = {"mask": valid_mask}

            # 执行批量预测
            # 只有当存在需要预测的数据时才执行
            if len(batch_histories) > 0:
                # 堆叠成一个大 Batch Tensor
                # Shape: (N_RSU, K, seq_len, 2) -> 例如 (20, 5, 10, 2)
                batch_input = torch.stack(batch_histories).to(self.device)

                # 执行推理
                with torch.no_grad():
                    # 输出 Shape: (N_RSU, K, 2)
                    # 输出的是未来 1 个时间步的 (x, y) 归一化坐标
                    batch_output = self.predictor_model(batch_input)
                    # 转回 CPU numpy 数组
                    batch_output = batch_output.cpu().numpy()

                # 处理结果并存入 Map
                for i, rsu_id in enumerate(batch_rsu_ids):
                    preds = batch_output[i]  # (K, 2) 当前 RSU 的预测结果
                    # 获取当前 RSU 的元数据 (这 K 辆车的 ID)
                    veh_ids = batch_meta_info[i]

                    # 根据预测位置，让下一个 RSU 提前缓存
                    for k in range(self.k_vehicles):
                        # 如果这是无效的填充位，跳过
                        if prediction_map[rsu_id]["mask"][k] == 0.0:
                            continue

                        vid = veh_ids[k]

                        # if vid is not None:
                        #     self.last_step_predictions[vid] = (preds[k, 0], preds[k, 1])

                        if vid is None or vid not in self.env.vehicles:
                            continue

                        # 获取预测的真实坐标
                        pred_x_real = preds[k, 0] * self.x_max
                        pred_y_real = preds[k, 1] * self.y_max
                        pred_coord = np.array([pred_x_real, pred_y_real])
                        # print(pred_coord)

                        # 查询 KDTree，找到预测位置最近的 RSU (即下一个 RSU)
                        # k=1 表示找最近的 1 个
                        distances, sorted_indices = self.env.rsu_tree.query(pred_coord, k=1)
                        next_rsu_id = sorted_indices  # query返回的是(dist, index)，这里直接取 index
                        # 注意: scipy 的 query 有时返回标量有时返回数组，取决于输入
                        if isinstance(next_rsu_id, np.ndarray):
                            next_rsu_id = next_rsu_id.item()  # 转为 int

                        # 简单判断：如果预测归属变了，且不是自己
                        if next_rsu_id != rsu_id:
                            target_rsu = self.env.rsus[next_rsu_id]
                            vehicle = self.env.vehicles[vid]

                            # 执行协同缓存：强制让目标 RSU 缓存该车辆需要的内容
                            # queue_jumping 会把内容插到队头 (高优先级)
                            # 注意：这里假设车辆需要的内容就是当前的 job_type
                            needed_content = vehicle.job.job_type

                            # 为了防止频繁抖动，可以加一个概率或距离阈值，这里直接执行
                            # 只有当内容不在缓存中时才操作，避免重复刷新
                            if needed_content not in target_rsu.caching_contents:
                                target_rsu.caching_contents.queue_jumping(needed_content)
                                # 打印日志观察效果
                                # print(f"Ref: 车辆 {vid} 即将从 RSU {rsu_id} 移动到 {next_rsu_id}，提前缓存内容 {needed_content}")

                    # 取出预测的 x, y 坐标
                    pred_x = preds[:, 0]
                    pred_y = preds[:, 1]
                    mask = prediction_map[rsu_id]["mask"]

                    # 扁平化拼接: [x1, x2...xK, y1, y2...yK, m1, m2...mK]
                    # 总长度 = K * 3 (例如 5*3=15)
                    flat_pred = np.concatenate([pred_x, pred_y, mask])

                    # 存入 map，供后续构建 Observation 使用
                    prediction_map[rsu_id]["final"] = flat_pred

        # 构造最终观测 (遍历所有 Agent)
        for idx, a in enumerate(self.env.agents):
            rsu = self.env.rsus[idx]

            # 获取邻居信息
            nb_id1, nb_id2 = self.env.rsu_network[idx]
            nb_rsus = [self.env.rsus[nb_id1], self.env.rsus[nb_id2]]

            # 获取邻居的任务处理列表
            nb1_h = nb_rsus[0].handling_jobs.olist
            nb2_h = nb_rsus[1].handling_jobs.olist

            # 计算归一化特征
            # 邻居 1 的计算负载 (处理任务权重 / 最大容量)
            norm_nb1_h = (
                    sum([v[1] for v in nb1_h if v is not None])
                    / nb_rsus[0].handling_jobs.max_size
            )
            # 邻居 2 的计算负载
            norm_nb2_h = (
                    sum([v[1] for v in nb2_h if v is not None])
                    / nb_rsus[1].handling_jobs.max_size
            )

            # 邻居 1 的连接队列负载 (排队数 / 最大队列长度)
            nb1_c = (
                    nb_rsus[0].connections_queue.size()
                    / nb_rsus[0].connections_queue.max_size
            )
            # 邻居 2 的连接队列负载
            nb2_c = (
                    nb_rsus[1].connections_queue.size()
                    / nb_rsus[1].connections_queue.max_size
            )

            # 邻居平均计算负载
            norm_nb_h = (norm_nb1_h + norm_nb2_h) / 2

            # 自己的任务数量负载 (数量 / 容量)
            norm_self_handling = rsu.handling_jobs.size() / rsu.handling_jobs.max_size

            # 自己的计算资源负载 (权重和 / 容量)
            norm_self_handling_ratio = (
                    sum([v[1] for v in rsu.handling_jobs.olist if v is not None])
                    / rsu.handling_jobs.max_size
            )

            # 自己的连接队列负载
            norm_num_conn_queue = (
                    rsu.connections_queue.size() / rsu.connections_queue.max_size
            )

            # 自己的已连接数负载
            norm_num_connected = rsu.connections.size() / rsu.connections.max_size

            # 组装 Global Observation
            # 包含自己和两个邻居的核心负载指标 (计算负载 + 排队负载)
            # 维度 = 6
            global_obs = (
                    [norm_self_handling_ratio]  # 自身计算负载
                    + [norm_num_conn_queue]  # 自身排队负载
                    + [norm_nb1_h]  # 邻居1 计算负载
                    + [nb1_c]  # 邻居1 排队负载
                    + [norm_nb2_h]  # 邻居2 计算负载
                    + [nb2_c]  # 邻居2 排队负载
            )

            # 组装 Local Observation
            # 包含自身详细状态 + 邻居平均状态
            # 基础维度 = 5
            local_obs = (
                    [norm_self_handling]  # 特征1: 任务数量比
                    + [norm_self_handling_ratio]  # 特征2: 计算负载比
                    + [norm_num_conn_queue]  # 特征3: 排队负载比
                    + [norm_num_connected]  # 特征4: 连接数比
                    + [norm_nb_h]  # 特征5: 邻居平均负载
            )

            # 拼接预测数据到 Local Obs
            if self.use_trajectory_predictor:
                # 从 Map 中获取该 RSU 的预测数据
                # 如果该 RSU 没有任何数据（极少见情况），构建一个全 0 数组兜底
                pred_data = prediction_map.get(rsu.id, {}).get(
                    "final",
                    np.zeros(self.k_vehicles * 3, dtype=np.float32)
                )

                # 将预测数据 (K*3 维) 拼接到 local_obs 后面
                # 最终 local_obs 维度 = 5 + 15 = 20
                local_obs = local_obs + list(pred_data)

            # 构建最终的观测字典
            observations[a] = {
                "local_obs": local_obs,
                "global_obs": global_obs,
                "action_mask": [],  # 暂不使用 Action Mask
            }

            # if self.use_trajectory_predictor:
            #     # 随机抽查一个 agent 的观测
            #     sample_agent = self.env.agents[0]
            #     obs = observations[sample_agent]["local_obs"]
            #
            #     # 获取观测向量的最后几位 (预测部分)
            #     # 假设 K=5, 每个车3个数据(x,y,mask)，共15个数据
            #     pred_len = self.k_vehicles * 3
            #     pred_part = obs[-pred_len:]
            #
            #     # 打印检查
            #     print(f"Agent 0 预测部分数据 (前6个): {pred_part[:6]}")
            #
            #     # 检查是否全是 0
            #     if np.all(pred_part == 0):
            #         print("预测数据全为 0！模型可能未正确加载或输入数据为空。")
            #     else:
            #         print("检测到有效的预测数值。")

        return observations

    def _update_job_handlings(self):
        """
        该方法在每一步仿真后被调用。

        遍历所有 RSU 的当前处理任务列表 (handling_jobs)。
        检查每个任务对应的车辆状态：
           - 车辆是否已离开地图 (not in vehicle_ids)？
           - 车辆是否已断开连接 (not in connections)？
           - 车辆是否切换了连接的 RSU (connected_rsu_id 变了)？
        如果发现无效任务（车辆已离开或断连），则将任务从 RSU 的处理队列中移除。
        调用 `job_deprocess` 方法进行后处理（如更新统计数据）。
        """
        for rsu in self.env.rsus:
            # 备份当前处理列表，用于下一次状态比较
            rsu.pre_handling_jobs.olist = list.copy(rsu.handling_jobs.olist)

            # 遍历处理队列中的所有车辆任务
            for tuple_veh in rsu.handling_jobs:
                if tuple_veh is None:
                    continue
                veh, ratio = tuple_veh  # 解包元组：(车辆对象, 资源分配比例)
                if veh is None:
                    continue
                veh: Vehicle

                # 情况 A: 车辆已完全离开仿真环境，或者已不在任何连接列表中
                if (
                        veh.vehicle_id not in self.env.vehicle_ids  # 车辆离开地图
                        or veh not in self.env.connections  # 车辆不在当前连接集合中
                ):
                    # 将车辆从 RSU 的任务队列中移除
                    rsu.remove_job(elem=veh)
                    # 执行任务去处理逻辑 (如标记失败、释放资源等)
                    veh.job_deprocess(self.env.rsus, self.env.rsu_network)

                # 情况 B: 车辆发生切换 (Handover)，离开了前一个 RSU 的范围
                if (
                        veh.connected_rsu_id != veh.pre_connected_rsu_id
                ):
                    # 从“前一个” RSU (pre_connected_rsu_id) 的列表中移除该车辆
                    self.env.rsus[veh.pre_connected_rsu_id].remove_job(elem=veh)
                    # 执行去处理逻辑
                    veh.job_deprocess(self.env.rsus, self.env.rsu_network)

    def _update_all_rsus_idle(self):
        """
        更新所有 RSU 的空闲 (Idle) 状态。

        第一轮循环：遍历所有 RSU，调用其内部的 `check_idle` 方法。
           `check_idle` 会判断 RSU 及其邻居是否有任务在处理，返回一个状态字典。
           同时更新 RSU 自身的 `idle` 属性。

        第二轮循环：根据第一轮收集到的信息，同步更新每个 RSU 的邻居节点的空闲状态。
           确保每个 RSU 都知道它的邻居是否空闲 (Global Observation 需要此信息)。
        """
        updates = {}
        # 第一阶段：检查自身状态并收集更新信息
        for rsu in self.env.rsus:
            updates[rsu.id] = rsu.check_idle(self.env.rsus, self.env.rsu_network)
            self.env.rsus[rsu.id].idle = updates[rsu.id]["self_idle"]

        # 第二阶段：同步更新邻居状态
        for rsu_id, update in updates.items():
            for neighbor_id, idle_state in update["neighbors_idle"].items():
                self.env.rsus[neighbor_id].idle = idle_state

    def _update_vehicles(self):
        """
        该方法负责将 SUMO 仿真器中的车辆状态同步到 Python 的 `self.env.vehicles` 字典中。
        识别新进入地图的车辆 (New Vehicles)：创建 Vehicle 对象并加入字典。
        识别已离开地图的车辆 (Removed Vehicles)：从字典中删除这些对象。
        更新现有车辆的状态 (Update)：更新每辆车的位置、方向、任务类型等信息。
        """
        # 获取当前 SUMO 仿真中所有的车辆 ID
        current_vehicle_ids = set(self.env.sumo.vehicle.getIDList())
        # 获取上一时刻 Python 环境中记录的车辆 ID
        previous_vehicle_ids = set(self.env.vehicle_ids)

        # 1. 找出新车 (当前有，之前没有)
        new_vehicle_ids = current_vehicle_ids - previous_vehicle_ids
        for vehicle_id in new_vehicle_ids:
            # 实例化 Vehicle 对象，初始化任务内容
            self.env.vehicles[vehicle_id] = Vehicle(
                vehicle_id,
                self.env.sumo,
                self.env.timestep,
                self.env.cache.get_content(
                    min(self.env.timestep // self.env.caching_step, 9)
                ),
            )

        # 2. 找出离开的车 (之前有，当前没有)
        removed_vehicle_ids = previous_vehicle_ids - current_vehicle_ids
        # 使用字典推导式过滤掉已离开的车辆
        self.env.vehicles = {
            veh_ids: vehicle
            for veh_ids, vehicle in self.env.vehicles.items()
            if veh_ids not in removed_vehicle_ids
        }

        # 更新全局 ID 列表
        self.env.vehicle_ids = list(current_vehicle_ids)

        # 3. 更新每辆车的实时状态
        for vehicle in self.env.vehicles.values():
            vehicle.update_pos_direction()  # 更新位置和方向
            # 更新任务类型 (基于时间步变化)
            vehicle.update_job_type(
                self.env.cache.get_content(
                    min(self.env.timestep // self.env.caching_step, 9)
                )
            )

    def _update_connections_queue(self):
        """
                计算所有车辆与 RSU 的距离。
                使用 KDTree (rsu_tree) 快速找到每辆车距离最近的 RSU。
                建立连接：将车辆 ID 加入最近 RSU 的 `range_connections` 列表。
                处理切换 (Handover)：如果最近的 RSU 变了，更新 `connected_rsu_id`。
                维护队列：将不再连接范围内的车辆从 `connections` 列表中移除。
        """
        # 清空旧的连接队列记录
        self.env.connections_queue = []

        # 清空所有 RSU 的范围连接列表
        for rsu in self.env.rsus:
            rsu.range_connections.clear()
            rsu.distances.clear()

        # 遍历每辆车，为其寻找最近的 RSU
        for veh_id, veh in self.env.vehicles.items():
            vehicle_x, vehicle_y = veh.position.x, veh.position.y
            vehicle_coord = np.array([vehicle_x, vehicle_y])

            # 使用 KDTree 查询最近的 1 个 RSU
            distances, sorted_indices = self.env.rsu_tree.query(
                vehicle_coord, k=len(env_config.RSU_POSITIONS)
            )
            idx = sorted_indices[0]  # 最近 RSU 的索引 ID
            dis = distances[0]  # 距离

            # 更新车辆的连接状态
            if veh.connected_rsu_id is not None:
                # 如果之前已连接，记录前一个 ID (用于判断切换)
                veh.pre_connected_rsu_id = veh.connected_rsu_id
                veh.connected_rsu_id = idx
                veh.first_time_caching = True
            else:
                # 如果是首次连接
                veh.pre_connected_rsu_id = idx
                veh.connected_rsu_id = idx

            veh.distance_to_rsu = dis

            # 将车辆加入该 RSU 的范围列表
            rsu = self.env.rsus[idx]
            rsu.range_connections.append(veh.vehicle_id)
            rsu.distances.append(dis)

        # 二次遍历 RSU，清理无效连接
        for rsu in self.env.rsus:
            # 将 range_connections (在范围内) 复制到 connections_queue
            rsu.connections_queue.olist = list.copy(rsu.range_connections.olist)

            # 检查当前已建立连接 (connections) 的车辆是否仍然有效
            for veh in rsu.connections:
                if veh is None:
                    continue
                veh: Vehicle

                # 判断断连条件：
                # 1. 车辆不在地图上了
                # 2. 车辆不在 RSU 的范围内 (range_connections)
                # 3. 车辆连接的 RSU ID 已经变成别的了
                if (
                        veh.vehicle_id not in self.env.vehicle_ids
                        or veh.vehicle_id not in rsu.range_connections
                        or veh.connected_rsu_id != rsu.id
                ):
                    # veh296不知道为什么不会被移除？
                    if veh.vehicle_id == "veh296":
                        pass
                    # 从连接列表中移除
                    rsu.connections.remove(veh)








# not imp
class MappoHandler(Handler):
    def __init__(self, env):
        self.env = env
        pass

    def step(self, actions):
        # if env not reset auto, reset before update env
        if not self.env.sumo_has_init:
            observations, terminations, _ = self.env.reset()

        # random
        random.seed(self.env.seed + self.env.timestep)
        # take action
        if self.env.is_discrete:
            self.take_action(actions)
        else:
            self.env._beta_take_box_actions(actions)
        # caculate rewards
        # dev tag: calculate per timestep? or per fps?
        # calculate frame reward!
        rewards = self.reward()

        # sumo simulation every 10 time steps
        if self.env.timestep % self.env.fps == 0:
            self.env.sumo.simulationStep()
            # update veh status(position and job type) after sim step
            self._update_vehicles()
            # update connections queue, very important
            self._update_connections_queue()
            # remove deprecated jobs 需要在上面的if里吗还是在外面
            self._update_job_handlings()

            self.env.render()

        # update observation space
        observations = self.update_observation()
        # time up or sumo done

        truncations = {a: False for a in self.env.agents}
        infos = {}

        self._update_all_rsus_idle()

        # self.env.sumo.simulation.getMinExpectedNumber() <= 0
        if self.env.timestep >= self.env.max_step:
            # bad transition means real terminal
            terminations = {a: True for a in self.env.agents}
            infos = {
                a: {"bad_transition": True, "idle": self.env.rsus[idx].idle}
                for idx, a in enumerate(self.env.agents)
            }
            print(f"cloud_count:{self.env.cloud_times}")
            print(f"full_count:{self.env.full_count}")
            self.env.sumo.close()
            self.env.sumo_has_init = False
        else:
            infos = {
                a: {"bad_transition": False, "idle": self.env.rsus[idx].idle}
                for idx, a in enumerate(self.env.agents)
            }
            terminations = {a: False for idx, a in enumerate(self.env.agents)}

        self.env.timestep += 1

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed):
        self.env.timestep = 0

        # reset rsus
        self.env.rsus = [
            Rsu(
                id=i,
                position=Point(env_config.RSU_POSITIONS[i]),
                max_connections=self.env.max_connections,
                max_cores=self.env.num_cores,
            )
            for i in range(self.env.num_rsus)
        ]

        # reset content
        if not self.env.content_loaded:
            self.env._content_init()
            self.env.content_loaded = True

        np.random.seed(seed)
        random.seed(seed)
        self.env.seed = seed

        # reset sumo
        if not self.env.sumo_has_init:
            self.env._sumo_init()
            self.env.sumo_has_init = True

        # step once
        # not sure need or not
        # self.env.sumo.simulationStep()

        self.env.vehicle_ids = self.env.sumo.vehicle.getIDList()

        self.env.vehicles = {
            vehicle_id: Vehicle(
                vehicle_id,
                self.env.sumo,
                self.env.timestep,
                self.env.cache.get_content(
                    min(self.env.timestep // self.env.caching_step, 9)
                ),
            )
            for vehicle_id in self.env.vehicle_ids
        }

        self._update_connections_queue()

        self.env.agents = list.copy(self.env.possible_agents)

        observations = self.update_observation()

        self._update_all_rsus_idle()

        infos = {
            a: {"bad_transition": False, "idle": self.env.rsus[idx].idle}
            for idx, a in enumerate(self.env.agents)
        }

        terminations = {a: False for idx, a in enumerate(self.env.agents)}

        infos = {
            a: {"idle": self.env.rsus[idx].idle}
            for idx, a in enumerate(self.env.agents)
        }

        return observations, terminations, infos

    def reward(self):
        rewards = {}
        reward_per_agent = []

        rsu_qoe_dict, caching_ratio_dict = utility.calculate_box_utility(
            vehs=self.env.vehicles,
            rsus=self.env.rsus,
            rsu_network=self.env.rsu_network,
            time_step=self.env.timestep,
            fps=self.env.fps,
            weight=self.env.max_weight,
        )

        self.env.rsu_qoe_dict = rsu_qoe_dict

        for rid, ratio in caching_ratio_dict.items():
            self.env.rsus[rid].hit_ratios.append(ratio)

        for idx, agent in enumerate(self.env.agents):
            # dev tag: factor may need specify
            a = rsu_qoe_dict[idx]

            sum = 0
            if len(a) > 0:
                for r in a:
                    sum += r
                try:
                    flattened = list(itertools.chain.from_iterable(a))
                except TypeError:
                    # 如果展平失败（例如 a 是单个可迭代对象，但不是嵌套的），直接使用 a
                    flattened = list(a)
                rewards[agent] = np.mean(flattened)
            else:
                rewards[agent] = 0.0

        for idx, agent in enumerate(self.env.agents):

            if not self.env.rsus[idx].idle:
                reward_per_agent.append(rewards[agent])

        self.env.ava_rewards.append(np.mean(reward_per_agent))

        return rewards
        pass

    # return obs space and action space
    def spaces_init(self):
        neighbor_num = len(self.env.rsu_network[0])

        # handling_jobs num / all
        # handling_jobs ratio * num / all,
        # queue_connections num / all,
        # connected num /all
        # all neighbor (only) handling_jobs ratio * num / job capicity / neighbor num
        self.env.local_neighbor_obs_space = spaces.Box(0.0, 1.0, shape=(5,))
        # neighbor rsus:
        # avg handling jobs = ratio * num / all job capicity per rsu
        # avg connections = connection_queue.size() / max size
        # 详细邻居情况（包含自己吗？包含的话0是自己）
        self.env.global_neighbor_obs_space = spaces.Box(
            0.0, 1.0, shape=((neighbor_num + 1) * 2,)
        )

        # 事实上由于norm的原因，这个越多基本上random越占优
        # 第一个动作，将connections queue的veh迁移至哪个邻居？0或1，box即0-0.49 0.5-1.0
        # 第一个动作可以改为每个rsu的任务分配比例（自己，邻居1，邻居2），这样适合box，
        # 按比例来搞任务分配的话似乎不太行，random太强势了，可以加个任务max分配大小
        # （max分配大小乘以其任务分配比例即该rsu，算能耗时也可以用到）
        # 所以第二个动作就是# 每个connections 的 max分配大小
        # 这样改的话handling_jobs就是个元组（veh, 任务比例）
        # 第3个动作，将handling jobs内的算力资源进行分配，observation需要修改
        # 第4个动作，将connections内的通信资源进行分配
        # 第5个动作, 分配总算力
        # 第6个动作, 缓存策略 math.floor(caching * self.env.max_content)
        self.env.box_neighbor_action_space = spaces.Box(
            0.0,
            1.0,
            shape=(
                (neighbor_num + 1) * self.env.max_connections
                + self.env.max_connections  # 每个connections 的 max分配大小
                + self.env.num_cores
                + self.env.max_connections
                + 1
                + self.env.max_caching,
            ),
        )

        # 离散化区间
        bins = self.env.bins

        self.env.action_space_dims = [
            self.env.max_connections,  # 动作1: 自己任务分配比例
            self.env.max_connections,  # 动作2: 邻居任务分配比例，与上数相操作可得
            self.env.max_connections,  # 动作3: 每个连接的最大分配大小，可不用但是如果不用random会很高？
            self.env.num_cores,  # 动作4: 算力资源分配
            self.env.max_connections,  # 动作5: 通信资源分配
            1,  # 动作6: 总算力分配，只需一个动作
            self.env.max_caching,  # 动作7: 缓存策略，不需要bin因为本来就是离散的
        ]

        # self.env.action_space_dims = []

        action_space_dims = self.env.action_space_dims

        self.env.md_discrete_action_space = spaces.MultiDiscrete(
            [bins] * action_space_dims[0]
            + [bins] * action_space_dims[1]
            + [bins] * action_space_dims[2]
            + [bins] * action_space_dims[3]
            + [bins] * action_space_dims[4]
            + [bins] * action_space_dims[5]
            + [self.env.max_content] * action_space_dims[6]
        )
        pass

    def take_action(self, actions):
        def _mig_job(rsu: Rsu, action, idx):
            if rsu.connections_queue.is_empty():
                return

            m_actions_self = action[: self.env.action_space_dims[0]]
            pre = self.env.action_space_dims[0]
            m_actions_nb = action[pre : pre + self.env.action_space_dims[1]]
            pre = pre + self.env.action_space_dims[1]
            m_actions_job_ratio = action[pre : pre + self.env.action_space_dims[2]]

            nb_rsu1: Rsu = self.env.rsus[self.env.rsu_network[rsu.id][0]]
            nb_rsu2: Rsu = self.env.rsus[self.env.rsu_network[rsu.id][1]]

            # 取样，这个list可以改为其他
            # 如果取样的是connections_queue，那需要注意prase idx
            for m_idx, ma in enumerate(rsu.connections_queue):

                veh_id: Vehicle = rsu.connections_queue.remove(index=m_idx)

                if veh_id is None:
                    continue

                veh = self.env.vehicles[veh_id]
                real_m_idx = m_idx % self.env.max_connections
                # 防止奖励突变+ 1e-6
                self_ratio = m_actions_self[real_m_idx] / self.env.bins + 1e-6
                nb_ratio = m_actions_nb[real_m_idx] / self.env.bins + 1e-6
                job_ratio = m_actions_job_ratio[real_m_idx] / self.env.bins + 1e-6

                sum_ratio = self_ratio + nb_ratio

                # 0就是不迁移
                if sum_ratio > 0:

                    # 这种都可以靠遍历，如果k>3 需要修改逻辑
                    is_full = [
                        rsu.handling_jobs.is_full(),
                        nb_rsu1.handling_jobs.is_full(),
                        nb_rsu2.handling_jobs.is_full(),
                    ]

                    index_in_rsu = rsu.handling_jobs.index((veh, 0))
                    index_in_nb_rsu1 = nb_rsu1.handling_jobs.index((veh, 0))
                    index_in_nb_rsu2 = nb_rsu2.handling_jobs.index((veh, 0))
                    idxs_in_rsus = [index_in_rsu, index_in_nb_rsu1, index_in_nb_rsu2]
                    in_rsu = [elem is not None for elem in idxs_in_rsus]

                    # 理论上到这里的都是在范围内，可以debug看下是不是
                    # 三个都满了直接cloud
                    if all(is_full):
                        # 都满了，却不在这三个任意一个
                        if not any(in_rsu):
                            veh.is_cloud = True
                            veh.job.is_cloud = True
                            self.env.cloud_times += 1
                            if not veh.job.processing_rsus.is_empty():
                                for rsu in veh.job.processing_rsus:
                                    if rsu is not None:
                                        rsu.remove_job(veh)
                                veh.job.processing_rsus.clear()
                            continue

                    # 假如已在里面只需调整ratio，理论上到这一步基本不会失败因为至少有个非full
                    # 或者至少在一个里面，但是存在只分配成功一个位置，因此需要最后计算ratio

                    self_ratio = self_ratio / sum_ratio
                    nb_ratio = nb_ratio / sum_ratio

                    mig_rsus = np.array([rsu, nb_rsu1, nb_rsu2])

                    mig_ratio = np.array(
                        [
                            self_ratio,
                            nb_ratio / 2,
                            nb_ratio / 2,
                        ]
                    )
                    # 能到这里说明三个rsu至少一个没空或至少有一个在三个rsu里
                    # is_full 为rsu是否满，in_rsu为是否在里面
                    is_full = np.array(is_full)
                    in_rsu = np.array(in_rsu)
                    # 生成布尔掩码
                    update_mask = in_rsu  # 需要更新的 RSU
                    store_mask = (
                        ~is_full & ~in_rsu
                    )  # 需要存入的 RSU，当且仅当没满且没在rsu里

                    # 合并更新和存入的掩码
                    valid_mask = update_mask | store_mask  # 需要更新或存入的 RSU

                    # 只对满足条件的 RSU 的 mig_ratio 进行归一化
                    if np.any(valid_mask):  # 如果有需要更新或存入的 RSU
                        valid_ratios = mig_ratio[valid_mask]  # 提取满足条件的 mig_ratio
                        total_ratio = np.sum(valid_ratios)  # 计算总和
                        if total_ratio > 0:  # 避免除以零
                            mig_ratio[valid_mask] = valid_ratios / total_ratio  # 归一化
                    else:
                        # 异常情况？
                        self.env.full_count += 1
                        continue

                    # 更新
                    for u_idx, u_rsu in enumerate(mig_rsus):
                        if update_mask[u_idx]:
                            u_rsu: Rsu
                            u_rsu.handling_jobs[idxs_in_rsus[u_idx]] = (
                                veh,
                                float(mig_ratio[u_idx] * job_ratio),
                            )

                    # 存入
                    for s_idx, s_rsu in enumerate(mig_rsus):
                        if store_mask[s_idx]:
                            s_rsu: Rsu
                            veh_disconnect: Vehicle = None
                            # connections有可能爆满
                            if veh not in rsu.connections:
                                # 会不会重复connection？
                                # 2.1 重复connection逻辑 fix！改为append！
                                veh_disconnect = rsu.connections.append_and_out(veh)
                                # rsu.connections.append(veh)

                            s_rsu.handling_jobs.append(
                                (veh, float(mig_ratio[s_idx] * job_ratio))
                            )
                            veh.job_process(s_idx, s_rsu)

                            # 假如有veh被断开连接
                            if veh_disconnect is not None:
                                veh_disconnect: Vehicle
                                veh_disconnect.job_deprocess(
                                    self.env.rsus, self.env.rsu_network
                                )
                else:
                    # cloud
                    self.env.cloud_times += 1
                    veh.is_cloud = True
                    veh.job.is_cloud = True
                    if not veh.job.processing_rsus.is_empty():
                        for rsu in veh.job.processing_rsus:
                            if rsu is not None:
                                rsu.remove_job(veh)
                        veh.job.processing_rsus.clear()

                pass

        # env 0
        actions = actions[0]

        # 将 actions 和它们的原始索引组合成元组列表
        indexed_actions = list(enumerate(actions))

        # 随机打乱元组列表或用什么权重方法，
        # 因为如果按顺序后面的车基本不可能能安排到邻居节点
        random.shuffle(indexed_actions)
        num_nb = len(self.env.rsu_network[0])

        for idx, action in indexed_actions:
            rsu: Rsu = self.env.rsus[idx]

            if rsu.idle:
                continue

            _mig_job(rsu=rsu, action=action, idx=idx)

        for idx, action in indexed_actions:
            rsu: Rsu = self.env.rsus[idx]

            if rsu.idle:
                continue

            # resource alloc after all handling
            dims = self.env.action_space_dims
            pre = sum(dims[:3])
            cp_alloc_actions = np.array(action[pre : pre + dims[3]]) / self.env.bins
            # print(f"cp_alloc:{cp_alloc_actions}")
            pre = pre + dims[3]
            bw_alloc_actions = np.array(action[pre : pre + dims[4]]) / self.env.bins
            # print(f"bw_alloc:{bw_alloc_actions}")
            pre = pre + dims[4]
            cp_usage = np.array(action[pre : pre + dims[5]]) / self.env.bins
            # print(f"cp_usage:{cp_usage}")
            pre = pre + dims[5]

            # 已经转为box了
            rsu.box_alloc_cp(alloc_cp_list=cp_alloc_actions, cp_usage=cp_usage)
            rsu.box_alloc_bw(
                alloc_bw_list=bw_alloc_actions, veh_ids=self.env.vehicle_ids
            )

            # independ caching policy here, 也可以每个时间步都caching
            a = action[pre:]

            rsu.frame_cache_content(a, self.env.max_content)

        pass

    def update_observation(self):
        observations = {}

        max_handling_job = self.env.rsus[0].handling_jobs.max_size
        usage_all = 0
        num_handling_job = 0
        count = 0

        for idx, a in enumerate(self.env.agents):
            rsu = self.env.rsus[idx]

            nb_id1, nb_id2 = self.env.rsu_network[idx]
            nb_rsus = [self.env.rsus[nb_id1], self.env.rsus[nb_id2]]

            nb1_h = nb_rsus[0].handling_jobs.olist
            nb2_h = nb_rsus[1].handling_jobs.olist

            norm_nb1_h = (
                sum([v[1] for v in nb1_h if v is not None])
                / nb_rsus[0].handling_jobs.max_size
            )
            norm_nb2_h = (
                sum([v[1] for v in nb2_h if v is not None])
                / nb_rsus[1].handling_jobs.max_size
            )

            nb1_c = (
                nb_rsus[0].connections_queue.size()
                / nb_rsus[0].connections_queue.max_size
            )
            nb2_c = (
                nb_rsus[1].connections_queue.size()
                / nb_rsus[1].connections_queue.max_size
            )

            norm_nb_h = (norm_nb1_h + norm_nb2_h) / 2

            norm_self_handling = rsu.handling_jobs.size() / rsu.handling_jobs.max_size

            norm_self_handling_ratio = (
                sum([v[1] for v in rsu.handling_jobs.olist if v is not None])
                / rsu.handling_jobs.max_size
            )

            norm_num_conn_queue = (
                rsu.connections_queue.size() / rsu.connections_queue.max_size
            )

            norm_num_connected = rsu.connections.size() / rsu.connections.max_size

            global_obs = (
                [norm_self_handling_ratio]
                + [norm_num_conn_queue]
                + [norm_nb1_h]
                + [nb1_c]
                + [norm_nb2_h]
                + [nb2_c]
            )

            local_obs = (
                [norm_self_handling]
                + [norm_self_handling_ratio]
                + [norm_num_conn_queue]
                + [norm_num_connected]
                + [norm_nb_h]
            )
            # act_mask = self.env._single_frame_discrete_action_mask(idx, time_step + 1)
            # act_mask = self.env._single_frame_discrete_action_mask(
            #     self.env.agents.index(a), time_step + 1
            # )

            observations[a] = {
                "local_obs": local_obs,
                "global_obs": global_obs,
                "action_mask": [],
            }

        return observations
        pass

    def _update_job_handlings(self):
        # for rsu in self.env.rsus:
        #     rsu.update_conn_list()

        # 更新所有rsu的handling_jobs，将无效的handling job移除（
        # 例如veh已经离开地图，或者veh已经离开前一个rsu的范围，
        # 这里需要给veh设置一个pre_rsu标识前一个rsu是谁）
        for rsu in self.env.rsus:
            rsu.pre_handling_jobs.olist = list.copy(rsu.handling_jobs.olist)
            for tuple_veh in rsu.handling_jobs:
                if tuple_veh is None:
                    continue
                veh, ratio = tuple_veh
                if veh is None:
                    continue
                veh: Vehicle
                if (
                    veh.vehicle_id not in self.env.vehicle_ids  # 离开地图
                    or veh not in self.env.connections
                ):
                    # 需不需要往前移（最好不要）？以及邻居是否也要移除该veh
                    # 并没有移除！
                    rsu.remove_job(elem=veh)

                    veh.job_deprocess(self.env.rsus, self.env.rsu_network)
                if (
                    veh.connected_rsu_id != veh.pre_connected_rsu_id
                ):  # 离开上一个rsu范围，不保证正确
                    self.env.rsus[veh.pre_connected_rsu_id].remove_job(elem=veh)
                    veh.job_deprocess(self.env.rsus, self.env.rsu_network)
        # may not necessary
        # for rsu in self.rsus:
        #     rsu.job_clean()

    # check_idle的外层循环
    def _update_all_rsus_idle(self):
        # 第一阶段：收集所有 RSU 的邻居更新状态，并更新自身RSU状态
        updates = {}
        for rsu in self.env.rsus:
            updates[rsu.id] = rsu.check_idle(self.env.rsus, self.env.rsu_network)
            self.env.rsus[rsu.id].idle = updates[rsu.id]["self_idle"]

        ...
        # 第二阶段：统一更新所有 RSU 的邻居状态，但不更新自己的状态
        for rsu_id, update in updates.items():

            for neighbor_id, idle_state in update["neighbors_idle"].items():
                self.env.rsus[neighbor_id].idle = idle_state

    def _update_vehicles(self):
        current_vehicle_ids = set(self.env.sumo.vehicle.getIDList())
        previous_vehicle_ids = set(self.env.vehicle_ids)

        # find new veh in map
        new_vehicle_ids = current_vehicle_ids - previous_vehicle_ids
        for vehicle_id in new_vehicle_ids:
            self.env.vehicles[vehicle_id] = Vehicle(
                vehicle_id,
                self.env.sumo,
                self.env.timestep,
                self.env.cache.get_content(
                    min(self.env.timestep // self.env.caching_step, 9)
                ),
            )

        # find leaving veh
        removed_vehicle_ids = previous_vehicle_ids - current_vehicle_ids

        self.env.vehicles = {
            veh_ids: vehicle
            for veh_ids, vehicle in self.env.vehicles.items()
            if veh_ids not in removed_vehicle_ids
        }

        # update vehicle_ids
        self.env.vehicle_ids = list(current_vehicle_ids)

        # update every veh's position and direction
        for vehicle in self.env.vehicles.values():
            vehicle.update_pos_direction()
            # dev tag: update content?
            vehicle.update_job_type(
                self.env.cache.get_content(
                    min(self.env.timestep // self.env.caching_step, 9)
                )
            )

        # vehs need pending job
        # self.pending_job_vehicles = [veh for veh in self.vehicles if not veh.job.done()]

    def _update_connections_queue(self):
        """
        connection logic
        """
        # clear connections
        self.env.connections_queue = []

        # connections update
        for rsu in self.env.rsus:
            rsu.range_connections.clear()
            rsu.distances.clear()

        for veh_id, veh in self.env.vehicles.items():
            vehicle_x, vehicle_y = veh.position.x, veh.position.y
            vehicle_coord = np.array([vehicle_x, vehicle_y])

            # 距离排序
            # only connect 1
            distances, sorted_indices = self.env.rsu_tree.query(
                vehicle_coord, k=len(env_config.RSU_POSITIONS)
            )
            idx = sorted_indices[0]
            dis = distances[0]

            # connected
            if veh.connected_rsu_id is not None:
                veh.pre_connected_rsu_id = veh.connected_rsu_id
                veh.connected_rsu_id = idx
                veh.first_time_caching = True
            else:
                veh.pre_connected_rsu_id = idx
                veh.connected_rsu_id = idx

            veh.distance_to_rsu = dis
            rsu = self.env.rsus[idx]
            rsu.range_connections.append(veh.vehicle_id)
            rsu.distances.append(dis)

        for rsu in self.env.rsus:
            rsu.connections_queue.olist = list.copy(rsu.range_connections.olist)
            # disconnect out of range jobs
            # connections里是object，range_connections是id
            for veh in rsu.connections:

                if veh is None:
                    continue
                veh: Vehicle
                # 车辆已离开
                if (
                    veh.vehicle_id not in self.env.vehicle_ids
                    or veh.vehicle_id not in rsu.range_connections
                    or veh.connected_rsu_id != rsu.id
                ):
                    # veh296不知道为什么不会被移除？
                    if veh.vehicle_id == "veh296":
                        ...
                    rsu.connections.remove(veh)
