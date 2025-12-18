import collections
import functools
import itertools
import math
import random
import sys

from vanet_env import utility
from vanet_env import env_config

sys.path.append("./")
import pandas as pd
from shapely import Point
from sklearn.preprocessing import MinMaxScaler

from vanet_env import data_preprocess

sys.path.append("./")

import os
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import parallel_to_aec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as transforms
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import poisson

# gym
import gymnasium as gym
from gymnasium import spaces
from gym import spaces as gym_spaces

# custom
from vanet_env import network, caching, handler
from vanet_env.utils import (
    RSU_MARKER,
    VEHICLE_MARKER,
    interpolate_color,
    sumo_detector,
    is_empty,
    discrete_to_multi_discrete,
    multi_discrete_space_to_discrete_space,
)
from vanet_env.entites import Connection, Rsu, CustomVehicle, Vehicle

# sumo
import traci
import libsumo
from sumolib import checkBinary

# next performance improvement
import multiprocessing as mp

import datetime
import torch
from torch_geometric.data import Data

# 定义一个函数，用于创建原始环境
def raw_env(render_mode=None):
    # 创建 Env 类的实例
    env = Env(render_mode=render_mode)
    # 将并行环境转换为回合制环境
    env = parallel_to_aec(env)
    return env

# 定义 Env 类，继承自 ParallelEnv
class Env(ParallelEnv):
    # 定义环境的元数据，包括环境名称和支持的渲染模式
    metadata = {"name": "sumo_vanet_environment_v0", "render_modes": ["human", None]}

    def __init__(
        self,
        render_mode="human",
        multi_core=8,
        caching_fps=10,  # 将时间分割为 10 份
        fps=10,
        max_step=36000,  # 需要关注帧率
        seed=env_config.SEED,
        is_discrete=True,
        handler=handler.TrajectoryHandler,
        map="seattle",
    ):
        # 是否使用离散动作空间
        self.is_discrete = is_discrete
        # RSU（路侧单元）的数量
        self.num_rsus = env_config.NUM_RSU
        # 道路宽度
        self.road_width = env_config.ROAD_WIDTH
        # 随机种子
        self.seed = seed
        # 渲染模式
        self.render_mode = render_mode
        # 多核数量
        self.multi_core = multi_core
        # 最大步数
        self.max_step = max_step
        # 缓存帧率
        self.caching_fps = caching_fps
        # 缓存步数
        self.caching_step = max_step // caching_fps
        # 缓存命中率字典
        self.caching_ratio_dict = []
        # 可用奖励列表
        self.ava_rewards = []
        # 使用的地图名称
        self.map = map

        # 设置随机种子
        random.seed(self.seed)

        # RSU 的最大连接数
        self.max_connections = env_config.MAX_CONNECTIONS
        # RSU 的核心数量
        self.num_cores = env_config.NUM_CORES
        # 最大内容数量
        self.max_content = env_config.NUM_CONTENT
        # RSU 的缓存容量
        self.max_caching = env_config.RSU_CACHING_CAPACITY
        # RSU 连接队列的最大长度
        self.max_queue_len = 10
        # 单个权重
        self.max_weight = 10
        # QoE（体验质量）权重
        self.qoe_weight = 10
        # 分箱数量
        self.bins = 5
        # 帧率
        self.fps = fps

        # 初始化 RSU 列表
        self.rsus = [
            Rsu(
                id=i,
                position=Point(env_config.RSU_POSITIONS[i]),
                max_connections=self.max_connections,
                max_cores=self.num_cores,
            )
            for i in range(self.num_rsus)
        ]
        # RSU 的 ID 列表
        self.rsu_ids = [i for i in range(self.num_rsus)]
        # 代理的 ID 列表
        self._agent_ids = [f"rsu_{i}" for i in range(self.num_rsus)]
        # 全局平均效用
        self.avg_u_global = 0
        # 局部平均效用
        self.avg_u_local = 0

        # 单个 RSU 的最大数据速率
        self.max_data_rate = network.max_rate(self.rsus[0])
        print(f"single atn max_data_rate:{self.max_data_rate:.2f}")

        # RSU 的位置列表
        self.rsu_positions = [rsu.position for rsu in self.rsus]
        # RSU 的坐标数组
        self.rsu_coords = np.array(
            [
                (self.rsu_positions[rsu.id].x, self.rsu_positions[rsu.id].y)
                for rsu in self.rsus
            ]
        )
        # 用于快速查找 RSU 的 KD 树
        self.rsu_tree = KDTree(self.rsu_coords)

        # 连接队列
        self.connections_queue = []
        # 连接列表
        self.connections = []
        # RSU 网络
        self.rsu_network = network.network(self.rsu_coords, self.rsu_tree)

        # RSU 的最大连接距离
        self.max_distance = network.max_distance_mbps(
            self.rsus[0], rate_tr=env_config.DATA_RATE_TR * 2
        )
        print(f"max_distance:{self.max_distance}")
        # SUMO 接口
        self.sumo = traci

        # 复制代理 ID 列表
        list_agents = list(self._agent_ids)
        # 当前活动的代理列表
        self.agents = list.copy(list_agents)
        # 所有可能的代理列表
        self.possible_agents = list.copy(list_agents)

        # 当前时间步
        self.timestep = 0
        # 监测无法编排的次数
        self.full_count = 0
        # 监测云调用的次数
        self.cloud_times = 0

        # 内容是否加载完成的标志
        self.content_loaded = False
        # 处理类
        self.handler_class = handler

        # 初始化处理类
        self._handler_init()
        # 初始化空间
        self._space_init()
        # 初始化 SUMO 仿真
        self._sumo_init()

    def _handler_init(self):
        # 创建处理类的实例
        self.handler = self.handler_class(self)

    def _content_init(self):
        # 创建缓存类的实例
        self.cache = caching.Caching(
            self.caching_fps, self.max_content, self.max_caching, self.seed
        )
        # 获取内容列表、聚合数据帧列表和聚合数据帧
        self.content_list, self.aggregated_df_list, self.aggregated_df = (
            self.cache.get_content_list()
        )
        # 获取当前时间步对应的内容
        self.cache.get_content(min(self.timestep // self.caching_step, 9))

    def _sumo_init(self):
        # 打印 SUMO 初始化信息
        print("sumo init")

        # SUMO 配置文件的路径
        self.cfg_file_path = os.path.join(
            os.path.dirname(__file__), "assets", self.map, "sumo", "osm.sumocfg"
        )

        # GUI 设置文件的路径，此处被注释掉
        # gui_settings_path = os.path.join(
        #     os.path.dirname(__file__), "assets", self.map, "sumo", "gui_hide_all.xml"
        # )

        # RSU 图标文件的路径
        self.icon_path = os.path.join(os.path.dirname(__file__), "assets", "rsu.png")

        # 如果渲染模式不为 None，则进行可视化设置
        if self.render_mode is not None:
            # 初始化数据列表
            self.steps = []
            self.utilities = []
            self.caching_ratios = []

            # 开启交互模式
            plt.ion()
            # 隐藏工具栏
            plt.rcParams["toolbar"] = "None"

            # 创建两个子图，上下排列
            self.fig, (self.ut_ax, self.ca_ax) = plt.subplots(
                2, 1, figsize=(8, 6)
            )
            # 创建初始的效用曲线
            (self.ut_line,) = self.ut_ax.plot(
                self.steps, self.utilities, "r-", label="Utility"
            )
            # 设置 x 轴标签
            self.ut_ax.set_xlabel("Step")
            # 设置 y 轴标签
            self.ut_ax.set_ylabel("Averge Utility")
            # 设置标题
            self.ut_ax.set_title("Utility over Steps")
            # 显示图例
            self.ut_ax.legend()

            # 创建初始的缓存命中率曲线
            (self.ca_line,) = self.ca_ax.plot(
                self.steps, self.caching_ratios, "b-", label="Hit Ratio"
            )
            # 设置 x 轴标签
            self.ca_ax.set_xlabel("Step")
            # 设置 y 轴标签
            self.ca_ax.set_ylabel("Averge Caching Hit Ratio")
            # 设置标题
            self.ca_ax.set_title("Caching Ratio over Steps")
            # 显示图例
            self.ca_ax.legend()

            # 调整子图间距
            plt.tight_layout()
            # 获取当前图形管理器
            fig_manager = plt.get_current_fig_manager()
            if hasattr(fig_manager, "window"):
                # 置顶窗口
                fig_manager.window.attributes("-topmost", 1)
                # 设置窗口大小和位置，此处被注释掉
                # fig_manager.window.geometry(
                #     "200x500+1300+300"
                # )

            # 启动 SUMO GUI 仿真
            self.sumo.start(
                ["sumo-gui", "-c", self.cfg_file_path, "--step-length", "1", "--start"]
            )
            # 在 SUMO 中添加 RSU 图标
            for rsu in self.rsus:
                poi_id = f"rsu_icon_{rsu.id}"
                self.sumo.poi.add(
                    poi_id,
                    rsu.position.x,
                    rsu.position.y,
                    (205, 254, 194, 255),  # 绿色
                    width=10,
                    height=10,
                    imgFile=self.icon_path,
                    layer=20,
                )

            # 绘制资源容量，此处被注释掉
            # for rsu in self.rsus:
            #     max_ee = env_config.MAX_EE
            #     offset = 2
            #     width = 2
            #     height = 1
            #     x1 = rsu.position.x + offset
            #     y1 = rsu.position.y
            #     x2 = rsu.position.x + offset
            #     y2 = rsu.position.y + height
            #     self.sumo.polygon.add(
            #         f"resource_rsu{rsu.id}",
            #         [(x1, y1), (x2, y2)],
            #         color=(205, 254, 194, 255),
            #         fill=False,
            #         lineWidth=2,
            #         layer=40,
            #     )

            # 绘制 RSU 的覆盖范围
            for rsu in self.rsus:
                num_segments = 36
                for i in range(num_segments):
                    angle1 = 2 * np.pi * i / num_segments
                    angle2 = 2 * np.pi * (i + 1) / num_segments
                    x1 = rsu.position.x + self.max_distance * np.cos(angle1)
                    y1 = rsu.position.y + self.max_distance * np.sin(angle1)
                    x2 = rsu.position.x + self.max_distance * np.cos(angle2)
                    y2 = rsu.position.y + self.max_distance * np.sin(angle2)
                    self.sumo.polygon.add(
                        f"circle_segment_rsu{rsu.id}_{i}",
                        [(x1, y1), (x2, y2)],
                        color=(255, 0, 0, 255),
                        fill=False,
                        lineWidth=0.2,
                        layer=20,
                    )
        else:
            # 检查 SUMO 二进制文件
            sumoBinary = checkBinary("sumo")

            # 启动无 GUI 的 SUMO 仿真
            self.sumo = libsumo
            libsumo.start(["sumo", "-c", self.cfg_file_path])
            # traci.start([sumoBinary, "-c", cfg_file_path])

        # 获取 SUMO 网络边界
        net_boundary = self.sumo.simulation.getNetBoundary()
        # 地图大小
        self.map_size = net_boundary[1]
        # SUMO 是否初始化完成的标志
        self.sumo_has_init = True

    def _space_init(self):
        # 初始化处理类的空间
        self.handler.spaces_init()

    def reset(self, seed=env_config.SEED, options=None):
        # 记录开始时间
        self.start_time = datetime.datetime.now()
        # 调用处理类的重置方法
        return self.handler.reset(seed)

    def step(self, actions):
        # 调用处理类的步骤方法
        return self.handler.step(actions)

    # 改进：返回多边形然后后面统一绘制？
    def _render_connections(self):
        # 用于批量添加多边形的列表
        polygons_to_add = []

        # 绘制资源利用率
        for rsu in self.rsus:
            rsu: Rsu
            max_ee = env_config.MAX_EE
            offset = 10
            width = 6
            height = 10

            # 计算资源利用率的高度
            x1 = rsu.position.x + offset
            y1 = rsu.position.y
            x2 = rsu.position.x + offset
            y2 = rsu.position.y + height * max(rsu.cp_usage, 0.1)

            # 根据 RSU 是否空闲设置颜色
            if not rsu.idle:
                color = interpolate_color(
                    0, 1, rsu.cp_usage, is_reverse=True
                )  # 越高越红，越低越绿
            else:
                y2 = rsu.position.y + height * 0.1
                color = interpolate_color(
                    0, 1, 0.1, is_reverse=True
                )  # 越高越红，越低越绿

            # 添加透明度
            color_with_alpha = (*color, 255)

            # 将多边形信息添加到列表中
            polygons_to_add.append(
                (
                    f"dynamic_resource_rsu{rsu.id}",
                    [
                        (
                            x1,
                            y1,
                        ),
                        (x2, y2),
                    ],
                    color_with_alpha,
                    False,
                    width,
                )
            )

        # 绘制 QoE（体验质量）
        for rsu in self.rsus:
            rsu: Rsu
            for veh in rsu.connections:
                if veh is None:
                    continue

                veh: Vehicle

                # 计算最大 QoE
                max_qoe = max(env_config.MAX_QOE * 0.7, veh.job.qoe)

                # 根据 QoE 值插值颜色
                color = interpolate_color(0, max_qoe * 0.7, veh.job.qoe)
                # 添加透明度
                color_with_alpha = (*color, 255)

                # 将多边形信息添加到列表中
                polygons_to_add.append(
                    (
                        f"dynamic_line_rsu{rsu.id}_to_{veh.vehicle_id}",
                        [
                            (
                                rsu.position.x,
                                rsu.position.y,
                            ),
                            (veh.position.x, veh.position.y),
                        ],
                        color_with_alpha,
                        False,
                        0.3,
                    )
                )

        # 批量添加多边形到 SUMO 仿真中
        for polygon_id, points, color, is_fill, line_width in polygons_to_add:
            self.sumo.polygon.add(
                polygon_id,
                points,
                color=color,
                fill=is_fill,
                lineWidth=line_width,
                layer=41,
            )

    def render(self, mode=None):
        # 仅在特定时间步进行渲染
        if self.timestep % self.fps == 0:
            # 获取渲染模式
            mode = self.render_mode if mode is None else mode
            # 如果渲染模式不为 None，则进行可视化
            if mode is not None:
                # 计算平均效用
                mean_ut = np.nanmean(self.ava_rewards)
                cas = []

                # 计算平均缓存命中率
                for rsu in self.rsus:
                    cas = np.nanmean(rsu.hit_ratios)

                # 如果平均效用不为 NaN，则添加到效用列表中
                if not np.isnan(mean_ut):
                    self.utilities.append(mean_ut)

                # 更新数据
                if self.utilities:
                    self.steps.append(self.timestep)
                    self.caching_ratios.append(np.mean(cas))

                    # 更新效用曲线
                    self.ut_line.set_xdata(self.steps)
                    self.ut_line.set_ydata(self.utilities)
                    self.ut_ax.relim()  # 重新计算坐标轴范围
                    self.ut_ax.autoscale_view()  # 自动调整坐标轴范围

                    # 更新缓存命中率曲线
                    self.ca_line.set_xdata(self.steps)
                    self.ca_line.set_ydata(self.caching_ratios)
                    self.ca_ax.relim()  # 重新计算坐标轴范围
                    self.ca_ax.autoscale_view()  # 自动调整坐标轴范围
                    # 绘制图形
                    plt.draw()
                    # 刷新图形事件
                    self.fig.canvas.flush_events()

                # 清除所有动态渲染的多边形
                for polygon_id in self.sumo.polygon.getIDList():
                    if polygon_id.startswith("dynamic_"):
                        self.sumo.polygon.remove(polygon_id)

                # 渲染连接信息
                self._render_connections()

                return

        return

    def close(self):
        # 关闭 SUMO 仿真
        self.sumo.close()
        # 设置 SUMO 初始化标志为 False
        self.sumo_has_init = False
        # 关闭交互模式
        plt.ioff()
        # 关闭图形窗口
        plt.close()
        # 记录结束时间
        self.endtime = datetime.datetime.now()

        # 调用父类的关闭方法
        return super().close()

    def get_graph_observation(self):
        node_features = []
        edge_index = []
        edge_attr = []

        # 1. 构建节点特征 (Node Features)
        # 假设特征为: [历史请求频率向量 (num_content) + 当前缓存状态 (num_content)]
        # 这里需要你在 Rsu 类或 Env 中维护一个请求频率记录
        for rsu in self.rsus:
            # 获取缓存状态 (来自 entites.py 的修改)
            cache_vec = rsu.get_cache_one_hot(self.max_content)
            # 获取请求模式 (这里假设你有一个变量 request_counts)
            # req_vec = rsu.get_request_pattern(...)

            # 简单起见，这里先只用缓存状态演示
            node_features.append(cache_vec)

        x = torch.tensor(np.array(node_features), dtype=torch.float)

        # 2. 构建边和边特征 (Edges & Edge Attributes)
        # self.rsu_network 已经是邻接表: {rsu_id: [neighbor_id, ...]}
        src_list = []
        dst_list = []
        weights = []

        for src_id, neighbors in self.rsu_network.items():
            src_rsu = self.rsus[src_id]
            for dst_id in neighbors:
                dst_rsu = self.rsus[dst_id]

                src_list.append(src_id)
                dst_list.append(dst_id)

                # **核心逻辑**: 边特征 = R2R 延迟
                # 使用 network.py 中的逻辑估算延迟
                # 这里简化为距离产生的延迟，或者直接用 hops * latency
                dist = src_rsu.real_distance(dst_rsu.position)
                # 归一化延迟作为权重 (越小越好，但 GNN 通常处理“强度”，可能需要取倒数或在模型内部处理)
                latency = dist / 1000.0  # 简单示例
                weights.append([latency])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(weights, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # 使用 LRU 缓存来提高性能
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # 返回局部邻居观测空间
        return self.local_neighbor_obs_space

    # 使用 LRU 缓存来提高性能
    @functools.lru_cache(maxsize=None)
    def global_observation_space(self, agent):
        # 返回全局邻居观测空间
        return self.global_neighbor_obs_space

    # 使用 LRU 缓存来提高性能
    @functools.lru_cache(maxsize=None)
    def local_observation_space(self, agent):
        # 返回局部邻居观测空间
        return self.local_neighbor_obs_space

    # 使用 LRU 缓存来提高性能
    @functools.lru_cache(maxsize=None)
    def multi_discrete_action_space(self, agent):
        # 返回多离散动作空间
        return self.md_discrete_action_space

    # 使用 LRU 缓存来提高性能
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # 返回连续动作空间
        return self.box_neighbor_action_space

    # def get_agent_ids(self):
    #     return self._agent_ids
