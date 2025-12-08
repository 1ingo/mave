import math
import random
import sys
from typing import List

from shapely import Point
from vanet_env import network
from vanet_env import env_config
import traci

sys.path.append("./")

import numpy as np
from vanet_env import utils


# rectange, unit (meters)
# class Road:
#     def __init__(self, x, y, width=20, ):
#         self.x = x
#         self.y = y
#         self.width = width
#         self.height = height


# 定义有序队列列表类
class OrderedQueueList:
    def __init__(self, max_size, init_num=None):
        """
        初始化有序队列列表。

        参数:
        max_size (int): 队列的最大容量
        init_num (int or None): 初始化队列元素的值，默认为 None
        """
        self.max_size = max_size
        if init_num is not None:
            # 若提供了初始值，则用该值填充队列
            self.olist = [init_num] * max_size
        else:
            # 否则，用 None 填充队列
            self.olist = [None] * max_size

    def queue_jumping(self, elem):
        """
        将元素插入队列头部，用于自身任务处理。

        参数:
        elem: 要插入的元素

        返回:
        队列末尾被挤出的元素
        """
        t = self.olist[-1]
        # 将元素插入队列头部
        self.olist = [elem] + self.olist[:-1]
        return t

    # 入队操作
    def append(self, elem):
        """
        将元素添加到队列中第一个为 None 的位置。

        参数:
        elem: 要添加的元素

        返回:
        bool: 如果成功添加，返回 True；否则返回 False
        """
        for i in range(self.max_size):
            if self.olist[i] is None:
                # 找到第一个为 None 的位置，插入元素
                self.olist[i] = elem
                return True
        return False

    # 如果队列满了，淘汰最后一位元素
    def append_and_out(self, elem):
        """
        将元素添加到队列中，如果队列已满则淘汰最后一位元素。

        参数:
        elem: 要添加的元素

        返回:
        若队列已满，返回被淘汰的元素；否则返回 None
        """
        for i in range(self.max_size):
            if self.olist[i] is None:
                # 找到第一个为 None 的位置，插入元素
                self.olist[i] = elem
                return None
        t = self.olist[-1]
        self.olist[-1] = elem
        return t

    def replace(self, elem, index):
        """
        替换指定索引位置的元素。

        参数:
        elem: 要替换的元素
        index (int): 要替换的索引位置

        返回:
        被替换掉的元素
        """
        temp = self.olist[index]
        self.olist[index] = elem
        return temp

    def top_and_sink(self):
        """
        将队列头部元素移到队列尾部。

        返回:
        队列头部元素
        """
        top = self.olist[0]
        # 将头部元素移到队列尾部
        self.olist = self.olist[1:] + [top]
        return top

    def pop(self):
        """
        移除并返回队列头部元素。

        返回:
        队列头部元素
        """
        top = self.olist[0]
        # 移除头部元素，并在末尾添加 None
        self.olist = self.olist[1:] + [None]
        return top

    def remove(self, elem=None, index=-1):
        """
        移除指定元素或指定索引位置的元素。

        参数:
        elem: 要移除的元素，默认为 None
        index (int): 要移除的索引位置，默认为 -1

        返回:
        被移除的元素
        """
        if elem is not None:
            # 如果提供了元素，则找到该元素的索引并移除
            return self.remove(index=self.index(elem))
        else:
            if index is None:
                return None

            if 0 <= index < self.max_size:
                t = self.olist[index]
                # 将指定索引位置置为 None
                self.olist[index] = None
                return t
            else:
                # 若索引超出范围，抛出异常
                assert IndexError("Index out of range")

    def remove_and_shift(self, elem=None, index=-1):
        """
        移除指定元素或指定索引位置的元素，并将后面的元素向前移动。
        不建议使用，移动操作可能会有迭代问题。

        参数:
        elem: 要移除的元素，默认为 None
        index (int): 要移除的索引位置，默认为 -1

        返回:
        被移除的元素
        """
        if elem is not None:
            # 如果提供了元素，则找到该元素的索引并移除
            return self.remove_and_shift(index=self.index(elem))
        else:
            if index is None:
                return None
                # 若索引为 None，抛出异常
                assert IndexError("Index out of range")

            if 0 <= index < self.max_size:
                t = self.olist[index]

                # 将 index 之后的元素向前移动
                while index + 1 < self.max_size:
                    self.olist[index] = self.olist[index + 1]
                    index += 1

                # 最后一个位置设置为 None
                self.olist[self.max_size - 1] = None

                return t
            else:
                # 若索引超出范围，抛出异常
                assert IndexError("Index out of range")

    def to_list_replace_none(self):
        """
        将队列中的 None 替换为 0，并返回列表。

        返回:
        替换后的列表
        """
        return [0 if x is None else x for x in self.olist]

    def size(self):
        """
        计算队列中不为 None 的元素数量。

        返回:
        int: 队列中不为 None 的元素数量
        """
        return sum(1 for elem in self.olist if elem is not None)

    def is_full(self):
        """
        判断队列是否已满。

        返回:
        bool: 如果队列已满，返回 True；否则返回 False
        """
        return self.size() >= self.max_size

    def is_empty(self):
        """
        判断队列是否为空。

        返回:
        bool: 如果队列为空，返回 True；否则返回 False
        """
        return self.size() == 0

    def last(self):
        """
        寻找队列中最后一个不为 None 的元素。

        返回:
        最后一个不为 None 的元素；若队列全为 None，返回 None
        """
        for v in reversed(self.olist):
            if v is not None:
                return v
        return None

    def avg(self):
        """
        计算队列中不为 None 的元素的平均值。

        返回:
        float: 平均值；若队列为空，返回 0
        """
        filtered_values = [v for v in self.olist if v is not None]
        if filtered_values:
            return self.sum() / len(filtered_values)
        else:
            return 0

    def sum(self):
        """
        计算队列中不为 None 的元素的总和。

        返回:
        总和
        """
        filtered_values = [v for v in self.olist if v is not None]
        return sum(filtered_values)

    def __iter__(self):
        """
        使队列可迭代。

        返回:
        队列的迭代器
        """
        return iter(self.olist)

    def __str__(self):
        """
        返回队列的字符串表示形式。

        返回:
        队列的字符串表示
        """
        return str(self.olist)

    def __getitem__(self, index):
        """
        获取指定索引位置的元素。

        参数:
        index (int): 索引位置

        返回:
        指定索引位置的元素；若索引超出范围，抛出异常
        """
        if index < self.max_size:
            return self.olist[index]
        else:
            # 若索引超出范围，抛出异常
            raise IndexError("Index out of range")

    def __setitem__(self, index, value):
        """
        设置指定索引位置的元素。

        参数:
        index (int): 索引位置
        value: 要设置的值

        若索引超出范围，抛出异常
        """
        if 0 <= index < self.max_size:
            self.olist[index] = value
        else:
            # 若索引超出范围，抛出异常
            raise IndexError("Index out of range")

    def __len__(self):
        """
        返回队列的最大容量。

        返回:
        int: 队列的最大容量
        """
        return self.max_size

    def __contains__(self, item):
        """
        支持 in 操作符，判断元素是否在队列中。

        参数:
        item: 要判断的元素

        返回:
        bool: 如果元素在队列中，返回 True；否则返回 False
        """
        if isinstance(item, tuple):
            # 找到第一个匹配 elem[0] 的元组的索引
            for i, elem in enumerate(self.olist):
                if isinstance(elem, tuple) and elem[0] == item[0]:
                    return True

            return False

        if isinstance(item, np.ndarray):  # 如果 item 是 NumPy 数组
            return any(np.array_equal(item, x) for x in self.olist if x is not None)
        else:  # 如果 item 是普通值
            return item in self.olist

    def clear(self):
        """
        清空队列，将队列元素全部置为 None。
        """
        self.olist = [None] * self.max_size

    def index(self, elem):
        """
        查找元素在队列中的索引。

        参数:
        elem: 要查找的元素

        返回:
        元素的索引；若元素为元组，匹配第一个元素；若未找到，返回 None
        """
        if isinstance(elem, tuple):
            # 找到第一个匹配 elem[0] 的元组的索引
            for i, item in enumerate(self.olist):
                if isinstance(item, tuple) and item[0] == elem[0]:
                    return i
            return None

        return self.olist.index(elem)


# a = OrderedQueueList(5)
# a.append((1, "a"))
# a.append((2, "a"))
# a.append((3, "a"))
# a.append((4, "a"))
# a.append((5, "a"))
# a.remove_and_shift(elem=(2, 0))
# a.remove_and_shift(elem=(4, 0))
# # 不存在的话报错或者return none
# a.remove_and_shift(elem=(6, 0))
# a.remove(elem=(3, 1))
# print(a)
# b = OrderedQueueList(5)
# b.append(0)
# b.append(1)
# b.append(2)
# b.remove(2)
# b.remove_and_shift(elem=0)
# print(b)


# 定义路侧单元（RSU）类
class Rsu:
    def __init__(
        self,
        id,
        position: Point,
        bw=env_config.RSU_MAX_TRANSMITTED_BANDWIDTH,
        frequency=env_config.RSU_FREQUENCY,
        transmitted_power=env_config.RSU_TRANSMITTED_POWER,
        height=env_config.RSU_ANTENNA_HEIGHT,
        noise_power=env_config.RSU_NOISE_POWER,
        snr_threshold=env_config.RSU_SNR_THRESHOLD,
        computation_power=env_config.RSU_COMPUTATION_POWER,
        caching_capacity=env_config.RSU_CACHING_CAPACITY,
        num_atn=env_config.RSU_NUM_ANTENNA,
        tx_gain=env_config.ANTENNA_GAIN,
        max_connections=env_config.MAX_CONNECTIONS,
        max_cores=env_config.NUM_CORES,
    ):
        """
        初始化路侧单元（RSU）对象。

        参数:
        id: RSU 的唯一标识符
        position (Point): RSU 的位置，使用 shapely 的 Point 对象表示
        bw (float): RSU 的最大传输带宽，默认为配置文件中的值
        frequency (float): RSU 的工作频率，默认为配置文件中的值
        transmitted_power (float): RSU 的发射功率，默认为配置文件中的值
        height (float): RSU 的天线高度，默认为配置文件中的值
        noise_power (float): RSU 的噪声功率，默认为配置文件中的值
        snr_threshold (float): RSU 的信噪比阈值，默认为配置文件中的值
        computation_power (float): RSU 的计算能力，默认为配置文件中的值
        caching_capacity (int): RSU 的缓存容量，默认为配置文件中的值
        num_atn (int): RSU 的天线数量，默认为配置文件中的值
        tx_gain (float): RSU 的发射增益，默认为配置文件中的值
        max_connections (int): RSU 最大连接数，默认为配置文件中的值
        max_cores (int): RSU 最大核心数，默认为配置文件中的值
        """
        # 初始化 RSU 的基本属性
        self.id = id
        self.position = position
        self.bw = bw
        self.frequency = frequency
        self.transmitted_power = transmitted_power
        self.noise_power = noise_power
        self.height = height
        self.computation_power = computation_power
        self.caching_capacity = caching_capacity
        self.snr_threshold = snr_threshold
        self.tx_gain = tx_gain
        self.max_connections = max_connections
        self.max_cores = max_cores
        self.num_atn = num_atn
        # 初始状态为空闲
        self.idle = False

        # 存储与车辆的距离，使用 OrderedQueueList 管理
        self.distances = OrderedQueueList(max_connections)

        # 后两张表为虚拟的，用以复制前一半表动作
        # 复制自 range_connections，用于更新连接时的队列管理
        self.connections_queue = OrderedQueueList(max_connections * 3)
        # 范围内的连接，仅在更新连接时修改
        self.range_connections = OrderedQueueList(max_connections * 3)
        self.connections = OrderedQueueList(max_connections * 3)
        # 处理作业的队列，可能不是必需的
        self.handling_job_queue = OrderedQueueList(max_cores)
        self.handling_jobs = OrderedQueueList(max_cores * 3)
        self.pre_handling_jobs = OrderedQueueList(max_cores * 3)
        # 带宽分配队列
        self.bw_alloc = OrderedQueueList(max_connections)
        # 计算能力分配队列
        self.computation_power_alloc = OrderedQueueList(max_cores)
        # 实际计算能力分配队列
        self.real_cp_alloc = OrderedQueueList(max_cores)
        # 缓存内容队列
        self.caching_contents = OrderedQueueList(caching_capacity)
        # 能量效率
        self.energy_efficiency = 0

        # 计算能力使用率，最大值为权重
        self.cp_usage = 1
        # 带宽比率
        self.bw_ratio = 5
        # 发射比率
        self.tx_ratio = 100

        # 计算能力分配归一化列表
        self.cp_norm = [0] * self.handling_jobs.max_size
        # 带宽分配归一化列表
        self.bw_norm = [0] * self.connections.max_size

        # 能量效率
        self.ee = 0
        # 最大能量效率
        self.max_ee = 3
        # 效用值
        self.utility = 0

        # 服务质量列表
        self.qoe_list = []
        # 平均效用值
        self.avg_u = 0
        # 命中率列表
        self.hit_ratios = []
        self.storage_capacity = env_config.RSU_STORAGE_CAPACITY  # 总容量 C_r
        self.current_storage_usage = 0.0  # 当前已用空间 sum(s_k * x)
        self.prev_caching_set = set()

    def check_idle(self, rsus, rsu_network):
        """
        检查 RSU 的空闲状态，并确定邻居 RSU 的空闲状态。

        参数:
        rsus (list): 所有 RSU 的列表
        rsu_network (dict): RSU 网络连接信息

        返回:
        dict: 包含当前 RSU 的空闲状态和邻居 RSU 空闲状态更新信息
        """
        # 第一阶段：计算当前 RSU 的 idle 状态，应用并且返回所有邻居应该的状态
        should_idle = True

        # 如果处理作业队列或连接队列不为空，则 RSU 不空闲
        if not self.handling_jobs.is_empty() or not self.connections_queue.is_empty():
            should_idle = False

        # 如果连接列表不为空，则 RSU 不空闲
        if not self.connections.is_empty():
            should_idle = False

        # 只有在 connections_queue 不为空时，才更新邻接 RSU 的状态
        neighbors_idle_updates = {}
        if not self.connections_queue.is_empty():
            for rsu_id in rsu_network[self.id]:
                # 将邻居 RSU 的空闲状态设置为 False
                neighbors_idle_updates[rsu_id] = False

        # 返回当前 RSU 的 idle 状态以及需要更新的邻接 RSU 状态
        # 需要进一步处理
        return {"self_idle": should_idle, "neighbors_idle": neighbors_idle_updates}

    def get_tx_power(self):
        """
        计算 RSU 的发射功率。

        返回:
        float: 计算得到的发射功率
        """
        return self.transmitted_power * self.tx_ratio / 100 + self.tx_gain

    def remove_job(self, elem):
        """
        从处理作业列表中移除指定作业。

        参数:
        elem: 要移除的作业元素
        """
        if isinstance(elem, tuple):
            if elem in self.handling_jobs:
                # 若元素为元组且在处理作业列表中，移除该元素
                self.handling_jobs.remove(elem)
        else:
            if (elem, 0) in self.handling_jobs:
                # 若元素不是元组，以 (elem, 0) 形式查找并移除
                self.handling_jobs.remove((elem, 0))

    def box_alloc_cp(self, alloc_cp_list, cp_usage):
        """
        进行计算能力的分配。

        参数:
        alloc_cp_list (list or np.ndarray): 计算能力分配列表
        cp_usage (float): 计算能力使用率
        """
        # 0 - 1
        self.cp_usage = cp_usage
        # 复制计算能力分配列表
        self.computation_power_alloc.olist = list.copy(alloc_cp_list.tolist())

        ava_alloc = []

        for idx, veh_info in enumerate(self.handling_jobs.olist[: self.max_cores]):
            veh_info1 = self.handling_jobs.olist[idx]
            veh_info2 = self.handling_jobs.olist[idx + self.max_cores]
            veh_info3 = self.handling_jobs.olist[idx + self.max_cores * 2]
            if veh_info1 is not None or veh_info2 is not None or veh_info3 is not None:
                # 若有车辆信息，则添加对应的计算能力分配值
                ava_alloc.append(self.computation_power_alloc[idx])
            else:
                # 否则添加 None
                ava_alloc.append(None)

        # 提高性能，如果可用分配列表全为 None，则直接返回
        if utils.all_none(ava_alloc):
            return

        # 计算可用分配列表的总和
        sum_alloc = np.sum([a if a is not None else 0 for a in ava_alloc])

        if sum_alloc != 0:
            # 若总和不为 0，进行归一化处理
            self.cp_norm = [
                (a / sum_alloc if a is not None else 0)
                for a_idx, a in enumerate(ava_alloc)
            ]
        else:
            # 若总和为 0，抛出异常
            assert NotImplementedError("why you here")

        # 计算实际分配的计算能力
        real_cp = self.computation_power * self.cp_usage
        self.real_cp_alloc.olist = [float(real_cp * cp_n) for cp_n in self.cp_norm]

    def box_alloc_bw(self, alloc_bw_list, veh_ids):
        """
        进行带宽的分配。

        参数:
        alloc_bw_list (list or np.ndarray): 带宽分配列表
        veh_ids (list): 车辆 ID 列表
        """
        # 复制带宽分配列表
        self.bw_alloc.olist = list.copy(alloc_bw_list.tolist())
        from vanet_env import network

        ava_alloc = []
        for idx, veh in enumerate(self.connections.olist[: self.max_connections]):
            veh: Vehicle

            veh1 = self.connections.olist[idx]
            veh2 = self.connections.olist[idx + self.max_connections]
            veh3 = self.connections.olist[idx + self.max_connections * 2]

            if (
                (veh1 is not None and veh1.vehicle_id in veh_ids)
                or (veh2 is not None and veh2.vehicle_id in veh_ids)
                or (veh3 is not None and veh3.vehicle_id in veh_ids)
            ):
                # 若车辆存在且 ID 在指定列表中，添加对应的带宽分配值
                ava_alloc.append(self.bw_alloc[idx % self.max_connections])
            else:
                # 否则添加 None
                ava_alloc.append(None)

        # 提高性能，如果可用分配列表全为 None，则直接返回
        if utils.all_none(ava_alloc):
            return

        # 计算可用分配列表的总和
        sum_alloc = np.sum([a if a is not None else 0 for a in ava_alloc])

        if sum_alloc != 0:
            # 若总和不为 0，进行归一化处理
            self.bw_norm = [
                float(a / sum_alloc) if a is not None else 0
                for a_idx, a in enumerate(ava_alloc)
            ]
        else:
            # 若总和为 0，抛出异常
            assert NotImplementedError("why you here")

        for idx, veh in enumerate(self.connections):
            veh: Vehicle
            if veh is not None and veh.vehicle_id in veh_ids:
                # 若车辆存在且 ID 在指定列表中，更新车辆的数据速率
                veh.data_rate = network.channel_capacity(
                    self,
                    veh,
                    veh.distance_to_rsu,
                    self.bw * self.bw_norm[idx % self.max_connections] * self.num_atn,
                )

    def frame_allocate_computing_power(
        self,
        alloc_index: int,
        cp_a: int,
        cp_usage: int,
        proc_veh_set: set["Vehicle"],
        veh_ids,
    ):
        """
        在一帧内分配计算能力。

        参数:
        alloc_index (int): 分配索引
        cp_a (int): 分配的计算能力
        cp_usage (int): 计算能力使用率
        proc_veh_set (set): 处理车辆集合
        veh_ids (list): 车辆 ID 列表
        """
        self.cp_usage = cp_usage
        # 替换指定索引位置的计算能力分配值
        self.computation_power_alloc.replace(cp_a, alloc_index)
        # 对计算能力分配列表进行归一化处理
        self.cp_norm = utils.normalize_array_np(self.computation_power_alloc)
        veh: Vehicle = self.handling_jobs[alloc_index]
        if veh is not None and veh.vehicle_id in veh_ids:
            # 若车辆存在且 ID 在指定列表中，将车辆添加到处理车辆集合中
            proc_veh_set.add(veh.vehicle_id)

    def _parse_decision(self, caching_decision):
        """
        解析缓存决策，将其转换为去重的内容 ID 列表。

        参数:
        caching_decision: 来自 Handler 的动作输出，通常是 content_id 的列表或数组

        返回:
        list: 包含有效 content_id 的列表
        """
        content_ids = []

        # 处理列表或数组类型的输入
        if isinstance(caching_decision, (list, np.ndarray)):
            seen = set()
            for item in caching_decision:
                # 转换为标准整数 (处理 numpy.int64 等类型)
                c_id = int(item)

                # 简单的去重逻辑：同一帧没必要重复缓存同一个内容
                if c_id not in seen:
                    content_ids.append(c_id)
                    seen.add(c_id)
        # 处理单个数值类型的输入
        else:
            content_ids.append(int(caching_decision))

        return content_ids

    def frame_cache_content(self, caching_decision, num_content, cache_module):
        """
        在一帧内缓存内容。

        参数:
        caching_decision: 缓存决策，可以是单个值或列表
        num_content (int): 内容数量
        """
        # 在更新前，记录当前的缓存状态为 "prev"
        current_list = self.caching_contents.to_list_replace_none()
        # 过滤掉 0 或 None (空槽位)
        self.prev_caching_set = set([c for c in current_list if c is not None and c != 0])
        # 1. 获取决策要缓存的内容列表
        # caching_decision 可能是 [1, 0, 1...] 或者直接是 content_id
        # 假设这里转换为了要缓存的 content_ids 列表
        new_content_ids = self._parse_decision(caching_decision)

        # 2. 计算所需空间
        needed_size = 0
        for cid in new_content_ids:
            needed_size += cache_module.get_size(cid)

        # 3. 简单的 FIFO/LRU 替换策略 (为了满足容量约束)
        # 如果空间不足，移除最早加入的内容
        while self.current_storage_usage + needed_size > self.storage_capacity:
            if self.caching_contents.is_empty():
                break  # 已经空了还是放不下，说明单个内容太大，或者逻辑异常

            # 移除最旧的内容
            removed_id = self.caching_contents.pop()  # 假设 pop 移除队尾
            removed_size = cache_module.get_size(removed_id)
            self.current_storage_usage -= removed_size

        # 4. 存入新内容
        for cid in new_content_ids:
            # 再次检查空间 (防止上面 break 的情况)
            size = cache_module.get_size(cid)
            if self.current_storage_usage + size <= self.storage_capacity:
                self.caching_contents.queue_jumping(cid)
                self.current_storage_usage += size

    def calculate_update_cost(self, rsus, rsu_network, cache_module):
        """
        计算缓存更新产生的总延迟惩罚 T_update
        """

        total_update_cost = 0.0

        # 获取当前(更新后)的缓存集合
        current_list = self.caching_contents.to_list_replace_none()
        current_set = set([c for c in current_list if c is not None and c != 0])

        # 找出本时刻新增的内容: 在当前集合中，但不在上一时刻集合中
        # y_{r,k,t} = 1
        new_contents = current_set - self.prev_caching_set

        for content_id in new_contents:
            # 计算 Fetch Cost
            t_fetch = network.calculate_fetch_delay(
                target_rsu=self,
                content_id=content_id,
                rsus=rsus,
                rsu_network=rsu_network,
                cache_module=cache_module
            )
            total_update_cost += t_fetch

        return total_update_cost

    # notice, cal utility only when connect this rsu
    def frame_allocate_bandwidth(
        self,
        alloc_index: int,
        bw_a: int,
        proc_veh_set: set["Vehicle"],
        veh_ids,
        bw_ratio=1,
    ):
        from vanet_env import network

        self.bw_ratio = bw_ratio
        # 替换指定索引位置的带宽分配值
        self.bw_alloc.replace(bw_a, alloc_index)
        # 对带宽分配列表进行归一化处理
        self.bw_norm = utils.normalize_array_np(self.bw_alloc)

        for idx, veh in enumerate(self.connections):
            veh: Vehicle
            if veh is not None and veh.vehicle_id in veh_ids:
                # 若车辆存在且 ID 在指定列表中，将车辆添加到处理车辆集合中
                proc_veh_set.add(veh.vehicle_id)
                # 更新车辆的数据速率
                veh.data_rate = network.channel_capacity(
                    self,
                    veh,
                    distance=veh.distance_to_rsu,
                    bw=self.bw * self.bw_norm[idx] * self.bw_ratio,
                )

    def allocate_computing_power(
        self, ac_list: list, cp_usage, proc_veh_set: set["Vehicle"]
    ):
        self.cp_usage = cp_usage
        # 复制计算能力分配列表
        self.computation_power_alloc = list.copy(ac_list)
        # 对计算能力分配列表进行归一化处理
        self.cp_norm = utils.normalize_array_np(self.computation_power_alloc)
        ...

    def cache_content(self, caching_decision: list):
        # 找到缓存决策列表中值为 1 的元素的索引，并取前 10 个
        content_index_list = np.where(caching_decision == 1)[0][:10].tolist()
        # 复制内容索引列表到缓存内容列表
        self.caching_contents = list.copy(content_index_list)
        ...

    def allocate_bandwidth(self, abw_list: list, bw_ratio):
        from vanet_env import network

        self.bw_ratio = bw_ratio
        # 复制带宽分配列表
        self.bw_alloc = list.copy(abw_list)
        # 对带宽分配列表进行归一化处理
        self.bw_norm = utils.normalize_array_np(self.bw_alloc)

        # 更新所有或更新一个？
        for idx, veh in enumerate(self.connections):
            if veh is not None:
                # 若车辆存在，更新车辆的数据速率
                veh.data_rate = network.channel_capacity(
                    self, veh, self.bw * self.bw_norm[idx] * self.bw_ratio
                )

    # dev tag: index connect?
    def connect(self, conn, jumping=True, index=-1):
        # 调用连接对象的 connect 方法
        conn.connect(self)

        if jumping:
            # 若插队，断开最后一个连接，并将新连接插入队列头部
            self.disconnect_last()
            self.connections.queue_jumping(conn)
        else:
            if index == -1:
                # 若未指定索引，将连接添加到队列中
                self.connections.append(conn)
            else:
                # 若指定索引，替换指定索引位置的连接
                self.connections.replace(conn, index)

    def disconnect_last(self):
        if self.connections[-1] is not None:
            # 若最后一个连接存在，断开该连接
            self.disconnect(self.connections[-1])

    def disconnect(self, conn):
        # 调用连接对象的 disconnect 方法
        conn.disconnect()
        # 从连接列表中移除该连接
        self.connections.remove(elem=conn)

    def update_conn_list(self):
        # 清理过期的连接
        for idx, conn in enumerate(self.connections):
            if conn is None:
                continue
            if conn not in self.range_connections:
                # 若连接不在范围内，断开该连接
                self.disconnect(conn)

    # def update_job_handling_list(self):
    #     for idx, veh_id in enumerate(self.handling_jobs):
    #          if veh_id is None:
    #             continue

    # clean deprecated jobs
    # for idx, hconn in enumerate(self.handling_jobs):
    #     if hconn is None:
    #         continue
    #     if hconn.connected == False:
    #         self.handling_jobs.remove(idx)

    # python 3.9+
    def frame_handling_job(
        self,
        proc_veh_set: set["Vehicle"],
        rsu: "Rsu",
        h_index: int,
        handling: int,
        veh_ids,
    ):
        # not handling 也 抛弃
        veh: Vehicle = self.handling_job_queue[h_index]

        if veh is not None and veh.vehicle_id in veh_ids:
            # 若车辆存在且 ID 在指定列表中，将车辆添加到处理车辆集合中
            proc_veh_set.add(veh.vehicle_id)

            if handling == 1:
                if veh not in self.handling_jobs:
                    # 若车辆不在处理作业列表中，更新车辆的处理 RSU ID
                    veh.job.processing_rsu_id = self.id
                    # 替换处理作业列表中的元素
                    veh_replaced: Vehicle = self.handling_jobs.replace(
                        elem=veh, index=h_index
                    )
                    # 替换自动云连接
                    if veh_replaced is not None:
                        # 若替换的车辆存在，将其设置为云处理
                        veh_replaced.is_cloud = True
                        veh_replaced.job.processing_rsu_id = env_config.NUM_RSU
            # cloud
            else:
                # 若不处理，将车辆设置为云处理
                veh.is_cloud = True
                veh.job.processing_rsu_id = env_config.NUM_RSU

    # python < 3.11
    def frame_queuing_job(
        self, conn_rsu: "Rsu", veh: "Vehicle", index: int, cloud: bool = False
    ):
        if cloud:
            # there can be modify to more adaptable
            # specify rsu process this
            # veh.connected_rsu_id = config.NUM_RSU
            # 若为云处理，将车辆设置为云处理，并更新作业的处理 RSU ID
            veh.is_cloud = True
            veh.job.processing_rsu_id = env_config.NUM_RSU

        # 唯一一个使is_cloud失效的地方
        # 若不为云处理，将车辆的云处理标志设置为 False
        veh.is_cloud = False
        # 替换处理作业队列中的元素
        self.handling_job_queue.replace(elem=veh, index=index)

    def handling_job(self, jbh_list: list):
        # handle
        for idx, hconn in enumerate(self.handling_job_queue):
            # handle if 1
            if jbh_list[idx]:
                # dev tag: append or direct change?
                # if append, need remove logic
                if hconn not in self.handling_jobs:
                    # 若作业不在处理作业列表中，更新作业的处理 RSU ID
                    hconn.veh.job.processing_rsu_id = self.id
                    # 替换处理作业列表中的元素
                    self.handling_jobs.replace(hconn, idx)
                    # dev tag: replace or append?
                    # 建立与作业的连接
                    hconn.rsu.connect(hconn)
            # else:
            #     hconn.disconnect()
        ...

    def queuing_job(self, conn, cloud=False):
        if cloud:
            # there can be modify to more adaptable
            # 若为云处理，更新连接对象的 RSU 和作业的处理 RSU ID
            conn.rsu = self
            conn.is_cloud = True
            conn.veh.job.processing_rsu_id = env_config.NUM_RSU

        # pending handle
        # if self handling, queue-jumping
        if conn.rsu.id == self.id:
            # 若连接的 RSU 为当前 RSU，将连接插入队列头部
            self.handling_job_queue.queue_jumping(conn)
        else:
            # 否则，将连接添加到队列中
            self.handling_job_queue.append(conn)

    def distance(self, vh_position):
        return np.sqrt(
            (self.position.x - vh_position.x) ** 2
            + (self.position.y - vh_position.y) ** 2
        )

    # to km
    def real_distance(self, vh_position):
        return self.distance(vh_position) / (1000 / env_config.COORDINATE_UNIT)

    def get_d1_d2(self, vh_position: Point, vh_direction):
        # vh_direction is angle in degree, 0 points north, 90 points east ...
        # for convince: 0-45°, 315°-360° is north; 135°-225° is south
        # if (0 <= vh_direction <= 45) or (315 <= vh_direction <= 360):
        #     # "North"
        #     return abs(self.position.y - vh_position.y), abs(
        #         self.position.x - vh_position.x
        #     )
        # elif 135 <= vh_direction <= 225:
        #     # "South"
        #     return abs(self.position.y - vh_position.y), abs(
        #         self.position.x - vh_position.x
        #     )
        # else:
        #     return abs(self.position.x - vh_position.x), abs(
        #         self.position.y - vh_position.y
        #     )

        # 将车辆的行驶方向从角度转换为弧度，因为 Python 的 math 库中的三角函数使用弧度制
        angle_rad = math.radians(vh_direction)

        # 计算RSU相对于车辆的相对位置，即RSU的坐标减去车辆的坐标
        # dx 是 x 轴方向的相对距离，dy 是 y 轴方向的相对距离
        dx = self.position.x - vh_position.x
        dy = self.position.y - vh_position.y

        # 计算车辆行驶方向的单位向量，单位向量的长度为 1，用于后续的投影计算
        # 单位向量的 x 分量使用正弦函数计算，y 分量使用余弦函数计算
        unit_x = math.sin(angle_rad)  # x-component of the unit vector
        unit_y = math.cos(angle_rad)  # y-component of the unit vector

        # 将相对位置向量投影到车辆的行驶方向上，得到水平距离
        # 水平距离是相对位置向量在车辆行驶方向上的投影长度，使用向量点积公式计算
        # 最后取绝对值确保距离为非负
        horizontal_distance = abs(dx * unit_x + dy * unit_y)

        # 计算垂直距离，即相对位置向量在与车辆行驶方向垂直的方向上的投影长度
        # 通过向量的叉积原理，使用相对位置向量与垂直于行驶方向的向量进行计算
        # 最后取绝对值确保距离为非负
        vertical_distance = abs(dx * unit_y - dy * unit_x)

        # 返回垂直距离和水平距离的元组
        return vertical_distance, horizontal_distance

    def has_content(self, content_id):
        """
        检查 RSU 是否缓存了指定内容
        参数:
        content_id: 内容的唯一标识符
        返回:
        bool: 如果缓存命中返回 True，否则 False
        """
        return content_id in self.caching_contents


class Job:
    def __init__(self, job_id, job_size, job_type):
        # 初始化作业对象
        # job_id: 作业的唯一标识符
        self.job_id = job_id
        # job_size: 作业的大小，此属性已弃用
        self.job_size = job_size
        # job_type: 作业的类型
        self.job_type = job_type
        # qoe: 作业的服务质量，初始值为 0
        self.qoe = 0
        # pre_qoe: 作业上一次的服务质量，初始值为 None
        self.pre_qoe = None
        # pre_trans_qoe: 作业上一次的传输服务质量，初始值为 0
        self.pre_trans_qoe = 0
        # pre_proc_qoe: 作业上一次的处理服务质量，初始值为 0
        self.pre_proc_qoe = 0
        # trans_qoe: 作业的当前传输服务质量，初始值为 0
        self.trans_qoe = 0
        # proc_qoe: 作业的当前处理服务质量，初始值为 0
        self.proc_qoe = 0
        # job_processed: 作业已处理的大小，此属性已弃用
        self.job_processed = 0
        # is_cloud: 作业是否在云端处理，初始值为 False
        self.is_cloud = False
        # processing_rsus: 处理该作业的路侧单元（RSU）列表，最大长度为 3
        self.processing_rsus = OrderedQueueList(3)

    # 此方法已弃用
    def done(self):
        # 判断作业是否完成，通过比较作业大小和已处理大小
        return self.job_size - self.job_processed <= 0

    # 此方法已弃用
    def job_info(self):
        # 返回作业剩余未处理的大小
        return max(self.job_size - self.job_processed, 0)


class Vehicle:
    def __init__(
        self,
        vehicle_id,
        transpower,
        sumo,
        join_time,
        init_all=True,
        seed=env_config.SEED,
        max_connections=4,
    ):
        # 初始化车辆对象
        self.vehicle_id = vehicle_id
        # height: 车辆天线的高度，从环境配置中获取
        self.height = env_config.VEHICLE_ANTENNA_HEIGHT

        self.transpower = env_config.VEHICLE_TRANSMITTED_POWER
        # position: 车辆的位置，初始值为 None
        self.position = None
        # angle: 车辆的角度，初始值为 None
        self.angle = None
        # sumo: SUMO 仿真对象，用于获取车辆的状态信息
        self.sumo = sumo
        # seed: 随机数种子，用于生成作业大小
        self.seed = seed
        # join_time: 车辆加入仿真的时间
        self.join_time = join_time
        # first_time_caching: 车辆是否首次进行缓存操作，初始值为 True
        self.first_time_caching = True


        # 设置随机数种子
        random.seed(self.seed)

        # 随机生成作业大小，范围在 8 到环境配置的最大作业大小之间
        job_size = random.randint(8, env_config.MAX_JOB_SIZE)

        # job_type: 作业的类型
        job_type = random.randint(0, env_config.NUM_CONTENT - 1)

        # job: 车辆的作业对象，作业 ID 为车辆 ID
        self.job = Job(vehicle_id, job_size, job_type)

        if init_all:
            # 如果 init_all 为 True，则初始化车辆的位置和角度
            self.position = Point(sumo.vehicle.getPosition(vehicle_id))
            self.angle = sumo.vehicle.getAngle(self.vehicle_id)

        # is_cloud: 车辆是否连接到云端，初始值为 False
        self.is_cloud = False
        # connected_rsu_id: 车辆当前连接的 RSU 的 ID，初始值为 None
        self.connected_rsu_id = None
        # pre_connected_rsu_id: 车辆上一次连接的 RSU 的 ID，初始值为 None
        self.pre_connected_rsu_id = None

        # data_rate: 车辆的数据传输速率，初始值为 0
        self.data_rate = 0
        # distance_to_rsu: 车辆到 RSU 的距离，初始值为 None
        self.distance_to_rsu = None
        # connected rsus, may not needed
        # self.connections = OrderedQueueList(max_connections)

    def job_process(self, idx, rsu):
        # 开始处理作业
        # idx: 处理该作业的 RSU 在 processing_rsus 列表中的索引
        # rsu: 处理该作业的 RSU 对象
        self.job.is_cloud = False
        self.is_cloud = False
        # 将处理该作业的 RSU 添加到 processing_rsus 列表中
        self.job.processing_rsus[idx] = rsu

    def job_deprocess(self, rsus, rsu_network):
        # 停止处理作业
        self.job.is_cloud = True
        self.is_cloud = True

        if not self.job.processing_rsus.is_empty():
            # 如果 processing_rsus 列表不为空
            for rsu in self.job.processing_rsus:
                if rsu is None:
                    continue
                rsu: Rsu
                # 从 RSU 的处理作业列表中移除该作业
                rsu.remove_job(elem=self)
                # 获取该 RSU 的两个邻居 RSU 的 ID
                nb_id1, nb_id2 = rsu_network[rsu.id]
                # 从邻居 RSU 的处理作业列表中移除该作业
                rsus[nb_id1].remove_job(elem=self)
                rsus[nb_id2].remove_job(elem=self)

        # 清空 processing_rsus 列表
        self.job.processing_rsus.clear()

    # 仅在 Connection.disconnect() 方法中使用
    def disconnect(self, conn):
        # 断开与某个连接的关联
        if conn in self.connections:
            # 如果该连接存在于车辆的连接列表中，则移除该连接
            self.connections.remove(elem=conn)

    def update_job_type(self, job_type):
        # 更新车辆作业的类型
        self.job_type = job_type

    def update_pos_direction(self):
        # 更新车辆的位置和角度
        self.position = Point(self.sumo.vehicle.getPosition(self.vehicle_id))
        self.angle = self.sumo.vehicle.getAngle(self.vehicle_id)

    def get_speed(self):
        # 获取车辆的速度
        return self.sumo.vehicle.getSpeed(self.vehicle_id)

    def set_speed(self, speed):
        # 设置车辆的速度
        self.sumo.vehicle.setSpeed(self.vehicle_id, speed)

    def get_position(self):
        # 获取车辆的位置
        return (
            self.position
            if self.position is not None
            else self.sumo.vehicle.getPosition(self.vehicle_id)
        )

    def get_angle(self):
        # 获取车辆的角度
        return (
            self.angle
            if self.angle is not None
            else self.sumo.vehicle.getAngle(self.vehicle_id)
        )


class CustomVehicle(Vehicle):
    def __init__(
        self,
        id,
        position: Point,
        sumo=traci,
        height=env_config.VEHICLE_ANTENNA_HEIGHT,
        transpower=env_config.VEHICLE_TRANSMITTED_POWER,
        direction=0,
    ):
        # 初始化自定义车辆对象，继承自 Vehicle 类
        # id: 车辆的唯一标识符
        self.id = id
        # position: 车辆的位置
        self.position = position
        # height: 车辆天线的高度，从环境配置中获取
        self.height = height

        self.transpower = transpower
        # speed: 车辆的速度，初始值为 0
        self.speed = 0
        # acceleration: 车辆的加速度，初始值为 0
        self.acceleration = 0
        # direction: 车辆的方向，0 表示北，1 表示南，2 表示西，3 表示东
        self.direction = direction
        # sumo: SUMO 仿真对象，用于获取车辆的状态信息
        self.sumo = sumo

    def get_speed(self):
        # 获取车辆的速度
        return self.sumo.vehicle.getSpeed(self.vehicle_id)

    def set_speed(self, speed):
        # 设置车辆的速度
        self.sumo.vehicle.setSpeed(self.vehicle_id, speed)

    def get_position(self):
        # 获取车辆的位置
        return self.position

    def get_angle(self):
        # 根据车辆的方向返回对应的角度
        if self.direction == 0:
            return 0
        elif self.direction == 1:
            return 180
        elif self.direction == 2:
            return 270
        else:
            return 90


class Connection:
    def __init__(self, rsu: Rsu, veh: Vehicle, data_rate=0.0, cloud=False):
        # 初始化连接对象
        # is_cloud: 连接是否是到云端，初始值由 cloud 参数决定
        self.is_cloud = cloud

        # veh: 连接的车辆对象
        self.veh = veh
        # rsu: 连接的 RSU 对象，如果是连接到云端则为 None
        self.rsu = None if self.is_cloud else rsu
        # data_rate: 连接的数据传输速率，初始值为 0.0
        self.data_rate = data_rate
        # qoe: 连接的服务质量，初始值为 0
        self.qoe = 0
        # connected: 连接是否已建立，初始值为 False
        self.connected = False
        # id: 连接的唯一标识符，由 RSU 的 ID 和车辆的 ID 组成
        self.id = str(rsu.id) + veh.vehicle_id

    def check_connection(self):
        # 检查连接状态
        # 如果连接未建立，则返回是否连接到云端的状态；否则返回 True 表示已连接
        return self.is_cloud if not self.connected else True

    def connect(self, rsu, is_cloud=False):
        """
        only connect when take action
        """
        # 建立连接，仅在采取行动时调用
        if is_cloud:
            # 如果连接到云端
            self.is_cloud = True
            # RSU 对象设为 None
            self.rsu = None
        else:
            # 如果连接到 RSU
            self.rsu = rsu
            # 标记为已连接
            self.connected = True

    # process by rsu
    def disconnect(self):
        # 断开连接，由 RSU 处理
        # 标记为不连接到云端
        self.is_cloud = False
        # 标记为未连接
        self.connected = False
        # 调用车辆对象的 disconnect 方法，从车辆的连接列表中移除该连接
        self.veh.disconnect(self)

    def __eq__(self, other):
        # 定义连接对象的相等比较方法
        if other is None:
            # 如果比较对象为 None，则返回 False
            return False
        # 比较两个连接对象的唯一标识符是否相等
        return self.id == other.id