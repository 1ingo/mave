from collections import defaultdict, deque
import sys

from shapely import Point

sys.path.append("./")
from vanet_env import env_config
from vanet_env.entites import Rsu, CustomVehicle, Vehicle
from vanet_env import utils
import numpy as np



# Okumura-Hata 模型类，用于计算路径损耗
class OkumuraHata:

    def __init__(self):
        # 定义一个极小的常量，用于避免在计算对数时出现除以零的错误
        self.EPSILON = 1e-9
        pass

    # Okumura-Hata 路径损耗的另一个版本（来自 GitHub）
    # 可能存在一些未修复的 bug
    def power_loss(self, rsu: Rsu, vh: CustomVehicle):
        # 计算 RSU 与车辆之间的实际距离
        distance = rsu.real_distance(vh.position)

        # 计算车辆天线高度修正因子 ch
        ch = (
            0.8
            + (1.1 * np.log10(rsu.frequency) - 0.7) * rsu.height
            - 1.56 * np.log10(rsu.frequency)
        )
        # 计算路径损耗公式中的第一项 tmp_1
        tmp_1 = (
            69.55 - ch + 26.16 * np.log10(rsu.frequency) - 13.82 * np.log10(rsu.height)
        )
        # 计算路径损耗公式中的第二项 tmp_2
        tmp_2 = 44.9 - 6.55 * np.log10(rsu.height)

        # 为避免距离为 0 时出现对数错误，添加一个小的 epsilon 值
        return tmp_1 + tmp_2 * np.log10(distance + self.EPSILON)

    # 距离单位为千米
    # Okumura-Hata 路径损耗公式
    # Lb​=69.55+26.16log10​(f)−13.82log10​(hb​)−a(hm​)+(44.9−6.55log10​(hb​))log10​(d)
    # 可能存在一些未修复的 bug
    def okumura_hata_path_loss(self, rsu: Rsu, vh: CustomVehicle, city_type="small"):
        # 计算 RSU 与车辆之间的实际距离
        distance = rsu.real_distance(vh.position)

        # 根据城市类型计算车辆天线高度修正因子 a_hm
        if city_type == "large":
            if rsu.frequency <= 200:
                a_hm = 8.29 * (np.log10(1.54 * vh.height)) ** 2 - 1.1
            else:
                a_hm = 3.2 * (np.log10(11.75 * vh.height)) ** 2 - 4.97
        else:
            a_hm = (1.1 * np.log10(rsu.frequency) - 0.7) * vh.height - (
                1.56 * np.log10(rsu.frequency) - 0.8
            )

        # 计算路径损耗
        path_loss = (
            69.55
            + 26.16 * np.log10(rsu.frequency)
            - 13.82 * np.log10(rsu.height)
            - a_hm
            + (44.9 - 6.55 * np.log10(rsu.height)) * np.log10(distance + self.EPSILON)
        )

        return path_loss

    @staticmethod
    def max_distance_oku(rsu: Rsu, city_type="small"):
        # 计算接收功率（瓦特）
        received_power_w = rsu.snr_threshold * rsu.noise_power
        # 将接收功率转换为 dBm
        received_power_dbm = 10 * np.log10(received_power_w) + 30

        # 计算路径损耗
        path_loss = rsu.get_tx_power() - received_power_dbm

        # 根据城市类型计算车辆天线高度修正因子 a_hm
        if city_type == "large":
            if rsu.frequency <= 200:
                a_hm = (
                    8.29 * (np.log10(1.54 * env_config.VEHICLE_ANTENNA_HEIGHT)) ** 2
                    - 1.1
                )
            else:
                a_hm = (
                    3.2 * (np.log10(11.75 * env_config.VEHICLE_ANTENNA_HEIGHT)) ** 2
                    - 4.97
                )
        else:
            a_hm = (
                1.1 * np.log10(rsu.frequency) - 0.7
            ) * env_config.VEHICLE_ANTENNA_HEIGHT - (
                1.56 * np.log10(rsu.frequency) - 0.8
            )

        # 计算公式中的第一项 term1
        term1 = (
            path_loss
            - 69.55
            - 26.16 * np.log10(rsu.frequency)
            + 13.82 * np.log10(rsu.height)
            + a_hm
        )
        # 计算公式中的第二项 term2
        term2 = 44.9 - 6.55 * np.log10(rsu.height)
        # 计算最大距离
        distance = 10 ** (term1 / term2)

        return distance

# 已弃用
# 距离单位为米
# 常见的对数距离路径损耗模型：PL(d) = PL(d_0) + 10nlog_10(d/d_0)
def path_loss(distance):
    # 获取参考距离
    d_0 = env_config.RSU_REFERENCE_DISTANCE

    # 如果距离小于等于参考距离，返回参考距离的路径损耗
    if distance <= d_0:
        return env_config.RSU_PATH_LOSS_REFERENCE_DISTANCE

    # 递归计算路径损耗
    return path_loss(d_0) + 10 * env_config.RSU_PATH_LOSS_EXPONENT * np.log10(
        distance / d_0
    )

# Winner + B1 模型类，用于计算路径损耗
class WinnerB1:
    def __init__(
        self,
        building_height=20,
        street_width=env_config.ROAD_WIDTH,
        building_separation=50,
        street_orientation=30,
    ):
        # 建筑物高度
        self.building_height = building_height
        # 街道宽度
        self.street_width = street_width
        # 建筑物间距
        self.building_separation = building_separation
        # 街道朝向
        self.street_orientation = street_orientation
        pass

    # 已弃用
    def path_loss_deprecated(self, rsu: Rsu, vh: CustomVehicle):
        # 计算 RSU 与车辆之间的实际距离
        distance = rsu.real_distance(vh.position)
        # 获取 RSU 的频率
        frequency = rsu.frequency
        # 获取 RSU 的高度
        base_station_height = rsu.height
        # 获取车辆的高度
        mobile_station_height = vh.height
        # 计算自由空间路径损耗 L0
        L0 = 32.4 + 20 * np.log10(distance) + 20 * np.log10(frequency)

        # 计算建筑物与车辆高度差
        delta_hm = self.building_height - mobile_station_height
        # 计算街道传播损耗 Lrts
        Lrts = (
            -16.9
            - 10 * np.log10(self.street_width)
            + 10 * np.log10(frequency)
            + 20 * np.log10(delta_hm)
            + self.street_orientation_loss(self.street_orientation)
        )

        # 计算基站与建筑物高度差
        delta_hb = base_station_height - self.building_height
        if delta_hb > 0:
            # 计算基站高度损耗 Lbsh
            Lbsh = -18 * np.log10(1 + delta_hb)
            ka = 54
            kd = 18
        else:
            Lbsh = 0
            kd = 18 - 15 * delta_hb / self.building_height
            if distance >= 0.5:
                ka = 54 - 0.8 * delta_hb
            else:
                ka = 54 - 1.6 * delta_hb * distance

        # 计算频率相关损耗 kf
        kf = -4 + 0.7 * (frequency / 925 - 1)
        # 计算多径损耗 Lmsd
        Lmsd = (
            Lbsh
            + ka
            + kd * np.log10(distance)
            + kf * np.log10(frequency)
            - 9 * np.log10(self.building_separation)
        )

        # 计算总路径损耗 L
        L = L0 + Lrts + Lmsd
        return L

    # 已弃用
    def street_orientation_loss(self, alpha):
        # 根据街道朝向计算损耗
        if 0 <= alpha < 35:
            return -10 + 0.354 * alpha
        elif 35 <= alpha < 55:
            return 2.5 + 0.075 * (alpha - 35)
        else:
            return 4 - 0.114 * (alpha - 55)

    # 计算断点距离 d'_BP，fc 是中心频率，单位为 Hz
    def breakpoint_distance(self, h_BS, h_MS, fc_hz):
        # 计算基站有效高度
        h_BS_eff = h_BS - 1.0
        # 计算移动台有效高度
        h_MS_eff = h_MS - 1.0
        # 计算断点距离
        d_BP = (4 * h_BS_eff * h_MS_eff * fc_hz) / env_config.C
        return d_BP

    def path_loss_los(self, d1, fc_ghz, h_BS, h_MS):
        # 确保距离不小于 11 米
        if d1 <= 10:
            d1 = 11

        # 计算基站有效高度
        h_BS_eff = h_BS - 1.0
        # 计算移动台有效高度
        h_MS_eff = h_MS - 1.0

        # 计算断点距离 d'_BP
        d_BP = self.breakpoint_distance(h_BS, h_MS, fc_ghz * 1e9)

        # 根据距离与断点距离的关系计算路径损耗
        if d1 < d_BP:
            path_loss = 22.7 * np.log10(d1) + 41.0 + 20 * np.log10(fc_ghz / 5.0)
        else:
            path_loss = (
                40 * np.log10(d1)
                + 9.45
                - 17.3 * np.log10(h_BS_eff)
                - 17.3 * np.log10(h_MS_eff)
                + 2.7 * np.log10(fc_ghz / 5.0)
            )

        return path_loss

    # 通常情况下
    # 返回路径损耗，单位为 dB
    def path_loss_nlos(self, d1, d2, fc_ghz, h_BS, h_MS):
        # 确保距离不小于 11 米
        if d1 <= 10:
            d1 = 11

        # 获取街道宽度
        w = self.street_width
        # 如果 d2 小于等于街道宽度的一半，按视距路径损耗计算
        if d2 <= w / 2:
            return self.path_loss_los(d1, fc_ghz, h_BS, h_MS)

        def pl(d1, d2):
            # 计算路径损耗指数 nj
            nj = max(2.8 - 0.0024 * d2, 1.84)
            # 计算非视距路径损耗
            pl = (
                self.path_loss_los(d1, fc_ghz, h_BS, h_MS)
                + 20
                - 12.5 * nj
                + 10 * nj * np.log10(d2)
                + 3 * np.log10(fc_ghz / 5.0)
            )
            return pl

        # 根据 d2 的范围计算路径损耗
        if w / 2 < d2 < 2000:
            return min(pl(d1, d2), pl(d2, d1))
        else:
            # 如果 d2 超出有效范围，抛出异常
            raise ValueError("d2 is out of the valid range.")

    def test(self):
        # 基站高度，单位为米
        h_BS = 10
        # 移动台高度，单位为米
        h_MS = 1.5
        # 频率，单位为 GHz（5905 MHz）
        fc_ghz = 5.905

        # 视距路径损耗计算的示例距离
        d1_values = [10, 100, 500]
        # 非视距路径损耗计算的示例距离
        d2_values = [
            50,
            100,
            200,
            300,
            400,
            500,
        ]
        # 计算并打印视距路径损耗
        print("LOS Path Loss:")
        for d1 in d1_values:
            pl_los = self.path_loss_los(d1, fc_ghz, h_BS, h_MS)
            print(f"d1: {d1} m, Path Loss: {pl_los:.2f} dB")

        # 计算并打印非视距路径损耗
        print("\nNLOS Path Loss:")
        for d2 in d2_values:
            pl_nlos = self.path_loss_nlos(d1_values[0], d2, fc_ghz, h_BS, h_MS)
            print(f"d1: {d1_values[0]} m, d2: {d2} m, Path Loss: {pl_nlos:.2f} dB")

# 反向使用 Okumura-Hata 路径损耗公式计算最大连接距离
# 通用的最大距离算法
# 不适用于 Winner + B1 模型
def max_distance(rsu: Rsu):
    # 初始化最大距离为 0
    max_distance = 0
    # 步长，单位为千米
    step = 0.001
    # 将实际距离转换为程序中的距离单位
    distance = utils.real_distance_to_distance(step)

    while True:
        # 创建一个自定义车辆对象，位置在 RSU 位置的基础上加上距离
        vh = CustomVehicle(0, Point((rsu.position.x + distance, rsu.position.y)))

        # 如果信噪比小于 RSU 的信噪比阈值，跳出循环
        if sinr(rsu, vh) < rsu.snr_threshold:
            break

        # 更新最大距离
        max_distance = distance
        # 增加距离
        distance += utils.real_distance_to_distance(step)

    return max_distance

# 计算满足特定数据速率要求的最大距离
def max_distance_mbps(rsu: Rsu, rate_tr=env_config.DATA_RATE_TR):
    # 初始化最大距离为 0
    max_distance = 0
    # 步长，单位为米
    step = 1
    # 初始化距离为步长
    distance = step

    while True:
        # 创建一个自定义车辆对象，位置在 RSU 位置的基础上加上距离
        vh = CustomVehicle(0, Point((rsu.position.x + distance, rsu.position.y)))

        # 计算信道容量
        rate = channel_capacity(rsu, vh)
        # 如果信道容量小于等于数据速率阈值，跳出循环
        if rate <= rate_tr:
            break

        # 更新最大距离
        max_distance = distance
        # 增加距离
        distance += step

    return max_distance

def max_rate(rsu: Rsu):
    # 距离为 1 米
    distance = 1
    # 创建一个自定义车辆对象，位置在 RSU 位置的基础上加上距离
    vh = CustomVehicle(0, Point((rsu.position.x + distance, rsu.position.y)))

    # 计算最大速率
    max_rate = channel_capacity(rsu, vh)

    return max_rate

# 将 dB 转换为 dBm
def db_to_dbm(P_db):
    return P_db + 30

# 将 dBm 转换为瓦特
def dbm_to_watt(P_dbm):
    return 10 ** ((P_dbm - 30) / 10)

# 将比特每秒转换为兆比特每秒
def bpsToMbps(bps):
    return bps / 1e6

# 创建 RSU 网络的邻接表
def network(coords, tree, k=3):
    # 创建邻接表
    network = defaultdict(list)
    for i, coord in enumerate(coords):
        # 查询最近邻（不包括自身）
        distances, indices = tree.query(
            coord, k=k
        )  # k=3 包括自身和它的两个最近邻
        for index in indices[1:]:  # 跳过第一个索引，因为它是自身
            network[i].append(index)

    return network

# 使用广度优先搜索（BFS）算法计算从起始 RSU 到目标 RSU 的跳数
def find_hops(start_rsu, target_rsu, rsu_network):
    # 初始化队列，存储 (当前 RSU, 跳数)
    queue = deque([(start_rsu, 0)])
    # 初始化已访问集合
    visited = set()

    while queue:
        # 从队列中取出当前 RSU 和跳数
        current_rsu, hops = queue.popleft()

        # 如果当前 RSU 是目标 RSU，返回跳数
        if current_rsu == target_rsu:
            return hops

        # 如果当前 RSU 未被访问过
        if current_rsu not in visited:
            # 标记当前 RSU 为已访问
            visited.add(current_rsu)
            # 遍历当前 RSU 的邻居
            for neighbor in rsu_network[current_rsu]:
                # 如果邻居未被访问过
                if neighbor not in visited:
                    # 将邻居加入队列，并增加跳数
                    queue.append((neighbor, hops + 1))

    # 如果无法到达目标 RSU，返回 -1
    return -1


def sinr(rsu: Rsu, vh: Vehicle, distance=None):
    """
    计算线性的信道功率增益 g_{i,j}
    g = 1 / 10^(PL / 10) = 10^(-PL / 10)
    """

    if distance is not None:
        # 确保 d1 不超过距离值
        d1 = 10
        d1 = min(d1, distance)
        # 计算 d2
        d2 = np.sqrt(distance ** 2 - d1 ** 2)
        # 确保 d2 不超过距离值
        d2 = min(d2, distance)
    else:
        # 获取 d1 和 d2 的值
        d1, d2 = rsu.get_d1_d2(vh.get_position(), vh.get_angle())
    path_loss = WinnerB1().path_loss_nlos(
        d1, d2, rsu.frequency * 1e-3, rsu.height, vh.height
    )

    # 路径损耗 PL = P_tx / P_rx (线性倍数)
    # 所以增益 g = P_rx / P_tx = 1 / PL
    g_ij = 10 ** (-path_loss / 10)

    # 获取发射功率
    p_ij = dbm_to_watt(env_config.VEHICLE_TRANSMITTED_POWER)

    # 计算分子: 信号功率 S = p * g
    signal_power = p_ij * g_ij

    interference_watt = 0

    for other_veh in rsu.connections:
        # 跳过空槽位
        if other_veh is None:
            continue

        # 排除自己
        if other_veh.vehicle_id == vh.vehicle_id:
            continue

        # 计算其他车辆到 RSU 的距离
        d1, d2 = rsu.get_d1_d2(other_veh.position, other_veh.angle)

        # 计算路径损耗
        pl = WinnerB1().path_loss_nlos(
            d1, d2,
            rsu.frequency * 1e-3,
            rsu.height,
            other_veh.height
        )

        rx_power_dbm = env_config.VEHICLE_TRANSMITTED_POWER - pl
        rx_power_watt = dbm_to_watt(rx_power_dbm)
        interference_watt += rx_power_watt

    # 获取分母: 噪声 + 干扰
    # rsu.noise_power 已经是瓦特单位
    noise_plus_interference = rsu.noise_power + interference_watt

    # 计算 SINR
    sinr = signal_power / noise_plus_interference

    return sinr


def channel_capacity(
    rsu: Rsu, vh: Vehicle, distance=None, bw=env_config.RSU_MAX_TRANSMITTED_BANDWIDTH
):
    if distance is not None:
        # 如果提供了距离，计算信道容量并转换为 Mbps
        return bpsToMbps(bw * np.log2(1 + sinr(rsu, vh, distance)))
    # 否则，直接计算信道容量并转换为 Mbps
    return bpsToMbps(bw * np.log2(1 + sinr(rsu, vh)))

def V2R_delay(rsu: Rsu, vh: Vehicle):
    job_size = vh.job.job_size
    data_rate = channel_capacity(rsu, vh)
    T_com = job_size / data_rate
    return T_com

def R2R_delay(vh: Vehicle, hops):
    job_size = vh.job.job_size
    trans_delay = job_size / env_config.R2R_BANDWIDTH
    prop_delay = hops * env_config.HOP_LATENCY
    return trans_delay + prop_delay

def R2C_delay(rsu: Rsu, vh: Vehicle):
    job_size = vh.job.job_size
    trans_delay = job_size / env_config.RSU_TO_CLOUD_BANDWIDTH
    prop_delay = env_config.CLOUD_TRANS_TIME
    return trans_delay + prop_delay

def V2C_delay(rsu: Rsu, vh: Vehicle):
    v2r_delay = V2R_delay(rsu, vh)
    r2c_delay = R2C_delay(rsu, vh)
    return v2r_delay + r2c_delay


def calculate_computation_time(vehicle, local_rsu, rsus, rsu_network, action):
    """
    计算计算延迟 T^{comp}

    参数:
    vehicle: 车辆对象
    local_rsu: 车辆连接的本地 RSU 对象
    rsus: 所有 RSU 的列表 (用于查找邻居)
    rsu_network: RSU 邻接网络图
    action: 卸载决策 (1: RSU, 0: Cloud)

    返回:
    float: 计算时间 T^{comp}
    """

    # c_v: 任务计算量
    c_v = env_config.JOB_CP_REQUIRE
    # content_id: 内容标识
    content_id = vehicle.job_type

    if action == 0:
        return env_config.CLOUD_COMPUTATIONALLY_TIME


    elif action == 1:
        f_r = env_config.RSU_COMPUTATION_POWER

        # 本地 RSU 命中
        if local_rsu.has_content(content_id):
            return c_v / f_r

        # 本地未命中，检查邻居 RSU
        else:
            # 获取邻居 ID 列表
            neighbor_ids = rsu_network[local_rsu.id]

            # 查找是否有邻居缓存了该内容
            min_r2r_delay = float('inf')
            found_in_neighbor = False

            for nid in neighbor_ids:
                neighbor = rsus[nid]
                if neighbor.has_content(content_id):
                    found_in_neighbor = True

                    # 计算 R2R 延迟 (数据传输延迟)
                    # 假设 s_k (数据大小) 用于计算从邻居拉取数据的延迟
                    s_k = vehicle.job.job_size

                    # 跳数 (邻居通常为 1 跳，或者是根据 find_hops 计算)
                    hops = 1  # 或者 use network.find_hops(local_rsu.id, nid, rsu_network)

                    # 使用之前的 calc_r2r_delay 函数 (需确保已定义)
                    # T_{com}^{R2R} = s_k / R_{R2R} + T_{prop}^{R2R}
                    # 这里假设我们手动计算或调用之前的函数:
                    r2r_bw = env_config.RSU_MAX_TRANSMITTED_BANDWIDTH  # 或者专用的 R2R 带宽
                    t_trans = s_k / (r2r_bw / 1e6)  # Mbps
                    t_prop = hops * (env_config.HOP_LATENCY / 1000.0)  # ms to s

                    r2r_delay = t_trans + t_prop

                    if r2r_delay < min_r2r_delay:
                        min_r2r_delay = r2r_delay

            if found_in_neighbor:
                # 邻居命中
                return (c_v / f_r) + min_r2r_delay
            else:
                # 子情况 C: 邻居也没命中 (Miss)
                # 需要从云端下载到 RSU 再计算 (增加了 R2C 延迟)
                # T = c_v / f_r + T_{R2C_download}
                return (c_v / f_r) + env_config.CLOUD_TRANS_TIME
    return None


def calculate_fetch_delay(target_rsu: Rsu, content_id, rsus: list, rsu_network: dict, cache_module):
    """
    计算缓存更新成本 T_fetch = min(T_R2C, min(T_R2R))

    参数:
    ...
    cache_module: Caching 类的实例，用于获取内容大小
    """

    # 动态获取内容大小 s_k (单位假设为 Mbit)
    # 确保内容 ID 是合法的
    if content_id is None:
        return 0.0

    content_size_mbit = cache_module.get_size(content_id)

    # 计算从云端获取的延迟 T_R2C
    # 公式: T_trans = Size / Bandwidth + T_prop
    # Bandwidth 单位是 Mbps, Size 单位是 Mbit -> 除法结果是秒
    # RSU_TO_CLOUD_BANDWIDTH = 500 (Mbps)
    # CLOUD_TRANS_TIME = 10 (s) (包含传播延迟)
    t_r2c = content_size_mbit / env_config.RSU_TO_CLOUD_BANDWIDTH + env_config.CLOUD_TRANS_TIME

    # 计算从邻居获取的延迟 T_R2R
    min_t_r2r = float('inf')

    # 获取邻居 ID 列表
    if target_rsu.id in rsu_network:
        neighbor_ids = rsu_network[target_rsu.id]

        for nid in neighbor_ids:
            neighbor_rsu = rsus[nid]

            # 检查邻居是否缓存了该内容
            if content_id in neighbor_rsu.caching_contents:
                # 计算 R2R 传输延迟
                # R2R_BANDWIDTH = 1000 (Mbps)
                t_trans = content_size_mbit / env_config.R2R_BANDWIDTH

                # 计算传播延迟 (假设跳数为 1)
                # HOP_LATENCY = 3 (ms) -> 需要除以 1000 转为秒
                hops = 1
                t_prop = hops * (env_config.HOP_LATENCY / 1000.0)

                t_r2r = t_trans + t_prop

                if t_r2r < min_t_r2r:
                    min_t_r2r = t_r2r

    # 4. 取云端和邻居中的最小值
    t_fetch = min(t_r2c, min_t_r2r)

    return t_fetch