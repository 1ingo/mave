import math
import sys

sys.path.append("./")

# from vanet_env.entites import Rsu, Vehicle
# class Config:
#     def __init__(self):
#         self.path_loss = "OkumuraHata"
#         self.rsu = Rsu()
# 随机种子，用于保证实验的可重复性
SEED = 114514
# 地图名称，用于指定使用的地图，这里设置为 "london"
MAP_NAME = "london"
# Canvas Size
# MAP_SIZE = (400, 400)

"""
Rsu Config
"""
# 道路宽度，单位为米
ROAD_WIDTH = 10

# RSU（路边单元）的位置列表
RSU_POSITIONS = []

# 根据地图名称生成 RSU 的位置
# 如果地图名称不是 "london"，按照以下规则生成 RSU 位置
if MAP_NAME != "london":
    # 在 x 方向上，从 40 开始，每隔 90 取一个值
    for x in range(40, 400, 90):
        # 在 y 方向上，从 0 开始，每隔 77 取一个值，共取 5 个值
        for y in range(0, 77 * 5, 77):
            # 将生成的 (x, y) 坐标添加到 RSU 位置列表中
            RSU_POSITIONS.append((x, y))
# 如果地图名称是 "london"，按照相同的规则生成 RSU 位置
else:
    for x in range(40, 400, 90):
        for y in range(0, 77 * 5, 77):
            RSU_POSITIONS.append((x, y))

# roadside_up
# for x in range(0, 400, 100):
#     for y in range(17, (77 + 17) * 5, 77 + 17):
#         RSU_POSITIONS.append((x, y))

# RSU 的数量，通过计算 RSU 位置列表的长度得到
NUM_RSU = len(RSU_POSITIONS)

# RSU 的计算能力，单位为 TFLOPs/s（每秒万亿次浮点运算）
# 这里将 RTX 4060 Ti 的 22.1 tflops 和 RTX 4070 的 29.1 tflops 相加，不过实际使用了 RTX 4060 Ti 22.1 tflops、RTX 4070 29.1 tflops、RTX 4080 48.7 tflops、RTX 4090 82.6 tflops 中的 82.6 + 48.7
RSU_COMPUTATION_POWER = 82.6 + 48.7
# RSU 的缓存容量
RSU_CACHING_CAPACITY = 10
# RSU 的发射功率，单位为 dBm（分贝毫瓦），这里对应 1 瓦特
RSU_TRANSMITTED_POWER = 30
# RSU 的噪声功率，单位为瓦特
RSU_NOISE_POWER = 1e-9
# RSU 的最大发射带宽，单位为赫兹
RSU_MAX_TRANSMITTED_BANDWIDTH = 20e6
# RSU 的工作频率，单位为兆赫兹
RSU_FREQUENCY = 5905
# RSU 的天线高度，单位为米
RSU_ANTENNA_HEIGHT = 10
# 路径损耗指数，用于计算信号传播过程中的损耗
RSU_PATH_LOSS_EXPONENT = 3
# 参考距离，单位为米
RSU_REFERENCE_DISTANCE = 1e-3
# 参考距离处的路径损耗，单位为分贝
RSU_PATH_LOSS_REFERENCE_DISTANCE = 40
# 信噪比阈值，用于判断信号质量
RSU_SNR_THRESHOLD = 2e-8
# RSU 的天线数量，用于多输入多输出（MIMO）技术
RSU_NUM_ANTENNA = 2
# 天线增益，单位为 dBi（相对于各向同性辐射器的增益）
ANTENNA_GAIN = 3
# 数据速率阈值，单位为未知（推测可能是某种数据传输速率的限制）
DATA_RATE_TR = 40
# RSU 的最大计算核心数
NUM_CORES = 5
# RSU 的最大下行连接数，最好等于计算核心数且能被帧率整除
MAX_CONNECTIONS = 5
# 能效比，用于衡量能源利用效率，计算公式为 MAX_EE - EE = cp_usage
MAX_EE = 0.2

"""
Vehicle Config
"""
# 车辆的数量
NUM_VEHICLES = 150
# 车辆的天线高度，单位为米
VEHICLE_ANTENNA_HEIGHT = 1.5
# 车辆的最大作业大小
MAX_JOB_SIZE = 256
# 内容的数量
NUM_CONTENT = 50

VEHICLE_TRANSMITTED_POWER = 23

VEHICLE_COMPUTATION_POWER = 50

"""
Cloud Config
"""
# 云端的计算时间
CLOUD_COMPUTATIONALLY_TIME = 0.5
# 云端的传输时间
CLOUD_TRANS_TIME = 10

"""
Others
"""
# 渲染配置，坐标单位，单位为米
COORDINATE_UNIT = 1
# 自由空间中的光速，单位为米每秒
C = 3e8

# RSU 回程网络带宽 (R2R), 单位: Mbps
# 使用光纤或高带宽微波链路，例如 1 Gbps
R2R_BANDWIDTH = 1000

# RSU 到云端上行带宽 (R2C), 单位: Mbps
# 假设光纤接入，例如 500 Mbps
RSU_TO_CLOUD_BANDWIDTH = 500

"""
utility
"""
# 非缓存延迟因子，用于计算非缓存情况下的延迟
NON_CACHING_FACTOR = 1.25
# 跳数延迟，单位为毫秒
HOP_LATENCY = 3e-3
# 每帧的跳数优化因子
HOP_OPT_FACTOR = 90
# VR 设备（Meta Quest 2 with Complex Avatar）的相关参数
# 分辨率宽度
W = 2160
# 分辨率高度
H = 2160
# 每个像素所需的计算能力
CP_REQUIRE_PIX = 9.5e4
# 所需的帧率
FPS_REQUIRE = 90
# 作业所需的计算能力，单位为 TFLOPs，通过计算得到并向上取整
JOB_CP_REQUIRE = math.ceil(W * H * CP_REQUIRE_PIX * FPS_REQUIRE * 1e-12)
# 作业所需的数据速率，单位为 Mbps
JOB_DR_REQUIRE = 40
# 作业所需的帧率
JOB_FPS_REQUIRE = 90
# 延迟因子，用于计算延迟相关的指标
LATENCY_FACTOR = 0.5
# 最大的用户体验质量（QOE）
MAX_QOE = 1.0
# 计算因子，用于计算计算相关的指标
COMPUTATIONAL_FACTOR = 0.5

# 打印提示信息，表示配置已加载
print("config loaded")