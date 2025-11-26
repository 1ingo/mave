import sys

sys.path.append("./")
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from vanet_env import env_config


def summnet_preprocess(num_content=env_config.NUM_CONTENT):
    """
    对 SMMnet 数据集进行预处理。

    参数:
    num_content (int): 内容的数量，默认为 env_config 中的 NUM_CONTENT

    返回:
    DataFrame: 经过预处理后的 DataFrame
    """
    # 构建数据文件的路径
    # 获取当前文件所在目录，然后拼接数据文件的相对路径
    path = os.path.join(os.path.dirname(__file__), "data", "SMMnet", "course-meta.csv")

    # 使用 pandas 读取 CSV 文件，指定分隔符为制表符
    df = pd.read_csv(path, sep="\\t")

    # 打印 DataFrame 的基本信息，包括列名、数据类型、非空值数量等
    print(df.info())

    # 将 'catch' 列转换为 pandas 的日期时间类型
    # format="mixed" 表示自动识别日期时间的格式
    df["catch"] = pd.to_datetime(df["catch"], format="mixed")

    # 从 'catch' 列中提取月份信息，并添加到新列 'month' 中
    df["month"] = df["catch"].dt.month

    # 从 'catch' 列中提取日期信息，并添加到新列 'day' 中
    df["day"] = df["catch"].dt.day

    # 从 'catch' 列中提取时间信息（小时:分钟:秒.微秒），并添加到新列 'time_only' 中
    df["time_only"] = df["catch"].dt.strftime("%H:%M:%S.%f")

    # 计算 'catch' 列中每个时间点对应的秒数，并添加到新列 'seconds' 中
    # 小时转换为秒：小时数 * 3600
    # 分钟转换为秒：分钟数 * 60
    # 秒数：直接使用秒数
    # 微秒转换为秒：微秒数 / 1e6
    df["seconds"] = (
        df["catch"].dt.hour * 3600
        + df["catch"].dt.minute * 60
        + df["catch"].dt.second
        + df["catch"].dt.microsecond / 1e6
    )

    # 以下代码注释掉了，作用是找到每个 'id' 的最后一次出现记录
    # df = df.groupby("id").last().reset_index()

    # 创建一个 MinMaxScaler 对象，用于数据归一化
    scaler = MinMaxScaler()

    # 对 'seconds' 列进行归一化处理，并将结果存储在新列 'seconds_normalized' 中
    df["seconds_normalized"] = scaler.fit_transform(df[["seconds"]])

    # 按照 'players' 列进行降序排序，并将排序后的结果存储在 sorted_players_df 中
    sorted_players_df = df.sort_values(by="players", ascending=False)

    # 按照 'attempts' 列进行降序排序，并将排序后的结果存储在 sorted_attempts_df 中
    sorted_attempts_df = df.sort_values(by="attempts", ascending=False)

    # 按照 'stars' 列进行降序排序，并将排序后的结果存储在 sorted_stars_df 中
    sorted_stars_df = df.sort_values(by="stars", ascending=False)

    # 返回经过预处理后的 DataFrame
    return df
