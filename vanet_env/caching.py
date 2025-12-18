import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from vanet_env import data_preprocess
import sys

sys.path.append("./")


class Caching:
    def __init__(self, caching_fps, num_content, num_caching, seed):
        """
        初始化 Caching 类的实例。

        参数:
        caching_fps (int): 缓存的帧率，用于将时间划分为不同的间隔
        num_content (int): 内容的总数
        num_caching (int): 要缓存的内容数量
        seed (int): 随机数种子，确保结果可复现
        """
        # 设置随机数种子
        np.random.seed(seed)
        # 缓存帧率
        self.fps = caching_fps
        # 内容总数
        self.num_content = num_content
        # 要缓存的内容数量
        self.num_caching = num_caching

        self.content_list = []
        self.content_sizes = []

    def get_content(self, time):
        """
        根据时间和内容的流行度，从 content_list 中随机选择一个内容。

        参数:
        time (int): 时间，范围在 (0, fps) 内

        返回:
        int: 选中内容在 content_list 中的索引
        """
        # 获取对应时间间隔的聚合数据框
        ontime_df = self.aggregated_df_list[time]
        # 筛选出聚合数据框中 id 存在于 content_list 中的记录
        filtered_df = ontime_df[ontime_df["id"].isin(self.content_list)]
        # 获取筛选后数据框中每条记录的流行度得分列表
        popularity_scores_list = filtered_df["popularity_score"].tolist()
        # 将流行度得分列表转换为 numpy 数组，作为初始概率
        probabilities = np.array(popularity_scores_list)

        # 将前 num_caching 个元素的概率调整为总和 80%
        # 提取前 num_caching 个概率
        top_probabilities = probabilities[: self.num_caching]
        top_probabilities = np.array(top_probabilities)
        # 对前 num_caching 个概率进行归一化处理，使它们的和为 1
        top_probabilities /= top_probabilities.sum()
        # 将归一化后的概率乘以 0.8，使其总和为 80%
        top_probabilities *= 0.80

        # 将后 num_content - num_caching 个元素的概率调整为总和 20%
        # 提取剩余的概率
        remaining_probabilities = probabilities[self.num_caching :]
        remaining_probabilities = np.array(remaining_probabilities)
        # 对剩余的概率进行归一化处理，使它们的和为 1
        remaining_probabilities /= remaining_probabilities.sum()
        # 将归一化后的概率乘以 0.2，使其总和为 20%
        remaining_probabilities *= 0.20

        # 合并调整后的概率
        adjusted_probabilities = np.concatenate(
            [top_probabilities, remaining_probabilities]
        )

        # 根据调整后的概率从 content_list 中随机选择一个 id
        selected_id = np.random.choice(self.content_list, p=adjusted_probabilities)

        # 获取选中 id 在 content_list 中的索引
        idx = self.content_list.index(selected_id)
        return idx

    def get_content_list(self):
        """
        生成内容列表以及相关的聚合数据框。

        返回:
        list: 不重复的内容 id 列表
        list: 每个时间间隔的聚合数据框列表
        DataFrame: 所有聚合数据框合并后的结果
        """
        # 调用 data_preprocess 模块中的 summnet_preprocess 函数进行数据预处理
        df = data_preprocess.summnet_preprocess()
        # 创建一个 MinMaxScaler 对象，用于数据归一化
        scaler = MinMaxScaler()

        # 对数据进行归一化处理
        # 对 players 列进行归一化处理，并将结果存储在新列 players_norm 中
        df["players_norm"] = scaler.fit_transform(df[["players"]])
        # 对 stars 列进行归一化处理，并将结果存储在新列 stars_norm 中
        df["stars_norm"] = scaler.fit_transform(df[["stars"]])
        # 对 attempts 列进行归一化处理，并将结果存储在新列 attempts_norm 中
        df["attempts_norm"] = scaler.fit_transform(df[["attempts"]])

        # 按照 seconds_normalized 列进行升序排序
        df = df.sort_values(by="seconds_normalized", ascending=True)

        # 生成时间间隔列表，每个间隔为 1/fps
        time_intervals = [(i / self.fps, (i + 1) / self.fps) for i in range(self.fps)]
        # 用于存储每个时间间隔的聚合数据框
        aggregated_df_list = []

        for start, end in time_intervals:
            # 筛选出处于当前时间间隔内的记录
            interval_records = df[
                (df["seconds_normalized"] >= start) & (df["seconds_normalized"] < end)
            ]

            # 对当前时间间隔内的记录按 id 进行分组，并计算每组的 stars_norm 最大值和记录数量
            aggregated_records = (
                interval_records.groupby("id")
                .agg({"stars_norm": "max", "id": "count"})
                .rename(columns={"id": "count"})
                .reset_index()
            )
            # 为聚合后的记录添加时间间隔信息
            aggregated_records["time_interval"] = f"{start}-{end}"

            # 按照记录数量进行降序排序，并重置索引
            aggregated_records = aggregated_records.sort_values(
                by="count", ascending=False
            ).reset_index(drop=True)

            # 对记录数量列进行归一化处理，并将结果存储在新列 count_norm 中
            aggregated_records["count_norm"] = scaler.fit_transform(
                aggregated_records[["count"]]
            )
            # 计算每条记录的流行度得分
            aggregated_records["popularity_score"] = (
                aggregated_records["count_norm"] * 0.9
                + aggregated_records["stars_norm"] * 0.1
            )

            # 将当前时间间隔的聚合数据框添加到列表中
            aggregated_df_list.append(aggregated_records)

        # 将聚合数据框列表赋值给类的属性
        self.aggregated_df_list = aggregated_df_list
        # 将所有聚合数据框合并为一个，并重置索引
        self.aggregated_df = pd.concat(aggregated_df_list).reset_index(drop=True)

        # 计算每个时间间隔需要选择的 id 数量
        n = self.num_content * self.fps
        # 用于存储每个时间间隔的前 n 个流行度得分最高的 id
        top_ids_per_interval = []

        for start, end in time_intervals:
            # 筛选出当前时间间隔的记录
            interval_records = self.aggregated_df[
                self.aggregated_df["time_interval"] == f"{start}-{end}"
            ]
            # 获取当前时间间隔内流行度得分最高的前 n 个 id
            top_n_ids = interval_records.nlargest(n, "popularity_score")["id"].tolist()
            # 将这些 id 添加到列表中
            top_ids_per_interval.extend(top_n_ids)

        # 用于存储不重复的 id
        unique_top_ids_ordered = []
        # 用于记录已经出现过的 id
        seen_ids = set()

        for id in top_ids_per_interval:
            # 如果 id 未出现过，则添加到不重复 id 列表中，并记录该 id 已出现
            if id not in seen_ids:
                unique_top_ids_ordered.append(id)
                seen_ids.add(id)
            # 如果不重复 id 列表的长度达到 num_content，则停止循环
            if len(unique_top_ids_ordered) == self.num_content:
                break

        self.content_list = unique_top_ids_ordered

        min_size = 512
        max_size = 2048

        # 生成与 content_list 长度一致的大小数组
        # content_sizes[k] 对应 content_list[k] 的大小
        self.content_sizes = np.random.randint(
            low=min_size,
            high=max_size,
            size=len(self.content_list)
        )

        # 打印一下以便调试
        # print(f"Generated sizes for {len(self.content_list)} contents.")

        # 返回不重复的 id 列表、聚合数据框列表和合并后的聚合数据框
        return self.content_list, self.aggregated_df_list, self.aggregated_df


    def get_size(self, content_idx):
        """
        根据内容索引 k 获取其大小 s_k
        """
        if 0 <= content_idx < len(self.content_sizes):
            return self.content_sizes[content_idx]
        else:
            return 512