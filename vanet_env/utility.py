from collections import defaultdict
from itertools import chain
import sys
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append("./")
from vanet_env import env_config
from vanet_env.entites import Rsu, Vehicle, OrderedQueueList
from vanet_env import network, utils


# 计算固定策略下的效用和缓存命中率
def fixed_calculate_utility(
    # 车辆字典，键为车辆ID，值为Vehicle对象
    vehs: Dict[str, Vehicle],
    # RSU列表，每个元素为Rsu对象
    rsus: List[Rsu],
    # RSU网络对象
    rsu_network,
    # 当前时间步
    time_step,
    # 帧率
    fps,
    # 权重参数
    weight,
    # 最大QoE值，默认为环境配置中的MAX_QOE
    max_qoe=env_config.MAX_QOE,
    # 是否使用整数效用，默认为False
    int_utility=False,
    # 最大连接数，默认为环境配置中的MAX_CONNECTIONS
    max_connections=env_config.MAX_CONNECTIONS,
    # 核心数，默认为环境配置中的NUM_CORES
    num_cores=env_config.NUM_CORES,
):
    # 用于存储每个RSU的效用列表
    rsu_utility_dict = defaultdict(list)
    # 用于存储每个RSU的缓存命中率
    caching_hit_ratios = {}
    # 跳数惩罚率
    hop_penalty_rate = 0.1
    # 最大能源效率
    max_ee = 0.2
    # QoE因子，用于平衡QoE和能源效率
    qoe_factor = 1 - max_ee

    # 观察reward用，这里暂时没有实际作用
    if time_step >= 50:
        time_step

    # 由车辆计算，只需计算在范围内车辆的QoE，这里可修改
    for v_id, veh in vehs.items():
        # 相当于idle的车辆不算
        if veh.vehicle_id not in rsus[veh.connected_rsu_id].range_connections:
            continue

        # 类型注解，明确veh为Vehicle对象
        veh: Vehicle

        # 如果车辆连接到云端
        if veh.is_cloud:
            # 云端连接的QoE为最大QoE的0.1倍
            qoe = max_qoe * 0.1
            # 将该车辆的QoE添加到连接的RSU的效用列表中
            rsu_utility_dict[veh.connected_rsu_id].append(qoe)
            # 如果车辆的前一个QoE为空，则将当前QoE赋值给前一个QoE
            if veh.job.pre_qoe is None:
                veh.job.pre_qoe = qoe
            else:
                # 否则将当前QoE赋值给前一个QoE
                veh.job.pre_qoe = veh.job.qoe
            # 更新车辆的当前QoE
            veh.job.qoe = qoe
            continue

        # 计算传输QoE，取车辆数据速率和作业数据速率要求的最小值，再除以作业数据速率要求
        trans_qoe = (
            min(veh.data_rate, env_config.JOB_DR_REQUIRE) / env_config.JOB_DR_REQUIRE
        )
        # 存储每个RSU的传输QoE列表
        trans_qoes = defaultdict(list)
        # 存储每个RSU的处理QoE列表
        proc_qoes = defaultdict(list)
        # 存储每个RSU的缓存命中状态列表
        caching_hit_states = defaultdict(list)

        # 如果车辆的作业处理RSU列表为空
        if veh.job.processing_rsus.is_empty():
            # 相当于在连接却无处理，一般不会
            if veh.connected_rsu_id != None:
                # 惩罚之，将该RSU的效用列表中添加0.0
                rsu_utility_dict[veh.connected_rsu_id].append(0.0)

        # 所有作业分配比例之和
        job_ratio_all = 0.0

        # 一般是邻居RSU
        for p_rsu in veh.job.processing_rsus:
            # 即单个作业分配比例
            # QoE计算在最后一个循环之后进行
            if p_rsu is not None:
                # 类型注解，明确p_rsu为Rsu对象
                p_rsu: Rsu
                # 获取车辆在处理RSU的处理作业列表中的索引
                p_idx = p_rsu.handling_jobs.index((veh, 0))
                # 获取作业分配比例
                job_ratio = p_rsu.handling_jobs[p_idx][1]
                # 作业分配比例累加
                job_ratio_all += job_ratio

                # 如果作业分配比例不为0
                if job_ratio != 0:
                    # 计算处理QoE，取处理RSU的实际CPU分配除以作业分配比例和作业CPU要求的最小值，再除以作业CPU要求
                    process_qoe = (
                        min(
                            p_rsu.real_cp_alloc[p_idx % p_rsu.max_cores] / job_ratio,
                            env_config.JOB_CP_REQUIRE,
                        )
                        / env_config.JOB_CP_REQUIRE
                    )
                else:
                    # 作业分配比例为0时，处理QoE为0.0
                    process_qoe = 0.0

                # 获取车辆连接的RSU
                trans_rsu: Rsu = rsus[veh.connected_rsu_id]

                # 如果处理RSU就是车辆连接的RSU
                if p_rsu.id == veh.connected_rsu_id:
                    # 取处理QoE和传输QoE的最小值作为QoE
                    qoe = min(process_qoe, trans_qoe)
                    # 缓存使用标志
                    use_caching = False
                    # 缓存调试
                    if veh.job.job_type in trans_rsu.caching_contents:
                        # 如果作业类型在缓存内容中，QoE增加15%，但不超过1
                        qoe = min(qoe + qoe * 0.15, 1)
                        use_caching = True
                    else:
                        # 如果作业类型不在缓存内容中，QoE减少10%，但不低于0
                        qoe = max(qoe - qoe * 0.1, 0)
                        use_caching = False

                    # 只有第一次进来才append
                    if use_caching or veh.connected_rsu_id != veh.pre_connected_rsu_id:
                        veh.first_time_caching = False
                        # 记录缓存命中状态为1
                        caching_hit_states[veh.connected_rsu_id].append(1)
                    else:
                        veh.first_time_caching = False
                        # 记录缓存命中状态为0
                        caching_hit_states[veh.connected_rsu_id].append(0)

                    # 计算传输RSU的能源效率
                    trans_rsu.ee = max_ee * (1 - trans_rsu.cp_usage)

                    # 将QoE乘以QoE因子加上传输RSU的能源效率，添加到传输QoE列表中
                    trans_qoes[veh.connected_rsu_id].append(
                        float(qoe * qoe_factor + trans_rsu.ee)
                    )
                    # 将QoE乘以QoE因子加上处理RSU的能源效率，添加到处理QoE列表中
                    proc_qoes[p_rsu.id].append(float(qoe * qoe_factor + p_rsu.ee))

                else:
                    # 如果处理RSU不是车辆连接的RSU
                    qoe = min(process_qoe, trans_qoe)
                    # 跳数惩罚，QoE减少跳数惩罚率，不低于0
                    qoe = max(qoe - qoe * hop_penalty_rate, 0)

                    if veh.job.job_type in rsus[veh.connected_rsu_id].caching_contents:
                        # 如果作业类型在连接RSU的缓存内容中，QoE增加15%，但不超过1
                        qoe = min(qoe + qoe * 0.15, 1)
                        use_caching = True
                    else:
                        # 如果作业类型不在连接RSU的缓存内容中，QoE减少10%，但不低于0
                        qoe = max(qoe - qoe * 0.1, 0)
                        use_caching = False

                    if use_caching or veh.connected_rsu_id != veh.pre_connected_rsu_id:
                        veh.first_time_caching = False
                        # 记录缓存命中状态为1
                        caching_hit_states[veh.connected_rsu_id].append(1)
                    else:
                        veh.first_time_caching = False
                        # 记录缓存命中状态为0
                        caching_hit_states[veh.connected_rsu_id].append(0)

                    # 计算处理RSU的能源效率
                    p_rsu.ee = max_ee * (1 - p_rsu.cp_usage)
                    # 计算传输RSU的能源效率
                    trans_rsu.ee = max_ee * (1 - trans_rsu.cp_usage)

                    # 将QoE乘以QoE因子加上传输RSU的能源效率，添加到传输QoE列表中
                    trans_qoes[veh.connected_rsu_id].append(
                        float(qoe * qoe_factor + trans_rsu.ee)
                    )
                    # fixed
                    proc_qoes[p_rsu.id].append(float(qoe * qoe_factor + p_rsu.ee))

        # 计算被处理RSU的平均QoE
        num_proc_rus = len(proc_qoes.keys())
        # 没有进入if-else
        if num_proc_rus == 0:
            continue
        else:
            # 作业分配比例之和不能超过1.0
            if job_ratio_all > 1.0:
                assert NotImplementedError("Impossible value")

            # 将传输QoE列表展平
            flattened_trans_qoes = list(chain.from_iterable(trans_qoes.values()))
            # 计算传输QoE的平均值
            avg_trans_qoes = np.mean(flattened_trans_qoes)
            # 计算加权平均传输QoE
            weighted_avg_trans_qoes = float(avg_trans_qoes * job_ratio_all)

            # 将处理QoE列表展平
            flattened_proc_qoes = list(chain.from_iterable(proc_qoes.values()))
            # 计算处理QoE的平均值
            avg_proc_qoes = np.mean(flattened_proc_qoes)
            # 计算加权平均处理QoE
            weighted_avg_proc_qoes = float(avg_proc_qoes * job_ratio_all)

            # 取加权平均传输QoE和加权平均处理QoE的最小值作为最终QoE
            qoe = min(weighted_avg_trans_qoes, weighted_avg_proc_qoes)

            # 预迁移QoE，刚进来时有多少job ratio就按系数比例加多少qoe
            # if veh.connected_rsu_id != veh.pre_connected_rsu_id:
            #     t_rsu: Rsu = rsus[veh.connected_rsu_id]
            #     idx_pre = t_rsu.pre_handling_jobs.index((veh, 0))
            #     idx_now = t_rsu.handling_jobs.index((veh, 0))
            #     eps = 0
            #     job_ratio_all_add_eps = job_ratio_all + eps

            #     if idx_pre is not None:
            #         veh, ratio = t_rsu.pre_handling_jobs[idx_pre]
            #         # prehandling 奖励
            #         qoe = max(qoe + qoe * ratio / job_ratio_all_add_eps, max_qoe)

            #     if idx_now is not None:
            #         veh, ratio = t_rsu.handling_jobs[idx_now]
            #         # nowhandling 奖励
            #         qoe = max(qoe + qoe * ratio / job_ratio_all_add_eps, max_qoe)

            # 导入车辆
            if veh.job.pre_qoe is None:
                # 如果车辆的前一个QoE为空，则将当前QoE赋值给前一个QoE
                veh.job.pre_qoe = qoe
                # 记录前一个处理QoE
                veh.job.pre_proc_qoe = weighted_avg_proc_qoes
                # 记录前一个传输QoE
                veh.job.pre_trans_qoe = weighted_avg_trans_qoes
                # 效用等于最终QoE
                utility = qoe
            else:
                # 抖动因子0.2，如果增加可以添加更多QoE，否则
                utility = 0.2 * (qoe - veh.job.pre_qoe) + 0.8 * qoe

                # 更新车辆的前一个QoE、处理QoE和传输QoE
                veh.job.pre_qoe = veh.job.qoe
                veh.job.pre_proc_qoe = veh.job.proc_qoe
                veh.job.pre_trans_qoe = veh.job.trans_qoe

            # 更新车辆的传输QoE、处理QoE和当前QoE
            veh.job.trans_qoe = weighted_avg_trans_qoes
            veh.job.proc_qoe = weighted_avg_proc_qoes
            veh.job.qoe = qoe

            # 效用计算
            # 如果要把这个策略清理，需要修改proc_qoes
            for rsu_id, qoes in proc_qoes.items():
                # 缓存命中次数除以总缓存访问次数
                if len(caching_hit_states[rsu_id]) != 0:
                    caching_hit_ratio = sum(caching_hit_states[rsu_id]) / len(
                        caching_hit_states[rsu_id]
                    )
                    caching_hit_ratios[rsu_id] = caching_hit_ratio

                if rsu_id == trans_rsu.id:
                    # 这个trans_rsu id理论上必有，且理论上必是这三个proc中的一个，重复会稀释，需要检查吗
                    # 是否分别导入trans和proc？
                    rsu_utility_dict[trans_rsu.id].append(utility)
                    # 不重复导入
                else:
                    rsu_utility_dict[rsu_id].append(utility)

    return rsu_utility_dict, caching_hit_ratios
