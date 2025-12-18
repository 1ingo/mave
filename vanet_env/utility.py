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
        vehs: Dict[str, Vehicle],
        rsus: List[Rsu],
        rsu_network,
        time_step,
        fps,
        weight,
        cache_module,  # 必须传入 cache 模块用于计算内容大小
        max_qoe=env_config.MAX_QOE,
        int_utility=False,
        max_connections=env_config.MAX_CONNECTIONS,
        num_cores=env_config.NUM_CORES,
):
    rsu_utility_dict = defaultdict(list)
    caching_hit_ratios = {}
    rsu_delay_stats = defaultdict(list)

    # 保持原有的权重结构
    hop_penalty_rate = 0.1
    time_factor = 0.8
    qoe_factor = 1 - time_factor

    if time_step >= 50:
        time_step

    for v_id, veh in vehs.items():
        if veh.vehicle_id not in rsus[veh.connected_rsu_id].range_connections:
            continue

        veh: Vehicle

        # 计算基准时间
        # 用于将物理延迟 (秒) 转换为 0-1 的得分
        t_ref_comm = veh.job.job_size / (env_config.JOB_DR_REQUIRE * 1e6) if env_config.JOB_DR_REQUIRE > 0 else 1.0
        t_ref_comp = env_config.JOB_CP_REQUIRE / (env_config.RSU_COMPUTATION_POWER / 2)
        t_reference = t_ref_comm + t_ref_comp + 0.1

        # 云端处理
        if veh.is_cloud:
            veh.job.job_processed = veh.job.job_size  # 瞬间完成

            # 云端只有 0.1 的基础分
            qoe = max_qoe * 0.1

            t_comm_total = network.v2c_delay(rsus[veh.connected_rsu_id], veh)
            t_comp_cloud = env_config.CLOUD_COMPUTATIONALLY_TIME
            t_total = t_comm_total + t_comp_cloud
            rsu_delay_stats[veh.connected_rsu_id].append(t_total)

            # 延迟越大，得分越低
            delay_score = min(t_reference / (t_total + 1e-9), 1.0)

            final_utility = qoe * qoe_factor + time_factor * delay_score
            rsu_utility_dict[veh.connected_rsu_id].append(float(final_utility))

            # 更新状态
            if veh.job.pre_qoe is None:
                veh.job.pre_qoe = qoe
            else:
                veh.job.pre_qoe = veh.job.qoe
            veh.job.qoe = qoe
            continue

        # 边缘 RSU 处理

        # 计算通信延迟 (T_comm)
        if veh.data_rate > 0:
            t_comm = veh.job.job_size / (veh.data_rate * 1e6)
        else:
            t_comm = 100.0

        # 计算传输 QoE (基于瞬时速率)
        trans_qoe = (
                min(veh.data_rate, env_config.JOB_DR_REQUIRE) / env_config.JOB_DR_REQUIRE
        )

        trans_qoes = defaultdict(list)
        proc_qoes = defaultdict(list)
        caching_hit_states = defaultdict(list)

        if veh.job.processing_rsus.is_empty():
            if veh.connected_rsu_id is not None:
                rsu_utility_dict[veh.connected_rsu_id].append(0.0)

        job_ratio_all = 0.0

        for p_rsu in veh.job.processing_rsus:
            if p_rsu is not None:
                p_rsu: Rsu
                p_idx = p_rsu.handling_jobs.index((veh, 0))
                job_ratio = p_rsu.handling_jobs[p_idx][1]
                job_ratio_all += job_ratio

                if job_ratio != 0:
                    allocated_cp = p_rsu.real_cp_alloc[p_idx % p_rsu.max_cores]

                    veh.job.job_processed += allocated_cp

                    # 计算计算 QoE (基于瞬时算力供给率)
                    process_qoe = (
                            min(
                                allocated_cp / job_ratio,
                                env_config.JOB_CP_REQUIRE,
                            )
                            / env_config.JOB_CP_REQUIRE
                    )

                    # 计算计算延迟 (T_comp)
                    if allocated_cp > 0:
                        t_comp = (env_config.JOB_CP_REQUIRE * job_ratio) / allocated_cp
                    else:
                        t_comp = 100.0
                else:
                    process_qoe = 0.0
                    t_comp = 100.0

                trans_rsu: Rsu = rsus[veh.connected_rsu_id]

                # 计算获取延迟 (T_fetch)
                needed_content_id = veh.job.job_type
                t_fetch = 0.0
                use_caching = False

                if needed_content_id in p_rsu.caching_contents:
                    t_fetch = 0.0
                    use_caching = True
                else:
                    # 未命中，计算去邻居或云端的时间
                    t_fetch = network.calculate_fetch_delay(
                        target_rsu=p_rsu, content_id=needed_content_id,
                        rsus=rsus, rsu_network=rsu_network, cache_module=cache_module
                    )
                    use_caching = False

                # 计算迁移延迟 (T_mig)
                t_mig = 0.0
                if p_rsu.id != veh.connected_rsu_id:
                    t_mig = veh.job.job_size / (env_config.R2R_BANDWIDTH * 1e6) + (env_config.HOP_LATENCY / 1000.0)

                # 计算总延迟并转换为得分
                t_total = t_comm + t_comp + t_fetch + t_mig
                rsu_delay_stats[veh.connected_rsu_id].append(t_total)
                # 延迟得分: 延迟越低越接近 1，延迟越高越接近 0
                delay_score = min(t_reference / (t_total + 1e-9), 1.0)

                # 处理本地 vs 远程的 QoE 惩罚逻辑
                if p_rsu.id == veh.connected_rsu_id:
                    qoe = min(process_qoe, trans_qoe)
                    if use_caching:
                        qoe = min(qoe + qoe * 0.15, 1)
                    else:
                        qoe = max(qoe - qoe * 0.1, 0)
                else:
                    qoe = min(process_qoe, trans_qoe)
                    qoe = max(qoe - qoe * hop_penalty_rate, 0)  # 跳数惩罚
                    if veh.job.job_type in rsus[veh.connected_rsu_id].caching_contents:  # 这里应该查连接RSU还是处理RSU? 原代码是连接RSU
                        qoe = min(qoe + qoe * 0.15, 1)
                        use_caching = True  # 修正原有逻辑的变量复用
                    else:
                        qoe = max(qoe - qoe * 0.1, 0)
                        use_caching = False

                # 记录缓存命中
                if use_caching or veh.connected_rsu_id != veh.pre_connected_rsu_id:
                    veh.first_time_caching = False
                    caching_hit_states[veh.connected_rsu_id].append(1)
                else:
                    veh.first_time_caching = False
                    caching_hit_states[veh.connected_rsu_id].append(0)

                final_utility = qoe * qoe_factor + time_factor * delay_score

                trans_qoes[veh.connected_rsu_id].append(float(final_utility))
                proc_qoes[p_rsu.id].append(float(final_utility))

        num_proc_rus = len(proc_qoes.keys())
        if num_proc_rus == 0:
            continue
        else:
            if job_ratio_all > 1.0:
                assert NotImplementedError("Impossible value")

            flattened_trans_qoes = list(chain.from_iterable(trans_qoes.values()))
            avg_trans_qoes = np.mean(flattened_trans_qoes)
            weighted_avg_trans_qoes = float(avg_trans_qoes * job_ratio_all)

            flattened_proc_qoes = list(chain.from_iterable(proc_qoes.values()))
            avg_proc_qoes = np.mean(flattened_proc_qoes)
            weighted_avg_proc_qoes = float(avg_proc_qoes * job_ratio_all)

            # 这里的 qoe 实际上已经是包含了 delay_score 的 utility 了
            utility_val = min(weighted_avg_trans_qoes, weighted_avg_proc_qoes)

            # 导入车辆历史平滑
            if veh.job.pre_qoe is None:
                veh.job.pre_qoe = utility_val  # 这里其实存的是 Utility
                veh.job.pre_proc_qoe = weighted_avg_proc_qoes
                veh.job.pre_trans_qoe = weighted_avg_trans_qoes
                final_utility = utility_val
            else:
                final_utility = 0.2 * (utility_val - veh.job.pre_qoe) + 0.8 * utility_val
                veh.job.pre_qoe = veh.job.qoe  # 滚动更新
                veh.job.pre_proc_qoe = veh.job.proc_qoe
                veh.job.pre_trans_qoe = veh.job.trans_qoe

            # 更新车辆状态
            veh.job.trans_qoe = weighted_avg_trans_qoes
            veh.job.proc_qoe = weighted_avg_proc_qoes
            veh.job.qoe = final_utility  # 这里的 qoe 属性实际上存的是 utility

            # 记录到 RSU 字典
            for rsu_id, qoes in proc_qoes.items():
                if len(caching_hit_states[rsu_id]) != 0:
                    caching_hit_ratio = sum(caching_hit_states[rsu_id]) / len(caching_hit_states[rsu_id])
                    caching_hit_ratios[rsu_id] = caching_hit_ratio

                if rsu_id == trans_rsu.id:
                    rsu_utility_dict[trans_rsu.id].append(final_utility)
                else:
                    rsu_utility_dict[rsu_id].append(final_utility)

            for rsu in rsus:
                if rsu.id in rsu_delay_stats and len(rsu_delay_stats[rsu.id]) > 0:
                    rsu.avg_total_delay = np.mean(rsu_delay_stats[rsu.id])
                else:
                    rsu.avg_total_delay = 0.0

    return rsu_utility_dict, caching_hit_ratios