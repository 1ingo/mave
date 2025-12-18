import cProfile
import math
import pstats
import sys

import pandas as pd

sys.path.append("./")

import numpy as np
import random
from collections import defaultdict
from vanet_env import env

seed = 114514
random.seed(seed)
np.random.seed(seed)

import sys
import os

sys.path.append("./")
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from vanet_env import handler
from vanet_env.env import Env
from vanet_env.onpolicy.config import get_config
from vanet_env.onpolicy.envs.env_wrappers import (
    ShareSubprocVecEnv,
    ShareDummyVecEnv,
)

# 环境的最大步数
env_max_step = 10850
# 总的最大步数
max_step = env_max_step * 1000
# 是否使用离散动作空间
is_discrete = True
# 渲染模式，这里设置为 "human" 表示以人类可见的方式渲染
render_mode = "human"
# 地图名称，这里设置为 "seattle"
# map_name = "london"
map_name = "seattle"


# 创建用于评估的环境
def make_eval_env():
    def get_env_fn():
        def init_env():
            # 初始化环境，设置渲染模式、最大步数、是否离散和地图名称
            env = Env(
                render_mode,
                max_step=env_max_step,
                is_discrete=is_discrete,
                map=map_name
            )
            return env

        return init_env

    # 使用 ShareDummyVecEnv 创建一个包含单个环境的向量环境
    return ShareDummyVecEnv([get_env_fn()], is_discrete=is_discrete)


# 创建用于训练的环境
def make_train_env():
    def get_env_fn():
        def init_env():
            # 初始化环境，不设置渲染模式，设置最大步数、是否离散和地图名称
            env = Env(
                None, max_step=env_max_step, is_discrete=is_discrete, map=map_name
            )
            return env

        return init_env

    # 使用 ShareDummyVecEnv 创建一个包含单个环境的向量环境
    return ShareDummyVecEnv([get_env_fn()], is_discrete=is_discrete)


# 多智能体策略类，包含多种策略和实验运行方法
class MultiAgentStrategies:
    def __init__(self, env):
        """
        初始化策略模块，接收一个环境实例。

        Args:
            env: 多智能体环境实例。
        """
        self.env = env
        # 智能体的数量
        self.num_agents = len(env.agents)
        # 每个智能体的动作空间
        self.action_spaces = {
            agent: env.multi_discrete_action_space(agent) for agent in env.agents
        }

    def random_strategy(self, obs, infos):
        """
        随机策略，每个路边单元（RSU）选择一个随机动作。

        Returns:
            actions: 一个字典，将智能体 ID 映射到它们的随机动作。
        """
        actions = [self.action_spaces[agent].sample() for agent in self.env.agents]
        return [actions]

    def greedy_strategy(self, obs, infos):
        """
        贪心策略，每个路边单元（RSU）选择能最大化即时奖励的动作。
        贪心策略：采样 max_steps 次然后存储。

        Returns:
            actions: 一个字典，将智能体 ID 映射到它们的贪心动作。
        """
        # actions = {}
        # for agent in self.env.agents:
        #     best_action = None
        #     best_reward = float("-inf")
        #     self.env.reset()
        #     for _ in range(100):  # 采样 100 个动作以近似最佳动作。
        #         action = [
        #             self.action_spaces[agent].sample() for agent in self.env.agents
        #         ]
        #         obs, reward, _, _, _ = self.env.step([action])
        #         if reward[agent] > best_reward:
        #             best_reward = reward[agent]
        #             best_action = action

        #     actions[agent] = best_action
        actions = {}
        for agent in self.env.agents:
            local_obs = obs[agent]["local_obs"]
            global_obs = obs[agent]["global_obs"]
            idles = infos[agent]["idle"]

        return [actions]

    def heuristic_strategy(self, obs, infos):
        """
        启发式策略，使自身和相邻的路边单元（RSU）平均资源和平均平衡。
        一种在相邻 RSU 之间平衡负载的启发式策略。
        """
        actions = []
        for idx, agent in enumerate(self.env.agents):
            agent_obs = obs[agent]
            local_obs = agent_obs["local_obs"]
            # 自身处理中的任务比例
            self_handlings = local_obs[1]
            # 相邻 RSU 的状态
            nb_state = local_obs[4]

            # 相邻 RSU 的空闲比例
            idle_nb_ratio = 1 - nb_state
            # 自身的空闲比例
            idle_self_ratio = 1 - self_handlings
            # 总的空闲比例
            idle_all_ratio = idle_self_ratio + idle_nb_ratio

            # 队列中的连接数
            queue_connections = local_obs[2]
            # 已连接的数量
            connected = local_obs[3]
            # 空闲的连接比例
            idle_connected = 1 - connected

            if idle_all_ratio == 0:
                # 几乎不可能都吃满，假如都吃满则均分
                self_mratio = [2] * self.env.action_space_dims[0]
                nb_mratio = [2] * self.env.action_space_dims[1]
            else:
                self_mratio = [
                                  math.floor(idle_self_ratio / idle_all_ratio * self.env.bins)
                              ] * self.env.action_space_dims[0]
                nb_mratio = [
                                math.floor(idle_nb_ratio / idle_all_ratio * self.env.bins)
                            ] * self.env.action_space_dims[1]
            # 顺便确认 nb 是否正常
            # 根据 idle_ratio 确定分配值？
            # choice 里的值可以改为 math.floor(... * self.env.bins)
            if idle_all_ratio >= 1:
                # 高空闲倾向于分配高 job_ratio 和高 alloc、cp_usage 等
                job_ratios = [
                                 int(np.random.choice([3, 4]))
                             ] * self.env.action_space_dims[2]
                cp_alloc = [int(np.random.choice([3, 4]))] * self.env.action_space_dims[
                    3
                ]
                # 节能选择
                cp_usage = [
                               int(np.random.choice([0, 1, 2]))
                           ] * self.env.action_space_dims[5]
            else:
                job_ratios = [
                                 int(np.random.choice([0, 1, 2]))
                             ] * self.env.action_space_dims[2]
                cp_alloc = [
                               int(np.random.choice([0, 1, 2]))
                           ] * self.env.action_space_dims[3]
                # 激进选择
                cp_usage = [int(np.random.choice([3, 4]))] * self.env.action_space_dims[
                    5
                ]

            # 空闲可支配大于需要的，bw 策略可以激进
            if idle_connected >= queue_connections:
                bw_alloc = [int(np.random.choice([3, 4]))] * self.env.action_space_dims[
                    4
                ]
            else:
                bw_alloc = [
                               int(np.random.choice([0, 1, 2]))
                           ] * self.env.action_space_dims[4]

            # 模拟 FIFO，这里应该写在 env 里然后这里调用，时间匆忙，直接写外面了
            # 模拟 LRU，这里应该写在 env 里然后这里调用，时间匆忙，直接写外面了
            # 模拟 Random，这里应该写在 env 里然后这里调用，时间匆忙，直接写外面了
            rsu = self.env.rsus[idx]
            veh_id = rsu.range_connections.last()
            # 根据 max_caching 改动这一项
            if veh_id in self.env.vehicle_ids:
                caching_content = [self.env.vehicles[veh_id].job_type]
            else:
                caching_content = [
                    int(np.random.choice([i for i in range(self.env.bins)]))
                ]

            action = (
                    self_mratio
                    + nb_mratio
                    + job_ratios
                    + cp_alloc
                    + bw_alloc
                    + cp_usage
                    + caching_content
            )

            if self.action_spaces[agent].contains(action):
                pass
            else:
                print("action 不在 action_space 内")
                assert IndexError("...")

            actions.append(action)

        return [actions]

    def fairalloc_strategy(self, obs, infos):
        """
        公平分配策略，使用 max-min 公平分配的思想，
        在自身与邻居之间均分可分配的资源 bins，并对其它资源指标采用中间值分配策略，
        以实现公平分配。

        与 heuristic_strategy 不同，该策略不再依据当前的空闲状态动态调整，
        而是采用固定的“中间值”作为各项指标的分配值，从而在负载不均或数据噪声较大的情况下，
        也能实现一种较为稳定的公平分配。
        """
        actions = []
        for idx, agent in enumerate(self.env.agents):
            agent_obs = obs[agent]
            local_obs = agent_obs["local_obs"]

            # 获取与自身有关的状态（仅用于其它判断，本策略中主要忽略空闲比例）
            self_handlings = local_obs[1]
            nb_state = local_obs[4]
            # 其余状态如 queue_connections、connected 等也可以用于其它判断
            queue_connections = local_obs[2]
            connected = local_obs[3]
            idle_connected = 1 - connected

            # 对于 self 和 neighbor 的资源分配，采用 max-min 公平原则，即尽可能均分
            # 注意：bins 总量可能为奇数，此处我们将其尽可能均分，自身获得 floor(bins/2)，邻居获得剩余部分
            fair_share_self = self.env.bins // 2
            fair_share_nb = self.env.bins - fair_share_self

            # action_space 中第 0 部分表示自有资源分配，第 1 部分表示邻居资源分配
            self_mratio = [fair_share_self] * self.env.action_space_dims[0]
            nb_mratio = [fair_share_nb] * self.env.action_space_dims[1]

            # 对于其它资源调度参数，由于不再考虑当前状态的极端情况，
            # 故我们统一使用中间值（比如 2）作为分配指标，达到公平分配的目的
            # 注意：这里给定的数值 2 是在允许的离散取值范围内的中间值，可根据环境实际情况修改
            job_ratios = [2] * self.env.action_space_dims[2]
            cp_alloc = [2] * self.env.action_space_dims[3]
            bw_alloc = [2] * self.env.action_space_dims[4]
            cp_usage = [2] * self.env.action_space_dims[5]

            # 对于缓存内容（caching），可以根据当前最近连接车辆的任务类型来决定
            rsu = self.env.rsus[idx]
            veh_id = rsu.range_connections.last()
            if veh_id in self.env.vehicle_ids:
                caching_content = [self.env.vehicles[veh_id].job_type]
            else:
                # 若当前无有效车辆连接，则随机选择一个 job_type
                caching_content = [
                    int(np.random.choice([i for i in range(self.env.bins)]))
                ]

            # 将各个部分拼接成一个完整的 action 向量
            action = (
                    self_mratio
                    + nb_mratio
                    + job_ratios
                    + cp_alloc
                    + bw_alloc
                    + cp_usage
                    + caching_content
            )

            # 检查生成的 action 是否在允许的 action_space 内
            if not self.action_spaces[agent].contains(action):
                print("action 不在 action_space 内")
                raise IndexError("生成的 action 超出合法范围！")

            actions.append(action)

        return [actions]

    def run_experiment(self, strategy=None, strategy_name=None, steps=1000):
        """
        使用给定的策略运行模拟实验。

        Args:
            strategy: 生成动作的策略函数。
            steps: 模拟的步数。

        Returns:
            metrics: 一个字典，包含 QoE、Delay (替换 EE)、奖励和资源使用情况随时间的变化。
        """
        if strategy is not None:
            self.strategy = strategy
        else:
            if strategy_name == "random_strategy":
                self.strategy = self.random_strategy
            elif strategy_name == "heuristic_strategy":
                self.strategy = self.heuristic_strategy
            else:
                self.strategy = self.fairalloc_strategy
        # 记录每个时间步的用户体验质量（QoE）
        qoe_records = []
        # 记录每个时间步的平均延迟（Delay），替换原来的 EE
        delay_records = []
        # 记录每个时间步的资源使用情况
        resource_records = []
        # 记录每个时间步的奖励
        reward_records = []
        # 记录每个时间步的可用奖励
        ava_reward_records = []
        # 记录每个时间步的命中率
        hit_ratio_records = []
        # 记录每个时间步的空闲掩码
        idle_masks = []

        # 重置环境，获取初始观测和信息
        obs, _, infos = self.env.reset()

        # 计算初始的空闲掩码
        idle_mask = np.array(
            [infos[agent_id]["idle"] for agent_id in self.env.possible_agents]
        )

        # 将初始的空闲掩码添加到列表中
        idle_masks.append(idle_mask)

        # 循环执行指定的步数
        for time_step in range(steps - 1):
            # 根据当前策略生成动作
            actions = self.strategy(obs, infos)
            # 执行动作，获取新的观测、奖励、终止标志和信息
            obs, rewards, _, _, infos = self.env.step(actions)

            # 计算当前时间步的空闲掩码
            idle_mask = np.array(
                [infos[agent_id]["idle"] for agent_id in self.env.possible_agents]
            )

            # 将当前时间步的空闲掩码添加到列表中
            idle_masks.append(idle_mask)

            # 收集指标
            # 计算所有车辆的平均 QoE
            qoe_real = []
            delays = []
            hit_ratios = []

            for rsu in self.env.rsus:
                # [新增] 读取每个 RSU 的平均处理延迟 (由 utility.py 计算)
                # 如果 RSU 没有处理任务，avg_total_delay 可能是 0.0
                delays.append(float(rsu.avg_total_delay))

                # 收集车辆 QoE
                for vid in rsu.range_connections:
                    if vid in self.env.vehicle_ids:
                        qoe_real.append(float(self.env.vehicles[vid].job.qoe))

                # 收集缓存命中率
                for hit_ratio in rsu.hit_ratios:
                    hit_ratios.append(hit_ratio)

            # 创建一个全为 1 的活动掩码
            active_masks = np.ones((self.num_agents), dtype=np.float32)

            # 将空闲的智能体对应的活动掩码设置为 0
            active_masks[(idle_masks[time_step] == True)] = 0
            # resource_usage = np.mean([rsu.cp_usage for rsu in self.env.rsus])

            # 初始化 reward 和 ava_rews，防止为空报错
            reward = 0.0
            ava_rews = 0.0
            if rewards != []:
                # 计算平均奖励
                reward = np.mean([reward for reward in rewards.values()])
                # 计算可用奖励
                if active_masks.sum() > 0:
                    ava_rews = (np.array(list(rewards.values())) * active_masks).sum() / (
                            active_masks.sum() + 1e-6
                    )

            # 将当前时间步的平均 QoE 添加到记录列表中
            qoe_records.append(np.mean(qoe_real) if qoe_real else 0.0)
            # 将当前时间步的平均 Delay 添加到记录列表中 (替换 EE)
            delay_records.append(np.mean(delays))

            # resource_records.append(resource_usage)
            # 将当前时间步的奖励添加到记录列表中
            reward_records.append(reward)
            # 将当前时间步的可用奖励添加到记录列表中
            ava_reward_records.append(ava_rews)
            # 将当前时间步的平均命中率添加到记录列表中
            hit_ratio_records.append(np.nanmean(hit_ratios) if hit_ratios else 0.0)

        # 关闭环境
        self.env.close()
        return {
            "QoE": qoe_records,
            "Delay": delay_records,  # 键名更改为 Delay
            "Rewards": reward_records,
            "Ava_rewards": ava_reward_records,
            "Hit_ratio": hit_ratio_records,
        }


# 解析命令行参数
def parse_args(args, parser):
    # 添加地图名称参数
    parser.add_argument(
        "--map_name", type=str, default="Seattle", help="Which sumo map to run on"
    )
    # 解析已知参数
    all_args = parser.parse_known_args(args)[0]
    return all_args


# 使用 rMAPPO 算法进行评估
def rmappo(args):
    from env_config import SEED

    # 训练线程数
    n_training_threads = 1
    # 是否使用确定性的 CUDA
    cuda_deterministic = False
    # 环境名称
    env_name = "vanet"
    # 算法名称
    alg_name = "rMAPPO_ts"
    # 是否使用 wandb 进行日志记录
    use_wandb = True
    # 随机种子
    seed = SEED
    # 是否进行评估
    is_eval = True
    # 是否使用评估环境
    use_eval = False
    # 获取配置解析器
    parser = get_config()
    # 解析命令行参数
    all_args = parse_args(args, parser)
    if not all_args.seed_specify:
        all_args.seed = np.random.randint(10000, 100000)

    print("seed is :", all_args.seed)

    all_args.num_env_steps = max_step
    all_args.episode_length = env_max_step
    all_args.log_interval = 1
    # 根据是否评估设置前缀
    prefix = "train" if not is_eval else "eval"
    all_args.algorithm_name = "rmappo"
    all_args.experiment_name = "Mulit_discrete" if is_discrete else "Box"
    # 模型目录
    model_dir = r"D:\code\Multi-Agent-Vanet-Env-main\results\vanet\rmappo\wandb\run-20251126_092344-io0w5iac\files"

    all_args.model_dir = model_dir

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print(
            "u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False"
        )
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # # cuda
    # if torch.cuda.is_available():
    #     print("choose to use gpu...")
    #     device = torch.device("cuda:0")
    #     torch.set_num_threads(n_training_threads)
    #     if cuda_deterministic:
    #         torch.backends.cudnn.benchmark = False
    #         torch.backends.cudnn.deterministic = True
    # else:
    #     print("choose to use cpu...")
    #     device = torch.device("cpu")
    #     torch.set_num_threads(n_training_threads)

    print("choose to use cpu...")
    device = torch.device("cpu")
    torch.set_num_threads(n_training_threads)

    # 运行目录
    run_dir = (
            Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
            / env_name
            / alg_name
    )

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            notes=socket.gethostname(),
            name=str(prefix)
                 + "_"
                 + str(all_args.algorithm_name)
                 + "_"
                 + str(all_args.experiment_name)
                 + "_"
                 + str("nb")
                 + "_seed"
                 + str(all_args.seed),
            group=all_args.map_name,
            dir=str(run_dir),
            job_type="training" if not is_eval else "evaling",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # 设置进程标题
    setproctitle.setproctitle(str(alg_name) + "-" + str(env_name))

    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 创建训练环境
    envs = make_train_env()
    # 创建评估环境
    eval_envs = make_eval_env()

    # eval_envs = make_eval_env()
    # 智能体数量
    num_agents = envs.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # 运行实验
    # if all_args.share_policy:
    #     from onpolicy.runner.shared.smac_runner import SMACRunner as Runner
    # else:
    #     from onpolicy.runner.separated.smac_runner import SMACRunner as Runner
    from vanet_env.onpolicy.runner.shared.vanet_runner2 import (
        VANETRunner as Runner,
    )

    runner = Runner(config)
    runner.run_eval()

    # 后处理
    envs.close()
    # if use_eval and eval_envs is not envs:
    #     eval_envs.close()

    if use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


# 运行其他策略的实验
def other_policy(args, render=None):
    # 实验名称
    exp_name = "multi_discrete"
    # 算法名称，可选择不同的策略
    # alg_name = "random_strategy"
    alg_name = "heuristic_strategy"
    # alg_name = "fairalloc_strategy"

    # 是否记录日志
    log = True

    # 步数
    step = env_max_step
    # 初始化环境
    env = Env(render, max_step=step, map=map_name)
    # 初始化策略模块
    strategies = MultiAgentStrategies(env)

    if log:
        # 运行实验并获取指标
        metrics = strategies.run_experiment(strategy_name=alg_name, steps=step)

        print(f"{alg_name}:{metrics}")
        # 计算平均奖励
        av = np.mean(metrics["Rewards"])
        # 计算平均 QoE
        avg_qoe = np.mean(metrics["QoE"])
        # 计算平均命中率
        avg_hit_ratio = np.nanmean(metrics["Hit_ratio"])
        # 计算平均可用奖励
        avg_ava_rew = np.mean(metrics["Ava_rewards"])  # 修正了之前的拼写错误
        print(f"{alg_name}_avg_step_reward:{av}")
        print(f"{alg_name}_avg_step_qoe:{avg_qoe}")
        print(f"{alg_name}_avg_step_hit_ratio:{avg_hit_ratio}")
        print(f"{alg_name}_avg_step_ava_reward:{avg_ava_rew}")

        # 将指标保存到 DataFrame 中
        # [修改] 替换 EE 为 Delay
        df = pd.DataFrame(
            metrics, columns=["QoE", "Delay", "Rewards", "Ava_rewards", "Hit_ratio"]
        )

        from datetime import datetime

        # 获取当前时间
        current_time = datetime.now()

        # 格式化时间
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        # 将 DataFrame 保存为 CSV 文件
        df.to_csv(f"{map_name}_{alg_name}__{formatted_time}.csv", index=False)
        print(f"CSV 文件已生成：{map_name}_{alg_name}__{formatted_time}.csv")
        print(f"仿真运行时间：{(env.endtime - env.start_time).seconds}s")
    # metrics_greedy = strategies.run_experiment(strategies.greedy_strategy, steps=3600)
    # print(f"random:{metrics_greedy}")

    # metrics_heuristic = strategies.run_experiment(
    #     strategies.heuristic_strategy, steps=36_000_000
    # )
    # print(f"random:{metrics_heuristic}")


def main(args):
    #
    # # cProfile.run("other_policy()", sort="time")

    # profiler = cProfile.Profile()
    # profiler.enable()

    # 使用其他策略或算法,如需使用rmappo,需要使用函数rmappo(args=args)
    # other_policy(args, None)

    rmappo(args)
    # profiler.disable()
    # # 创建 Stats 对象并排序
    # stats = pstats.Stats(profiler)
    # stats.sort_stats("time")  # 按内部时间排序
    # stats.reverse_order()  # 反转排序顺序（从升序变为降序，或从降序变为升序）
    # stats.print_stats()  # 打印结果


if __name__ == "__main__":
    main(sys.argv[1:])