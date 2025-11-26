#!/usr/bin/env python
import math
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
# 是否为离散动作空间
is_discrete = True
# 要使用的地图名称
map_name = "london"

# 创建用于训练的环境
def make_train_env():
    def get_env_fn():
        def init_env():
            # 初始化环境，传入相关参数
            env = Env(
                None, max_step=env_max_step, is_discrete=is_discrete, map=map_name
            )
            return env
        return init_env
    # 使用 ShareDummyVecEnv 包装环境，适用于单线程环境
    return ShareDummyVecEnv([get_env_fn()], is_discrete=is_discrete)

# 创建用于评估的环境
def make_eval_env():
    def get_env_fn():
        def init_env():
            # 初始化环境，传入相关参数
            env = Env(
                None, max_step=env_max_step, is_discrete=is_discrete, map=map_name
            )
            return env
        return init_env
    # 使用 ShareDummyVecEnv 包装环境，适用于单线程环境
    return ShareDummyVecEnv([get_env_fn()], is_discrete=is_discrete)

# 解析命令行参数
def parse_args(args, parser):
    # 添加自定义的命令行参数，指定要运行的地图名称
    parser.add_argument(
        "--map_name", type=str, default="Seattle", help="Which sumo map to run on"
    )
    # 解析命令行参数，返回解析后的参数对象
    all_args = parser.parse_known_args(args)[0]
    return all_args

# 主函数，程序的入口点
def main(args):
    # 从 env_config 模块中导入随机种子
    from env_config import SEED

    # 训练线程的数量
    n_training_threads = 1
    # 是否使用确定性的 CUDA 操作
    cuda_deterministic = False
    # 时间分割技巧开关
    time_spilit = False
    # 是否使用 CADP 技巧
    use_cadp = True
    # CADP 技巧的断点步数
    cadp_breakpoint = math.floor(max_step * 0.01)
    # 环境名称
    env_name = "vanet"
    # 算法名称
    alg_name = "rmappo"
    # alg_name = "ippo"
    # 实验名称前缀
    exp_prefix = "time_all" if not time_spilit else "time_spilitted"
    # 是否使用 wandb 进行日志记录
    use_wandb = True
    # 随机种子
    seed = SEED
    # 是否进行评估
    use_eval = False
    # 获取默认的配置解析器
    parser = get_config()
    # 解析命令行参数
    all_args = parse_args(args, parser)
    # 设置地图名称
    all_args.map_name = map_name

    # 如果没有指定随机种子，则随机生成一个
    if not all_args.seed_specify:
        all_args.seed = np.random.randint(10000, 100000)

    print("seed is :", all_args.seed)

    # 设置是否使用 CADP 技巧和 CADP 断点步数
    all_args.use_cadp = use_cadp
    all_args.cadp_breakpoint = cadp_breakpoint
    # 设置评论家网络的学习率
    all_args.critic_lr = 4e-4
    # 设置学习率
    all_args.lr = 5e-4
    # 设置环境的总步数
    all_args.num_env_steps = max_step
    # 设置每个回合的最大步数
    all_args.episode_length = env_max_step if not time_spilit else env_max_step // 10
    # 设置日志记录的间隔
    all_args.log_interval = 1
    # 设置算法名称
    all_args.algorithm_name = alg_name
    # 设置实验名称
    all_args.experiment_name = (
        (
            exp_prefix + "_" + "Mulit_discrete"
            if is_discrete
            else exp_prefix + "_" + "Box"
        )
        + "_cadp"
        if use_cadp
        else ""
    )

    # 根据选择的算法设置相关参数
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

    # # 检查是否有可用的 CUDA 设备
    # if torch.cuda.is_available():
    #     print("choose to use gpu...")
    #     # 使用第一个 GPU 设备
    #     device = torch.device("cuda:0")
    #     # 设置 PyTorch 的线程数
    #     torch.set_num_threads(n_training_threads)
    #     if cuda_deterministic:
    #         # 禁用 CUDA 的自动调优，保证结果可复现
    #         torch.backends.cudnn.benchmark = False
    #         torch.backends.cudnn.deterministic = True
    # else:
    #     print("choose to use cpu...")
    #     # 使用 CPU 设备
    #     device = torch.device("cpu")
    #     # 设置 PyTorch 的线程数
    #     torch.set_num_threads(n_training_threads)


    # 使用 CPU 设备
    device = torch.device("cpu")
    # 设置 PyTorch 的线程数
    torch.set_num_threads(n_training_threads)


    # 定义保存运行结果的目录
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / env_name
        / alg_name
    )

    # 如果目录不存在，则创建目录
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 如果使用 wandb 进行日志记录
    if use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name)
            + "_"
            + str(all_args.experiment_name)
            + "_"
            + str("nb")
            + "_seed"
            + str(all_args.seed),
            group=all_args.map_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            # 获取已存在的运行目录编号
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
        # 如果目录不存在，则创建目录
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # 设置进程名称
    setproctitle.setproctitle(str(alg_name) + "-" + str(env_name))

    # 设置随机种子，保证结果可复现
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 创建训练环境
    envs = make_train_env()
    # 创建评估环境
    eval_envs = make_eval_env()
    # 获取环境中的智能体数量
    num_agents = envs.num_agents

    # 定义配置字典，包含所有必要的参数
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # 导入 VANETRunner 类
    from vanet_env.onpolicy.runner.shared.vanet_runner2 import (
        VANETRunner as Runner,
    )

    # 创建 Runner 实例
    runner = Runner(config)
    # 开始训练
    runner.run()

    # 训练结束后关闭训练环境
    envs.close()
    # 如果使用评估环境且评估环境与训练环境不同，则关闭评估环境
    if use_eval and eval_envs is not envs:
        eval_envs.close()

    # 如果使用 wandb 进行日志记录，则结束 wandb 运行
    if use_wandb:
        run.finish()
    else:
        # 导出日志数据到 JSON 文件
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        # 关闭日志写入器
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
