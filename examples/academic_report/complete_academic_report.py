from academic_report import AcademicReport
from academic_report import ModelArchitecture
from academic_report import ModelTraining
from academic_report import Reference
from academic_report import ReferenceType
from hdr.contracts.std import Image

topic = "科研导向的人形机器人底层运控调研"
author = "github@5eqn"


holo_references = [
    Reference(
        type=ReferenceType.Repository,
        title="HoloMotion: A Foundation Model for Whole-Body Humanoid Control",
        link="https://github.com/HorizonRobotics/HoloMotion",
    ),
    Reference(
        type=ReferenceType.Checkpoint,
        title="HorizonRobotics/HoloMotion_v1.2",
        link="https://huggingface.co/HorizonRobotics/HoloMotion_v1.2",
    ),
    Reference(
        type=ReferenceType.Checkpoint,
        title="feat(release): uploaded v1.2 motion tracking model",
        link="https://huggingface.co/HorizonRobotics/HoloMotion_v1.2/commit/b656e063caa7b564b9dce9044bb1e38afc54832a",
    ),
    Reference(
        type=ReferenceType.Checkpoint,
        title="feat(model): added holomotion v1.2 velocity tracking model trained with AMP",
        link="https://huggingface.co/HorizonRobotics/HoloMotion_v1.2/commit/02293f6efc81bea65a1126654659462dfeb72992",
    ),
]

sonic_references = [
    Reference(
        type=ReferenceType.Article,
        title="SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control",
        link="https://arxiv.org/abs/2511.07820",
    ),
    Reference(
        type=ReferenceType.Repository,
        title="GR00T-WholeBodyControl",
        link="https://github.com/NVlabs/GR00T-WholeBodyControl",
    ),
    Reference(
        type=ReferenceType.Checkpoint,
        title="nvidia/GEAR-SONIC",
        link="https://huggingface.co/nvidia/GEAR-SONIC",
    ),
    Reference(
        type=ReferenceType.Article,
        title="Training Guide",
        link="https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/training.html",
    ),
]

legged_gym_references = [
    Reference(
        type=ReferenceType.Repository,
        title="unitree_rl_gym",
        link="https://github.com/unitreerobotics/unitree_rl_gym",
    ),
    Reference(
        type=ReferenceType.Repository,
        title="legged_gym",
        link="https://github.com/leggedrobotics/legged_gym",
    ),
    Reference(
        type=ReferenceType.Article,
        title="Training",
        link="https://deepwiki.com/unitreerobotics/unitree_rl_gym/2.1-training",
    ),
]


figure_dir = "examples/academic_report/figures"

holo_architecture_figure = Image(path=f"{figure_dir}/holo_architecture.svg")
holo_training_figure = Image(path=f"{figure_dir}/holo_training.svg")
sonic_architecture_figure = Image(path=f"{figure_dir}/sonic_architecture.svg")
sonic_training_figure = Image(path=f"{figure_dir}/sonic_training.svg")
legged_gym_architecture_figure = Image(path=f"{figure_dir}/legged_gym_architecture.svg")
legged_gym_training_figure = Image(path=f"{figure_dir}/legged_gym_training.svg")


report = AcademicReport(
    topic=topic,
    author=author,
    content=[
        ModelArchitecture(
            content=(
                "HoloMotion v1.2 的控制模型不是一个端到端 VLA，而是面向 Unitree G1 29DoF/23DoF "
                "全身运动跟踪与速度跟踪的低层策略族。系统级架构按数据流拆成：运动数据整理、人体到机器人"
                "重定向、IsaacLab 分布式训练、评估导出、ROS2/ONNX 实机部署。v1.2 发布了 motion tracking "
                "与 velocity tracking 两个预训练 ONNX 模型，其中 motion tracking 配置显示 `dof_mode: '23'`，"
                "机器人动作维度由 `robot_action_dim` 决定，部署评估目标指向 `assets/robots/unitree/G1/29dof/scene_29dof.xml`。"
                "\n\n"
                "核心 actor 是 `ReferenceRoutedGroupedMoETransformerPolicy`：输入分为当前参考状态、机器人自身状态、"
                "上一帧动作和未来参考片段。当前参考项包括参考重力投影、base 线速度/角速度、参考关节角、"
                "参考根高度；机器人自身项包括投影重力、相对根角速度、关节位置、关节速度、上一动作；未来项"
                "包括未来参考关节角、根高度、参考重力、base 线速度和角速度。actor 先用 2048 维观测嵌入和 "
                "2048 维 router 嵌入构造 token，再进入 3 层 transformer/MoE：`d_model=512`、`n_heads=8`、"
                "`n_kv_heads=4`、`num_fine_experts=16`、`num_shared_experts=1`、`top_k=2`、gated attention 开启、"
                "FFN 倍率为 2.0，dense FFN 倍率为 4，最大上下文长度 32。输出直接是机器人动作，通常解释为 "
                "PD 目标偏移或底层控制目标。"
                "\n\n"
                "critic 是独立 MLP，使用更强的特权观测：当前/未来参考关节与根运动、全局 anchor 差、机器人"
                "body link 线速度/角速度、相对 body link 位置/旋转、关节位置/速度、上一动作等。critic "
                "结构为 4 层 2048 hidden dims，SiLU 激活，RMSNorm hidden norm，输出 1 维价值。actor 和 critic "
                "都有 EMA observation normalization、clip range 10.0，并每 4 个 step 同步统计量。"
            ),
            references=holo_references,
            figures=[holo_architecture_figure],
        ),
        ModelTraining(
            content=(
                "HoloMotion v1.2 的训练路径是强化学习加模仿/运动先验：先把大规模人类动作数据转成 AMASS-compatible "
                "SMPL，再通过 GMR 重定向成机器人 HDF5，随后在 IsaacLab 中训练 motion tracking 或 velocity tracking "
                "策略。官方 README 明确支持 motion tracking 和 velocity tracking 两类任务，并把 v1.2 定位为可直接"
                "部署的预训练运动跟踪/速度跟踪模型。"
                "\n\n"
                "motion tracking v1.2 配置给出可复现实验骨架：`num_envs=8192`、`num_processes=64`、headless 训练，"
                "算法目标为 `holomotion.src.algo.ppo_tf.PPOTF`，`num_learning_iterations=20001`，每环境 rollout "
                "`num_steps_per_env=32`，每轮 `num_learning_epochs=3`、`num_mini_batches=24`。PPO 超参为 "
                "`clip_param=0.2`、`gamma=0.99`、`lam=0.95`、`value_loss_coef=1.0`、`entropy_coef=0.005`、"
                "`max_grad_norm=1.0`、clipped value loss 开启、desired KL 0.013。优化器是 AdamW，actor learning "
                "rate 3e-5、critic learning rate 5e-5，自适应 schedule，并有 KL early stop。采样策略使用 "
                "weighted bin：AMASS 0.6、LAFAN1 0.35、pico 0.05。"
                "\n\n"
                "训练中的鲁棒性来自三处：一是 rough terrain generator，4x4 地形网格、20m x 20m 单元、0.1m "
                "horizontal scale、0.005m vertical scale；二是 domain randomization，包括 root pose 随机化、"
                "执行器 stiffness/damping 0.9-1.1 scale 扰动、观测噪声；三是辅助状态预测，包含 base linear "
                "velocity、keybody contact、参考/机器人 keybody 相对位置等辅助损失。导出侧生成 ONNX，并按 "
                "`model_1000.onnx` 到 `model_20000.onnx` 做 MuJoCo eval。velocity tracking v1.2 的 Hugging Face "
                "提交说明其模型同样使用 AMP 训练，导出 checkpoint 为 `model_29500.onnx`。"
            ),
            references=holo_references,
            figures=[holo_training_figure],
        ),
        ModelArchitecture(
            content=(
                "SONIC / GR00T-WBC 是 NVIDIA GEAR 团队面向 Unitree G1 29DoF 的通用低层 whole-body controller。"
                "它在 GR00T 系统里扮演 fast System 1：上层 VLA 或交互规划器给出目标运动、速度、VR 姿态、视频/"
                "文本/音乐生成的人体动作，SONIC 把这些目标统一编码到 token space，再解码成稳定的机器人关节动作。"
                "\n\n"
                "论文和训练文档给出的核心结构是 universal-token architecture。它不是为每种输入写一个控制器，而是"
                "使用多个并行 encoder 接收不同运动格式：G1 robot joint trajectories、VR teleop 3 点 tracking "
                "targets（头和双腕）、SMPL 人体模型关节位置，以及可选 SOMA/BVH skeleton joint positions。每个 encoder "
                "把输入投影到共享 latent token space，并通过 FSQ（Finite Scalar Quantization）离散化/规整 token。"
                "随后单一 decoder 从共享 token 产生 G1 关节动作，因此同一策略可以接 VR 全身遥操作、3 点移动操作、"
                "人体视频估计、文本/音乐生成动作、VLA 输出的 head/wrist/base/navigation command。"
                "\n\n"
                "控制栈还包含一个实时 kinematic planner：它按当前机器人状态和用户命令自回归生成未来 0.8s-2.4s "
                "运动片段，笔记本推理低于 5ms，Jetson Orin 约 12ms，可每 100ms 或命令变化时重规划。规划器输出"
                "未来运动，universal policy 负责物理跟踪。这使 SONIC 在架构上分成三层：输入适配 encoder、共享"
                "token/decoder tracking policy、planner 或 VLA 等上层指令源。GR00T-WholeBodyControl 仓库同时保留"
                "了 Decoupled WBC（下肢 RL + 上肢 IK）与 GEAR-SONIC；SONIC 是更统一的全身策略路线。"
            ),
            references=sonic_references,
            figures=[sonic_architecture_figure],
        ),
        ModelTraining(
            content=(
                "SONIC 的训练目标是把 motion tracking 规模化，而不是手写单项 locomotion reward。论文摘要报告了"
                "三个缩放轴：模型从 1.2M 参数扩到 42M 参数，数据超过 100M frames / 700 小时高质量 motion data，"
                "训练 compute 约 9k GPU hours；正文实验还报告大规模 tracker 在 128 GPUs 训练 3 天，约 32k GPU hours，"
                "用于 100M frames 的运动跟踪评估。这里可理解为公开摘要与实验段落采用了不同统计口径或模型版本，"
                "报告时应保留原文数字而不混成一个单值。"
                "\n\n"
                "公开训练指南给出的可复现训练入口是 `gear_sonic/train_agent_trl.py`，默认配置 "
                "`+exp=manager/universal_token/all_modes/sonic_release`，`num_envs=4096`、`headless=True`。数据需要"
                " motion_lib PKL：Bones-SEED CSV 先以 120fps source 转 30fps，再过滤掉家具交互、车辆、杂技、 elevated "
                "surface 等 G1 不可执行动作；过滤后约 130K/142K motions 保留。默认 `sonic_release` 使用 G1、teleop、"
                "SMPL 三个 encoder；`sonic_bones_seed` 增加 SOMA 第四 encoder。"
                "\n\n"
                "训练算法是 Isaac Lab simulation 中的 PPO，加 auxiliary losses。全量训练示例指定 "
                "`motion_file=data/motion_lib_bones_seed/robot_filtered` 和 `smpl_motion_file=data/smpl_filtered`；"
                "从 released checkpoint 微调用 `+checkpoint=sonic_release/last.pt`。官方建议 64+ GPUs 获得合理收敛"
                "时间；8x L40 约 5-7 天到 100K iterations，8x A100 约 3-5 天，8x H100 约 2-3 天。监控指标包括 "
                "`rewards/total > 3.0`、root position error < 0.15m、body position error < 0.10m、throughput "
                "约 4000+ fps。评估侧要求 well-converged policy 达到 >0.98 success rate 和 <29mm local MPJPE；"
                "导出 ONNX 时按输入模态生成 `*_smpl.onnx`、`*_g1.onnx`、`*_teleop.onnx`、`*_encoder.onnx`、"
                "`*_decoder.onnx`。"
            ),
            references=sonic_references,
            figures=[sonic_training_figure],
        ),
        ModelArchitecture(
            content=(
                "legged_gym / Unitree RL Gym 代表传统可复现的底层运动控制基线：Isaac Gym 并行仿真环境 + RSL-RL PPO "
                "actor-critic。原始 legged_gym 面向 ANYmal 等足式机器人，Unitree fork 把同一范式扩到 Go2、G1、H1、"
                "H1_2。环境输出 observation、privileged observation、reward、done 和 infos；训练时 actor 看常规"
                "观测，critic 可看 privileged observation 做 asymmetric actor-critic。"
                "\n\n"
                "原始 legged_gym 默认策略是 `ActorCritic` MLP：actor hidden dims `[512, 256, 128]`，critic hidden dims "
                "`[512, 256, 128]`，activation `elu`，连续动作输出作为关节 PD target offset 或 torque/control command。"
                "观测一般包含 base linear/angular velocity、projected gravity、commands、关节位置/速度、上一动作，"
                "粗糙地形任务还会加入 height measurements。动作通过 `action_scale` 缩放后叠加 default joint angle，"
                "低层控制通常是 P control，`decimation=4` 表示策略频率低于物理仿真频率。"
                "\n\n"
                "Unitree G1 配置把架构缩成更适合 12DoF 下肢的 recurrent policy：`num_observations=47`、"
                "`num_privileged_obs=50`、`num_actions=12`，URDF 为 `g1_12dof.urdf`，foot name 是 `ankle_roll`。"
                "PPO policy class 为 `ActorCriticRecurrent`，actor/critic hidden dims 都是 `[32]`，activation `elu`，"
                "`rnn_type='lstm'`、`rnn_hidden_size=64`、`rnn_num_layers=1`，`init_noise_std=0.8`。这类结构的优势是"
                "小、快、可部署；劣势是技能覆盖主要由 reward 和命令空间决定，不具备 SONIC/HoloMotion 那种大规模"
                "运动先验 token 化表达。"
            ),
            references=legged_gym_references,
            figures=[legged_gym_architecture_figure],
        ),
        ModelTraining(
            content=(
                "legged_gym 训练以 PPO 直接最大化 locomotion reward。原始默认配置为 `num_envs=4096`、episode 20s、"
                "Isaac Gym GPU parallel simulation；runner 使用 `OnPolicyRunner`，算法类 `PPO`，每轮 "
                "`num_steps_per_env=24`，默认 `max_iterations=1500`。PPO 超参包括 `value_loss_coef=1.0`、"
                "`use_clipped_value_loss=True`、`clip_param=0.2`、`entropy_coef=0.01`、`num_learning_epochs=5`、"
                "`num_mini_batches=4`、`learning_rate=1e-3`、adaptive schedule、`gamma=0.99`、`lam=0.95`、"
                "`desired_kl=0.01`、`max_grad_norm=1.0`。"
                "\n\n"
                "Unitree RL Gym 的训练命令是 `python legged_gym/scripts/train.py --task=xxx`，task 可为 `go2`、`g1`、"
                "`h1`、`h1_2`。训练输出保存到 `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`；"
                "Play 阶段会导出 actor network 到 `logs/{experiment_name}/exported/policies`，标准 MLP 为 "
                "`policy_1.pt`，RNN 为 `policy_lstm_1.pt`，之后再走 Sim2Sim 和 Sim2Real。"
                "\n\n"
                "G1 配置的训练环境强化了 sim-to-real 鲁棒性：随机摩擦 `friction_range=[0.1, 1.25]`，随机 base mass "
                "`added_mass_range=[-1, 3]`，每 5s push robot，最大 XY push velocity 1.5；控制为 P control，hip yaw/"
                "roll/pitch stiffness 100、knee 150、ankle 40，damping 分别约 2/4/2，`action_scale=0.25`，"
                "`decimation=4`。奖励由 tracking linear/angular velocity、orientation、base height、dof acceleration/"
                "velocity、action rate、dof limits、alive、contact 等项组成。G1 PPO runner 把 `max_iterations` 提到 "
                "10000，实验名 `g1`。这套训练非常适合作为科研基线：资源需求远小于 SONIC/HoloMotion，超参透明，"
                "但需要为每个技能和机器人仔细调 reward、domain randomization 与课程。"
            ),
            references=legged_gym_references,
            figures=[legged_gym_training_figure],
        ),
    ],
)


print(f"Academic report completed with {len(report.content)} slides.")
