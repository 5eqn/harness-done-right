from academic_report import AcademicReport
from academic_report import Insight
from academic_report import ModelArchitecture
from academic_report import ModelTraining
from academic_report import Reference
from academic_report import ReferenceType

topic = "科研导向的人形机器人底层运控调研"
author = "github@5eqn"


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

switch_references = [
    Reference(
        type=ReferenceType.Article,
        title="Switch: Learning Agile Skills Switching for Humanoid Robots",
        link="https://arxiv.org/html/2604.14834v1",
    ),
]

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

safety_references = [
    Reference(
        type=ReferenceType.Article,
        title="Real-time motion control of robotic manipulators for safe human-robot coexistence",
        link="https://www.sciencedirect.com/science/article/abs/pii/S0736584521001022",
    ),
]


legged_gym_architecture = ModelArchitecture(
    content=(
        "legged_gym / Unitree RL Gym 是最小可复现的底层运控基线：Isaac Gym 并行仿真环境 + "
        "RSL-RL PPO actor-critic。它不向上层开放丰富接口；上层给定的是速度命令等低维 command，"
        "策略直接把 robot observation 映射到关节 PD target offset 或控制命令。原始 legged_gym "
        "默认 actor/critic 都是 `[512, 256, 128]` 的 MLP，activation 为 `elu`；actor 看普通观测，"
        "critic 可以看 privileged observations。"
        "\n\n"
        "Unitree G1 配置更轻，面向 12DoF 下肢：`num_observations=47`、`num_privileged_obs=50`、"
        "`num_actions=12`，policy class 是 `ActorCriticRecurrent`，actor/critic hidden dims 都是 "
        "`[32]`，`rnn_type='lstm'`、`rnn_hidden_size=64`、`rnn_num_layers=1`。这个架构的价值是透明、"
        "便宜、可复现；限制是技能基本被 reward 和 command space 锁死，不是一个可组合的运动接口。"
    ),
    references=legged_gym_references,
    figures=[],
)

legged_gym_training = ModelTraining(
    content=(
        "legged_gym 用 PPO 在 Isaac Gym GPU 并行仿真中直接优化 locomotion reward。原始默认配置为 "
        "`num_envs=4096`、episode 20s、`num_steps_per_env=24`、`max_iterations=1500`。PPO 超参包括 "
        "`clip_param=0.2`、`gamma=0.99`、`lam=0.95`、`learning_rate=1e-3`、`entropy_coef=0.01`、"
        "`num_learning_epochs=5`、`num_mini_batches=4`。"
        "\n\n"
        "Unitree G1 训练命令是 `python legged_gym/scripts/train.py --task=g1`，训练后导出 MLP 或 "
        "LSTM actor，再进入 Sim2Sim / Sim2Real。G1 配置通过 friction randomization、base mass "
        "randomization、push robot、action rate penalty、joint limit penalty 等机制做 sim-to-real "
        "鲁棒性。它是研究新接口之前的必要基线：先证明一个简单闭环 PPO policy 能稳定工作，再讨论应该"
        "向上层开放什么控制接口。"
    ),
    references=legged_gym_references,
    figures=[],
)

interface_insight = Insight(
    content=(
        "从 legged_gym 往后，问题不再只是“怎样训练一个会走路的 policy”，而是“底层控制器应该向"
        "上层智能开放什么 interface”。如果完全不开放接口，系统稳定但能力封闭；如果直接开放关节角、"
        "速度或力矩，上层拥有最大自由度，但必须承担 actuator saturation、joint range、speed limits、"
        "obstacle avoidance 和人机共处安全约束。受 constrained control / reference governor 工作启发，"
        "直接暴露低层执行参考并不天然安全，接口本身需要约束、过滤和恢复机制。"
        "\n\n"
        "接下来四条路线可以放在同一张谱系上：legged_gym 是封闭低维 command；SWITCH 开放“选择具体"
        "技能”的接口；HoloMotion 开放 motion-tracking-ish 的全身参考接口；SONIC 进一步把 motion "
        "tracking 与 VR、SMPL、VLA 等抽象意图接到统一 token space。"
    ),
    references=[*legged_gym_references, *safety_references],
    figures=[],
)

switch_architecture = ModelArchitecture(
    content=(
        "SWITCH 解决的是“在多个具体技能之间如何安全切换”。它不是让上层直接输出关节角，也不是给一个"
        "任意连续 motion target；它开放的接口更离散：上层选择目标 skill label，系统负责找到一条可执行"
        "过渡路径。架构由三部分组成：Skill Graph、whole-body tracking policy、online skill scheduler。"
        "\n\n"
        "Skill Graph 把 motion data frame 作为节点，把同一技能内的相邻帧作为边，并根据局部姿态/速度"
        "相似度加入跨技能边；当跨技能边跨度过大时，系统插入 buffer nodes，让过渡更平滑。部署时，online "
        "scheduler 在用户切换技能或 tracking error 超阈值时做图搜索，给低层 tracking policy 提供实时"
        "参考。低层 policy 的动作是 desired joint positions，由 PD controller 跟踪。"
    ),
    references=switch_references,
    figures=[],
)

switch_training = ModelTraining(
    content=(
        "SWITCH 在仿真中用 PPO 训练 whole-body tracking policy。训练数据不是孤立技能片段，而是 Skill "
        "Graph 增广后的多技能轨迹，包括跨技能连接和 buffer nodes。训练机制包括 Reference State "
        "Initialization、buffer-aware imitation learning、统一 imitation reward，以及针对 foot-ground "
        "contact 的额外奖励。"
        "\n\n"
        "课程学习分三层：先以 90% 单技能轨迹和 10% 增广轨迹学习基础运动，再逐步提高 skill augmentation "
        "比例；同时逐步引入 regularization / penalty；最后收紧 termination threshold。实验在 Unitree G1 "
        "29DoF 上部署，Jetson Orin NX 运行 learned policy。论文报告 SWITCH 在 easy/medium/hard skill "
        "switching 中显著提高 Skill Switching Success Rate，并能在 tracking error 或扰动后通过 scheduler "
        "进入恢复路径。"
    ),
    references=switch_references,
    figures=[],
)

holo_architecture = ModelArchitecture(
    content=(
        "HoloMotion v1.2 把接口推进到 motion-tracking-ish：上层可以给出当前和未来参考运动，低层策略"
        "负责把它变成 Unitree G1 的全身动作。系统级架构包括动作数据整理、人体到机器人重定向、IsaacLab "
        "分布式训练、评估导出、ROS2/ONNX 实机部署。v1.2 发布了 motion tracking 和 velocity tracking "
        "两个预训练 ONNX 模型。"
        "\n\n"
        "核心 actor 是 `ReferenceRoutedGroupedMoETransformerPolicy`。输入包含当前参考状态、机器人自身"
        "状态、上一动作和未来参考片段；actor 用 2048 维观测嵌入和 2048 维 router 嵌入构造 token，再进入 "
        "3 层 transformer/MoE：`d_model=512`、`n_heads=8`、`num_fine_experts=16`、`top_k=2`。critic 是 "
        "4 层 2048 hidden dims MLP，使用更强的 privileged observations。"
    ),
    references=holo_references,
    figures=[],
)

holo_training = ModelTraining(
    content=(
        "HoloMotion v1.2 的训练路径是强化学习加运动先验：先把大规模人类动作数据转成 AMASS-compatible "
        "SMPL，再通过 GMR 重定向成机器人 HDF5，随后在 IsaacLab 中训练 motion tracking 或 velocity "
        "tracking 策略。motion tracking v1.2 配置给出 `num_envs=8192`、`num_processes=64`、"
        "`num_steps_per_env=32`、`num_learning_iterations=20001`。"
        "\n\n"
        "PPO 超参包括 `clip_param=0.2`、`gamma=0.99`、`lam=0.95`、`entropy_coef=0.005`、actor LR "
        "`3e-5`、critic LR `5e-5`。数据采样使用 AMASS / LAFAN1 / pico weighted bin。鲁棒性来自 rough "
        "terrain、执行器 stiffness/damping randomization、观测噪声和辅助状态预测。导出后生成 ONNX，并"
        "通过 MuJoCo / ROS2 评估部署。"
    ),
    references=holo_references,
    figures=[],
)

sonic_architecture = ModelArchitecture(
    content=(
        "SONIC / GR00T-WBC 进一步扩展接口：不只跟踪一种 motion reference，而是把多种动作来源编码到"
        "同一个 universal token space。它面向 Unitree G1 29DoF，在 GR00T 系统里扮演 fast System 1："
        "上层 VLA、VR、视频人体动作估计、文本/音乐生成动作或 planner 给出意图，SONIC 把它们变成全身"
        "动作。"
        "\n\n"
        "核心结构是多个 encoder + 共享 token + 单一 decoder。encoder 接收 G1 joint trajectories、VR "
        "teleop 三点 tracking targets、SMPL 人体关节位置，以及可选 SOMA/BVH skeleton positions；每个 "
        "encoder 投影到共享 latent token space，并通过 FSQ 规整 token；decoder 从共享 token 产生 G1 "
        "关节动作。"
    ),
    references=sonic_references,
    figures=[],
)

sonic_training = ModelTraining(
    content=(
        "SONIC 的训练目标是把 motion tracking supersize。论文摘要报告模型从 1.2M 参数扩到 42M 参数，"
        "数据超过 100M frames / 700 小时高质量 motion data，训练 compute 约 9k GPU hours；实验段落还"
        "报告大规模 tracker 在 128 GPUs 训练 3 天，约 32k GPU hours。"
        "\n\n"
        "公开训练入口是 `gear_sonic/train_agent_trl.py`，默认配置 `+exp=manager/universal_token/all_modes/"
        "sonic_release`、`num_envs=4096`。Bones-SEED CSV 先从 120fps 转 30fps，再过滤 G1 不可执行动作。"
        "`sonic_release` 使用 G1、teleop、SMPL 三个 encoder；`sonic_bones_seed` 增加 SOMA 第四 encoder。"
        "训练算法是 Isaac Lab simulation 中的 PPO 加 auxiliary losses，官方建议 64+ GPUs 获得合理收敛时间。"
    ),
    references=sonic_references,
    figures=[],
)

final_insight = Insight(
    content=(
        "四个系统形成一条清晰的接口演化线。legged_gym 证明最小闭环 policy 可以工作，但能力封闭；SWITCH "
        "把接口开放到“选择技能”，用图搜索和 scheduler 管住技能切换风险；HoloMotion 把接口开放到"
        "全身参考运动，让底层策略成为运动跟踪基础模型；SONIC 把 motion tracking 接到更抽象的多模态意图，"
        "让低层控制成为上层智能的动作接口。"
        "\n\n"
        "下一步研究的关键不是简单追求更大的 policy，而是设计更好的接口边界：足够开放，让 VLA、planner、"
        "人类遥操作或生成模型能表达意图；足够受约束，让底层控制器可以在关节限制、接触事件、扰动和失败恢复"
        "中保持安全。"
    ),
    references=[
        *legged_gym_references,
        *safety_references,
        *switch_references,
        *holo_references,
        *sonic_references,
    ],
    figures=[],
)

report = AcademicReport(
    topic=topic,
    author=author,
    content=[
        legged_gym_architecture,
        legged_gym_training,
        interface_insight,
        switch_architecture,
        switch_training,
        holo_architecture,
        holo_training,
        sonic_architecture,
        sonic_training,
        final_insight,
    ],
)


print(f"Academic report completed with {len(report.content)} slides.")
