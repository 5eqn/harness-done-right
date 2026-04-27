from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

task_registry = File(
    path="mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py"
)
g1_env_cfg = File(
    path="mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py"
)
tracking_rl_cfg = File(
    path="mjlab/src/mjlab/tasks/tracking/config/g1/rl_cfg.py"
)
rl_config = File(path="mjlab/src/mjlab/rl/config.py")
train_py = File(path="mjlab/src/mjlab/scripts/train.py")
mjlab_runner = File(path="mjlab/src/mjlab/rl/runner.py")
tracking_runner = File(
    path="mjlab/src/mjlab/tasks/tracking/rl/runner.py"
)
pyproject = File(path="mjlab/pyproject.toml")

loss_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=task_registry,
            content="""
Scope anchor: this proof is for exactly `Mjlab-Tracking-Flat-Unitree-G1`. The registered task uses `unitree_g1_flat_tracking_env_cfg()` without disabling state estimation, binds `unitree_g1_tracking_ppo_runner_cfg()`, and selects `MotionTrackingOnPolicyRunner`. The adjacent no-state-estimation task has a different task id and is not part of this proof.""",
            text="""
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)""",
        ),
        Highlight(
            file=g1_env_cfg,
            content="""
The G1 flat tracking factory defaults to state estimation enabled. Since the scoped task above calls it without `has_state_estimation=False`, the loss/training-objective evidence applies to the state-estimation version of Unitree G1 tracking.""",
            text="""
def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:""",
        ),
        Highlight(
            file=tracking_rl_cfg,
            content="""
The task-specific runner config supplies the PPO objective/training hyperparameters requested here: value loss coefficient 1.0, clipped value loss enabled, PPO ratio clip 0.2, entropy coefficient 0.005, 5 epochs, 4 minibatches, learning rate 1e-3, adaptive schedule with desired KL 0.01, gamma 0.99, lambda 0.95, max gradient norm 1.0, 24 rollout steps per environment, and 30,000 learning iterations.""",
            text="""
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,""",
        ),
        Highlight(
            file=rl_config,
            content="""
MJLab's repo defines `RslRlPpoAlgorithmCfg` as a PPO algorithm configuration object with exactly the knobs used above. This proves MJLab configures clipped value loss, PPO clipping, entropy regularization, rollout-update epochs/minibatches, adaptive KL scheduling, discounting/GAE, gradient clipping, and the RSL-RL algorithm class name; it does not reimplement the PPO loss formula in this repo.""",
            text="""
@dataclass
class RslRlPpoAlgorithmCfg:
  \"\"\"Config for the PPO algorithm.\"\"\"

  num_learning_epochs: int = 5
  \"\"\"The number of learning epochs per update.\"\"\"
  num_mini_batches: int = 4
  \"\"\"The number of mini-batches per update.
  mini batch size = num_envs * num_steps / num_mini_batches
  \"\"\"
  learning_rate: float = 1e-3
  \"\"\"The learning rate.\"\"\"
  schedule: Literal[\"adaptive\", \"fixed\"] = \"adaptive\"
  \"\"\"The learning rate schedule.\"\"\"
  gamma: float = 0.99
  \"\"\"The discount factor.\"\"\"
  lam: float = 0.95
  \"\"\"The lambda parameter for Generalized Advantage Estimation (GAE).\"\"\"
  entropy_coef: float = 0.005
  \"\"\"The coefficient for the entropy loss.\"\"\"
  desired_kl: float = 0.01
  \"\"\"The desired KL divergence between the new and old policies.\"\"\"
  max_grad_norm: float = 1.0
  \"\"\"The maximum gradient norm for the policy.\"\"\"
  value_loss_coef: float = 1.0
  \"\"\"The coefficient for the value loss.\"\"\"
  use_clipped_value_loss: bool = True
  \"\"\"Whether to use clipped value loss.\"\"\"
  clip_param: float = 0.2
  \"\"\"The clipping parameter for the policy.\"\"\"""",
        ),
        Highlight(
            file=rl_config,
            content="""
The runner config embeds that algorithm config inside an on-policy runner config whose class name is `OnPolicyRunner`. Thus the task config is shaped for RSL-RL's on-policy PPO runner path, not for a separate trainer or custom MJLab loss.""",
            text="""
@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  \"\"\"The runner class name. Default is OnPolicyRunner.\"\"\"
  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      }
    )
  )
  \"\"\"The actor configuration.\"\"\"
  critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  \"\"\"The critic configuration.\"\"\"
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  \"\"\"The algorithm configuration.\"\"\"""",
        ),
        Highlight(
            file=rl_config,
            content="""
The algorithm config's `class_name` is `PPO`, which is the name resolved by the external RSL-RL runner. This is the repo-level proof that the configured loss/training objective is PPO.""",
            text="""
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  \"\"\"The optimizer to use.\"\"\"
  share_cnn_encoders: bool = False
  \"\"\"Share CNN encoders between actor and critic.\"\"\"
  class_name: str = "PPO"
  \"\"\"Algorithm class name resolved by RSL-RL.\"\"\"""",
        ),
        Highlight(
            file=pyproject,
            content="""
The detailed PPO loss implementation is outside MJLab: the project depends on the external package `rsl-rl-lib==5.0.1`. The MJLab repository proves correct configuration and dispatch into RSL-RL, but the line-by-line clipped surrogate/value-loss math lives in that dependency rather than in MJLab source.""",
            text="""
  "tensordict",
  "rsl-rl-lib==5.0.1",
  "tensorboard>=2.20.0",""",
        ),
        Highlight(
            file=train_py,
            content="""
Training converts the selected `cfg.agent` into `agent_cfg`, loads the task's registered runner class, adds the tracking registry name, writes the exact agent config before runner mutation, and then passes `agent_cfg` into the runner constructor. This is the concrete handoff from MJLab task config to the RSL-RL runner stack.""",
            text="""
  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name

  # Write config files before runner creation, since the runner mutates agent_cfg
  # in-place (e.g., injecting non-serializable objects).
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)""",
        ),
        Highlight(
            file=mjlab_runner,
            content="""
`MjlabOnPolicyRunner` imports `OnPolicyRunner` from external `rsl_rl.runners` and delegates construction with `super().__init__(env, train_cfg, log_dir, device)`. Therefore the `agent_cfg` dictionary passed by `train.py` becomes the `train_cfg` consumed by RSL-RL's `OnPolicyRunner`, after only MJLab's local cleanup of `None` optional model fields.""",
            text="""
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabOnPolicyRunner(OnPolicyRunner):
  \"\"\"Base runner that persists environment state across checkpoints.\"\"\"

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Strip None-valued optional configs so MLPModel doesn't receive them.
    for key in ("actor", "critic"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)""",
        ),
        Highlight(
            file=tracking_runner,
            content="""
The tracking-specific runner does not replace PPO or define a custom loss. It subclasses `MjlabOnPolicyRunner`, forwards the same `train_cfg` to `super().__init__`, and only stores `registry_name` for tracking-specific export/context behavior.""",
            text="""
class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name""",
        ),
    ]
)
