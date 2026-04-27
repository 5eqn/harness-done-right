from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode


task_registration = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py")
tracking_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py")
tracking_rl_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/rl_cfg.py")
rl_config = File(path="mjlab/src/mjlab/rl/config.py")
train_script = File(path="mjlab/src/mjlab/scripts/train.py")
base_runner = File(path="mjlab/src/mjlab/rl/runner.py")
tracking_runner = File(path="mjlab/src/mjlab/tasks/tracking/rl/runner.py")
vecenv_wrapper = File(path="mjlab/src/mjlab/rl/vecenv_wrapper.py")


model_correct = ProofFromCode(
  highlights=[
    Highlight(
      file=task_registration,
      content="""
Scope anchor: the analyzed model is for exactly `Mjlab-Tracking-Flat-Unitree-G1`. That task registers the default G1 flat tracking environment, the G1 tracking PPO config, and `MotionTrackingOnPolicyRunner`; the no-state-estimation sibling is a separate task id and is not the requested target.""",
      text="""
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)""",
    ),
    Highlight(
      file=tracking_env_cfg,
      content="""
The scoped task is the state-estimation variant because `has_state_estimation` defaults to `True`; only explicit callers of the separate no-state-estimation task remove state-estimation actor inputs.""",
      text="""
def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:""",
    ),
    Highlight(
      file=tracking_rl_cfg,
      content="""
The actor model is configured as an RSL-RL MLP-style model with hidden layers `(512, 256, 128)`, ELU activation, observation normalization, and a stochastic Gaussian action distribution initialized with scalar standard deviation 1.0.""",
      text="""
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),""",
    ),
    Highlight(
      file=tracking_rl_cfg,
      content="""
The critic is a separate RSL-RL model with the same feed-forward hidden dimensions and ELU activation, also using observation normalization, but with no distribution config because it predicts values rather than actions.""",
      text="""
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),""",
    ),
    Highlight(
      file=rl_config,
      content="""
`RslRlModelCfg` is the local schema that makes those actor/critic settings meaningful: it exposes hidden dimensions, activation, observation normalization, optional distribution config, optional CNN/RNN fields, and defaults `class_name` to `MLPModel`. Therefore the configured model type is an MLP unless the config explicitly overrides it, which the G1 tracking config does not.""",
      text="""
@dataclass
class RslRlModelCfg:
  \"\"\"Config for a single neural network model (Actor or Critic).\"\"\"

  hidden_dims: Tuple[int, ...] = (128, 128, 128)
  \"\"\"The hidden dimensions of the network.\"\"\"
  activation: str = "elu"
  \"\"\"The activation function.\"\"\"
  obs_normalization: bool = False
  \"\"\"Whether to normalize the observations. Default is False.\"\"\"
  cnn_cfg: dict[str, Any] | None = None
  \"\"\"CNN encoder config. When set, class_name should be "CNNModel".""",
    ),
    Highlight(
      file=rl_config,
      content="""
The remainder of `RslRlModelCfg` confirms the distribution config is optional actor-side stochastic-output plumbing, recurrent settings are opt-in, and `class_name="MLPModel"` is the default model class consumed by RSL-RL.""",
      text="""
  distribution_cfg: dict[str, Any] | None = None
  \"\"\"Distribution config dict passed to rsl_rl. Example::

    {"class_name": "GaussianDistribution",
     "init_std": 1.0, "std_type": "scalar"}

  ``None`` means deterministic output (use for critic).
  \"\"\"
  rnn_type: str | None = None""",
    ),
    Highlight(
      file=rl_config,
      content="""
This final field is the decisive model-type default: absent an override in `rl_cfg.py`, actor and critic are RSL-RL `MLPModel` instances, not transformers or recurrent networks.""",
      text="""
  class_name: str = "MLPModel"
  \"\"\"Model class name resolved by RSL-RL (MLPModel, CNNModel, or RNNModel).\"\"\"""",
    ),
    Highlight(
      file=rl_config,
      content="""
The runner config groups the actor, critic, and PPO algorithm into the on-policy runner payload. It also declares `obs_groups`, so RSL-RL receives the `actor` observation group for the actor and the `critic` observation group for the critic.""",
      text="""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)},
  )
  save_interval: int = 50""",
    ),
    Highlight(
      file=rl_config,
      content="""
`RslRlOnPolicyRunnerCfg` is the concrete config object returned by the G1 tracking RL factory: it contains `actor`, `critic`, and `algorithm` dataclasses, with `class_name="OnPolicyRunner"` for RSL-RL compatibility.""",
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
  )""",
    ),
    Highlight(
      file=tracking_rl_cfg,
      content="""
The model is trained under PPO with explicit loss/objective hyperparameters: clipped value loss, PPO clip range 0.2, entropy coefficient 0.005, five epochs, four minibatches, adaptive LR schedule, GAE `(gamma=0.99, lam=0.95)`, desired KL 0.01, and gradient norm clipping at 1.0.""",
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
    ),""",
    ),
    Highlight(
      file=tracking_rl_cfg,
      content="""
The rollout/training schedule is also part of the model's training contract: 24 steps per environment per update, checkpoints every 500 iterations, and a long 30,000-iteration training budget under experiment name `g1_tracking`.""",
      text="""
    experiment_name="g1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,""",
    ),
    Highlight(
      file=train_script,
      content="""
The training script converts the dataclass config to dictionaries immediately before runner construction. This is the exact handoff point where the actor/critic/algorithm model config is passed to the selected runner.""",
      text="""
  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner""",
    ),
    Highlight(
      file=train_script,
      content="""
For tracking tasks, the registry-provided runner receives the converted `agent_cfg`; therefore the scoped task trains the registered `MotionTrackingOnPolicyRunner` with the G1 actor/critic/PPO config above.""",
      text="""
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))""",
    ),
    Highlight(
      file=base_runner,
      content="""
`MjlabOnPolicyRunner` deliberately strips only `None` optional fields before delegating to RSL-RL. Since the G1 actor has a real `distribution_cfg` and both actor and critic keep their `hidden_dims`, `activation`, `obs_normalization`, and default `class_name`, those model-defining fields survive the handoff.""",
      text="""
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
            train_cfg[key].pop(opt, None)""",
    ),
    Highlight(
      file=base_runner,
      content="""
After cleaning optional fields, MJLab delegates to `rsl_rl.runners.OnPolicyRunner`; MJLab does not replace the model with a hidden local network implementation.""",
      text="""
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)""",
    ),
    Highlight(
      file=tracking_runner,
      content="""
The tracking-specific runner is also thin at construction time: it calls the MJLab base runner constructor and only stores `registry_name`. Thus model creation still follows the RSL-RL on-policy runner path configured above.""",
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
    Highlight(
      file=vecenv_wrapper,
      content="""
The action output size used by RSL-RL comes from `env.action_manager.total_action_dim`; for this task the action proof grounds that manager to the G1 joint-position action term. This connects the stochastic actor head to the actual environment action dimension.""",
      text="""
    self.num_envs = self.unwrapped.num_envs
    self.device = torch.device(self.unwrapped.device)
    self.max_episode_length = self.unwrapped.max_episode_length
    self.num_actions = self.unwrapped.action_manager.total_action_dim
    self._modify_action_space()""",
    ),
  ]
)
