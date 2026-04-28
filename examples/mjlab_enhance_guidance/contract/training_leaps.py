from enhance_guidance import AtomicLeap, File, Highlight


holomotion_train = File(path="HoloMotion/holomotion/src/training/train.py")
holomotion_algo_base = File(path="HoloMotion/holomotion/src/algo/algo_base.py")
holomotion_ppo_tf = File(path="HoloMotion/holomotion/src/algo/ppo_tf.py")
mjlab_train = File(path="mjlab/src/mjlab/scripts/train.py")
mjlab_rl_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/rl_cfg.py")
mjlab_rl_config = File(path="mjlab/src/mjlab/rl/config.py")


training_leaps: list[AtomicLeap] = [
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_train,
            content=(
                "HoloMotion resolves the algorithm class from config and runs "
                "the selected implementation through a uniform load/learn API."
            ),
            text="""
    log_dir = config.experiment_save_dir
    headless = config.headless
    algo_class = get_class(config.algo._target_)
    algo = algo_class(
        env_config=config.env,
        config=config.algo.config,
        log_dir=log_dir,
        headless=headless,
    )

    algo.load(config.checkpoint)
    algo.learn()""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_train,
            content=(
                "MJLab currently resolves only the runner class from the task "
                "registry and adds tracking-specific runner kwargs."
            ),
            text="""
  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name""",
        ),
        change_direction=(
            "Move MJLab training from task-registered runner selection to "
            "config-declared algorithm dispatch. The tracking task should keep "
            "its environment registration, but the train path should resolve a "
            "generalized tracker algorithm class from the agent config, construct "
            "it with env config, algorithm config, log directory, and device "
            "metadata, then call load/learn through that algorithm interface."
        ),
        change_reason=(
            "HoloMotion can swap plain PPO, PPOTF, and TF-MoE variants without "
            "rewiring task registration. MJLab currently makes the task registry "
            "choose a runner class, which couples generalized tracking behavior "
            "to one G1 task. Dispatching from config gives engineers a clean "
            "place to add PPOTF/transformer training while preserving the "
            "existing MJLab task and environment system."
        ),
        changed_code=r'''
from mjlab.utils.imports import get_class


@dataclass
class RslRlBaseRunnerCfg:
  algo_target: str = "mjlab.rl.algorithms.PPO"
  algo_config: dict[str, object] = field(default_factory=dict)


def build_algorithm(cfg: TrainConfig, env: RslRlVecEnvWrapper, log_dir: Path, device: str):
  algo_cls = get_class(cfg.agent.algo_target)
  return algo_cls(
    env=env,
    env_cfg=cfg.env,
    config=cfg.agent.algo_config,
    log_dir=str(log_dir),
    device=device,
  )


algo = build_algorithm(cfg, env, log_dir, device)
if resume_path is not None:
  algo.load(str(resume_path))
algo.learn(num_learning_iterations=cfg.agent.max_iterations)
''',
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_ppo_tf,
            content=(
                "PPOTF is the HoloMotion transformer-policy PPO algorithm, and "
                "its actor wrapper dispatch chooses the appropriate transformer "
                "or reference-routed MoE actor."
            ),
            text="""
class PPOTF(PPO):
    \"\"\"Transformer-policy PPO with TensorDict rollout and sequence update.\"\"\"

    @staticmethod
    def _select_actor_wrapper_cls(actor_cfg: dict):
        actor_type = str(actor_cfg.get("type", ""))
        use_future_cross_attn = bool(
            actor_cfg.get("use_future_cross_attn", False)
        )""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_rl_cfg,
            content=(
                "The MJLab G1 tracking task configures standard PPO scalar "
                "hyperparameters rather than a sequence-aware PPOTF algorithm."
            ),
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
        change_direction=(
            "Add a PPOTF-style algorithm option for motion tracking instead of "
            "using only scalar RSL-RL PPO settings. The new option should train "
            "sequence-capable transformer actors, keep rollout data as structured "
            "actor/critic/reference tensors, and run minibatches over complete "
            "environment sequences rather than flattening all time steps as "
            "independent samples."
        ),
        change_reason=(
            "Generalized tracking depends on temporal reference context and "
            "routing decisions. HoloMotion makes that explicit with PPOTF, while "
            "MJLab's current G1 tracking config only names generic PPO losses and "
            "does not encode sequence update semantics. Without a PPOTF training "
            "path, a transformer/MoE actor added elsewhere would still be trained "
            "like a feed-forward policy."
        ),
        changed_code=r'''
@dataclass
class MjlabPpoTfAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "PPOTF"
  num_learning_epochs: int = 3
  num_mini_batches: int = 24
  actor_learning_rate: float = 3.0e-5
  critic_learning_rate: float = 5.0e-5
  use_kv_cache: bool = True
  sequence_batching: bool = True
  aux_state_pred: dict[str, object] = field(
    default_factory=lambda: {
      "enabled": True,
      "w_base_lin_vel": 1.0e-2,
      "w_keybody_contact": 1.0e-2,
      "w_ref_keybody_rel_pos": 1.0e-1,
      "w_robot_keybody_rel_pos": 1.0e-1,
    }
  )


def update(self) -> dict[str, float]:
  generator = self.sequence_batches(
    num_mini_batches=self.cfg.num_mini_batches,
    num_learning_epochs=self.cfg.num_learning_epochs,
  )
  for batch in generator:
    actor_loss, critic_loss, aux_loss = self.compute_ppotf_losses(batch)
    self.step_actor_critic(actor_loss + aux_loss, critic_loss)
  return self.last_update_metrics
''',
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_algo_base,
            content=(
                "HoloMotion initializes accelerator, logging, environment, "
                "configs, storage, algorithm components, and models in a fixed "
                "algorithm lifecycle."
            ),
            text="""
        self._setup_accelerator()
        self.algo_logger = AlgoLogger(
            self.accelerator,
            self.log_dir,
            is_main_process=self.is_main_process,
        )
        self._setup_environment()
        self._setup_configs()
        self._setup_seeding()
        self._setup_data_buffers()
        self._setup_algo_components()
        self._setup_models_and_optimizer()""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_train,
            content=(
                "MJLab's train script performs rank, device, EGL, and seed "
                "selection inline before environment construction."
            ),
            text="""
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank""",
        ),
        change_direction=(
            "Introduce an algorithm initialization phase that prepares "
            "distributed state before creating the environment or models. MJLab "
            "can still set MuJoCo EGL per local rank, but the generalized tracker "
            "should centralize process id, world size, main-process logging, "
            "device selection, seeding, storage allocation, and optimizer/model "
            "preparation in the algorithm lifecycle."
        ),
        change_reason=(
            "MJLab currently performs rank/device handling in the train script "
            "and delegates distributed details to the runner stack. HoloMotion's "
            "ordering makes rank-aware logging, distributed tensors, device "
            "placement, and later optimizer preparation deterministic before "
            "rollouts begin. That structure is safer for TF-MoE training, where "
            "sequence buffers, routing metrics, and optimizer state must agree "
            "across workers."
        ),
        changed_code=r'''
class MjlabGeneralizedTrackerAlgorithm:
  def __init__(self, env_cfg, config, log_dir: str, headless: bool = True) -> None:
    self.env_cfg = env_cfg
    self.config = config
    self.log_dir = log_dir
    self.headless = headless

    self.setup_distributed()
    self.setup_logger()
    self.setup_environment()
    self.setup_configs()
    self.setup_seeding()
    self.setup_rollout_storage()
    self.setup_algorithm_components()
    self.setup_models_and_optimizers()

  def setup_distributed(self) -> None:
    self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    self.process_id = int(os.environ.get("RANK", "0"))
    self.num_processes = int(os.environ.get("WORLD_SIZE", "1"))
    self.main_process = self.process_id == 0
    self.device = torch.device(
      f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    if self.device.type == "cuda":
      torch.cuda.set_device(self.local_rank)
      os.environ["MUJOCO_EGL_DEVICE_ID"] = str(self.local_rank)
''',
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_ppo_tf,
            content=(
                "HoloMotion PPOTF builds a critic optimizer separately and "
                "prepares actor, critic, and both optimizers together."
            ),
            text="""
        self.critic_optimizer = optimizer_class(
            self.critic.parameters(),
            lr=self.critic_learning_rate,
            betas=(self.critic_beta1, self.critic_beta2),
            **optimizer_kwargs,
        )

        (
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.critic_optimizer,
        ) = self.accelerator.prepare(
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.critic_optimizer,
        )""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_rl_config,
            content=(
                "MJLab's shared PPO config exposes one learning rate for the "
                "algorithm rather than independent actor and critic rates."
            ),
            text="""
  learning_rate: float = 1e-3
  \"\"\"The learning rate.\"\"\"
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  \"\"\"The learning rate schedule.\"\"\"
  gamma: float = 0.99
  \"\"\"The discount factor.\"\"\"""",
        ),
        change_direction=(
            "Split MJLab's single PPO learning-rate field into actor and critic "
            "optimizer settings for generalized tracking. The PPOTF algorithm "
            "should construct actor and critic optimizers separately, optionally "
            "apply actor-side AdamW decay grouping, and prepare both models and "
            "optimizers together through the distributed preparation layer."
        ),
        change_reason=(
            "HoloMotion tunes transformer policy and critic optimization "
            "independently, which is important when the actor carries KV cache, "
            "routing, auxiliary losses, and stochastic sigma parameters while "
            "the critic remains a value estimator. MJLab's single learning rate "
            "is adequate for basic PPO but too blunt for stable generalized "
            "tracker training."
        ),
        changed_code=r'''
@dataclass
class MjlabPpoTfAlgorithmCfg(RslRlPpoAlgorithmCfg):
  actor_learning_rate: float = 3.0e-5
  critic_learning_rate: float = 5.0e-5
  actor_weight_decay: float = 0.01


def setup_models_and_optimizers(self) -> None:
  self.actor = build_transformer_actor(self.actor_cfg).to(self.device)
  self.critic = build_critic(self.critic_cfg).to(self.device)

  decay_params, non_decay_params = split_decay_params(self.actor)
  self.actor_optimizer = torch.optim.AdamW(
    [
      {"params": decay_params, "weight_decay": self.cfg.actor_weight_decay},
      {"params": non_decay_params, "weight_decay": 0.0},
    ],
    lr=self.cfg.actor_learning_rate,
  )
  self.critic_optimizer = torch.optim.AdamW(
    self.critic.parameters(),
    lr=self.cfg.critic_learning_rate,
  )
  self.actor, self.critic, self.actor_optimizer, self.critic_optimizer = (
    self.distributed.prepare(
      self.actor,
      self.critic,
      self.actor_optimizer,
      self.critic_optimizer,
    )
  )
''',
    ),
]
