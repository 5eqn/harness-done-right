from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

task_registration = File(
  path="mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py"
)
g1_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py")
tracking_rl_cfg = File(
  path="mjlab/src/mjlab/tasks/tracking/config/g1/rl_cfg.py"
)
rl_config = File(path="mjlab/src/mjlab/rl/config.py")
train_script = File(path="mjlab/src/mjlab/scripts/train.py")
os_utils = File(path="mjlab/src/mjlab/utils/os.py")
on_policy_runner = File(
  path="mjlab/.venv/lib/python3.13/site-packages/rsl_rl/runners/on_policy_runner.py"
)
rsl_logger = File(
  path="mjlab/.venv/lib/python3.13/site-packages/rsl_rl/utils/logger.py"
)
wandb_utils = File(
  path="mjlab/.venv/lib/python3.13/site-packages/rsl_rl/utils/wandb_utils.py"
)
manager_env = File(path="mjlab/src/mjlab/envs/manager_based_rl_env.py")
reward_manager = File(path="mjlab/src/mjlab/managers/reward_manager.py")
command_manager = File(path="mjlab/src/mjlab/managers/command_manager.py")
tracking_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py")
motion_commands = File(path="mjlab/src/mjlab/tasks/tracking/mdp/commands.py")
video_recorder = File(path="mjlab/src/mjlab/utils/wrappers/video_recorder.py")


logs_correct = ProofFromCode(
  highlights=[
    Highlight(
      file=task_registration,
      content="""
The proof is scoped to the state-estimation MJLab task only: `Mjlab-Tracking-Flat-Unitree-G1` registers the default G1 tracking env, while the no-state-estimation sibling is a different task id and therefore excluded.""",
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
      file=g1_env_cfg,
      content="""
The G1 flat tracking factory defaults `has_state_estimation` to true and only removes state-estimation actor terms inside the false branch, so the scoped registration keeps state-estimation observations.""",
      text="""
def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:""",
    ),
    Highlight(
      file=tracking_rl_cfg,
      content="""
The task-specific RL config sets `experiment_name="g1_tracking"`, so its run root is the G1 tracking experiment bucket.""",
      text="""
    experiment_name="g1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,""",
    ),
    Highlight(
      file=train_script,
      content="""
Training constructs the concrete log directory as `logs/rsl_rl/g1_tracking/<timestamp>`, appending `_<run_name>` only when a run name is configured.""",
      text="""
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name""",
    ),
    Highlight(
      file=train_script,
      content="""
The train script prints both the selected device/seed/rank and the final log directory before creating the environment, making the training console identify where logs are going.""",
      text="""
  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None""",
    ),
    Highlight(
      file=train_script,
      content="""
Rank zero prints the concrete experiment directory, providing the second required console log about logging destination.""",
      text="""
  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )""",
    ),
    Highlight(
      file=train_script,
      content="""
The environment and agent configs are converted to dictionaries and dumped to `params/env.yaml` and `params/agent.yaml` before runner construction, preserving the exact training inputs before RSL-RL mutates config dictionaries.""",
      text="""
  # Write config files before runner creation, since the runner mutates agent_cfg
  # in-place (e.g., injecting non-serializable objects).
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)""",
    ),
    Highlight(
      file=os_utils,
      content="""
`dump_yaml` creates missing parent directories and writes YAML data, so the train script's env/agent dumps really materialize as YAML files under the run directory.""",
      text="""
  if not filename.suffix:
    filename = filename.with_suffix(".yaml")
  filename.parent.mkdir(parents=True, exist_ok=True)
  with open(filename, "w") as f:
    yaml.dump(data, f, sort_keys=sort_keys)""",
    ),
    Highlight(
      file=rl_config,
      content="""
The base runner config defaults `logger` to `"wandb"`, with project `"mjlab"`, so this task uses W&B unless the user explicitly overrides the logger.""",
      text='''
  logger: Literal["wandb", "tensorboard"] = "wandb"
  """The logger to use. Default is wandb."""
  wandb_project: str = "mjlab"
  """The wandb project name."""''',
    ),
    Highlight(
      file=on_policy_runner,
      content="""
The RSL-RL runner is constructed with the log directory and creates the `Logger` object from the same training config, environment config, device, and distributed rank data.""",
      text="""
        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )""",
    ),
    Highlight(
      file=rsl_logger,
      content="""
When logging starts, RSL-RL reads the configured logger type, lowercases it, and for `"wandb"` instantiates `WandbSummaryWriter` at the same `log_dir`.""",
      text="""
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)""",
    ),
    Highlight(
      file=wandb_utils,
      content="""
The W&B writer names the W&B run after the run directory basename and initializes W&B with `config={"log_dir": log_dir}`, tying remote logging back to the local timestamp/run-name directory.""",
      text="""
        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        # Get wandb project and entity
        try:
            project = cfg["wandb_project"]""",
    ),
    Highlight(
      file=wandb_utils,
      content="""
The W&B writer logs every scalar both to TensorBoard and to W&B, so scalar tags emitted by RSL-RL become W&B metrics by default.""",
      text="""
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({tag: scalar_value}, step=global_step)""",
    ),
    Highlight(
      file=manager_env,
      content="""
At reset, the RL environment rebuilds `extras["log"]` and merges reward, metrics, command, event, and termination reset info into it, making those episode summaries available to the RSL-RL logger.""",
      text="""
    # rewards manager.
    info = self.reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    # metrics manager.
    info = self.metrics_manager.reset(env_ids)
    self.extras["log"].update(info)
    # curriculum manager.
    info = self.curriculum_manager.reset(env_ids)
    self.extras["log"].update(info)
    # command manager.
    info = self.command_manager.reset(env_ids)
    self.extras["log"].update(info)""",
    ),
    Highlight(
      file=reward_manager,
      content="""
Reward episode logs are emitted with the `Episode_Reward/` prefix and normalized by max episode duration before their per-episode accumulators are reset.""",
      text="""
    for key in self._episode_sums.keys():
      episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
      extras["Episode_Reward/" + key] = (
        episodic_sum_avg / self._env.max_episode_length_s
      )
      self._episode_sums[key][env_ids] = 0.0""",
    ),
    Highlight(
      file=rsl_logger,
      content="""
RSL-RL consumes `extras["log"]` as episode extras, averages each key across buffered episode infos, writes slash-containing keys as exact scalar tags, and includes them in console output.""",
      text="""
            if self.ep_extras:
                # Iterate over all keys in the episode info dictionary
                for key in self.ep_extras[0]:
                    infotensor = torch.tensor([], device=self.device)
                    # Iterate over all steps
                    for ep_info in self.ep_extras:""",
    ),
    Highlight(
      file=rsl_logger,
      content="""
Slash-prefixed episode keys such as `Episode_Reward/...` and `Metrics/motion/...` are written to the logger without being moved under another namespace.""",
      text=r'''
                    if "/" in key:
                        self.writer.add_scalar(key, value, it)  # type: ignore
                        extras_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar("Episode/" + key, value, it)  # type: ignore
                        extras_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""''',
    ),
    Highlight(
      file=tracking_env_cfg,
      content="""
The base tracking environment registers the command under the name `motion`, so command reset metrics for this task are prefixed as `Metrics/motion/...` by the command manager.""",
      text="""
  commands: dict[str, CommandTermCfg] = {
    "motion": MotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,""",
    ),
    Highlight(
      file=command_manager,
      content="""
The command manager prefixes every command metric as `Metrics/{name}/{metric_name}`; for the G1 tracking command named `motion`, this is exactly `Metrics/motion/*`.""",
      text="""
    for name, term in self._terms.items():
      metrics = term.reset(env_ids=env_ids)
      for metric_name, metric_value in metrics.items():
        extras[f"Metrics/{name}/{metric_name}"] = metric_value
    return extras""",
    ),
    Highlight(
      file=motion_commands,
      content="""
The `MotionCommand` term initializes concrete tracking-error and sampling metrics, providing the metric names that flow into `Metrics/motion/*`.""",
      text="""
    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)""",
    ),
    Highlight(
      file=motion_commands,
      content="""
During command computation, motion tracking errors are updated from the reference motion and current Unitree G1 robot state before reset logging reads them.""",
      text="""
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )""",
    ),
    Highlight(
      file=train_script,
      content="""
Videos are optional because the training config defaults `video` to false and only enables `rgb_array` rendering when `cfg.video` is true.""",
      text="""
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False""",
    ),
    Highlight(
      file=train_script,
      content="""
When video is explicitly enabled on rank zero, training wraps the environment with `VideoRecorder` and writes MP4s under `log_dir/videos/train` at the configured interval/length.""",
      text="""
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")""",
    ),
    Highlight(
      file=video_recorder,
      content="""
The video wrapper creates MP4 filenames from the triggering step or episode and stores them in the configured video folder.""",
      text="""
    if self.trigger_type == "step":
      video_filename = f"{self.name_prefix}-step-{self.step_count}.mp4"
    elif self.trigger_type == "episode":
      video_filename = f"{self.name_prefix}-episode-{self.episode_count}.mp4"
    else:
      assert_never(self.trigger_type)

    self.current_video_path = self.video_folder / video_filename""",
    ),
    Highlight(
      file=rsl_logger,
      content="""
For W&B runs, the logger scans the run directory for any generated MP4s and uploads each available video, which keeps video logging optional and automatic when files exist.""",
      text="""
            # Upload available videos
            if self.logger_type == "wandb":
                for video in pathlib.Path(self.log_dir).rglob("*.mp4"):  # type: ignore
                    self.writer.save_video(video, it)  # type: ignore""",
    ),
  ]
)
