from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

task_registry = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py")
registry = File(path="mjlab/src/mjlab/tasks/registry.py")
train_script = File(path="mjlab/src/mjlab/scripts/train.py")
tracking_runner = File(path="mjlab/src/mjlab/tasks/tracking/rl/runner.py")
base_runner = File(path="mjlab/src/mjlab/rl/runner.py")
manager_env = File(path="mjlab/src/mjlab/envs/manager_based_rl_env.py")
vecenv_wrapper = File(path="mjlab/src/mjlab/rl/vecenv_wrapper.py")

trainer_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=task_registry,
            content="""
The scoped training target is exactly `Mjlab-Tracking-Flat-Unitree-G1`. Its registration binds the default G1 flat tracking environment config, the G1 tracking PPO runner config, and the custom `MotionTrackingOnPolicyRunner`; the adjacent no-state-estimation task has a different id and is outside this proof.""",
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
            file=registry,
            content="""
Task lookup returns deep copies of the registered train/play environment config and RL config, so launching the scoped task receives the registered `ManagerBasedRlEnvCfg` and `RslRlBaseRunnerCfg` without mutating the global registry.""",
            text="""
def load_env_cfg(task_name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  \"\"\"Load environment configuration for a task.

  Returns a deep copy to prevent mutation of the registered config.
  \"\"\"
  return deepcopy(
    _REGISTRY[task_name].env_cfg if not play else _REGISTRY[task_name].play_env_cfg
  )


def load_rl_cfg(task_name: str) -> RslRlBaseRunnerCfg:
  \"\"\"Load RL configuration for a task.

  Returns a deep copy to prevent mutation of the registered config.
  \"\"\"
  return deepcopy(_REGISTRY[task_name].rl_cfg)""",
        ),
        Highlight(
            file=train_script,
            content="""
The train CLI materializes launch config from the selected task by calling `load_env_cfg(task_id)` and `load_rl_cfg(task_id)`. For `Mjlab-Tracking-Flat-Unitree-G1`, those calls load the registered state-estimation G1 tracking env and PPO agent config above.""",
            text="""
  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)""",
        ),
        Highlight(
            file=train_script,
            content="""
The launcher selects CPU versus CUDA from `CUDA_VISIBLE_DEVICES`, maps each distributed process to its `LOCAL_RANK`, sets `MUJOCO_EGL_DEVICE_ID` to the same local GPU, offsets the seed per rank, and writes the chosen seed back into both agent and environment configs.""",
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
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed""",
        ),
        Highlight(
            file=train_script,
            content="""
Launch-time GPU plumbing is grounded: selected GPU ids become `CUDA_VISIBLE_DEVICES`, MuJoCo is forced to EGL rendering, single-device runs call `run_train` directly, and multi-GPU runs are delegated to `torchrunx`.""",
            text="""
  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.""",
        ),
        Highlight(
            file=train_script,
            content="""
The multi-GPU path is not speculative: it imports `torchrunx`, configures a per-run log directory when none is already set, and launches one worker per selected GPU while copying MuJoCo-related environment variables into the worker processes.""",
            text="""
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)""",
        ),
        Highlight(
            file=train_script,
            content="""
Tracking-task motion-file resolution is explicit. The trainer detects a `MotionCommandCfg`, accepts an existing local `motion_file`, otherwise downloads `motion.npz` from a W&B artifact registry name, and rejects tracking launches without one of those two sources.""",
            text="""
  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")
    elif cfg.registry_name:
      # Download from WandB registry.
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      raise ValueError(
        "For tracking tasks, provide either:\\n"
        "  --registry-name your-org/motions/motion-name (download from WandB)\\n"
        "  --env.commands.motion.motion-file /path/to/motion.npz (local file)"
      )""",
        ),
        Highlight(
            file=train_script,
            content="""
The trainer constructs the real MJLab manager-based RL environment on the selected device, optionally enabling RGB-array render mode for video recording. This is the environment whose managers perform action, reward, command, termination, and observation updates.""",
            text="""
  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )""",
        ),
        Highlight(
            file=train_script,
            content="""
Before runner creation, the trainer wraps the environment in `RslRlVecEnvWrapper`, converts dataclass configs with `asdict`, loads the registered runner class, passes the W&B registry name only for tracking tasks, and dumps env/agent YAML configs on rank zero. The comment is important: dumping happens before runner mutation of `agent_cfg`.""",
            text="""
  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

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
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)""",
        ),
        Highlight(
            file=train_script,
            content="""
The runner is created from the registered task runner class and receives the wrapped env, serialized agent config, log directory, selected device, and tracking-only keyword arguments. For the scoped task, `load_runner_cls` returns `MotionTrackingOnPolicyRunner` from the task registration.""",
            text="""
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)""",
        ),
        Highlight(
            file=tracking_runner,
            content="""
`MotionTrackingOnPolicyRunner` is the custom trainer class registered for the scoped G1 tracking task. It subclasses `MjlabOnPolicyRunner`, delegates actual on-policy runner construction to `super().__init__`, and stores the registry name for later motion artifact linkage.""",
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
            file=train_script,
            content="""
Resume plumbing supports both W&B and local files. If `cfg.agent.resume` is true, W&B resume resolves through `get_wandb_checkpoint_path`; otherwise local resume resolves through `get_checkpoint_path` using the configured run/checkpoint selectors.""",
            text="""
  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )""",
        ),
        Highlight(
            file=train_script,
            content="""
If a resume path was resolved, the script calls `runner.load(str(resume_path))` before training. Training then launches through RSL-RL with the task config's `max_iterations` and `init_at_random_ep_len=True`, exactly matching the requested training launch contract.""",
            text="""
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )""",
        ),
        Highlight(
            file=base_runner,
            content="""
`MjlabOnPolicyRunner.load` restores trainer and environment state, not just neural-network weights. It loads the checkpoint, migrates legacy state-dict formats when needed, calls the algorithm loader, updates the learning iteration, and restores `common_step_counter` from checkpoint infos.""",
            text="""
    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos""",
        ),
        Highlight(
            file=manager_env,
            content="""
The environment step consumes policy actions first: it routes the action through the action manager on the selected device, then for every decimated physics substep applies processed actions, writes scene data, steps MuJoCo, updates scene state, and computes substep metrics.""",
            text="""
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
      self.metrics_manager.compute_substep()""",
        ),
        Highlight(
            file=manager_env,
            content="""
After physics, `env.step` increments counters, computes terminations, keeps separated `terminated` and `time_outs` buffers, computes rewards at the environment step dt, and computes step metrics. This is the update/reward half of the trainer feedback loop.""",
            text="""
    # Update env counters.
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # Check terminations and compute rewards.
    # NOTE: Derived quantities (xpos, xquat, ...) are stale by one physics
    # substep here. See the docstring above for why this is acceptable.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()""",
        ),
        Highlight(
            file=manager_env,
            content="""
The same step handles reset/update/observation ordering: reset envs are reset and written, MuJoCo forward refreshes derived quantities for all envs, commands and step/interval events update, sensors run, observations are recomputed with history, and the method returns obs, reward, terminated, timeout, and extras.""",
            text="""
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_pre_reset(reset_env_ids)
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()

    # Single forward() call: recompute derived quantities from current
    # qpos/qvel for every env. For non-reset envs this resolves the
    # one-substep staleness left by mj_step; for reset envs it picks up
    # the freshly written reset state.
    self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)

    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)""",
        ),
        Highlight(
            file=manager_env,
            content="""
The return signature from `ManagerBasedRlEnv.step` preserves the split between termination and timeout, which the RSL-RL wrapper later folds into a single done tensor.""",
            text="""
    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )""",
        ),
        Highlight(
            file=vecenv_wrapper,
            content="""
The RSL-RL wrapper adapts MJLab's Gym-style split termination API into the RSL-RL vector-env API: it optionally clips actions, calls `env.step`, ORs `terminated | truncated`, converts the boolean result to long `dones`, records `time_outs` for infinite-horizon configs, and returns TensorDict observations, rewards, dones, and extras.""",
            text="""
  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    term_or_trunc = terminated | truncated
    assert isinstance(rew, torch.Tensor)
    assert isinstance(term_or_trunc, torch.Tensor)
    dones = term_or_trunc.to(dtype=torch.long)
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    return (
      TensorDict(obs_dict, batch_size=[self.num_envs]),
      rew,
      dones,
      extras,
    )""",
        ),
    ]
)
