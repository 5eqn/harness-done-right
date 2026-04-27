from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

task_registry = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py")
g1_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py")
tracking_rl_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/rl_cfg.py")
base_runner = File(path="mjlab/src/mjlab/rl/runner.py")
tracking_runner = File(path="mjlab/src/mjlab/tasks/tracking/rl/runner.py")
rsl_on_policy_runner = File(
  path="mjlab/.venv/lib/python3.13/site-packages/rsl_rl/runners/on_policy_runner.py"
)
rsl_ppo = File(
  path="mjlab/.venv/lib/python3.13/site-packages/rsl_rl/algorithms/ppo.py"
)
train_script = File(path="mjlab/src/mjlab/scripts/train.py")
evaluate_script = File(path="mjlab/src/mjlab/tasks/tracking/scripts/evaluate.py")
play_script = File(path="mjlab/src/mjlab/scripts/play.py")
checkpoint_utils = File(path="mjlab/src/mjlab/utils/os.py")

checkpoint_correct = ProofFromCode(
  highlights=[
    Highlight(
      file=task_registry,
      content="""
Scope anchor: this proof is for exactly `Mjlab-Tracking-Flat-Unitree-G1`. That task registers the default G1 flat tracking env, its PPO runner config, and `MotionTrackingOnPolicyRunner`; the no-state-estimation variant is a separate task id below it.""",
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
      file=task_registry,
      content="""
The adjacent task id explicitly disables state estimation, proving it is outside this checkpoint analysis scope.""",
      text="""
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
The scoped task is the state-estimation version because the environment factory defaults `has_state_estimation` to `True`, and the registered scoped task calls it without overriding that argument.""",
      text="""
def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  \"\"\"Create Unitree G1 flat terrain tracking configuration.\"\"\"
  cfg = make_tracking_env_cfg()""",
    ),
    Highlight(
      file=tracking_rl_cfg,
      content="""
The Unitree G1 tracking runner config sets `save_interval=500`, so the save cadence required by this task is part of the scoped task's RL configuration.""",
      text="""
    experiment_name="g1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,""",
    ),
    Highlight(
      file=rsl_on_policy_runner,
      content="""
RSL-RL's learning loop honors the configured save interval by saving `model_{it}.pt` whenever the logger exists and the iteration is divisible by `self.cfg["save_interval"]`, then saves a final checkpoint at `self.current_learning_iteration` after training.""",
      text="""
            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()""",
    ),
    Highlight(
      file=rsl_ppo,
      content="""
The algorithm state saved by `self.alg.save()` includes actor weights, critic weights, and optimizer state, with optional RND model and optimizer state if RND is enabled.""",
      text="""
    def save(self) -> dict:
        \"\"\"Return a dict of all models for saving.\"\"\"
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd_optimizer.state_dict()
        return saved_dict""",
    ),
    Highlight(
      file=base_runner,
      content="""
MJLab's base runner overrides save to add the environment `common_step_counter` into `infos["env_state"]`, store `alg.save()` state, the current iteration, and infos in the checkpoint, and only call the logger's model upload when `self.cfg["upload_model"]` is true.""",
      text="""
  def save(self, path: str, infos=None) -> None:
    \"\"\"Save checkpoint.

    Extends the base implementation to persist the environment's
    common_step_counter and to respect the ``upload_model`` config flag.
    \"\"\"
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)""",
    ),
    Highlight(
      file=base_runner,
      content="""
Load starts from a raw `torch.load(..., weights_only=False)` and migrates old monolithic `model_state_dict` checkpoints into the current actor/critic state dict format, including actor, critic, actor observation normalizer, critic observation normalizer, and old standard-deviation keys.""",
      text="""
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    if "model_state_dict" in loaded_dict:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = loaded_dict.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():
        # Migrate actor keys.
        if key.startswith("actor."):
          new_key = key.replace("actor.", "mlp.")
          actor_state_dict[new_key] = value
        elif key.startswith("actor_obs_normalizer."):
          new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
          actor_state_dict[new_key] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        # Migrate critic keys.
        if key.startswith("critic."):
          new_key = key.replace("critic.", "mlp.")
          critic_state_dict[new_key] = value
        elif key.startswith("critic_obs_normalizer."):
          new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
          critic_state_dict[new_key] = value

      loaded_dict["actor_state_dict"] = actor_state_dict
      loaded_dict["critic_state_dict"] = critic_state_dict""",
    ),
    Highlight(
      file=base_runner,
      content="""
Load also migrates RSL-RL 4.x actor `std`/`log_std` entries to the 5.x distribution parameter names, delegates loading to the algorithm, restores the runner iteration when requested, and writes the saved `common_step_counter` back into the unwrapped environment.""",
      text="""
    # Migrate rsl-rl 4.x actor keys to 5.x distribution keys.
    actor_sd = loaded_dict.get("actor_state_dict", {})
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos""",
    ),
    Highlight(
      file=rsl_ppo,
      content="""
The algorithm load path restores exactly the requested model and optimizer components: actor, critic, optimizer, optional RND state, and optional RND optimizer, and returns whether iteration should be restored.""",
      text="""
    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        \"\"\"Load specified models from a saved dict.\"\"\"
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)""",
    ),
    Highlight(
      file=tracking_runner,
      content="""
The tracking ONNX model is not just a policy wrapper: it registers motion buffers for joint position/velocity, global body pose, and global body velocities, then its forward pass emits policy actions plus the motion reference slice for the requested time step.""",
      text="""
  def __init__(self, actor, motion):
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
    self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
    self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
    self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
    self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
    self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
    self.time_step_total: int = self.joint_pos.shape[0]  # type: ignore[index]

  def forward(self, x, time_step):
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.policy(x),
      self.joint_pos[time_step_clamped],  # type: ignore[index]
      self.joint_vel[time_step_clamped],  # type: ignore[index]
      self.body_pos_w[time_step_clamped],  # type: ignore[index]
      self.body_quat_w[time_step_clamped],  # type: ignore[index]
      self.body_lin_vel_w[time_step_clamped],  # type: ignore[index]
      self.body_ang_vel_w[time_step_clamped],  # type: ignore[index]
    )""",
    ),
    Highlight(
      file=tracking_runner,
      content="""
The tracking runner exports that motion-aware wrapper to ONNX with two inputs (`obs`, `time_step`) and seven outputs (`actions` plus the six motion reference outputs), so each checkpoint save can produce an ONNX file that carries the motion buffers needed for playback/inspection.""",
      text="""
  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    os.makedirs(path, exist_ok=True)
    cmd = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
    model = _OnnxMotionModel(self.alg.get_policy(), cmd.motion)
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      model,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs", "time_step"],
      output_names=[
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ],
      dynamic_axes={},
      dynamo=False,
    )""",
    ),
    Highlight(
      file=tracking_runner,
      content="""
After base checkpoint save, the tracking runner resolves an ONNX path from the checkpoint path, exports the ONNX, attaches base metadata plus tracking-specific `anchor_body_name` and `body_names`, and uploads the ONNX to W&B only when the logger is W&B and `upload_model` is true.""",
      text="""
  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      motion_term = cast(
        MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
      )
      metadata.update(
        {
          "anchor_body_name": motion_term.cfg.anchor_body_name,
          "body_names": list(motion_term.cfg.body_names),
        }
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
        if self.registry_name is not None:
          wandb.run.use_artifact(self.registry_name)  # type: ignore
          self.registry_name = None
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")""",
    ),
    Highlight(
      file=train_script,
      content="""
Training can resume from W&B: when `cfg.agent.resume` is true and `cfg.wandb_run_path` is provided, it downloads or reuses the selected W&B checkpoint and reports the checkpoint/run/cache status.""",
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
        )""",
    ),
    Highlight(
      file=train_script,
      content="""
Training can also resume locally: without `wandb_run_path`, it resolves the checkpoint from `logs/rsl_rl/{experiment_name}` using `load_run` and `load_checkpoint`, then passes the resulting path to the runner's load method before learning.""",
      text="""
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.""",
    ),
    Highlight(
      file=train_script,
      content="""
The resolved training resume path is actually loaded into the task-specific runner before `runner.learn(...)`, so restoration is not merely discovery.""",
      text="""
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )""",
    ),
    Highlight(
      file=evaluate_script,
      content="""
Tracking evaluation resolves its policy checkpoint from W&B via the same utility, prints the concrete path, constructs the registered runner class, and loads the checkpoint before requesting the inference policy.""",
      text="""
  log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
  resume_path, _ = get_wandb_checkpoint_path(
    log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
  )
  print(f"[INFO] Loading checkpoint: {resume_path}")

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)""",
    ),
    Highlight(
      file=play_script,
      content="""
Play mode supports local checkpoint files: for trained agents, `checkpoint_file` is accepted directly, must exist, is announced by name, and supplies the resume directory used for playback artifacts.""",
      text="""
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:""",
    ),
    Highlight(
      file=play_script,
      content="""
Play mode also supports W&B checkpoints when no local file is supplied: it requires `wandb_run_path`, downloads or reuses the selected checkpoint, and reports checkpoint name, run id, and cache/download status.""",
      text="""
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent""",
    ),
    Highlight(
      file=play_script,
      content="""
The resolved play checkpoint is loaded with actor-only `load_cfg`, strict state dict checking, and map-location to the chosen device before the inference policy is created.""",
      text="""
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)""",
    ),
    Highlight(
      file=play_script,
      content="""
The play viewer can hot-swap local checkpoints by listing `*.pt` files in the checkpoint directory, sorting them by parsed iteration, and loading the selected file through `_reload_policy`.""",
      text="""
    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )""",
    ),
    Highlight(
      file=play_script,
      content="""
The play viewer can hot-swap W&B checkpoints too: it lists `.pt` files from the W&B run, sorts by parsed iteration, and resolves the selected checkpoint through `get_wandb_checkpoint_path` before reloading the policy.""",
      text="""
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )""",
    ),
    Highlight(
      file=checkpoint_utils,
      content="""
Local checkpoint resolution deliberately ignores the W&B cache directory, matches run and checkpoint names by regex, sorts matching runs/checkpoints, and returns the latest matching checkpoint path.""",
      text="""
  if not log_path.exists():
    raise ValueError(f"Log path does not exist: {log_path}")
  # Exclude wandb_checkpoints directory which is used for caching downloaded checkpoints.
  runs = [
    log_path / run.name
    for run in log_path.iterdir()
    if run.is_dir() and run.name != "wandb_checkpoints" and re.match(run_dir, run.name)
  ]
  if len(runs) == 0:
    raise ValueError(f"No run directories found in {log_path} matching '{run_dir}'")
  if sort_alpha:
    runs.sort()
  else:
    runs = sorted(runs, key=lambda p: p.stat().st_mtime)
  run_path = runs[-1]

  model_checkpoints = [
    f.name for f in run_path.iterdir() if re.match(checkpoint, f.name)
  ]
  if len(model_checkpoints) == 0:
    raise ValueError(f"No checkpoint found in {run_path} matching {checkpoint}")
  model_checkpoints.sort(key=lambda m: f"{m:0>15}")
  checkpoint_file = model_checkpoints[-1]
  return run_path / checkpoint_file""",
    ),
    Highlight(
      file=checkpoint_utils,
      content="""
W&B checkpoint resolution queries `model_%.pt` files, keeps only strict `model_<digits>.pt` names, chooses the highest numeric iteration unless a specific checkpoint name is supplied, validates explicit names, caches downloads under `wandb_checkpoints/{run_id}`, and returns both path and cache status.""",
      text=r"""
  # Query wandb API to find the latest checkpoint.
  api = wandb.Api()
  wandb_run = api.run(str(run_path))
  files = [
    file.name
    for file in wandb_run.files(pattern="model_%.pt")
    if re.match(r"^model_\d+\.pt$", file.name)
  ]
  if checkpoint_name is None:
    checkpoint_file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
  else:
    if checkpoint_name not in files:
      raise ValueError(
        f"Checkpoint '{checkpoint_name}' not found in run {run_path}."
        f" Available: {files}"
      )
    checkpoint_file = checkpoint_name

  checkpoint_path = download_dir / checkpoint_file

  # If this checkpoint is not cached locally, download it.
  was_cached = checkpoint_path.exists()
  if not was_cached:
    download_dir.mkdir(parents=True, exist_ok=True)
    wandb_file = wandb_run.file(str(checkpoint_file))
    wandb_file.download(str(download_dir), replace=True)

  return checkpoint_path, was_cached""",
    ),
  ]
)
