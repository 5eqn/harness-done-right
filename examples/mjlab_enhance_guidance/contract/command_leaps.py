from enhance_guidance import AtomicLeap, File, Highlight


holomotion_command = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_motion_tracking_command.py"
)
holomotion_h5 = File(path="HoloMotion/holomotion/src/training/h5_dataloader.py")
mjlab_commands = File(path="mjlab/src/mjlab/tasks/tracking/mdp/commands.py")


command_leaps: list[AtomicLeap] = [
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_command,
            content="HoloMotion builds HDF5-v2 train/validation motion datasets from config before constructing the cache-backed command pipeline.",
            text="""
            (
                train_dataset,
                val_dataset,
                cache_kwargs,
            ) = build_motion_datasets_from_cfg(
                motion_cfg=mcfg,
                max_frame_length=max_frame_length,
                min_window_length=min_frame_length,
                world_frame_normalization=world_frame_norm,
                handpicked_motion_names=mcfg.get(
                    "handpicked_motion_names", None
                ),
                excluded_motion_names=mcfg.get("excluded_motion_names", None),
                allowed_prefixes=allowed_prefixes,
            )
            if len(train_dataset) == 0:
                raise ValueError(
                    "Training dataset is empty. Check that all HDF5 v2 "
                    "roots contain valid clips with length "
                    f">= {min_frame_length}"
                )""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_commands,
            content="MJLab eagerly loads one `.npz` motion file into tensors and derives one global motion length.",
            text="""
class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = self.joint_pos.shape[0]""",
        ),
        change_direction=(
            "Replace MJLab's eager single-file `.npz` `MotionLoader` with an "
            "HDF5-v2 motion-library builder that accepts train/validation roots, "
            "enumerates valid root/DoF windows, performs FK-derived reference "
            "materialization, and fails fast when the training dataset is empty."
        ),
        change_reason=(
            "A single in-memory `.npz` forces every environment to track one "
            "motion stream and cannot scale to AMASS-style multi-clip training. "
            "The HDF5-v2 root/DoF dataset makes motion diversity, filtering, "
            "world-frame normalization, validation splits, and bounded windows "
            "first-class parts of the command pipeline."
        ),
        changed_code="""
class Hdf5V2MotionLibrary:
  def __init__(self, cfg: MotionLibraryCfg, device: torch.device) -> None:
    self.cfg = cfg
    self.device = device
    self.train_dataset, self.val_dataset, cache_kwargs = build_motion_datasets_from_cfg(
      motion_cfg=cfg.to_dict(),
      max_frame_length=cfg.max_frame_length,
      min_window_length=cfg.min_frame_length,
      world_frame_normalization=cfg.world_frame_normalization,
      handpicked_motion_names=cfg.handpicked_motion_names,
      excluded_motion_names=cfg.excluded_motion_names,
      allowed_prefixes=cfg.cache.allowed_prefixes,
    )
    if len(self.train_dataset) == 0:
      raise ValueError(
        "Training dataset is empty. Check HDF5 v2 roots and min_frame_length."
      )

    self.cache = MotionClipBatchCache(
      train_dataset=self.train_dataset,
      val_dataset=self.val_dataset,
      batch_size=cfg.cache.max_num_clips,
      stage_device=self._stage_device(cfg.cache.device),
      num_workers=cfg.dataloader.num_workers,
      prefetch_factor=cfg.dataloader.prefetch_factor,
      pin_memory=cfg.dataloader.pin_memory,
      persistent_workers=cfg.dataloader.persistent_workers,
      sampler_rank=cfg.process_id,
      sampler_world_size=cfg.num_processes,
      allowed_prefixes=cfg.cache.allowed_prefixes,
      swap_interval_steps=cfg.cache.swap_interval_steps,
      seed=cfg.seed,
      loader_timeout=cfg.dataloader.timeout,
      **cache_kwargs,
    )
""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_h5,
            content="HoloMotion enumerates bounded HDF5 windows per manifest clip, preserving source key, shard, start, and length.",
            text="""
            aliases = self._build_motion_key_aliases(motion_key, meta)
            if self.handpicked_motion_names is not None and not any(
                alias in self.handpicked_motion_names for alias in aliases
            ):
                continue
            if self.excluded_motion_names is not None and any(
                alias in self.excluded_motion_names for alias in aliases
            ):
                continue

            shard_index = int(meta.get("shard", 0))
            start = int(meta.get("start", 0))
            length = int(meta.get("length", 0))

            if length <= 0:
                continue

            remaining = length
            offset = 0
            window_index = 0
            while remaining > 0:
                window_length = min(self.max_frame_length, remaining)
                if window_length >= self.min_window_length:
                    win_start = start + offset
                    unique_key = (
                        f"{motion_key}__start_{win_start}_len_{window_length}"
                    )
                    windows.append(
                        MotionWindow(
                            motion_key=unique_key,
                            shard_index=shard_index,
                            start=win_start,
                            length=window_length,
                            raw_motion_key=motion_key,
                            window_index=window_index,
                        )
                    )
                    window_index += 1""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_commands,
            content="MJLab advances a single per-environment frame counter through the one loaded motion length.",
            text="""
  def _update_command(self):
    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    self.update_relative_body_poses()""",
        ),
        change_direction=(
            "Replace the global `time_steps` loop over one motion length with "
            "per-environment cache assignments: each environment stores a clip "
            "row, a current frame, and the original sampled start frame from a "
            "bounded HDF5 window."
        ),
        change_reason=(
            "Window-level cache assignment lets different environments train on "
            "different clips and start offsets, prevents future-frame reads from "
            "running past each window, and decouples command progression from a "
            "single `.npz` timeline."
        ),
        changed_code="""
def _init_per_env_cache(self) -> None:
  self._clip_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
  self._frame_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
  self._start_frame_indices = torch.zeros(
    self.num_envs, dtype=torch.long, device=self.device
  )
  clip_idx, frame_idx = self.motion_library.cache.sample_env_assignments(
    self.num_envs,
    self.cfg.n_future_frames,
    self.device,
    deterministic_start=self.is_evaluating,
  )
  self._clip_indices[:] = clip_idx
  self._frame_indices[:] = frame_idx
  self._start_frame_indices[:] = frame_idx

def _update_command(self) -> None:
  active_env_ids = torch.arange(self.num_envs, device=self.device)
  if self._env.episode_length_buf is not None:
    active_env_ids = active_env_ids[self._env.episode_length_buf != 0]
  self._frame_indices[active_env_ids] += 1
  self._resample_when_window_ends(active_env_ids)
""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_h5,
            content="HoloMotion gathers current plus future timesteps and clamps each gather to the selected clip length.",
            text="""
        temporal_span = 1 + int(n_future_frames)
        time_offsets = torch.arange(
            temporal_span, device=staged_device, dtype=torch.long
        )
        gather_timesteps = frame_indices[:, None] + time_offsets[None, :]

        lengths = batch.lengths
        max_valid = torch.clamp(
            lengths.index_select(0, selected_clips) - 1, min=0
        )
        gather_timesteps = torch.minimum(
            gather_timesteps, max_valid[:, None]
        ).clone()""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_commands,
            content="MJLab command observation exposes only the current joint position and velocity frame.",
            text="""
  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]""",
        ),
        change_direction=(
            "Route command access through cache gathers that return the current "
            "frame plus the configured future horizon, then define current, "
            "future, and immediate-next getters on top of the same gather path."
        ),
        change_reason=(
            "Generalized tracking needs reference windows, not just one current "
            "joint frame. Current+future gathers give observations, rewards, "
            "terminations, and reset logic a coherent temporal target while "
            "clamping safely at the end of each cached clip."
        ),
        changed_code="""
def _get_ref_state_array(self, base_key: str, prefix: str = "ref_") -> torch.Tensor:
  tensor_key = resolve_reference_tensor_key(
    batch_tensors=self.motion_library.cache.current_batch.tensors,
    base_key=base_key,
    prefix=prefix,
  )
  return self.motion_library.cache.gather_tensor(
    tensor_key,
    clip_indices=self._clip_indices,
    frame_indices=self._frame_indices,
    n_future_frames=self.cfg.n_future_frames,
  )

@property
def command(self) -> torch.Tensor:
  dof_pos = self._get_ref_state_array("dof_pos")[:, 0, :][..., self.urdf2sim_dof_idx]
  dof_vel = self._get_ref_state_array("dof_vel")[:, 0, :][..., self.urdf2sim_dof_idx]
  return torch.cat([dof_pos, dof_vel], dim=-1)

@property
def command_fut(self) -> torch.Tensor:
  dof_pos = self._get_ref_state_array("dof_pos")[:, 1:, :][..., self.urdf2sim_dof_idx]
  dof_vel = self._get_ref_state_array("dof_vel")[:, 1:, :][..., self.urdf2sim_dof_idx]
  return torch.cat([dof_pos, dof_vel], dim=-1).flatten(start_dim=1)
""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_command,
            content="HoloMotion reset resampling chooses new cache assignments, refreshes reference access, and aligns simulator state.",
            text="""
        self._record_completion_rate_for_envs(idxs)
        clip_idx, frame_idx = self._motion_cache.sample_env_assignments(
            len(idxs),
            self.cfg.n_fut_frames,
            self.device,
            deterministic_start=(eval or self._is_evaluating),
        )
        self._clip_indices[idxs] = clip_idx
        self._frame_indices[idxs] = frame_idx
        self._start_frame_indices[idxs] = frame_idx
        self._reward_sum_since_assign[idxs] = 0.0
        self._step_count_since_assign[idxs] = 0.0
        self._update_ref_motion_state_from_cache(env_ids=idxs)
        self._align_root_to_ref(idxs)
        self._align_dof_to_ref(idxs)""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_commands,
            content="MJLab reset alignment samples root state from the current `.npz` frame before adding RSI perturbations.",
            text="""
    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )""",
        ),
        change_direction=(
            "Keep reset-time RSI, but make it cache-assignment aware: record "
            "completion for the old window, sample a new clip/start frame, reset "
            "assignment accumulators, gather the new current reference, and then "
            "align root and DoF state from cache-backed getters."
        ),
        change_reason=(
            "With many cached windows, reset cannot merely choose another index "
            "inside the same `.npz`. It must move the environment to a new "
            "clip/window assignment and align simulator state to that assignment "
            "so the first post-reset observation, reward, and termination targets "
            "all reference the same HDF5 window."
        ),
        changed_code="""
def _resample_command(self, env_ids: torch.Tensor, *, eval: bool = False) -> None:
  if env_ids.numel() == 0:
    return
  self._record_completion_rate_for_envs(env_ids)
  clip_idx, frame_idx = self.motion_library.cache.sample_env_assignments(
    int(env_ids.numel()),
    self.cfg.n_future_frames,
    self.device,
    deterministic_start=(eval or self.is_evaluating),
  )
  self._clip_indices[env_ids] = clip_idx
  self._frame_indices[env_ids] = frame_idx
  self._start_frame_indices[env_ids] = frame_idx
  self._reward_sum_since_assign[env_ids] = 0.0
  self._step_count_since_assign[env_ids] = 0.0

  root_pos = self.get_ref_motion_root_global_pos_cur().clone()
  root_rot = self.get_ref_motion_root_global_rot_quat_wxyz_cur().clone()
  root_lin_vel = self.get_ref_motion_root_global_lin_vel_cur().clone()
  root_ang_vel = self.get_ref_motion_root_global_ang_vel_cur().clone()
  root_pos, root_rot, root_lin_vel, root_ang_vel = self._apply_root_perturbations(
    env_ids, root_pos, root_rot, root_lin_vel, root_ang_vel
  )
  self._write_root_state_to_sim(env_ids, root_pos, root_rot, root_lin_vel, root_ang_vel)
  self._align_dof_to_ref(env_ids)
""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_command,
            content="HoloMotion supports uniform, weighted-bin, and cache-curriculum sampling as cache-level strategies.",
            text="""
        sampling_strategy_cfg = mcfg.get("sampling_strategy", None)
        if sampling_strategy_cfg is None:
            sampling_strategy = "uniform"
        else:
            sampling_strategy = str(sampling_strategy_cfg).lower()
        if sampling_strategy == "weighted_bin":
            weighted_bin_cfg = mcfg.get("weighted_bin", {})
            self._motion_cache.enable_weighted_bin_sampling(
                cfg=dict(weighted_bin_cfg or {})
            )
        elif sampling_strategy == "curriculum":
            curriculum_cfg = dict(mcfg.get("curriculum", {}) or {})
            self._motion_cache.enable_cache_curriculum_sampling(
                cfg=curriculum_cfg
            )
        elif sampling_strategy not in ("uniform", "curriculum"):
            raise ValueError(
                f"Invalid sampling_strategy '{sampling_strategy}'. "
                "Expected one of ['curriculum', 'uniform', 'weighted_bin']."
            )""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_commands,
            content="MJLab adaptive sampling biases temporal bins inside the single loaded motion based on termination failures.",
            text="""
  def _adaptive_sampling(self, env_ids: torch.Tensor):
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)""",
        ),
        change_direction=(
            "Move adaptive sampling from time-bin failure counts over one motion "
            "into cache-level sampling strategies: default uniform windows, "
            "optional regex weighted-bin selection, and optional window-level "
            "curriculum driven by completion/error statistics."
        ),
        change_reason=(
            "MJLab's adaptive bins can only bias temporal regions inside a single "
            "loaded motion. HoloMotion-style sampling lets engineers rebalance "
            "whole datasets by source or difficulty and prioritize hard cached "
            "windows without losing broad uniform coverage."
        ),
        changed_code="""
def _configure_motion_sampling(self) -> None:
  strategy = (self.cfg.motion.sampling_strategy or "uniform").lower()
  if strategy == "weighted_bin":
    self.motion_library.cache.enable_weighted_bin_sampling(
      cfg=dict(self.cfg.motion.weighted_bin)
    )
  elif strategy == "curriculum":
    self.motion_library.cache.enable_cache_curriculum_sampling(
      cfg=dict(self.cfg.motion.curriculum)
    )
  elif strategy != "uniform":
    raise ValueError(
      f"Invalid sampling_strategy {strategy!r}; expected uniform, weighted_bin, or curriculum."
    )
  self._sampling_strategy = strategy

def update_curriculum_reward_accumulators(self, rewards: torch.Tensor) -> None:
  if self._sampling_strategy != "curriculum":
    return
  ids = torch.arange(self.num_envs, device=self.device)
  self._reward_sum_since_assign[ids] += rewards.view(-1)[ids]
  self._mpkpe_sum_since_assign[ids] += self.metrics["mpkpe_whole_body"][ids]
  self._step_count_since_assign[ids] += 1.0
""",
    ),
]
