from project_analysis import File, Highlight, ProofFromCode


training_cfg = File(
    path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_mlp.yaml"
)
algo_cfg = File(path="HoloMotion/holomotion/config/algo/ppo.yaml")
env_cfg = File(path="HoloMotion/holomotion/config/env/motion_tracking.yaml")
robot_cfg = File(
    path="HoloMotion/holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"
)
motion_env = File(path="HoloMotion/holomotion/src/env/motion_tracking.py")
motion_command = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_motion_tracking_command.py"
)
h5_dataloader = File(path="HoloMotion/holomotion/src/training/h5_dataloader.py")
observation_impl = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_observation.py"
)
reward_impl = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_rewards.py"
)
termination_impl = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_termination.py"
)
algo_base = File(path="HoloMotion/holomotion/src/algo/algo_base.py")
ppo = File(path="HoloMotion/holomotion/src/algo/ppo.py")


command_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=training_cfg,
            content="""
Scope anchor: this proof is for the G1 29-DoF motion-tracking MLP job. That composition selects HoloMotion's `motion_tracking` environment, the motion-tracking observation/reward/termination tables, and the Unitree G1 robot config whose `robot.motion` subtree supplies the HDF5 motion library below.""",
            text="""
defaults:
  - /training: train_base
  - /algo: ppo
  - /robot: unitree/G1/29dof/29dof_training_isaaclab
  - /env: motion_tracking
  - /env/terminations: termination_motion_tracking
  - /env/observations: motion_tracking/obs_motion_tracking_mlp
  - /env/rewards: motion_tracking/rew_motion_tracking
  - /env/domain_randomization: domain_rand_medium
  - /env/terrain: isaaclab_rough
  - /modules: motion_tracking/motion_tracking_mlp""",
        ),
        Highlight(
            file=training_cfg,
            content="""
The concrete training dataset root is resolved through Hydra into `robot.motion.train_hdf5_roots` and `val_hdf5_roots`. For this default training entry point, both train and validation references come from `data/h5v2_datasets/AMASS_test` unless the run overrides `train_hdf5_roots`.""",
            text="""
train_hdf5_roots:
  - data/h5v2_datasets/AMASS_test""",
        ),
        Highlight(
            file=env_cfg,
            content="""
The environment registers one motion command named `ref_motion`. It is a `MotionCommandCfg` with `command_obs_name: bydmmc_ref_motion`, the robot motion-library config, URDF DoF/body order, body groupings, `torso_link` anchor, perturbation ranges, a long 100-second resample interval, 10 future frames through `${obs.n_fut_frames}`, and 50 Hz target motion timing.""",
            text="""
    commands:
      ref_motion:
        type: MotionCommandCfg
        params:
          command_obs_name: bydmmc_ref_motion
          motion_lib_cfg: ${robot.motion}
          urdf_dof_names: ${robot.dof_names}
          urdf_body_names: ${robot.body_names}
          arm_dof_names: ${robot.arm_dof_names}
          waist_dof_names: ${robot.waist_dof_names}
          leg_dof_names: ${robot.leg_dof_names}
          arm_body_names: ${robot.arm_body_names}
          torso_body_names: ${robot.torso_body_names}
          leg_body_names: ${robot.leg_body_names}
          anchor_bodylink_name: ${robot.anchor_body}
          asset_name: robot
          debug_vis: true
          root_pose_perturb_range: ${domain_rand.motion_init_perturb.root_pose_perturb_range}
          root_vel_perturb_range: ${domain_rand.motion_init_perturb.root_vel_perturb_range}
          dof_pos_perturb_range: ${domain_rand.motion_init_perturb.dof_pos_perturb_range}
          dof_vel_perturb_range: ${domain_rand.motion_init_perturb.dof_vel_perturb_range}
          resample_time_interval_s: 100
          n_fut_frames: ${obs.n_fut_frames}
          target_fps: 50""",
        ),
        Highlight(
            file=motion_env,
            content="""
HoloMotion resolves the command YAML to a Python dict, separates `MotionCommandCfg` terms from velocity terms, injects the per-rank seed, process id, process count, and evaluation flag into `ref_motion`, then builds an IsaacLab commands config. This is the construction bridge from config to the live `RefMotionCommand` term.""",
            text="""
            # Populate RefMotionCommand distributed params when present.
            if "ref_motion" in motion_cmds:
                if self.accelerator is not None:
                    cmd_process_id = self.accelerator.process_index
                    cmd_num_processes = self.accelerator.num_processes
                else:
                    cmd_process_id = getattr(self.config, "process_id", 0)
                    cmd_num_processes = getattr(
                        self.config, "num_processes", 1
                    )
                motion_cmds["ref_motion"]["params"].update(
                    {
                        "seed": int(seed_val),
                        "process_id": cmd_process_id,
                        "num_processes": cmd_num_processes,
                        "is_evaluating": self.is_evaluating,
                    }
                )""",
        ),
        Highlight(
            file=motion_command,
            content="""
The command config class pins the runtime implementation: `MotionCommandCfg.class_type` is `RefMotionCommand`. Required fields include the reference observation schema, URDF DoF/body names, motion library config, distributed seed/process settings, evaluation flag, resample timing, future horizon, target FPS, anchor, asset name, and perturbation ranges.""",
            text="""
@configclass
class MotionCommandCfg(CommandTermCfg):
    \"\"\"Configuration for the motion command.\"\"\"

    class_type: type = RefMotionCommand

    command_obs_name: str = MISSING
    urdf_dof_names: list[str] = MISSING
    urdf_body_names: list[str] = MISSING""",
        ),
        Highlight(
            file=motion_command,
            content="""
`RefMotionCommand` construction immediately initializes the robot handle, tensor buffers, and the motion library/cache. This means the command term owns both simulator mapping and HDF5-backed reference access before the first reset or step.""",
            text="""
        super().__init__(cfg, env)
        self._env = env
        self._is_evaluating = self.cfg.is_evaluating
        self._runtime_process_id = int(self.cfg.process_id)
        self._runtime_num_processes = max(1, int(self.cfg.num_processes))

        self._init_robot_handle()
        self._init_buffers()
        self._init_motion_lib()""",
        ),
        Highlight(
            file=robot_cfg,
            content="""
The selected robot motion library uses the algorithm's sampling strategy knobs, supports weighted-bin and curriculum subconfigs, writes optional sampled-key diagnostics, bounds windows to 50-300 frames, normalizes world frame, and explicitly selects the `hdf5_v2` backend with train and validation roots resolved from the training config.""",
            text="""
  motion:
    asset:
      assetRoot: "./"
      assetFileName: "assets/robots/${robot.humanoid_type}/g1_29dof_rev_1_0.xml"

    sampling_strategy: ${algo.config.sampling_strategy}
    weighted_bin: ${algo.config.weighted_bin}
    curriculum: ${algo.config.curriculum}
    dump_sampled_motion_keys: false
    dump_sampled_motion_keys_interval: 1
    dump_sampled_motion_keys_dir: "sampled_motion_cache_keys"

    max_frame_length: 300 # 6s
    min_frame_length: 50 # 1s
    handpicked_motion_names: null
    excluded_motion_names: null

    world_frame_normalization: true

    backend: "hdf5_v2" # hdf5, hdf5_v2
    train_hdf5_roots: ${train_hdf5_roots}
    val_hdf5_roots: ${train_hdf5_roots}""",
        ),
        Highlight(
            file=robot_cfg,
            content="""
The cache is sized to one motion clip per environment, stages on CUDA by default, swaps after `max_frame_length` command steps, and permits both raw `ref_` and filtered `ft_ref_` reference tensors. The swap interval equals 300 under the default motion config, so cache replacement happens at rollout barriers rather than every environment step.""",
            text="""
    cache:
      batch_progress_bar: false
      max_num_clips: ${num_envs} # Batch size for motion clips
      device: "cuda" # "cuda" or "cpu"; cuda stages on GPU
      swap_interval_steps: ${robot.motion.max_frame_length} # Swap cache every N steps
      allowed_prefixes:
        - "ref_"
        - "ft_ref_\"""",
        ),
        Highlight(
            file=motion_command,
            content="""
For `hdf5_v2`, the command builds datasets through `build_motion_datasets_from_cfg`, rejects an empty training set, and then creates the `MotionClipBatchCache` with the resolved train/val datasets, cache batch size, staging device, dataloader worker settings, distributed sampler rank/world size, allowed prefixes, swap interval, seed, timeout, and hdf5-v2-specific cache kwargs.""",
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
        Highlight(
            file=motion_command,
            content="""
After cache construction, the command selects the configured sampling strategy. The default `uniform` path is accepted without extra setup; if the config switches to `weighted_bin`, regex-bin sampling is enabled; if it switches to `curriculum`, cache-curriculum prioritized sampling is enabled. Any other strategy fails fast.""",
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
        Highlight(
            file=algo_cfg,
            content="""
The scoped default algorithm config sets `sampling_strategy: uniform`; weighted-bin and curriculum parameters are present but not active unless the run overrides the strategy. Thus this training job randomizes clip/window starts uniformly, while the code path for weighted-bin or curriculum is grounded above and below.""",
            text="""
    # --- Sampling Strategy ---
    sampling_strategy: uniform
    curriculum:
      p_a_ratio: 0.5
      ema_alpha_signal: 0.2
      ema_alpha_rel_improve: 0.2
      relative_eps: 1.0e-6
      dump_whole_window_scores_json: false
      dump_whole_window_scores_every_swaps: 10

    weighted_bin:
      bin_regex_patterns: []
      dump_sampled_keys: false
      dump_sampled_keys_interval: 1000""",
        ),
        Highlight(
            file=h5_dataloader,
            content="""
The HDF5 v2 dataset enumerates contiguous windows from manifest clips. Each source clip is split into windows no longer than `max_frame_length`; windows shorter than `min_window_length` are skipped; each window records shard index, absolute start, length, raw motion key, and a unique `motion_key__start_<frame>_len_<length>` identifier.""",
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
        Highlight(
            file=h5_dataloader,
            content="""
HDF5 v2 reads the minimal root/DoF representation directly from each shard: mandatory `ref_root_pos`, `ref_root_rot`, and `ref_dof_pos`; optional `frame_flag`; plus scalar motion FPS and sampled filter cutoff metadata. The code rejects missing mandatory datasets and invalid FPS, so malformed HDF5 references fail before training can consume them.""",
            text="""
        for dataset_name in ("ref_root_pos", "ref_root_rot", "ref_dof_pos"):
            if dataset_name not in shard_handle:
                raise KeyError(
                    f"Missing mandatory dataset '{dataset_name}' in shard index "
                    f"{window.shard_index}"
                )
            np_array = np.asarray(shard_handle[dataset_name][start:end, ...])
            arrays[dataset_name] = self._cast_motion_np(np_array, dataset_name)""",
        ),
        Highlight(
            file=h5_dataloader,
            content="""
The dataset materializes full reference tensors from the root/DoF HDF5 fields by running FK, optional filtered FK, optional world-frame normalization, and root-state derivation. That is why later command getters can read `ref_dof_pos`, `ref_dof_vel`, `ref_rg_pos`, `ref_rb_rot`, `ref_body_vel`, `ref_body_ang_vel`, `ref_root_pos`, `ref_root_rot`, `ref_root_vel`, and `ref_root_ang_vel` from the cache.""",
            text="""
        self._fk_transform(
            arrays,
            motion_fps,
            vel_smoothing_sigma=self._ref_vel_smoothing_sigma,
        )
        if self._online_filter_enabled and "ft_ref_" in self._allowed_prefixes:
            self._add_online_filtered_reference_tensors(
                arrays,
                motion_fps,
                cutoff_hz,
            )
        if self._world_frame_transform is not None:
            self._world_frame_transform(arrays)

        self._derive_root_state_tensors(arrays, prefix="ref_")
        self._derive_root_state_tensors(arrays, prefix="ft_ref_")""",
        ),
        Highlight(
            file=h5_dataloader,
            content="""
Cache collation pads every sampled window to the batch's max frame length and repeats the last valid frame after `sample.length`. This makes future-frame gathers safe at the end of a shorter window while preserving `lengths` as the authoritative end condition for resampling.""",
            text="""
                target = batched_tensors[name]
                valid_frames = sample.length
                target[batch_idx, :valid_frames] = tensor

                if valid_frames < max_frame_length and valid_frames > 0:
                    target[batch_idx, valid_frames:] = tensor[valid_frames - 1]""",
        ),
        Highlight(
            file=h5_dataloader,
            content="""
Training assignment sampling first picks random cache rows, then computes the latest legal start frame that leaves room for `n_future_frames`. Evaluation passes `deterministic_start=True` and starts at frame zero; training leaves it false and samples a uniform integer start in `[0, max_start]`.""",
            text="""
        clip_indices = torch.randint(
            low=0, high=total, size=(num_envs,), device=device
        )

        max_start = torch.clamp(
            lengths[clip_indices] - 1 - n_future_frames, min=0
        )
        if deterministic_start:
            frame_starts = torch.zeros_like(max_start)
        else:
            rand = torch.rand_like(max_start, dtype=torch.float32)
            frame_starts = torch.floor(rand * (max_start + 1).float()).to(
                torch.long
            )

        return clip_indices, frame_starts""",
        ),
        Highlight(
            file=motion_command,
            content="""
Initial per-environment command assignment happens after cache construction. Every env gets a sampled clip row and frame index, evaluation uses deterministic frame-zero starts, reward/step accumulators are cleared, and reference access is refreshed through the cache-backed command getters.""",
            text="""
        # Initial assignment
        clip_idx, frame_idx = self._motion_cache.sample_env_assignments(
            self.num_envs,
            self.cfg.n_fut_frames,
            self.device,
            deterministic_start=(self._is_evaluating),
        )
        self._clip_indices[:] = clip_idx
        self._frame_indices[:] = frame_idx
        self._start_frame_indices[:] = frame_idx
        self._reward_sum_since_assign[:] = 0.0
        self._step_count_since_assign[:] = 0.0
        self._update_ref_motion_state_from_cache()""",
        ),
        Highlight(
            file=motion_command,
            content="""
Reference tensors are not copied into persistent per-env buffers; the current `_clip_indices` and `_frame_indices` are the state. `_get_ref_state_array` resolves the requested prefixed tensor key in the current cache batch and asks the cache to gather the current plus future frames for every environment.""",
            text="""
        batch_tensors = self._motion_cache.current_batch.tensors
        tensor_key = resolve_reference_tensor_key(
            batch_tensors=batch_tensors,
            base_key=base_key,
            prefix=prefix,
        )
        return self._motion_cache.gather_tensor(
            tensor_key,
            clip_indices=self._clip_indices,
            frame_indices=self._frame_indices,
            n_future_frames=self.cfg.n_fut_frames,
        )""",
        ),
        Highlight(
            file=h5_dataloader,
            content="""
The cache gather operation builds a `[current + future]` timestep range per environment, clamps timesteps to each selected clip's last valid frame, and indexes the batched tensor by selected clip row and timestep. This is the exact loop mechanism by which frame indices become reference tensors for observations, rewards, resets, and evaluation.""",
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
        Highlight(
            file=motion_command,
            content="""
The current command observation is exactly the current reference DoF position concatenated with current reference DoF velocity in simulator DoF order. Because the G1 has 29 DoFs, `bydmmc_ref_motion` is 58 values per environment; `command_fut` applies the same schema to future frames.""",
            text="""
    def _get_obs_bydmmc_ref_motion(
        self,
        obs_prefix: str = "ref_",
    ) -> torch.Tensor:
        base_pos = self._get_ref_state_array("dof_pos", obs_prefix)[:, 0, ...][
            ..., self.urdf2sim_dof_idx
        ]
        base_vel = self._get_ref_state_array("dof_vel", obs_prefix)[:, 0, ...][
            ..., self.urdf2sim_dof_idx
        ]
        num_envs = base_pos.shape[0]
        cur_ref_dof_pos_flat = base_pos.reshape(num_envs, -1)
        cur_ref_dof_vel_flat = base_vel.reshape(num_envs, -1)
        return torch.cat([cur_ref_dof_pos_flat, cur_ref_dof_vel_flat], dim=-1)""",
        ),
        Highlight(
            file=motion_command,
            content="""
The command advances the reference one frame per active environment step, but skips just-reset envs whose episode length buffer is zero. When the cache swap counter reaches the configured swap interval, it marks a pending swap; actual cache replacement is deferred to the PPO rollout barrier. After advancing, end-of-window environments are resampled and reference access is refreshed.""",
            text="""
    def _update_command(self):
        all_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        motion_ids = self._filter_env_ids_for_motion_task(all_ids)
        if motion_ids.numel() == 0:
            return

        continue_ids = motion_ids
        episode_length_buf = getattr(self._env, "episode_length_buf", None)
        if episode_length_buf is not None:
            continue_mask = episode_length_buf[motion_ids] != 0
            continue_ids = motion_ids[continue_mask]
        if continue_ids.numel() > 0:
            self._frame_indices[continue_ids] += 1
        self._swap_step_counter += 1

        if self._swap_step_counter >= self._motion_cache.swap_interval_steps:
            self._swap_pending = True

        # Resample when motion ends
        self._resample_when_motion_end_cache()
        self._update_ref_motion_state_from_cache()""",
        ),
        Highlight(
            file=motion_command,
            content="""
End-of-window resampling is length-aware. It computes each selected clip's latest legal frame as `length - 1 - n_fut_frames`; envs past that bound are assigned a new cache row/start, realigned to the new reference root and DoF state, and marked in `motion_end_mask` for termination logic.""",
            text="""
        lengths = self._motion_cache.lengths_for_indices(self._clip_indices)
        max_valid_frame = torch.clamp(
            lengths - 1 - self.cfg.n_fut_frames, min=0
        )
        need_resample = (
            self._frame_indices[motion_ids] > max_valid_frame[motion_ids]
        )

        if torch.any(need_resample):
            resample_ids = motion_ids[torch.nonzero(need_resample).squeeze(-1)]
            # Resample these envs
            self._record_completion_rate_for_envs(resample_ids)
            clip_idx, frame_idx = self._motion_cache.sample_env_assignments(
                len(resample_ids),
                self.cfg.n_fut_frames,
                self.device,
                deterministic_start=self._is_evaluating,
            )""",
        ),
        Highlight(
            file=motion_command,
            content="""
Reset-time resampling records completion for the old assignment, samples a new cache row/start with deterministic starts only in evaluation, writes the clip/frame/start indices, clears assignment accumulators, refreshes cache-backed reference access, and applies the reference to root and DoF simulator state.""",
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
        Highlight(
            file=motion_command,
            content="""
Root alignment applies the command to simulation. It reads current reference root position, orientation, linear velocity, and angular velocity, converts the XYZW reference quaternion to WXYZ for IsaacLab, samples bounded root pose/velocity perturbations from config, applies them only to the reset envs, and writes the concatenated root state to the robot.""",
            text="""
        root_pos = self.get_ref_motion_root_global_pos_cur().clone()
        root_rot_xyzw = self.get_ref_motion_root_global_rot_quat_xyzw_cur()
        root_rot = root_rot_xyzw[..., [3, 0, 1, 2]].clone()
        root_lin_vel = self.get_ref_motion_root_global_lin_vel_cur().clone()
        root_ang_vel = self.get_ref_motion_root_global_ang_vel_cur().clone()

        pos_rot_range_list = [
            self.cfg.root_pose_perturb_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        pos_rot_ranges = torch.tensor(pos_rot_range_list, device=self.device)
        pos_rot_rand_deltas = isaaclab_math.sample_uniform(
            pos_rot_ranges[:, 0],
            pos_rot_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )""",
        ),
        Highlight(
            file=motion_command,
            content="""
DoF alignment applies the command to all simulator joints. It reads current reference DoF position/velocity in simulator order, samples the configured DoF-position perturbation, clips reset envs to soft joint limits, and writes joint position/velocity to the articulation.""",
            text="""
        dof_pos = self.get_ref_motion_dof_pos_cur().clone()
        dof_vel = self.get_ref_motion_dof_vel_cur().clone()

        dof_pos += isaaclab_math.sample_uniform(
            *self.cfg.dof_pos_perturb_range,
            dof_pos.shape,
            dof_pos.device,
        )
        soft_dof_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        dof_pos[env_ids] = torch.clip(
            dof_pos[env_ids],
            soft_dof_pos_limits[:, :, 0],
            soft_dof_pos_limits[:, :, 1],
        )

        self.robot.write_joint_state_to_sim(
            dof_pos[env_ids],
            dof_vel[env_ids],
            env_ids=env_ids,
        )""",
        ),
        Highlight(
            file=observation_impl,
            content="""
Reference motion observations are not separate duplicated logic: observation functions fetch the live `ref_motion` command term and call the configured command observation getter. The actor/critic therefore observe the same cache-backed current/future tensors that reset and reward code use.""",
            text="""
    def _get_obs_ref_motion_states(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ):
        \"\"\"Reference motion states (flattened) via RefMotionCommand schema.\"\"\"
        command = env.command_manager.get_term(ref_motion_command_name)
        obs_fn_name = f"_get_obs_{command.cfg.command_obs_name}"
        obs_fn = getattr(command, obs_fn_name)
        return obs_fn(obs_prefix=ref_prefix)""",
        ),
        Highlight(
            file=reward_impl,
            content="""
Rewards also read the command term directly. For DoF-position tracking, the target is the immediate-next reference frame, not the stale current frame; it compares robot joint positions to `get_ref_motion_dof_pos_immediate_next` and returns an exponential tracking reward.""",
            text="""
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keydof_idxs = _get_dof_indices(command.robot, key_dofs)
    ref_dof_pos = command.get_ref_motion_dof_pos_immediate_next(
        prefix=ref_prefix
    )
    error = torch.sum(
        torch.square(
            command.robot.data.joint_pos[:, keydof_idxs]
            - ref_dof_pos[:, keydof_idxs]
        ),
        dim=-1,
    )
    return torch.exp(-error / std**2)""",
        ),
        Highlight(
            file=termination_impl,
            content="""
Termination uses the command's own motion-end mask. Because `_resample_when_motion_end_cache` sets this mask when a reference window ends, the termination manager can end episodes exactly when the command loop reaches the last legal frame.""",
            text="""
def motion_end(
    env: ManagerBasedRLEnv,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    \"\"\"Terminate when reference motion frames exceed their end frames.

    Returns a boolean mask of shape [num_envs].
    \"\"\"
    command: motion_tracking_command.RefMotionCommand = (
        env.command_manager.get_term(command_name)
    )
    result = command.motion_end_mask.clone().bool()
    return result""",
        ),
        Highlight(
            file=algo_base,
            content="""
PPO setup keeps a direct handle to the command term, passes distributed runtime context into it, and then seeds Python, NumPy, Torch, the IsaacLab env, and the motion cache consistently. This makes random clip/start sampling reproducible per rank instead of being an uncontrolled source of nondeterminism.""",
            text="""
        self.command_name = list(self.env.config.commands.keys())[0]
        self.command_term = self.env._env.command_manager.get_term(
            self.command_name
        )
        if self.command_name == "ref_motion":
            self.command_term.set_runtime_distributed_context(
                process_id=int(self.accelerator.process_index),
                num_processes=int(self.accelerator.num_processes),
            )
            self.command_term.setup_dumping_dir(self.log_dir)""",
        ),
        Highlight(
            file=algo_base,
            content="""
Training rewards are fed back into the command after each environment step. The command accumulates reward, MPJPE, MPKPE, and step counts per assigned window, which are later used by the cache curriculum path when that sampling strategy is enabled.""",
            text="""
        if self.command_name != "ref_motion":
            return
        motion_term = self.env._env.command_manager.get_term("ref_motion")
        if motion_term is None:
            return
        motion_term.update_curriculum_reward_accumulators(rewards)""",
        ),
        Highlight(
            file=ppo,
            content="""
Cache swaps are deliberately applied after each PPO iteration, not in the middle of a rollout. This calls the command's barrier method, which updates curriculum state, advances the cache, reassigns motion envs, realigns simulator state, and clears the pending flag.""",
            text="""
    def _post_iteration_hook(self, it: int) -> None:
        if self.command_name == "ref_motion":
            motion_cmd = self.env._env.command_manager.get_term("ref_motion")
            motion_cmd.apply_cache_swap_if_pending_barrier(
                accelerator=self.accelerator
            )""",
        ),
        Highlight(
            file=motion_command,
            content="""
At the cache-swap barrier, curriculum state is updated from aggregated per-window rollout statistics before advancing the cache. After advancing, the command samples fresh assignments using deterministic starts only during evaluation, rewrites clip/frame/start indices, refreshes references, and realigns root and DoF state to the new references.""",
            text="""
        self._record_completion_rate_for_envs(motion_ids)
        next_swap_index = int(self._motion_cache.swap_index) + 1
        self._update_cache_curriculum_state(
            accelerator=accelerator,
            swap_index=next_swap_index,
        )

        # Advance cache and reset counters
        self._motion_cache.advance()
        self._maybe_dump_sampled_motion_keys()
        self._swap_pending = False
        self._swap_step_counter = 0""",
        ),
        Highlight(
            file=motion_command,
            content="""
The curriculum update path is concrete: only `sampling_strategy == "curriculum"` pushes window indices, MPKPE signal, completion rate, and counts into the cache; non-curriculum strategies clear stats. In distributed training it gathers those tensors across ranks before updating cache priorities.""",
            text="""
        if self._sampling_strategy != "curriculum":
            self._reset_window_curriculum_stats()
            return

        (
            row_window_indices,
            row_mpkpe_signal,
            row_completion_rate,
            row_count,
        ) = self._build_window_curriculum_stats_from_current_batch()

        if accelerator is not None and int(accelerator.num_processes) > 1:
            gather_window_indices = accelerator.gather(row_window_indices)
            gather_mpkpe_signal = accelerator.gather(row_mpkpe_signal)
            gather_completion_rate = accelerator.gather(row_completion_rate)
            gather_count = accelerator.gather(row_count)""",
        ),
        Highlight(
            file=h5_dataloader,
            content="""
Weighted-bin sampling is regex-based over manifest/raw motion keys. It assigns every motion window to the first matching configured bin or to `others`, validates positive-ratio bins are non-empty, and converts ratios into exact integer cache-batch counts. This is the optional weighted-bin command-sampling mechanism if the config enables it.""",
            text="""
    for idx, motion_key in enumerate(keys):
        assigned = False
        for b_idx, pat in enumerate(compiled_patterns):
            if pat["compiled"].search(motion_key):
                bin_indices[b_idx].append(idx)
                assigned = True
                break
        if not assigned:
            bin_indices[-1].append(idx)

    # Combine explicit ratios with implicit "others" ratio
    all_ratios: List[float] = list(ratios)
    all_ratios.append(others_ratio)""",
        ),
        Highlight(
            file=h5_dataloader,
            content="""
The cache-curriculum sampler prioritizes windows by observed learning signal: priority is positive relative completion improvement times remaining difficulty times whether the window has been seen. Fresh uniform samples and prioritized samples are mixed according to `p_a_ratio` and the prioritized pool size.""",
            text="""
        progress = torch.clamp(
            self._ema_completion_rel_improve.index_select(0, idx),
            min=0.0,
            max=1.0,
        )
        remaining_difficulty = torch.clamp(
            1.0 - self._ema_completion_rate.index_select(0, idx),
            min=0.0,
            max=1.0,
        )
        seen = self._seen_mask.index_select(0, idx).to(dtype=torch.float32)
        return progress * remaining_difficulty * seen""",
        ),
        Highlight(
            file=ppo,
            content="""
Offline evaluation forces deterministic command behavior. It sets the command evaluation flag, switches the cache to validation mode, resets all environments for each validation batch, then calls `setup_offline_eval_deterministic` before recomputing observations.""",
            text="""
        # Evaluation flag and cache batch-size adjustment (ensure batch_size == num_envs)
        motion_cmd._is_evaluating = True
        num_envs = self.env.num_envs
        try:
            if getattr(cache, "_batch_size", None) != num_envs:
                from holomotion.src.training.h5_dataloader import (
                    MotionClipBatchCache,
                )""",
        ),
        Highlight(
            file=ppo,
            content="""
The evaluation loop uses validation batches deterministically: it calls `cache.set_mode("val")`, advances one cache batch per eval batch, resets, assigns env `i` to cache row `i` at frame zero through the command, optionally realigns without perturbation, and recomputes observations from that deterministic state.""",
            text="""
        # Switch to validation cache and iterate all batches
        if hasattr(cache, "set_mode"):
            cache.set_mode("val")
        # Determine policy/video FPS from command config (align wallclock time)
        motion_fps = int(getattr(motion_cmd.cfg, "target_fps", 50))
        total_batches = int(getattr(cache, "num_batches", 1))
        with torch.no_grad():
            for batch_idx in tqdm(
                range(total_batches), desc="Evaluating batches"
            ):
                if batch_idx > 0:
                    cache.advance()
                # Reset envs first, then apply deterministic mapping on the active cache batch
                _ = self.env.reset_all()
                if hasattr(motion_cmd, "setup_offline_eval_deterministic"):
                    motion_cmd.setup_offline_eval_deterministic(
                        apply_pending_swap=False
                    )
                self._reset_rollout_forward_state()""",
        ),
        Highlight(
            file=motion_command,
            content="""
The deterministic evaluation setup is explicit: optional pending swaps are applied, clip/frame indices are zeroed, active envs are mapped one-to-one to cache rows, and reference access is refreshed. The caller then realigns root/DoF without perturbation, so eval starts from exact HDF5 reference frame zero for each clip.""",
            text="""
        clip_count = int(self._motion_cache.clip_count)
        active_count = min(int(self.num_envs), clip_count)

        # Reset indices
        self._clip_indices[:] = 0
        self._frame_indices[:] = 0

        if active_count > 0:
            active_ids = torch.arange(
                active_count, dtype=torch.long, device=self.device
            )
            self._clip_indices[active_ids] = torch.arange(
                active_count, dtype=torch.long, device=self.device
            )

        self._update_ref_motion_state_from_cache()""",
        ),
    ]
)
