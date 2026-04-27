from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

motion_env_cfg = File(path="HoloMotion/holomotion/config/env/motion_tracking.yaml")
robot_cfg = File(
    path="HoloMotion/holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"
)
train_tf_moe_cfg = File(
    path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_tf-moe.yaml"
)
algo_cfg = File(path="HoloMotion/holomotion/config/algo/ppo.yaml")
motion_command = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_motion_tracking_command.py"
)
h5_dataloader = File(path="HoloMotion/holomotion/src/training/h5_dataloader.py")
motion_env = File(path="HoloMotion/holomotion/src/env/motion_tracking.py")
obs_cfg = File(
    path="HoloMotion/holomotion/config/env/observations/motion_tracking/obs_motion_tracking_tf-moe.yaml"
)
obs_impl = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_observation.py"
)
rew_cfg = File(
    path="HoloMotion/holomotion/config/env/rewards/motion_tracking/rew_motion_tracking.yaml"
)
ppo = File(path="HoloMotion/holomotion/src/algo/ppo.py")
velocity_env_cfg = File(path="HoloMotion/holomotion/config/env/velocity_tracking.yaml")
velocity_command = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_velocity_tracking_command.py"
)

command_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=train_tf_moe_cfg,
            content="""
The TF-MoE motion-tracking run is not a synthetic-command task. Its defaults select `/env: motion_tracking`, the motion-tracking observations/rewards, and the motion-tracking module, then provide `train_hdf5_roots` as the source of reference motions.""",
            text="""
defaults:
  - /training: train_base
  - /algo: ppo_tf
  - /robot: unitree/G1/29dof/29dof_training_isaaclab
  - /env: motion_tracking
  - /env/terminations: termination_motion_tracking
  - /env/observations: motion_tracking/obs_motion_tracking_tf-moe
  - /env/rewards: motion_tracking/rew_motion_tracking
  - /env/domain_randomization: domain_rand_medium
  - /env/terrain: isaaclab_rough
  - /modules: motion_tracking/motion_tracking_tf-moe""",
        ),
        Highlight(
            file=train_tf_moe_cfg,
            content="""
The concrete demo training config points motion loading at `data/h5v2_datasets/AMASS_test`. In other words, the motion command samples windows from a packed HDF5 motion dataset, not from a procedurally generated pose command.""",
            text="""
train_hdf5_roots:
  - data/h5v2_datasets/AMASS_test""",
        ),
        Highlight(
            file=motion_env_cfg,
            content="""
The command term registered for the motion-tracking environment is named `ref_motion`, has type `MotionCommandCfg`, and is configured to expose the command observation as `bydmmc_ref_motion`. It receives the robot's motion library config, body/joint name mappings, perturbation ranges, a long command resample interval, future-frame count, and target FPS.""",
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
            file=robot_cfg,
            content="""
The `robot.motion` object passed into `ref_motion` is where the data backend and sampling policy live. It takes `sampling_strategy` and curriculum knobs from PPO config, sets HDF5 v2 as the backend, and uses the training roots from the training config for both train and validation unless overridden.""",
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
            file=algo_cfg,
            content="""
The default sampling mode is uniform. A cache-level curriculum exists, but it is only activated if `algo.config.sampling_strategy` is changed to `curriculum`; the environment-level `curriculum.enabled: false` in `motion_tracking.yaml` is separate domain/reward curriculum, not the motion-window sampler.""",
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
            file=motion_command,
            content="""
At runtime `RefMotionCommand` materializes the HDF5 v2 pipeline by calling `build_motion_datasets_from_cfg(...)`. The resulting train/val datasets plus cache kwargs are fed into `MotionClipBatchCache`, so command data enters IsaacLab through a batch cache owned by the command term.""",
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
            )""",
        ),
        Highlight(
            file=motion_command,
            content="""
The command cache is seeded with the distributed process id/world size and the configured seed. This is the training-time data loader/cache that all later per-env random command assignments sample from.""",
            text="""
            self._motion_cache = MotionClipBatchCache(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=int(cache_cfg.get("max_num_clips", 1024)),
                stage_device=stage_device,
                num_workers=int(dataloader_cfg.get("num_workers", 4)),
                prefetch_factor=dataloader_cfg.get("prefetch_factor", None),
                pin_memory=bool(dataloader_cfg.get("pin_memory", True)),
                persistent_workers=bool(
                    dataloader_cfg.get("persistent_workers", True)
                ),
                batch_progress_bar=bool(
                    cache_cfg.get("batch_progress_bar", False)
                ),
                sampler_rank=int(self.cfg.process_id),
                sampler_world_size=int(self.cfg.num_processes),
                allowed_prefixes=allowed_prefixes,
                swap_interval_steps=int(
                    cache_cfg.get("swap_interval_steps", max_frame_length)
                ),
                seed=int(self.cfg.seed),
                loader_timeout=float(dataloader_cfg.get("timeout", 0.0)),
                **cache_kwargs,
            )""",
        ),
        Highlight(
            file=motion_command,
            content="""
The command term supports three sampling policies: uniform, weighted-bin, and cache curriculum. Uniform is the default; weighted-bin changes cache batches by regex-defined bins; curriculum changes cache batches by online rollout performance.""",
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
            )

        self._sampling_strategy = sampling_strategy""",
        ),
        Highlight(
            file=h5_dataloader,
            content="""
HDF5 v2 does not treat an original motion clip as one indivisible command. `Hdf5RootDofDataset._enumerate_windows` splits every manifest clip into contiguous windows capped by `max_frame_length`; windows shorter than `min_window_length` are dropped. With the default G1 motion config, that means up to 300-frame/6-second windows and at least 50-frame/1-second windows at 50 Hz.""",
            text="""
    def _enumerate_windows(self) -> List[MotionWindow]:
        windows: List[MotionWindow] = []
        for motion_key, meta in self.clips.items():
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
                    window_index += 1
                offset += window_length
                remaining = max(0, length - offset)""",
        ),
        Highlight(
            file=h5_dataloader,
            content="""
Each sampled window reads only root position, root rotation, and DoF position from HDF5 v2, then derives the richer reference tensors by FK. This matters for command correctness: observations/rewards later see a complete reference state generated consistently from the sampled window, not independent random signals.""",
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
HDF5 v2 then runs FK, optional online filtering, world-frame normalization, and root-state derivation before returning the `MotionClipSample`. Thus the sampled command contains current/future root, body, velocity, angular velocity, DoF, and metadata tensors derived from one coherent clip window.""",
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
Uniform/random assignment happens in `sample_env_assignments`: each environment chooses a clip row from the current cache batch with `torch.randint`, then chooses a start frame uniformly from all frames that still leave room for the configured future frames. Evaluation flips `deterministic_start=True`, which pins the start frame to zero but still uses the current batch's clip rows.""",
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
            )""",
        ),
        Highlight(
            file=motion_command,
            content="""
At command initialization, the first per-env cache assignment is immediately written into `_clip_indices`, `_frame_indices`, and `_start_frame_indices`. Those per-env indices are the live command identity and phase used by all current/future reference accessors.""",
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
On environment reset or motion end, `RefMotionCommand` resamples only the affected motion-tracking envs and then realigns the simulator root and joints to the sampled reference. The command is therefore not only an observation target; it also determines reset initial state for RL rollouts.""",
            text="""
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
            file=h5_dataloader,
            content="""
When observations or rewards request a reference tensor, the cache gathers `1 + n_future_frames` timesteps from each env's selected clip and current frame. Future timesteps are clamped at the clip length, so the command remains valid at the tail of short windows.""",
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
The command term's reference accessor resolves `ref_` versus `ft_ref_` prefixes against the active cache batch, then delegates to `gather_tensor` using the per-env clip/frame indices. This is the central bridge from sampled command to every reference observation/reward helper.""",
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
            file=obs_cfg,
            content="""
Actor observations directly include current and future reference command terms. In the TF-MoE config the actor sees current reference gravity/base velocity/base angular velocity/DoF/root height plus future reference DoF/root height/gravity/base velocity/base angular velocity.""",
            text="""
  obs_groups:
    unified:
      atomic_obs_list:
        - actor_ref_gravity_projection_cur:
            func: ref_gravity_projection_cur
            history_length: ${obs.context_length}
            flatten_history_dim: false
            params:
              ref_prefix: ${obs.actor_obs_prefix}
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_ref_gravity_projection_cur.n_min}
                n_max: ${domain_rand.obs_noise.actor_ref_gravity_projection_cur.n_max}

        - actor_ref_gravity_projection_fut:
            func: ref_gravity_projection_fut
            params:
              ref_prefix: ${obs.actor_obs_prefix}""",
        ),
        Highlight(
            file=obs_impl,
            content="""
Observation builders do not duplicate sampling. They fetch the `ref_motion` command term from IsaacLab's command manager and call its reference accessors. This proves the policy's input is the same command selected by the cache sampler.""",
            text="""
        command = env.command_manager.get_term(ref_motion_command_name)
        return command.get_ref_motion_dof_pos_cur(prefix=ref_prefix)""",
        ),
        Highlight(
            file=obs_impl,
            content="""
Future-reference observation follows the same path: get the `ref_motion` term, ask it for future DoF positions, optionally reshape, and return it to the observation manager. The sampled command's future frames therefore condition the actor during rollout.""",
            text="""
        command = env.command_manager.get_term(ref_motion_command_name)
        dof_pos_fut = command.get_ref_motion_dof_pos_fut(
            prefix=ref_prefix
        )  # [B, T, D(sim)]
        dof_pos_fut = ObservationFunctions._slice_future_frames(
            dof_pos_fut,
            num_frames=num_frames,
            obs_name="ref_dof_pos_fut",
        )
        return dof_pos_fut""",
        ),
        Highlight(
            file=rew_cfg,
            content="""
Motion-tracking rewards are also command-relative: the reward config uses `reward_prefix: "ref_"` and passes that prefix into root/body tracking reward terms. The sampled motion command is therefore the target that the PPO reward optimizes against.""",
            text="""
rewards:
  _config:
    reward_prefix: "ref_"

  is_alive:
    weight: 0.5
    params: {}

  root_pos_xy_tracking_exp:
    weight: 1.0
    params:
      std: 0.2
      ref_prefix: ${rewards._config.reward_prefix}

  root_rot_tracking_exp:
    weight: 0.5
    params:
      std: 0.4
      ref_prefix: ${rewards._config.reward_prefix}""",
        ),
        Highlight(
            file=motion_command,
            content="""
Every simulation step, command time advances by incrementing `_frame_indices` for active motion envs. The command can request a cache swap after `swap_interval_steps`, and it resamples any env whose selected window has run out of valid frames.""",
            text="""
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
            file=ppo,
            content="""
PPO applies pending cache swaps only after an iteration/rollout barrier. This avoids the command dataset changing midway through a rollout storage window; sampled commands remain stable while transitions are being collected.""",
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
The cache-curriculum path is closed-loop: before a pending swap, the command records completion/MPKPE statistics for active envs, updates the cache curriculum sampler, advances the cache, and then reassigns every motion env to the new curriculum-shaped batch. This is how curriculum-generated command sampling, when enabled, actually affects future RL rollouts.""",
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
        self._swap_step_counter = 0

        # Reassign motion envs to the new cache batch
        clip_idx, frame_idx = self._motion_cache.sample_env_assignments(
            int(motion_ids.numel()),
            self.cfg.n_fut_frames,
            self.device,
            deterministic_start=(self._is_evaluating),
        )""",
        ),
        Highlight(
            file=h5_dataloader,
            content="""
The curriculum sampler update consumes per-window MPKPE signal and completion-rate means, then refreshes the prefetched batch if sampler state changed. That makes harder/underperforming windows more likely to appear in later command batches, according to `PrioritizedInfiniteSampler` scoring.""",
            text="""
    def update_cache_curriculum(
        self,
        *,
        window_indices: Tensor,
        mpkpe_signal_means: Tensor,
        completion_rate_means: Tensor,
        counts: Tensor,
        swap_index: int,
    ) -> bool:
        if self._cache_curriculum_sampler is None:
            return False
        updated = (
            self._cache_curriculum_sampler.maybe_update_from_observations(
                window_indices=window_indices,
                mpkpe_signal_means=mpkpe_signal_means,
                completion_rate_means=completion_rate_means,
                counts=counts,
                swap_index=swap_index,
            )
        )
        if updated:
            self._refresh_prefetched_batch()
        self._maybe_dump_cache_curriculum_scores_json(swap_index=swap_index)
        return updated""",
        ),
        Highlight(
            file=ppo,
            content="""
Offline evaluation is deterministic over the validation cache rather than random per-env starts: it switches cache mode to `val`, resets, calls deterministic setup, maps env i to clip i in the active batch, and records robot/reference traces. This is the evaluation counterpart to the training sampler.""",
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
                    )""",
        ),
        Highlight(
            file=velocity_env_cfg,
            content="""
For completeness, HoloMotion also has a synthetic velocity-command task. That task is separate from motion tracking: its config registers `base_velocity`, with ranges and resampling times for uniform velocity commands.""",
            text="""
    commands:
      base_velocity:
        type: HoloMotionUniformVelocityCommandCfg
        params:
          asset_name: robot
          resampling_time_range: [3, 10.0]
          rel_standing_envs: 0.20
          rel_yaw_envs: 0.30  # actual prob for sampled yaw-only is 0.3 * (1-0.2) = 0.24
          rel_heading_envs: 1.0
          heading_command: false
          heading_control_stiffness: 0.5
          debug_vis: true
          ranges:
            lin_vel_x: [-0.6, 1.0]
            lin_vel_y: [-0.5, 0.5]
            ang_vel_z: [-1.0, 1.0]
            heading: [-3.14, 3.14]""",
        ),
        Highlight(
            file=velocity_command,
            content="""
The velocity task's random command generator samples linear x/y velocity and yaw velocity uniformly, then independently marks yaw-only and standing envs. This is the procedurally random command path, but it belongs to velocity tracking, not the TF-MoE motion-tracking report path.""",
            text="""
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = (
                r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
            )
        self.is_yaw_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.rel_yaw_envs
        )""",
        ),
    ]
)
