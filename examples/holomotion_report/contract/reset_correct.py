from project_analysis import File, Highlight, ProofFromCode

training_cfg = File(
    path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_mlp.yaml"
)
motion_tracking_cfg = File(path="HoloMotion/holomotion/config/env/motion_tracking.yaml")
domain_rand_medium = File(
    path="HoloMotion/holomotion/config/env/domain_randomization/domain_rand_medium.yaml"
)
robot_cfg = File(
    path="HoloMotion/holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"
)
motion_env = File(path="HoloMotion/holomotion/src/env/motion_tracking.py")
motion_command = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_motion_tracking_command.py"
)
h5_dataloader = File(path="HoloMotion/holomotion/src/training/h5_dataloader.py")
ppo = File(path="HoloMotion/holomotion/src/algo/ppo.py")

reset_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=training_cfg,
            content="""
The scoped training entry point is the G1 29-DOF motion-tracking MLP job: it composes the motion-tracking environment, motion-tracking terminations, motion-tracking observations and rewards, medium domain randomization, rough IsaacLab terrain, and the motion-tracking MLP module. Therefore the reset proof is about motion imitation on the Unitree G1 task, not the velocity-tracking environment.""",
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
            file=motion_tracking_cfg,
            content="""
The motion command is the reset-critical command term. It receives the robot motion library, URDF/IsaacLab DOF and body naming, anchor body, root and DOF perturbation ranges from domain randomization, a long resample interval, future-frame horizon, and a 50 Hz target FPS. This is the configuration link that makes reset motion-aware rather than a generic zero-state reset.""",
            text="""
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
            file=domain_rand_medium,
            content="""
Training resets are intentionally not exact reference copies: medium domain randomization gives bounded root position/orientation perturbations, bounded root velocity perturbations, and bounded DOF-position perturbations while leaving DOF velocity perturbation at zero. These ranges keep the reset close enough to the reference motion for stable imitation but varied enough for robustness.""",
            text="""
  motion_init_perturb:
    root_pose_perturb_range:
      x: [-0.05, 0.05]
      y: [-0.05, 0.05]
      z: [-0.01, 0.01]
      roll: [-0.1, 0.1]
      pitch: [-0.1, 0.1]
      yaw: [-0.2, 0.2]
    root_vel_perturb_range:
      x: [-0.5, 0.5]
      y: [-0.5, 0.5]
      z: [-0.2, 0.2]
      roll: [-0.5, 0.5]
      pitch: [-0.5, 0.5]
      yaw: [-0.2, 0.2]
    dof_pos_perturb_range: [-0.1, 0.1]
    dof_vel_perturb_range: [0.0, 0.0]""",
        ),
        Highlight(
            file=robot_cfg,
            content="""
The robot motion library uses the configured algorithm sampling strategy, optional weighted-bin or curriculum settings, HDF5 v2 training and validation roots, bounded clip windows, and a cache with one clip batch per environment. Reset-time command sampling is therefore backed by real motion windows loaded from the training HDF5 dataset.""",
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
            file=motion_env,
            content="""
The HoloMotion wrapper exposes both indexed and all-env reset entry points, and both delegate to the underlying `ManagerBasedRLEnv.reset`. This is where IsaacLab's standard episode bookkeeping and manager reset order are entered; HoloMotion's own reset-specific guarantees are then supplied by the configured command term below.""",
            text="""
    def reset_idx(self, env_ids: torch.Tensor):
        return self._env.reset(env_ids=env_ids)

    def reset_all(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        out = self._env.reset(env_ids=env_ids)
        return out""",
        ),
        Highlight(
            file=motion_command,
            content="""
The command-term reset is motion-aware. It runs the base command reset, normalizes env ids onto the command device, clears motion-end masks and per-env motion-end counters, filters to envs assigned to the motion-tracking task, and resamples only those envs. This prevents non-motion task ids from receiving motion state writes and makes reset local to the requested environments.""",
            text="""
    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, float]:
        extras = super().reset(env_ids)

        if env_ids is None:
            env_ids = slice(None)

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(self.device)
        self._motion_end_mask[env_ids] = False
        self.motion_end_counter[env_ids] = 0

        # Do not apply cache swap inside per-env reset; defer to PPO barrier.
        # Always resample only the requested envs here.
        motion_ids = self._filter_env_ids_for_motion_task(env_ids.view(-1))
        self._resample_command(motion_ids, eval=self._is_evaluating)

        return extras""",
        ),
        Highlight(
            file=motion_command,
            content="""
The resample command records the previous assignment's completion, samples a new motion clip and start frame, writes the new clip/frame/start indices, clears per-assignment reward and step accumulators, refreshes reference access, and realigns both root and DOF state. This is the core reset chain that initializes a stable per-episode motion state.""",
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
            file=h5_dataloader,
            content="""
Reset-time starts are randomized during training and deterministic during evaluation. The cache samples a random clip index from the current batch; for each selected clip it computes the latest valid start that still leaves room for future frames, then either picks frame zero in deterministic mode or samples a uniform frame start in `[0, max_start]` for training.""",
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
Root reset state comes directly from the current reference motion frame: root position, orientation, linear velocity, and angular velocity are cloned from reference getters. Bounded pose and velocity deltas are sampled from the configured ranges and applied before writing a complete root state tensor to simulation.""",
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
The root write is complete and physically scoped: translation and orientation deltas are applied only to reset env ids, velocity deltas are applied to linear and angular velocity, and the final concatenated root position, WXYZ rotation, linear velocity, and angular velocity are written through IsaacLab's articulation API for exactly those env ids.""",
            text="""
        root_pos[env_ids] += translation_delta
        root_rot[env_ids] = isaaclab_math.quat_mul(
            rotation_delta,
            root_rot[env_ids],
        )

        lin_ang_vel_range_list = [
            self.cfg.root_vel_perturb_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        lin_ang_vel_ranges = torch.tensor(
            lin_ang_vel_range_list, device=self.device
        )

        lin_ang_vel_rand_deltas = isaaclab_math.sample_uniform(
            lin_ang_vel_ranges[:, 0],
            lin_ang_vel_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        root_lin_vel[env_ids] += lin_ang_vel_rand_deltas[:, :3]
        root_ang_vel[env_ids] += lin_ang_vel_rand_deltas[:, 3:]

        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos[env_ids],
                    root_rot[env_ids],
                    root_lin_vel[env_ids],
                    root_ang_vel[env_ids],
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )""",
        ),
        Highlight(
            file=motion_command,
            content="""
Joint reset state is also reference-driven. It clones current reference DOF positions and velocities, adds bounded position noise, clips reset positions to each environment's soft joint limits, and writes joint position/velocity to simulation for the reset env ids. The clipping is the key stability guard against randomization pushing joints outside soft limits.""",
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
            file=motion_command,
            content="""
Immediately after reset, command advancement is guarded by `episode_length_buf`: envs whose episode length is zero do not advance their reference frame in `_update_command`. That avoids an off-by-one jump on the first post-reset command update and keeps the freshly sampled reference frame aligned with the root and joint state just written.""",
            text="""
        continue_ids = motion_ids
        episode_length_buf = getattr(self._env, "episode_length_buf", None)
        if episode_length_buf is not None:
            continue_mask = episode_length_buf[motion_ids] != 0
            continue_ids = motion_ids[continue_mask]
        if continue_ids.numel() > 0:
            self._frame_indices[continue_ids] += 1""",
        ),
        Highlight(
            file=motion_command,
            content="""
The reset-related buffers are initialized conservatively: motion frame ids, motion-end masks and counters, cache indices, start-frame indices, reward and error accumulators, step counters, completion statistics, and transient history/acceleration buffers all start from zeros or `None`. The reset path then clears the per-env motion-end mask/counter and assignment accumulators before writing a new state.""",
            text="""
        self._start_frame_indices = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._reward_sum_since_assign = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._mpjpe_sum_since_assign = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._mpkpe_sum_since_assign = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._step_count_since_assign = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._completion_rate_sum_by_window: Dict[int, float] = {}
        self._completion_rate_count_by_window: Dict[int, int] = {}
        self._mpkpe_signal_sum_by_window: Dict[int, float] = {}
        self._mpkpe_signal_count_by_window: Dict[int, int] = {}

        self.pos_history_buffer = None
        self.rot_history_buffer = None
        self.ref_pos_history_buffer = None
        self.current_accel = None
        self.ref_body_accel = None
        self.current_ang_accel = None  # Placeholder for angular acceleration""",
        ),
        Highlight(
            file=motion_command,
            content="""
Curriculum sampling is optional and mutually exclusive with weighted-bin sampling. When the robot motion config selects `sampling_strategy: curriculum`, the cache enables a curriculum sampler from the configured algorithm parameters; otherwise the default `uniform` strategy remains active. This is the concrete "if present" path for reset-time curriculum sampling.""",
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
When cache curriculum is enabled, it builds a mixed prioritized/uniform batch: prioritized indices are sampled, fresh uniform indices are sampled while excluding prioritized picks, any shortfall is filled uniformly, and the sampler rejects incomplete batches. That gives reset resampling a curriculum source without losing coverage of fresh motion windows.""",
            text="""
    def _sample_batch_indices(self, generator: torch.Generator) -> Tensor:
        uniform_count, prioritized_count = self._pool_batch_sizes()
        prioritized_indices = self._sample_prioritized_indices(
            generator, prioritized_count
        )
        uniform_indices = self._sample_uniform_indices(
            generator,
            uniform_count,
            exclude=prioritized_indices,
        )
        sampled_indices = torch.cat(
            [uniform_indices, prioritized_indices], dim=0
        )
        if sampled_indices.numel() < self._batch_size:
            extra_indices = self._sample_uniform_indices(
                generator,
                self._batch_size - int(sampled_indices.numel()),
                exclude=sampled_indices,
            )
            sampled_indices = torch.cat(
                [sampled_indices, extra_indices], dim=0
            )
        if sampled_indices.numel() != self._batch_size:
            raise ValueError(
                "Prioritized sampler failed to assemble a full cache batch."
            )""",
        ),
        Highlight(
            file=ppo,
            content="""
The offline deterministic evaluation path explicitly resets observation history for active envs before recomputing observations with `update_history=True` after forced no-perturb reference realignment. This separate path confirms HoloMotion treats observation history as reset-sensitive state when it needs a deterministic starting condition.""",
            text="""
                # Recompute observations after deterministic setup
                obs_mgr = self.env._env.observation_manager
                if active_count > 0:
                    obs_mgr.reset(active_ids)
                    obs_dict = obs_mgr.compute(update_history=True)
                else:
                    obs_dict = obs_mgr.compute(update_history=True)
                obs = self._wrap_obs_dict(obs_dict)""",
        ),
    ]
)
