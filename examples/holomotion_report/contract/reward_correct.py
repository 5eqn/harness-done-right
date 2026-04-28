from project_analysis import File, Highlight, ProofFromCode


training_cfg = File(
    path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_mlp.yaml"
)
reward_cfg = File(
    path="HoloMotion/holomotion/config/env/rewards/motion_tracking/rew_motion_tracking.yaml"
)
robot_cfg = File(
    path="HoloMotion/holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"
)
env_cfg = File(path="HoloMotion/holomotion/config/env/motion_tracking.yaml")
motion_tracking_env = File(
    path="HoloMotion/holomotion/src/env/motion_tracking.py"
)
scene_builder = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_scene.py"
)
reward_impl = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_rewards.py"
)
motion_command = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_motion_tracking_command.py"
)


reward_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=training_cfg,
            content="""
Scope anchor: the G1 29-DoF motion-tracking MLP training config includes the motion-tracking environment and specifically selects `motion_tracking/rew_motion_tracking` as the reward table. The proof below is therefore for the IsaacLab motion-tracking reward setup, not the separate velocity-tracking reward set.""",
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
            file=reward_cfg,
            content="""
The configured reward table has one alive bonus, six positive motion-reference tracking terms, and three negative regularization/contact terms. The reference terms use the `ref_` tensor prefix and weights/stds: root XY +1.0/std 0.2, root rotation +0.5/std 0.4, root-relative keybody position +1.0/std 0.3, root-relative keybody rotation +2.0/std 0.4, global keybody linear velocity +1.0/std 1.0, and global keybody angular velocity +1.0/std 3.14.""",
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
      ref_prefix: ${rewards._config.reward_prefix}

  root_rel_keybodylink_pos_tracking_l2_exp:
    weight: 1.0
    params:
      keybody_names: ${robot.key_bodies}
      std: 0.3
      ref_prefix: ${rewards._config.reward_prefix}

  root_rel_keybodylink_rot_tracking_l2_exp:
    weight: 2.0
    params:
      keybody_names: ${robot.key_bodies}
      std: 0.4
      ref_prefix: ${rewards._config.reward_prefix}

  global_keybodylink_lin_vel_tracking_l2_exp:
    weight: 1.0
    params:
      keybody_names: ${robot.key_bodies}
      std: 1.0
      ref_prefix: ${rewards._config.reward_prefix}

  global_keybodylink_ang_vel_tracking_l2_exp:
    weight: 1.0
    params:
      keybody_names: ${robot.key_bodies}
      std: 3.14
      ref_prefix: ${rewards._config.reward_prefix}""",
        ),
        Highlight(
            file=reward_cfg,
            content="""
The regularizers are also explicit in the same reward table. `action_rate_l2` penalizes action changes with weight -0.1; `joint_pos_limits` uses `SceneEntityCfg(name=robot, joint_names=.*)` and weight -10.0 to discourage every robot joint from crossing limits; `undesired_contacts` uses the `contact_forces` sensor, threshold 1.0, and a robot body-name regex with weight -0.1 to discourage contacts outside the allowed ankle/wrist links.""",
            text="""
  action_rate_l2:
    weight: -0.1
    params: {}

  # joint_acc_l2:
  #   weight: -1.0e-6
  #   params: {}

  joint_pos_limits:
    weight: -10.0
    params:
      asset_cfg:
        _target_: isaaclab.managers.scene_entity_cfg.SceneEntityCfg
        name: robot
        joint_names:
          - ".*"

  undesired_contacts:
    weight: -0.1
    params:
      threshold: 1.0
      sensor_cfg:
        _target_: isaaclab.managers.scene_entity_cfg.SceneEntityCfg
        name: contact_forces
        body_names:
          - ${robot.undesired_contacts_regrex}""",
        ),
        Highlight(
            file=robot_cfg,
            content="""
The keybody rewards above resolve `${robot.key_bodies}` to these 14 G1 links, and the contact regularizer resolves `${robot.undesired_contacts_regrex}` to a regex that excludes only the ankle-roll and wrist-yaw links from the undesired-contact body set. This grounds which robot parts the body tracking and contact penalties operate on.""",
            text="""
  undesired_contacts_regrex: "^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
  torso_name: "torso_link"
  anchor_body: "torso_link"

  key_bodies:
    - "pelvis"
    - "left_hip_roll_link"
    - "left_knee_link"
    - "left_ankle_pitch_link"
    - "right_hip_roll_link"
    - "right_knee_link"
    - "right_ankle_pitch_link"
    - "torso_link"
    - "left_shoulder_roll_link"
    - "left_elbow_link"
    - "left_wrist_yaw_link"
    - "right_shoulder_roll_link"
    - "right_elbow_link"
    - "right_wrist_yaw_link\"""",
        ),
        Highlight(
            file=env_cfg,
            content="""
The environment config passes the selected reward table through `config.rewards`, configures a single joint-position action term over all robot joints, and defines the `ref_motion` command that reward functions read by default. The `anchor_bodylink_name` is `${robot.anchor_body}`, which the robot config resolves to `torso_link`.""",
            text="""
    robot: ${robot}
    domain_rand: ${domain_rand}
    rewards: ${rewards}
    terrain: ${terrain}
    obs: ${obs}
    terminations: ${terminations}

    simulation:
      episode_length_s: 10 # Long episodes for fluid motion-based termination
      sim_freq: 200
      control_decimation: 4
      physx:
        bounce_threshold_velocity: 0.5
        gpu_max_rigid_patch_count: 327680 # 10 * 2**15

    scene:
      terrain: ${terrain}
      lighting:
        distant_light_intensity: 3000.0
        dome_light_intensity: 1000.0
      contact_sensor:
        history_length: 3
        force_threshold: 10.0
        track_air_time: true
        debug_vis: false

    actions:
      dof_pos:
        type: joint_position
        params:
          asset_name: robot
          joint_names:
            - ".*"
          use_default_offset: true
          scale: ${robot.actuators.action_scale}

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
          anchor_bodylink_name: ${robot.anchor_body}""",
        ),
        Highlight(
            file=motion_tracking_env,
            content="""
At environment construction time, HoloMotion materializes the OmegaConf reward subtree into `_rewards_config_dict` and passes it into `build_rewards_config`. That is the bridge from Hydra YAML to the IsaacLab reward manager configuration object.""",
            text="""
        _rewards_config_dict = EasyDict(
            OmegaConf.to_container(self.config.rewards, resolve=True)
        )""",
        ),
        Highlight(
            file=motion_tracking_env,
            content="""
The same `_init_isaaclab_env` path builds observations and then constructs `rewards: RewardsCfg = build_rewards_config(_rewards_config_dict)`. The resulting `rewards` object is later put into the manager-based IsaacLab environment config, so the RewardTermCfg instances created below are the runtime reward terms.""",
            text="""
            observations: ObservationsCfg = build_observations_config(
                _obs_config_dict.obs_groups
            )
            rewards: RewardsCfg = build_rewards_config(_rewards_config_dict)

            if _terminations_config_dict:
                terminations: TerminationsCfg = build_terminations_config(
                    _terminations_config_dict
                )
            else:
                terminations: TerminationsCfg = TerminationsCfg()""",
        ),
        Highlight(
            file=reward_impl,
            content="""
`build_rewards_config` handles the flat reward table used here by skipping `_config`, resolving Hydra/Holo config values, resolving the function first from HoloMotion globals and then from `isaaclab.envs.mdp`, and setting an attribute with an IsaacLab `RewardTermCfg(func=..., weight=..., params=...)`. Thus local reference-tracking functions use HoloMotion implementations, while generic `action_rate_l2`, `joint_pos_limits`, `undesired_contacts`, and `is_alive` are delegated to IsaacLab MDP implementations by matching name.""",
            text="""
    if not is_grouped:
        for reward_name, reward_cfg in reward_config_dict.items():
            if reward_name == "_config":
                continue
            reward_cfg = resolve_holo_config(reward_cfg)
            base_params = resolve_holo_config(reward_cfg["params"])
            method_name = f"{reward_name}"
            func = globals().get(method_name, None)
            if func is None:
                func = getattr(isaaclab_mdp, reward_name, None)
            if func is None:
                raise ValueError(f"Unknown reward function: {reward_name}")
            params = dict(base_params)
            setattr(
                rewards_cfg,
                reward_name,
                RewardTermCfg(
                    func=func,
                    weight=reward_cfg["weight"],
                    params=params,
                ),
            )
        return rewards_cfg""",
        ),
        Highlight(
            file=scene_builder,
            content="""
The `undesired_contacts` regularizer is grounded in an actual IsaacLab contact sensor named `contact_forces`: HoloMotion builds it from the env config with prim path defaulting to all robot bodies, history length 3, track-air-time enabled, force threshold 10.0, and debug visualization disabled.""",
            text="""
    def build_contact_sensor_config(config: dict) -> ContactSensorCfg:
        \"\"\"Build contact sensor configuration.\"\"\"
        prim_path = config.get("prim_path", "{ENV_REGEX_NS}/Robot/.*")
        history_length = config.get("history_length", 3)
        force_threshold = config.get("force_threshold", 10.0)
        track_air_time = config.get("track_air_time", True)
        debug_vis = config.get("debug_vis", False)

        return ContactSensorCfg(
            prim_path=prim_path,
            history_length=history_length,
            track_air_time=track_air_time,
            force_threshold=force_threshold,
            debug_vis=debug_vis,
        )""",
        ),
        Highlight(
            file=motion_command,
            content="""
All local reference-tracking rewards read reference tensors through `RefMotionCommand`. `_get_ref_state_array` resolves a concrete cache key from the requested base key plus logical prefix such as `ref_`, then gathers the current clip/frame assignment with `1 + n_fut_frames`. This is the source of the reference root/body poses and velocities used by the reward formulas.""",
            text="""
    def _get_ref_state_array(
        self,
        base_key: str,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        \"\"\"Gather a reference tensor from the current cache batch.

        Args:
            base_key: Base key in the motion cache (e.g. \\"dof_pos\\", \\"root_pos\\").
            prefix: Optional logical prefix (e.g. \\"\\", \\"ref_\\", \\"ft_ref_\\", \\"robot_\\").

        Returns:
            Tensor of shape ``[num_envs, 1 + n_fut_frames, ...]`` gathered for
            the envs' current clip/frame assignments.
        \"\"\"
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
            file=motion_command,
            content="""
The reward functions consistently compare the robot against the immediate next reference frame rather than the current reference frame. For root position it adds `env_origins` back to cache positions, while rotation and velocity getters convert or slice the corresponding immediate-next cache tensors; current-frame helpers remain adjacent but the reward code below calls the immediate-next variants.""",
            text="""
    def get_ref_motion_root_global_pos_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_immediate_next_ref_state_array("root_pos", prefix)
        return base + self._env_origins

    def get_ref_motion_root_global_rot_quat_xyzw_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self._get_ref_state_array("root_rot", prefix)[:, 0, ...]

    def get_ref_motion_root_global_rot_quat_xyzw_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self._get_immediate_next_ref_state_array("root_rot", prefix)

    def get_ref_motion_root_global_rot_quat_wxyz_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self.get_ref_motion_root_global_rot_quat_xyzw_cur(
            prefix=prefix
        )[..., [3, 0, 1, 2]]

    def get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self.get_ref_motion_root_global_rot_quat_xyzw_immediate_next(
            prefix=prefix
        )[..., [3, 0, 1, 2]]

    def get_ref_motion_root_global_lin_vel_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("root_vel", prefix)
        return base[:, 0, ...]

    def get_ref_motion_root_global_lin_vel_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self._get_immediate_next_ref_state_array("root_vel", prefix)""",
        ),
        Highlight(
            file=reward_impl,
            content="""
Root-position tracking optimizes the robot base XY position toward the immediate-next reference root XY. It reads `ref_motion`, takes `ref_root_pos[:, :2]` and `command.robot.data.root_pos_w[:, :2]`, sums squared XY error, and returns an exponential reward `exp(-error/std^2)`; the YAML gives it weight +1.0 and std 0.2.""",
            text="""
def root_pos_xy_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    ref_root_pos = command.get_ref_motion_root_global_pos_immediate_next(
        prefix=ref_prefix
    )
    error = torch.sum(
        torch.square(
            ref_root_pos[:, :2] - command.robot.data.root_pos_w[:, :2]
        ),
        dim=-1,
    )
    return torch.exp(-error / std**2)""",
        ),
        Highlight(
            file=reward_impl,
            content="""
Root-rotation tracking optimizes robot base orientation toward the immediate-next reference root quaternion. It converts the reference getter to wxyz, compares against IsaacLab `root_quat_w(env)` with squared quaternion error magnitude, and applies the same exponential kernel; the YAML gives it weight +0.5 and std 0.4.""",
            text="""
def root_rot_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    ref_root_quat = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )
    error = (
        isaaclab_math.quat_error_magnitude(
            ref_root_quat,
            isaaclab_mdp.root_quat_w(env),
        )
        ** 2
    )
    return torch.exp(-error / std**2)""",
        ),
        Highlight(
            file=reward_impl,
            content="""
Root-relative keybody position tracking optimizes the 14 configured key bodies to match the reference skeleton in their own root frames. It maps names to body indices, converts reference body/root positions into environment frame, computes robot and reference root-relative vectors with the respective root pose, averages squared XYZ error over key bodies, and returns `exp(-mean_error/std^2)`; the YAML gives it weight +1.0 and std 0.3.""",
            text="""
def root_rel_keybodylink_pos_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    \"\"\"Track root-relative keybody positions using environment-frame positions.

    IsaacLab MDP root position helpers are expressed in the environment frame
    (simulation-world position minus `env.scene.env_origins`). This reward
    converts body positions into the same environment frame before computing
    root-relative vectors.
    \"\"\"
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Get reference and robot root positions/orientations
    ref_root_pos_env = positions_world_to_env_frame(
        command.get_ref_motion_root_global_pos_immediate_next(
            prefix=ref_prefix
        ),
        env.scene.env_origins,
    )  # [B, 3]
    ref_root_quat_w = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, 4] (w,x,y,z)
    robot_root_pos_env = isaaclab_mdp.root_pos_w(env)  # [B, 3]
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)  # [B, 4] (w,x,y,z)

    # Select relevant body indices first
    ref_body_pos_env = positions_world_to_env_frame(
        command.get_ref_motion_bodylink_global_pos_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs],
        env.scene.env_origins,
    )
    robot_body_pos_root_rel = (
        root_relative_positions_from_mixed_position_frames(
            body_pos_w=command.robot.data.body_pos_w[:, keybody_idxs],
            root_pos_env=robot_root_pos_env,
            root_quat_w=robot_root_quat_w,
            env_origins=env.scene.env_origins,
        )
    )
    ref_body_pos_root_rel = root_relative_positions_from_env_frame(
        body_pos_env=ref_body_pos_env,
        root_pos_env=ref_root_pos_env,
        root_quat_w=ref_root_quat_w,
    )

    # Compute error
    error = torch.sum(
        torch.square(ref_body_pos_root_rel - robot_body_pos_root_rel),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)""",
        ),
        Highlight(
            file=reward_impl,
            content="""
Root-relative keybody rotation tracking optimizes relative limb/link orientations, not absolute world quaternions. It transforms robot and reference body quaternions into their respective root frames using inverse root quaternions, computes squared quaternion error per key body, averages across bodies, and returns the exponential reward; the YAML gives it the largest positive imitation weight, +2.0, with std 0.4.""",
            text="""
def root_rel_keybodylink_rot_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    \"\"\"Track root-relative keybody rotations in each entity's root frame.

    Returns: [B]
    \"\"\"
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Root orientations
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)  # [B, 4]
    ref_root_quat_w = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, 4]

    # Body orientations (world)
    robot_body_quat_w = command.robot.data.body_quat_w[
        :, keybody_idxs
    ]  # [B, N, 4]
    ref_body_quat_w = (
        command.get_ref_motion_bodylink_global_rot_wxyz_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs]
    )  # [B, N, 4]

    # Relative (q_rel = q_root^{-1} * q_body)
    num_bodies = len(keybody_idxs)
    robot_root_quat_inv_exp = isaaclab_math.quat_inv(robot_root_quat_w)[
        :, None, :
    ].expand(-1, num_bodies, -1)
    ref_root_quat_inv_exp = isaaclab_math.quat_inv(ref_root_quat_w)[
        :, None, :
    ].expand(-1, num_bodies, -1)

    robot_rel_quat = isaaclab_math.quat_mul(
        robot_root_quat_inv_exp,
        robot_body_quat_w,
    )  # [B, N, 4]
    ref_rel_quat = isaaclab_math.quat_mul(
        ref_root_quat_inv_exp,
        ref_body_quat_w,
    )  # [B, N, 4]

    error = (
        isaaclab_math.quat_error_magnitude(ref_rel_quat, robot_rel_quat) ** 2
    )  # [B, N]
    return torch.exp(-error.mean(-1) / std**2)""",
        ),
        Highlight(
            file=reward_impl,
            content="""
Global keybody velocity tracking optimizes dynamic motion matching. The linear term compares immediate-next reference body linear velocities to robot body linear velocities for the configured key bodies; the angular term does the same for angular velocity. Both reduce squared vector error per body, average across bodies, and use exponential rewards with weights +1.0 and stds 1.0 and 3.14.""",
            text="""
def global_keybodylink_lin_vel_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    \"\"\"Track global keybody linear velocities.\"\"\"
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    ref_global_keybody_lin_vel = (
        command.get_ref_motion_bodylink_global_lin_vel_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs]
    )  # [B, N, 3]
    robot_keybody_lin_vel = command.robot.data.body_lin_vel_w[
        :, keybody_idxs
    ]  # [B, N, 3]

    error = torch.sum(
        torch.square(ref_global_keybody_lin_vel - robot_keybody_lin_vel),
        dim=-1,
    )  # [B, N]
    return torch.exp(-error.mean(-1) / std**2)


def global_keybodylink_ang_vel_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    \"\"\"Track global keybody angular velocities.\"\"\"
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    ref_global_keybody_ang_vel = (
        command.get_ref_motion_bodylink_global_ang_vel_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs]
    )  # [B, N, 3]
    robot_keybody_ang_vel = command.robot.data.body_ang_vel_w[
        :, keybody_idxs
    ]  # [B, N, 3]

    error = torch.sum(
        torch.square(ref_global_keybody_ang_vel - robot_keybody_ang_vel),
        dim=-1,
    )  # [B, N]
    return torch.exp(-error.mean(-1) / std**2)""",
        ),
        Highlight(
            file=env_cfg,
            content="""
The regularization penalties are curriculum-controlled: as completion rate crosses thresholds, the final configured weights are restored for action-rate, joint-limit, and undesired-contact terms. This shows the negative terms are not dead config; they are intentionally scheduled to full strength during training.""",
            text="""
      action_rate_l2_completion_rate:
        enabled: true
        func: reward_term_weight_by_completion_rate
        params:
          reward_term_name: "action_rate_l2"
          final_weight: -0.1
          start_scale: 0.1
          num_updates: 5
          cr_thresholds: [0.10, 0.20, 0.28, 0.34, 0.40]

      joint_pos_limits_completion_rate:
        enabled: true
        func: reward_term_weight_by_completion_rate
        params:
          reward_term_name: "joint_pos_limits"
          final_weight: -10.0
          start_scale: 0.1
          num_updates: 5
          cr_thresholds: [0.10, 0.20, 0.28, 0.34, 0.40]

      undesired_contacts_completion_rate:
        enabled: true
        func: reward_term_weight_by_completion_rate
        params:
          reward_term_name: "undesired_contacts"
          final_weight: -0.1
          start_scale: 0.1
          num_updates: 5
          cr_thresholds: [0.10, 0.20, 0.28, 0.34, 0.40]""",
        ),
    ]
)
