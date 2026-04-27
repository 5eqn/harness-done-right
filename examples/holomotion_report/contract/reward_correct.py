from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

train_cfg = File(path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_mlp.yaml")
env_impl = File(path="HoloMotion/holomotion/src/env/motion_tracking.py")
robot_cfg = File(path="HoloMotion/holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml")
reward_cfg = File(path="HoloMotion/holomotion/config/env/rewards/motion_tracking/rew_motion_tracking.yaml")
reward_builder = File(path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_rewards.py")

reward_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=train_cfg,
            content="""
Step 1: establish which reward catalog is active. The default MLP motion-tracking training config includes `/env/rewards: motion_tracking/rew_motion_tracking`, so the active reward catalog is `HoloMotion/holomotion/config/env/rewards/motion_tracking/rew_motion_tracking.yaml`.""",
            text="""
  - /env/rewards: motion_tracking/rew_motion_tracking""",
        ),
        Highlight(
            file=env_impl,
            content="""
Step 2: establish how the catalog is installed. `MotionTrackingEnv` passes the resolved reward config dictionary to `build_rewards_config`, producing the IsaacLab reward manager config used by the environment.""",
            text="""
            rewards: RewardsCfg = build_rewards_config(_rewards_config_dict)""",
        ),
        Highlight(
            file=reward_builder,
            content="""
Step 3: establish the binding semantics. For the flat active catalog, every YAML key except `_config` becomes one `RewardTermCfg`. Resolution order is HoloMotion reward functions first (`globals()`), then IsaacLab MDP rewards. The exact YAML `weight` and `params` are preserved in the resulting term config.""",
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
            file=robot_cfg,
            content="""
Step 4: establish robot constants used by the reward catalog. The G1 policy has 29 DoFs and 30 bodies. The reward catalog's `${robot.key_bodies}` resolves to 14 named key bodies, so keybody tracking rewards average over K=14 bodies. The undesired-contact regex excludes the two ankle-roll links and two wrist-yaw links from the 30 bodies, so it targets the other 26 body names after IsaacLab resolves the regex.""",
            text='''
  dof_obs_size: 29
  actions_dim: 29
  num_bodies: 30
  num_extend_bodies: 0
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
    - "right_wrist_yaw_link"''',
        ),
        Highlight(
            file=reward_cfg,
            content="""
Step 5: exact active reward catalog. There are 10 enabled terms:
1. `is_alive`, weight +0.5.
2. `root_pos_xy_tracking_exp`, weight +1.0, std 0.2.
3. `root_rot_tracking_exp`, weight +0.5, std 0.4.
4. `root_rel_keybodylink_pos_tracking_l2_exp`, weight +1.0, K=14 key bodies, std 0.3.
5. `root_rel_keybodylink_rot_tracking_l2_exp`, weight +2.0, K=14 key bodies, std 0.4.
6. `global_keybodylink_lin_vel_tracking_l2_exp`, weight +1.0, K=14 key bodies, std 1.0.
7. `global_keybodylink_ang_vel_tracking_l2_exp`, weight +1.0, K=14 key bodies, std 3.14.
8. `action_rate_l2`, weight -0.1.
9. `joint_pos_limits`, weight -10.0 over all joints matched by `".*"`, i.e. the 29 configured DoFs.
10. `undesired_contacts`, weight -0.1, contact threshold 1.0, body regex matching all but the 4 excluded ankle/wrist contact bodies.""",
            text="""
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
      ref_prefix: ${rewards._config.reward_prefix}

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
            file=reward_builder,
            content="""
Step 6: root pose formulas. `root_pos_xy_tracking_exp` is a 2D XY squared error between immediate-next reference root position and current robot root position, then `exp(-error / 0.2^2)` from the catalog. `root_rot_tracking_exp` uses squared quaternion error magnitude against IsaacLab's current root quaternion, then `exp(-error / 0.4^2)`.""",
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
    return torch.exp(-error / std**2)


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
            file=reward_builder,
            content="""
Step 7: keybody position formula. For K=14 key bodies, the position reward selects those body indices, converts reference and robot body positions into consistent root-relative/environment frames, computes per-body squared 3D error `[B,K]`, averages across K, then applies `exp(-mean_error / 0.3^2)`. With catalog weight +1.0, this contributes up to +1.0 before reward-manager timestep scaling.""",
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
    keybody_idxs = _get_body_indices(command.robot, keybody_names)""",
        ),
        Highlight(
            file=reward_builder,
            content="""
Step 8: keybody position terminal computation. The implementation's final computation confirms the numeric structure: `sum(square(...), dim=-1)` gives a scalar 3D L2 error for each key body, `mean(-1)` averages over the K=14 configured bodies, and the exponential kernel uses the configured `std`.""",
            text="""
    # Compute error
    error = torch.sum(
        torch.square(ref_body_pos_root_rel - robot_body_pos_root_rel),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)""",
        ),
        Highlight(
            file=reward_builder,
            content="""
Step 9: keybody rotation formula. `root_rel_keybodylink_rot_tracking_l2_exp` converts both robot and reference body quaternions into their own root frames, computes squared quaternion error magnitude per selected body `[B,K]`, averages over the K=14 key bodies, and applies `exp(-mean_error / 0.4^2)`. The catalog gives this term the largest positive tracking weight, +2.0.""",
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
    keybody_idxs = _get_body_indices(command.robot, keybody_names)""",
        ),
        Highlight(
            file=reward_builder,
            content="""
Step 10: keybody rotation terminal computation. The selected relative quaternions are compared with `quat_error_magnitude`, squared, averaged across K=14 bodies, then exponentiated with the configured std.""",
            text="""
    error = (
        isaaclab_math.quat_error_magnitude(ref_rel_quat, robot_rel_quat) ** 2
    )  # [B, N]
    return torch.exp(-error.mean(-1) / std**2)""",
        ),
        Highlight(
            file=reward_builder,
            content="""
Step 11: global keybody linear velocity formula. For the same K=14 key bodies, the linear velocity term compares immediate-next reference global body linear velocity `[B,K,3]` against robot body linear velocity `[B,K,3]`, reduces 3D squared error per body, averages over K, and applies `exp(-mean_error / 1.0^2)` with weight +1.0.""",
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
    return torch.exp(-error.mean(-1) / std**2)""",
        ),
        Highlight(
            file=reward_builder,
            content="""
Step 12: global keybody angular velocity formula. It mirrors the linear-velocity term but uses angular velocity `[B,K,3]` and catalog std 3.14. This looser std makes angular velocity tracking less sharply decayed than position or root XY tracking.""",
            text="""
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
            file=reward_cfg,
            content="""
Step 13: penalty terms. The three enabled penalties are action smoothness (`action_rate_l2`, weight -0.1), joint position limits over all matched joints (`joint_pos_limits`, weight -10.0 over `".*"`), and contact penalties (`undesired_contacts`, weight -0.1, threshold 1.0, contact sensor `contact_forces`). These are not custom HoloMotion functions in this file, so Step 3's binding rule resolves them from IsaacLab MDP.""",
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
    ]
)
