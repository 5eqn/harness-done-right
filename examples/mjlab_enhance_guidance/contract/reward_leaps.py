from enhance_guidance import AtomicLeap, File, Highlight


holomotion_rewards_cfg = File(
    path="HoloMotion/holomotion/config/env/rewards/motion_tracking/rew_motion_tracking.yaml"
)
holomotion_rewards = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_rewards.py"
)
mjlab_tracking_cfg = File(
    path="mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py"
)
mjlab_tracking_rewards = File(
    path="mjlab/src/mjlab/tasks/tracking/mdp/rewards.py"
)
mjlab_common_rewards = File(path="mjlab/src/mjlab/envs/mdp/rewards.py")


reward_leaps: list[AtomicLeap] = [
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_rewards_cfg,
            content=(
                "HoloMotion rewards root pose separately and weights "
                "root-relative keybody pose/rotation as the central motion "
                "tracking signal."
            ),
            text="""
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
      ref_prefix: ${rewards._config.reward_prefix}""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_tracking_cfg,
            content=(
                "MJLab currently rewards anchor and whole-body terms in a "
                "more global/anchor-relative style without a keybody-first "
                "root-relative reward split."
            ),
            text="""
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),""",
        ),
        change_direction=(
            "Replace MJLab's anchor/global tracking reward bundle with a "
            "HoloMotion-style bundle: root xy and root rotation are separate "
            "coarse terms, while selected key bodies are tracked in the "
            "robot/reference root frames with stronger rotation weighting."
        ),
        change_reason=(
            "Large motion libraries contain clips with different global "
            "placements, headings, retargeting offsets, and limb salience. "
            "Root-relative keybody rewards make the policy learn reusable "
            "whole-body shape and phase alignment instead of overfitting to "
            "absolute anchor coordinates or every body link equally."
        ),
        changed_code="""
rewards: dict[str, RewardTermCfg] = {
  "root_pos_xy_tracking_exp": RewardTermCfg(
    func=mdp.root_pos_xy_tracking_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.2, "ref_prefix": "ref_"},
  ),
  "root_rot_tracking_exp": RewardTermCfg(
    func=mdp.root_rot_tracking_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.4, "ref_prefix": "ref_"},
  ),
  "root_rel_keybodylink_pos_tracking_l2_exp": RewardTermCfg(
    func=mdp.root_rel_keybodylink_pos_tracking_l2_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.3,
      "keybody_names": G1_KEY_BODIES,
      "ref_prefix": "ref_",
    },
  ),
  "root_rel_keybodylink_rot_tracking_l2_exp": RewardTermCfg(
    func=mdp.root_rel_keybodylink_rot_tracking_l2_exp,
    weight=2.0,
    params={
      "command_name": "motion",
      "std": 0.4,
      "keybody_names": G1_KEY_BODIES,
      "ref_prefix": "ref_",
    },
  ),
}""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_rewards,
            content=(
                "HoloMotion reward terms explicitly request the immediate next "
                "reference frame through a configurable prefix."
            ),
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
        mjlab_reference=Highlight(
            file=mjlab_tracking_rewards,
            content=(
                "MJLab reward terms read the command's current cached motion "
                "state directly, so reward timing is tied to the command "
                "property rather than next-frame reference plumbing."
            ),
            text="""
def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)""",
        ),
        change_direction=(
            "Thread `ref_prefix` through MJLab tracking reward functions and "
            "query immediate-next reference state from the motion command "
            "instead of relying on current-frame command properties."
        ),
        change_reason=(
            "Generalized tracking trains against reference snippets sampled "
            "from many clips. Reward, observation, and reset code need to agree "
            "on which reference frame is being tracked; immediate-next reward "
            "targets avoid one-step lag and make it possible to swap filtered, "
            "unfiltered, or curriculum reference prefixes without duplicating "
            "reward functions."
        ),
        changed_code="""
def root_pos_xy_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  ref_prefix: str = "ref_",
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  ref_root_pos = command.get_ref_motion_root_global_pos_immediate_next(
    prefix=ref_prefix
  )
  robot_root_pos = command.robot.data.root_pos_w
  error = torch.sum(torch.square(ref_root_pos[:, :2] - robot_root_pos[:, :2]), dim=-1)
  return torch.exp(-error / std**2)


def key_dof_position_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  key_dofs: tuple[str, ...] | None = None,
  ref_prefix: str = "ref_",
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  key_dof_ids = command.find_joint_ids(key_dofs)
  ref_dof_pos = command.get_ref_motion_dof_pos_immediate_next(prefix=ref_prefix)
  error = torch.sum(
    torch.square(command.robot.data.joint_pos[:, key_dof_ids] - ref_dof_pos[:, key_dof_ids]),
    dim=-1,
  )
  return torch.exp(-error / std**2)""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_rewards_cfg,
            content=(
                "HoloMotion keeps the policy alive with a small positive term "
                "and uses comparatively gentle action, joint-limit, and contact "
                "regularizers."
            ),
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
        mjlab_reference=Highlight(
            file=mjlab_tracking_cfg,
            content=(
                "MJLab's tracking config has action-rate and joint-limit "
                "penalties, but no alive shaping and a much harsher "
                "self-collision penalty."
            ),
            text="""
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),""",
        ),
        change_direction=(
            "Add an alive reward to MJLab tracking, keep action-rate and "
            "joint-limit penalties, and convert the broad self-collision cost "
            "into a lower-weight undesired-contact regularizer over configured "
            "non-support bodies."
        ),
        change_reason=(
            "With a diverse motion library, many valid motions include brief "
            "incidental contacts or unusual poses. A severe all-or-nothing "
            "self-collision cost can dominate tracking and reject useful clips; "
            "a small alive reward plus scoped contact regularization preserves "
            "safety while letting the tracker learn rare but valid motion modes."
        ),
        changed_code="""
rewards.update(
  {
    "is_alive": RewardTermCfg(func=mdp.is_alive, weight=0.5),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts,
      weight=-0.1,
      params={
        "threshold": 1.0,
        "sensor_cfg": SceneEntityCfg(
          "contact_forces",
          body_names=(G1_UNDESIRED_CONTACTS_REGEX,),
        ),
      },
    ),
  }
)""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_rewards,
            content=(
                "HoloMotion includes normalized mechanical regularizers that "
                "scale by actuator effort and joint velocity limits."
            ),
            text="""
        current_torque = asset.data.applied_torque[:, joint_ids]
        current_joint_vel = asset.data.joint_vel[:, joint_ids]
        joint_vel_limits = asset.data.joint_vel_limits[:, joint_ids]

        if not torch.all(torch.isfinite(joint_vel_limits)) or not torch.all(
            joint_vel_limits > 0.0
        ):
            raise ValueError(
                "normed_positive_work requires finite, strictly positive "
                "joint velocity limits for all selected joints."
            )

        normalized_power = (current_torque * inv_effort_limit) * (
            current_joint_vel / joint_vel_limits
        )
        return torch.sum(torch.relu(normalized_power), dim=-1)""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_common_rewards,
            content=(
                "MJLab has an electrical-power cost, but it sums raw positive "
                "mechanical power without effort/velocity normalization."
            ),
            text="""
  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.qfrc_actuator[:, self._joint_ids]
    qd = asset.data.joint_vel[:, self._joint_ids]
    mech = tau * qd
    mech_pos = torch.clamp(mech, min=0.0)  # Don't penalize regen.
    return torch.sum(mech_pos, dim=1)""",
        ),
        change_direction=(
            "Replace raw electrical-power scaling in tracking with optional "
            "normalized positive-work and torque-rate terms that divide by "
            "per-joint effort and velocity limits before summing."
        ),
        change_reason=(
            "Raw power penalties can bias the policy away from high-energy but "
            "legitimate motions in a large library and can over-penalize joints "
            "with larger actuator limits. Normalized work/torque regularizers "
            "make the smoothness and efficiency pressure comparable across "
            "joints, robots, and clips."
        ),
        changed_code="""
class normed_positive_work:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    joint_ids, _ = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    effort_limit = asset.data.actuator_effort_limit[:, self._joint_ids]
    if effort_limit.ndim == 2:
      effort_limit = effort_limit[0]
    if not torch.all(torch.isfinite(effort_limit)) or not torch.all(effort_limit > 0.0):
      raise ValueError("normed_positive_work requires positive finite effort limits.")
    self._inv_effort_limit = effort_limit.reciprocal()

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    torque = asset.data.qfrc_actuator[:, self._joint_ids]
    joint_vel = asset.data.joint_vel[:, self._joint_ids]
    joint_vel_limits = asset.data.joint_vel_limits[:, self._joint_ids]
    if not torch.all(torch.isfinite(joint_vel_limits)) or not torch.all(joint_vel_limits > 0.0):
      raise ValueError("normed_positive_work requires positive finite velocity limits.")
    normalized_power = (torque * self._inv_effort_limit) * (joint_vel / joint_vel_limits)
    return torch.sum(torch.relu(normalized_power), dim=1)""",
    ),
]
