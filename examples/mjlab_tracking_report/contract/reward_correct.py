from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

task_registry = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py")
tracking_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py")
g1_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py")
tracking_rewards = File(path="mjlab/src/mjlab/tasks/tracking/mdp/rewards.py")
motion_command = File(path="mjlab/src/mjlab/tasks/tracking/mdp/commands.py")
common_rewards = File(path="mjlab/src/mjlab/envs/mdp/rewards.py")
reward_manager = File(path="mjlab/src/mjlab/managers/reward_manager.py")
manager_env = File(path="mjlab/src/mjlab/envs/manager_based_rl_env.py")

reward_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=task_registry,
            content="""
Scope anchor: this proof is for exactly `Mjlab-Tracking-Flat-Unitree-G1`. The registered task calls `unitree_g1_flat_tracking_env_cfg()` without overriding `has_state_estimation`, while the no-state-estimation variant is a separate task id. Therefore the active task is the Unitree G1 flat tracking environment with state estimation enabled.""",
            text="""
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

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
The G1 flat tracking factory defaults `has_state_estimation` to `True`, and only removes state-estimation actor observations inside the `if not has_state_estimation` branch. Since the scoped task uses the default call above, reward evidence below is for the state-estimation version, not the no-state-estimation sibling.""",
            text="""
def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  \"\"\"Create Unitree G1 flat terrain tracking configuration.\"\"\"
  cfg = make_tracking_env_cfg()""",
        ),
        Highlight(
            file=g1_env_cfg,
            content="""
The G1 specialization fixes the motion command body set and anchor. Rewards that operate on `command.cfg.body_names` therefore use these 14 Unitree G1 bodies, anchored at `torso_link`, with the command term named `motion` from the base config.""",
            text="""
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )""",
        ),
        Highlight(
            file=tracking_env_cfg,
            content="""
The active reward table has exactly six positive motion imitation exponentials and three penalties. The six positive terms all read command field `motion` and use stds 0.3, 0.4, 0.3, 0.4, 1.0, and 3.14 with weights 0.5, 0.5, 1.0, 1.0, 1.0, and 1.0. The penalties are `action_rate_l2` at -0.1, `joint_limit` using all robot joints at -10.0, and `self_collisions` at -10.0 with force threshold 10.0.""",
            text="""
  rewards: dict[str, RewardTermCfg] = {
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
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
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
    ),
  }""",
        ),
        Highlight(
            file=tracking_rewards,
            content="""
Root/anchor position reward formula: get the `motion` command, compare command field `anchor_pos_w` with robot field `robot_anchor_pos_w`, reduce squared XYZ error with `sum(..., dim=-1)`, and return `exp(-error / std^2)`. With the active config this is weighted +0.5 and uses std 0.3.""",
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
        Highlight(
            file=tracking_rewards,
            content="""
Root/anchor orientation reward formula: get the `motion` command, compare command field `anchor_quat_w` with robot field `robot_anchor_quat_w`, square the quaternion error magnitude, and return `exp(-error / std^2)`. With the active config this is weighted +0.5 and uses std 0.4.""",
            text="""
def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)""",
        ),
        Highlight(
            file=tracking_rewards,
            content="""
Body position reward formula: select command body indexes from the G1 `body_names`, compare command field `body_pos_relative_w` with robot field `robot_body_pos_w`, reduce squared XYZ error per body, average over bodies, and return `exp(-mean_error / std^2)`. With the active config this is weighted +1.0 and uses std 0.3.""",
            text="""
def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)""",
        ),
        Highlight(
            file=tracking_rewards,
            content="""
Body orientation reward formula: select the same command body indexes, compare command field `body_quat_relative_w` with robot field `robot_body_quat_w`, square quaternion error magnitude per body, average over bodies, and return `exp(-mean_error / std^2)`. With the active config this is weighted +1.0 and uses std 0.4.""",
            text="""
def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)""",
        ),
        Highlight(
            file=tracking_rewards,
            content="""
Body linear velocity reward formula: select the G1 command body indexes, compare command field `body_lin_vel_w` with robot field `robot_body_lin_vel_w`, reduce squared XYZ velocity error per body, average over bodies, and return `exp(-mean_error / std^2)`. With the active config this is weighted +1.0 and uses std 1.0.""",
            text="""
def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)""",
        ),
        Highlight(
            file=tracking_rewards,
            content="""
Body angular velocity reward formula: select the G1 command body indexes, compare command field `body_ang_vel_w` with robot field `robot_body_ang_vel_w`, reduce squared XYZ angular velocity error per body, average over bodies, and return `exp(-mean_error / std^2)`. With the active config this is weighted +1.0 and uses std 3.14.""",
            text="""
def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)""",
        ),
        Highlight(
            file=motion_command,
            content="""
The reward functions above are grounded in concrete `MotionCommand` fields: command root/body positions, quaternions, and velocities come from the loaded motion at `time_steps`, while robot counterparts come from the Unitree G1 entity's current body link tensors at the ordered body indexes.""",
            text="""
  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]""",
        ),
        Highlight(
            file=motion_command,
            content="""
The anchor command fields used by the two root rewards are also concrete: `anchor_pos_w` and `anchor_quat_w` are the motion body's position/quaternion for the configured `torso_link` anchor, with positions shifted into each environment origin.""",
            text="""
  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]""",
        ),
        Highlight(
            file=motion_command,
            content="""
The robot fields used by all six positive rewards are direct G1 entity data: selected body link positions, quaternions, linear velocities, angular velocities, plus the configured anchor body's position and quaternion.""",
            text="""
  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]""",
        ),
        Highlight(
            file=motion_command,
            content="""
The relative body command fields used by body position/orientation rewards are recalculated from the reference body pose, reference anchor pose, and robot anchor pose. This aligns the reference motion to the robot anchor yaw and anchor XY while keeping the reference anchor height, then produces `body_pos_relative_w` and `body_quat_relative_w` before rewards read them.""",
            text="""
  def update_relative_body_poses(self) -> None:
    \"\"\"Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``.

    Called after ``reset_to_frame`` so that termination checks that
    compare relative body positions see the correct state.
    \"\"\"
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )""",
        ),
        Highlight(
            file=common_rewards,
            content="""
`action_rate_l2` penalty formula: compute the squared L2 norm of raw policy action change, `sum((action - prev_action)^2)`. The active table gives this positive cost a negative weight of -0.1, so larger action jumps reduce reward.""",
            text="""
def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  \"\"\"Penalize the rate of change of the actions using L2 squared kernel.

  Operates on raw policy output (before per-term scale/offset).
  \"\"\"
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )""",
        ),
        Highlight(
            file=common_rewards,
            content="""
`joint_limit` penalty formula: for every matched robot joint, add how far it is below its soft lower limit plus how far it is above its soft upper limit, then sum across joints. The active table matches `joint_names=(".*",)` and weights this violation magnitude by -10.0.""",
            text="""
def joint_pos_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  \"\"\"Penalize joint positions if they cross the soft limits.\"\"\"
  asset: Entity = env.scene[asset_cfg.name]
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  out_of_limits = -(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
  ).clip(max=0.0)
  out_of_limits += (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
  ).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)""",
        ),
        Highlight(
            file=g1_env_cfg,
            content="""
The G1 config installs the `self_collision` sensor used by the self-collision reward. It watches pelvis-subtree versus pelvis-subtree contacts, records contact force, keeps four substeps of history, and exposes one contact slot for the reward function.""",
            text="""
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)""",
        ),
        Highlight(
            file=tracking_rewards,
            content="""
`self_collisions` penalty formula: read `env.scene["self_collision"]`; when force history exists, compute force magnitudes, mark each history substep where any contact slot exceeds force threshold 10.0, and return the count of hit substeps. The active table weights that count by -10.0. If no history exists, it falls back to instantaneous found count.""",
            text="""
def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  \"\"\"Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  \"\"\"
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)""",
        ),
        Highlight(
            file=manager_env,
            content="""
The environment constructs the reward manager from exactly `cfg.rewards` and passes `cfg.scale_rewards_by_dt`, so the reward terms above are the terms the manager computes for the scoped environment.""",
            text="""
    self.reward_manager = RewardManager(
      self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.reward_manager}")""",
        ),
        Highlight(
            file=manager_env,
            content="""
By default, `scale_rewards_by_dt` is true. Therefore the manager multiplies every weighted reward term by the environment step duration unless a task explicitly disables this, which the G1 tracking config does not do.""",
            text="""
  scale_rewards_by_dt: bool = True
  \"\"\"Whether to multiply rewards by the environment step duration (dt).

  When True (default), reward values are scaled by step_dt to normalize cumulative
  episodic rewards across different simulation frequencies. Set to False for
  algorithms that expect unscaled reward signals (e.g., HER, static reward scaling).
  \"\"\"""",
        ),
        Highlight(
            file=manager_env,
            content="""
The reward manager runs after the decimated physics step and termination check, using `dt=self.step_dt`. That is the dt used for scaling the weighted reward terms.""",
            text="""
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()""",
        ),
        Highlight(
            file=reward_manager,
            content="""
Reward manager calculation: for every active term, compute `raw_value * weight * scale`, where `scale` is `dt` when dt scaling is enabled. It zeros NaN, positive infinity, and negative infinity with `torch.nan_to_num`, accumulates both the total reward buffer and episode sums, and stores unscaled reward rate as `value / scale` for per-term visualization.""",
            text="""
  def compute(self, dt: float) -> torch.Tensor:
    self._reward_buf[:] = 0.0
    scale = dt if self._scale_by_dt else 1.0
    for term_idx, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      if term_cfg.weight == 0.0:
        self._step_reward[:, term_idx] = 0.0
        continue
      value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * scale
      # NaN/Inf can occur from corrupted physics state; zero them to avoid policy crash.
      value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
      self._reward_buf += value
      self._episode_sums[name] += value
      self._step_reward[:, term_idx] = value / scale
    return self._reward_buf""",
        ),
    ]
)
