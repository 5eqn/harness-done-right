from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

tracking_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py")
g1_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py")
g1_task_registry = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py")
tracking_observations = File(
    path="mjlab/src/mjlab/tasks/tracking/mdp/observations.py"
)
common_observations = File(path="mjlab/src/mjlab/envs/mdp/observations.py")
motion_commands = File(path="mjlab/src/mjlab/tasks/tracking/mdp/commands.py")
observation_manager = File(path="mjlab/src/mjlab/managers/observation_manager.py")
command_manager = File(path="mjlab/src/mjlab/managers/command_manager.py")
g1_constants = File(
    path="mjlab/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py"
)
g1_xml = File(path="mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml")

observation_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=g1_task_registry,
            content="""
This pins the proof to the requested task id only. `Mjlab-Tracking-Flat-Unitree-G1` is registered with the default `unitree_g1_flat_tracking_env_cfg()`, not the separate `No-State-Estimation` variant.""",
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
            file=g1_env_cfg,
            content="""
The G1 default has state estimation because `has_state_estimation` defaults to True. The only code that removes state-estimation-dependent actor inputs runs under `if not has_state_estimation`; therefore the requested task keeps both `motion_anchor_pos_b` and `base_lin_vel` in the actor observations.""",
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
This is the state-estimation fork. It removes exactly `motion_anchor_pos_b` and `base_lin_vel` from actor terms only for the no-state-estimation configuration, which is not the requested task id.""",
            text="""
  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )""",
        ),
        Highlight(
            file=tracking_env_cfg,
            content="""
Actor schema evidence. The actor group contains exactly `command`, `motion_anchor_pos_b`, `motion_anchor_ori_b`, `base_lin_vel`, `base_ang_vel`, `joint_pos`, `joint_vel`, and `actions`. Actor corruption/noise is enabled, so the per-term noise declared here applies before concatenation.""",
            text="""
  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }""",
        ),
        Highlight(
            file=tracking_env_cfg,
            content="""
Critic schema evidence. The critic group contains the same command/anchor/base/joint/action families plus `body_pos` and `body_ori`. It disables corruption, so the critic receives clean terms. Its configured terms are concatenated, not returned as a dict.""",
            text="""
  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }""",
        ),
        Highlight(
            file=tracking_env_cfg,
            content="""
This proves the configured actor and critic terms are flattened into one tensor per group. The actor has corruption enabled; the critic does not.""",
            text="""
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }""",
        ),
        Highlight(
            file=observation_manager,
            content="""
Observation manager evidence: for each active term it computes the function output, applies noise only if the group's `enable_corruption` survived preparation, stores each term under its configured name, and finally concatenates `group_obs.values()` along the configured dimension. The dimension table is also built from each term's runtime tensor shape, excluding the batch dimension.""",
            text="""
    if self._group_obs_concatenate[group_name]:
      result = torch.cat(
        list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name]
      )
      # Final check for concatenated result (non-per-term checking).
      if not group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
        result = self._check_and_handle_nans(
          result, context=group_name, policy=group_cfg.nan_policy
        )
      return result""",
        ),
        Highlight(
            file=observation_manager,
            content="""
This second manager excerpt is the noise and shape-preparation evidence. If a group disables corruption, term noise is cleared. Each term dimension is recorded as `obs_dims[1:]`, so all arithmetic below is over the non-batch feature dimension actually returned by each observation function.""",
            text="""
        if not group_cfg.enable_corruption:
          term_cfg.noise = None
        if group_cfg.history_length is not None:
          term_cfg.history_length = group_cfg.history_length
          term_cfg.flatten_history_dim = group_cfg.flatten_history_dim
        self._group_obs_term_names[group_name].append(term_name)
        self._group_obs_term_cfgs[group_name].append(term_cfg)
        if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
          self._group_obs_class_term_cfgs[group_name].append(term_cfg)

        obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)""",
        ),
        Highlight(
            file=motion_commands,
            content="""
Command shape evidence. `MotionCommand.command` is exactly `joint_pos` concatenated with `joint_vel`, both indexed from the same motion time steps. For G1's 29 joints, command contributes 29 + 29 = 58 dimensions.""",
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
        Highlight(
            file=common_observations,
            content="""
This connects the actor/critic `command`, joint, action, and built-in sensor terms to concrete tensors: generated command comes from the command manager, joint positions/velocities are all selected robot joints, actions are the current action vector, and IMU observations are raw built-in sensor tensors.""",
            text="""
def joint_vel_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]


##
# Actions.
##


def last_action(env: ManagerBasedRlEnv, action_name: str | None = None) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


##
# Commands.
##


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command


##
# Sensors.
##


def builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  \"\"\"Get observation from a built-in sensor by name.\"\"\"
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data""",
        ),
        Highlight(
            file=command_manager,
            content="""
This closes the `generated_commands` path: `get_command("motion")` returns the `MotionCommand.command` property highlighted above, so the observation named `command` has the 58-dimensional joint position plus joint velocity shape.""",
            text="""
  def get_command(self, name: str) -> torch.Tensor:
    return self._terms[name].command

  def get_term(self, name: str) -> CommandTerm:
    return self._terms[name]""",
        ),
        Highlight(
            file=tracking_observations,
            content="""
Anchor and body shape evidence. Anchor position is flattened from a single 3-vector, anchor orientation converts quaternion delta to a rotation matrix and keeps the first two columns, so rot6d = 6. Body position repeats over `len(command.cfg.body_names)` bodies and flattens 3 coordinates each; body orientation also keeps two rotation-matrix columns for 6 coordinates per body.""",
            text="""
def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)""",
        ),
        Highlight(
            file=tracking_observations,
            content="""
Critic body shapes come directly from this code. With the G1 config's 14 body names, `robot_body_pos_b` is 14 * 3 = 42 and `robot_body_ori_b` is 14 * 6 = 84.""",
            text="""
def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)""",
        ),
        Highlight(
            file=g1_env_cfg,
            content="""
The requested G1 task sets the tracking anchor to `torso_link` and defines exactly 14 tracked body names. Those 14 names ground the critic body dimensions: body position 14 * 3 = 42 and body orientation rot6d 14 * 6 = 84.""",
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
            file=g1_xml,
            content="""
The IMU terms named in the observation config are MuJoCo gyro and velocimeter sensors on the G1 XML. Each is a 3-vector, so `base_ang_vel` contributes 3 and `base_lin_vel` contributes 3.""",
            text="""
  <sensor>
    <gyro name="imu_ang_vel" site="imu_in_pelvis"/>
    <velocimeter name="imu_lin_vel" site="imu_in_pelvis"/>
    <accelerometer name="imu_lin_acc" site="imu_in_pelvis"/>
    <subtreeangmom name="root_angmom" body="pelvis"/>
  </sensor>""",
        ),
        Highlight(
            file=g1_constants,
            content="""
Action and joint count evidence from the G1 actuator constants. These target-name regex groups cover the 29 actuated G1 joints: 10 shoulder/elbow/wrist-roll joints, 5 hip-pitch/hip-yaw/waist-yaw joints, 4 hip-roll/knee joints, 4 wrist-pitch/wrist-yaw joints, 2 waist pitch/roll joints, and 4 ankle pitch/roll joints. `G1_ACTION_SCALE` is built over these actuator target names, and the G1 tracking config assigns it to the joint-position action, so action dimension is 29. The same full-joint indexing makes `joint_pos` and `joint_vel` 29 each.""",
            text="""
G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_ACTUATOR_4010 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)""",
        ),
        Highlight(
            file=g1_constants,
            content="""
This completes the actuator groups used for the 29-dimensional G1 action and joint observations, and proves that `G1_ACTION_SCALE` is generated from every configured actuator target expression.""",
            text="""
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)""",
        ),
        Highlight(
            file=g1_constants,
            content="""
The `G1_ACTION_SCALE` dictionary is programmatically derived from the G1 articulation's actuator target names, and the tracking task installs that scale into the joint-position action. This is why the last-action observation has the same 29-dimensional action width as the joint-position action.""",
            text="""
G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_ACTION_SCALE[n] = 0.25 * e / s""",
        ),
        Highlight(
            file=g1_env_cfg,
            content="""
The G1 tracking task replaces the base action scale with the G1-specific action scale. Together with the `joint_pos` action using all actuators, this grounds the action count used by `last_action`: 29.""",
            text="""
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE""",
        ),
        Highlight(
            file=tracking_env_cfg,
            content="""
The base tracking action is a single joint-position action over all robot actuators. For the G1 task, the G1-specific scale above is assigned to this action, so `env.action_manager.action` and the `actions` observation have 29 elements.""",
            text="""
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }""",
        ),
        Highlight(
            file=tracking_env_cfg,
            content="""
Final arithmetic. Actor terms are command 58, anchor position 3, anchor rot6d 6, base linear velocity 3, base angular velocity 3, joint_pos 29, joint_vel 29, actions 29. Actor total: 58 + 3 + 6 + 3 + 3 + 29 + 29 + 29 = 189. Critic adds body_pos 42 and body_ori 84 to the same families: 58 + 3 + 6 + 42 + 84 + 3 + 3 + 29 + 29 + 29 = 286. These are concatenated group vectors, with actor noise enabled and critic noise disabled.""",
            text="""
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }""",
        ),
    ]
)
