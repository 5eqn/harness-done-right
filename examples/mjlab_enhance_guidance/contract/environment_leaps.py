from enhance_guidance import AtomicLeap, File, Highlight


holomotion_rough_terrain_cfg = File(
    path="HoloMotion/holomotion/config/env/terrain/isaaclab_rough.yaml"
)
holomotion_domain_rand_cfg = File(
    path="HoloMotion/holomotion/config/env/domain_randomization/domain_rand_medium.yaml"
)
holomotion_scene_impl = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_scene.py"
)
holomotion_termination_cfg = File(
    path="HoloMotion/holomotion/config/env/terminations/termination_motion_tracking.yaml"
)
mjlab_tracking_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py")
mjlab_g1_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py")
mjlab_action_manager = File(path="mjlab/src/mjlab/managers/action_manager.py")


environment_leaps: list[AtomicLeap] = [
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_rough_terrain_cfg,
            content=(
                "HoloMotion trains motion tracking on generated rough terrain "
                "and explicitly randomizes spawn location inside each terrain."
            ),
            text="""
terrain:
  terrain_type: generator
  prim_path: /World/ground
  static_friction: 1.0
  dynamic_friction: 1.0
  restitution: 0.0
  friction_combine_mode: multiply
  restitution_combine_mode: multiply
  debug_vis: false
  max_init_terrain_level: 4

  # Randomize spawn position within each sub-terrain (recommended for locomotion).
  random_spawn: true

  random_spawn_margin: 4.0""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_tracking_env_cfg,
            content=(
                "MJLab's base tracking environment still assembles a flat plane "
                "scene, so the G1 tracker is not routinely exposed to uneven "
                "terrain or randomized terrain origins."
            ),
            text="""
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,""",
        ),
        change_direction=(
            "Replace the flat-only tracking scene with an MJLab procedural "
            "rough-terrain scene for training, and add a reset event that "
            "randomizes each environment's terrain origin before motion reset. "
            "Keep play/evaluation free to override back to flat terrain."
        ),
        change_reason=(
            "Generalized motion tracking should not learn only one perfectly "
            "flat contact manifold. Rough sub-terrains and randomized spawn "
            "origins force the policy to absorb small contact-height changes, "
            "spawn offsets, and terrain-level variation without making this an "
            "IsaacLab-specific port."
        ),
        changed_code="""
from mjlab.terrains.config import ROUGH_TERRAINS_CFG


events["randomize_terrain"] = EventTermCfg(
  func=mdp.randomize_terrain,
  mode="reset",
)

return ManagerBasedRlEnvCfg(
  scene=SceneCfg(
    terrain=TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=ROUGH_TERRAINS_CFG,
      max_init_terrain_level=4,
    ),
    num_envs=1,
  ),
  observations=observations,
  actions=actions,
  commands=commands,
  events=events,
  rewards=rewards,
  terminations=terminations,
  viewer=viewer,
  sim=sim,
  decimation=4,
  episode_length_s=10.0,
)""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_scene_impl,
            content=(
                "HoloMotion threads action-delay domain randomization into the "
                "Unitree actuator configuration instead of treating action "
                "latency as an evaluation-only concern."
            ),
            text="""
    action_delay_cfg = copy.deepcopy(
        domain_rand_config.get("action_delay", {})
    )
    if action_delay_cfg.get("enabled", False):
        delay_kwargs = {
            "min_delay": int(action_delay_cfg.get("min_delay", 0)),
            "max_delay": int(action_delay_cfg.get("max_delay", 0)),
        }
    else:
        delay_kwargs = {"min_delay": 0, "max_delay": 0}""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_action_manager,
            content=(
                "MJLab currently stores and routes the current policy action "
                "directly to action terms, preserving history for observation "
                "but not delaying what reaches the actuator target."
            ),
            text="""
    # Shift history: prev_prev ← prev ← current ← new.
    self._prev_prev_action[:] = self._prev_action
    self._prev_action[:] = self._action
    self._action[:] = action.to(self.device)
    # Split the flat action vector and route each slice to its term.
    idx = 0
    for term in self._terms.values():
      term_actions = action[:, idx : idx + term.action_dim]
      term.process_actions(term_actions)
      idx += term.action_dim""",
        ),
        change_direction=(
            "Add policy-step action delay to MJLab's action manager or base "
            "action term: configure min/max lag per environment, sample a lag "
            "on reset or step, push raw policy actions into a bounded buffer, "
            "and route the delayed action slice to each action term."
        ),
        change_reason=(
            "Robust humanoid tracking must tolerate controller, inference, and "
            "communication latency. Keeping only last-action history improves "
            "observability, but it does not train the closed loop under delayed "
            "actuation; HoloMotion makes that latency part of the environment."
        ),
        changed_code="""
@dataclass(kw_only=True)
class ActionDelayCfg:
  min_lag: int = 0
  max_lag: int = 0
  per_env: bool = True
  resample_on_reset: bool = True


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg:
  action_delay: ActionDelayCfg = field(default_factory=ActionDelayCfg)


class ActionManager(ManagerBase):
  def _sample_action_lag(self, env_ids: torch.Tensor | slice | None = None) -> None:
    cfg = self._env.cfg.action_delay
    if cfg.max_lag <= 0:
      self._action_lag[:] = 0
      return
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(self.num_envs, device=self.device)
    self._action_lag[env_ids] = torch.randint(
      cfg.min_lag,
      cfg.max_lag + 1,
      (len(env_ids),),
      device=self.device,
    )

  def process_action(self, action: torch.Tensor) -> None:
    if self.total_action_dim != action.shape[1]:
      raise ValueError(
        f"Invalid action shape, expected: {self.total_action_dim},"
        f" received: {action.shape[1]}."
      )
    self._prev_prev_action[:] = self._prev_action
    self._prev_action[:] = self._action
    self._action[:] = action.to(self.device)
    self._action_delay_buffer.append(self._action.clone())
    delayed_action = self._action_delay_buffer.gather_by_lag(self._action_lag)

    idx = 0
    for term in self._terms.values():
      term_actions = delayed_action[:, idx : idx + term.action_dim]
      term.process_actions(term_actions)
      idx += term.action_dim""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_domain_rand_cfg,
            content=(
                "HoloMotion randomizes more than pushes and friction: default "
                "joint bias, torso COM, pelvis/torso mass, material properties, "
                "disturbance pushes, and actuator gains are all configured."
            ),
            text="""
  default_dof_pos_bias:
    mode: startup
    params:
      joint_names: [".*"]
      pos_distribution_params: [-0.01, 0.01]
      operation: add
      distribution: uniform

  rigid_body_com:
    mode: startup
    params:
      body_names: torso_link
      com_range:
        x: [-0.075, 0.075]
        y: [-0.1, 0.1]
        z: [-0.1, 0.1]

  randomize_mass:
    mode: startup
    params:
      body_names:
        - "pelvis"
        - "torso_link"
      mass_range: [-1.0, 2.0]""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_tracking_env_cfg,
            content=(
                "MJLab's tracking domain randomization has useful pushes, COM "
                "offset, encoder bias, and foot friction, but it does not yet "
                "cover action latency, default joint bias, body mass, material "
                "restitution, or actuator gain variation."
            ),
            text="""
  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),""",
        ),
        change_direction=(
            "Expand MJLab's tracking startup randomization to cover nominal "
            "joint-position bias, pelvis/torso mass, wider torso COM offsets, "
            "full material parameters, and actuator gains while keeping the "
            "existing push, encoder-bias, and shared foot-friction events."
        ),
        change_reason=(
            "The current environment covers several disturbances, but a "
            "general tracker also needs robustness to calibration drift, "
            "body-parameter mismatch, contact-material variation, and actuator "
            "gain mismatch. These are environment robustness gaps, not simulator "
            "API differences."
        ),
        changed_code="""
events.update(
  {
    "default_joint_pos_bias": EventTermCfg(
      mode="startup",
      func=dr.default_joint_pos_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "add",
        "ranges": (-0.01, 0.01),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": {
          0: (-0.075, 0.075),
          1: (-0.10, 0.10),
          2: (-0.10, 0.10),
        },
      },
    ),
    "body_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis", "torso_link")),
        "operation": "add",
        "ranges": (-1.0, 2.0),
      },
    ),
    "actuator_gains": EventTermCfg(
      mode="startup",
      func=dr.actuator_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "stiffness_range": (0.9, 1.1),
        "damping_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
  }
)""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_termination_cfg,
            content=(
                "HoloMotion terminates when the pelvis, ankles, or wrists drift "
                "too far from reference height, not just when the anchor fails."
            ),
            text="""
  keybody_ref_z_far:
    params:
      threshold: 0.25
      ref_prefix: ${rewards._config.reward_prefix}
      keybody_names:
        - pelvis
        - left_ankle_roll_link
        - right_ankle_roll_link
        - left_wrist_yaw_link
        - right_wrist_yaw_link""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_g1_env_cfg,
            content=(
                "MJLab binds the end-effector height termination to ankles and "
                "wrists only, leaving pelvis drift to the separate anchor check "
                "instead of using one key-body safety set."
            ),
            text="""
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )""",
        ),
        change_direction=(
            "Promote MJLab's end-effector termination into a key-body tracking "
            "termination that includes pelvis plus ankles and wrists, and add "
            "an optional full-position key-body termination for late-stage "
            "training once policies are no longer collapsing early."
        ),
        change_reason=(
            "Whole-body motion tracking can fail through pelvis drift while the "
            "hands and feet still satisfy a z-only check. A pelvis-inclusive "
            "key-body termination catches fallen or badly translated states "
            "earlier and makes termination semantics match the generalized "
            "motion-tracking target."
        ),
        changed_code="""
KEY_BODY_TERMINATION_NAMES = (
  "pelvis",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)

cfg.terminations["key_body_z"] = TerminationTermCfg(
  func=mdp.bad_motion_body_pos_z_only,
  params={
    "command_name": "motion",
    "threshold": 0.25,
    "body_names": KEY_BODY_TERMINATION_NAMES,
  },
)
cfg.terminations["key_body_pos"] = TerminationTermCfg(
  func=mdp.bad_motion_body_pos,
  params={
    "command_name": "motion",
    "threshold": 0.5,
    "body_names": KEY_BODY_TERMINATION_NAMES,
  },
)
cfg.terminations.pop("ee_body_pos", None)""",
    ),
]
