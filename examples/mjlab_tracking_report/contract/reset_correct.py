from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

g1_registry = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py")
g1_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py")
tracking_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py")
manager_env = File(path="mjlab/src/mjlab/envs/manager_based_rl_env.py")
command_manager = File(path="mjlab/src/mjlab/managers/command_manager.py")
motion_commands = File(path="mjlab/src/mjlab/tasks/tracking/mdp/commands.py")
tracking_terminations = File(
    path="mjlab/src/mjlab/tasks/tracking/mdp/terminations.py"
)
event_manager = File(path="mjlab/src/mjlab/managers/event_manager.py")

reset_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=g1_registry,
            content="""
The scoped task is exactly `Mjlab-Tracking-Flat-Unitree-G1`, and its training and play configs both call `unitree_g1_flat_tracking_env_cfg()` without overriding `has_state_estimation`; the separate no-state-estimation task has a different id and is therefore out of scope.""",
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
The G1 tracking config defaults to `has_state_estimation=True`, so the scoped task keeps state-estimation observations unless the caller explicitly selects the different no-state-estimation registration.""",
            text="""
def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  \"\"\"Create Unitree G1 flat terrain tracking configuration.\"\"\"
  cfg = make_tracking_env_cfg()""",
        ),
        Highlight(
            file=manager_env,
            content="""
A public environment reset first selects all env ids when none are provided, seeds if requested, runs the internal reset, writes reset data into MuJoCo, forwards the simulator, computes commands at `dt=0.0`, senses, and then computes fresh observations. That ordering makes the post-reset observation reflect the reset command and written simulation state.""",
            text="""
  def reset(
    self,
    *,
    seed: int | None = None,
    env_ids: torch.Tensor | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[types.VecEnvObs, dict]:
    del options  # Unused.
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
    if seed is not None:
      self.seed(seed)
    self._reset_idx(env_ids)
    self.scene.write_data_to_sim()
    self.sim.forward()
    self.command_manager.compute(dt=0.0)
    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)
    self.recorder_manager.record_post_reset(env_ids)
    return self.obs_buf, self.extras""",
        ),
        Highlight(
            file=manager_env,
            content="""
The internal reset order is stable and explicit: curricula are computed, the simulator and scene are reset, reset-mode events run if configured, manager buffers are reset in a fixed order, command reset happens before event/termination manager reset, and episode/manual-reset state is cleared last.""",
            text="""
  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    self.curriculum_manager.compute(env_ids=env_ids)
    self.sim.reset(env_ids)
    self.scene.reset(env_ids)

    if "reset" in self.event_manager.available_modes:
      env_step_count = self._sim_step_counter // self.cfg.decimation
      self.event_manager.apply(
        mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
      )

    # NOTE: This is order sensitive.
    self.extras["log"] = dict()
    # observation manager.
    info = self.observation_manager.reset(env_ids)
    self.extras["log"].update(info)
    # action manager.
    info = self.action_manager.reset(env_ids)
    self.extras["log"].update(info)
    # rewards manager.
    info = self.reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    # metrics manager.
    info = self.metrics_manager.reset(env_ids)
    self.extras["log"].update(info)
    # curriculum manager.
    info = self.curriculum_manager.reset(env_ids)
    self.extras["log"].update(info)
    # command manager.
    info = self.command_manager.reset(env_ids)
    self.extras["log"].update(info)
    # event manager.
    info = self.event_manager.reset(env_ids)
    self.extras["log"].update(info)
    # termination manager.
    info = self.termination_manager.reset(env_ids)
    self.extras["log"].update(info)
    # reset the episode length buffer.
    self.episode_length_buf[env_ids] = 0
    self._manual_reset_pending[env_ids] = False""",
        ),
        Highlight(
            file=command_manager,
            content="""
Every command-term reset clears per-command counters, logs and zeros metrics for the reset env ids, and then calls `_resample(env_ids)`. Thus the motion command necessarily receives a reset-time resample call through the environment reset chain.""",
            text="""
  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    extras = {}
    for metric_name, metric_value in self.metrics.items():
      extras[metric_name] = torch.mean(metric_value[env_ids]).item()
      metric_value[env_ids] = 0.0
    self.command_counter[env_ids] = 0
    self._resample(env_ids)
    return extras""",
        ),
        Highlight(
            file=command_manager,
            content="""
`_resample` is guarded against empty resets, refreshes the per-env time-left value from the configured range, invokes the concrete `_resample_command`, and increments the command counter. For the tracking task, that concrete implementation is `MotionCommand._resample_command`.""",
            text="""
  def _resample(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) != 0:
      self.time_left[env_ids] = self.time_left[env_ids].uniform_(
        *self.cfg.resampling_time_range
      )
      self._resample_command(env_ids)
      self.command_counter[env_ids] += 1""",
        ),
        Highlight(
            file=tracking_env_cfg,
            content="""
The base tracking command is the reset source for reference-state initialization. It is configured with effectively infinite time-based resampling, debug visualization, explicit root pose RSI ranges, velocity RSI ranges, and a small joint-position perturbation range before robot-specific fields are filled.""",
            text="""
  commands: dict[str, CommandTermCfg] = {
    "motion": MotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      # Override in robot cfg.
      motion_file="",
      anchor_body_name="",
      body_names=(),
    )
  }""",
        ),
        Highlight(
            file=tracking_env_cfg,
            content="""
The concrete velocity perturbation ranges are bounded and physically modest for RSI and interval pushes: half-meter-per-second horizontal velocity, smaller vertical velocity, and bounded roll/pitch/yaw angular velocity perturbations.""",
            text="""
VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}""",
        ),
        Highlight(
            file=motion_commands,
            content="""
`MotionCommand` supports three reset sampling modes. `start` deterministically uses frame zero, `uniform` samples a random frame uniformly across the motion, and `adaptive` samples from failure-weighted bins; no other mode is accepted because the fallback asserts adaptive.""",
            text="""
  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)""",
        ),
        Highlight(
            file=motion_commands,
            content="""
Uniform sampling chooses arbitrary reference frames over the whole loaded motion, while adaptive sampling uses termination failures to bias bins; that makes reset sampling capable of both broad coverage and targeted recovery from unstable parts of the motion.""",
            text="""
  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = torch.randint(
      0, self.motion.time_step_total, (len(env_ids),), device=self.device
    )
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.""",
        ),
        Highlight(
            file=motion_commands,
            content="""
Adaptive sampling is explicitly coupled to termination results: failed episodes mark the current time bin, and the next sampling probabilities are built from accumulated failure counts plus a uniform floor. That keeps difficult reset frames in the training distribution without collapsing exploration.""",
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
        Highlight(
            file=motion_commands,
            content="""
Reset-state writing is centralized: joint positions are clipped to the robot's per-env soft joint limits, joint state is written, root pose and velocity are concatenated and written, and the robot entity is reset afterward. This is the core stability guarantee for RSI writes.""",
            text="""
  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    \"\"\"Clip joint positions and write root + joint state to sim.\"\"\"
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)""",
        ),
        Highlight(
            file=motion_commands,
            content="""
After choosing the reference frame, RSI copies the reference root state and applies bounded pose, orientation, linear-velocity, angular-velocity, and joint-position perturbations before writing the final clipped state. Joint velocities come directly from the reference frame.""",
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
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )""",
        ),
        Highlight(
            file=g1_env_cfg,
            content="""
Play mode disables stochastic pieces only for play: it extends the episode, turns off actor observation corruption, removes interval pushes, zeros RSI pose/velocity randomization, and pins sampling to the first frame. These overrides are inside `if play`, so training keeps the randomization above.""",
            text="""
  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg""",
        ),
        Highlight(
            file=tracking_env_cfg,
            content="""
Startup randomization is configured on the scoped tracking task: torso COM, encoder bias, and foot friction are randomized once at environment construction, while the interval push randomizes root velocity during training unless play removes it.""",
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
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
  }""",
        ),
        Highlight(
            file=manager_env,
            content="""
Startup events actually run during environment construction when available, and reset events are also applied by `_reset_idx` when configured. The tracking task currently uses startup DR plus command-reset RSI; the reset hook is still present and ordered before manager resets for tasks that configure reset events.""",
            text="""
    # Initialize startup events if defined.
    if "startup" in self.event_manager.available_modes:
      self.event_manager.apply(mode="startup")""",
        ),
        Highlight(
            file=event_manager,
            content="""
The event manager's reset path requires the global step count, supports per-env reset triggering and minimum-step gates, calls the configured reset function with selected env ids, and recomputes MuJoCo constants when a randomization function declares that model constants changed.""",
            text="""
      elif mode == "reset":
        assert global_env_step_count is not None
        min_step_count = term_cfg.min_step_count_between_reset
        if env_ids is None:
          env_ids = slice(None)
        if min_step_count == 0:
          self._reset_term_last_triggered_step_id[index][env_ids] = (
            global_env_step_count
          )
          self._reset_term_last_triggered_once[index][env_ids] = True
          term_cfg.func(self._env, env_ids, **term_cfg.params)
          fired = True
        else:
          last_triggered_step = self._reset_term_last_triggered_step_id[index][env_ids]
          triggered_at_least_once = self._reset_term_last_triggered_once[index][env_ids]
          steps_since_triggered = global_env_step_count - last_triggered_step
          valid_trigger = steps_since_triggered >= min_step_count
          valid_trigger |= (last_triggered_step == 0) & ~triggered_at_least_once
          if isinstance(env_ids, torch.Tensor):
            valid_env_ids = env_ids[valid_trigger]
          else:
            valid_env_ids = valid_trigger.nonzero().flatten()
          if len(valid_env_ids) > 0:
            self._reset_term_last_triggered_once[index][valid_env_ids] = True
            self._reset_term_last_triggered_step_id[index][valid_env_ids] = (
              global_env_step_count
            )
            term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
            fired = True""",
        ),
        Highlight(
            file=tracking_env_cfg,
            content="""
Termination guards are tied directly to drift from the reference motion: root/anchor height drift, anchor orientation drift, and selected end-effector height drift terminate episodes, while timeout is tracked separately as a truncation. This prevents unstable reset samples or failed tracking from drifting indefinitely.""",
            text="""
  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (),  # Set per-robot.
      },
    ),
  }""",
        ),
        Highlight(
            file=g1_env_cfg,
            content="""
For Unitree G1 specifically, the end-effector drift guard is bound to both ankles and wrists, so terminal drift checks cover the important limbs for flat G1 whole-body motion tracking.""",
            text="""
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )""",
        ),
        Highlight(
            file=tracking_terminations,
            content="""
The termination implementations compare live robot state against command reference state, not arbitrary thresholds alone. Anchor height, projected gravity/orientation, and selected body height errors are computed from `MotionCommand` fields and return boolean termination masks per environment.""",
            text="""
def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )""",
        ),
    ]
)
