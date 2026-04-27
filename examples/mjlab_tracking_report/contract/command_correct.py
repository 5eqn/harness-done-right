from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

task_registry = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py")
g1_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py")
tracking_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py")
motion_commands = File(path="mjlab/src/mjlab/tasks/tracking/mdp/commands.py")
command_manager = File(path="mjlab/src/mjlab/managers/command_manager.py")
manager_env = File(path="mjlab/src/mjlab/envs/manager_based_rl_env.py")
train_script = File(path="mjlab/src/mjlab/scripts/train.py")
play_script = File(path="mjlab/src/mjlab/scripts/play.py")
evaluate_script = File(path="mjlab/src/mjlab/tasks/tracking/scripts/evaluate.py")

command_correct = ProofFromCode(
  highlights=[
    Highlight(
      file=task_registry,
      content="""
Scope anchor: this command proof is for exactly `Mjlab-Tracking-Flat-Unitree-G1`. The registered training and play configs both call `unitree_g1_flat_tracking_env_cfg()` without `has_state_estimation=False`; the adjacent no-state-estimation task is a separate task id and is outside this proof.""",
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
      file=task_registry,
      content="""
The registry explicitly separates the no-state-estimation variant. Because the requested task id is not this one, none of the command proof below relies on the no-state-estimation config branch.""",
      text="""
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
The Unitree G1 flat tracking config defaults to state estimation. For the scoped task above, the registry calls this function with no override, so `has_state_estimation` remains True.""",
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
The G1 specialization fixes the command anchor and the ordered body list. The anchor is `torso_link`; the body order is pelvis, left leg chain, right leg chain, torso, left arm chain, right arm chain. Later body-index resolution preserves this order, so rewards, critic body observations, relative pose alignment, and debug visuals all refer to these same 14 bodies.""",
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
The base tracking environment registers a single command term named `motion`. It has effectively infinite time-based resampling, enables debug visualization, declares bounded RSI pose/velocity/joint perturbations, and leaves the motion file, anchor, and body list for the robot-specific G1 config and train/play script to fill.""",
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
      file=train_script,
      content="""
Training does not proceed with an empty command dataset. For tracking tasks, the script either keeps an existing local `motion_file`, downloads a W&B motion artifact and points the command at `motion.npz`, or raises with instructions. Thus the runtime `MotionCommand` always receives a concrete `.npz` path.""",
      text="""
  if is_tracking_task:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")
    elif cfg.registry_name:
      # Download from WandB registry.
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      raise ValueError(
        "For tracking tasks, provide either:\\n"
        "  --registry-name your-org/motions/motion-name (download from WandB)\\n"
        "  --env.commands.motion.motion-file /path/to/motion.npz (local file)"
      )""",
    ),
    Highlight(
      file=command_manager,
      content="""
The command manager turns the configured `MotionCommandCfg` into a live `MotionCommand` by calling `term_cfg.build(self._env)`. This is the construction point that invokes `MotionCommand.__init__`, resolves G1 body indexes, and loads the motion file.""",
      text="""
  def _prepare_terms(self):
    for term_name, term_cfg in self.cfg.items():
      term_cfg: CommandTermCfg | None
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      term = term_cfg.build(self._env)
      if not isinstance(term, CommandTerm):
        raise TypeError(
          f"Returned object for the term {term_name} is not of type CommandType."
        )
      self._terms[term_name] = term""",
    ),
    Highlight(
      file=motion_commands,
      content="""
Initialization binds the command to the `robot` scene entity, finds the robot anchor body by name, finds the motion anchor inside the configured G1 command body list, and resolves G1 body indexes with `preserve_order=True`. That order is then passed to `MotionLoader`, so the `.npz` body tensors are sliced into exactly the configured G1 body order.""",
      text="""
    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MotionLoader(
      self.cfg.motion_file, self.body_indexes, device=self.device
    )""",
    ),
    Highlight(
      file=motion_commands,
      content="""
`MotionLoader` is the exact `.npz` contract: it requires `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`. The body tensors are selected by the resolved G1 body indexes, and `time_step_total` is derived from the number of joint-position frames.""",
      text="""
class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = self.joint_pos.shape[0]""",
    ),
    Highlight(
      file=motion_commands,
      content="""
The command observation tensor itself is exactly the current reference joint positions concatenated with current reference joint velocities. Both properties index the same `self.time_steps`, so the 58-dimensional G1 command is one coherent frame: 29 joint positions plus 29 joint velocities.""",
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
      file=motion_commands,
      content="""
The loaded body command fields are also frame-indexed by `self.time_steps`. World positions are shifted by each environment origin, quaternions and velocities come directly from the selected `.npz` frame, and the anchor fields are the configured `torso_link` slice inside the selected command body list.""",
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
      file=command_manager,
      content="""
The command manager's compute loop first updates metrics, then decrements the time-based resampling timer, resamples only if the timer expires, and always calls `_update_command`. Since G1 tracking sets `(1.0e9, 1.0e9)`, normal training/play steps do not time-resample every few seconds; the command advances by dataset frame update until the motion reaches its end.""",
      text="""
  def compute(self, dt: float) -> None:
    self._update_metrics()
    self.time_left -= dt
    resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
    if len(resample_env_ids) > 0:
      self._resample(resample_env_ids)
    self._update_command()""",
    ),
    Highlight(
      file=motion_commands,
      content="""
Dataset stepping is explicit: every command update increments `time_steps` by one frame. Only environments whose frame index reaches `motion.time_step_total` are resampled. This proves that the huge resampling interval means the command keeps advancing through the loaded dataset until the dataset end, rather than drawing a new reference at a short fixed interval.""",
      text="""
  def _update_command(self):
    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    self.update_relative_body_poses()""",
    ),
    Highlight(
      file=command_manager,
      content="""
When a reset or timer resample does occur, the command term refreshes the per-env time-left value, calls the concrete `MotionCommand._resample_command`, and increments the command counter. With the G1 `1e9` range, this timer is effectively a guard; frame-end resampling in `_update_command` is the normal loop point through the dataset.""",
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
      file=motion_commands,
      content="""
All three reset sampling modes are implemented here. `start` pins envs to frame zero, `uniform` samples anywhere in the motion, and the remaining allowed mode is asserted to be `adaptive`, which samples through the failure-bin curriculum path.""",
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
Uniform sampling draws arbitrary frames from the full loaded motion and records maximum entropy metrics. The top-bin fields are set to neutral/default values, which distinguishes uniform from the adaptive failure-biased distribution.""",
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
Adaptive sampling starts by attributing terminated episodes to temporal bins. It maps the current `time_steps` to bin indexes, selects only failed envs, and accumulates those failures in `_current_bin_failed`; this is the failure-bin signal that later reshapes command sampling.""",
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
The adaptive curriculum converts accumulated failure counts plus a uniform floor into a smoothed probability distribution, samples temporal bins with `torch.multinomial`, then samples a random point inside each selected bin. It also logs normalized entropy, the top probability, and the top bin, making the curriculum's concentration observable.""",
      text="""
    # Sample.
    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (self.motion.time_step_total - 1)
    ).long()

    # Update metrics.
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count""",
    ),
    Highlight(
      file=motion_commands,
      content="""
After every frame update, adaptive mode folds the current failure-bin counts into the long-lived bin curriculum with an exponential moving average controlled by `adaptive_alpha`, then clears the current batch. That closes the loop from failed rollouts to future command reset sampling.""",
      text="""
    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()""",
    ),
    Highlight(
      file=motion_commands,
      content="""
The reset-state initialization path takes the chosen reference frame, then applies bounded RSI perturbations to root position, root orientation, root linear velocity, and root angular velocity before writing. These perturbations are active in training because only play mode clears the pose and velocity ranges.""",
      text="""
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
    root_ang_vel += rand_samples[:, 3:]""",
    ),
    Highlight(
      file=motion_commands,
      content="""
RSI also perturbs joint positions around the selected reference frame, while joint velocities remain the exact reference velocities. The final state is immediately passed to `_write_reference_state_to_sim`, so command sampling changes the simulator initial condition, not just the observation target.""",
      text="""
    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )""",
    ),
    Highlight(
      file=motion_commands,
      content="""
The simulation write is centralized and safety-clipped: joint positions are clipped to soft limits, joint state is written, root pose/velocity are concatenated into MuJoCo root state, and the entity is reset. This proves the command's sampled and perturbed reference frame is applied to the Unitree G1 simulator state.""",
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
Relative body pose alignment is recomputed from the reference anchor, robot anchor, and selected body pose. The command aligns reference body poses to the robot anchor yaw and XY while preserving the reference anchor height, then stores `body_quat_relative_w` and `body_pos_relative_w` for rewards, terminations, and critic observations.""",
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
      file=manager_env,
      content="""
The environment reset path writes reset data to MuJoCo, forwards derived quantities, computes commands at `dt=0.0`, senses, and then computes observations. That ordering means reset-time command sampling and sim writes are reflected in the first actor/critic observation.""",
      text="""
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
During normal RL stepping, actions are applied through the MuJoCo decimation loop first; after termination/reward/reset handling and a single `sim.forward()`, the command manager computes the next command frame before interval events, sensing, and observation recomputation. This is where `_update_command` advances the dataset one frame per environment step.""",
      text="""
    self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)

    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)""",
    ),
    Highlight(
      file=g1_env_cfg,
      content="""
Play mode changes the command behavior only for play configs: it disables stochastic actor corruption and interval pushes, clears RSI pose/velocity randomization, and sets `sampling_mode` to `start`. Thus play mode starts at the first motion frame and removes reset-time root randomization while preserving the same Unitree G1 command implementation.""",
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

    motion_cmd.sampling_mode = "start\"""",
    ),
    Highlight(
      file=play_script,
      content="""
The general play script has a demo override: if `_demo_mode` is set, tracking commands use uniform sampling to show multiple motion phases across environments. This is a play/demo diversity path, separate from the registered play config's default `start` sampling.""",
      text="""
  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform\"""",
    ),
    Highlight(
      file=evaluate_script,
      content="""
The dedicated tracking evaluation script forces deterministic start sampling after resolving the motion artifact. This corroborates that evaluation/playback from a trained W&B run starts from the beginning of the reference motion rather than adaptive or uniform reset frames.""",
      text="""
  # Evaluation config.
  motion_cmd.sampling_mode = "start"
  env_cfg.observations["actor"].enable_corruption = True
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = cfg.num_envs""",
    ),
    Highlight(
      file=motion_commands,
      content="""
The dataclass-level defaults confirm the intended command modes and curriculum knobs: adaptive is the training default, the allowed modes are exactly `adaptive`, `uniform`, and `start`, and the adaptive kernel/uniform-ratio/EMA parameters are stored on the command config itself.""",
      text="""
class MotionCommandCfg(CommandTermCfg):
  motion_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive\"""",
    ),
  ]
)
