from project_analysis import File, Highlight, ProofFromCode

task_registry = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py")
g1_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py")
tracking_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py")
terrain_entity = File(path="mjlab/src/mjlab/terrains/terrain_entity.py")
scene_impl = File(path="mjlab/src/mjlab/scene/scene.py")
manager_env = File(path="mjlab/src/mjlab/envs/manager_based_rl_env.py")

scene_correct = ProofFromCode(
    highlights=[
        ProofFromCode(
            highlights=[
                Highlight(
                    file=task_registry,
                    content="""
The scoped task is exactly `Mjlab-Tracking-Flat-Unitree-G1`: it is registered separately from the no-state-estimation variant and uses the default G1 flat tracking config, whose `has_state_estimation` parameter defaults to true.""",
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
The no-state-estimation behavior is gated behind `if not has_state_estimation`; therefore the target task above keeps the state-estimation-dependent actor observations because it calls the factory without overriding the true default.""",
                    text="""
  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=g1_env_cfg,
                    content="""
The target environment inserts a single named scene entity, `robot`, built by `get_g1_robot_cfg()`, so the scene contains the Unitree G1 robot under the entity name used by actions, commands, terminations, and sensors.""",
                    text="""
  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),""",
                ),
                Highlight(
                    file=g1_env_cfg,
                    content="""
The self-collision sensor is not merely declared locally: the G1 config assigns it into `cfg.scene.sensors`, requesting contact presence and force, no reduction, one slot, and history length four. This matches the environment decimation and lets the reward/termination pipeline see brief self-contact impulses over a full policy step.""",
                    text="""
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)""",
                ),
                Highlight(
                    file=scene_impl,
                    content="""
Scene construction materializes the configured terrain, robot entity, and sensor objects before compilation; sensors edit the MuJoCo spec using the already-added entities, so the self-collision sensor is attached to the actual G1 scene rather than floating as unused config.""",
                    text="""
    self._spec = mujoco.MjSpec.from_file(str(_SCENE_XML))
    if self._cfg.extent is not None:
      self._spec.stat.extent = self._cfg.extent
    self._add_terrain()
    self._add_entities()
    self._add_sensors()
    if self._cfg.spec_fn is not None:
      self._cfg.spec_fn(self._spec)""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=tracking_env_cfg,
                    content="""
The base tracking config creates a `SceneCfg` with a `TerrainEntityCfg(terrain_type="plane")`, so the target flat G1 task starts from a flat plane rather than a rough or generated terrain.""",
                    text="""
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,""",
                ),
                Highlight(
                    file=terrain_entity,
                    content="""
At terrain build time, `terrain_type == "plane"` imports a ground plane and uses grid origins, with no generated heightfield or sub-terrain curriculum. That is the concrete implementation behind the flat-plane scene claim.""",
                    text="""
    elif self.cfg.terrain_type == "plane":
      self._import_ground_plane("terrain")
      self._configure_env_origins()
      self._flat_patches: dict[str, torch.Tensor] = {}
      self._flat_patch_radii: dict[str, float] = {}""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=tracking_env_cfg,
                    content="""
The simulation is configured at a 0.005 second MuJoCo timestep with ten solver iterations, twenty line-search iterations, and a decimation of four. The resulting policy/environment step is 0.02 seconds, which is a stable 50 Hz control cadence for the tracking task.""",
                    text="""
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,""",
                ),
                Highlight(
                    file=manager_env,
                    content="""
The runtime environment enforces that timing relationship: every action is applied for `cfg.decimation` MuJoCo physics steps, scene state and sensors update at the physics timestep, and later reward/termination logic uses the computed environment step duration.""",
                    text="""
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
      self.metrics_manager.compute_substep()""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=tracking_env_cfg,
                    content="""
The tracking scene is stabilized with explicit domain-randomization events: interval pushes train disturbance recovery, startup COM offset covers mass-model variation, encoder bias covers state-estimation noise, and shared foot friction randomization covers contact variation while keeping all foot geoms physically consistent.""",
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
      func=dr.body_com_offset,""",
                ),
                Highlight(
                    file=tracking_env_cfg,
                    content="""
The same event table includes encoder-bias and foot-friction startup randomization, grounding the claim that stable training is supported by randomized sensor and contact conditions rather than a brittle single deterministic scene.""",
                    text="""
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
      func=dr.geom_friction,""",
                ),
                Highlight(
                    file=tracking_env_cfg,
                    content="""
The termination set stops rollouts on timeouts, bad anchor height, bad anchor orientation, and bad end-effector/body height tracking; these are the concrete stability checks that prevent training from continuing through fallen or badly divergent motion-tracking states.""",
                    text="""
  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,""",
                ),
                Highlight(
                    file=g1_env_cfg,
                    content="""
The robot-specific G1 config fills in the body names for the end-effector tracking termination, binding the generic stability rule to G1 ankles and wrists rather than leaving the base config's placeholder empty.""",
                    text="""
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )""",
                ),
            ]
        ),
    ]
)
