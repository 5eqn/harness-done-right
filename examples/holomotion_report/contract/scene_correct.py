from project_analysis import File, Highlight, ProofFromCode


train_cfg = File(
    path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_tf-moe.yaml"
)
env_cfg = File(path="HoloMotion/holomotion/config/env/motion_tracking.yaml")
robot_cfg = File(
    path="HoloMotion/holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"
)
rough_terrain_cfg = File(
    path="HoloMotion/holomotion/config/env/terrain/isaaclab_rough.yaml"
)
domain_rand_cfg = File(
    path="HoloMotion/holomotion/config/env/domain_randomization/domain_rand_medium.yaml"
)
termination_cfg = File(
    path="HoloMotion/holomotion/config/env/terminations/termination_motion_tracking.yaml"
)
motion_tracking_env = File(path="HoloMotion/holomotion/src/env/motion_tracking.py")
scene_impl = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_scene.py"
)
terrain_impl = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_terrain.py"
)
domain_rand_impl = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_domain_rand.py"
)
termination_impl = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_termination.py"
)
mujoco_eval = File(
    path="HoloMotion/holomotion/src/evaluation/eval_mujoco_sim2sim.py"
)
mujoco_scene_xml = File(
    path="HoloMotion/assets/robots/unitree/G1/29dof/scene_29dof.xml"
)


scene_correct = ProofFromCode(
    highlights=[
        ProofFromCode(
            highlights=[
                Highlight(
                    file=train_cfg,
                    content="""
The motion-tracking training target composes the Unitree G1 29-DoF robot, the IsaacLab motion-tracking environment, motion-tracking terminations, medium domain randomization, rough IsaacLab terrain, and the TF-MoE module in one Hydra default stack. This means the scene proof is grounded in the actual training configuration rather than an isolated helper.""",
                    text="""
defaults:
  - /training: train_base
  - /algo: ppo_tf
  - /robot: unitree/G1/29dof/29dof_training_isaaclab
  - /env: motion_tracking
  - /env/terminations: termination_motion_tracking
  - /env/observations: motion_tracking/obs_motion_tracking_tf-moe
  - /env/rewards: motion_tracking/rew_motion_tracking
  - /env/domain_randomization: domain_rand_medium
  - /env/terrain: isaaclab_rough
  - /modules: motion_tracking/motion_tracking_tf-moe""",
                ),
                Highlight(
                    file=motion_tracking_env,
                    content="""
At runtime the wrapper converts the resolved Hydra config into a single IsaacLab scene dictionary containing environment count/spacing/physics replication plus the resolved robot, terrain, domain randomization, lighting, and contact sensor configs. This is the concrete handoff from project config into IsaacLab scene construction.""",
                    text="""
            scene_config_dict = {
                "num_envs": self.config.num_envs,
                "env_spacing": self.config.env_spacing,
                "replicate_physics": self.config.replicate_physics,
                "robot": _robot_config_dict,
                "terrain": _terrain_config_dict,
                "domain_rand": _domain_rand_config_dict,
                "lighting": _scene_config_dict.lighting,
                "contact_sensor": _scene_config_dict.contact_sensor,
            }""",
                ),
                Highlight(
                    file=motion_tracking_env,
                    content="""
The scene dictionary is materialized by `build_scene_config` with process metadata, so distributed runs still create the same semantic scene while allowing robot USD generation to be isolated per rank.""",
                    text="""
            scene: MotionTrackingSceneCfg = build_scene_config(
                scene_config_dict,
                main_process=main_process,
                process_id=process_id,
                num_processes=num_processes,
            )""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=robot_cfg,
                    content="""
The selected robot config is Unitree G1 29-DoF: it declares 29 action dimensions, 29 observed DoFs, and 30 bodies, matching the full-body humanoid motion-tracking task rather than a reduced leg-only model.""",
                    text="""
  humanoid_type: unitree/G1/29dof

  dof_obs_size: 29
  actions_dim: 29
  num_bodies: 30""",
                ),
                Highlight(
                    file=robot_cfg,
                    content="""
The robot asset paths point to the G1 29-DoF URDF for IsaacLab conversion and the paired MuJoCo XML for motion assets/sim2sim, with an unfixed base and forced USD conversion. That supports a free-floating humanoid in training instead of a fixed-base manipulator scene.""",
                    text="""
  asset:
    collapse_fixed_joints: True
    replace_cylinder_with_capsule: True
    flip_visual_attachments: False
    max_angular_velocity: 1000.
    max_linear_velocity: 1000.
    density: 0.001
    angular_damping: 0.
    linear_damping: 0.

    asset_root: "./"
    urdf_file: "assets/robots/${robot.humanoid_type}/g1_29dof_rev_1_0.urdf"
    assetFileName: "assets/robots/${robot.humanoid_type}/g1_29dof_rev_1_0.xml"
    fix_base_link: false
    force_usd_conversion: true""",
                ),
                Highlight(
                    file=scene_impl,
                    content="""
`build_robot_config` turns that URDF into an IsaacLab `ArticulationCfg` at `{ENV_REGEX_NS}/Robot`, uses the resolved URDF path and per-rank USD output directory, keeps the base free, merges fixed joints, names the configured root link, and activates contact sensors. Those fields are the actual robot-spawn contract used by the scene.""",
                    text="""
        articulation_cfg = ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UrdfFileCfg(
                asset_path=os.path.abspath(urdf_path),
                usd_dir=os.path.abspath(usd_dir),
                force_usd_conversion=force_usd_conversion,
                fix_base=False,
                merge_fixed_joints=True,
                root_link_name=root_link_name,
                replace_cylinders_with_capsules=True,
                activate_contact_sensors=True,""",
                ),
                Highlight(
                    file=scene_impl,
                    content="""
The articulation starts from the configured base position and default joint angles, zeroes joint velocities, applies a soft joint-position limit factor, and attaches the configured actuator object. This is a stable initial pose and actuation setup for RL rollouts.""",
                    text="""
            init_state=ArticulationCfg.InitialStateCfg(
                pos=init_pos,
                joint_pos=default_joint_positions,
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=0.9,
            actuators=actuators,""",
                ),
                Highlight(
                    file=scene_impl,
                    content="""
The scene builder assigns the robot articulation to `scene_cfg.robot`, then adds terrain, a ray-caster height scanner attached to the robot, lighting, and contact forces. The robot, terrain, height query, and contact sensor therefore coexist in one IsaacLab `InteractiveSceneCfg`.""",
                    text="""
    # Build robot configuration with process info
    if "robot" in scene_config_dict:
        robot_config = scene_config_dict["robot"]
        scene_cfg.robot = SceneFunctions.build_robot_config(
            robot_config,
            domain_rand_config=scene_config_dict.get("domain_rand", {}),
            main_process=main_process,
            process_id=process_id,
            num_processes=num_processes,
        )""",
                ),
                Highlight(
                    file=scene_impl,
                    content="""
Contact sensing is wired from config into `scene_cfg.contact_forces` with a default robot-wide prim path, history length, air-time tracking, force threshold, and debug flag, which is the data path for contact-aware rewards/metrics.""",
                    text="""
    # Build contact sensor configuration
    if "contact_sensor" in scene_config_dict:
        contact_config = scene_config_dict["contact_sensor"]
        scene_cfg.contact_forces = SceneFunctions.build_contact_sensor_config(
            contact_config
        )""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=rough_terrain_cfg,
                    content="""
The training terrain is not the evaluation plane: the selected training default is `isaaclab_rough`, which is a generated `/World/ground` terrain with friction, no restitution, four initial terrain levels, random spawn enabled, and a four-meter margin from terrain edges.""",
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
                Highlight(
                    file=rough_terrain_cfg,
                    content="""
The generated terrain is a 4-by-4 set of 20m rough sub-terrains with random-uniform height noise up to 4cm, so the locomotion scene exposes the policy to mild uneven terrain while keeping the difficulty bounded.""",
                    text="""
    sub_terrains:
      rough:
        type: random_uniform
        proportion: 1.0
        noise_range: [0.0, 0.04]
        noise_step: 0.05
        downsampled_scale: 1.0""",
                ),
                Highlight(
                    file=terrain_impl,
                    content="""
The implementation maps those YAML sub-terrain dictionaries into IsaacLab height-field configs, including `random_uniform`, `plane`, `discrete_obstacles`, and `pyramid_sloped`. For the training config above, the `rough` entry takes the `HfRandomUniformTerrainCfg` branch.""",
                    text="""
            if sub_type == "random_uniform":
                hf_cfg = HfRandomUniformTerrainCfg(
                    proportion=sub_proportion, **sub_params
                )
            elif sub_type == "plane":
                hf_cfg = HfPlaneTerrainCfg(
                    proportion=sub_proportion, **sub_params
                )""",
                ),
                Highlight(
                    file=terrain_impl,
                    content="""
The final terrain importer uses IsaacLab's `TerrainGeneratorCfg`, passes through the generated sub-terrain map, applies physical friction/restitution material, and switches to `RandomSpawnTerrainImporter` when random spawn is enabled. That grounds both the rough terrain geometry and the randomized environment origins.""",
                    text="""
    terrain_generator = terrain_gen.TerrainGeneratorCfg(
        **{
            key: value
            for key, value in generator_params.items()
            if key
            in (
                "size",
                "border_width",
                "border_height",
                "num_rows",
                "num_cols",
                "horizontal_scale",
                "vertical_scale",
                "slope_threshold",
                "difficulty_range",
                "color_scheme",
                "curriculum",
                "seed",
                "use_cache",
                "cache_dir",
            )
        },
        sub_terrains=sub_terrains_cfg,
    )""",
                ),
                Highlight(
                    file=terrain_impl,
                    content="""
Random spawn is a real importer-class change, not just metadata: when `random_spawn` is true, the configured terrain uses `RandomSpawnTerrainImporter`, and that importer samples x/y offsets inside the sub-terrain bounds.""",
                    text="""
    # Configure random spawning within sub-terrains if requested
    random_spawn = config.get("random_spawn", False)
    terrain_importer_class = (
        RandomSpawnTerrainImporter if random_spawn else TerrainImporter
    )""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=env_cfg,
                    content="""
IsaacLab simulation timing is explicit: episodes last 10 seconds, physics runs at 200 Hz, and controls are decimated by four. That yields a 0.005 second physics dt and a 0.02 second policy/action interval, matching a 50 Hz motion-reference cadence.""",
                    text="""
    simulation:
      episode_length_s: 10 # Long episodes for fluid motion-based termination
      sim_freq: 200
      control_decimation: 4
      physx:
        bounce_threshold_velocity: 0.5
        gpu_max_rigid_patch_count: 327680 # 10 * 2**15""",
                ),
                Highlight(
                    file=motion_tracking_env,
                    content="""
The runtime config computes `dt` directly from `sim_freq`, stores the control decimation and episode length in the IsaacLab `ManagerBasedRLEnvCfg`, and enables PhysX stabilization. This makes the 200 Hz physics / 50 Hz control relationship executable rather than merely documented.""",
                    text="""
            decimation: int = _simulation_config_dict.control_decimation
            episode_length_s: int = _simulation_config_dict.episode_length_s
            sim_freq = _simulation_config_dict.sim_freq
            dt = 1.0 / sim_freq
            physx = PhysxCfg(
                bounce_threshold_velocity=_simulation_config_dict.physx.bounce_threshold_velocity,
                gpu_max_rigid_patch_count=_simulation_config_dict.physx.gpu_max_rigid_patch_count,
                enable_stabilization=True,
            )""",
                ),
                Highlight(
                    file=motion_tracking_env,
                    content="""
The final IsaacLab `SimulationCfg` uses that `dt`, sets render interval to the same decimation, keeps scene queries available for sensors/terrain, then reasserts GPU rigid-patch capacity and stabilization before attaching the terrain physics material to the simulation.""",
                    text="""
            sim: SimulationCfg = SimulationCfg(
                dt=dt,
                render_interval=decimation,
                physx=physx,
                device=_device,
                enable_scene_query_support=True,
            )
            sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
            sim.physx.enable_stabilization = True
            sim.physics_material = scene.terrain.physics_material""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=domain_rand_cfg,
                    content="""
Medium domain randomization enables action delay from 0 to 2 policy steps. Because the robot uses Unitree-style actuators, this delay is consumed by the actuator builder and changes low-level actuation timing during training.""",
                    text="""
  action_delay:
    enabled: true
    min_delay: 0
    max_delay: 2""",
                ),
                Highlight(
                    file=scene_impl,
                    content="""
The Unitree actuator builder reads `domain_rand.action_delay` and converts it into `min_delay`/`max_delay` kwargs for the actuator cfg. Thus action latency is part of the robot actuation model inside the scene, not an external training-only perturbation.""",
                    text="""
    action_delay_cfg = copy.deepcopy(
        domain_rand_config.get("action_delay", {})
    )
    if action_delay_cfg.get("enabled", False):
        delay_kwargs = {
            "min_delay": int(action_delay_cfg.get("min_delay", 0)),
            "max_delay": int(action_delay_cfg.get("max_delay", 0)),
        }""",
                ),
                Highlight(
                    file=domain_rand_cfg,
                    content="""
The same randomization config defines startup randomization for default joint-position bias, torso COM offset, pelvis/torso mass, material friction/restitution, and actuator gains, plus interval pushes. Those are the core scene/environment perturbations used to make the learned controller robust instead of overfitting one deterministic world.""",
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
      mass_range: [-1.0, 2.0]

  rigid_body_material:
    mode: startup
    params:
      body_names: ".*"
      static_friction_range: [0.3, 1.6]
      dynamic_friction_range: [0.3, 1.2]
      restitution_range: [0.0, 0.5]
      num_buckets: 64

  push_by_setting_velocity:
    mode: interval
    interval_range_s: [1.0, 3.0]
    params:
      velocity_range:
        x: [-0.5, 0.5]
        y: [-0.5, 0.5]
        z: [-0.2, 0.2]
        roll: [-0.52, 0.52]
        pitch: [-0.52, 0.52]
        yaw: [-0.78, 0.78]

  randomize_actuator_gains:
    mode: startup
    params:
      asset_name: robot
      body_names: ".*"
      stiffness_distribution_params: [0.9, 1.1]
      damping_distribution_params: [0.9, 1.1]
      operation: scale
      distribution: uniform""",
                ),
                Highlight(
                    file=domain_rand_impl,
                    content="""
Only dictionary entries with a `mode` become IsaacLab event terms, which correctly excludes non-event support settings such as observation noise/action-delay metadata while registering actual startup and interval events through the project's `DomainRandFunctions` table.""",
                    text="""
    for event_name, cfg in domain_rand_config_dict.items():
        # Keep non-event config under `domain_rand` available for Hydra
        # references without forcing it through the Isaac Lab event builder.
        if not (isinstance(cfg, dict) and "mode" in cfg):
            continue

        try:
            func = getattr(DomainRandFunctions, f"_get_dr_{event_name}")""",
                ),
                Highlight(
                    file=domain_rand_impl,
                    content="""
The event builder wraps each resolved randomization function in `EventTermCfg` and attaches it to `EventsCfg`, so IsaacLab's manager system can execute the events at startup or interval according to the YAML mode.""",
                    text="""
        term = EventTermCfg(
            func=func,
            **cfg,
        )
        setattr(events_cfg, event_name, term)""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=termination_cfg,
                    content="""
Training stability is guarded by a timeout, a projected-gravity mismatch cutoff, and a key-body z-error cutoff over pelvis, ankles, and wrists. This prevents fallen or severely divergent tracking states from continuing as valid rollouts.""",
                    text="""
terminations:

  time_out:
    time_out: true

  ref_gravity_projection_far:
    params:
      threshold: 0.8
      ref_prefix: ${rewards._config.reward_prefix}

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
                Highlight(
                    file=termination_impl,
                    content="""
Termination config names are resolved to either custom project functions or IsaacLab native terminations, and `time_out` is explicitly marked as timeout rather than failure termination. That preserves the distinction between successful completion and instability.""",
                    text="""
        term_cfg = TerminationTermCfg(
            func=func,
            params=params,
            time_out=(termination_name == "time_out")
            or termination_cfg.get("time_out", False),
        )
        setattr(terminations_cfg, termination_name, term_cfg)""",
                ),
                Highlight(
                    file=termination_impl,
                    content="""
The gravity-projection termination compares the reference and simulated anchor-frame gravity z-components after inverse quaternion projection, so large orientation mismatch is detected even when position alone has not yet diverged.""",
                    text="""
    return (
        motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
    ).abs() > threshold""",
                ),
                Highlight(
                    file=termination_impl,
                    content="""
The key-body z termination measures absolute z error between reference and simulated body positions and terminates if any selected body exceeds the threshold, directly catching pelvis/ankle/wrist height failures.""",
                    text="""
    error_z = (ref_pos_w[..., 2] - robot_pos_w[..., 2]).abs()  # [B, Nb]
    return torch.any(error_z > threshold, dim=-1)  # [B]""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=mujoco_scene_xml,
                    content="""
The MuJoCo scene used for sim2sim includes the G1 29-DoF robot XML and defines viewer statistics/visual setup around a humanoid centered near z=1.0. This is the MuJoCo-side scene counterpart to the IsaacLab training scene.""",
                    text="""
  <include file="g1_29dof.xml"/>

  <!-- setup scene -->
  <statistic center="0.0 0.0 1.0" extent="0.8"/>""",
                ),
                Highlight(
                    file=mujoco_scene_xml,
                    content="""
The MuJoCo worldbody includes a named plane floor with collision enabled, so sim2sim evaluation is not a body-only model: the G1 is evaluated against a concrete ground plane.""",
                    text="""
   <worldbody>
    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" contype="1" conaffinity="0" priority="1" condim="3"/>""",
                ),
                Highlight(
                    file=mujoco_eval,
                    content="""
The MuJoCo evaluator uses the same 200 Hz simulation and 50 Hz policy timing as IsaacLab, with control decimation four and action count derived from the configured robot DOF names. That timing alignment is critical for sim2sim consistency.""",
                    text="""
        self.simulation_dt = 1 / 200
        self.policy_dt = 1 / 50
        self.control_decimation = 4
        self.dof_names_ref_motion = list(config.robot.dof_names)
        self.num_actions = len(self.dof_names_ref_motion)""",
                ),
                Highlight(
                    file=mujoco_eval,
                    content="""
The evaluator loads the configured MuJoCo XML through `MjModel.from_xml_path`, allocates `MjData`, and sets `m.opt.timestep` to the 1/200 second simulation dt, making the XML scene executable with the intended physics step.""",
                    text="""
        logger.info(f"Loading MuJoCo model from {xml_path}")
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt""",
                ),
                Highlight(
                    file=mujoco_eval,
                    content="""
During MuJoCo rollout, low-level PD torques are written to controls, then `mujoco.mj_step` advances the same model/data pair. Optional torque and foot-contact logging is sampled immediately after each physics step, grounding the environment dynamics and contact metrics in the executed MuJoCo scene.""",
                    text="""
            mujoco.mj_step(self.m, self.d)
            if record_low_level_torque:
                self._robot_low_level_dof_torque_seq.append(torque_ref)
                self._record_low_level_foot_contact_sample()
            if sleep:
                time.sleep(self.simulation_dt)""",
                ),
            ]
        ),
    ]
)
