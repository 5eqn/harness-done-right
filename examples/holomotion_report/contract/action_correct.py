from project_analysis import File, Highlight, ProofFromCode

training_cfg = File(
    path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_mlp.yaml"
)
robot_cfg = File(
    path="HoloMotion/holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"
)
motion_env_cfg = File(path="HoloMotion/holomotion/config/env/motion_tracking.yaml")
motion_actor_cfg = File(
    path="HoloMotion/holomotion/config/modules/motion_tracking/motion_tracking_mlp.yaml"
)
motion_actor_tf_moe_cfg = File(
    path="HoloMotion/holomotion/config/modules/motion_tracking/motion_tracking_tf-moe.yaml"
)
domain_rand_medium = File(
    path="HoloMotion/holomotion/config/env/domain_randomization/domain_rand_medium.yaml"
)
isaaclab_actions = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_actions.py"
)
isaaclab_scene = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_scene.py"
)
unitree_actuators = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/unitree_actuators.py"
)
motion_env_py = File(path="HoloMotion/holomotion/src/env/motion_tracking.py")
algo_base = File(path="HoloMotion/holomotion/src/algo/algo_base.py")
ppo_py = File(path="HoloMotion/holomotion/src/algo/ppo.py")
agent_modules = File(path="HoloMotion/holomotion/src/modules/agent_modules.py")
onnx_export = File(path="HoloMotion/holomotion/src/utils/onnx_export.py")
mujoco_eval = File(path="HoloMotion/holomotion/src/evaluation/eval_mujoco_sim2sim.py")
eval_cfg = File(path="HoloMotion/holomotion/config/evaluation/eval_mujoco_sim2sim.yaml")


def make_proof(*, highlights: list[Highlight | ProofFromCode]) -> ProofFromCode:
    return ProofFromCode(highlights=highlights)


training_stack = make_proof(
    highlights=[
        Highlight(
            file=training_cfg,
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
            content="""
The top-level defaults make the proof concrete: the environment below is `motion_tracking`, the robot below is the Unitree G1 29-DOF IsaacLab config, and the default actor module is `motion_tracking_mlp`.""",
        ),
    ],
)

action_dimension = make_proof(
    highlights=[
        Highlight(
            file=robot_cfg,
            text="""
  dof_obs_size: 29
  actions_dim: 29
  num_bodies: 30""",
            content="""
The robot config declares the policy action dimension as 29 for the G1 29-DOF setup.""",
        ),
        Highlight(
            file=algo_base,
            text="""
    def _setup_configs(self) -> None:
        self.num_envs: int = self.env.config.num_envs
        self.num_privileged_obs = 0
        self.num_actions = self.env.config.robot.actions_dim""",
            content="""
The algorithm does not infer an unrelated action size; it takes the action count directly from `robot.actions_dim`.""",
        ),
        Highlight(
            file=motion_actor_cfg,
            text="""
    output_dim: robot_action_dim""",
            content="""
The default MLP motion-tracking actor asks for the robot action dimension as its output dimension.""",
        ),
        Highlight(
            file=motion_actor_tf_moe_cfg,
            text="""
    output_dim: robot_action_dim""",
            content="""
The alternate TF-MoE motion-tracking actor uses the same robot action dimension placeholder.""",
        ),
        Highlight(
            file=agent_modules,
            text="""
        # Resolve output_dim placeholders when present.
        if "output_dim" in module_config_dict:
            output_dim = module_config_dict["output_dim"]
            if isinstance(output_dim, list):
                raise ValueError(
                    "PPOActor expects module_config_dict.output_dim to be a scalar. "
                    "List-valued output_dim is not supported."
                )
            if output_dim == "robot_action_dim":
                module_config_dict["output_dim"] = num_actions""",
            content="""
`PPOActor` resolves `robot_action_dim` to `num_actions`, so the actor's mean/action output dimension becomes 29 for this robot.""",
        ),
        Highlight(
            file=ppo_py,
            text="""
        self.actor = PPOActor(
            obs_schema=actor_schema,
            module_config_dict=actor_cfg,
            num_actions=self.num_actions,
            init_noise_std=self.config.init_noise_std,
            obs_example=sample_td,
        ).to(self.device)""",
            content="""
PPO passes that resolved 29-action count into the actor constructor used during training.""",
        ),
        Highlight(
            file=agent_modules,
            text="""
        if mode == "inference":
            actions_out = mu
            td.set("actions", actions_out)
            return td

        self.distribution = Normal(mu, sigma)
        if mode == "sampling":
            actions_out = self.distribution.sample()
        else:
            if actions is None:
                raise ValueError("actions must be provided when mode='logp'")
            actions_out = actions

        td.set("actions", actions_out)""",
            content="""
The actor returns the 29-D mean as deterministic inference actions, or samples a 29-D Normal action in rollout/training mode.""",
        ),
    ],
)

isaaclab_action_application = make_proof(
    highlights=[
        Highlight(
            file=motion_env_cfg,
            text="""
    actions:
      dof_pos:
        type: joint_position
        params:
          asset_name: robot
          joint_names:
            - ".*"
          use_default_offset: true
          scale: ${robot.actuators.action_scale}""",
            content="""
The environment has a single policy action term named `dof_pos`; it targets all robot joints, uses default joint-position offset, and scales raw actions with `robot.actuators.action_scale`.""",
        ),
        Highlight(
            file=robot_cfg,
            text="""
    action_scale:
      ".*_hip_pitch_joint": 0.548
      ".*_hip_roll_joint": 0.351
      ".*_hip_yaw_joint": 0.548
      ".*_knee_joint": 0.351
      ".*_ankle_pitch_joint": 0.439
      ".*_ankle_roll_joint": 0.439
      "waist_yaw_joint": 0.548
      "waist_roll_joint": 0.439
      "waist_pitch_joint": 0.439
      ".*_shoulder_pitch_joint": 0.439
      ".*_shoulder_roll_joint": 0.439
      ".*_shoulder_yaw_joint": 0.439
      ".*_elbow_joint": 0.439
      ".*_wrist_roll_joint": 0.439
      ".*_wrist_pitch_joint": 0.075
      ".*_wrist_yaw_joint": 0.075""",
            content="""
The scale is a joint-pattern mapping, not a scalar fallback; hips, knees, ankles, waist, shoulders, elbows, and wrists each get explicit scale values.""",
        ),
        Highlight(
            file=isaaclab_actions,
            text="""
    def joint_position_action(
        asset_name: str = "robot",
        joint_names: list[str] | None = None,
        use_default_offset: bool = True,
        scale: float = 1.0,
    ) -> mdp.JointPositionActionCfg:
        \"\"\"Joint position control action.\"\"\"
        if joint_names is None:
            joint_names = [".*"]
        return mdp.JointPositionActionCfg(
            asset_name=asset_name,
            joint_names=joint_names,
            use_default_offset=use_default_offset,
            scale=scale,
        )""",
            content="""
The local action builder concretely turns the YAML action into IsaacLab's `JointPositionActionCfg`, carrying through the all-joint selector, default-offset flag, and scale mapping.""",
        ),
        Highlight(
            file=isaaclab_actions,
            text="""
        if action_type == "joint_position":
            action_term = ActionFunctions.joint_position_action(**params)
        elif action_type == "joint_velocity":
            action_term = ActionFunctions.joint_velocity_action(**params)
        elif action_type == "joint_effort":
            action_term = ActionFunctions.joint_effort_action(**params)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        setattr(actions_cfg, action_name, action_term)""",
            content="""
`build_actions_config` keeps the action name from the YAML (`dof_pos`) and attaches the corresponding IsaacLab action term to the environment config.""",
        ),
        Highlight(
            file=motion_env_py,
            text="""
            curriculum: CurriculumCfg = build_curriculum_config(
                _curriculum_config_dict
            )
            actions: ActionsCfg = build_actions_config(_actions_config_dict)
            sim: SimulationCfg = SimulationCfg(
                dt=dt,
                render_interval=decimation,
                physx=physx,
                device=_device,
                enable_scene_query_support=True,
            )""",
            content="""
The motion-tracking environment installs the built action term in the IsaacLab `ManagerBasedRLEnvCfg` class.""",
        ),
        Highlight(
            file=algo_base,
            text="""
        actions = actor_out.get("actions")
        self._last_rollout_actions = actions
        obs_dict, rewards, dones, time_outs, infos = self.env.step(actions)""",
            content="""
The PPO rollout takes the actor's 29-D `actions` tensor and sends it directly to the environment step.""",
        ),
        Highlight(
            file=motion_env_py,
            text="""
    def step(self, actor_state: dict):
        obs_dict, rewards, terminated, time_outs, infos = self._env.step(
            actor_state
        )""",
            content="""
The wrapper delegates that tensor to IsaacLab's `ManagerBasedRLEnv.step`, where the configured `dof_pos` action term is applied.""",
        ),
    ],
)

delay_and_ema = make_proof(
    highlights=[
        Highlight(
            file=domain_rand_medium,
            text="""
domain_rand:
  action_delay:
    enabled: true
    min_delay: 0
    max_delay: 2""",
            content="""
The default training stack enables randomized actuator delay in the medium domain-randomization config used by the training defaults.""",
        ),
        Highlight(
            file=isaaclab_scene,
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
            content="""
Scene construction converts the domain-randomization action-delay section into actuator `min_delay` and `max_delay` kwargs.""",
        ),
        Highlight(
            file=robot_cfg,
            text="""
  actuators:
    actuator_type: unitree_erfi # implicit, unitree, or unitree_erfi
    ema_filter_enabled: false
    ema_filter_alpha: 1.0""",
            content="""
The selected actuator family supports ERFI/EMA, but the default robot config disables EMA filtering and sets alpha to 1.0.""",
        ),
        Highlight(
            file=isaaclab_scene,
            text="""
    if config.get("actuator_type", "unitree") == "unitree_erfi":
        erfi_cfg = copy.deepcopy(domain_rand_config.get("erfi", {}))
        actuator_filter_kwargs = {
            "ema_filter_enabled": bool(
                config.get("ema_filter_enabled", False)
            ),
            "ema_filter_alpha": config.get("ema_filter_alpha", 1.0),
        }""",
            content="""
The robot config's EMA fields are passed into the Unitree ERFI actuator config when that actuator type is selected.""",
        ),
        Highlight(
            file=isaaclab_scene,
            text="""
        actuator_cfg = UnitreeErfiActuatorCfg(**actuator_kwargs)
        actuator_cfg.class_type = UnitreeErfiActuator
    else:
        actuator_kwargs = {**base_kwargs, **delay_kwargs}
        actuator_cfg = UnitreeActuatorCfg(**actuator_kwargs)
        actuator_cfg.class_type = UnitreeActuator""",
            content="""
With `actuator_type: unitree_erfi`, the configured actuator class is `UnitreeErfiActuator`; otherwise HoloMotion uses the simpler `UnitreeActuator` with the same delay kwargs.""",
        ),
        Highlight(
            file=unitree_actuators,
            text="""
    def _filter_joint_position_action(
        self, control_action: ArticulationActions
    ) -> ArticulationActions:
        if not self.cfg.ema_filter_enabled:
            self._maybe_dump_ema_filter_debug_skip("ema_filter_disabled")
            return control_action
        if control_action.joint_positions is None:
            self._maybe_dump_ema_filter_debug_skip("joint_positions_none")
            return control_action""",
            content="""
At runtime the ERFI actuator bypasses filtering when `ema_filter_enabled` is false, exactly matching the default robot config.""",
        ),
        Highlight(
            file=unitree_actuators,
            text="""
        filtered_joint_positions = raw_joint_positions.clone()
        if torch.any(~needs_init):
            filtered_joint_positions = torch.where(
                needs_init.unsqueeze(-1),
                raw_joint_positions,
                self._ema_filter_alpha * raw_joint_positions
                + (1.0 - self._ema_filter_alpha) * self._ema_filter_state,
            )
        self._maybe_dump_ema_filter_debug_verification(
            raw_joint_positions=raw_joint_positions,
            filtered_joint_positions=filtered_joint_positions,
            previous_filtered_joint_positions=previous_filtered_joint_positions,
            needs_init=needs_init,
        )
        self._ema_filter_state[:] = filtered_joint_positions
        self._ema_filter_initialized[:] = True""",
            content="""
If EMA is enabled, the filtered joint-position target is `alpha * raw + (1 - alpha) * previous_filtered`, with reset-time initialization handled per environment.""",
        ),
    ],
)

isaaclab_torque_application = make_proof(
    highlights=[
        Highlight(
            file=isaaclab_scene,
            text="""
            actuators=actuators,
        )

        return articulation_cfg""",
            content="""
The robot articulation receives the actuator config built from the Unitree actuator setup.""",
        ),
        Highlight(
            file=isaaclab_scene,
            text="""
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "waist_yaw_joint",
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],""",
            content="""
The Unitree actuator group covers all anatomical joint families controlled by the 29-D action.""",
        ),
        Highlight(
            file=robot_cfg,
            text="""
      stiffness:
        ".*_hip_pitch_joint": 40.17923847
        ".*_hip_roll_joint": 99.09842778
        ".*_hip_yaw_joint": 40.17923847
        ".*_knee_joint": 99.09842778
        ".*_ankle_pitch_joint": 28.50124620
        ".*_ankle_roll_joint": 28.50124620
        "waist_yaw_joint": 40.17923847
        "waist_roll_joint": 28.50124620
        "waist_pitch_joint": 28.50124620
        ".*_shoulder_pitch_joint": 14.25062310
        ".*_shoulder_roll_joint": 14.25062310
        ".*_shoulder_yaw_joint": 14.25062310
        ".*_elbow_joint": 14.25062310
        ".*_wrist_roll_joint": 14.25062309787429
        ".*_wrist_pitch_joint": 16.77832748089279
        ".*_wrist_yaw_joint": 16.77832748089279

      damping:
        ".*_hip_pitch_joint": 2.55788977
        ".*_hip_roll_joint": 6.30880185
        ".*_hip_yaw_joint": 2.55788977
        ".*_knee_joint": 6.30880185
        ".*_ankle_pitch_joint": 1.81444569
        ".*_ankle_roll_joint": 1.81444569
        "waist_yaw_joint": 2.55788977
        "waist_roll_joint": 1.81444569
        "waist_pitch_joint": 1.81444569
        ".*_shoulder_pitch_joint": 0.90722284
        ".*_shoulder_roll_joint": 0.90722284
        ".*_shoulder_yaw_joint": 0.90722284
        ".*_elbow_joint": 0.90722284
        ".*_wrist_roll_joint": 0.907222843292423
        ".*_wrist_pitch_joint": 1.06814150219
        ".*_wrist_yaw_joint": 1.06814150219""",
            content="""
The PD gains are configured per joint pattern in the robot file, so position targets are converted by nonzero stiffness/damping gains.""",
        ),
        Highlight(
            file=unitree_actuators,
            text="""
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg""",
            content="""
The custom Unitree actuator is built on IsaacLab's delayed PD actuator classes.""",
        ),
        Highlight(
            file=unitree_actuators,
            text="""
class UnitreeActuator(DelayedPDActuator):
    \"\"\"Unitree actuator class that implements a torque-speed curve for the actuators.""",
            content="""
`UnitreeActuator` is an IsaacLab delayed-PD actuator subclass, so its control input is converted through that PD actuator machinery.""",
        ),
        Highlight(
            file=unitree_actuators,
            text="""
    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # calculate the desired joint torques
        control_action = super().compute(control_action, joint_pos, joint_vel)""",
            content="""
The Unitree actuator first stores joint velocities and then delegates desired torque calculation to IsaacLab's delayed PD base implementation.""",
        ),
        Highlight(
            file=unitree_actuators,
            text="""
        # apply friction model on the torque
        self.applied_effort -= (
            self._friction_static
            * torch.tanh(joint_vel / self._activation_vel)
            + self._friction_dynamic * joint_vel
        )

        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort""",
            content="""
After PD computation, HoloMotion applies the Unitree friction model and replaces position/velocity commands with the resulting effort command.""",
        ),
        Highlight(
            file=unitree_actuators,
            text="""
    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # check if the effort is the same direction as the joint velocity
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(
            same_direction, self._effort_y1, self._effort_y2
        )
        # check if the joint velocity is less than the max speed at full torque
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1,
            max_effort,
            self._compute_effort_limit(max_effort),
        )
        return torch.clip(effort, -max_effort, max_effort)""",
            content="""
The IsaacLab training path clips the computed effort by a Unitree torque-speed curve before it is applied.""",
        ),
    ],
)

mujoco_eval_action_application = make_proof(
    highlights=[
        Highlight(
            file=onnx_export,
            text="""
    metadata = {
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"]
        .data.default_joint_stiffness[0]
        .cpu()
        .tolist(),
        "joint_damping": env.scene["robot"]
        .data.default_joint_damping[0]
        .cpu()
        .tolist(),
        "default_joint_pos": env.scene["robot"]
        .data.default_joint_pos[0]
        .cpu()
        .tolist(),
        "action_scale": env.action_manager.get_term("dof_pos")
        ._scale[0]
        .cpu()
        .tolist(),
    }""",
            content="""
Export records the authoritative IsaacLab joint order, PD gains, default positions, and `dof_pos` action scale into the ONNX file.""",
        ),
        Highlight(
            file=mujoco_eval,
            text="""
        result = {}
        result["action_scale"] = _parse_floats(meta["action_scale"])
        result["kps"] = _parse_floats(meta["joint_stiffness"])
        result["kds"] = _parse_floats(meta["joint_damping"])
        result["default_joint_pos"] = _parse_floats(meta["default_joint_pos"])
        result["joint_names"] = [
            x for x in meta["joint_names"].split(",") if x != ""
        ]""",
            content="""
MuJoCo evaluation parses those same metadata fields back into arrays and names.""",
        ),
        Highlight(
            file=mujoco_eval,
            text="""
        self.dof_names_onnx = meta["joint_names"]
        self.action_scale_onnx = meta["action_scale"].astype(np.float32)
        self.kps_onnx = meta["kps"].astype(np.float32)
        self.kds_onnx = meta["kds"].astype(np.float32)
        self.default_angles_onnx = meta["default_joint_pos"].astype(np.float32)""",
            content="""
The parsed metadata becomes the evaluator's ONNX-order action scale, PD gains, and default joint positions.""",
        ),
        Highlight(
            file=mujoco_eval,
            text="""
        # Map ONNX <-> MJCF for control
        self.onnx_to_mu = [
            self.dof_names_onnx.index(name) for name in self.mjcf_dof_names
        ]
        self.mu_to_onnx = [
            self.mjcf_dof_names.index(name) for name in self.dof_names_onnx
        ]
        self.ref_to_onnx = [
            self.dof_names_ref_motion.index(name)
            for name in self.dof_names_onnx
        ]""",
            content="""
The evaluator explicitly builds the ONNX-to-MuJoCo and reference-order mappings by joint name, preventing an order-only assumption.""",
        ),
        Highlight(
            file=eval_cfg,
            text="""
policy_action_delay_step: 0 # max random action delay in 50 Hz policy steps; 0 disables delay
action_delay_type: "episode" # "episode" samples once per reset, "step" re-samples every policy step""",
            content="""
MuJoCo evaluation defaults to no policy action delay, but exposes the same policy-step delay concept for sim2sim tests.""",
        ),
        Highlight(
            file=mujoco_eval,
            text="""
    def _apply_action_delay(self, raw_actions: np.ndarray) -> np.ndarray:
        raw_actions = np.asarray(raw_actions, dtype=np.float32)
        if self.policy_action_delay_step <= 0:
            return raw_actions.copy()

        expected_buffer_len = max(1, self.policy_action_delay_step + 1)
        if (
            not hasattr(self, "_policy_action_delay_buffer")
            or self._policy_action_delay_buffer.maxlen != expected_buffer_len
        ):
            self._reset_action_delay_randomization()""",
            content="""
When sim2sim delay is configured, raw policy actions go through a delay buffer; otherwise the function returns the action unchanged.""",
        ),
        Highlight(
            file=mujoco_eval,
            text="""
        filtered_actions = (
            self.action_ema_filter_alpha * raw_actions
            + (1.0 - self.action_ema_filter_alpha)
            * self._filtered_actions_onnx
        ).astype(np.float32, copy=False)
        self._filtered_actions_onnx = filtered_actions.copy()
        return self._filtered_actions_onnx.copy()""",
            content="""
The MuJoCo sim2sim path has the same EMA formula when the exported/configured actuator settings enable it.""",
        ),
        Highlight(
            file=mujoco_eval,
            text="""
        onnx_output = self.policy_session.run(output_names, input_feed)
        if self.dump_onnx_io_npy:
            self._record_onnx_io_frame(input_feed, output_names, onnx_output)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        raw_actions_onnx = onnx_output[0].reshape(-1)
        filtered_actions_onnx = self._apply_action_ema_filter(raw_actions_onnx)
        self.actions_onnx = self._apply_action_delay(filtered_actions_onnx)""",
            content="""
The ONNX policy output is flattened into the raw action vector, then EMA and delay are applied before scaling to targets.""",
        ),
        Highlight(
            file=mujoco_eval,
            text="""
        self.target_dof_pos_onnx = (
            self.actions_onnx * self.action_scale_onnx
            + self.default_angles_onnx
        )
        self.target_dof_pos_mu = self.target_dof_pos_onnx[self.onnx_to_mu]
        for i, dof_name in enumerate(self.mjcf_dof_names):
            self.target_dof_pos_by_name[dof_name] = float(
                self.target_dof_pos_mu[i]
            )""",
            content="""
The MuJoCo path applies exactly the same action-scale/default-offset affine transform and then maps targets into MuJoCo actuator/joint order by name.""",
        ),
        Highlight(
            file=mujoco_eval,
            text="""
    def _apply_control(self, sleep: bool):
        \"\"\"Apply PD targets via Unitree lowcmd, step MuJoCo, optionally sleep.\"\"\"
        for _ in range(self.control_decimation):""",
            content="""
Each policy action is held for `control_decimation` low-level MuJoCo steps, matching the 50 Hz policy over a 200 Hz simulator.""",
        ),
        Highlight(
            file=mujoco_eval,
            text="""
                tau = (
                    feedforward_tau
                    + kp * (target_q - current_q)
                    + kd * (target_dq - current_dq)
                )
                if (
                    act_idx in self.actuator_force_range
                    and self.actuator_force_range[act_idx] is not None
                ):
                    min_force, max_force = self.actuator_force_range[act_idx]
                    tau = np.clip(tau, min_force, max_force)
                self.d.ctrl[mu_idx] = tau""",
            content="""
MuJoCo low-level application is an explicit PD torque formula with force-range clipping, written into MuJoCo control slots.""",
        ),
        Highlight(
            file=mujoco_eval,
            text="""
    def _run_eval_step(self, max_steps: int) -> bool:
        self._update_policy()
        self.counter += 1
        self._apply_control(sleep=True)
        if self._video_writer is not None:
            self._maybe_record_frame()
        return self._advance_eval_frame(max_steps)""",
            content="""
A sim2sim evaluation step always updates the policy target before applying the low-level clipped PD controller.""",
        ),
    ],
)

action_highlights: list[Highlight | ProofFromCode] = [
    training_stack,
    action_dimension,
    isaaclab_action_application,
    delay_and_ema,
    isaaclab_torque_application,
    mujoco_eval_action_application,
]

action_correct = make_proof(highlights=action_highlights)
