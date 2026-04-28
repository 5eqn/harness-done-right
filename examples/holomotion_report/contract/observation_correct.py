from project_analysis import File, Highlight, ProofFromCode


training_cfg = File(
    path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_mlp.yaml"
)
motion_tracking_env_cfg = File(
    path="HoloMotion/holomotion/config/env/motion_tracking.yaml"
)
motion_obs_cfg = File(
    path="HoloMotion/holomotion/config/env/observations/motion_tracking/obs_motion_tracking_mlp.yaml"
)
motion_modules_cfg = File(
    path="HoloMotion/holomotion/config/modules/motion_tracking/motion_tracking_mlp.yaml"
)
g1_robot_cfg = File(
    path="HoloMotion/holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"
)
motion_tracking_env = File(
    path="HoloMotion/holomotion/src/env/motion_tracking.py"
)
isaaclab_observation = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_observation.py"
)
motion_command = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_motion_tracking_command.py"
)
agent_modules = File(path="HoloMotion/holomotion/src/modules/agent_modules.py")
network_modules = File(path="HoloMotion/holomotion/src/modules/network_modules.py")
eval_obs_builder = File(path="HoloMotion/holomotion/src/evaluation/obs/obs_builder.py")
eval_mujoco = File(
    path="HoloMotion/holomotion/src/evaluation/eval_mujoco_sim2sim.py"
)


observation_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=training_cfg,
            content="""
This pins the proof to the requested HoloMotion training stack. The motion-tracking G1 29-DoF MLP run uses the `motion_tracking` environment, `obs_motion_tracking_mlp` observation config, and `motion_tracking_mlp` module schema.""",
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
        ),
        Highlight(
            file=motion_tracking_env_cfg,
            content="""
The observation terms are collected together with the `ref_motion` command used by all reference-motion observation functions. The command is configured for the same G1 DoF/body lists, a `torso_link` anchor, 10 future frames, and 50 Hz. Actions are joint-position actions over every robot joint, so last-action observation width matches the 29 action dimensions grounded below. `clip_observations` is an environment-level clipping limit; learned normalization is handled in the module proof below.""",
            text="""
    actions:
      dof_pos:
        type: joint_position
        params:
          asset_name: robot
          joint_names:
            - ".*"
          use_default_offset: true
          scale: ${robot.actuators.action_scale}

    commands:
      ref_motion:
        type: MotionCommandCfg
        params:
          command_obs_name: bydmmc_ref_motion
          motion_lib_cfg: ${robot.motion}
          urdf_dof_names: ${robot.dof_names}
          urdf_body_names: ${robot.body_names}
          arm_dof_names: ${robot.arm_dof_names}
          waist_dof_names: ${robot.waist_dof_names}
          leg_dof_names: ${robot.leg_dof_names}
          arm_body_names: ${robot.arm_body_names}
          torso_body_names: ${robot.torso_body_names}
          leg_body_names: ${robot.leg_body_names}
          anchor_bodylink_name: ${robot.anchor_body}
          asset_name: robot
          debug_vis: true
          root_pose_perturb_range: ${domain_rand.motion_init_perturb.root_pose_perturb_range}
          root_vel_perturb_range: ${domain_rand.motion_init_perturb.root_vel_perturb_range}
          dof_pos_perturb_range: ${domain_rand.motion_init_perturb.dof_pos_perturb_range}
          dof_vel_perturb_range: ${domain_rand.motion_init_perturb.dof_vel_perturb_range}
          resample_time_interval_s: 100
          n_fut_frames: ${obs.n_fut_frames}
          target_fps: 50

    normalization:
      clip_actions: 100.0
      clip_observations: 100.0""",
        ),
        Highlight(
            file=g1_robot_cfg,
            content="""
Robot dimensional ground truth. The configured G1 has 29 observed/action DoFs and 30 bodies; its anchor is `torso_link`. This makes DoF position, DoF velocity, reference DoF position, and last action 29 each, while all-body critic bodylink terms use 30 bodies.""",
            text="""
robot:
  humanoid_type: unitree/G1/29dof

  dof_obs_size: 29
  actions_dim: 29
  num_bodies: 30
  num_extend_bodies: 0
  undesired_contacts_regrex: "^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
  torso_name: "torso_link"
  anchor_body: "torso_link\"""",
        ),
        Highlight(
            file=g1_robot_cfg,
            content="""
This is the complete G1 DoF order used by the command and simulator mapping. Counting these names gives the 29-wide DoF/reference/action vectors used in the actor and critic arithmetic.""",
            text="""
  dof_names:
    - "left_hip_pitch_joint"
    - "left_hip_roll_joint"
    - "left_hip_yaw_joint"
    - "left_knee_joint"
    - "left_ankle_pitch_joint"
    - "left_ankle_roll_joint"
    - "right_hip_pitch_joint"
    - "right_hip_roll_joint"
    - "right_hip_yaw_joint"
    - "right_knee_joint"
    - "right_ankle_pitch_joint"
    - "right_ankle_roll_joint"
    - "waist_yaw_joint"
    - "waist_roll_joint"
    - "waist_pitch_joint"
    - "left_shoulder_pitch_joint"
    - "left_shoulder_roll_joint"
    - "left_shoulder_yaw_joint"
    - "left_elbow_joint"
    - "left_wrist_roll_joint"
    - "left_wrist_pitch_joint"
    - "left_wrist_yaw_joint"
    - "right_shoulder_pitch_joint"
    - "right_shoulder_roll_joint"
    - "right_shoulder_yaw_joint"
    - "right_elbow_joint"
    - "right_wrist_roll_joint"
    - "right_wrist_pitch_joint"
    - "right_wrist_yaw_joint\"""",
        ),
        Highlight(
            file=g1_robot_cfg,
            content="""
The full body list has 30 entries, matching `num_bodies`. Critic bodylink flats therefore contribute 30 * 3 = 90 for each position/linear-velocity/angular-velocity all-body term and 30 * 6 = 180 for the 6D rotation-matrix body term.""",
            text="""
  body_names:
    - "pelvis"
    - "left_hip_pitch_link"
    - "left_hip_roll_link"
    - "left_hip_yaw_link"
    - "left_knee_link"
    - "left_ankle_pitch_link"
    - "left_ankle_roll_link"
    - "right_hip_pitch_link"
    - "right_hip_roll_link"
    - "right_hip_yaw_link"
    - "right_knee_link"
    - "right_ankle_pitch_link"
    - "right_ankle_roll_link"
    - "waist_yaw_link"
    - "waist_roll_link"
    - "torso_link"
    - "left_shoulder_pitch_link"
    - "left_shoulder_roll_link"
    - "left_shoulder_yaw_link"
    - "left_elbow_link"
    - "left_wrist_roll_link"
    - "left_wrist_pitch_link"
    - "left_wrist_yaw_link"
    - "right_shoulder_pitch_link"
    - "right_shoulder_roll_link"
    - "right_shoulder_yaw_link"
    - "right_elbow_link"
    - "right_wrist_roll_link"
    - "right_wrist_pitch_link"
    - "right_wrist_yaw_link\"""",
        ),
        Highlight(
            file=motion_obs_cfg,
            content="""
Actor observation schema evidence. `context_length` is 32 and `n_fut_frames` is 10. The actor current/proprioceptive side contains reference gravity/base velocities/DoF/root-height plus actual projected gravity, root angular velocity, DoF position, DoF velocity, and last action, all with 32-frame histories where marked. The actor future/reference side contains 10-frame future reference DoF position, root height, gravity, base linear velocity, and base angular velocity. The eight listed key bodies would make keybody current/future terms 8 * 3 = 24, but those two keybody actor terms are not consumed by the MLP actor schema highlighted later.""",
            text="""
obs:
  context_length: 32
  n_fut_frames: 10
  target_fps: 50
  actor_obs_prefix: "ref_"
  critic_obs_prefix: "ref_"

  obs_groups:
    unified:
      atomic_obs_list:
        - actor_ref_gravity_projection_cur:
            func: ref_gravity_projection_cur
            history_length: ${obs.context_length}
            flatten_history_dim: false
            params:
              ref_prefix: ${obs.actor_obs_prefix}
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_ref_gravity_projection_cur.n_min}
                n_max: ${domain_rand.obs_noise.actor_ref_gravity_projection_cur.n_max}

        - actor_ref_gravity_projection_fut:
            func: ref_gravity_projection_fut
            params:
              ref_prefix: ${obs.actor_obs_prefix}
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_ref_gravity_projection_fut.n_min}
                n_max: ${domain_rand.obs_noise.actor_ref_gravity_projection_fut.n_max}

        # Reference base linear velocity
        - actor_ref_base_linvel_cur:
            func: ref_base_linvel_cur
            history_length: ${obs.context_length}
            flatten_history_dim: false
            params:
              ref_prefix: ${obs.actor_obs_prefix}
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_ref_base_linvel_cur.n_min}
                n_max: ${domain_rand.obs_noise.actor_ref_base_linvel_cur.n_max}
                n_min_z: ${domain_rand.obs_noise.actor_ref_base_linvel_cur.n_min_z}
                n_max_z: ${domain_rand.obs_noise.actor_ref_base_linvel_cur.n_max_z}

        - actor_ref_base_linvel_fut:
            func: ref_base_linvel_fut
            params:
              ref_prefix: ${obs.actor_obs_prefix}
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_ref_base_linvel_fut.n_min}
                n_max: ${domain_rand.obs_noise.actor_ref_base_linvel_fut.n_max}
                n_min_z: ${domain_rand.obs_noise.actor_ref_base_linvel_fut.n_min_z}
                n_max_z: ${domain_rand.obs_noise.actor_ref_base_linvel_fut.n_max_z}

        - actor_ref_base_angvel_cur:
            func: ref_base_angvel_cur
            history_length: ${obs.context_length}
            flatten_history_dim: false
            params:
              ref_prefix: ${obs.actor_obs_prefix}
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_ref_base_angvel_cur.n_min}
                n_max: ${domain_rand.obs_noise.actor_ref_base_angvel_cur.n_max}
                n_min_z: ${domain_rand.obs_noise.actor_ref_base_angvel_cur.n_min_z}
                n_max_z: ${domain_rand.obs_noise.actor_ref_base_angvel_cur.n_max_z}

        - actor_ref_base_angvel_fut:
            func: ref_base_angvel_fut
            params:
              ref_prefix: ${obs.actor_obs_prefix}
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_ref_base_angvel_fut.n_min}
                n_max: ${domain_rand.obs_noise.actor_ref_base_angvel_fut.n_max}
                n_min_z: ${domain_rand.obs_noise.actor_ref_base_angvel_fut.n_min_z}
                n_max_z: ${domain_rand.obs_noise.actor_ref_base_angvel_fut.n_max_z}

        - actor_ref_dof_pos_cur:
            func: ref_dof_pos_cur
            history_length: ${obs.context_length}
            flatten_history_dim: false
            params:
              ref_prefix: ${obs.actor_obs_prefix}
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_ref_dof_pos_cur.n_min}
                n_max: ${domain_rand.obs_noise.actor_ref_dof_pos_cur.n_max}

        - actor_ref_dof_pos_fut:
            func: ref_dof_pos_fut
            params:
              ref_prefix: ${obs.actor_obs_prefix}
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_ref_dof_pos_fut.n_min}
                n_max: ${domain_rand.obs_noise.actor_ref_dof_pos_fut.n_max}

        - actor_ref_motion_filter_cutoff_hz:
            func: ref_motion_filter_cutoff_hz

        - actor_ref_root_height_cur:
            func: ref_root_height_cur
            history_length: ${obs.context_length}
            flatten_history_dim: false
            params:
              ref_prefix: ${obs.actor_obs_prefix}
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_ref_root_height_cur.n_min}
                n_max: ${domain_rand.obs_noise.actor_ref_root_height_cur.n_max}

        - actor_ref_root_height_fut:
            func: ref_root_height_fut
            params:
              ref_prefix: ${obs.actor_obs_prefix}
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_ref_root_height_fut.n_min}
                n_max: ${domain_rand.obs_noise.actor_ref_root_height_fut.n_max}

        - actor_ref_keybody_rel_pos_cur:
            func: ref_keybody_rel_pos_cur
            history_length: ${obs.context_length}
            flatten_history_dim: false
            params:
              ref_prefix: ${obs.actor_obs_prefix}
              keybody_names:
                - "left_knee_link"
                - "right_knee_link"
                - "left_ankle_roll_link"
                - "right_ankle_roll_link"
                - "left_elbow_link"
                - "right_elbow_link"
                - "left_wrist_yaw_link"
                - "right_wrist_yaw_link"
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_ref_keybody_rel_pos_cur.n_min}
                n_max: ${domain_rand.obs_noise.actor_ref_keybody_rel_pos_cur.n_max}

        - actor_ref_keybody_rel_pos_fut:
            func: ref_keybody_rel_pos_fut
            params:
              ref_prefix: ${obs.actor_obs_prefix}
              keybody_names:
                - "left_knee_link"
                - "right_knee_link"
                - "left_ankle_roll_link"
                - "right_ankle_roll_link"
                - "left_elbow_link"
                - "right_elbow_link"
                - "left_wrist_yaw_link"
                - "right_wrist_yaw_link"
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_ref_keybody_rel_pos_fut.n_min}
                n_max: ${domain_rand.obs_noise.actor_ref_keybody_rel_pos_fut.n_max}

        - actor_projected_gravity:
            func: projected_gravity
            history_length: ${obs.context_length}
            flatten_history_dim: false
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_projected_gravity.n_min}
                n_max: ${domain_rand.obs_noise.actor_projected_gravity.n_max}

        - actor_rel_robot_root_ang_vel:
            func: rel_robot_root_ang_vel
            history_length: ${obs.context_length}
            flatten_history_dim: false
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_rel_robot_root_ang_vel.n_min}
                n_max: ${domain_rand.obs_noise.actor_rel_robot_root_ang_vel.n_max}

        - actor_dof_pos:
            func: dof_pos
            history_length: ${obs.context_length}
            flatten_history_dim: false
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_dof_pos.n_min}
                n_max: ${domain_rand.obs_noise.actor_dof_pos.n_max}

        - actor_dof_vel:
            func: dof_vel
            history_length: ${obs.context_length}
            flatten_history_dim: false
            noise:
              type: AdditiveUniformNoiseCfg
              params:
                n_min: ${domain_rand.obs_noise.actor_dof_vel.n_min}
                n_max: ${domain_rand.obs_noise.actor_dof_vel.n_max}

        - actor_last_action:
            func: last_action
            history_length: ${obs.context_length}
            flatten_history_dim: false""",
        ),
        Highlight(
            file=motion_obs_cfg,
            content="""
Critic observation schema evidence. The critic receives current reference DoF/root/anchor/root-heading terms, actual root velocities, all-body link velocities/positions/rotations, DoF state, last action, and 10-frame future reference DoF/root/root-heading terms. Because the unified group has `concatenate_terms: false`, these terms remain separately keyed for the module schema instead of being one IsaacLab-concatenated tensor.""",
            text="""
        - critic_ref_dof_pos_cur:
            func: ref_dof_pos_cur
            params:
              ref_prefix: ${obs.critic_obs_prefix}

        - critic_ref_dof_pos_fut:
            func: ref_dof_pos_fut
            params:
              ref_prefix: ${obs.critic_obs_prefix}

        - critic_ref_root_height_fut:
            func: ref_root_height_fut
            params:
              ref_prefix: ${obs.critic_obs_prefix}

        - critic_ref_root_height_cur:
            func: ref_root_height_cur
            params:
              ref_prefix: ${obs.critic_obs_prefix}

        - critic_global_anchor_diff:
            func: global_anchor_diff
            params:
              ref_prefix: ${obs.critic_obs_prefix}

        - critic_ref_motion_cur_heading_aligned_root_pos:
            func: ref_motion_cur_heading_aligned_root_pos
            params:
              ref_prefix: ${obs.critic_obs_prefix}

        - critic_ref_motion_fut_heading_aligned_root_pos:
            func: ref_motion_fut_heading_aligned_root_pos
            params:
              ref_prefix: ${obs.critic_obs_prefix}

        - critic_ref_motion_cur_heading_aligned_root_rot6d:
            func: ref_motion_cur_heading_aligned_root_rot6d
            params:
              ref_prefix: ${obs.critic_obs_prefix}

        - critic_ref_motion_fut_heading_aligned_root_rot6d:
            func: ref_motion_fut_heading_aligned_root_rot6d
            params:
              ref_prefix: ${obs.critic_obs_prefix}

        - critic_ref_motion_cur_heading_aligned_root_lin_vel:
            func: ref_motion_cur_heading_aligned_root_lin_vel
            params:
              ref_prefix: ${obs.critic_obs_prefix}

        - critic_ref_motion_fut_heading_aligned_root_lin_vel:
            func: ref_motion_fut_heading_aligned_root_lin_vel
            params:
              ref_prefix: ${obs.critic_obs_prefix}

        - critic_ref_motion_cur_heading_aligned_root_ang_vel:
            func: ref_motion_cur_heading_aligned_root_ang_vel
            params:
              ref_prefix: ${obs.critic_obs_prefix}

        - critic_ref_motion_fut_heading_aligned_root_ang_vel:
            func: ref_motion_fut_heading_aligned_root_ang_vel
            params:
              ref_prefix: ${obs.critic_obs_prefix}

        - critic_rel_robot_root_lin_vel:
            func: rel_robot_root_lin_vel

        - critic_rel_robot_root_ang_vel:
            func: rel_robot_root_ang_vel

        - critic_global_robot_bodylink_lin_vel_flat:
            func: global_robot_bodylink_lin_vel_flat

        - critic_global_robot_bodylink_ang_vel_flat:
            func: global_robot_bodylink_ang_vel_flat

        - critic_root_rel_robot_bodylink_pos_flat:
            func: root_rel_robot_bodylink_pos_flat

        - critic_root_rel_robot_bodylink_rot_mat_flat:
            func: root_rel_robot_bodylink_rot_mat_flat

        - critic_dof_pos:
            func: dof_pos

        - critic_dof_vel:
            func: dof_vel

        - critic_last_action:
            func: last_action

      enable_corruption: true
      concatenate_terms: false""",
        ),
        Highlight(
            file=motion_tracking_env,
            content="""
Collection wiring evidence. The resolved `config.obs.obs_groups` dictionary is passed through `build_observations_config`, then installed in the `ManagerBasedRLEnvCfg`. During rollout, `step` returns the observation dictionary produced by IsaacLab's observation manager.""",
            text="""
            observations: ObservationsCfg = build_observations_config(
                _obs_config_dict.obs_groups
            )
            rewards: RewardsCfg = build_rewards_config(_rewards_config_dict)

            if _terminations_config_dict:
                terminations: TerminationsCfg = build_terminations_config(
                    _terminations_config_dict
                )
            else:
                terminations: TerminationsCfg = TerminationsCfg()""",
        ),
        Highlight(
            file=motion_tracking_env,
            content="""
The built observation config is part of the IsaacLab environment config, and `ManagerBasedRLEnv` is the runtime collector. This is the bridge from HoloMotion YAML to live tensors.""",
            text="""
        self._env = ManagerBasedRLEnv(isaaclab_env_cfg, self.render_mode)

        logger.info("IsaacLab environment initialized !")
        return self._env""",
        ),
        Highlight(
            file=motion_tracking_env,
            content="""
Rollout evidence: every policy step calls the IsaacLab environment and returns `obs_dict` along with rewards/done/info. That `obs_dict` is the object later consumed by the actor and critic TensorDict assemblers.""",
            text="""
    def step(self, actor_state: dict):
        obs_dict, rewards, terminated, time_outs, infos = self._env.step(
            actor_state
        )
        # IsaacLab separates terminated vs time_outs, combine them for consistency
        dones = terminated | time_outs
        self._update_completion_rate_stats(terminated, time_outs, infos)
        self._update_robot_metrics(infos)
        return obs_dict, rewards, dones, time_outs, infos""",
        ),
        Highlight(
            file=isaaclab_observation,
            content="""
Observation-building evidence. For each configured atomic term, the builder resolves `func` first to an `ObservationFunctions._get_obs_*` method and then to IsaacLab MDP functions such as `last_action`. It carries through `params`, noise, `history_length`, and `flatten_history_dim`, then attaches each term to the configured group. This proves the YAML is not documentation; it is executable observation config.""",
            text="""
        # Add observation terms to the group
        for obs_term_dict in group_cfg["atomic_obs_list"]:
            for obs_name, obs_params in obs_term_dict.items():
                obs_params = resolve_holo_config(obs_params)
                func_name = obs_params.get("func", obs_name)
                method_name = f"_get_obs_{func_name}"

                if hasattr(ObservationFunctions, method_name):
                    func = getattr(ObservationFunctions, method_name)
                elif hasattr(isaaclab_mdp, func_name):
                    func = getattr(isaaclab_mdp, func_name)
                else:
                    raise ValueError(
                        f"Unknown observation function: {func_name}"
                    )

                obs_term_kwargs = {"func": func}
                try:
                    params_cfg = obs_params.get("params", {})
                except AttributeError:
                    print(f"No params found for {obs_name}")

                obs_term_kwargs["params"] = resolve_holo_config(params_cfg)

                noise_cfg = obs_params.get("noise")
                if noise_cfg is not None:
                    obs_term_kwargs["noise"] = _build_noise_cfg(noise_cfg)

                for field_name in obs_term_field_names:
                    if field_name in {"func", "params", "noise"}:
                        continue
                    if field_name in obs_params:
                        obs_term_kwargs[field_name] = obs_params[field_name]

                obs_term = ObsTerm(**obs_term_kwargs)

                # Add observation term to group
                setattr(isaaclab_obs_group_cfg, obs_name, obs_term)""",
        ),
        Highlight(
            file=isaaclab_observation,
            content="""
This completes group construction. Since the config sets `concatenate_terms: false`, the group is installed as a non-concatenated `unified` observation group with each term addressable as `unified/<term>`.""",
            text="""
                setattr(isaaclab_obs_group_cfg, obs_name, obs_term)

        # Add group to main observations config
        setattr(obs_cfg, group_name, isaaclab_obs_group_cfg)

    return obs_cfg""",
        ),
        Highlight(
            file=motion_command,
            content="""
Reference collection evidence. Every reference state accessor gathers tensors of shape `[num_envs, 1 + n_fut_frames, ...]` from the current motion-cache batch using the envs' current clip/frame assignments. Current observations use index 0; future observations use the following `n_fut_frames` entries.""",
            text="""
        batch_tensors = self._motion_cache.current_batch.tensors
        tensor_key = resolve_reference_tensor_key(
            batch_tensors=batch_tensors,
            base_key=base_key,
            prefix=prefix,
        )
        return self._motion_cache.gather_tensor(
            tensor_key,
            clip_indices=self._clip_indices,
            frame_indices=self._frame_indices,
            n_future_frames=self.cfg.n_fut_frames,
        )""",
        ),
        Highlight(
            file=isaaclab_observation,
            content="""
Reference DoF/root-position observation evidence. Current reference DoF position is `[B,29]`; future reference DoF position is `[B,10,29]` after the configured frame slice; current root height is `[B,1]`; future root height is `[B,10,1]` because the code unsqueezes the per-frame z-height.""",
            text="""
    def _get_obs_ref_dof_pos_cur(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, num_dofs]
        \"\"\"Reference current DoF positions in simulator DoF order.\"\"\"
        command = env.command_manager.get_term(ref_motion_command_name)
        return command.get_ref_motion_dof_pos_cur(prefix=ref_prefix)""",
        ),
        Highlight(
            file=isaaclab_observation,
            content="""
Future reference DoF position remains a 3D `[batch, future, dof]` tensor, not a pre-flattened vector, for IsaacLab training. The TensorDict assembler later validates `seq_len=10` and flattens it for the MLP.""",
            text="""
    def _get_obs_ref_dof_pos_fut(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
        num_frames: int | None = None,
    ) -> torch.Tensor:  # [num_envs, n_fut_frames * num_dofs]
        \"\"\"Future reference DoF positions (flattened over time) in simulator DoF order.\"\"\"
        command = env.command_manager.get_term(ref_motion_command_name)
        dof_pos_fut = command.get_ref_motion_dof_pos_fut(
            prefix=ref_prefix
        )  # [B, T, D(sim)]
        dof_pos_fut = ObservationFunctions._slice_future_frames(
            dof_pos_fut,
            num_frames=num_frames,
            obs_name="ref_dof_pos_fut",
        )
        return dof_pos_fut""",
        ),
        Highlight(
            file=isaaclab_observation,
            content="""
Reference gravity and base-velocity evidence. Gravity projection, base linear velocity, and base angular velocity are all 3-vectors in current form and `[B,T,3]` in future form. These define the actor current 3+3+3 and future 3+3+3 contributions, and the same shapes are reused for the critic's heading-aligned root velocity terms.""",
            text="""
    def _get_obs_ref_gravity_projection_cur(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, 3]
        \"\"\"Reference gravity projection.\"\"\"
        command = env.command_manager.get_term(ref_motion_command_name)
        gravity_projection = command.get_ref_motion_gravity_projection_cur(
            prefix=ref_prefix
        )
        return gravity_projection

    @staticmethod
    def _get_obs_ref_gravity_projection_fut(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
        num_frames: int | None = None,
    ) -> torch.Tensor:  # [num_envs, T, 3]
        \"\"\"Future reference gravity projection.\"\"\"
        command = env.command_manager.get_term(ref_motion_command_name)
        gravity_projection = command.get_ref_motion_gravity_projection_fut(
            prefix=ref_prefix
        )
        gravity_projection = ObservationFunctions._slice_future_frames(
            gravity_projection,
            num_frames=num_frames,
            obs_name="ref_gravity_projection_fut",
        )
        return gravity_projection""",
        ),
        Highlight(
            file=motion_command,
            content="""
Reference meaning evidence. Gravity is the world gravity vector rotated into the reference root frame. Base linear and angular velocities are reference root global velocities rotated into the reference root frame. Thus the actor reference command is not an arbitrary command vector; it is physically meaningful reference-motion state in the local reference frame.""",
            text="""
    def get_ref_motion_gravity_projection_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        \"\"\"Current reference gravity projected into reference root frame.\"\"\"
        g_w = self.robot.data.GRAVITY_VEC_W  # [B, 3]
        ref_root_rot_wxyz = self.get_ref_motion_root_global_rot_quat_wxyz_cur(
            prefix=prefix
        )  # [B, 4]
        return isaaclab_math.quat_apply_inverse(ref_root_rot_wxyz, g_w)

    def get_ref_motion_gravity_projection_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        g_w = self.robot.data.GRAVITY_VEC_W  # [B, 3]
        ref_root_rot_wxyz = (
            self.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
                prefix=prefix
            )
        )
        return isaaclab_math.quat_apply_inverse(ref_root_rot_wxyz, g_w)

    def get_ref_motion_gravity_projection_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        \"\"\"Future reference gravity projected into reference root frame.\"\"\"
        g_w = self.robot.data.GRAVITY_VEC_W  # [B, 3]
        ref_root_rot_wxyz_fut = (
            self.get_ref_motion_root_global_rot_quat_wxyz_fut(prefix=prefix)
        )  # [B, T, 4]
        gravity_fut = g_w[:, None, :].expand(
            -1, ref_root_rot_wxyz_fut.shape[1], -1
        )  # [B, T, 3]
        return isaaclab_math.quat_apply_inverse(
            ref_root_rot_wxyz_fut, gravity_fut
        )  # [B, T, 3]

    def get_ref_motion_base_linvel_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        \"\"\"Current reference base linear velocity in reference root frame.\"\"\"
        ref_root_lin_vel_w = self.get_ref_motion_root_global_lin_vel_cur(
            prefix=prefix
        )  # [B, 3]
        ref_root_rot_wxyz = self.get_ref_motion_root_global_rot_quat_wxyz_cur(
            prefix=prefix
        )  # [B, 4]
        return isaaclab_math.quat_apply_inverse(
            ref_root_rot_wxyz, ref_root_lin_vel_w
        )  # [B, 3]""",
        ),
        Highlight(
            file=isaaclab_observation,
            content="""
Proprioception evidence. Actual root linear/angular velocities are 3-vectors in the root frame; projected gravity is also a 3-vector; DoF position and DoF velocity are all 29 simulator DoFs. The `last_action` term is resolved through IsaacLab's `last_action` MDP function by the builder, so it is `[B, num_actions] = [B,29]` from the configured action space.""",
            text="""
    def _get_obs_rel_robot_root_lin_vel(env: ManagerBasedRLEnv):
        \"\"\"Relative root linear velocity in the root frame.\"\"\"
        return isaaclab_mdp.base_lin_vel(env)  # [num_envs, 3]

    @staticmethod
    def _get_obs_rel_robot_root_ang_vel(env: ManagerBasedRLEnv):
        \"\"\"Relative root angular velocity in the root frame.\"\"\"
        return isaaclab_mdp.base_ang_vel(env)  # [num_envs, 3]""",
        ),
        Highlight(
            file=isaaclab_observation,
            content="""
More proprioception evidence: the actual robot DoF observations are IsaacLab joint position and velocity relative to defaults over every selected DoF, so they are 29-wide for this G1 config.""",
            text="""
    # ------- Robot DoF States -------
    @staticmethod
    def _get_obs_dof_pos(env: ManagerBasedRLEnv):
        \"\"\"Joint positions relative to the default joint angles.\"\"\"
        return isaaclab_mdp.joint_pos_rel(env)  # [num_envs, num_dofs]

    @staticmethod
    def _get_obs_dof_vel(env: ManagerBasedRLEnv):
        \"\"\"Joint velocities.\"\"\"
        return isaaclab_mdp.joint_vel_rel(env)  # [num_envs, num_dofs]""",
        ),
        Highlight(
            file=isaaclab_observation,
            content="""
Critic body-shape evidence. With `keybody_names=None`, `_get_body_indices` selects all `robot.num_bodies`; for this G1 config that is 30. The flattened bodylink velocity and position observations are therefore 90-wide, and the rotation-matrix term is 180-wide because it keeps six numbers per body.""",
            text="""
    def _get_obs_global_robot_bodylink_lin_vel_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        \"\"\"Flattened linear velocities of specified bodylinks in the environment frame.\"\"\"
        bodylink_lin_vel = (
            ObservationFunctions._get_obs_global_robot_bodylink_lin_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        return bodylink_lin_vel.reshape(
            bodylink_lin_vel.shape[0], -1
        )  # [num_envs, num_keybodies * 3]

    @staticmethod
    def _get_obs_global_robot_bodylink_ang_vel_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        \"\"\"Flattened angular velocities of specified bodylinks in the environment frame.\"\"\"
        bodylink_ang_vel = (
            ObservationFunctions._get_obs_global_robot_bodylink_ang_vel(
                env, robot_asset_name, keybody_names
            )
        )  # [num_envs, num_keybodies, 3]
        return bodylink_ang_vel.reshape(
            bodylink_ang_vel.shape[0], -1
        )  # [num_envs, num_keybodies * 3]""",
        ),
        Highlight(
            file=isaaclab_observation,
            content="""
The critic anchor-difference term is exactly 9 dimensions: 3 position deltas plus the first two columns of a rotation matrix, i.e. rot6d. It compares the robot anchor body to the current reference anchor body in the robot-anchor frame.""",
            text="""
        pos_diff, rot_diff = isaaclab_math.subtract_frame_transforms(
            t01=global_robot_anchor_pos,
            q01=global_robot_anchor_rot_wxyz,
            t02=env_ref_motion_anchor_pos,
            q02=global_ref_motino_anchor_rot_wxyz,
        )
        rot_diff_mat = isaaclab_math.matrix_from_quat(rot_diff)
        return torch.cat(
            [
                pos_diff,
                rot_diff_mat[..., :2].reshape(env.num_envs, -1),
            ],
            dim=-1,
        )  # [num_envs, 9]""",
        ),
        Highlight(
            file=motion_command,
            content="""
Critic heading-aligned reference-motion evidence. Current and future root position, rot6d, linear velocity, and angular velocity are expressed in the current robot heading frame. Shapes are `[B,3]`, `[B,6]`, `[B,3]`, `[B,3]` for current and `[B,10,3]`, `[B,10,6]`, `[B,10,3]`, `[B,10,3]` for future.""",
            text="""
    def get_ref_motion_cur_heading_aligned_root_pos(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        # prepare current frame robot root global poses
        robot_cur_global_root_pos = self.robot.data.root_pos_w
        robot_cur_global_root_rot = self.robot.data.root_quat_w  # wxyz
        yaw_quat = isaaclab_math.yaw_quat(robot_cur_global_root_rot)

        # transform the current goal frame root poses into the relative heading aligned frame
        global_pos_diff = (
            self.get_ref_motion_root_global_pos_cur(prefix=prefix)
            - robot_cur_global_root_pos
        )
        global_pos_diff_heading_aligned = isaaclab_math.quat_apply_inverse(
            yaw_quat, global_pos_diff
        )
        return global_pos_diff_heading_aligned""",
        ),
        Highlight(
            file=motion_command,
            content="""
The rot6d term is grounded here: HoloMotion converts the reference root quaternion into the robot-heading frame, builds a rotation matrix, and flattens the first two columns into 6 numbers. The future version preserves the future-frame axis.""",
            text="""
        ref_root_rot6d = isaaclab_math.matrix_from_quat(
            ref_root_quat_in_heading_wxyz
        )[..., :2].reshape(ref_root_quat_wxyz.shape[0], 6)  # [B, 6]
        return ref_root_rot6d

    def get_ref_motion_fut_heading_aligned_root_rot6d(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        \"\"\"Future reference root rotations (rot6d) in heading-aligned frame.

        Returns:
            torch.Tensor: [B, T, 6]
        \"\"\"""",
        ),
        Highlight(
            file=motion_modules_cfg,
            content="""
Actor assembly and normalization config. The actor consumes exactly ten current/history terms and five future terms. Current per-frame width is 3 + 3 + 3 + 29 + 1 + 3 + 3 + 29 + 29 + 29 = 132; with 32 frames this is 4224. Future per-frame width is 29 + 1 + 3 + 3 + 3 = 39; with 10 frames this is 390. The MLP actor input is therefore 4614. Observation normalization is enabled with EMA stats, epsilon 1e-8, and clamp range 10.""",
            text="""
    obs_norm:
      enabled: true
      epsilon: 1.0e-8 # Reduced for better stability in DDP
      update_method: ema # ema or cumulative
      ema_momentum: 1.0e-4
      update_at_train: true
      update_at_eval: false
      enable_clipping: true # Enable clipping for DDP stability
      clip_range: 10.0 # Reduced clip range for better stability
      sync_interval_steps: 4 # Periodically sync obs normalizers across ranks during rollout

    # Observation schema for motion tracking, from the actor's perspective.
    obs_schema:
      flattened_obs:
        seq_len: ${obs.context_length}
        terms:
          - unified/actor_ref_gravity_projection_cur
          - unified/actor_ref_base_linvel_cur
          - unified/actor_ref_base_angvel_cur
          - unified/actor_ref_dof_pos_cur
          - unified/actor_ref_root_height_cur
          - unified/actor_projected_gravity
          - unified/actor_rel_robot_root_ang_vel
          - unified/actor_dof_pos
          - unified/actor_dof_vel
          - unified/actor_last_action
      flattened_obs_fut:
        seq_len: ${obs.n_fut_frames}
        terms:
          - unified/actor_ref_dof_pos_fut
          - unified/actor_ref_root_height_fut
          - unified/actor_ref_gravity_projection_fut
          - unified/actor_ref_base_linvel_fut
          - unified/actor_ref_base_angvel_fut

    output_dim: robot_action_dim""",
        ),
        Highlight(
            file=motion_modules_cfg,
            content="""
Critic assembly and normalization config. Current critic width is 596: reference DoF 29, anchor diff 9, heading-aligned current root 3+6+3+3, actual root velocity 3+3, all-body velocity/position/rotation 90+90+90+180, DoF state 29+29, and last action 29. Future critic width is 10 * (29+1+3+6+3+3) = 450. The MLP critic input is therefore 1046, with the same EMA observation normalization settings.""",
            text="""
    obs_schema:
      flattened_obs:
        seq_len: 1
        terms:
          - unified/critic_ref_dof_pos_cur
          - unified/critic_global_anchor_diff
          - unified/critic_ref_motion_cur_heading_aligned_root_pos
          - unified/critic_ref_motion_cur_heading_aligned_root_rot6d
          - unified/critic_ref_motion_cur_heading_aligned_root_lin_vel
          - unified/critic_ref_motion_cur_heading_aligned_root_ang_vel
          - unified/critic_rel_robot_root_lin_vel
          - unified/critic_rel_robot_root_ang_vel
          - unified/critic_global_robot_bodylink_lin_vel_flat
          - unified/critic_global_robot_bodylink_ang_vel_flat
          - unified/critic_root_rel_robot_bodylink_pos_flat
          - unified/critic_root_rel_robot_bodylink_rot_mat_flat
          - unified/critic_dof_pos
          - unified/critic_dof_vel
          - unified/critic_last_action
      flattened_obs_fut:
        seq_len: ${obs.n_fut_frames}
        terms:
          - unified/critic_ref_dof_pos_fut
          - unified/critic_ref_root_height_fut
          - unified/critic_ref_motion_fut_heading_aligned_root_pos
          - unified/critic_ref_motion_fut_heading_aligned_root_rot6d
          - unified/critic_ref_motion_fut_heading_aligned_root_lin_vel
          - unified/critic_ref_motion_fut_heading_aligned_root_ang_vel

    output_dim: 1""",
        ),
        Highlight(
            file=agent_modules,
            content="""
Assembler evidence. MLP actor/critic modules use `output_mode="flat"`. A 2D term is accepted only for `seq_len=1`; a 3D term must match the configured sequence length and is flattened from `[B,T,D]` to `[B,T*D]`. Then all schema terms are concatenated in order. This proves the 4614 actor and 1046 critic dimensions above are the actual network inputs.""",
            text="""
    def _validate_and_flatten(
        self,
        tensor: torch.Tensor,
        seq_len: int,
        term: str,
    ) -> torch.Tensor:
        if tensor.ndim == 2:
            # [B, D] treat as seq_len=1
            if seq_len != 1:
                raise ValueError(
                    f"Term '{term}' expected seq_len={seq_len} but tensor is 2D {tensor.shape}"
                )
            return tensor
        if tensor.ndim == 3:
            if tensor.shape[1] != seq_len:
                raise ValueError(
                    f"Term '{term}' seq_len mismatch: expected {seq_len}, got {tensor.shape[1]}"
                )
            b, t, d = tensor.shape
            return tensor.reshape(b, t * d)
        raise ValueError(
            f"Term '{term}' tensor ndim must be 2 or 3, got {tensor.ndim}"
        )""",
        ),
        Highlight(
            file=agent_modules,
            content="""
Normalization execution evidence. When observation normalization is enabled, actor observations update the empirical normalizer during rollout/training, then use `normalize_only`, then clamp to the configured range. For 3D sequence tensors, stats are updated over `B * seq_len` rows and reshaped back; for this MLP stack the assembler has already produced 2D flat tensors.""",
            text="""
    def _normalize_actor_obs(
        self, obs: torch.Tensor, update: bool
    ) -> torch.Tensor:
        if not self.obs_norm_enabled:
            return obs
        clip = float(self.obs_norm_clip)
        if obs.ndim == 3:
            b, seq_len, d = obs.shape
            flat_obs = obs.reshape(b * seq_len, d)
            if update:
                self.obs_normalizer.update(flat_obs)
            flat_obs = self.obs_normalizer.normalize_only(flat_obs)
            obs = flat_obs.reshape(b, seq_len, d)
        else:
            if update:
                self.obs_normalizer.update(obs)
            obs = self.obs_normalizer.normalize_only(obs)
        if clip > 0.0:
            obs = torch.clamp(obs, -clip, clip)
        return obs""",
        ),
        Highlight(
            file=network_modules,
            content="""
Normalizer internals. `EmpiricalNormalization` stores mean/std buffers over the assembled observation dimension and supports EMA updates, exactly matching the motion module config. The normalized value is `(x - mean) / (std + eps)`.""",
            text="""
class EmpiricalNormalization(nn.Module):
    \"\"\"Normalize mean and variance of values based on empirical values.\"\"\"

    def __init__(
        self,
        shape,
        eps: float = 1e-2,
        until: int | None = None,
        *,
        update_method: str = "cumulative",
        ema_momentum: float | None = None,
    ):
        \"\"\"Initialize EmpiricalNormalization module.""",
        ),
        Highlight(
            file=network_modules,
            content="""
This is the normalizer formula and EMA update path used when `update_method: ema` is configured. It updates running mean and second moment, derives variance/std, and `normalize_only` applies the stored statistics without mutating them.""",
            text="""
    def normalize_only(self, x):
        return (x - self._mean) / (self._std + self.eps)

    @torch.compiler.disable
    @torch.jit.unused
    def update(self, x):
        \"\"\"Learn input values without computing the output values of them.\"\"\"

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        if self.update_method == "ema":
            m = float(self.ema_momentum)
            mean_x = torch.mean(x, dim=0, keepdim=True)
            ex2_x = torch.mean(x * x, dim=0, keepdim=True)
            self._mean.mul_(1.0 - m).add_(mean_x, alpha=m)
            self._ex2.mul_(1.0 - m).add_(ex2_x, alpha=m)
            var = torch.clamp(self._ex2 - self._mean * self._mean, min=0.0)
            self._var.copy_(var)
            self._std.copy_(torch.sqrt(self._var))
            return""",
        ),
        Highlight(
            file=eval_mujoco,
            content="""
Evaluation observation-builder evidence. MuJoCo sim2sim accepts the same unified observation config, filters out critic terms, then reorders actor terms by `modules.actor.obs_schema`. That keeps the deployed/eval policy input order aligned with training even when the unified observation group contains extra actor terms such as keybody or filter metadata not consumed by the MLP actor.""",
            text="""
        if obs_groups.get("unified", None) is not None:
            entries = []
            for term_dict in obs_groups.unified.atomic_obs_list:
                term_name = str(list(term_dict.keys())[0])
                if term_name.startswith("critic_"):
                    continue
                entries.append(
                    (
                        "unified",
                        term_name,
                        self._to_plain_obs_cfg(term_dict[term_name]),
                    )
                )
            if not entries:
                raise ValueError(
                    "obs_groups.unified found but contains no non-critic terms."
                )
            return entries""",
        ),
        Highlight(
            file=eval_mujoco,
            content="""
The eval builder converts that schema-ordered actor term list into `PolicyObsBuilder`, then every policy update calls `build_policy_obs()` and feeds the resulting flat vector into ONNX. This proves evaluation uses the same actor observation terms and order as training.""",
            text="""
    def _init_obs_buffers(self):
        atomic_list = self._get_policy_atomic_obs_list()
        obs_policy_cfg = {"atomic_obs_list": atomic_list}
        self.obs_builder = PolicyObsBuilder(
            dof_names_onnx=self.dof_names_onnx,
            default_angles_onnx=self.default_angles_onnx,
            evaluator=self,
            obs_policy_cfg=obs_policy_cfg,
        )""",
        ),
        Highlight(
            file=eval_obs_builder,
            content="""
PolicyObsBuilder mirrors the training-side temporal structure in deployment/eval: history terms create circular buffers, the first value fills the whole buffer for warm start, configured terms are appended in order, and the final policy observation is a single concatenated `float32` vector.""",
            text="""
    def build_policy_obs(self) -> np.ndarray:
        \"\"\"Append one step using evaluator-provided observation terms and return flattened obs.\"\"\"
        # Compute per-term outputs
        values: Dict[str, np.ndarray] = {}

        for spec in self.term_specs:
            name = spec["name"]
            scale = spec.get("scale", 1.0)
            values[name] = self._compute_term(name) * scale

        # Lazily initialize buffers with inferred feature dims
        if len(self._buffers) == 0:
            for spec in self.term_specs:
                name = spec["name"]
                hist_len = int(spec.get("history_length", 0))
                if hist_len <= 0:
                    continue
                feat_dim = int(values[name].reshape(-1).shape[0])
                self._buffers[name] = _CircularBuffer(
                    hist_len, feat_dim, "cpu"
                )
        # Append current step to buffers (skip terms without history)
        for spec in self.term_specs:
            name = spec["name"]
            if name in self._buffers:
                item = torch.as_tensor(
                    values[name].reshape(1, -1),
                    dtype=torch.float32,
                    device="cpu",
                )
                self._buffers[name].append(item)
        # Assemble flat list according to term ordering and history flatten rules
        flat_list: List[np.ndarray] = []
        for spec in self.term_specs:
            name = spec["name"]
            flatten = bool(spec.get("flatten", True))
            if name in self._buffers:
                buf = self._buffers[name].buffer[0]  # [hist, feat]
                arr = (
                    buf.reshape(-1).detach().cpu().numpy()
                    if flatten
                    else buf[-1].detach().cpu().numpy()
                )
                flat_list.append(arr.astype(np.float32))
            else:
                # no history -> use computed value directly
                flat_list.append(values[name].reshape(-1).astype(np.float32))

        if len(flat_list) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(flat_list, axis=0).astype(np.float32)""",
        ),
    ]
)
