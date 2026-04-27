from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

obs_cfg = File(path="HoloMotion/holomotion/config/env/observations/motion_tracking/obs_motion_tracking_mlp.yaml")
module_cfg = File(path="HoloMotion/holomotion/config/modules/motion_tracking/motion_tracking_mlp.yaml")
robot_cfg = File(path="HoloMotion/holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml")
obs_builder = File(path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_observation.py")
agent_modules = File(path="HoloMotion/holomotion/src/modules/agent_modules.py")

observation_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=obs_cfg,
            content="""
Step 1: establish observation time constants. The active MLP observation config has actor history length H=32 and future-frame length T=10. Therefore yes, future frames are inside observation.""",
            text='''
obs:
  context_length: 32
  n_fut_frames: 10
  target_fps: 50
  actor_obs_prefix: "ref_"
  critic_obs_prefix: "ref_"''',
        ),
        Highlight(
            file=robot_cfg,
            content="""
Step 2: establish robot-size constants. The robot has D=29 observed DoFs, A=29 actions, and N=30 bodies. These constants drive the DoF/action/body observation dimensions below.""",
            text="""
  dof_obs_size: 29
  actions_dim: 29
  num_bodies: 30
  num_extend_bodies: 0""",
        ),
        Highlight(
            file=module_cfg,
            content="""
Step 3: actor schema evidence. The actor consumes 10 current-history terms and 5 future terms. Current terms are under `flattened_obs` with `seq_len: ${obs.context_length}` = 32; future terms are under `flattened_obs_fut` with `seq_len: ${obs.n_fut_frames}` = 10.

Actor current per-step dimensions:
- actor_ref_gravity_projection_cur: 3.
- actor_ref_base_linvel_cur: 3.
- actor_ref_base_angvel_cur: 3.
- actor_ref_dof_pos_cur: D = 29.
- actor_ref_root_height_cur: 1.
- actor_projected_gravity: 3.
- actor_rel_robot_root_ang_vel: 3.
- actor_dof_pos: D = 29.
- actor_dof_vel: D = 29.
- actor_last_action: A = 29.

Actor current per-step sum: 3 + 3 + 3 + 29 + 1 + 3 + 3 + 29 + 29 + 29 = 132.
Actor current contribution: H * 132 = 32 * 132 = 4224.

Actor future per-frame dimensions:
- actor_ref_dof_pos_fut: D = 29.
- actor_ref_root_height_fut: 1.
- actor_ref_gravity_projection_fut: 3.
- actor_ref_base_linvel_fut: 3.
- actor_ref_base_angvel_fut: 3.

Actor future per-frame sum: 29 + 1 + 3 + 3 + 3 = 39.
Actor future contribution: T * 39 = 10 * 39 = 390.
Actor total consumed observation dimension: 4224 + 390 = 4614.""",
            text="""
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
          - unified/actor_ref_base_angvel_fut""",
        ),
        Highlight(
            file=module_cfg,
            content="""
Step 4: critic future schema evidence. The critic consumes six future reference terms via `flattened_obs_fut`, again with T=10 future frames.

Critic future per-frame dimensions:
- critic_ref_dof_pos_fut: D = 29.
- critic_ref_root_height_fut: 1.
- critic_ref_motion_fut_heading_aligned_root_pos: 3.
- critic_ref_motion_fut_heading_aligned_root_rot6d: 6.
- critic_ref_motion_fut_heading_aligned_root_lin_vel: 3.
- critic_ref_motion_fut_heading_aligned_root_ang_vel: 3.

Critic future per-frame sum: 29 + 1 + 3 + 6 + 3 + 3 = 45.
Critic future contribution: T * 45 = 10 * 45 = 450.""",
            text="""
      flattened_obs_fut:
        seq_len: ${obs.n_fut_frames}
        terms:
          - unified/critic_ref_dof_pos_fut
          - unified/critic_ref_root_height_fut
          - unified/critic_ref_motion_fut_heading_aligned_root_pos
          - unified/critic_ref_motion_fut_heading_aligned_root_rot6d
          - unified/critic_ref_motion_fut_heading_aligned_root_lin_vel
          - unified/critic_ref_motion_fut_heading_aligned_root_ang_vel""",
        ),
        Highlight(
            file=obs_builder,
            content="""
Observation configs are converted to IsaacLab `ObservationTermCfg` objects by resolving each `func` name against HoloMotion observation functions or IsaacLab MDP helpers.""",
            text="""
                if hasattr(ObservationFunctions, method_name):
                    func = getattr(ObservationFunctions, method_name)
                elif hasattr(isaaclab_mdp, func_name):
                    func = getattr(isaaclab_mdp, func_name)
                else:
                    raise ValueError(
                        f"Unknown observation function: {func_name}"
                    )""",
        ),
        Highlight(
            file=obs_builder,
            content="""
Step 5: DoF reference dimensions. Current reference DoF position is `[B, D]`; future reference DoF position is `[B, T, D]`. With D=29, those terms contribute 29 per current step or future frame.""",
            text='''
    def _get_obs_ref_dof_pos_cur(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
    ) -> torch.Tensor:  # [num_envs, num_dofs]
        """Reference current DoF positions in simulator DoF order."""
        command = env.command_manager.get_term(ref_motion_command_name)
        return command.get_ref_motion_dof_pos_cur(prefix=ref_prefix)''',
        ),
        Highlight(
            file=obs_builder,
            content="""
Step 6: future-vector dimensions. Future gravity/base-velocity helpers return `[B, T, 3]`, so each contributes 3 per future frame. The same shape pattern is used by future heading-aligned root position, linear velocity, and angular velocity. Future root rotation is explicitly rot6d, so it contributes 6 per future frame.""",
            text="""
    def _get_obs_ref_gravity_projection_fut(
        env: ManagerBasedRLEnv,
        ref_motion_command_name: str = "ref_motion",
        ref_prefix: str = "ref_",
        num_frames: int | None = None,
    ) -> torch.Tensor:  # [num_envs, T, 3]
        \"\"\"Future reference gravity projection.\"\"\"""",
        ),
        Highlight(
            file=obs_builder,
            content="""
Step 7: critic current special term. `critic_global_anchor_diff` is position delta plus rot6d, so it contributes 3 + 6 = 9 dimensions.""",
            text="""
        return torch.cat(
            [
                pos_diff,
                rot_diff_mat[..., :2].reshape(env.num_envs, -1),
            ],
            dim=-1,
        )  # [num_envs, 9]""",
        ),
        Highlight(
            file=obs_builder,
            content="""
Step 8: critic body-flat dimensions. With N=30 bodies, flattened all-body linear velocity is 30 * 3 = 90, all-body angular velocity is 30 * 3 = 90, root-relative body position is 30 * 3 = 90, and root-relative body rot6d is 30 * 6 = 180.""",
            text="""
    def _get_obs_global_robot_bodylink_lin_vel_flat(
        env: ManagerBasedRLEnv,
        robot_asset_name: str = "robot",
        keybody_names: list[str] | None = None,
    ) -> torch.Tensor:  # [num_envs, num_keybodies * 3]
        \"\"\"Flattened linear velocities of specified bodylinks in the environment frame.\"\"\"""",
        ),
        Highlight(
            file=module_cfg,
            content="""
Step 9: critic current schema evidence and arithmetic. The critic current `flattened_obs` terms are:
- critic_ref_dof_pos_cur: 29.
- critic_global_anchor_diff: 9.
- critic_ref_motion_cur_heading_aligned_root_pos: 3.
- critic_ref_motion_cur_heading_aligned_root_rot6d: 6.
- critic_ref_motion_cur_heading_aligned_root_lin_vel: 3.
- critic_ref_motion_cur_heading_aligned_root_ang_vel: 3.
- critic_rel_robot_root_lin_vel: 3.
- critic_rel_robot_root_ang_vel: 3.
- critic_global_robot_bodylink_lin_vel_flat: 90.
- critic_global_robot_bodylink_ang_vel_flat: 90.
- critic_root_rel_robot_bodylink_pos_flat: 90.
- critic_root_rel_robot_bodylink_rot_mat_flat: 180.
- critic_dof_pos: 29.
- critic_dof_vel: 29.
- critic_last_action: 29.

Critic current sum: 29 + 9 + 3 + 6 + 3 + 3 + 3 + 3 + 90 + 90 + 90 + 180 + 29 + 29 + 29 = 596.
Critic total consumed observation dimension: current 596 + future 450 = 1046.""",
            text="""
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
          - unified/critic_last_action""",
        ),
        Highlight(
            file=agent_modules,
            content="""
Step 10: final assembly evidence. The TensorDict assembler fetches only terms named in the actor/critic `obs_schema` and concatenates them. Therefore configured atomic terms that are not named in the schema do not affect the 4614/1046 totals.

Configured but not consumed by the active MLP actor schema:
- actor_ref_motion_filter_cutoff_hz: [B, 1].
- actor_ref_keybody_rel_pos_cur: 8 selected keybodies * 3 = 24 per history step.
- actor_ref_keybody_rel_pos_fut: 8 selected keybodies * 3 = 24 per future frame.

These are real configured observation terms, but they are not part of the active MLP actor `obs_schema` and therefore are excluded from actor total 4614.""",
            text="""
                tensor = self._get_from_data(data, term)
                if tensor is None:
                    raise KeyError(
                        f"Missing term '{term}' in TensorDict input for assembler. "
                        "Use explicit hierarchical terms (e.g. 'group/term') "
                        "for nested TensorDict keys."
                    )""",
        ),
    ]
)
