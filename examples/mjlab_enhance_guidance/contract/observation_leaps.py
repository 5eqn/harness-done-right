from enhance_guidance import AtomicLeap, File, Highlight


holomotion_obs_tf_moe = File(
    path="HoloMotion/holomotion/config/env/observations/motion_tracking/obs_motion_tracking_tf-moe.yaml"
)
holomotion_env_cfg = File(path="HoloMotion/holomotion/config/env/motion_tracking.yaml")
mjlab_tracking_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py")
mjlab_g1_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py")
mjlab_g1_rl_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/rl_cfg.py")


observation_leaps: list[AtomicLeap] = [
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_obs_tf_moe,
            content="HoloMotion declares one unified observation group with explicit context length, future-frame horizon, and actor/critic reference prefixes.",
            text="""
obs:
  context_length: 1
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
              ref_prefix: ${obs.actor_obs_prefix}""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_tracking_env_cfg,
            content="MJLab exposes separate actor and critic groups, concatenating each group's terms into flat tensors.",
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
        change_direction=(
            "Replace the two hard-coded concatenated MJLab observation groups with a "
            "single declarative observation schema that can emit named atomic terms "
            "for actor, critic, and TF-MoE routing. Keep actor and critic prefixes "
            "explicit so the same observation functions can select deployment-safe "
            "or privileged reference tensors without duplicating code."
        ),
        change_reason=(
            "Generalized tracking needs stable named features, not just two flat "
            "vectors. A unified schema lets the policy module route actor, critic, "
            "expert, and auxiliary terms consistently while still preserving "
            "actor/critic visibility boundaries."
        ),
        changed_code="""
from dataclasses import dataclass, field

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.tracking import mdp


@dataclass
class TrackingObservationSchema:
  context_length: int = 1
  n_fut_frames: int = 10
  target_fps: int = 50
  actor_obs_prefix: str = "ref_"
  critic_obs_prefix: str = "ref_"


def make_unified_tracking_observations(schema: TrackingObservationSchema):
  atomic_terms = {
    "actor_ref_gravity_projection_cur": ObservationTermCfg(
      func=mdp.ref_gravity_projection_cur,
      params={"ref_prefix": schema.actor_obs_prefix},
      history_length=schema.context_length,
      flatten_history_dim=False,
    ),
    "actor_ref_dof_pos_cur": ObservationTermCfg(
      func=mdp.ref_dof_pos_cur,
      params={"ref_prefix": schema.actor_obs_prefix},
      history_length=schema.context_length,
      flatten_history_dim=False,
    ),
    "actor_dof_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      history_length=schema.context_length,
      flatten_history_dim=False,
    ),
    "actor_dof_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      history_length=schema.context_length,
      flatten_history_dim=False,
    ),
    "actor_last_action": ObservationTermCfg(
      func=mdp.last_action,
      history_length=schema.context_length,
      flatten_history_dim=False,
    ),
  }
  return {
    "unified": ObservationGroupCfg(
      terms=atomic_terms,
      concatenate_terms=False,
      enable_corruption=True,
    )
  }
""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_obs_tf_moe,
            content="HoloMotion adds actor future-reference observations alongside current reference observations and per-term noise configuration.",
            text="""
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
            func: ref_base_linvel_cur""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_tracking_env_cfg,
            content="MJLab's actor begins with current generated command and current anchor pose terms, without an explicit future reference horizon.",
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
    ),""",
        ),
        change_direction=(
            "Expand MJLab's current-frame command and anchor observations into "
            "explicit current and future reference-frame features. Add future "
            "reference gravity, base velocities, DoF targets, root height, and "
            "key-body relative positions using the command's configured future "
            "horizon instead of overloading a single generated command tensor."
        ),
        change_reason=(
            "A single current motion command can imitate one clip locally, but it "
            "does not tell the controller where the sequence is going. Future "
            "reference frames make phase, upcoming contacts, and motion transitions "
            "observable, which is essential for robust multi-motion generalization."
        ),
        changed_code="""
def add_future_reference_terms(terms, *, prefix: str, n_fut_frames: int):
  terms.update(
    {
      "actor_ref_gravity_projection_fut": ObservationTermCfg(
        func=mdp.ref_gravity_projection_fut,
        params={"ref_prefix": prefix, "num_frames": n_fut_frames},
      ),
      "actor_ref_base_linvel_fut": ObservationTermCfg(
        func=mdp.ref_base_linvel_fut,
        params={"ref_prefix": prefix, "num_frames": n_fut_frames},
      ),
      "actor_ref_base_angvel_fut": ObservationTermCfg(
        func=mdp.ref_base_angvel_fut,
        params={"ref_prefix": prefix, "num_frames": n_fut_frames},
      ),
      "actor_ref_dof_pos_fut": ObservationTermCfg(
        func=mdp.ref_dof_pos_fut,
        params={"ref_prefix": prefix, "num_frames": n_fut_frames},
      ),
      "actor_ref_keybody_rel_pos_fut": ObservationTermCfg(
        func=mdp.ref_keybody_rel_pos_fut,
        params={
          "ref_prefix": prefix,
          "num_frames": n_fut_frames,
          "keybody_names": (
            "left_knee_link",
            "right_knee_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_elbow_link",
            "right_elbow_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
          ),
        },
      ),
    }
  )
  return terms
""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_obs_tf_moe,
            content="HoloMotion critic observations include global anchor error and heading-aligned current/future root reference features.",
            text="""
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
              ref_prefix: ${obs.critic_obs_prefix}""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_tracking_env_cfg,
            content="MJLab's critic starts from command, anchor, and body pose terms rather than the richer HoloMotion privileged schema.",
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
    ),""",
        ),
        change_direction=(
            "Upgrade the critic from clean copies of the actor plus body pose to a "
            "privileged tracking-state schema. Add global anchor error, "
            "heading-aligned current/future root pose, root velocities, all-body "
            "linear/angular velocities, root-relative body positions, body rotation "
            "features, DoF state, and last action."
        ),
        change_reason=(
            "The critic should explain value across diverse motions and reference "
            "quality, not merely observe the same anchor-relative pose as the actor. "
            "Richer privileged state stabilizes learning while keeping the actor "
            "deployable."
        ),
        changed_code="""
def make_privileged_critic_terms(*, prefix: str):
  return {
    "critic_ref_dof_pos_cur": ObservationTermCfg(
      func=mdp.ref_dof_pos_cur,
      params={"ref_prefix": prefix},
    ),
    "critic_ref_dof_pos_fut": ObservationTermCfg(
      func=mdp.ref_dof_pos_fut,
      params={"ref_prefix": prefix},
    ),
    "critic_global_anchor_diff": ObservationTermCfg(
      func=mdp.global_anchor_diff,
      params={"ref_prefix": prefix},
    ),
    "critic_ref_motion_cur_heading_aligned_root_pos": ObservationTermCfg(
      func=mdp.ref_motion_cur_heading_aligned_root_pos,
      params={"ref_prefix": prefix},
    ),
    "critic_ref_motion_fut_heading_aligned_root_pos": ObservationTermCfg(
      func=mdp.ref_motion_fut_heading_aligned_root_pos,
      params={"ref_prefix": prefix},
    ),
    "critic_rel_robot_root_lin_vel": ObservationTermCfg(
      func=mdp.rel_robot_root_lin_vel
    ),
    "critic_root_rel_robot_bodylink_pos_flat": ObservationTermCfg(
      func=mdp.root_rel_robot_bodylink_pos_flat
    ),
    "critic_root_rel_robot_bodylink_rot_mat_flat": ObservationTermCfg(
      func=mdp.root_rel_robot_bodylink_rot_mat_flat
    ),
    "critic_dof_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "critic_dof_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "critic_last_action": ObservationTermCfg(func=mdp.last_action),
  }
""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_env_cfg,
            content="HoloMotion routes reference observations through a named ref_motion command configured with robot DoF and body subsets.",
            text="""
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
          arm_body_names: ${robot.arm_body_names}""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_g1_env_cfg,
            content="MJLab specializes observations with an ad-hoc state-estimation branch that removes actor terms.",
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
        change_direction=(
            "Move observation specialization out of ad-hoc state-estimation forks "
            "and into reference-router-aware term parameters. Observation terms "
            "should resolve features through a named reference prefix and command "
            "source, while noise, history, and normalization remain configurable "
            "per term or model."
        ),
        change_reason=(
            "Generalized tracking must support multiple reference sources and "
            "deployment stacks without deleting observation terms by hand. Router "
            "references preserve one schema for training, evaluation, and export, "
            "while controlled corruption/history/normalization maintain robustness."
        ),
        changed_code="""
def build_router_reference_observations(schema: TrackingObservationSchema):
  terms = make_unified_tracking_observations(schema)["unified"].terms
  terms["actor_ref_motion_filter_cutoff_hz"] = ObservationTermCfg(
    func=mdp.ref_motion_filter_cutoff_hz
  )
  terms["actor_ref_root_height_cur"] = ObservationTermCfg(
    func=mdp.ref_root_height_cur,
    params={"ref_prefix": schema.actor_obs_prefix},
    history_length=schema.context_length,
    flatten_history_dim=False,
  )
  terms = add_future_reference_terms(
    terms,
    prefix=schema.actor_obs_prefix,
    n_fut_frames=schema.n_fut_frames,
  )
  terms.update(make_privileged_critic_terms(prefix=schema.critic_obs_prefix))
  return {
    "unified": ObservationGroupCfg(
      terms=terms,
      concatenate_terms=False,
      enable_corruption=True,
    )
  }


def unitree_g1_generalized_tracking_ppo_runner_cfg():
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.actor.obs_normalization = True
  cfg.critic.obs_normalization = True
  cfg.obs_groups = {"unified": ("unified",)}
  return cfg
""",
    ),
]
