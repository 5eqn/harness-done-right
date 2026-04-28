from enhance_guidance import AtomicLeap, File, Highlight


holomotion_tf_moe_config = File(
    path="HoloMotion/holomotion/config/modules/motion_tracking/motion_tracking_tf-moe.yaml"
)
holomotion_ppo_tf = File(path="HoloMotion/holomotion/src/algo/ppo_tf.py")
mjlab_tracking_rl_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/rl_cfg.py")
mjlab_rl_config = File(path="mjlab/src/mjlab/rl/config.py")
mjlab_runner = File(path="mjlab/src/mjlab/rl/runner.py")


model_leaps: list[AtomicLeap] = [
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_tf_moe_config,
            content=(
                "HoloMotion configures the tracking actor with explicit "
                "reference-routed MoE capacity and router hyperparameters."
            ),
            text="""
    # MoE-specific hyperparameters
    num_fine_experts: 16
    num_shared_experts: 1
    top_k: 2
    moe_loss_coef: 0.0
    routing_score_fn: ${algo.config.moe_router.routing_score_fn}
    routing_scale: ${algo.config.moe_router.routing_scale}
    use_dynamic_bias: ${algo.config.moe_router.use_dynamic_bias}
    bias_update_rate: ${algo.config.moe_router.bias_update_rate}
    expert_bias_clip: ${algo.config.moe_router.expert_bias_clip}""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_tracking_rl_cfg,
            content=(
                "MJLab currently configures the G1 tracking actor as a compact "
                "feed-forward RSL-RL model with Gaussian action sampling."
            ),
            text="""
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),""",
        ),
        change_direction=(
            "Replace the tracking actor's plain MLP configuration with a "
            "reference-routed grouped MoE transformer actor: keep Gaussian "
            "action sampling, but express the policy as a 512-wide, 3-layer "
            "transformer with 16 fine experts, one shared expert, top-2 "
            "routing, and explicit router knobs."
        ),
        change_reason=(
            "The current MLP compresses every motion command into one dense "
            "feed-forward path. HoloMotion's generalized tracker needs a model "
            "that can specialize across diverse reference motions while sharing "
            "common locomotion structure, so the actor must expose TF-MoE "
            "capacity and routing as first-class config rather than hidden "
            "RSL-RL defaults."
        ),
        changed_code="""
actor=RslRlModelCfg(
  class_name="ReferenceRoutedGroupedMoETransformerPolicy",
  obs_normalization=True,
  distribution_cfg={
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  },
  transformer_cfg={
    "obs_embed_mlp_hidden": 2048,
    "router_embed_mlp_hidden": 2048,
    "d_model": 512,
    "n_heads": 8,
    "n_kv_heads": 4,
    "use_gated_attn": True,
    "n_layers": 3,
    "ff_mult": 2.0,
    "ff_mult_dense": 4,
    "attn_dropout": 0.0,
    "mlp_dropout": 0.0,
    "max_ctx_len": 32,
  },
  moe_cfg={
    "num_fine_experts": 16,
    "num_shared_experts": 1,
    "top_k": 2,
    "moe_loss_coef": 0.0,
    "routing_score_fn": "softmax",
    "routing_scale": 1.0,
    "use_dynamic_bias": False,
    "bias_update_rate": 0.001,
    "expert_bias_clip": 0.0,
  },
)""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_tf_moe_config,
            content=(
                "HoloMotion preserves sequence structure and marks reference "
                "features in the actor observation schema."
            ),
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
        mjlab_reference=Highlight(
            file=mjlab_rl_config,
            content=(
                "MJLab's model dataclass exposes dense network fields but no "
                "schema for temporal or reference-routed actor inputs."
            ),
            text="""
@dataclass
class RslRlModelCfg:
  \"\"\"Config for a single neural network model (Actor or Critic).\"\"\"

  hidden_dims: Tuple[int, ...] = (128, 128, 128)
  \"\"\"The hidden dimensions of the network.\"\"\"
  activation: str = "elu"
  \"\"\"The activation function.\"\"\"
  obs_normalization: bool = False
  \"\"\"Whether to normalize the observations. Default is False.\"\"\"
  cnn_cfg: dict[str, Any] | None = None
  \"\"\"CNN encoder config. When set, class_name should be "CNNModel".""",
        ),
        change_direction=(
            "Extend MJLab's model config with an actor observation schema that "
            "separates current/history reference terms, proprioceptive terms, "
            "and future reference terms with explicit sequence lengths."
        ),
        change_reason=(
            "A TF-MoE policy cannot be a drop-in dense layer swap: it needs the "
            "model input to preserve temporal structure and identify "
            "`actor_ref_*` features for routing. Without a schema, MJLab will "
            "continue to flatten observations opaquely and the router cannot "
            "learn motion-conditioned expert selection."
        ),
        changed_code="""
@dataclass
class RslRlModelCfg:
  hidden_dims: Tuple[int, ...] = (128, 128, 128)
  activation: str = "elu"
  obs_normalization: bool = False
  cnn_cfg: dict[str, Any] | None = None
  distribution_cfg: dict[str, Any] | None = None
  rnn_type: str | None = None
  rnn_hidden_dim: int = 256
  rnn_num_layers: int = 1
  class_name: str = "MLPModel"
  obs_schema: dict[str, dict[str, Any]] | None = None
  transformer_cfg: dict[str, Any] | None = None
  moe_cfg: dict[str, Any] | None = None


ACTOR_TF_MOE_OBS_SCHEMA = {
  "flattened_obs": {
    "seq_len": "context_length",
    "terms": (
      "actor_ref_gravity_projection_cur",
      "actor_ref_base_linvel_cur",
      "actor_ref_base_angvel_cur",
      "actor_ref_dof_pos_cur",
      "actor_ref_root_height_cur",
      "actor_projected_gravity",
      "actor_rel_robot_root_ang_vel",
      "actor_dof_pos",
      "actor_dof_vel",
      "actor_last_action",
    ),
  },
  "flattened_obs_fut": {
    "seq_len": "n_fut_frames",
    "terms": (
      "actor_ref_dof_pos_fut",
      "actor_ref_root_height_fut",
      "actor_ref_gravity_projection_fut",
      "actor_ref_base_linvel_fut",
      "actor_ref_base_angvel_fut",
    ),
  },
}""",
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_ppo_tf,
            content=(
                "HoloMotion constructs the selected transformer actor wrapper "
                "with the actor schema, config, action size, and sample obs."
            ),
            text="""
        self.actor = actor_cls(
            obs_schema=actor_schema,
            module_config_dict=actor_cfg,
            num_actions=self.num_actions,
            init_noise_std=self.config.init_noise_std,
            obs_example=sample_td,
        ).to(self.device)""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_runner,
            content=(
                "MJLab's base runner only strips optional fields, then delegates "
                "model construction to stock RSL-RL."
            ),
            text="""
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)""",
        ),
        change_direction=(
            "Introduce a tracking-specific on-policy runner path that detects "
            "the TF-MoE actor class, builds the reference-routed transformer "
            "actor from the schema and a sample observation, and only falls "
            "back to stock RSL-RL for legacy MLP/RNN/CNN models."
        ),
        change_reason=(
            "MJLab's current runner only cleans optional fields before handing "
            "the config to RSL-RL, which means model construction is limited to "
            "RSL-RL's generic modules. HoloMotion's actor requires wrapper "
            "logic that computes observation dimensions, infers router feature "
            "indices, and passes an observation example into the policy module."
        ),
        changed_code="""
class MotionTrackingTfMoeRunner(MjlabOnPolicyRunner):
  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    actor_cfg = train_cfg.get("actor", {})
    if actor_cfg.get("class_name") != "ReferenceRoutedGroupedMoETransformerPolicy":
      super().__init__(env, train_cfg, log_dir, device)
      return

    obs_schema = actor_cfg["obs_schema"]
    obs_example = env.get_observations()
    actor_obs_dim = TensorDictAssembler(obs_schema, output_mode="flat").infer_output_dim(
      obs_example
    )
    module_cfg = {
      **actor_cfg["transformer_cfg"],
      **actor_cfg["moe_cfg"],
      "input_dim_override": int(actor_obs_dim),
    }
    self.actor = PPOTFRefRouterActor(
      obs_schema=obs_schema,
      module_config_dict=module_cfg,
      num_actions=env.num_actions,
      init_noise_std=actor_cfg["distribution_cfg"]["init_std"],
      obs_example=obs_example,
    ).to(device)
    self.critic = build_rsl_rl_critic(env, train_cfg["critic"], device)
    self.alg = build_ppo_algorithm(self.actor, self.critic, train_cfg["algorithm"])""",
    ),
]
