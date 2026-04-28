from project_analysis import File, Highlight, ProofFromCode


train_config = File(
    path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_tf-moe.yaml"
)
module_config = File(
    path="HoloMotion/holomotion/config/modules/motion_tracking/motion_tracking_tf-moe.yaml"
)
ppo_tf_config = File(path="HoloMotion/holomotion/config/algo/ppo_tf.yaml")
ppo_config = File(path="HoloMotion/holomotion/config/algo/ppo.yaml")
train_py = File(path="HoloMotion/holomotion/src/training/train.py")
ppo_py = File(path="HoloMotion/holomotion/src/algo/ppo.py")
ppo_tf_py = File(path="HoloMotion/holomotion/src/algo/ppo_tf.py")
agent_modules = File(path="HoloMotion/holomotion/src/modules/agent_modules.py")
network_modules = File(path="HoloMotion/holomotion/src/modules/network_modules.py")


model_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=train_config,
            content="""
Scope anchor: this proof follows the HoloMotion G1 29-DoF motion-tracking TF-MoE training configuration. That config composes the PPOTF algorithm, the motion-tracking environment and observation/reward groups, the medium domain randomization and rough terrain groups, and the `motion_tracking_tf-moe` model module group.""",
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
            file=train_py,
            content="""
The training entry point resolves `config.algo._target_`, constructs that algorithm with the compiled environment and algorithm config, loads an optional checkpoint, and calls `learn()`. For the scoped config, this instantiates `holomotion.src.algo.ppo_tf.PPOTF`, so the trainable model is the actor and critic built by `PPOTF`.""",
            text="""
    algo_class = get_class(config.algo._target_)
    algo = algo_class(
        env_config=config.env,
        config=config.algo.config,
        log_dir=log_dir,
        headless=headless,
    )

    algo.load(config.checkpoint)
    algo.learn()""",
        ),
        Highlight(
            file=ppo_tf_config,
            content="""
The algorithm config is PPOTF, a PPO variant with transformer-policy sequence update. The optimization surface is explicit: 32 rollout steps per environment, adaptive schedule, actor LR 3e-5, critic LR 5e-5, three epochs, 24 minibatches, PPO clip 0.2, entropy coefficient 5e-3, KL target 0.01, log-standard-deviation action noise initialized at 1.0, and trainable sigma because `fix_sigma` is false.""",
            text="""
algo:
  _target_: holomotion.src.algo.ppo_tf.PPOTF
  config:
    num_steps_per_env: 32
    kl_coef: 0.0
    schedule: adaptive
    actor_learning_rate: 3.0e-5
    critic_learning_rate: 5.0e-5

    num_learning_epochs: 3
    num_mini_batches: 24

    clip_param: 0.2
    entropy_coef: 5.0e-3
    desired_kl: 0.01
    noise_std_type: log
    fix_sigma: false
    init_noise_std: 1.0""",
        ),
        Highlight(
            file=module_config,
            content="""
The actor core is not a plain MLP: the configured actor type is `ReferenceRoutedGroupedMoETransformerPolicy`. It is a transformer policy with reference-routed MoE feed-forward blocks: 16 fine experts, one shared expert, top-2 routing, softmax router scores from the PPOTF router config, no dynamic bias, and routing scale 1.0.""",
            text="""
modules:
  actor:
    type: ReferenceRoutedGroupedMoETransformerPolicy

    use_checkpointing: false # use gradient checkpointing to save GRAM significantly

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
        Highlight(
            file=ppo_tf_config,
            content="""
Those interpolated router fields resolve to softmax routing with unit scale, disabled dynamic bias, update rate 0.001, and no expert-bias clipping. The selected-expert margin auxiliary is disabled while the dead-expert margin auxiliary is enabled with weight 10.0.""",
            text="""
    dead_expert_margin_to_topk:
      enabled: true
      weight: 10.0

    aux_router_command_recon:
      enabled: false
      weight: 0.0
      hidden_dim: 0
      term_prefix: actor_ref_

    aux_router_switch_penalty:
      enabled: false
      weight: 0.0

    router_expert_orthogonal:
      enabled: false
      weight: 0.0
      min_active_usage: 1.0e-3
      eps: 1.0e-8

    selected_expert_margin_to_unselected:
      enabled: false
      weight: 0.0
      target: 0.0

    moe_router:
      routing_score_fn: softmax
      routing_scale: 1.0
      use_dynamic_bias: false
      bias_update_rate: 0.001
      expert_bias_clip: 0.0""",
        ),
        Highlight(
            file=module_config,
            content="""
The actor transformer dimensions are fixed in config: observation and router embedding MLP hidden size 2048, model width 512, eight query heads, four KV heads, gated attention enabled, three transformer layers, dense FF multiplier 4 for dense layers, MoE FF multiplier 2.0 for MoE layers, zero attention/MLP dropout, and max context length 32.""",
            text="""
    # Transformer hyperparameters - smaller model for stability
    obs_embed_mlp_hidden: 2048
    router_embed_mlp_hidden: 2048
    d_model: 512
    n_heads: 8
    n_kv_heads: 4
    use_gated_attn: true

    n_layers: 3
    ff_mult: 2.0
    ff_mult_dense: 4
    attn_dropout: 0.0
    mlp_dropout: 0.0
    max_ctx_len: 32""",
        ),
        Highlight(
            file=module_config,
            content="""
Both actor and critic enable observation normalization with EMA statistics, epsilon 1e-8, train-time updates, no eval-time updates, clipping at 10.0, and cross-rank sync every four rollout steps. The actor output dimension is `robot_action_dim`, so the mean head is sized to the environment action dimension.""",
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

    # Observation schema for motion tracking, from the actor's perspective.""",
        ),
        Highlight(
            file=module_config,
            content="""
The actor input schema is sequence-aware: current/history reference terms and robot proprioceptive terms use `obs.context_length`, while future reference terms use `obs.n_fut_frames`. These named terms are the source for the flattened actor tensor and for the reference-router feature subset.""",
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
          - unified/actor_ref_base_angvel_fut

    output_dim: robot_action_dim""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
PPOTF maps the configured actor type to `PPOTFRefRouterActor`. That wrapper rejects future cross-attention for this V1 reference-routed policy and otherwise falls back to the generic transformer actor only for non-reference-routed types.""",
            text="""
    @staticmethod
    def _select_actor_wrapper_cls(actor_cfg: dict):
        actor_type = str(actor_cfg.get("type", ""))
        use_future_cross_attn = bool(
            actor_cfg.get("use_future_cross_attn", False)
        )
        if actor_type == "ReferenceRoutedGroupedMoETransformerPolicy":
            if use_future_cross_attn:
                raise ValueError(
                    "ReferenceRoutedGroupedMoETransformerPolicy does not "
                    "support use_future_cross_attn=True."
                )
            return PPOTFRefRouterActor""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
At model setup time PPOTF resolves actor/critic module config, injects the action-noise bounds and auxiliary configs into the actor, unwraps the observation schemas, infers flattened actor dimension from a sampled TensorDict, selects the wrapper class, and passes `input_dim_override` when no future cross-attention is used. This is the concrete path from YAML config into model construction.""",
            text="""
    def _setup_models_and_optimizer(self):
        sample_obs_dict = self.env.reset_all()[0]
        sample_td = self._wrap_obs_dict(sample_obs_dict)

        actor_cfg = OmegaConf.to_container(
            self.config.module_dict.actor, resolve=True
        )
        critic_cfg = OmegaConf.to_container(
            self.config.module_dict.critic, resolve=True
        )
        actor_cfg["noise_std_type"] = getattr(
            self.config, "noise_std_type", "log"
        )
        actor_cfg["min_sigma"] = getattr(self.config, "min_sigma", 0.1)
        actor_cfg["max_sigma"] = getattr(self.config, "max_sigma", 1.5)
        actor_cfg["fix_sigma"] = getattr(self.config, "fix_sigma", False)""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
The actual module instantiation is `self.actor = actor_cls(...)` and `self.critic = PPOCritic(...)`, both moved to the accelerator device. Therefore the trainable actor is `PPOTFRefRouterActor` wrapping `ReferenceRoutedGroupedMoETransformerPolicy`; the trainable critic is `PPOCritic` wrapping the configured MLP.""",
            text="""
        self.actor = actor_cls(
            obs_schema=actor_schema,
            module_config_dict=actor_cfg,
            num_actions=self.num_actions,
            init_noise_std=self.config.init_noise_std,
            obs_example=sample_td,
        ).to(self.device)
        actor_module_unwrapped = self.actor.actor_module
        self.aux_command_router_num_moe_layers = int(
            getattr(actor_module_unwrapped, "_num_moe_layers", 0)
        )
        self.aux_command_router_num_fine_experts = int(
            getattr(actor_module_unwrapped, "num_fine_experts", 0)
        )""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
The critic construction immediately follows the actor construction and uses the resolved critic schema/config. The actor and critic each receive their own optimizer, and `accelerator.prepare` wraps both models and optimizers for the training run.""",
            text="""
        self.critic = PPOCritic(
            obs_schema=critic_schema,
            module_config_dict=critic_cfg,
            obs_example=sample_td,
        ).to(self.device)

        if self.is_main_process:
            actor = self.accelerator.unwrap_model(self.actor)
            critic = self.accelerator.unwrap_model(self.critic)""",
        ),
        Highlight(
            file=agent_modules,
            content="""
`PPOTFRefRouterActor` computes router features from all actor observation leaves whose names start with `actor_ref_`, then writes both `router_input_dim` and `router_feature_indices` into the actor module config before calling the parent transformer actor constructor. This is the reference-routed part of the architecture, and it is derived from the configured observation schema rather than hardcoded dimensions.""",
            text="""
        router_feature_indices = self.infer_router_feature_indices(
            obs_schema, obs_example
        )
        actor_module_cfg["router_input_dim"] = int(len(router_feature_indices))
        actor_module_cfg["router_feature_indices"] = list(
            router_feature_indices
        )
        if "router_embed_mlp_hidden" not in actor_module_cfg:
            actor_module_cfg["router_embed_mlp_hidden"] = int(
                actor_module_cfg.get("obs_embed_mlp_hidden", 1024)
            )

        super().__init__(
            obs_schema=obs_schema,
            module_config_dict=actor_module_cfg,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            obs_example=obs_example,
        )""",
        ),
        Highlight(
            file=agent_modules,
            content="""
The generic `PPOActor` constructor resolves the configured network class from `network_modules`, builds a TensorDict assembler from the observation schema, constructs the actor module with inferred input dimension and configured output dimension, and creates a learnable Gaussian action noise parameter. With `noise_std_type: log`, the parameter is `log_std`; with the scoped config, it is initialized from `init_noise_std=1.0` and remains trainable.""",
            text="""
        self.actor_net_type = module_config_dict.get("type", "MLP")

        logger.info(f"actor_net_type: {self.actor_net_type}")

        actor_net_class = getattr(NM, self.actor_net_type, None)

        if actor_net_class is NM.MLP and obs_schema is None:
            raise ValueError(
                "PPOActor(Mlp) requires obs_schema so the agent module can assemble"
                "TensorDict observations into a flat tensor."
            )""",
        ),
        Highlight(
            file=agent_modules,
            content="""
This second constructor excerpt completes the actor path: TensorDict observations are assembled to a flat tensor, optional observation normalization is installed, the configured actor module is instantiated, and the Gaussian distribution's standard-deviation parameter is allocated outside the actor core network so the optimizer can train it.""",
            text="""
        if obs_schema is not None:
            output_mode = "seq" if actor_net_class is NM.ConvMLP else "flat"
            self.assembler = TensorDictAssembler(
                obs_schema, output_mode=output_mode
            )
            if obs_example is not None:
                self.assembler.infer_output_dim(obs_example)
            if self.assembler.output_dim is None:
                raise ValueError(
                    "TensorDictAssembler could not infer output_dim"
                )
            input_dim_for_net = int(self.assembler.output_dim)
        else:
            raise ValueError("obs_schema can't be None!")""",
        ),
        Highlight(
            file=agent_modules,
            content="""
The stochastic actor head is a diagonal Normal distribution: the actor core predicts `mu`, sigma is obtained from the trainable parameter and clamped to configured bounds, sampling uses `Normal(mu, sigma)`, and inference returns the mean action directly.""",
            text="""
        sigma = self._sigma_like(mu)
        td.set("mu", mu)
        td.set("sigma", sigma)

        if mode == "inference":
            actions_out = mu
            td.set("actions", actions_out)
            return td

        self.distribution = Normal(mu, sigma)
        if mode == "sampling":
            actions_out = self.distribution.sample()""",
        ),
        Highlight(
            file=network_modules,
            content="""
`ReferenceRoutedGroupedMoETransformerPolicy` validates router dimensions and inherits the grouped MoE transformer. It then builds a dedicated router observation embedding MLP: router-selected reference features go through Linear(router_input_dim, 2048), SiLU, and Linear(2048, d_model=512), because the scoped config sets `router_embed_mlp_hidden` to 2048 and `d_model` to 512.""",
            text="""
        self.router_obs_embed = nn.Sequential(
            nn.Linear(self.router_input_dim, self.router_embed_mlp_hidden),
            nn.SiLU(),
            nn.Linear(self.router_embed_mlp_hidden, self.d_model),
        )
        self._apply_freeze_router_state()""",
        ),
        Highlight(
            file=network_modules,
            content="""
The grouped MoE transformer constructor stores the configured expert counts and routing policy, then stores transformer dimensions including layer count, head count, KV-head count, FF multipliers, dropouts, context length, QK norm, gated attention, and checkpointing. With the scoped config, this resolves to 3 layers, d_model 512, 8 heads, 4 KV heads, ff_mult 2.0, dense ff_mult 4, zero dropout, and max context 32.""",
            text="""
        self.num_fine_experts = module_config_dict["num_fine_experts"]
        self.num_shared_experts = module_config_dict["num_shared_experts"]
        self.top_k = module_config_dict["top_k"]
        self.use_dynamic_bias = module_config_dict.get(
            "use_dynamic_bias", False
        )
        self.bias_update_rate = module_config_dict.get(
            "bias_update_rate", 0.001
        )
        self.routing_score_fn = str(
            module_config_dict.get("routing_score_fn", "softmax")
        ).lower()
        self.freeze_router = bool(
            module_config_dict.get("freeze_router", False)
        )
        self.routing_scale = float(
            module_config_dict.get("routing_scale", 1.0)
        )""",
        ),
        Highlight(
            file=network_modules,
            content="""
The actor observation embedding and action mean head both use SiLU. For the scoped actor, non-cross-attention mode uses Linear(flat_actor_obs_dim, 2048), SiLU, Linear(2048, 512), then the final action head uses Linear(512, 512), SiLU, Linear(512, robot_action_dim).""",
            text="""
            self.obs_embed = nn.Sequential(
                nn.Linear(obs_in, self.obs_embed_mlp_hidden),
                nn.SiLU(),
                nn.Linear(self.obs_embed_mlp_hidden, self.d_model),
            )
            self.state_obs_embed = None
            self.future_obs_embed = None
            self.future_pos_embed = None""",
        ),
        Highlight(
            file=network_modules,
            content="""
Layer composition is deterministic: layer 0 is always a dense `ModernTransformerBlock`; since `dense_layer_at_last` defaults false and is not set in the scoped config, layers 1 and 2 are `GroupedMoEBlock`s. Each block receives the configured attention dimensions and either dense or MoE FF multiplier.""",
            text="""
        # Stack of TransformerBlocks: first layer is always dense; the last
        # layer is also dense when dense_layer_at_last=True.
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            use_dense_layer = i == 0 or (
                self.dense_layer_at_last and i == self.n_layers - 1
            )
            if use_dense_layer:
                layer = ModernTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    n_kv_heads=self.n_kv_heads,
                    ff_mult=self.ff_mult_dense,""",
        ),
        Highlight(
            file=network_modules,
            content="""
The MoE layers use `GroupedMoEBlock` with the same attention dimensions plus the configured fine/shared expert counts, top-k, routing function, optional dynamic bias, routing scale, and dead/selected margin flags. This grounds the actor as a transformer with two sparse MoE FFN layers, not merely a dense transformer.""",
            text="""
            else:
                layer = GroupedMoEBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    n_kv_heads=self.n_kv_heads,
                    ff_mult=self.ff_mult,
                    use_qk_norm=self.use_qk_norm,
                    use_gated_attn=self.use_gated_attn,
                    gated_attn_type=self.gated_attn_type,
                    attn_dropout=self.attn_dropout,
                    mlp_dropout=self.mlp_dropout,
                    num_fine_experts=self.num_fine_experts,
                    num_shared_experts=self.num_shared_experts,
                    top_k=self.top_k,
                    use_dynamic_bias=self.use_dynamic_bias,
                    bias_update_rate=self.bias_update_rate,
                    routing_score_fn=self.routing_score_fn,""",
        ),
        Highlight(
            file=network_modules,
            content="""
The transformer actor's final normalization and mean head are explicit: RMSNorm over d_model, then a two-layer SiLU MLP to the action output dimension. Auxiliary state-prediction heads are also attached when enabled by config, but they do not replace the action mean head.""",
            text="""
        self.norm_f = RMSNorm(self.d_model)
        self.action_mu_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.output_dim),
        )
        aux_cfg = module_config_dict.get("aux_state_pred", {})""",
        ),
        Highlight(
            file=network_modules,
            content="""
A `GroupedMoEBlock` contains a second RMSNorm, a router Linear(d_model, num_fine_experts), and grouped gate/up expert parameters. The scoped actor therefore routes each MoE token from 512 hidden units into 16 fine-expert logits before applying fine-expert feed-forward projections.""",
            text="""
        self.norm2 = RMSNorm(d_model)
        self.intermediate_dim = int(d_model * ff_mult)

        self.router = nn.Linear(d_model, num_fine_experts, bias=False)
        self._apply_freeze_router_state()

        # Gate + Up (Combined)
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                num_fine_experts, self.d_model, 2 * self.intermediate_dim
            )
        )""",
        ),
        Highlight(
            file=network_modules,
            content="""
The MoE router computation first applies all shared experts, computes router logits from `router_x` when provided by the reference router, takes the configured top-k experts, normalizes selected probabilities, and adds sparse expert output to the shared expert output. This grounds the actual MoE/router behavior used by the actor.""",
            text="""
        # 1. Shared Experts (Dense Path)
        shared_out = self.shared_experts(x)

        # 2. Router (Gating)
        router_input = x if router_x is None else router_x
        if router_input.shape != x.shape:
            raise ValueError(
                "router_x shape must match x shape in compute_moe_ffn: "
                f"x={tuple(x.shape)}, router_x={tuple(router_input.shape)}"
            )
        if self.freeze_router:
            with torch.no_grad():
                logits = self.router(router_input)
        else:
            logits = self.router(router_input)
        logits_fp32 = logits.to(torch.float32)""",
        ),
        Highlight(
            file=network_modules,
            content="""
For the scoped softmax router, expert choice uses `torch.topk(..., self.top_k)`, where config sets top_k=2. The selected experts receive probabilities from the softmax distribution over all 16 fine-expert logits, renormalized over the two selected experts.""",
            text="""
        if self.routing_score_fn == "softmax":
            choice_logits = logits_fp32
            if bias_fp32 is not None:
                # Keep dynamic bias as a selection correction, not a mixture-weight shaper.
                choice_logits = choice_logits + bias_fp32
            choice_scores = choice_logits
            _, topk_idx = torch.topk(choice_scores, self.top_k, dim=-1)
            dense_distribution = torch.softmax(logits_fp32, dim=-1)
            if torch.onnx.is_in_onnx_export():
                selected_probs = dense_distribution.gather(-1, topk_idx)
            else:
                selected_logits = logits_fp32.gather(-1, topk_idx)""",
        ),
        Highlight(
            file=module_config,
            content="""
The critic is a separate MLP value model, not a transformer/MoE. Its observation normalizer mirrors the actor normalizer; its hidden trunk uses four 2048-wide layers, RMSNorm after each hidden Linear, SiLU activations, and scalar output dimension 1.""",
            text="""
  critic:
    type: MLP

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

    hidden_norm: rmsnorm

    layer_config:
      hidden_dims:
        - 2048
        - 2048
        - 2048
        - 2048
      activation: SiLU""",
        ),
        Highlight(
            file=module_config,
            content="""
The critic schema combines current critic reference, robot state, body-link kinematics, DOF state, last action, and future reference terms. Its `output_dim: 1` is the scalar value prediction consumed by PPO.""",
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
          - unified/critic_last_action""",
        ),
        Highlight(
            file=agent_modules,
            content="""
`PPOCritic` resolves its network class the same way the actor does, assembles TensorDict observations through the configured critic schema, creates an observation normalizer when enabled, and instantiates the configured critic module with inferred input dimension and `output_dim=1`.""",
            text="""
        critic_net_class = getattr(NM, self.critic_net_type, None)
        if critic_net_class is None:
            critic_net_class = globals().get(self.critic_net_type, None)
        if critic_net_class is None or not isinstance(critic_net_class, type):
            available_classes = [
                name
                for name in dir(NM)
                if isinstance(getattr(NM, name, None), type)
            ] + [
                name
                for name, obj in globals().items()
                if isinstance(obj, type)
            ]""",
        ),
        Highlight(
            file=network_modules,
            content="""
The MLP implementation makes the critic layer dimensions and activations operational: for each configured hidden size it appends Linear, the configured normalization (`rmsnorm` for this critic), and the configured activation (`SiLU`), then appends one final Linear to output dimension 1.""",
            text="""
        layer_config = self.module_config_dict["layer_config"]
        hidden_dims: list[int] = list(layer_config.get("hidden_dims", []))
        activation = getattr(nn, str(layer_config["activation"]))()

        layers: list[nn.Module] = []
        prev = self.input_dim
        for h in hidden_dims:
            h_i = int(h)
            layers.append(nn.Linear(prev, h_i))
            layers.append(
                _make_norm(
                    self.hidden_norm_type,
                    h_i,
                    eps=self.hidden_norm_eps,
                )
            )
            layers.append(activation)
            prev = h_i
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_head = nn.Linear(prev, self.output_dim)""",
        ),
        Highlight(
            file=ppo_config,
            content="""
The base PPO config enables policy export and KV-cache export by default, so this architecture is export-relevant after checkpoints are saved. It also requests AdamW, adaptive LR, gradient norm clipping 1.0, clipped value loss, GAE gamma 0.99/lambda 0.95, and 10,001 learning iterations.""",
            text="""
    enable_online_eval: false
    num_learning_iterations: 10001
    log_interval: 5
    save_interval: 500
    export_policy: true
    onnx_name_suffix: null
    use_kv_cache: true
    eval_interval: null
    load_optimizer: true
    headless: ${headless}""",
        ),
        Highlight(
            file=ppo_py,
            content="""
Checkpoint saving triggers ONNX export whenever `export_policy` is true, passing `use_kv_cache` through to the actor export method. Therefore the trainable actor's export path is part of the model contract rather than an unrelated utility.""",
            text="""
        if bool(self.config.get("export_policy", False)):
            export_policy_to_onnx_common(
                self,
                path,
                onnx_name_suffix=self.config.get("onnx_name_suffix", None),
                use_kv_cache=bool(self.config.get("use_kv_cache", True)),
            )""",
        ),
        Highlight(
            file=agent_modules,
            content="""
`PPOTFActor.export_onnx` exports a single-step policy with `obs`, `past_key_values`, and `step_idx` inputs when `use_kv_cache=True`. The exported outputs include `actions`, `present_key_values`, and per-MoE-layer expert indices/logits, so MoE routing is observable in the ONNX artifact.""",
            text="""
            output_names = [
                "actions",
                "present_key_values",
                *self.onnx_routing_output_names(),
            ]

            torch.onnx.export(
                exporter,
                (obs, past_key_values, step_idx),
                str(export_path),
                export_params=True,
                opset_version=opset_version,
                verbose=False,
                dynamo=False,
                input_names=["obs", "past_key_values", "step_idx"],
                output_names=output_names,
            )""",
        ),
    ]
)
