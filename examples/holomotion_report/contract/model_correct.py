from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

algo_cfg = File(path="HoloMotion/holomotion/config/algo/ppo.yaml")
algo_tf_cfg = File(path="HoloMotion/holomotion/config/algo/ppo_tf.yaml")
train_tf_moe_cfg = File(path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_tf-moe.yaml")
module_tf_moe_cfg = File(path="HoloMotion/holomotion/config/modules/motion_tracking/motion_tracking_tf-moe.yaml")
ppo = File(path="HoloMotion/holomotion/src/algo/ppo.py")
ppo_tf = File(path="HoloMotion/holomotion/src/algo/ppo_tf.py")
agent_modules = File(path="HoloMotion/holomotion/src/modules/agent_modules.py")
network_modules = File(path="HoloMotion/holomotion/src/modules/network_modules.py")

model_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=algo_cfg,
            content="""
Base answer: HoloMotion is actor-critic PPO. The base PPO config selects the PPO algorithm and passes separate actor/critic module configs through `module_dict`.""",
            text="""
algo:
  _target_: holomotion.src.algo.ppo.PPO
  _recursive_: false
  config:
    # --- General Settings ---""",
        ),
        Highlight(
            file=ppo,
            content="""
The normal PPO setup constructs two separate modules: `self.actor = PPOActor(...)` and `self.critic = PPOCritic(...)`. That is the actor-critic split.""",
            text="""
        self.actor = PPOActor(
            obs_schema=actor_schema,
            module_config_dict=actor_cfg,
            num_actions=self.num_actions,
            init_noise_std=self.config.init_noise_std,
            obs_example=sample_td,
        ).to(self.device)""",
        ),
        Highlight(
            file=ppo,
            content="""
The value side is explicitly a separate critic object built from `critic_schema` and `critic_cfg`.""",
            text="""
        self.critic = PPOCritic(
            obs_schema=critic_schema,
            module_config_dict=critic_cfg,
            obs_example=sample_td,
        ).to(self.device)""",
        ),
        Highlight(
            file=train_tf_moe_cfg,
            content="""
When choosing the Transformer-MoE path, the training defaults switch to `/algo: ppo_tf`, `/env/observations: motion_tracking/obs_motion_tracking_tf-moe`, and `/modules: motion_tracking/motion_tracking_tf-moe`.""",
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
            file=algo_tf_cfg,
            content="""
The Transformer-MoE training path targets `holomotion.src.algo.ppo_tf.PPOTF`, so the PPO actor-critic trainer becomes the Transformer-policy PPO subclass.""",
            text="""
defaults:
  - ppo

algo:
  _target_: holomotion.src.algo.ppo_tf.PPOTF
  config:""",
        ),
        Highlight(
            file=ppo_tf,
            content="""
Important distinction: `PPOTF` chooses a specialized actor wrapper for `ReferenceRoutedGroupedMoETransformerPolicy`, but this selection is for the actor only. The full architecture is Transformer-MoE actor plus separate critic, not a Transformer-MoE critic.""",
            text="""
        if actor_type == "ReferenceRoutedGroupedMoETransformerPolicy":
            if use_future_cross_attn:
                raise ValueError(
                    "ReferenceRoutedGroupedMoETransformerPolicy does not "
                    "support use_future_cross_attn=True."
                )
            return PPOTFRefRouterActor""",
        ),
        Highlight(
            file=ppo_tf,
            content="""
`PPOTF` infers the flattened actor observation dimension from the actor schema, selects the actor wrapper class, and injects `input_dim_override` before construction. For the TF-MoE config, actor observation is H=1 current frame plus T=10 future frames: current 132 + future 390 = 522 input dims.""",
            text="""
        actor_obs_dim = int(
            TensorDictAssembler(
                actor_schema, output_mode="flat"
            ).infer_output_dim(sample_td)
        )
        use_future_cross_attn = bool(
            actor_cfg.get("use_future_cross_attn", False)
        )
        actor_cls = self._select_actor_wrapper_cls(actor_cfg)""",
        ),
        Highlight(
            file=ppo_tf,
            content="""
After actor setup, `PPOTF` still constructs the critic with `PPOCritic`, using the critic schema and critic module config.""",
            text="""
        self.critic = PPOCritic(
            obs_schema=critic_schema,
            module_config_dict=critic_cfg,
            obs_example=sample_td,
        ).to(self.device)""",
        ),
        Highlight(
            file=module_tf_moe_cfg,
            content="""
Transformer-MoE actor config:
- type: `ReferenceRoutedGroupedMoETransformerPolicy`.
- fine experts: 16.
- shared experts: 1.
- top_k: 2 selected fine experts per token.
- output: `robot_action_dim`, so for the 29-DoF G1 policy this emits 29 actor means.
This is the actor/policy network.""",
            text="""
modules:
  actor:
    type: ReferenceRoutedGroupedMoETransformerPolicy

    use_checkpointing: false # use gradient checkpointing to save GRAM significantly

    # MoE-specific hyperparameters
    num_fine_experts: 16
    num_shared_experts: 1
    top_k: 2""",
        ),
        Highlight(
            file=module_tf_moe_cfg,
            content="""
Transformer-MoE actor Transformer shape:
- observation embedding hidden: 2048.
- router/reference embedding hidden: 2048.
- d_model: 512.
- attention heads: 8 query heads and 4 KV heads.
- gated attention: enabled.
- layers: 3.
- max context length: 32.
- dense FF multiplier: 4.
- MoE FF multiplier: 2.0.
- attention and MLP dropout: 0.0.""",
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
            file=module_tf_moe_cfg,
            content="""
The TF-MoE actor consumes a schema split into current observations (`flattened_obs`, seq_len=1 in the TF-MoE obs config) and future reference observations (`flattened_obs_fut`, seq_len=10). The current terms contribute 132 dims and future terms contribute 390 dims, so actor input is 522 dims when choosing TF-MoE.""",
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
            file=network_modules,
            content="""
Causal-attention evidence, step 1: the Transformer actor API is sequence-based. `sequence_mu` receives flat observation tokens shaped `[B, T, D]` and returns action means `[B, T, A]`. Its own docstring states that `attn_mask=None` means causal attention. For the TF-MoE config, D=522 and A=29.""",
            text="""
    def sequence_mu(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
        return_pre_moe_hidden: bool = False,
        return_router_features: bool = False,
        return_router_temporal_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        \"\"\"Compute per-token action mean for sequences.

        Args:
            x: [B, T, D] flat obs per token.
            attn_mask: [B, T, T] boolean mask (True if attend allowed), or None for causal.
            return_hidden: If True, also return the hidden states.

        Returns:
            mu: [B, T, A]
            h: [B, T, d_model] (only if return_hidden=True)
        \"\"\"""",
        ),
        Highlight(
            file=network_modules,
            content="""
Causal-attention evidence, step 2: inside the attention kernel call, HoloMotion passes `is_causal=(mask is None)`. Therefore, when the actor calls `sequence_mu(..., attn_mask=None)`, the Transformer self-attention is causal over the token/history dimension; it is not a future-reference cross-attention path.""",
            text="""
        attn_out = export_safe_scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=(mask is None),
            enable_gqa=enable_gqa,
        )""",
        ),
        Highlight(
            file=module_tf_moe_cfg,
            content="""
Critic config when choosing Transformer-MoE:
- critic type: `MLP`, not Transformer-MoE.
- observation normalization enabled.
- hidden norm: RMSNorm.
The critic is the value function side of the actor-critic architecture.""",
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

    hidden_norm: rmsnorm""",
        ),
        Highlight(
            file=module_tf_moe_cfg,
            content="""
Critic MLP details: the value network is a 4-layer 2048-wide MLP with SiLU activation and output dimension 1. Using the critic observation schema, its input dimension is 1046.""",
            text="""
    layer_config:
      hidden_dims:
        - 2048
        - 2048
        - 2048
        - 2048
      activation: SiLU""",
        ),
        Highlight(
            file=module_tf_moe_cfg,
            content="""
The critic output dimension is one scalar value estimate.""",
            text="""
    output_dim: 1""",
        ),
        Highlight(
            file=network_modules,
            content="""
The implementation confirms the actor class is `ReferenceRoutedGroupedMoETransformerPolicy`, a subclass of `GroupedMoETransformerPolicy`.""",
            text="""
class ReferenceRoutedGroupedMoETransformerPolicy(GroupedMoETransformerPolicy):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        module_config_dict: dict,
    ):""",
        ),
        Highlight(
            file=network_modules,
            content="""
The base grouped-MoE Transformer embeds observations into d_model, then stacks Transformer blocks. Layer 0 is always dense; with n_layers=3 and no dense final layer, layers 1 and 2 are `GroupedMoEBlock`s.""",
            text="""
        # Stack of TransformerBlocks: first layer is always dense; the last
        # layer is also dense when dense_layer_at_last=True.
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            use_dense_layer = i == 0 or (
                self.dense_layer_at_last and i == self.n_layers - 1
            )""",
        ),
        Highlight(
            file=network_modules,
            content="""
The actor output head normalizes the final hidden state and maps 512 -> 512 -> action dimension with SiLU in between. This produces actor mean actions (`mu`).""",
            text="""
        self.norm_f = RMSNorm(self.d_model)
        self.action_mu_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.output_dim),
        )""",
        ),
        Highlight(
            file=network_modules,
            content="""
Each MoE block has modern attention plus a router and sparse expert parameters: router linear to 16 fine experts, expert-specific gate/up weights, expert-specific down weights, and shared experts.""",
            text="""
        self.router = nn.Linear(d_model, num_fine_experts, bias=False)
        self._apply_freeze_router_state()

        # Gate + Up (Combined)
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                num_fine_experts, self.d_model, 2 * self.intermediate_dim
            )
        )
        # Down
        self.down_proj = nn.Parameter(
            torch.empty(num_fine_experts, self.intermediate_dim, self.d_model)
        )""",
        ),
        Highlight(
            file=network_modules,
            content="""
The MoE forward path computes a dense shared-expert output, routes each token, computes sparse top-k expert output, and combines shared plus sparse paths.""",
            text="""
        # 1. Shared Experts (Dense Path)
        shared_out = self.shared_experts(x)

        # 2. Router (Gating)
        router_input = x if router_x is None else router_x
        if router_input.shape != x.shape:
            raise ValueError(
                "router_x shape must match x shape in compute_moe_ffn: "
                f"x={tuple(x.shape)}, router_x={tuple(router_input.shape)}"
            )""",
        ),
        Highlight(
            file=network_modules,
            content="""
The reference-routed actor builds router features from current and future reference motion, processes them through a reference history attention module, projects them per MoE layer, and supplies those projections as `router_x` to MoE blocks. That is the meaning of reference-routed Transformer-MoE here.""",
            text="""
        ref_hist_h = self._build_ref_hist_hidden(
            ref_motion_x=ref_motion_x,
            pos=pos,
            tgt_mask=tgt_mask,
        )
        shared_router_summary = self._build_shared_router_summary(ref_hist_h)
        router_h_per_layer = self._build_router_h_per_layer(
            shared_router_summary
        )
        cos, sin = self.get_cos_sin(h, pos)
        if return_hidden and return_pre_moe_hidden:
            raise ValueError(
                "return_hidden and return_pre_moe_hidden cannot both be True."
            )""",
        ),
        Highlight(
            file=agent_modules,
            content="""
The critic wrapper resolves `type: MLP`, assembles critic observations, normalizes them, runs the MLP, and writes `values`, confirming the critic is a conventional value network.""",
            text="""
        self.critic_net_type = module_config_dict.get("type", "MLP")
        obs_norm_cfg = module_config_dict.get("obs_norm", {})
        self.obs_norm_enabled = bool(obs_norm_cfg.get("enabled", False))

        if self.obs_norm_enabled:
            self.obs_norm_clip = float(obs_norm_cfg.get("clip_range", 0.0))
            self.obs_norm_eps = float(obs_norm_cfg.get("epsilon", 1.0e-8))""",
        ),
    ]
)
