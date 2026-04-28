from project_analysis import File, Highlight, ProofFromCode

train_cfg = File(
    path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_tf-moe.yaml"
)
ppo_cfg = File(path="HoloMotion/holomotion/config/algo/ppo.yaml")
ppo_tf_cfg = File(path="HoloMotion/holomotion/config/algo/ppo_tf.yaml")
train_py = File(path="HoloMotion/holomotion/src/training/train.py")
algo_base = File(path="HoloMotion/holomotion/src/algo/algo_base.py")
ppo_py = File(path="HoloMotion/holomotion/src/algo/ppo.py")
ppo_tf_py = File(path="HoloMotion/holomotion/src/algo/ppo_tf.py")
agent_modules = File(path="HoloMotion/holomotion/src/modules/agent_modules.py")

loss_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=train_cfg,
            content="""
Scope anchor: the motion-tracking TF-MoE training preset selects the transformer PPO algorithm config, the motion-tracking environment/reward stack, and the TF-MoE module config. Therefore the loss proof for the report must include both the shared PPO base loss and the PPOTF sequence/auxiliary extensions used by this preset.""",
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
Training resolves the Hydra `_target_` to a Python class, constructs it with the environment config, algorithm config, log directory, and headless setting, loads an optional checkpoint, and calls `learn()`. This is the dispatch path that makes the configured PPO/PPOTF loss implementation the actual training objective.""",
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
            file=ppo_cfg,
            content="""
The base PPO config declares the canonical PPO loss knobs: 32 rollout steps, 3 epochs, 4 minibatches, ratio clip 0.2, gamma/lambda for returns and GAE, value-loss coefficient 1.0, entropy/noise regularization coefficient 5e-3, gradient clipping at 1.0, clipped value loss enabled, target KL 0.01, and Gaussian initial action noise std 1.0. Symmetry is defined but disabled by default in the base config.""",
            text="""
    num_steps_per_env: 32
    num_learning_epochs: 3
    num_mini_batches: 4
    clip_param: 0.2
    gamma: 0.99
    lam: 0.95
    value_loss_coef: 1.0
    entropy_coef: 5.0e-3
    anneal_entropy: false
    zero_entropy_point: 1.0
    max_grad_norm: 1.0
    use_clipped_value_loss: true
    desired_kl: 0.01
    init_noise_std: 1.0""",
        ),
        Highlight(
            file=ppo_tf_cfg,
            content="""
The TF-MoE preset extends the base PPO config by targeting `PPOTF`, using sequence minibatching, keeping the same ratio clip, entropy coefficient, target KL, and action-noise bounds, and enabling auxiliary state prediction plus the dead-expert MoE margin. Router command/switch/orthogonal/selected-margin terms are configured but disabled here unless a user turns them on.""",
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
    init_noise_std: 1.0
    min_sigma: 0.01
    max_sigma: 1.2
    aux_state_pred:
      enabled: true""",
        ),
        Highlight(
            file=ppo_tf_cfg,
            content="""
For the default TF-MoE motion-tracking preset, auxiliary state prediction contributes contact BCE, base linear-velocity NLL, reference keybody relative-position MSE, and robot keybody relative-position MSE; the enabled MoE regularizer is `dead_expert_margin_to_topk` with weight 10.0. The other router/MoE regularizers have explicit zero/false config here, so the proof treats them as conditional, not as active default loss terms.""",
            text="""
      w_keybody_contact: 1.0e-2
      w_base_lin_vel: 1.0e-2
      w_ref_keybody_rel_pos: 1.0e-1
      w_robot_keybody_rel_pos: 1.0e-1

      min_std: 0.01
      max_std: 2.0

      keybody_contact_names:
        - left_hip_pitch_link
        - right_hip_pitch_link
        - left_knee_link
        - right_knee_link
        - left_ankle_roll_link
        - right_ankle_roll_link
        - left_elbow_link
        - right_elbow_link
        - left_wrist_yaw_link
        - right_wrist_yaw_link

      keybody_rel_pos_names:
        - left_knee_link
        - right_knee_link
        - left_ankle_roll_link
        - right_ankle_roll_link
        - left_elbow_link
        - right_elbow_link
        - left_wrist_yaw_link
        - right_wrist_yaw_link

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
      target: 0.0""",
        ),
        Highlight(
            file=algo_base,
            content="""
Reward is not itself the optimized loss. During rollout, the actor samples actions, the environment returns `rewards`, and those rewards are stored via `process_env_step`; only after `num_steps_per_env` does the runner compute returns/advantages for PPO. This separates environment reward calculation from the later gradient loss computation.""",
            text="""
        obs_dict, rewards, dones, time_outs, infos = self.env.step(actions)

        next_obs_td = self._wrap_obs_dict(obs_dict)
        dones = dones.to(self.device)
        self._last_rollout_dones = dones

        if collect_transition:
            rewards = rewards.to(self.device)
            time_outs = time_outs.to(self.device)
            self.process_env_step(rewards, dones, time_outs, infos)

        if track_episode_stats:
            rewards_for_stats = rewards.to(self.device)
            self._track_episode_stats(rewards_for_stats, dones, infos)
        return next_obs_td""",
        ),
        Highlight(
            file=algo_base,
            content="""
The stored rewards are converted into value targets and advantages with the configured `gamma` and `lam`, then `learn()` calls `update()` to compute and backpropagate the PPO losses. This is the reward-to-return-to-loss boundary: rewards feed targets, but the loss is calculated in the update method.""",
            text="""
            self.storage.compute_returns(
                last_values,
                self.gamma,
                self.lam,
                normalize_advantage=False,
            )

        if getattr(self, "global_advantage_norm", False):
            accelerator = self.accelerator if self.is_distributed else None
            self.storage.normalize_advantages_global_by_command(
                command_name=self.command_name,
                accelerator=accelerator,
                eps=1.0e-8,
            )

    def rollout_policy(self, obs_td: TensorDict) -> TensorDict:
        \"\"\"Collect one rollout with current policy and compute returns.\"\"\"
        actor_was_training = self.actor.training
        critic_was_training = self.critic.training
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            self._reset_rollout_forward_state()
            for _ in range(self.num_steps_per_env):
                obs_td = self._rollout_forward(obs_td)
            self._compute_returns(obs_td)""",
        ),
        Highlight(
            file=algo_base,
            content="""
The outer learning loop confirms the order: collect rollout, then call `update()` for losses and optimizer steps, then log the returned `loss_dict`. This proves reward collection and loss optimization are separate phases in each training iteration.""",
            text="""
            obs_td = self.rollout_policy(obs_td)

            stop = time.time()
            collection_time = stop - start
            start = stop

            loss_dict = self.update()

            stop = time.time()
            learn_time = stop - start

            if self.is_main_process and it % self.log_interval == 0:
                self._log_iteration(
                    it=it,
                    loss_dict=loss_dict,
                    collection_time=collection_time,
                    learn_time=learn_time,
                )""",
        ),
        Highlight(
            file=agent_modules,
            content="""
The actor distribution provides the log probabilities, sigma/noise, and entropy consumed by the PPO loss. In sequence mode, PPOTF computes Gaussian log-probability from `(actions - mu)^2 / sigma^2`, expands the learnable/clamped sigma across action dimensions, and returns entropy as the diagonal Normal entropy sum.""",
            text="""
        # Match sampling-time clamping for stability and consistent KL/log-prob
        sigma_vec = self._sigma_from_params().clamp(
            self.min_sigma, self.max_sigma
        )
        sigma = sigma_vec[None, None, :].expand_as(mu)
        var = sigma * sigma
        logp = -0.5 * (
            ((actions - mu) ** 2) / (var + 1.0e-8)
            + 2.0 * torch.log(sigma + 1.0e-8)
            + math.log(2.0 * math.pi)
        ).sum(dim=-1, keepdim=True)
        entropy = (
            0.5 + 0.5 * math.log(2.0 * math.pi) + torch.log(sigma + 1.0e-8)
        ).sum(dim=-1, keepdim=True)
        return mu, sigma, logp, entropy, aux_preds""",
        ),
        Highlight(
            file=ppo_py,
            content="""
Base PPO reads every relevant loss hyperparameter from config, including clipping, epochs/minibatches, gamma/lambda, value coefficient, entropy schedule, gradient norm, clipped-value flag, and optional symmetry loss. The symmetry term is conditional: `_symmetry_loss_active()` requires the base-velocity command, enabled flag, and positive coefficient.""",
            text="""
        self.clip_param = self.config.clip_param
        self.num_learning_epochs = int(self.config.num_learning_epochs)
        self.configured_num_mini_batches = int(self.config.num_mini_batches)
        if self.configured_num_mini_batches < 1:
            raise ValueError("num_mini_batches must be >= 1.")
        distributed_update_cfg = self.config.get("distributed_update", {})
        self.distributed_update_mode = str(
            distributed_update_cfg.get("mode", "legacy")
        ).lower()
        if self.distributed_update_mode not in {"legacy", "scalable"}:
            raise ValueError(
                "distributed_update.mode must be one of "
                "{'legacy', 'scalable'}."
            )
        self.requested_num_mini_batches = self._resolve_num_mini_batches(
            self.configured_num_mini_batches
        )
        self.num_mini_batches = self.requested_num_mini_batches
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.value_loss_coef = self.config.value_loss_coef
        self.initial_entropy_coef = float(self.config.entropy_coef)""",
        ),
        Highlight(
            file=ppo_py,
            content="""
Entropy annealing is validated and resolved into a scalar coefficient before PPO update. With the default configs above, annealing is disabled, so this returns the configured initial entropy coefficient; if enabled, it linearly decays to zero by `zero_entropy_point` of total iterations.""",
            text="""
    def _get_effective_entropy_coef(self) -> float:
        if self.initial_entropy_coef <= 0.0 or not self.anneal_entropy:
            return float(self.initial_entropy_coef)
        total_learning_iterations = int(
            getattr(
                self,
                "total_learning_iterations",
                self.current_learning_iteration
                + int(self.num_learning_iterations),
            )
        )
        total_learning_iterations = max(1, total_learning_iterations)
        zero_entropy_iteration = float(self.zero_entropy_point) * float(
            total_learning_iterations
        )
        anneal_scale = max(
            0.0,
            1.0
            - float(self.current_learning_iteration) / zero_entropy_iteration,
        )
        return float(self.initial_entropy_coef * anneal_scale)""",
        ),
        Highlight(
            file=ppo_py,
            content="""
This is the base PPO loss implementation. It recomputes actor log-probabilities and critic values on minibatches, forms the PPO ratio `exp(new_logp - old_logp)`, computes the clipped surrogate loss as the mean of the worse unclipped/clipped objective, computes clipped or unclipped value MSE, builds separate actor and critic losses, subtracts entropy regularization, optionally adds the symmetry MSE penalty, then backpropagates, clips gradients, and steps both optimizers.""",
            text="""
            ratio = torch.exp(
                actions_log_prob_batch
                - torch.squeeze(old_actions_log_prob_batch).float()
            )
            clip_fraction = self._compute_clip_fraction(ratio)
            clip_fraction_batch_mean += clip_fraction
            clip_fraction_batch_last = clip_fraction
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch_norm + (
                    value_batch - target_values_batch_norm
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch_norm).pow(2)
                value_losses_clipped = (
                    value_clipped - returns_batch_norm
                ).pow(2)
                value_loss = torch.max(
                    value_losses, value_losses_clipped
                ).mean()
            else:
                value_loss = (returns_batch_norm - value_batch).pow(2).mean()

            actor_loss = surrogate_loss
            critic_loss = self.value_loss_coef * value_loss

            if entropy_coef > 0.0:
                entropy_loss = entropy_batch.mean()
                actor_loss = actor_loss - entropy_coef * entropy_loss
            if symmetry_loss is not None:
                actor_loss = (
                    actor_loss + self.symmetry_loss_coef * symmetry_loss
                )

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.accelerator.backward(actor_loss)
            self.accelerator.backward(critic_loss)

            if self.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_grad_norm,
                )
                self.accelerator.clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_grad_norm,
                )

            self.actor_optimizer.step()
            self.critic_optimizer.step()""",
        ),
        Highlight(
            file=ppo_py,
            content="""
The optional symmetry loss is grounded in mirrored observations and mirrored-back actions: when active, the actor's current `mu` is penalized with MSE against the mirrored policy action transformed back into the original action convention. This is not part of the motion-tracking TF-MoE default, but it is implemented for velocity-tracking PPO when configured.""",
            text="""
                symmetry_loss = None
                if self._symmetry_loss_active():
                    mirrored_obs_batch = self._mirror_actor_obs(obs_batch)
                    mirrored_actor_out = self.actor(
                        mirrored_obs_batch,
                        actions=None,
                        mode="inference",
                        update_obs_norm=False,
                    )
                    mirrored_actions = mirrored_actor_out.get("actions")
                    mirrored_actions_back = self._mirror_env_action(
                        mirrored_actions
                    )
                    symmetry_loss = F.mse_loss(
                        mu_batch.float(),
                        mirrored_actions_back.float(),
                    )""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
PPOTF is a subclass of PPO, so it inherits config setup, rollout/return handling, entropy scheduling, KL helpers, optimizer helpers, and other base behavior unless overridden. Its class-level purpose is the TensorDict sequence-update path for transformer policies.""",
            text="""
class PPOTF(PPO):
    \"\"\"Transformer-policy PPO with TensorDict rollout and sequence update.\"\"\"""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
PPOTF validates and materializes the conditional auxiliary/router/MoE configuration. The booleans and nonnegative weights here decide whether router command reconstruction, router future reconstruction, router switch penalty, dead-expert margin, router-expert orthogonality, and selected-vs-unselected expert margin losses can enter `actor_loss` later.""",
            text="""
        aux_cmd_cfg = self.config.get("aux_router_command_recon", {})
        self.use_aux_router_command_recon: bool = bool(
            aux_cmd_cfg.get("enabled", False)
        )
        self.aux_router_command_recon_weight = float(
            aux_cmd_cfg.get("weight", 0.0)
        )
        self.aux_router_command_recon_hidden_dim = int(
            aux_cmd_cfg.get("hidden_dim", 0)
        )
        self.aux_router_command_recon_term_prefix = str(
            aux_cmd_cfg.get("term_prefix", "actor_ref_")
        )
        aux_switch_cfg = self.config.get("aux_router_switch_penalty", {})
        self.use_aux_router_switch_penalty = bool(
            aux_switch_cfg.get("enabled", False)
        )
        self.aux_router_switch_penalty_weight = float(
            aux_switch_cfg.get("weight", 0.0)
        )
        self.aux_router_switch_penalty_metric = str(
            aux_switch_cfg.get("metric", "js")
        ).lower()
        self.aux_router_switch_penalty_beta = float(
            aux_switch_cfg.get("beta", 1.0)
        )
        aux_router_future_cfg = self.config.get("aux_router_future_recon", {})
        self.use_aux_router_future_recon = bool(
            aux_router_future_cfg.get("enabled", False)
        )
        self.aux_router_future_recon_weight = float(
            aux_router_future_cfg.get("weight", 0.0)
        )
        self.aux_router_future_recon_hidden_dim = int(
            aux_router_future_cfg.get("hidden_dim", 0)
        )
        self.aux_router_future_recon_huber_beta = float(
            aux_router_future_cfg.get("huber_beta", 1.0)
        )
        dead_margin_cfg = self.config.get("dead_expert_margin_to_topk", {})
        self.use_dead_expert_margin_to_topk = bool(
            dead_margin_cfg.get("enabled", False)
        )
        self.dead_expert_margin_to_topk_weight = float(
            dead_margin_cfg.get("weight", 0.0)
        )
        orth_cfg = self.config.get("router_expert_orthogonal", {})
        self.use_router_expert_orthogonal = bool(
            orth_cfg.get("enabled", False)
        )
        self.router_expert_orthogonal_weight = float(
            orth_cfg.get("weight", 0.0)
        )""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
PPOTF computes the same clipped PPO objective as base PPO, but on valid sequence tokens. It forms `ratio`, clamps it, uses `-min(s1, s2)` masked by `valid_tok` for the surrogate, uses clipped value loss over valid tokens, and initializes actor/critic losses before auxiliary additions.""",
            text="""
            logp_new = logp_new_b.squeeze(-1).float()
            logp_old = old_logp_b.squeeze(-1).float()
            ratio = torch.exp(logp_new - logp_old)
            clip_fraction = self._compute_clip_fraction(
                ratio, weight=valid_tok
            )
            clip_fraction_batch_mean += clip_fraction
            clip_fraction_batch_last = clip_fraction
            adv = advantages_b.squeeze(-1)
            s1 = ratio * adv
            s2 = (
                torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                * adv
            )
            surrogate_loss = (
                -torch.min(s1, s2) * valid_tok
            ).sum() / valid_count

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch_norm + (
                    value_batch - target_values_batch_norm
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch_norm).pow(2)
                value_losses_clipped = (
                    value_clipped - returns_batch_norm
                ).pow(2)
                v_max = torch.max(value_losses, value_losses_clipped).squeeze(
                    -1
                )
                value_loss = (v_max * valid_tok).sum() / valid_count
            else:
                v_err = (returns_batch_norm - value_batch).pow(2).squeeze(-1)
                value_loss = (v_err * valid_tok).sum() / valid_count

            actor_loss = surrogate_loss
            critic_loss = self.value_loss_coef * value_loss""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
When auxiliary state prediction is enabled, PPOTF adds supervised actor losses for base linear velocity Gaussian NLL, root-height Gaussian NLL if enabled, keybody contact BCE, reference keybody relative-position MSE, robot keybody relative-position MSE, and denoising Huber terms if their config weights activate them. The default TF-MoE config enables the base velocity/contact/reference keybody/robot keybody branches and leaves root-height/denoise branches off unless configured.""",
            text="""
            if self.use_aux_state_pred:
                aux_base_lin_vel_loc = actor_out.get("aux_base_lin_vel_loc")
                aux_base_lin_vel_log_std = actor_out.get(
                    "aux_base_lin_vel_log_std"
                )
                aux_base_lin_vel_std = torch.clamp(
                    torch.exp(aux_base_lin_vel_log_std),
                    min=self.aux_state_pred_min_std,
                    max=self.aux_state_pred_max_std,
                )
                aux_base_lin_vel_nll = 0.5 * (
                    torch.square(
                        (gt_base_lin_vel_b - aux_base_lin_vel_loc)
                        / aux_base_lin_vel_std
                    )
                    + 2.0 * torch.log(aux_base_lin_vel_std + 1.0e-8)
                ).sum(dim=-1)
                aux_base_lin_vel_loss = (
                    aux_base_lin_vel_nll * valid_tok
                ).sum() / valid_count
                actor_loss = (
                    actor_loss
                    + self.aux_state_pred_w_base_lin_vel
                    * aux_base_lin_vel_loss
                )
                aux_root_height_loc = actor_out.get("aux_root_height_loc")
                aux_root_height_log_std = actor_out.get(
                    "aux_root_height_log_std"
                )
                if self.use_aux_root_height and gt_root_height_b is not None:
                    aux_root_height_std = torch.clamp(
                        torch.exp(aux_root_height_log_std),
                        min=self.aux_state_pred_min_std,
                        max=self.aux_state_pred_max_std,
                    )
                    aux_root_height_nll = 0.5 * (
                        torch.square(
                            (gt_root_height_b - aux_root_height_loc)
                            / aux_root_height_std
                        )
                        + 2.0 * torch.log(aux_root_height_std + 1.0e-8)
                    ).sum(dim=-1)
                    aux_root_height_loss = (
                        aux_root_height_nll * valid_tok
                    ).sum() / valid_count
                    actor_loss = (
                        actor_loss
                        + self.aux_state_pred_w_root_height
                        * aux_root_height_loss
                    )
                else:
                    actor_loss = actor_loss + 0.0 * (
                        aux_root_height_loc.sum()
                        + aux_root_height_log_std.sum()
                    )
                if (
                    self.aux_state_pred_num_contact_bodies > 0
                    and gt_keybody_contact_b is not None
                ):
                    aux_keybody_contact_logits = actor_out.get(
                        "aux_keybody_contact_logits"
                    )
                    contact_bce = F.binary_cross_entropy_with_logits(
                        aux_keybody_contact_logits,
                        gt_keybody_contact_b,
                        reduction="none",
                    ).mean(dim=-1)
                    aux_keybody_contact_loss = (
                        contact_bce * valid_tok
                    ).sum() / valid_count
                    actor_loss = (
                        actor_loss
                        + self.aux_state_pred_w_keybody_contact
                        * aux_keybody_contact_loss
                    )""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
The remaining PPOTF auxiliary-state branches add reference and robot keybody MSEs and optional denoising Huber losses with their configured weights. These losses are actor-side additions, not environment rewards.""",
            text="""
                if (
                    self.aux_state_pred_num_keybody_bodies > 0
                    and gt_ref_keybody_rel_pos_b is not None
                ):
                    aux_ref_keybody_rel_pos_loss = (
                        self._masked_aux_keybody_mse(
                            aux_ref_keybody_rel_pos,
                            gt_ref_keybody_rel_pos_b,
                            valid_tok,
                        )
                    )
                    actor_loss = (
                        actor_loss
                        + self.aux_state_pred_w_ref_keybody_rel_pos
                        * aux_ref_keybody_rel_pos_loss
                    )
                elif aux_ref_keybody_rel_pos.numel() > 0:
                    actor_loss = (
                        actor_loss + 0.0 * aux_ref_keybody_rel_pos.sum()
                    )
                if (
                    self.aux_state_pred_num_keybody_bodies > 0
                    and gt_robot_keybody_rel_pos_b is not None
                ):
                    aux_robot_keybody_rel_pos_loss = (
                        self._masked_aux_keybody_mse(
                            aux_robot_keybody_rel_pos,
                            gt_robot_keybody_rel_pos_b,
                            valid_tok,
                        )
                    )
                    actor_loss = (
                        actor_loss
                        + self.aux_state_pred_w_robot_keybody_rel_pos
                        * aux_robot_keybody_rel_pos_loss
                    )
                elif aux_robot_keybody_rel_pos.numel() > 0:
                    actor_loss = (
                        actor_loss + 0.0 * aux_robot_keybody_rel_pos.sum()
                    )
                if self.use_aux_denoise_ref_root_lin_vel:
                    aux_denoise_ref_root_lin_vel_residual = actor_out.get(
                        "aux_denoise_ref_root_lin_vel_residual"
                    )
                    aux_denoise_ref_root_lin_vel_loss = self._masked_aux_huber(
                        pred=aux_denoise_ref_root_lin_vel_residual,
                        target=gt_denoise_ref_root_lin_vel_b,
                        valid_tok=valid_tok,
                        beta=self.aux_denoise_residual_huber_beta,
                    )
                    actor_loss = (
                        actor_loss
                        + self.aux_state_pred_w_denoise_ref_root_lin_vel
                        * aux_denoise_ref_root_lin_vel_loss
                    )
                if self.use_aux_denoise_ref_root_ang_vel:
                    aux_denoise_ref_root_ang_vel_residual = actor_out.get(
                        "aux_denoise_ref_root_ang_vel_residual"
                    )
                    aux_denoise_ref_root_ang_vel_loss = self._masked_aux_huber(
                        pred=aux_denoise_ref_root_ang_vel_residual,
                        target=gt_denoise_ref_root_ang_vel_b,
                        valid_tok=valid_tok,
                        beta=self.aux_denoise_residual_huber_beta,
                    )
                    actor_loss = (
                        actor_loss
                        + self.aux_state_pred_w_denoise_ref_root_ang_vel
                        * aux_denoise_ref_root_ang_vel_loss
                    )
                if self.use_aux_denoise_ref_dof_pos:
                    aux_denoise_ref_dof_pos_residual = actor_out.get(
                        "aux_denoise_ref_dof_pos_residual"
                    )
                    aux_denoise_ref_dof_pos_loss = self._masked_aux_huber(
                        pred=aux_denoise_ref_dof_pos_residual,
                        target=gt_denoise_ref_dof_pos_b,
                        valid_tok=valid_tok,
                        beta=self.aux_denoise_residual_huber_beta,
                    )
                    actor_loss = (
                        actor_loss
                        + self.aux_state_pred_w_denoise_ref_dof_pos
                        * aux_denoise_ref_dof_pos_loss
                    )""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
When configured, PPOTF adds router command reconstruction MSE, future reconstruction Huber, adjacent-router switch penalty, dead-expert top-k margin, routed-expert orthogonality, selected-expert margin, optional token KL penalty, and entropy regularization to the actor loss. In the default TF-MoE preset, the dead-expert margin is active while the other router regularizers and `kl_coef` are disabled by config.""",
            text="""
            if self.use_aux_router_command_recon:
                if self.aux_router_command_recon_assembler is None:
                    raise ValueError(
                        "aux_router_command_recon is enabled but command "
                        "assembler was not initialized."
                    )
                aux_router_command_recon_pred = actor_out.get(
                    "aux_router_command_recon"
                )
                gt_aux_router_command_recon_b = (
                    self.aux_router_command_recon_assembler(
                        obs_b.flatten(0, 1)
                    ).reshape(b, t, -1)
                )
                aux_router_command_recon_loss = self._masked_aux_mse(
                    aux_router_command_recon_pred,
                    gt_aux_router_command_recon_b,
                    valid_tok,
                )
                actor_loss = (
                    actor_loss
                    + self.aux_router_command_recon_weight
                    * aux_router_command_recon_loss
                )
            if self.use_aux_router_future_recon:
                aux_router_future_recon_loss = (
                    self._compute_aux_router_future_recon_loss(
                        actor_wrapper=actor_unwrapped,
                        actor_out=actor_out,
                        obs_b=obs_b,
                        valid_tok=valid_tok,
                    )
                )
                actor_loss = (
                    actor_loss
                    + self.aux_router_future_recon_weight
                    * aux_router_future_recon_loss
                )
            if self.use_aux_router_switch_penalty:
                if self.aux_router_switch_penalty_metric == "js":
                    aux_router_features = actor_out.get("router_features")
                    aux_router_switch_penalty_loss = self._masked_adjacent_router_js(
                        router_features=aux_router_features,
                        valid_tok=valid_tok,
                        num_moe_layers=self.aux_command_router_num_moe_layers,
                        num_fine_experts=self.aux_command_router_num_fine_experts,
                    )
                else:
                    aux_router_temporal_features = actor_out.get(
                        "router_temporal_features"
                    )
                    aux_router_switch_penalty_loss = self._masked_adjacent_router_normed_smooth_l1(
                        router_temporal_features=aux_router_temporal_features,
                        valid_tok=valid_tok,
                        num_moe_layers=self.aux_command_router_num_moe_layers,
                        num_fine_experts=self.aux_command_router_num_fine_experts,
                        beta=self.aux_router_switch_penalty_beta,
                    )
                aux_router_switch_penalty_loss = (
                    aux_router_switch_penalty_loss.to(actor_loss.dtype)
                )
                actor_loss = (
                    actor_loss
                    + self.aux_router_switch_penalty_weight
                    * aux_router_switch_penalty_loss
                )
            if self.use_dead_expert_margin_to_topk and len(moe_layers) > 0:
                margin_losses = [
                    layer.last_dead_expert_margin_to_topk_loss
                    for layer in moe_layers
                    if layer.last_dead_expert_margin_to_topk_loss is not None
                ]
                if len(margin_losses) > 0:
                    dead_expert_margin_to_topk_loss = torch.stack(
                        [
                            loss.to(actor_loss.device, dtype=actor_loss.dtype)
                            for loss in margin_losses
                        ]
                    ).mean()
                    actor_loss = (
                        actor_loss
                        + self.dead_expert_margin_to_topk_weight
                        * dead_expert_margin_to_topk_loss
                    )""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
This closes the PPOTF gradient/update flow: after all PPO, auxiliary, router/MoE, KL, and entropy actor-loss terms are combined, the actor and critic optimizers are zeroed, actor and critic losses are backpropagated separately through Accelerate, gradients are clipped if configured, and both optimizers step.""",
            text="""
            if entropy_coef > 0.0:
                ent_tok = entropy_b.squeeze(-1)
                entropy_loss = (ent_tok * valid_tok).sum() / valid_count
                actor_loss = actor_loss - entropy_coef * entropy_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.accelerator.backward(actor_loss)
            self.accelerator.backward(critic_loss)

            if self.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.accelerator.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
            self.actor_optimizer.step()
            self.critic_optimizer.step()""",
        ),
    ]
)
