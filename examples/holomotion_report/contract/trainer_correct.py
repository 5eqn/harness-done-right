from project_analysis import File, Highlight, ProofFromCode


train_cfg = File(
    path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_tf-moe.yaml"
)
env_cfg = File(path="HoloMotion/holomotion/config/env/motion_tracking.yaml")
ppo_cfg = File(path="HoloMotion/holomotion/config/algo/ppo.yaml")
ppo_tf_cfg = File(path="HoloMotion/holomotion/config/algo/ppo_tf.yaml")
train_py = File(path="HoloMotion/holomotion/src/training/train.py")
config_py = File(path="HoloMotion/holomotion/src/utils/config.py")
algo_base = File(path="HoloMotion/holomotion/src/algo/algo_base.py")
ppo_tf_py = File(path="HoloMotion/holomotion/src/algo/ppo_tf.py")
motion_tracking_env = File(path="HoloMotion/holomotion/src/env/motion_tracking.py")


trainer_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=train_cfg,
            content="""
Scope anchor: this proof follows the G1 29-DoF motion-tracking TF-MoE training preset. The Hydra defaults compose the train base, PPOTF algorithm, Unitree G1 robot, motion-tracking environment, termination/observation/reward groups, medium domain randomization, rough terrain, and TF-MoE module group into the trainer run.""",
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
The training entrypoint is a Hydra application rooted at `holomotion/config` with base config `training/train_base`. Supplying the scoped `training/motion_tracking/train_g1_29dof_motion_tracking_tf-moe` config replaces/extends that base through the defaults stack above, so launch-time composition is Hydra-driven rather than manual Python argument parsing.""",
            text="""
@hydra.main(
    config_path="../../config",
    config_name="training/train_base",
    version_base=None,
)
def main(config: OmegaConf):
    \"\"\"Train the motion tracking model.""",
        ),
        Highlight(
            file=train_py,
            content="""
The entrypoint immediately compiles the resolved config. For distributed runs it creates an `Accelerator` early, broadcasts rank zero's resolved experiment directory to every worker, then uses that shared path as `log_dir`; this avoids each rank writing to a different timestamped Hydra directory.""",
            text="""
    config = compile_config(config, accelerator=None)
    dist = None

    # In distributed runs, Hydra resolves ${now:...} per process so experiment_save_dir
    # can differ by rank (e.g. staggered startup). Use Accelerator to init the process
    # group, then broadcast rank 0's path so all ranks write to the same directory.
    if getattr(config, "num_processes", 1) > 1:
        project_config = ProjectConfiguration(
            project_dir=config.experiment_save_dir,
            logging_dir=config.experiment_save_dir,
        )
        _accelerator = Accelerator(project_config=project_config)
        import torch.distributed as dist

        path_list = (
            [config.experiment_save_dir]
            if _accelerator.is_main_process
            else [None]
        )
        dist.broadcast_object_list(path_list, src=0)
        config.experiment_save_dir = path_list[0]""",
        ),
        Highlight(
            file=train_py,
            content="""
After config compilation, the trainer resolves the configured algorithm `_target_`, constructs that class with the resolved environment config and algorithm config, loads an optional checkpoint, and starts `learn()`. With the scoped config, this dispatches to `holomotion.src.algo.ppo_tf.PPOTF`.""",
            text="""
    log_dir = config.experiment_save_dir
    headless = config.headless
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
            file=ppo_tf_cfg,
            content="""
The composed algorithm is explicitly PPOTF, not the plain PPO fallback. The config fixes the rollout horizon, PPO epochs/minibatches, clipping, entropy, target KL, actor/critic learning rates, trainable sigma bounds, and auxiliary state-prediction switch used by the update path.""",
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
            file=ppo_cfg,
            content="""
The inherited PPO base config supplies trainer-level loop, checkpoint/log cadence, Accelerate precision/compile settings, optimizer type, adaptive schedule, distributed advantage normalization, and the actor/critic module dictionary. PPOTF overrides selected values but still inherits these operational trainer settings through `defaults: - ppo`.""",
            text="""
    # --- General Settings ---
    enable_online_eval: false
    num_learning_iterations: 10001
    log_interval: 5
    save_interval: 500
    export_policy: true
    onnx_name_suffix: null
    use_kv_cache: true
    eval_interval: null
    load_optimizer: true
    headless: ${headless}
    # ---

    # --- Accelerate Settings ---
    mixed_precision: null # "fp16", "bf16", or null. Use "bf16" for A100/H100, "fp16" for older GPUs
    dynamo_backend: "inductor" # "inductor", "aot_eager", "cudagraphs", or null. Enables automatic model compilation during prepare()
    # ---""",
        ),
        Highlight(
            file=config_py,
            content="""
`compile_config` is the launch-time normalization pass: it registers resolvers, deep-copies the OmegaConf object, fills Accelerate/distributed metadata, creates directories, then propagates device/process data into nested environment config fields.""",
            text="""
    setup_hydra_resolvers()
    config = copy.deepcopy(config)
    config = compile_config_hf_accelerate(config, accelerator)
    config = compile_config_directories(config, eval)
    config = compile_config_devices(config)
    return config""",
        ),
        Highlight(
            file=config_py,
            content="""
When an `Accelerator` instance is not yet available, config compilation still derives rank, world size, local rank, and device from `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, and Accelerate equivalents. It writes `process_id`, `num_processes`, and `main_process` back into the config, making distributed metadata concrete before environment construction.""",
            text="""
    else:
        # Best-effort distributed metadata when running under torchrun / Accelerate launch,
        # even if an Accelerator instance is not provided yet.
        process_idx = int(
            os.environ.get(
                "RANK", os.environ.get("ACCELERATE_PROCESS_INDEX", "0")
            )
        )
        total_processes = int(
            os.environ.get(
                "WORLD_SIZE", os.environ.get("ACCELERATE_NUM_PROCESSES", "1")
            )
        )
        local_rank = int(
            os.environ.get(
                "LOCAL_RANK",
                os.environ.get("ACCELERATE_LOCAL_PROCESS_INDEX", "0"),
            )
        )
        is_main_process = process_idx == 0
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

    config.process_id = process_idx
    config.num_processes = total_processes
    config.main_process = is_main_process""",
        ),
        Highlight(
            file=config_py,
            content="""
The resolved distributed metadata is then copied into `config.env.config`, along with the selected simulation/RL/PhysX device strings. This is the concrete bridge from launch process state into the IsaacLab environment configuration.""",
            text="""
        env_cfg.num_processes = world_size
        env_cfg.process_id = process_rank
        env_cfg.main_process = is_main_process
        env_cfg.simulation_device = device_str
        for key in [
            "sim_device",
            "rl_device",
            "compute_device",
            "physx_device",
        ]:
            setattr(env_cfg, key, device_str)""",
        ),
        Highlight(
            file=algo_base,
            content="""
The algorithm constructor always sets up Accelerator first, then the logger, environment, configs, seeding, buffers, algorithm components, and models/optimizers. That order matters: device/rank state exists before IsaacLab is launched or modules are prepared.""",
            text="""
        self._setup_accelerator()
        self.algo_logger = AlgoLogger(
            self.accelerator,
            self.log_dir,
            is_main_process=self.is_main_process,
        )
        self._setup_environment()
        self._setup_configs()
        self._setup_seeding()
        self._setup_data_buffers()
        self._setup_algo_components()
        self._setup_models_and_optimizer()""",
        ),
        Highlight(
            file=algo_base,
            content="""
Accelerator setup is concrete: it creates an `Accelerator` with TensorBoard project config, records local rank, uses the Accelerator device, sets the active CUDA device when applicable, and stores process/world/rank fields used by later distributed logic.""",
            text="""
        accelerator_kwargs["project_config"] = project_config
        self.accelerator = Accelerator(**accelerator_kwargs)
        self.local_rank = getattr(
            self.accelerator, "local_process_index", None
        )
        if self.local_rank is None:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.device = self.accelerator.device
        if torch.cuda.is_available() and self.device.type == "cuda":
            dev_index = self.device.index
            if dev_index is None:
                dev_index = int(self.local_rank)
                self.device = torch.device("cuda", dev_index)
            else:
                dev_index = int(dev_index)
            torch.cuda.set_device(dev_index)
        self.is_main_process = self.accelerator.is_main_process""",
        ),
        Highlight(
            file=algo_base,
            content="""
The trainer launches IsaacLab on the Accelerator-selected device. It optionally staggers AppLauncher startup by local rank, disables Omniverse multi-GPU rendering per process, and passes `headless`, camera/video, device, and kit args into `AppLauncher`.""",
            text="""
        if self.is_distributed:
            self.accelerator.wait_for_everyone()
            base_delay_s = float(
                os.environ.get("HOLOMOTION_ISAAC_STAGGER_SEC", "5.0")
            )
            local_rank = int(self.local_rank)
            delay_s = base_delay_s * float(local_rank)
            if delay_s > 0.0:
                logger.info(
                    f"[Global Rank {self.gpu_global_rank}, Local Rank {local_rank}] "
                    f"Sleeping {delay_s:.1f}s before IsaacSim AppLauncher init"
                )
            time.sleep(delay_s)""",
        ),
        Highlight(
            file=algo_base,
            content="""
Environment construction is dynamic but grounded: `env_config._target_` resolves to the configured class, render mode follows the video flag, and the class receives the resolved env config, device string, log dir, headless flag, live Accelerator, and render mode before an initial `reset_all()`.""",
            text="""
        env_class = get_class(self.env_config._target_)

        render_mode = (
            "rgb_array"
            if bool(self.config.get("record_video", False))
            else None
        )
        self.env = env_class(
            config=self.env_config.config,
            device=device_str,
            headless=self.headless,
            log_dir=self.log_dir,
            accelerator=self.accelerator,
            render_mode=render_mode,
        )

        _ = self.env.reset_all()""",
        ),
        Highlight(
            file=env_cfg,
            content="""
The environment target in the composed config is the motion-tracking wrapper, and its config declares the shared process metadata, robot/domain-randomization/reward/terrain/observation/termination groups, position-control action term, and `ref_motion` command term used during trainer rollout.""",
            text="""
env:
  _target_: holomotion.src.env.motion_tracking.MotionTrackingEnv
  _recursive_: False
  config:
    experiment_name: ${experiment_name}
    num_envs: ${num_envs}
    env_spacing: 2.5 # meters
    replicate_physics: true
    headless: ${headless}
    num_processes: ${num_processes}
    main_process: ${main_process}
    process_id: ${process_id}
    ckpt_dir: null
    disable_ref_viz: false
    eval_log_dir: null
    save_rendering_dir: null""",
        ),
        Highlight(
            file=motion_tracking_env,
            content="""
Inside `MotionTrackingEnv`, the wrapper materializes OmegaConf groups into plain config dictionaries for robot, terrain, observations, rewards, domain randomization, terminations, scene, commands, simulation, actions, and curriculum. These are the exact inputs used to build the IsaacLab manager-based environment.""",
            text="""
        _robot_config_dict = EasyDict(
            OmegaConf.to_container(self.config.robot, resolve=True)
        )
        _terrain_config_dict = EasyDict(
            OmegaConf.to_container(self.config.terrain, resolve=True)
        )
        _obs_config_dict = EasyDict(
            OmegaConf.to_container(self.config.obs, resolve=True)
        )
        _rewards_config_dict = EasyDict(
            OmegaConf.to_container(self.config.rewards, resolve=True)
        )
        _domain_rand_config_dict = (
            EasyDict(
                OmegaConf.to_container(
                    self.config.domain_rand,
                    resolve=True,
                )
            )
            if self.config.domain_rand is not None
            else {}
        )""",
        ),
        Highlight(
            file=motion_tracking_env,
            content="""
The IsaacLab config class wires scene, commands, observations, rewards, terminations, domain-randomization events, curriculum, actions, and simulation parameters, then instantiates `ManagerBasedRLEnv`. This grounds the trainer's `env.step()` calls in IsaacLab manager-based action/reward/termination/observation execution.""",
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
                terminations: TerminationsCfg = TerminationsCfg()

            if _domain_rand_config_dict:
                events: EventsCfg = build_domain_rand_config(
                    _domain_rand_config_dict
                )
            else:
                events: EventsCfg = EventsCfg()""",
        ),
        Highlight(
            file=motion_tracking_env,
            content="""
The final environment object is the IsaacLab `ManagerBasedRLEnv` created from `MotionTrackingEnvCfg` and the requested render mode. The config is also dumped to the log directory, giving the run a reproducible copy of the concrete environment config.""",
            text="""
        isaaclab_envconfig_dump_path = os.path.join(
            self.log_dir, "isaaclab_env_cfg.yaml"
        )
        dump_yaml(isaaclab_envconfig_dump_path, isaaclab_env_cfg)

        self._env = ManagerBasedRLEnv(isaaclab_env_cfg, self.render_mode)

        logger.info("IsaacLab environment initialized !")
        return self._env""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
PPOTF builds actor and critic from a real reset observation sample, resolves actor/critic module configs, injects sigma and auxiliary settings, validates actor schema, chooses the TF actor wrapper, and constructs the actor and critic on the selected device.""",
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
Optimizer setup is separate for actor and critic. AdamW uses parameter groups so actor weight decay skips low-dimensional/bias/norm/log-std parameters, while the critic optimizer uses all critic parameters. Both optimizers are then wrapped with `accelerator.prepare`, so DDP/mixed precision preparation covers models and optimizers together.""",
            text="""
        optimizer_class = getattr(optim, self.optimizer_type)
        optimizer_kwargs = self._build_optimizer_kwargs(optimizer_class)
        if self.optimizer_type == "AdamW":
            decay_params = []
            non_decay_params = []
            for name, p in self.actor.named_parameters():
                if not p.requires_grad:
                    continue
                if (
                    p.ndim < 2
                    or ("log_std" in name)
                    or ("bias" in name)
                    or ("norm" in name)
                ):
                    non_decay_params.append(p)
                else:
                    decay_params.append(p)""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
The actor, critic, and their optimizers are passed through Accelerate as a single unit. After preparation, PPOTF resets the actor KV cache for the vectorized environment, so the rollout state is sized to the actual number of envs on the selected device.""",
            text="""
        (
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.critic_optimizer,
        ) = self.accelerator.prepare(
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.critic_optimizer,
        )

        actor_for_kv = self.accelerator.unwrap_model(self.actor)
        if hasattr(actor_for_kv, "reset_kv_cache"):
            actor_for_kv.reset_kv_cache(self.env.num_envs, self.device)""",
        ),
        Highlight(
            file=algo_base,
            content="""
Each rollout step runs the actor under Accelerator autocast, optionally evaluates the critic for storage, builds a transition, extracts sampled actions, and calls `self.env.step(actions)`. Rewards/timeouts are stored through `process_env_step`, while done flags and episode stats are tracked for logging.""",
            text="""
        with self.accelerator.autocast():
            actor_out: TensorDict = self.actor(
                obs_td,
                actions=None,
                mode=actor_mode,
                update_obs_norm=update_obs_norm,
            )
            critic_out: TensorDict | None = None
            if collect_transition:
                critic_out = self.critic(
                    obs_td, update_obs_norm=update_obs_norm
                )

        if collect_transition:
            self.transition_td = self._build_transition(
                obs_td,
                actor_out,
                critic_out,
            )

        actions = actor_out.get("actions")
        self._last_rollout_actions = actions
        obs_dict, rewards, dones, time_outs, infos = self.env.step(actions)""",
        ),
        Highlight(
            file=motion_tracking_env,
            content="""
The HoloMotion environment step is a thin but important wrapper over IsaacLab: it delegates actor actions to `ManagerBasedRLEnv.step`, merges `terminated` and `time_outs` into PPO-style `dones`, updates completion/robot metrics, and returns observations, rewards, dones, timeouts, and infos to the trainer.""",
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
            file=algo_base,
            content="""
Rollout collection is bounded by `num_steps_per_env` from config. The trainer temporarily puts actor/critic in eval mode, disables gradients, resets rollout state, performs exactly that many environment steps, then computes returns from the final observation.""",
            text="""
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
Transition storage is explicit: rewards are cloned, timeout bootstrap rewards are added from transition values, rewards and dones are written into the transition tensorclass, the transition is appended to rollout storage, and the motion-command post-step hook can update curriculum accumulators.""",
            text="""
        raw_rewards = rewards.clone().view(-1, 1)
        rewards = raw_rewards.clone()
        dones = dones.view(-1, 1)

        # Bootstrapping on time outs
        rewards += self.gamma * (
            self.transition_td.values * time_outs[:, None]
        )
        self.transition_td.rewards = rewards
        self.transition_td.dones = dones.to(dtype=torch.bool)

        self.storage.add(self.transition_td)
        self._post_env_step_hook(raw_rewards, dones, time_outs, infos)

        self.transition_td = None""",
        ),
        Highlight(
            file=algo_base,
            content="""
The outer training loop proves the phase order for every iteration: reset/wrap initial observations, ensure storage, set train mode, synchronize workers, collect rollout, call `update()`, log losses/performance, checkpoint on interval, run iteration hooks, clear episode infos, and synchronize again.""",
            text="""
    def learn(self):
        \"\"\"Main learning loop with runner logic shared across on-policy algorithms.\"\"\"
        obs_dict = self.env.reset_all()[0]
        obs_td = self._wrap_obs_dict(obs_dict)
        self._ensure_storage(obs_td)
        self.train_mode()

        start_it = self.current_learning_iteration
        total_it = start_it + int(self.num_learning_iterations)
        self.total_learning_iterations = total_it

        self.accelerator.wait_for_everyone()
        if self.is_main_process:
            logger.info(
                f"Starting training for {self.num_learning_iterations} iterations "
                f"from iteration {self.current_learning_iteration}"
            )""",
        ),
        Highlight(
            file=algo_base,
            content="""
Within that loop, model updates happen only after rollout collection. The `loss_dict = self.update()` call is the trainer boundary between environment interaction and actor/critic optimization, followed by timing, logging, checkpointing, hooks, and distributed barrier synchronization.""",
            text="""
        for it in range(start_it, total_it):
            self.current_learning_iteration = it
            start = time.time()
            obs_td = self.rollout_policy(obs_td)

            stop = time.time()
            collection_time = stop - start
            start = stop

            loss_dict = self.update()

            stop = time.time()
            learn_time = stop - start""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
PPOTF update computes fresh actor log-probability/statistics and critic values for each sequence batch, forms PPO ratio/clipped surrogate and value losses, applies active auxiliary/router/entropy/KL terms, then backpropagates actor and critic losses separately through Accelerate.""",
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
