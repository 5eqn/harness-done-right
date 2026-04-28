from project_analysis import File, Highlight, ProofFromCode


train_config = File(
    path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_tf-moe.yaml"
)
train_base_config = File(
    path="HoloMotion/holomotion/config/training/train_base.yaml"
)
env_config = File(path="HoloMotion/holomotion/config/env/motion_tracking.yaml")
algo_config = File(path="HoloMotion/holomotion/config/algo/ppo_tf.yaml")
ppo_config = File(path="HoloMotion/holomotion/config/algo/ppo.yaml")
train_script = File(path="HoloMotion/holomotion/src/training/train.py")
config_utils = File(path="HoloMotion/holomotion/src/utils/config.py")
algo_base = File(path="HoloMotion/holomotion/src/algo/algo_base.py")
algo_utils = File(path="HoloMotion/holomotion/src/algo/algo_utils.py")
ppo = File(path="HoloMotion/holomotion/src/algo/ppo.py")
ppo_tf = File(path="HoloMotion/holomotion/src/algo/ppo_tf.py")
motion_tracking_env = File(path="HoloMotion/holomotion/src/env/motion_tracking.py")
motion_command = File(
    path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_motion_tracking_command.py"
)


logs_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=train_config,
            content="""
This proof is scoped to the actual HoloMotion Unitree G1 29-DoF TF-MoE motion-tracking training stack. The train config composes the PPOTF algorithm, motion-tracking environment, motion-tracking observations/rewards, and the TF-MoE module, so the logging evidence follows the train path used by this report rather than an unrelated example.""",
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
            file=train_base_config,
            content="""
The run directory is configured as `logs/<project_name>/<timestamp>-<experiment_name>`, with Hydra save/output subdirectories derived under the same experiment directory. This gives every training run a deterministic local logging root once Hydra resolves the timestamp and experiment name.""",
            text="""
timestamp: ${now:%Y%m%d_%H%M%S}
base_dir: logs
experiment_dir: ${base_dir}/${project_name}/${timestamp}-${experiment_name}
save_dir: ${experiment_dir}/.hydra
output_dir: ${experiment_dir}/output
experiment_save_dir: ???""",
        ),
        Highlight(
            file=config_utils,
            content="""
`compile_config_directories` materializes that directory, stores it back as `config.experiment_save_dir`, and writes the unresolved config YAML only on the main process. The run therefore has a real filesystem target and one authoritative config snapshot, not one racing write per rank.""",
            text="""
    experiment_save_dir = Path(config.experiment_dir)
    experiment_save_dir.mkdir(exist_ok=True, parents=True)
    config.experiment_save_dir = str(experiment_save_dir)
    if hasattr(config, "env"):
        config.env.config.save_rendering_dir = str(
            Path(config.experiment_dir) / "renderings_training"
        )
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    if config.main_process:
        logger.info(f"Saving config file to {experiment_save_dir}")
        with open(experiment_save_dir / "config.yaml", "w") as file:
            OmegaConf.save(unresolved_conf, file)""",
        ),
        Highlight(
            file=train_script,
            content="""
Distributed launches are normalized before the algorithm is built: rank 0's experiment directory is broadcast to all ranks after Accelerate initializes the process group. This prevents each process from logging checkpoints, TensorBoard state, and run logs into different timestamp directories.""",
            text="""
        path_list = (
            [config.experiment_save_dir]
            if _accelerator.is_main_process
            else [None]
        )
        dist.broadcast_object_list(path_list, src=0)
        config.experiment_save_dir = path_list[0]""",
        ),
        Highlight(
            file=train_script,
            content="""
The training entry point passes the compiled `experiment_save_dir` into the resolved algorithm class as `log_dir`, then calls `load()` and `learn()`. All logging evidence below is therefore connected to the concrete train invocation, not just helper classes.""",
            text="""
    log_dir = config.experiment_save_dir
    headless = config.headless
    algo_class = get_class(config.algo._target_)
    algo = algo_class(
        env_config=config.env,
        config=config.algo.config,
        log_dir=log_dir,
        headless=headless,
    )""",
        ),
        Highlight(
            file=algo_base,
            content="""
HoloMotion configures the present scalar backend as TensorBoard through Accelerate: the accelerator is constructed with `log_with="tensorboard"` and both `project_dir` and `logging_dir` set to the run log directory. There is no W&B branch in this training logger path; the available remote/scalar integration here is TensorBoard via Accelerate trackers.""",
            text="""
        accelerator_kwargs["log_with"] = "tensorboard"
        project_config = ProjectConfiguration(
            project_dir=self.log_dir,
            logging_dir=self.log_dir,
        )
        accelerator_kwargs["project_config"] = project_config
        self.accelerator = Accelerator(**accelerator_kwargs)""",
        ),
        Highlight(
            file=algo_base,
            content="""
Accelerate tracker initialization records precision and compile settings, giving TensorBoard run metadata in addition to scalar values. The code does this before training starts, so later `accelerator.log(...)` calls have an initialized tracker.""",
            text="""
        self.accelerator.init_trackers(
            project_name="holomotion",
            config={
                "precision": mixed_precision if mixed_precision else "fp32",
                "dynamo_backend": dynamo_backend if dynamo_backend else "none",
                "dynamo_dynamic": bool(self.config.get("dynamo_dynamic", True))
                if dynamo_backend
                else False,
            },
        )""",
        ),
        Highlight(
            file=algo_base,
            content="""
Loguru is present and deliberately separated by rank: every process gets a `run_rank_XXXX.log` file, while only the main process also gets stdout and aggregate `run.log`. This is the correct distributed behavior for human-readable logs because worker ranks keep diagnostics without duplicating console output.""",
            text="""
        if self.log_dir:
            rank_log_file_name = (
                "offline_eval_rank" if self.is_offline_eval else "run_rank"
            )
            logger.add(
                os.path.join(
                    self.log_dir,
                    f"{rank_log_file_name}_{int(self.accelerator.process_index):04d}.log",
                ),
                level=log_level,
                colorize=False,
            )
        if self.is_main_process:
            logger.add(
                sys.stdout,
                level=log_level,
                colorize=True,
            )
            log_file_name = (
                "offline_eval.log" if self.is_offline_eval else "run.log"
            )
            logger.add(
                os.path.join(self.log_dir, log_file_name),
                level=log_level,
                colorize=False,
            )""",
        ),
        Highlight(
            file=algo_base,
            content="""
The learning loop only calls `_log_iteration` on the main process and only at the configured interval. This keeps scalar logging, console tables, and episode aggregation single-writer in distributed training.""",
            text="""
            if self.is_main_process and it % self.log_interval == 0:
                self._log_iteration(
                    it=it,
                    loss_dict=loss_dict,
                    collection_time=collection_time,
                    learn_time=learn_time,
                )""",
        ),
        Highlight(
            file=ppo_config,
            content="""
For the scoped PPOTF config, iteration logging runs every five training iterations and checkpoints every 500 iterations. The proof's main-process logging gate therefore uses explicit user-configurable cadence, not an accidental per-step print.""",
            text="""
    num_learning_iterations: 10001
    log_interval: 5
    save_interval: 500
    export_policy: true
    onnx_name_suffix: null""",
        ),
        Highlight(
            file=algo_base,
            content="""
Episode reward and length are tracked from real environment rewards and done masks. The code accumulates per-env reward/length every step, pushes completed episodes into bounded buffers, and resets only the completed env slots, which is the expected accounting for vectorized RL environments.""",
            text="""
        self.cur_reward_sum += rewards
        self.cur_episode_length += 1

        done_ids = (dones > 0).nonzero(as_tuple=False)
        self.rewbuffer.extend(
            self.cur_reward_sum[done_ids][:, 0].cpu().numpy().tolist()
        )
        self.lenbuffer.extend(
            self.cur_episode_length[done_ids][:, 0].cpu().numpy().tolist()
        )
        self.cur_reward_sum[done_ids] = 0
        self.cur_episode_length[done_ids] = 0""",
        ),
        Highlight(
            file=algo_base,
            content="""
Environment-provided episode log dictionaries are collected only on the main process, and every value is converted to CPU scalar tensors before being buffered. This is the handoff from IsaacLab/HoloMotion `infos["log"]` entries into trainer-level episode metric aggregation.""",
            text="""
        log_info = infos.get("log")
        if self.is_main_process and isinstance(log_info, dict):
            cpu_log_info: Dict[str, torch.Tensor] = {}
            for key, value in log_info.items():
                cpu_value = self._log_value_to_cpu_tensor(value)
                if cpu_value is not None and cpu_value.numel() > 0:
                    cpu_log_info[key] = cpu_value
            if len(cpu_log_info) > 0:
                self.ep_infos.append(cpu_log_info)""",
        ),
        Highlight(
            file=algo_base,
            content="""
Episode extras are averaged across all buffered scalar elements, and slash-qualified names are preserved exactly while plain names are placed under `Episode/`. That means HoloMotion metrics like `Metrics/Robot/...` and `Metrics/ref_motion/...` keep their namespaces, while generic episode terms still get a clear TensorBoard prefix.""",
            text="""
        for key, total in metric_sums.items():
            count = metric_counts.get(key, 0)
            if count <= 0:
                continue
            mean_value = total / float(count)
            metric_key = key if "/" in key else f"Episode/{key}"
            metrics[metric_key] = mean_value""",
        ),
        Highlight(
            file=algo_base,
            content="""
Iteration logging always includes iteration counters, total iterations, loss scalars, and performance timing/FPS. It then adds mean episode reward/length when completed episodes exist, and finally merges episode extras and algorithm-specific metrics into one scalar dictionary.""",
            text="""
        iteration_metrics: Dict[str, Any] = {
            "0-Train/iteration": int(it),
            "0-Train/iterations_total": total_learning_iterations,
        }

        for key, value in loss_dict.items():
            if value is None:
                continue
            scalar = float(value)
            iteration_metrics[f"Loss/{key}"] = scalar

        iteration_metrics.update(
            {
                "1-Perf/total_fps": float(fps),
                "1-Perf/collection_time": float(collection_time),
                "1-Perf/learning_time": float(learn_time),
            }
        )""",
        ),
        Highlight(
            file=algo_base,
            content="""
Mean reward and mean episode length are logged from the completed-episode buffers, so the standard RL progress metrics are not synthetic: they are derived from `rewbuffer` and `lenbuffer` populated by environment step outcomes.""",
            text="""
        elif len(self.rewbuffer) > 0:
            mean_reward = float(statistics.mean(self.rewbuffer))
            mean_episode_length = float(statistics.mean(self.lenbuffer))
            iteration_metrics["0-Train/mean_reward"] = mean_reward
            iteration_metrics["0-Train/mean_episode_length"] = (
                mean_episode_length
            )""",
        ),
        Highlight(
            file=algo_base,
            content="""
The final iteration metric set explicitly includes both averaged episode extras and additional algorithm metrics before it is handed to `AlgoLogger`. This is where reward/episode, robot, command, cache, optimization, and performance scalars converge.""",
            text="""
        iteration_metrics.update(self._aggregate_episode_log_metrics())
        iteration_metrics.update(self._get_additional_log_metrics())

        self.algo_logger.log_iteration(
            step=it,
            total_learning_iterations=total_learning_iterations,
            metrics=iteration_metrics,
        )""",
        ),
        Highlight(
            file=algo_utils,
            content="""
`AlgoLogger` filters the metric map down to true scalars before calling `accelerator.log(..., step=...)`; non-scalar debug objects can still appear in console formatting, but they cannot poison TensorBoard scalar streams.""",
            text="""
        tensorboard_metrics: dict[str, float] = {}
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if value is None or not self._is_scalar_metric(value):
                continue
            tensorboard_metrics[key] = self._to_scalar(value)

        if len(tensorboard_metrics) > 0:
            self.accelerator.log(tensorboard_metrics, step=int(step))""",
        ),
        Highlight(
            file=algo_utils,
            content="""
The same logger builds a sorted tabular console record and appends the absolute logging directory, so the loguru output mirrors the scalar stream and points reviewers to the exact local run folder.""",
            text="""
        console_log = self._build_console_log(
            step=step,
            total_learning_iterations=total_learning_iterations,
            console_metrics=console_metrics,
        )
        logger.info(console_log)""",
        ),
        Highlight(
            file=motion_tracking_env,
            content="""
Every environment step passes through HoloMotion's wrapper after IsaacLab returns observations, rewards, terminations, timeouts, and infos. The wrapper updates completion-rate command metrics and robot low-level metrics before the trainer sees `infos`, so those fields are available to `_track_episode_stats` in the same rollout step.""",
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
            file=motion_tracking_env,
            content="""
Robot metrics are grounded in concrete simulation tensors: action change per second, joint acceleration, applied torque norm, mechanical energy proxy, and effort-normalized torque-rate. Each is stored as a mean scalar before logging.""",
            text="""
        dof_acc_norm = torch.norm(dof_acc, dim=-1)  # [B]
        dof_torque_norm = torch.norm(dof_torque, dim=-1)  # [B]
        energy = torch.sum(
            torch.abs(dof_vel) * torch.abs(dof_torque), dim=-1
        )  # [B]

        self.metrics["Robot/Action_Rate"] = action_rate.mean()
        self.metrics["Robot/DOF_Acc"] = dof_acc_norm.mean()
        self.metrics["Robot/DOF_Torque"] = dof_torque_norm.mean()
        self.metrics["Robot/Energy"] = energy.mean()
        self.metrics["Robot/Normed_Torque_Rate"] = normed_torque_rate.mean()""",
        ),
        Highlight(
            file=motion_tracking_env,
            content="""
Those robot scalars are written into `infos["log"]` under stable `Metrics/Robot/...` keys. Because the trainer preserves slash-qualified episode keys, these become TensorBoard/loguru scalar names without losing their robot namespace.""",
            text="""
        infos["log"]["Metrics/Robot/Action_Rate"] = self.metrics[
            "Robot/Action_Rate"
        ]
        infos["log"]["Metrics/Robot/DOF_Acc"] = self.metrics["Robot/DOF_Acc"]
        infos["log"]["Metrics/Robot/DOF_Torque"] = self.metrics[
            "Robot/DOF_Torque"
        ]
        infos["log"]["Metrics/Robot/Energy"] = self.metrics["Robot/Energy"]
        infos["log"]["Metrics/Robot/Normed_Torque_Rate"] = self.metrics[
            "Robot/Normed_Torque_Rate"
        ]""",
        ),
        Highlight(
            file=motion_tracking_env,
            content="""
The task completion metric is command-scoped: done environments are counted as successful only when they time out without termination, and the rolling rate is injected directly as `Metrics/ref_motion/Task/Completion_Rate` on the same device as the environment.""",
            text="""
        denom = sum(self._completion_total_queue)
        completion_rate = (
            float(sum(self._completion_success_queue)) / float(denom)
            if denom > 0
            else 0.0
        )
        if ("log" not in infos) or (not isinstance(infos["log"], dict)):
            infos["log"] = {}
        infos["log"]["Metrics/ref_motion/Task/Completion_Rate"] = torch.tensor(
            completion_rate, device=self.device, dtype=torch.float32
        )""",
        ),
        Highlight(
            file=env_config,
            content="""
The motion-tracking environment names its command `ref_motion` and wires it to `MotionCommandCfg`, so command metrics and command-specific log keys in this proof are scoped to the reference-motion command used for imitation training.""",
            text="""
    commands:
      ref_motion:
        type: MotionCommandCfg
        params:
          command_obs_name: bydmmc_ref_motion
          motion_lib_cfg: ${robot.motion}""",
        ),
        Highlight(
            file=algo_base,
            content="""
During algorithm setup the first configured command is resolved from IsaacLab's command manager; for `ref_motion`, HoloMotion also passes process id/world size and the log directory into the command term. This grounds command metrics and optional cache dumps in the same distributed context and run directory as the trainer.""",
            text="""
        self.command_name = list(self.env.config.commands.keys())[0]
        self.command_term = self.env._env.command_manager.get_term(
            self.command_name
        )
        if self.command_name == "ref_motion":
            self.command_term.set_runtime_distributed_context(
                process_id=int(self.accelerator.process_index),
                num_processes=int(self.accelerator.num_processes),
            )
            self.command_term.setup_dumping_dir(self.log_dir)""",
        ),
        Highlight(
            file=motion_command,
            content="""
The reference-motion command computes concrete tracking-error metrics each command update: MPJPE for whole body, arms, waist, and legs from joint position error against the immediate-next reference frame. These are not labels only; they are derived from current robot joint positions and reference motion joint positions.""",
            text="""
        # Update metric values
        self.metrics["Task/MPJPE_WholeBody"][:] = mpjpe_wholebody
        self.metrics["Task/MPJPE_Arms"][:] = mpjpe_arms
        self.metrics["Task/MPJPE_Waist"][:] = mpjpe_waist
        self.metrics["Task/MPJPE_Legs"][:] = mpjpe_legs""",
        ),
        Highlight(
            file=motion_command,
            content="""
The same command computes MPKPE metrics from current body world positions versus reference body positions, again split into whole body, arms, waist, and legs. Together with the `ref_motion` command name, these are the concrete command-side tracking metrics exposed by the motion command term.""",
            text="""
        # Update metric values
        self.metrics["Task/MPKPE_WholeBody"][:] = mpkpe_wholebody
        self.metrics["Task/MPKPE_Arms"][:] = mpkpe_arms
        self.metrics["Task/MPKPE_Waist"][:] = mpkpe_waist
        self.metrics["Task/MPKPE_Legs"][:] = mpkpe_legs""",
        ),
        Highlight(
            file=ppo,
            content="""
PPO adds optimization and cache/runtime metrics to the same scalar stream: actor/critic learning rates, effective entropy coefficient, last update statistics, and mean action-noise standard deviation. These fill out the standard training diagnostics alongside reward and episode metrics.""",
            text="""
        if "actor_learning_rate" in self.__dict__:
            iteration_metrics["0-Train/actor_learning_rate"] = float(
                self.actor_learning_rate
            )

        if "critic_learning_rate" in self.__dict__:
            iteration_metrics["0-Train/critic_learning_rate"] = float(
                self.critic_learning_rate
            )

        if "initial_entropy_coef" in self.__dict__:
            iteration_metrics["0-Train/entropy_coef_effective"] = float(
                self._get_effective_entropy_coef()
            )""",
        ),
        Highlight(
            file=ppo,
            content="""
For the reference-motion command, PPO also logs cache state under `1-Perf/Cache/...`, which makes command dataset/cache progression visible in the same TensorBoard/loguru iteration record as policy training.""",
            text="""
        if self.command_name != "ref_motion":
            return iteration_metrics

        motion_cmd = self.env._env.command_manager.get_term("ref_motion")
        cache = motion_cmd._motion_cache
        iteration_metrics["1-Perf/Cache/swap_index"] = float(cache.swap_index)""",
        ),
        Highlight(
            file=ppo_tf,
            content="""
The PPOTF update returns a dense scalar loss dictionary covering value loss, surrogate policy loss, entropy, KL signals, auxiliary prediction losses, and MoE/router diagnostics. Base logging prefixes these keys with `Loss/`, so the TF-MoE-specific training objective is visible in scalar logs.""",
            text="""
        loss_out = {
            "value_function": mean_value_loss,
            "critic_explained_variance": critic_explained_variance,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "kl_token": mean_kl_token,
            "kl_loss": mean_kl_loss,
            "kl_analytic": mean_kl_analytic,
            "aux_base_lin_vel_nll": mean_aux_base_lin_vel_nll,
            "aux_root_height_nll": mean_aux_root_height_nll,
            "aux_base_lin_vel_std": mean_aux_base_lin_vel_std,
            "aux_root_height_std": mean_aux_root_height_std,""",
        ),
        Highlight(
            file=ppo_tf,
            content="""
Distributed PPOTF loss metrics are reduced across ranks before returning to the main-process logger. That means the reported `Loss/...` scalars are rank-averaged training diagnostics instead of only rank-0 local minibatch values.""",
            text="""
        if self.is_distributed:
            reduced_out = {}
            for k, v in loss_out.items():
                if v is None:
                    reduced_out[k] = None
                    continue
                t = torch.tensor(v, device=self.device, dtype=torch.float32)
                reduced_t = self.accelerator.reduce(t, reduction="mean")
                reduced_out[k] = float(reduced_t.item())
            loss_out = reduced_out""",
        ),
    ]
)
