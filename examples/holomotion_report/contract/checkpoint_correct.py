from project_analysis import File, Highlight, ProofFromCode


train_config = File(
    path="HoloMotion/holomotion/config/training/motion_tracking/train_g1_29dof_motion_tracking_tf-moe.yaml"
)
train_base_config = File(path="HoloMotion/holomotion/config/training/train_base.yaml")
ppo_config = File(path="HoloMotion/holomotion/config/algo/ppo.yaml")
ppo_tf_config = File(path="HoloMotion/holomotion/config/algo/ppo_tf.yaml")
train_py = File(path="HoloMotion/holomotion/src/training/train.py")
eval_motion_tracking_single = File(
    path="HoloMotion/holomotion/src/evaluation/eval_motion_tracking_single.py"
)
algo_base = File(path="HoloMotion/holomotion/src/algo/algo_base.py")
ppo_py = File(path="HoloMotion/holomotion/src/algo/ppo.py")
ppo_tf_py = File(path="HoloMotion/holomotion/src/algo/ppo_tf.py")
agent_modules = File(path="HoloMotion/holomotion/src/modules/agent_modules.py")
onnx_export = File(path="HoloMotion/holomotion/src/utils/onnx_export.py")


checkpoint_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=train_config,
            content="""
Scope anchor: this proof follows the G1 29-DoF motion-tracking TF-MoE training target. The scoped config composes the base training config, the PPOTF algorithm config, the Unitree G1 robot config, the motion-tracking env and observation/reward groups, rough terrain, and the `motion_tracking_tf-moe` module group.""",
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
The training base owns the top-level checkpoint path: it defaults to `null`, so fresh training is the default and resume/evaluate behavior is enabled by explicitly setting `checkpoint` in the Hydra config or CLI.""",
            text="""
motion_h5_path: ???
checkpoint: null

num_processes: ???
main_process: ???
process_id: ???""",
        ),
        Highlight(
            file=train_py,
            content="""
Distributed path handling is explicit before the algorithm is built: when more than one process is requested, rank 0's `experiment_save_dir` is broadcast to every rank. This prevents per-rank Hydra timestamps from producing different checkpoint directories.""",
            text="""
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
The training runner resolves the algorithm class from config, constructs it with `config.algo.config`, calls `algo.load(config.checkpoint)` before learning, and then enters `algo.learn()`. Because the scoped config selects PPOTF, resume loading is on the PPOTF/PPO code path below.""",
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
            file=eval_motion_tracking_single,
            content="""
Evaluation/export has the reciprocal checkpoint path handling: it first looks for `config.yaml` next to the checkpoint, falls back one directory up for exported/checkpoint layouts, then writes the chosen checkpoint path into both `train_config.checkpoint` and `train_config.algo.config.checkpoint` before merging eval overrides.""",
            text="""
    checkpoint = Path(checkpoint_path)
    config_path = checkpoint.parent / "config.yaml"

    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
        if not config_path.exists():
            logger.warning(
                f"Training config not found at {config_path}, using evaluation config"
            )
            return eval_config

    logger.info(f"Loading training config from {config_path}")
    with open(config_path) as file:
        train_config = OmegaConf.load(file)

    # Apply eval_overrides from training config if they exist
    if train_config.get("eval_overrides") is not None:
        train_config = OmegaConf.merge(
            train_config, train_config.eval_overrides
        )

    # Set checkpoint path
    train_config.checkpoint = checkpoint_path
    train_config.algo.config.checkpoint = checkpoint_path""",
        ),
        Highlight(
            file=ppo_tf_config,
            content="""
The scoped algorithm is PPOTF, with trainable log-sigma (`noise_std_type: log`, `fix_sigma: false`) and clamped sigma bounds. Those settings make actor sigma part of the actor module state, not a derived runtime-only value.""",
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
    max_sigma: 1.2""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
PPOTF subclasses `PPO`, so unless it overrides a method, the checkpoint save/load implementation used for the scoped TF-MoE run is the `PPO.save` and `PPO.load` code below.""",
            text="""
from holomotion.src.algo.ppo import PPO
from holomotion.src.modules.agent_modules import (
    PPOCondTFActor,
    PPOCritic,
    PPOTFActor,
    PPOTFRefRouterActor,
    PPOTFRefRouterSeqActor,
    PPOTFRefRouterV3Actor,
    TensorDictAssembler,
)
from holomotion.src.modules.network_modules import GroupedMoEBlock
from loguru import logger
from omegaconf import OmegaConf
from tabulate import tabulate
from tensordict import TensorDict


class PPOTF(PPO):
    \"\"\"Transformer-policy PPO with TensorDict rollout and sequence update.\"\"\"""",
        ),
        Highlight(
            file=algo_base,
            content="""
The periodic checkpoint cadence is grounded in the algorithm config: `_setup_configs` copies `config.save_interval` into `self.save_interval` along with log interval, rollout length, and learning iterations.""",
            text="""
        self.save_interval = self.config.save_interval
        self.log_interval = self.config.log_interval
        self.num_steps_per_env = self.config.num_steps_per_env
        self.num_learning_iterations = self.config.num_learning_iterations
        self.total_learning_iterations = int(self.num_learning_iterations)""",
        ),
        Highlight(
            file=ppo_config,
            content="""
The inherited PPO base config enables periodic checkpointing every 500 learning iterations and enables policy export on every save.""",
            text="""
    num_learning_iterations: 10001
    log_interval: 5
    save_interval: 500
    export_policy: true
    onnx_name_suffix: null
    use_kv_cache: true
    eval_interval: null""",
        ),
        Highlight(
            file=algo_base,
            content="""
The learning loop saves periodically only on the main process, naming each checkpoint `model_{current_learning_iteration}.pt`, then saves a final checkpoint after the loop. Every iteration ends with `wait_for_everyone()`, keeping distributed ranks synchronized around rank-0 saves.""",
            text="""
            if self.is_main_process and it % self.save_interval == 0:
                self.save(
                    os.path.join(
                        self.log_dir,
                        f"model_{self.current_learning_iteration}.pt",
                    )
                )
                self._release_cuda_cache()

            self._post_iteration_hook(it)
            self.ep_infos.clear()
            self.accelerator.wait_for_everyone()

        final_checkpoint_path = os.path.join(
            self.log_dir, f"model_{self.current_learning_iteration}.pt"
        )
        if self.is_main_process:
            self.save(final_checkpoint_path)
            self._release_cuda_cache()""",
        ),
        Highlight(
            file=ppo_py,
            content="""
`PPO.save` has its own main-process guard, so even direct calls from export/eval helpers cannot write duplicate checkpoints from non-main ranks. It derives the Accelerate model directory by stripping `.pt` from the checkpoint path, creates the directory, and saves actor and critic model states under sibling `actor` and `critic` subdirectories.""",
            text="""
    def save(self, path, infos=None):
        if not self.is_main_process:
            return

        logger.info(f"Saving checkpoint to {path}")
        base_path = path.replace(".pt", "")
        os.makedirs(
            os.path.dirname(base_path) if os.path.dirname(base_path) else ".",
            exist_ok=True,
        )

        self.accelerator.save_model(
            self.actor, os.path.join(base_path, "actor")
        )
        self.accelerator.save_model(
            self.critic, os.path.join(base_path, "critic")
        )""",
        ),
        Highlight(
            file=ppo_py,
            content="""
The `.pt` payload contains optimizer state for both actor and critic, the learning iteration, caller-provided infos, and any subclass extra checkpoint state. `_checkpoint_state_to_cpu` is applied before `torch.save`, so nested tensors in the metadata are detached and moved to CPU for portable loading.""",
            text="""
        custom_state = {
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        custom_state.update(self._extra_checkpoint_state())
        torch.save(_checkpoint_state_to_cpu(custom_state), path)""",
        ),
        Highlight(
            file=ppo_py,
            content="""
The CPU conversion used by save is recursive across tensors, dictionaries, lists, and tuples, so optimizer states and any future extra checkpoint tensors are serialized independently of the training device.""",
            text="""
def _checkpoint_state_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _checkpoint_state_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_checkpoint_state_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_state_to_cpu(v) for v in value)
    return value""",
        ),
        Highlight(
            file=agent_modules,
            content="""
The actor module state saved by `accelerator.save_model(self.actor, ...)` includes the actor observation normalizer when enabled, because `self.obs_normalizer` is a submodule. In the scoped config it is enabled, so actor normalization statistics are checkpointed with the actor weights.""",
            text="""
        self.out_keys = [
            "actions",
            "actions_log_prob",
            "mu",
            "sigma",
            "entropy",
        ]
        if self.obs_norm_enabled and self.assembler is not None:
            self.obs_normalizer = EmpiricalNormalization(
                shape=self.assembler.output_dim,
                eps=self.obs_norm_eps,
                update_method=self.obs_norm_update_method,
                ema_momentum=self.obs_norm_ema_momentum,
            )
        else:
            self.obs_normalizer = nn.Identity()""",
        ),
        Highlight(
            file=agent_modules,
            content="""
Actor sigma is also actor module state. For the scoped `noise_std_type: log` config, `self.log_std` is an `nn.Parameter`; because `fix_sigma` is false, it remains trainable and is saved/restored through the actor model checkpoint.""",
            text="""
        # Action noise parameters (kept outside nets so optimizer updates them)
        if self.noise_std_type == "log":
            logger.info("Using log-std parameterization for action noise")
            self.log_std = nn.Parameter(
                torch.log(torch.ones(num_actions) * init_noise_std)
            )
            if self.fix_sigma:
                self.log_std.requires_grad = False
        else:  # scalar (default)
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            if self.fix_sigma:
                self.std.requires_grad = False""",
        ),
        Highlight(
            file=agent_modules,
            content="""
The critic has the same normalizer persistence shape: when critic obs normalization is enabled, the critic wrapper attaches an `EmpiricalNormalization` submodule before constructing the critic network, so critic normalization statistics are saved in the critic model directory.""",
            text="""
        self.in_keys = critic_in_keys
        self.out_keys = ["values"]

        if self.obs_norm_enabled and self.assembler is not None:
            self.obs_normalizer = EmpiricalNormalization(
                shape=self.assembler.output_dim,
                eps=self.obs_norm_eps,
                update_method=self.obs_norm_update_method,
                ema_momentum=self.obs_norm_ema_momentum,
            )
        else:
            self.obs_normalizer = nn.Identity()""",
        ),
        Highlight(
            file=ppo_tf_py,
            content="""
PPOTF constructs actor and critic modules, then creates actor and critic optimizers from their parameters and wraps all four objects with `accelerator.prepare`. Therefore the objects saved by `save_model` and the optimizer states in the `.pt` payload are the prepared training objects actually used during PPO updates.""",
            text="""
        self.critic_optimizer = optimizer_class(
            self.critic.parameters(),
            lr=self.critic_learning_rate,
            betas=(self.critic_beta1, self.critic_beta2),
            **optimizer_kwargs,
        )

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
        )""",
        ),
        Highlight(
            file=ppo_py,
            content="""
Loading mirrors saving: a `None` checkpoint is a no-op, otherwise actor and critic model directories are resolved from the `.pt` path, loaded strictly, and the `.pt` metadata is read onto the algorithm device. In normal training resume mode, both actor and critic optimizer states are restored; offline evaluation intentionally skips optimizer restore.""",
            text="""
    def load(self, ckpt_path):
        if ckpt_path is None:
            return None
        if self.is_main_process:
            logger.info(f"Loading checkpoint from {ckpt_path}")

        actor_model_path = self._resolve_model_file_path(ckpt_path, "actor")
        critic_model_path = self._resolve_model_file_path(ckpt_path, "critic")
        self._load_accelerate_model(self.actor, actor_model_path, strict=True)
        self._load_accelerate_model(
            self.critic, critic_model_path, strict=True
        )

        loaded_dict = torch.load(ckpt_path, map_location=self.device)
        if not getattr(self, "is_offline_eval", False):
            self._restore_optimizer_state(
                self.actor_optimizer,
                loaded_dict["actor_optimizer_state_dict"],
                optimizer_name="actor",
            )
            self._restore_optimizer_state(
                self.critic_optimizer,
                loaded_dict["critic_optimizer_state_dict"],
                optimizer_name="critic",
            )""",
        ),
        Highlight(
            file=ppo_py,
            content="""
After optimizer restoration, load restores the learning iteration, optionally reapplies configured sigma override, delegates subclass-specific extra state loading, and returns checkpoint infos. That proves the resume point is not merely weights-only.""",
            text="""
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        self._maybe_override_loaded_actor_sigma()
        self._load_extra_checkpoint_state(loaded_dict)
        return loaded_dict.get("infos", None)""",
        ),
        Highlight(
            file=algo_base,
            content="""
The actor/critic model path resolver proves the checkpoint directory contract: `model_500.pt` corresponds to `model_500/actor` and `model_500/critic`, and missing directories fail loudly before partial loading can proceed.""",
            text="""
    def _resolve_model_file_path(self, ckpt_path: str, model_name: str) -> str:
        \"\"\"Resolve per-model Accelerate checkpoint directory from *.pt path.\"\"\"
        base_path = ckpt_path.replace(".pt", "")
        model_path = os.path.join(base_path, model_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Missing accelerate checkpoint directory for {model_name}: "
                f"{model_path}"
            )
        return model_path""",
        ),
        Highlight(
            file=algo_base,
            content="""
Accelerate model loading supports both directory and single-file variants: it prefers `model.safetensors`, falls back to `pytorch_model.bin`, or delegates to `load_checkpoint_in_model` for full Accelerate directories, then calls `_load_model_state` for strict state-dict loading.""",
            text="""
    def _load_accelerate_model(
        self, model, model_path: str, *, strict: bool = True
    ) -> None:
        \"\"\"Load model params from Accelerate checkpoint directory/file.\"\"\"
        checkpoint_path = model_path
        if os.path.isdir(model_path):
            safetensors_path = os.path.join(model_path, "model.safetensors")
            pytorch_bin_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.isfile(safetensors_path):
                checkpoint_path = safetensors_path
            elif os.path.isfile(pytorch_bin_path):
                checkpoint_path = pytorch_bin_path
            else:
                target = self.accelerator.unwrap_model(model)
                load_checkpoint_in_model(target, model_path, strict=strict)
                return
        state_dict = load_state_dict(checkpoint_path)
        self._load_model_state(model, state_dict, strict=strict)""",
        ),
        Highlight(
            file=ppo_py,
            content="""
The sigma override branch is deliberately after checkpoint load. If enabled, it replaces the loaded actor sigma parameter and logs on the main process only, which makes sigma resume behavior explicit rather than accidental.""",
            text="""
        override_sigma(sigma_override)
        if self.is_main_process:
            logger.info(
                "Reapplied sigma override after checkpoint load: {}",
                sigma_override,
            )""",
        ),
        Highlight(
            file=ppo_py,
            content="""
ONNX export is coupled to saving only when config enables it. Since the scoped inherited PPO config sets `export_policy: true`, each successful main-process checkpoint save also calls the common ONNX exporter with the saved checkpoint path, optional suffix, and KV-cache setting.""",
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
            file=onnx_export,
            content="""
The ONNX export path is deterministic from the checkpoint path: exported files go under `<checkpoint_parent>/exported`, use the checkpoint filename with `.pt` replaced by `.onnx`, and optional suffixes are sanitized before insertion.""",
            text="""
    checkpoint = Path(checkpoint_path)
    export_dir = checkpoint.parent / "exported"
    export_dir.mkdir(exist_ok=True)

    onnx_name = checkpoint.name.replace(".pt", ".onnx")
    if onnx_name_suffix is not None:
        suffix = re.sub(r"[\\s+]", "_", str(onnx_name_suffix))
        onnx_name = onnx_name.replace(".onnx", f"_{suffix}.onnx")
    onnx_path = export_dir / onnx_name""",
        ),
        Highlight(
            file=onnx_export,
            content="""
The exporter unwraps the prepared actor, strips a possible compiled `_orig_mod`, forwards `use_kv_cache` only if the actor export signature accepts it, attaches HoloMotion environment metadata, and restores actor/critic training modes in `finally`.""",
            text="""
    try:
        actor_for_export = algo.accelerator.unwrap_model(algo.actor)
        orig_mod = getattr(actor_for_export, "_orig_mod", None)
        if orig_mod is not None:
            actor_for_export = orig_mod

        export_signature = inspect.signature(actor_for_export.export_onnx)
        export_kwargs = {"onnx_path": onnx_path, "opset_version": 17}
        if "use_kv_cache" in export_signature.parameters:
            export_kwargs["use_kv_cache"] = bool(use_kv_cache)

        onnx_path_str = actor_for_export.export_onnx(**export_kwargs)
        attach_onnx_metadata_holomotion(algo.env._env, onnx_path=onnx_path_str)
        logger.info(
            f"Successfully exported minimal policy to: {onnx_path_str}"
        )
        return onnx_path_str
    finally:
        if actor_was_training is not None:
            algo.actor.train(actor_was_training)
        if critic_was_training is not None:
            algo.critic.train(critic_was_training)""",
        ),
    ]
)
