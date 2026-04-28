from enhance_guidance import AtomicLeap, File, Highlight


holomotion_metrics = File(path="HoloMotion/holomotion/src/evaluation/metrics.py")
holomotion_onnx_export = File(path="HoloMotion/holomotion/src/utils/onnx_export.py")
holomotion_gmr_to_holomotion = File(
    path="HoloMotion/holomotion/src/motion_retargeting/gmr_to_holomotion.py"
)
mjlab_evaluate = File(path="mjlab/src/mjlab/tasks/tracking/scripts/evaluate.py")
mjlab_base_runner = File(path="mjlab/src/mjlab/rl/runner.py")
mjlab_train = File(path="mjlab/src/mjlab/scripts/train.py")


other_leaps: list[AtomicLeap] = [
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_metrics,
            content=(
                "HoloMotion computes offline dataset metrics per clip, records "
                "motion-completion success, and persists clip-level CSV output."
            ),
            text="""
            max_body_err = float(
                np.nanmax(df_frames["frame_max_body_pos_err"].to_numpy())
            )
            success = 1.0 if max_body_err <= failure_pos_err_thresh_m else 0.0
            clip_meta_entry = {
                "motion_key": motion_key,
                "num_frames": num_frames_clip,
                "clip_length": clip_length,
                "success": success,
                "max_body_pos_err": max_body_err,
                "failure_threshold_m": float(failure_pos_err_thresh_m),
                **chatter_summary,
                **stability_summary,
            }""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_evaluate,
            content=(
                "MJLab evaluates a single run against the training motion "
                "artifact and reports only aggregate online episode metrics."
            ),
            text="""
  metrics = {
    "success_rate": success.float().mean().item(),
    "mpkpe": means[0].mean().item(),
    "r_mpkpe": means[1].mean().item(),
    "joint_vel_error": means[2].mean().item(),
    "ee_pos_error": means[3].mean().item(),
    "ee_ori_error": means[4].mean().item(),
  }""",
        ),
        change_direction=(
            "Add an offline held-out evaluation mode to MJLab tracking that can "
            "evaluate a checkpoint on an explicit unseen motion set, dump each "
            "rollout to NPZ, compute per-frame and per-clip metrics, and write "
            "both `per_clip_metrics.csv` and `whole_dataset_metrics.json`. Keep "
            "the current online aggregate path, but make it a thin convenience "
            "wrapper over the richer evaluator."
        ),
        change_reason=(
            "Generalized motion tracking cannot be validated by replaying only "
            "the training artifact and averaging one scalar per environment. "
            "Engineers need held-out clip completion, MPJPE-style body-link "
            "errors, stability/chatter metrics, and per-motion failure rows to "
            "see which unseen motions regress as model, command, and reward "
            "changes land."
        ),
        changed_code=r'''
@dataclass(frozen=True)
class EvaluateConfig:
  wandb_run_path: str
  wandb_checkpoint_name: str | None = None
  motion_file: str | None = None
  unseen_motion_file: str | None = None
  output_dir: str = "logs/eval/tracking"
  num_envs: int = 1024
  failure_pos_err_thresh_m: float = 0.25
  metric_calculation: Literal["per_clip", "per_frame"] = "per_clip"
  dump_rollouts: bool = True


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> dict[str, float]:
  motion_path = cfg.unseen_motion_file or cfg.motion_file or _motion_from_wandb(cfg)
  rollout_dir = Path(cfg.output_dir) / Path(motion_path).stem
  rollout_dir.mkdir(parents=True, exist_ok=True)

  rollouts = collect_tracking_rollouts(
    task_id=task_id,
    checkpoint=resolve_checkpoint(cfg),
    motion_file=motion_path,
    num_envs=cfg.num_envs,
    output_dir=rollout_dir if cfg.dump_rollouts else None,
  )

  per_clip = compute_tracking_metric_table(
    rollouts=rollouts,
    failure_pos_err_thresh_m=cfg.failure_pos_err_thresh_m,
  )
  per_clip.to_csv(rollout_dir / "per_clip_metrics.csv", index=False)

  dataset = summarize_tracking_metrics(per_clip, mode=cfg.metric_calculation)
  with open(rollout_dir / "whole_dataset_metrics.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)
  return dataset["mean"]
''',
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_onnx_export,
            content=(
                "HoloMotion delegates ONNX export to the actor implementation, "
                "so transformer/MoE actors can own their input signatures, "
                "KV-cache behavior, and export-specific outputs."
            ),
            text="""
        export_signature = inspect.signature(actor_for_export.export_onnx)
        export_kwargs = {"onnx_path": onnx_path, "opset_version": 17}
        if "use_kv_cache" in export_signature.parameters:
            export_kwargs["use_kv_cache"] = bool(use_kv_cache)

        onnx_path_str = actor_for_export.export_onnx(**export_kwargs)
        attach_onnx_metadata_holomotion(algo.env._env, onnx_path=onnx_path_str)
        logger.info(
            f"Successfully exported minimal policy to: {onnx_path_str}"
        )""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_base_runner,
            content=(
                "MJLab's generic export path assumes the RSL-RL policy can be "
                "converted with `as_onnx()` and dummy inputs from that wrapper."
            ),
            text="""
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]""",
        ),
        change_direction=(
            "Make MJLab export actor-driven: unwrap the trained actor, prefer an "
            "`export_onnx()` method when present, pass export options such as "
            "`use_kv_cache`, and fall back to the legacy `as_onnx()` path only "
            "for old MLP actors. Export metadata should still be attached after "
            "the actor writes the ONNX file."
        ),
        change_reason=(
            "The generalized tracker's policy may be a transformer or MoE actor "
            "with multiple named reference inputs, cache tensors, and optional "
            "routing diagnostics. A hard-coded MLP `as_onnx()` export will either "
            "fail or silently produce an artifact that cannot be consumed by "
            "sim2sim/deployment."
        ),
        changed_code=r'''
def export_policy_to_onnx(
  self,
  path: str,
  filename: str = "policy.onnx",
  verbose: bool = False,
  *,
  use_kv_cache: bool = True,
) -> None:
  os.makedirs(path, exist_ok=True)
  onnx_path = Path(path) / filename
  policy = self.alg.get_policy()
  actor = getattr(policy, "actor", policy)
  actor = getattr(actor, "_orig_mod", actor)
  actor.eval()

  if hasattr(actor, "export_onnx"):
    kwargs = {"onnx_path": onnx_path, "opset_version": 18}
    signature = inspect.signature(actor.export_onnx)
    if "use_kv_cache" in signature.parameters:
      kwargs["use_kv_cache"] = use_kv_cache
    actor.export_onnx(**kwargs)
    return

  onnx_model = policy.as_onnx(verbose=verbose)
  onnx_model.to("cpu")
  onnx_model.eval()
  torch.onnx.export(
    onnx_model,
    onnx_model.get_dummy_inputs(),
    str(onnx_path),
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=onnx_model.input_names,
    output_names=onnx_model.output_names,
    dynamic_axes={},
    dynamo=False,
  )
''',
    ),
    AtomicLeap(
        holomotion_reference=Highlight(
            file=holomotion_gmr_to_holomotion,
            content=(
                "HoloMotion persists the fully resolved motion-retargeting "
                "configuration beside generated artifacts for reproducibility."
            ),
            text="""
    # dump resolved config used
    (out_root).mkdir(parents=True, exist_ok=True)
    with open(out_root / "config_used.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))""",
        ),
        mjlab_reference=Highlight(
            file=mjlab_train,
            content=(
                "MJLab writes env and agent YAML before runner construction, but "
                "does not persist a unified resolved manifest for train/eval/export."
            ),
            text="""
  # Write config files before runner creation, since the runner mutates agent_cfg
  # in-place (e.g., injecting non-serializable objects).
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)""",
        ),
        change_direction=(
            "Add a single resolved run manifest to MJLab tracking logs before "
            "training starts and reuse it during evaluation/export. The manifest "
            "should include task id, env config, agent config, motion artifact or "
            "file path, observation term order, action metadata, CLI overrides, "
            "git commit, seed/rank/device metadata, and later the exported ONNX "
            "path and evaluation split."
        ),
        change_reason=(
            "For generalized tracking, a checkpoint is not meaningful without "
            "the exact motion database, observation schema, export signature, and "
            "run provenance. Separate env/agent YAML files are helpful, but they "
            "do not give sim2sim, offline evaluation, or another engineer a "
            "single source of truth to reproduce a result."
        ),
        changed_code=r'''
def write_tracking_run_manifest(
  log_dir: Path,
  *,
  task_id: str,
  env_cfg: object,
  agent_cfg: object,
  motion_source: str | None,
  env: ManagerBasedRlEnv | None,
  cli_overrides: Sequence[str],
  phase: Literal["train", "eval", "export"] = "train",
  extra: Mapping[str, object] | None = None,
) -> Path:
  manifest = {
    "phase": phase,
    "task_id": task_id,
    "git_commit": get_git_commit(),
    "cli_overrides": list(cli_overrides),
    "env": to_plain_dict(env_cfg),
    "agent": to_plain_dict(agent_cfg),
    "motion_source": motion_source,
    "observation_order": (
      env.observation_manager.active_terms["actor"] if env is not None else None
    ),
    "action_terms": (
      list(env.action_manager.active_terms) if env is not None else None
    ),
    "distributed": {
      "rank": int(os.environ.get("RANK", "0")),
      "world_size": int(os.environ.get("WORLD_SIZE", "1")),
      "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
    },
    "extra": dict(extra or {}),
  }
  manifest_path = log_dir / "params" / "resolved_run_manifest.yaml"
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  dump_yaml(manifest_path, manifest)
  return manifest_path
''',
    ),
]
