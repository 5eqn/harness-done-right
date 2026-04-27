from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

play_script = File(path="mjlab/src/mjlab/scripts/play.py")
g1_registry = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py")
g1_env_cfg = File(path="mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py")
motion_commands = File(path="mjlab/src/mjlab/tasks/tracking/mdp/commands.py")
viser_viewer = File(path="mjlab/src/mjlab/viewer/viser/viewer.py")
native_viewer = File(path="mjlab/src/mjlab/viewer/native/viewer.py")

viewer_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=g1_registry,
            content="""
The scoped viewer proof is for exactly `Mjlab-Tracking-Flat-Unitree-G1`: its registered play config calls `unitree_g1_flat_tracking_env_cfg(play=True)` without disabling state estimation, while the no-state-estimation task is a separate id.""",
            text="""
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)""",
        ),
        Highlight(
            file=g1_env_cfg,
            content="""
The G1 flat tracking factory defaults `has_state_estimation` to true, so the scoped task's play environment keeps the state-estimation actor inputs unless the different no-state-estimation registration is selected.""",
            text="""
def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  \"\"\"Create Unitree G1 flat terrain tracking configuration.\"\"\"
  cfg = make_tracking_env_cfg()""",
        ),
        Highlight(
            file=play_script,
            content="""
The play entrypoint loads the task-specific play environment config and the matching RL config by task id before doing any tracking or viewer setup.""",
            text="""
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)""",
        ),
        Highlight(
            file=play_script,
            content="""
The CLI chooses the task from the populated registry and passes that chosen task id into `run_play`, so launching with `Mjlab-Tracking-Flat-Unitree-G1` selects the exact registered play config above.""",
            text="""
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )""",
        ),
        Highlight(
            file=play_script,
            content="""
Tracking detection is explicit: the script only enters the motion-file handling path when the loaded play config has a `motion` command whose config type is `MotionCommandCfg`.""",
            text="""
  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )""",
        ),
        Highlight(
            file=play_script,
            content="""
For tracking play, a local motion file takes precedence; dummy tracking agents must otherwise provide a motion registry artifact, which is downloaded and assigned to `motion_cmd.motion_file`.""",
            text="""
    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\\n"
          "  --motion-file /path/to/motion.npz (local file)\\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )""",
        ),
        Highlight(
            file=play_script,
            content="""
In trained tracking mode, checkpoint files require an explicit motion file unless a W&B run path is available; with a run path, the script resolves the motion artifact used by the run and assigns its `motion.npz`.""",
            text="""
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")""",
        ),
        Highlight(
            file=play_script,
            content="""
The trained play path also resolves a checkpoint: either a local checkpoint file that must exist, or a checkpoint fetched from the W&B run, with `log_dir` set to the checkpoint parent for video/checkpoint UI use.""",
            text="""
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )""",
        ),
        Highlight(
            file=play_script,
            content="""
After the environment is created and wrapped, trained play constructs the registered runner class, loads the actor from the checkpoint strictly, and obtains the inference policy used by the viewer.""",
            text="""
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)""",
        ),
        Highlight(
            file=g1_env_cfg,
            content="""
The G1 play-mode environment is intentionally viewer-friendly: it uses an effectively infinite episode, disables actor corruption, removes interval pushes, disables RSI pose/velocity randomization, and starts playback from the first motion frame.""",
            text="""
  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start\"""",
        ),
        Highlight(
            file=g1_env_cfg,
            content="""
The viewer follows the G1 torso, matching the tracking anchor body rather than an arbitrary camera body.""",
            text="""
  cfg.viewer.body_name = "torso_link\"""",
        ),
        Highlight(
            file=play_script,
            content="""
Viewer backend launch is explicit. `auto` uses the native MuJoCo viewer when a display is present and Viser otherwise; explicit `native` and `viser` route to `NativeMujocoViewer.run()` and `ViserPlayViewer.run()` respectively.""",
            text="""
  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")""",
        ),
        Highlight(
            file=viser_viewer,
            content="""
When Viser is used, the play viewer constructs a Viser server and scene with debug visualization enabled by default, so command debug overlays can be shown immediately.""",
            text="""
    self._external_server = viser_server is not None
    self._server = viser_server or viser.ViserServer(label="mjlab")

  @override
  def setup(self) -> None:
    \"\"\"Setup the viewer resources.\"\"\"
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)

    self._threadpool = ThreadPoolExecutor(max_workers=1)
    self._counter = 0
    self._pending_update_reasons: set[UpdateReason] = set()

    # Create MjlabViserScene for all 3D visualization (with debug visualization enabled).
    self._scene = MjlabViserScene(
      server=self._server,
      mj_model=sim.mj_model,
      num_envs=self.env.num_envs,
    )""",
        ),
        Highlight(
            file=viser_viewer,
            content="""
The Viser controls give command terms their own GUI and debug-visualization controls, wiring command GUI changes back into scene updates and viewer actions.""",
            text="""
      # Let command terms create their own GUI controls.
      env = self.env.unwrapped
      if env.command_manager.active_terms:
        with self._server.gui.add_folder("Commands"):
          env.command_manager.create_gui(
            self._server,
            lambda: self._scene.env_idx,
            on_change=self._scene.request_update,
            request_action=self.request_action,
          )

      # Add standard visualization options from MjlabViserScene.
      def _debug_viz_extra() -> None:
        env.command_manager.create_debug_vis_gui(
          self._server, on_change=self._scene.request_update
        )""",
        ),
        Highlight(
            file=native_viewer,
            content="""
The native backend launches MuJoCo's passive viewer directly and raises if the native viewer handle is missing, proving the native method is a real viewer launch path.""",
            text="""
    self.viewer = mujoco.viewer.launch_passive(
      self.mjm,
      self.mjd,
      key_callback=self._safe_key_callback,
      show_left_ui=False,
      show_right_ui=False,
    )
    if self.viewer is None:
      raise RuntimeError("Failed to launch MuJoCo viewer")""",
        ),
        Highlight(
            file=native_viewer,
            content="""
Each native viewer sync updates debug visualizers before rendering and syncing the passive viewer, so MotionCommand debug geometry can appear in the native path as well.""",
            text="""
      self._update_debug_visualizers(v)
      self._render_other_env_geoms(v, sim, sim_data)

      # Pin tracking camera to body frame origin so DR-induced COM shifts don't move
      # the camera.
      if sim.expanded_fields & self._INERTIAL_FIELDS:
        self._stabilize_tracking_camera()""",
        ),
        Highlight(
            file=motion_commands,
            content="""
`MotionCommand` debug visualization supports ghost mode by creating a visual-only ghost model and drawing a ghost mesh at the current reference root pose and joint positions.""",
            text="""
    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        # Build a ghost model with only visual geoms visible. Collision geoms (nonzero
        # contype/conaffinity) get alpha=0 so the viewer's alpha filter excludes them.
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if (
            self._ghost_model.geom_contype[gi] != 0
            or self._ghost_model.geom_conaffinity[gi] != 0
          ):
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color""",
        ),
        Highlight(
            file=motion_commands,
            content="""
The same debug method also supports frame mode, drawing desired and current body frames for every configured tracking body plus larger desired/current anchor frames.""",
            text="""
    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()""",
        ),
        Highlight(
            file=motion_commands,
            content="""
MotionCommand creates a Viser scrubber with a frame slider, an all-envs checkbox, and a Start Here button that requests a GUI reset action instead of passively changing text-only state.""",
            text="""
  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    \"\"\"Create motion scrubber controls in the Viser viewer.\"\"\"
    max_frame = int(self.motion.time_step_total) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )""",
        ),
        Highlight(
            file=motion_commands,
            content="""
The scrubber's Start Here action maps to `apply_gui_reset`, which resets selected environments to the exact chosen reference frame and refreshes relative body poses for termination/reward consistency.""",
            text="""
  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    self.update_relative_body_poses()
    return True""",
        ),
        Highlight(
            file=viser_viewer,
            content="""
The Viser viewer consumes the GUI reset action, resets either all envs or the selected env, applies the command's GUI reset, writes the changed scene to MuJoCo, forwards and senses, then refreshes the UI state.""",
            text="""
  def _handle_gui_reset(self, all_envs: bool) -> None:
    \"\"\"Reset environment(s) and apply GUI-selected command state.\"\"\"
    env = self.env.unwrapped
    if all_envs:
      env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
    else:
      env_ids = torch.tensor(
        [self._scene.env_idx], dtype=torch.int64, device=env.device
      )

    with self._sim_lock:
      env.reset(env_ids=env_ids)
      if env.command_manager.apply_gui_reset(env_ids):
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.sim.sense()""",
        ),
    ]
)
