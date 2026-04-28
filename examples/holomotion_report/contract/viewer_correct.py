from project_analysis import File, Highlight, ProofFromCode


eval_doc = File(path="HoloMotion/docs/evaluate_motion_tracking.md")
mujoco_doc = File(path="HoloMotion/docs/mujoco_sim2sim.md")
eval_script = File(
    path="HoloMotion/holomotion/scripts/evaluation/eval_motion_tracking.sh"
)
mujoco_script = File(
    path="HoloMotion/holomotion/scripts/evaluation/eval_mujoco_sim2sim.sh"
)
mujoco_viz_script = File(
    path="HoloMotion/holomotion/scripts/motion_retargeting/run_motion_viz_mujoco.sh"
)
eval_cfg = File(path="HoloMotion/holomotion/config/evaluation/eval_isaaclab.yaml")
mujoco_eval_cfg = File(
    path="HoloMotion/holomotion/config/evaluation/eval_mujoco_sim2sim.yaml"
)
eval_single = File(
    path="HoloMotion/holomotion/src/evaluation/eval_motion_tracking_single.py"
)
algo_base = File(path="HoloMotion/holomotion/src/algo/algo_base.py")
ppo = File(path="HoloMotion/holomotion/src/algo/ppo.py")
motion_tracking_env = File(path="HoloMotion/holomotion/src/env/motion_tracking.py")
mujoco_eval = File(
    path="HoloMotion/holomotion/src/evaluation/eval_mujoco_sim2sim.py"
)
mujoco_viz = File(
    path="HoloMotion/holomotion/src/motion_retargeting/utils/visualize_with_mujoco.py"
)


viewer_correct = ProofFromCode(
    highlights=[
        ProofFromCode(
            highlights=[
                Highlight(
                    file=eval_doc,
                    content="""
The user-facing evaluation document names the supported offline IsaacLab evaluation command, so the viewer/evaluation proof starts from a documented launch surface rather than a private helper.""",
                    text="""
```bash
bash ./holomotion/scripts/evaluation/eval_motion_tracking.sh
```""",
                ),
                Highlight(
                    file=eval_script,
                    content="""
The documented IsaacLab evaluation script is explicitly headless by default and launches `eval_motion_tracking_single.py` through Accelerate with checkpoint, dataset, policy export, NPZ dump, metric, and report flags. This is the concrete evaluation entrypoint used before visualization.""",
                    text="""
HEADLESS=true
CONFIG_NAME="eval_isaaclab"

CKPT_PATH="logs/HoloMotionMotrackV1.2/your_log_dir/model_xxx.pt"

eval_h5_dataset_path="['data/h5v2_datasets/lafan1']"

num_envs=4


${Train_CONDA_PREFIX}/bin/accelerate launch \\
    holomotion/src/evaluation/eval_motion_tracking_single.py \\
    --config-name=evaluation/${CONFIG_NAME} \\
    headless=${HEADLESS} \\
    num_envs=${num_envs} \\
    export_policy=true \\
    dump_npzs=true \\
    calc_per_clip_metrics=true \\
    generate_report=true \\
    motion_h5_path=${eval_h5_dataset_path} \\
    +use_kv_cache=true \\
    export_only=false \\
    checkpoint=$CKPT_PATH \\
    project_name="HoloMotionMoTrack\"""",
                ),
                Highlight(
                    file=eval_cfg,
                    content="""
The IsaacLab evaluation Hydra config requires the launcher to provide `headless`, `num_envs`, and checkpoint/data inputs, and it exposes `dump_npzs`, `calc_per_clip_metrics`, and `generate_report` toggles used by the script above.""",
                    text="""
num_envs: ???
headless: ???

motion_h5_path: null
checkpoint: null
log_dir: null
ckpt_pt_names: null""",
                ),
                Highlight(
                    file=eval_cfg,
                    content="""
The same config makes output generation explicit: policy export is available but disabled by default, while offline NPZ dumps, per-clip metrics, and report generation are separate launch-time switches.""",
                    text="""
export_policy: false
export_only: false

dump_npzs: false
calc_per_clip_metrics: false
generate_report: false
dof_mode: "23\"""",
                ),
                Highlight(
                    file=eval_single,
                    content="""
The Python evaluation entrypoint binds to Hydra config `evaluation/eval_isaaclab`, matching the shell script's `--config-name=evaluation/${CONFIG_NAME}`. That proves the documented script reaches this code path.""",
                    text="""
@hydra.main(
    config_path="../../config",
    config_name="evaluation/eval_isaaclab",
    version_base=None,
)""",
                ),
                Highlight(
                    file=eval_single,
                    content="""
The evaluation entrypoint refuses to run without a checkpoint, loads the training config from that checkpoint, compiles the merged config, and preserves `config.headless` for the PPO/evaluator construction.""",
                    text="""
    export_only = bool(config.get("export_only", False))
    if export_only:
        checkpoint_paths = _resolve_export_ckpt_paths(config)
        config = load_training_config(str(checkpoint_paths[0]), config)
    else:
        if config.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for evaluation")
        checkpoint_paths = [Path(str(config.checkpoint))]
        config = load_training_config(config.checkpoint, config)

    # Compile config without accelerator (PPO will create it)
    config = compile_config(config, accelerator=None)

    # Use checkpoint directory as log_dir for offline evaluation/export.
    log_dir = str(checkpoint_paths[0].parent)
    headless = config.headless""",
                ),
                Highlight(
                    file=eval_single,
                    content="""
The evaluated algorithm is constructed with `headless=headless` and `is_offline_eval=True`, so the IsaacLab viewer/headless choice from the launch script is carried into environment/AppLauncher setup instead of being ignored.""",
                    text="""
    # PPO creates Accelerator, AppLauncher, and environment internally
    algo_class = get_class(config.algo._target_)
    algo = algo_class(
        env_config=config.env,
        config=config.algo.config,
        log_dir=log_dir,
        headless=headless,
        is_offline_eval=True,
    )""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=algo_base,
                    content="""
IsaacLab launch flags are built in one place from the evaluation/training config. Cameras are enabled when recording or GUI mode needs them; the actual AppLauncher receives `headless`, `enable_cameras`, `video`, `device`, and Kit arguments that disable Omniverse multi-GPU rendering.""",
                    text="""
        # Create AppLauncher with accelerator device
        # Enable cameras only when needed:
        # - headless & recording: True (offscreen rendering)
        # - headless & not recording: False (maximize performance)
        # - with GUI: True
        _record_video = bool(self.config.get("record_video", False))
        enable_cameras = _record_video or (not self.headless)""",
                ),
                Highlight(
                    file=algo_base,
                    content="""
This is the exact AppLauncher invocation: the proof of viewer/headless correctness rests on concrete flags passed to IsaacLab, including `video` for recording and `headless` for GUI suppression.""",
                    text="""
        app_launcher_flags = {
            "headless": self.headless,
            "enable_cameras": enable_cameras,
            "video": _record_video,
            "device": device_str,
            "kit_args": kit_args_str,
        }

        self._sim_app_launcher = AppLauncher(**app_launcher_flags)
        self._sim_app = self._sim_app_launcher.app""",
                ),
                Highlight(
                    file=algo_base,
                    content="""
When `record_video` is true, the environment is created with `render_mode="rgb_array"`; otherwise it is non-rendering. The same constructor passes the headless value to the HoloMotion environment wrapper.""",
                    text="""
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
        )""",
                ),
                Highlight(
                    file=motion_tracking_env,
                    content="""
The concrete IsaacLab environment config includes a world-origin `ViewerCfg`, so a non-headless launch has a real IsaacLab viewer configuration available.""",
                    text="""
            viewer: ViewerCfg = ViewerCfg(origin_type="world")

            motion_cmds = {}
            vel_cmds = {}""",
                ),
                Highlight(
                    file=motion_tracking_env,
                    content="""
The wrapper passes the selected render mode into `ManagerBasedRLEnv`, completing the chain from launch flag to IsaacLab environment instantiation and render behavior.""",
                    text="""
        self._env = ManagerBasedRLEnv(isaaclab_env_cfg, self.render_mode)

        logger.info("IsaacLab environment initialized !")
        return self._env""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=eval_single,
                    content="""
After checkpoint loading, evaluation calls `offline_evaluate_policy(dump_npzs)` and optionally runs metrics/report generation on the resulting output directory. This proves the IsaacLab evaluation launch has a non-interactive output path for later viewing.""",
                    text="""
    result = algo.offline_evaluate_policy(dump_npzs)
    algo.accelerator.wait_for_everyone()

    if algo.accelerator.is_main_process:
        logger.info("Evaluation completed successfully!")
        output_dir = (
            result.get("output_dir") if isinstance(result, dict) else None
        )
        if output_dir is not None:
            logger.info(f"NPZs saved to: {output_dir}")""",
                ),
                Highlight(
                    file=ppo,
                    content="""
Offline evaluation is deterministic and clip-oriented: it iterates validation cache batches, resets environments, aligns each env to one clip, rolls the policy forward, and saves one result NPZ per motion clip.""",
                    text="""
        # Switch to validation cache and iterate all batches
        if hasattr(cache, "set_mode"):
            cache.set_mode("val")
        # Determine policy/video FPS from command config (align wallclock time)
        motion_fps = int(getattr(motion_cmd.cfg, "target_fps", 50))
        total_batches = int(getattr(cache, "num_batches", 1))
        with torch.no_grad():
            for batch_idx in tqdm(
                range(total_batches), desc="Evaluating batches"
            ):
                if batch_idx > 0:
                    cache.advance()
                # Reset envs first, then apply deterministic mapping on the active cache batch
                _ = self.env.reset_all()""",
                ),
                Highlight(
                    file=ppo,
                    content="""
The offline evaluation render-free loop still advances the live environment by calling `_rollout_forward` every rollout step, then records robot/reference state and saves NPZs at the end. This is the evaluation-side analogue of a render loop: it proves time advances through the simulated policy rather than just exporting static data.""",
                    text="""
                    # No mid-rollout finalize; we defer to end using valid masks
                    # Inference and step (advance sim)
                    obs = self._rollout_forward(
                        obs,
                        actor_mode="inference",
                        collect_transition=False,
                        track_episode_stats=False,
                    )
                    dones = self._last_rollout_dones
                    if dones is None:
                        raise RuntimeError(
                            "Rollout forward did not return dones during offline evaluation."
                        )""",
                ),
                Highlight(
                    file=eval_doc,
                    content="""
The docs tie the offline NPZ outputs to a visual inspection step through MuJoCo, including the required `robot_` prefix for recorded robot states and the expected MP4 output path.""",
                    text="""
Generate video outputs to validate the motion tracking quality from the `.npz` result files by setting the `motion_npz_root` to the evaluation npz folder. Note that in order to properly visualize the recorded robot data, you should set the `+key_prefix="robot_"` .""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=mujoco_doc,
                    content="""
The sim2sim documentation names the MuJoCo verification entry script and the three launch inputs that matter for viewing an exported policy against a reference motion: robot XML, ONNX policy, and motion NPZ.""",
                    text="""
The entry script is `holomotion/scripts/evaluation/eval_mujoco_sim2sim.sh`, you should set these variables before running:

- `robot_xml_path`: The scene mjcf .xml file for the robot
- `ONNX_PATH`: The exported ONNX model file
- `motion_npz_path`: The npz file containing the reference motion""",
                ),
                Highlight(
                    file=mujoco_script,
                    content="""
The MuJoCo sim2sim shell script has an explicit headless/viewer switch: headless mode selects OSMesa and video recording, while GUI mode selects EGL and disables recording by default.""",
                    text="""
export HEADLESS=false
if $HEADLESS; then
    export MUJOCO_GL="osmesa"
    export RECORD_VIDEO=true
else
    export MUJOCO_GL="egl"
    export RECORD_VIDEO=false
fi""",
                ),
                Highlight(
                    file=mujoco_script,
                    content="""
The same script launches the MuJoCo evaluator with `record_video`, `headless`, camera-tracking controls, ONNX path, motion NPZ path, and robot XML path, which is exactly the configurable viewer/video surface described in the docs.""",
                    text="""
${Train_CONDA_PREFIX}/bin/python holomotion/src/evaluation/eval_mujoco_sim2sim.py \\
    record_video=$RECORD_VIDEO \\
    headless=$HEADLESS \\
    camera_tracking=true \\
    camera_distance=7.0 \\
    +model_type=${model_type} \\
    use_gpu=true \\
    dump_npzs=true \\
    dump_onnx_io_npy=false \\
    calc_per_clip_metrics=true \\
    generate_report=true \\
    ray_actors_per_gpu=12 \\
    policy_action_delay_step=0 \\
    action_delay_type=step \\
    +ckpt_onnx_path="$ONNX_PATH" \\
    +motion_npz_path='${oc.env:motion_npz_path}' \\
    robot_xml_path=${robot_xml_path}""",
                ),
                Highlight(
                    file=mujoco_eval_cfg,
                    content="""
The MuJoCo Hydra config exposes the viewer/video defaults directly: GUI is enabled by default (`headless: false`), video is opt-in, and video resolution/FPS plus camera tracking/framing are configurable.""",
                    text="""
# Evaluation toggles
headless: false # true to run without GUI
record_video: false # true to save MP4 recordings
video_width: 1280
video_height: 720
video_fps: 30
camera_tracking: true # true to make camera follow robot root body""",
                ),
                Highlight(
                    file=mujoco_eval,
                    content="""
The MuJoCo evaluator is also bound to Hydra config `evaluation/eval_mujoco_sim2sim`, so the shell overrides land in this exact evaluator rather than a separate script.""",
                    text="""
@hydra.main(
    config_path="../../config",
    config_name="evaluation/eval_mujoco_sim2sim",
    version_base=None,
)""",
                ),
                Highlight(
                    file=mujoco_eval,
                    content="""
MuJoCo sim2sim dispatch is explicit: `headless=true` chooses the headless loop, otherwise it launches the interactive Unitree/MuJoCo viewer path.""",
                    text="""
    def run_simulation(self):
        if bool(self.config.get("headless", False)):
            logger.info("Running MuJoCo sim2sim headless")
            self.run_simulation_unitree_headless()
        else:
            self.run_simulation_unitree()""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=mujoco_eval,
                    content="""
The GUI sim2sim path launches a passive MuJoCo viewer from the loaded model/data, configures the shared camera, and optionally initializes video tools when `record_video` is enabled.""",
                    text="""
        viewer = mujoco.viewer.launch_passive(self.m, self.d)

        # Configure viewer camera to use shared align / tracking settings
        self._configure_viewer_camera(viewer)

        # Start keyboard listener for velocity tracking
        if (
            self.command_mode == "velocity_tracking"
            and self.keyboard_handler is not None
        ):
            self.keyboard_handler.start_listener()

        # Optional recording in viewer mode
        if bool(self.config.get("record_video", False)):
            self._init_video_tools(tag="viewer")""",
                ),
                Highlight(
                    file=mujoco_eval,
                    content="""
The interactive MuJoCo viewer is not a placeholder: it runs simulation and viewer threads until the viewer closes or max steps are reached, updating camera tracking, drawing reference spheres, and calling `viewer.sync()` every viewer tick.""",
                    text="""
        def physics_viewer_thread():
            while viewer.is_running() and not stop_event.is_set():
                with locker:
                    # Update camera lookat to track robot root (with small offset for framing)
                    self._update_camera_lookat(viewer.cam)

                    # Draw reference global bodylink positions as blue spheres when available
                    self._draw_ref_body_spheres_to_scene(
                        viewer.user_scn, reset_ngeom=True
                    )

                    viewer.sync()
                time.sleep(viewer_dt)""",
                ),
                Highlight(
                    file=mujoco_eval,
                    content="""
The headless MuJoCo loop is also real: it resets simulation state, optionally starts recording, repeatedly runs `_run_eval_step`, updates progress, closes video tools, and dumps robot-augmented NPZ output.""",
                    text="""
        running = True
        while running:
            running = self._run_eval_step(max_steps)
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()""",
                ),
                Highlight(
                    file=mujoco_eval,
                    content="""
Video output is constructed from config values and the ONNX/motion stems. It creates an OpenCV MP4 writer and a MuJoCo offscreen renderer that shares the same camera settings as the viewer path.""",
                    text="""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            out_path, fourcc, fps, (width, height)
        )
        self._offscreen = OffscreenRenderer(
            self.m,
            height,
            width,
            distance=self._camera_distance,
            azimuth=self._camera_azimuth,
            elevation=self._camera_elevation,
        )""",
                ),
                Highlight(
                    file=mujoco_eval,
                    content="""
The offscreen frame path actually renders MuJoCo scene state and writes MP4 frames, converting RGB to OpenCV's BGR format before writing.""",
                    text="""
            frame_rgb = self._offscreen.render(self.d)
            # Convert RGB (MuJoCo) -> BGR (OpenCV) before writing
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self._video_writer.write(frame_bgr)
            self._last_frame_time = now""",
                ),
                Highlight(
                    file=mujoco_eval,
                    content="""
The offscreen renderer is grounded in MuJoCo APIs: it updates the scene, draws optional overlays, renders into a viewport, reads pixels, and flips the image before returning it as an RGB frame.""",
                    text="""
    def render(self, data) -> np.ndarray:
        mujoco.mjv_updateScene(
            self.model,
            data,
            self._opt,
            None,
            self._cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self._scene,
        )
        if self._overlay_callback is not None:
            self._overlay_callback(self._scene)
        mujoco.mjr_render(self._viewport, self._scene, self._con)
        mujoco.mjr_readPixels(self._rgb, None, self._viewport, self._con)
        return np.flipud(self._rgb)""",
                ),
            ]
        ),
        ProofFromCode(
            highlights=[
                Highlight(
                    file=mujoco_viz_script,
                    content="""
The separate MuJoCo visualization script is present for post-evaluation rendering of NPZ files. It uses OSMesa for offscreen rendering and passes the documented `robot_` key prefix plus reference overlay flags to the visualizer.""",
                    text="""
export MUJOCO_GL="osmesa"

motion_npz_root="path_to_your_npz_dir"

export motion_name="all"


$Train_CONDA_PREFIX/bin/python holomotion/src/motion_retargeting/utils/visualize_with_mujoco.py \\
    +key_prefix="robot_" \\
    +draw_ref_body_spheres=true \\
    +ref_key_prefix="ref_" \\
    +motion_npz_root=${motion_npz_root} \\
    skip_frames=6 \\
    max_workers=11 \\
    +motion_name='${oc.env:motion_name}'""",
                ),
                Highlight(
                    file=mujoco_viz,
                    content="""
The NPZ visualizer creates a MuJoCo model/data pair, an offscreen renderer, derives output FPS from metadata and skip rate, and opens an MP4 writer at `cfg.video_dir/{motion_name}.mp4`.""",
                    text="""
        mj_model = mujoco.MjModel.from_xml_path(cfg.robot.asset.assetFileName)
        mj_data = mujoco.MjData(mj_model)

        width, height = 1280, 720
        renderer = OffscreenRenderer(mj_model, height, width)

        src_fps = _infer_fps_from_meta(metadata, default_fps=50.0)
        skip_frames = getattr(cfg, "skip_frames", 1)
        actual_fps = src_fps / max(1, int(skip_frames))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(cfg.video_dir, f"{motion_name}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out = cv2.VideoWriter(out_path, fourcc, actual_fps, (width, height))""",
                ),
                Highlight(
                    file=mujoco_viz,
                    content="""
The visualizer has a concrete render loop over motion frames: each selected frame writes root pose, joint positions, forwards MuJoCo kinematics, renders with optional reference markers, converts to BGR, and writes the MP4 frame.""",
                    text="""
            for t in tqdm(
                range(0, T, max(1, int(skip_frames))),
                desc=f"Rendering {motion_name}",
            ):
                root_pos = gpos[t, 0]
                root_quat_xyzw = grot[t, 0]
                root_quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]]

                mj_data.qpos[:3] = root_pos
                mj_data.qpos[3:7] = root_quat_wxyz
                mj_data.qpos[7:] = dof_pos[t]

                mujoco.mj_forward(mj_model, mj_data)""",
                ),
                Highlight(
                    file=mujoco_viz,
                    content="""
Frame production in the NPZ visualizer is a real MuJoCo-to-video path: the rendered frame is converted from RGB to BGR and emitted through the OpenCV writer before resources are closed.""",
                    text="""
                frame = renderer.render(
                    mj_data,
                    ref_body_positions=frame_ref_body_positions,
                )
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        finally:
            out.release()
            renderer.close()""",
                ),
                Highlight(
                    file=mujoco_viz,
                    content="""
The visualizer supports both bulk rendering and single-motion rendering: `motion_name == "all"` creates Ray tasks for every collected motion, otherwise it instantiates `MotionRendererNPZ` and processes the selected clips locally.""",
                    text="""
        # Ray parallel or single-thread mode
        if cfg.motion_name == "all":
            if not ray.is_initialized():
                num_cpus = min(os.cpu_count(), cfg.get("max_workers", 8))
                ray.init(num_cpus=num_cpus)
                print(f"Initialized Ray with {num_cpus} workers")""",
                ),
            ]
        ),
    ]
)
