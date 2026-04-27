from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

train_script = File(path="HoloMotion/holomotion/scripts/training/train_motion_tracking.sh")
train_py = File(path="HoloMotion/holomotion/src/training/train.py")
env_cfg = File(path="HoloMotion/holomotion/config/env/motion_tracking.yaml")
motion_env = File(path="HoloMotion/holomotion/src/env/motion_tracking.py")
scene_builder = File(path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_scene.py")

scene_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=train_script,
            content="""
The shell entrypoint selects the motion-tracking Hydra config and launches the Python training module through Accelerate.""",
            text="""
COMMON_ARGS=(
    "holomotion/src/training/train.py"
    "--config-name=training/motion_tracking/${config_name}"
    "num_envs=${num_envs}"
    "headless=true"
    "experiment_name=${config_name}"
)""",
        ),
        Highlight(
            file=train_py,
            content="""
The training module compiles the Hydra config, resolves `config.algo._target_`, and passes the full environment config into the algorithm constructor.""",
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
            file=env_cfg,
            content="""
The environment config declares `MotionTrackingEnv` as the runtime environment and wires robot, terrain, rewards, observations, and terminations into it.""",
            text="""
env:
  _target_: holomotion.src.env.motion_tracking.MotionTrackingEnv
  _recursive_: False
  config:
    experiment_name: ${experiment_name}
    num_envs: ${num_envs}
    env_spacing: 2.5 # meters""",
        ),
        Highlight(
            file=motion_env,
            content="""
`MotionTrackingEnv` turns resolved robot, terrain, lighting, and contact-sensor dictionaries into an IsaacLab scene config.""",
            text="""
            scene: MotionTrackingSceneCfg = build_scene_config(
                scene_config_dict,
                main_process=main_process,
                process_id=process_id,
                num_processes=num_processes,
            )""",
        ),
        Highlight(
            file=scene_builder,
            content="""
The scene builder validates the robot asset, converts URDF to USD in a controlled output directory, and returns an IsaacLab `ArticulationCfg`.""",
            text="""
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        # Configure USD output directory. Optionally isolate per rank to avoid races.
        usd_base_dir = os.path.join(os.path.dirname(urdf_path), "usd")""",
        ),
    ]
)
