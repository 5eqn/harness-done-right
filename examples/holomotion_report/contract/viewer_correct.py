from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

train_py = File(path="HoloMotion/holomotion/src/training/train.py")
mujoco_eval = File(path="HoloMotion/holomotion/src/evaluation/eval_mujoco_sim2sim.py")
eval_cfg = File(path="HoloMotion/holomotion/config/evaluation/eval_mujoco_sim2sim.yaml")

viewer_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=train_py,
            content="""
Post-training MuJoCo evaluation is launched only after ONNX export is enabled and exported checkpoints are resolved.""",
            text="""
    if not bool(config.mujoco_eval.get("enabled", False)):
        return
    if not bool(config.algo.config.get("export_policy", False)):
        msg = (
            "mujoco_eval.enabled=true requires "
            "algo.config.export_policy=true to export ONNX "
            "before post-training evaluation."
        )""",
        ),
        Highlight(
            file=eval_cfg,
            content="""
The MuJoCo eval config includes shared viewer/offscreen camera settings and a Unitree viewer update rate.""",
            text="""
camera_azimuth: 150.0 # default viewer/offscreen azimuth (deg), side-ish view
camera_elevation: -20.0 # default viewer/offscreen elevation (deg), slight downward angle""",
        ),
        Highlight(
            file=mujoco_eval,
            content="""
The viewer path launches MuJoCo passive viewer, applies the configured camera, and optionally initializes recording in viewer mode.""",
            text="""
        viewer = mujoco.viewer.launch_passive(self.m, self.d)

        # Configure viewer camera to use shared align / tracking settings
        self._configure_viewer_camera(viewer)""",
        ),
        Highlight(
            file=mujoco_eval,
            content="""
The viewer loop continues while the MuJoCo viewer is running, updates camera/overlays, and synchronizes viewer state each tick.""",
            text="""
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
    ]
)
