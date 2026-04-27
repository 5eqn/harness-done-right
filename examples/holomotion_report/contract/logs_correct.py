from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

algo_base = File(path="HoloMotion/holomotion/src/algo/algo_base.py")
algo_utils = File(path="HoloMotion/holomotion/src/algo/algo_utils.py")
motion_env = File(path="HoloMotion/holomotion/src/env/motion_tracking.py")

logs_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=algo_base,
            content="""
Accelerate initializes TensorBoard trackers and writes rank-specific plus main-process log files under `log_dir`.""",
            text="""
        self.accelerator.init_trackers(
            project_name="holomotion",
            config={
                "precision": mixed_precision if mixed_precision else "fp32",
                "dynamo_backend": dynamo_backend if dynamo_backend else "none",""",
        ),
        Highlight(
            file=algo_utils,
            content="""
`AlgoLogger` sends scalar metrics to Accelerate/TensorBoard and builds a console table for the same iteration.""",
            text="""
        if len(tensorboard_metrics) > 0:
            self.accelerator.log(tensorboard_metrics, step=int(step))

        console_metrics = {
            key: self._format_console_value(value)
            for key, value in metrics.items()
            if value is not None
        }""",
        ),
        Highlight(
            file=motion_env,
            content="""
The environment records robot action rate, acceleration, torque, energy, and normalized torque-rate metrics into `infos["log"]`.""",
            text="""
        infos["log"]["Metrics/Robot/Action_Rate"] = self.metrics[
            "Robot/Action_Rate"
        ]
        infos["log"]["Metrics/Robot/DOF_Acc"] = self.metrics["Robot/DOF_Acc"]
        infos["log"]["Metrics/Robot/DOF_Torque"] = self.metrics[
            "Robot/DOF_Torque"
        ]""",
        ),
    ]
)
