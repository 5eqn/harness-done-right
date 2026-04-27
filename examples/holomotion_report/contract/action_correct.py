from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

env_cfg = File(path="HoloMotion/holomotion/config/env/motion_tracking.yaml")
actions_builder = File(path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_actions.py")
algo_base = File(path="HoloMotion/holomotion/src/algo/algo_base.py")

action_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=env_cfg,
            content="""
The motion-tracking config defines a joint-position action term over all robot joints, scaled by the robot actuator config.""",
            text="""
    actions:
      dof_pos:
        type: joint_position
        params:
          asset_name: robot
          joint_names:
            - ".*"
          use_default_offset: true
          scale: ${robot.actuators.action_scale}""",
        ),
        Highlight(
            file=actions_builder,
            content="""
The action builder maps the `joint_position` config into IsaacLab's `JointPositionActionCfg`, preserving asset, joint, offset, and scale fields.""",
            text="""
    def joint_position_action(
        asset_name: str = "robot",
        joint_names: list[str] | None = None,
        use_default_offset: bool = True,
        scale: float = 1.0,
    ) -> mdp.JointPositionActionCfg:""",
        ),
        Highlight(
            file=algo_base,
            content="""
During rollout, the actor output named `actions` is sent directly to `self.env.step(actions)`.""",
            text="""
        actions = actor_out.get("actions")
        self._last_rollout_actions = actions
        obs_dict, rewards, dones, time_outs, infos = self.env.step(actions)""",
        ),
    ]
)
