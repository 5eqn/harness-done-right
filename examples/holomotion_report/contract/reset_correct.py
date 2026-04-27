from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

motion_env = File(path="HoloMotion/holomotion/src/env/motion_tracking.py")
algo_base = File(path="HoloMotion/holomotion/src/algo/algo_base.py")
motion_command = File(path="HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_motion_tracking_command.py")

reset_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=motion_env,
            content="""
The environment exposes both per-index and all-env reset paths directly to the wrapped IsaacLab env.""",
            text="""
    def reset_idx(self, env_ids: torch.Tensor):
        return self._env.reset(env_ids=env_ids)

    def reset_all(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        out = self._env.reset(env_ids=env_ids)
        return out""",
        ),
        Highlight(
            file=algo_base,
            content="""
The learning loop begins from a full environment reset before wrapping observations into a TensorDict and building rollout storage.""",
            text='''
    def learn(self):
        """Main learning loop with runner logic shared across on-policy algorithms."""
        obs_dict = self.env.reset_all()[0]
        obs_td = self._wrap_obs_dict(obs_dict)
        self._ensure_storage(obs_td)
        self.train_mode()''',
        ),
        Highlight(
            file=motion_command,
            content="""
The reference-motion command owns reset-time motion sampling through its `reset` method, so environment resets also refresh command state.""",
            text="""
    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, float]:""",
        ),
    ]
)
