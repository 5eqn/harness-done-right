from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

algo_cfg = File(path="HoloMotion/holomotion/config/algo/ppo.yaml")
ppo = File(path="HoloMotion/holomotion/src/algo/ppo.py")
storage = File(path="HoloMotion/holomotion/src/algo/algo_utils.py")

loss_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=algo_cfg,
            content="""
The PPO config sets the concrete loss hyperparameters: clipping, discounting, value coefficient, entropy coefficient, KL target, and gradient norm.""",
            text="""
    clip_param: 0.2
    gamma: 0.99
    lam: 0.95
    value_loss_coef: 1.0
    entropy_coef: 5.0e-3
    anneal_entropy: false""",
        ),
        Highlight(
            file=storage,
            content="""
Rollout storage computes GAE returns from rewards, values, dones, gamma, and lambda before PPO updates.""",
            text="""
            delta = (
                rewards[step]
                + next_is_not_terminal * gamma * next_values
                - values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            returns[step] = advantage + values[step]""",
        ),
        Highlight(
            file=ppo,
            content="""
The actor loss uses PPO's clipped surrogate ratio against old action log probabilities.""",
            text="""
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()""",
        ),
        Highlight(
            file=ppo,
            content="""
The value loss optionally uses PPO value clipping, then actor and critic losses are backpropagated separately.""",
            text="""
            actor_loss = surrogate_loss
            critic_loss = self.value_loss_coef * value_loss

            if entropy_coef > 0.0:
                entropy_loss = entropy_batch.mean()
                actor_loss = actor_loss - entropy_coef * entropy_loss""",
        ),
    ]
)
