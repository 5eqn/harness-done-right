from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

algo_base = File(path="HoloMotion/holomotion/src/algo/algo_base.py")
ppo = File(path="HoloMotion/holomotion/src/algo/ppo.py")
train_doc = File(path="HoloMotion/docs/train_motion_tracking.md")

checkpoint_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=algo_base,
            content="""
The training loop saves periodic checkpoints at `save_interval` and also saves a final model at the current learning iteration.""",
            text="""
            if self.is_main_process and it % self.save_interval == 0:
                self.save(
                    os.path.join(
                        self.log_dir,
                        f"model_{self.current_learning_iteration}.pt",
                    )
                )
                self._release_cuda_cache()""",
        ),
        Highlight(
            file=ppo,
            content="""
PPO checkpoint loading restores actor, critic, optimizer states, current iteration, sigma override, and extra subclass state.""",
            text="""
        actor_model_path = self._resolve_model_file_path(ckpt_path, "actor")
        critic_model_path = self._resolve_model_file_path(ckpt_path, "critic")
        self._load_accelerate_model(self.actor, actor_model_path, strict=True)
        self._load_accelerate_model(
            self.critic, critic_model_path, strict=True
        )""",
        ),
        Highlight(
            file=ppo,
            content="""
PPO saving writes Accelerate actor/critic directories, optimizer and iteration metadata, plus optional ONNX export.""",
            text="""
        self.accelerator.save_model(
            self.actor, os.path.join(base_path, "actor")
        )
        self.accelerator.save_model(
            self.critic, os.path.join(base_path, "critic")
        )""",
        ),
        Highlight(
            file=train_doc,
            content="""
The user documentation exposes the checkpoint interval and resume path override expected by the code.""",
            text="""
#### How to resume training from a checkpoint ?

To resume training from a pretrained checkpoint, you can find the checkpoint in the log directory, and then add the option like this: `checkpoint=logs/HoloMotion/20250728_214414-train_unitree_g1_21dof_teacher/model_X.pt`""",
        ),
    ]
)
