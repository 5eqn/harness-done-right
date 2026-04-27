from hdr.contracts.std import File
from project_analysis import Highlight, ProofFromCode

train_script = File(path="HoloMotion/holomotion/scripts/training/train_motion_tracking.sh")
train_py = File(path="HoloMotion/holomotion/src/training/train.py")
algo_base = File(path="HoloMotion/holomotion/src/algo/algo_base.py")

trainer_correct = ProofFromCode(
    highlights=[
        Highlight(
            file=train_script,
            content="""
The training shell script chooses single-GPU or multi-GPU Accelerate launch based on `CUDA_VISIBLE_DEVICES`.""",
            text="""
if [[ "${USE_MULTI_GPU}" == "true" ]]; then
    ${Train_CONDA_PREFIX}/bin/accelerate launch \\
        --multi_gpu \\
        "${COMMON_ARGS[@]}"
else
    ${Train_CONDA_PREFIX}/bin/accelerate launch \\
        "${COMMON_ARGS[@]}"
fi""",
        ),
        Highlight(
            file=train_py,
            content="""
The trainer loads any checkpoint and delegates all optimization to the algorithm `learn` method.""",
            text="""
    algo.load(config.checkpoint)
    algo.learn()""",
        ),
        Highlight(
            file=algo_base,
            content="""
The base on-policy loop alternates rollout collection, PPO update, logging, checkpointing, and distributed synchronization.""",
            text="""
            obs_td = self.rollout_policy(obs_td)

            stop = time.time()
            collection_time = stop - start
            start = stop

            loss_dict = self.update()""",
        ),
    ]
)
