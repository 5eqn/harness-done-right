# HoloMotion HDR Report Example

## Local Catalog

- `HoloMotion` is a symlink to `/Users/wujimi/HoloMotion`; all proof `File(...)` paths should be relative to this example root and normally start with `HoloMotion/...`.
- `contract/project_analysis.py` defines the HDR contract types: `Highlight`, `ProofFromCode`, and `ProjectAnalysis`.
- `contract/*_correct.py` are the evidence modules. Each file should export one variable matching its filename, for example `action_correct`.
- `contract/finish_project_analysis.py` imports all proof variables and constructs the final `ProjectAnalysis`.

The proof modules map to the final report fields:

- `scene_correct.py`: IsaacLab scene construction, G1 robot asset, terrain, timing, stability checks.
- `model_correct.py`: actor/critic architecture, MLP or TF-MoE policy details, observation schema, trainable noise.
- `action_correct.py`: 29-DoF action shape, joint-position action term, delay/EMA, actuator torque path, MuJoCo sim2sim action path.
- `observation_correct.py`: actor/critic observation terms, dimensions, histories, future reference frames, normalization.
- `reset_correct.py`: motion-aware reset, sampled clip/start frame, root/DoF alignment and perturbation.
- `reward_correct.py`: motion-tracking reward table, weights, key bodies, local reward implementations and IsaacLab fallback terms.
- `loss_correct.py`: PPO/PPOTF losses, value loss, entropy, clipping, auxiliary TF/MoE losses.
- `trainer_correct.py`: Hydra launch, Accelerate setup, env construction, rollout/update flow.
- `logs_correct.py`: TensorBoard/loguru output, episode metrics, robot/command metrics, distributed logging behavior.
- `checkpoint_correct.py`: save/load path, checkpoint contents, main-process writes, ONNX export metadata.
- `viewer_correct.py`: IsaacLab evaluation launch, headless/viewer flags, MuJoCo sim2sim visualization.
- `command_correct.py`: HDF5 motion command/cache, random vs deterministic sampling, curriculum/weighted-bin hooks, reference tensor update path.

## How To Run

Run from `examples/holomotion_report`:

```bash
uv run python contract/finish_project_analysis.py
```

For a single proof:

```bash
uv run python contract/action_correct.py
uv run pyright contract/action_correct.py
uv run ruff check contract/action_correct.py
```

## Things To Remember

- `Highlight.text` must be an exact full-line excerpt from `Highlight.file`, start with a newline, not end with a newline, and appear exactly once in that file.
- Put explanation in `Highlight.content`; `ProofFromCode` only accepts `highlights`.
- If grouping evidence, use named intermediate proofs:

```python
actor_shape = ProofFromCode(highlights=[...])
actuator_path = ProofFromCode(highlights=[...])

action_correct = ProofFromCode(
    highlights=[
        actor_shape,
        actuator_path,
    ]
)
```

- Keep paths portable inside this example. Do not use `/Users/...` in `File(path=...)`; use `HoloMotion/...`.
- Prefer bottom-up proof: quote config, then builder/wiring code, then runtime use. Avoid claims that are only true in docs or comments unless the implementation also supports them.
- HoloMotion has both MLP and TF-MoE motion-tracking stacks. Say which config a proof is scoped to, and do not silently mix the two unless the proof explicitly compares them.
- `command_correct.py` is usually the hardest proof. Ground dataset/cache claims in `isaaclab_motion_tracking_command.py` and `h5_dataloader.py`, and distinguish default `uniform` sampling from optional `weighted_bin` and `curriculum`.
- `action_correct.py` can trigger pyright issues if nested `ProofFromCode(...)` objects are written inline with extra `content=` fields. Use named proof variables for groups.
