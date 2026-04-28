# MJLab Tracking HDR Report Example

## Local Catalog

- `mjlab` is a symlink to `/Users/wujimi/mjlab`; all proof `File(...)` paths should be relative to this example root and normally start with `mjlab/...`.
- `contract/project_analysis.py` defines the HDR contract types: `Highlight`, `ProofFromCode`, and `ProjectAnalysis`.
- `contract/*_correct.py` are the evidence modules. Each file should export one variable matching its filename, for example `scene_correct`.
- `contract/finish_project_analysis.py` imports all proof variables and constructs the final `ProjectAnalysis`.

The proof modules map to the final report fields:

- `scene_correct.py`: MJLab task registration, G1 scene entity, flat terrain, MuJoCo timing, domain randomization, termination stability.
- `model_correct.py`: policy/value model architecture, observation schema, action output dimension, activation and hidden sizes.
- `action_correct.py`: action shape, action manager wiring, scaling/offsets, robot joint application.
- `observation_correct.py`: actor/critic observation groups, state-estimation-dependent terms, dimensions and meanings.
- `reset_correct.py`: reset events, robot root/joint initialization, command/motion state reset.
- `reward_correct.py`: reward table, term weights, reward functions, sensor/command dependencies.
- `loss_correct.py`: PPO loss terms, value loss, entropy, clipping, optimizer step.
- `trainer_correct.py`: task launch path, runner construction, rollout loop, model/environment updates.
- `logs_correct.py`: reward/step metrics, logger/TensorBoard handling, iteration summaries.
- `checkpoint_correct.py`: checkpoint save/load behavior, policy state, optimizer state, export/resume path.
- `viewer_correct.py`: play/viewer script and runtime viewer launch path.
- `command_correct.py`: motion command construction, dataset/window sampling, resampling/curriculum behavior, command use in training/eval.

## How To Run

Run from `examples/mjlab_tracking_report`:

```bash
uv run python contract/finish_project_analysis.py
```

For a single proof:

```bash
uv run python contract/scene_correct.py
uv run pyright contract/scene_correct.py
uv run ruff check contract/scene_correct.py
```

## Things To Remember

- `Highlight.text` must be an exact full-line excerpt from `Highlight.file`, start with a newline, not end with a newline, and appear exactly once in that file.
- Put explanation in `Highlight.content`; `ProofFromCode` only accepts `highlights`.
- If grouping evidence, use named intermediate proofs:

```python
task_registration = ProofFromCode(highlights=[...])
scene_runtime = ProofFromCode(highlights=[...])

scene_correct = ProofFromCode(
    highlights=[
        task_registration,
        scene_runtime,
    ]
)
```

- Keep paths portable inside this example. Do not use `/Users/...` in `File(path=...)`; use `mjlab/...`.
- Prefer bottom-up proof: quote task config, then builder/wiring code, then runtime use.
- MJLab proofs often depend on distinguishing the registered task from nearby variants. Anchor the proof in the exact task id/config first.
- For observation proofs, be explicit about state-estimation variants; some terms are removed when state estimation is disabled.
- For scene and reward proofs, connect sensors and terrain from config to implementation, not just declarations.
- For command proofs, focus on how the dataset/window sampler advances and how command state reaches observations, rewards, resets, and evaluation.
