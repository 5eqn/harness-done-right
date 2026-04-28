# MJLab Enhance Guidance HDR Example

## Local Catalog

- `mjlab` is a symlink to `/Users/wujimi/mjlab`; MJLab `File(...)` paths should be relative to this example root and start with `mjlab/...`.
- `HoloMotion` is a symlink to `/Users/wujimi/HoloMotion`; HoloMotion `File(...)` paths should be relative to this example root and start with `HoloMotion/...`.
- `contract/enhance_guidance.py` defines the HDR contract types: `Highlight`, `AtomicLeap`, and `EnhanceGuidance`.
- `contract/*_leaps.py` are the guidance modules. Each file should export one variable matching its filename, for example `command_leaps`.
- `contract/finish_enhance_guidance.py` imports all leap variables and constructs the final `EnhanceGuidance`.

The leap modules map to the final guidance fields:

- `model_leaps.py`: model architecture changes from MJLab MLP PPO toward HoloMotion TF-MoE/reference-routed tracking.
- `training_leaps.py`: trainer and optimizer changes, PPOTF-style sequence updates, distributed setup, and actor/critic optimizer separation.
- `command_leaps.py`: motion-library and command-pipeline changes from single `.npz` tracking to HDF5-v2 windows, cache assignment, future-frame gathers, and dataset-level sampling.
- `environment_leaps.py`: robustness changes such as rough terrain, action delay, wider domain randomization, and stronger tracking terminations.
- `observation_leaps.py`: named unified observations, future references, privileged critic state, and router-aware reference terms.
- `reward_leaps.py`: reward shaping changes for root/keybody tracking, immediate-next reference targets, scoped contact regularization, and normalized effort terms.
- `other_leaps.py`: supporting evaluation, export, and reproducibility changes that do not fit the core six categories.

## How To Run

Run from `examples/mjlab_enhance_guidance`:

```bash
uv run python contract/finish_enhance_guidance.py
```

For a single leap module:

```bash
uv run python -c "import sys; sys.path.insert(0, 'contract'); import command_leaps"
uv run pyright contract/command_leaps.py
uv run ruff check contract/command_leaps.py
```

## Things To Remember

- `Highlight.text` must be an exact full-line excerpt from `Highlight.file`, start with a newline, not end with a newline, and appear exactly once in that file.
- Every `AtomicLeap` needs one HoloMotion highlight, one MJLab highlight, `change_direction`, `change_reason`, and code-only `changed_code`.
- Keep paths portable inside this example. Do not use `/Users/...` in `File(path=...)`; use `HoloMotion/...` and `mjlab/...`.
- This example is guidance, not a proof report. `changed_code` should be draft implementation direction that engineers can act on, not claims that the code already exists.
- Omit differences that are only simulator/framework choices. Prefer leaps that help MJLab become a generalized motion tracker while staying implementable in MJLab.
- Keep each leap atomic: one coherent change, one reason it matters, and a small code sketch. If a change spans data, model, and trainer, split it across the relevant `*_leaps.py` files.
- Ground HoloMotion-side claims in TF-MoE/generalized-tracking code where possible, and ground MJLab-side references in the current tracking task/code waiting to change.
