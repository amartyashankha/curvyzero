# LightZero Stock Evaluator Action-Mask Gap - 2026-05-09

Scope: official LightZero Atari Pong evaluator/action-mask gap only. No
training. No pytest.

## Verdict

Fix for the evaluator API gap: use LightZero's stock MuZero evaluator path,
`lzero.worker.MuZeroEvaluator`, not DI-engine's generic
`ding.worker.InteractionSerialEvaluator`.

`InteractionSerialEvaluator` is the wrong stock path for MuZero Atari parity. It
forwards only the generic observation payload to `policy.forward`, so
`MuZeroPolicy._forward_eval(data, action_mask, to_play, ...)` is reached without
the required `action_mask`. That is why the earlier stock DI-engine evaluator
attempt failed with:

```text
MuZeroPolicy._forward_eval() missing 1 required positional argument: 'action_mask'
```

Upstream LightZero's `train_muzero` wires `lzero.worker.MuZeroEvaluator` for
MuZero, and that evaluator performs the MuZero-specific collation:

- reads `action_mask`, `to_play`, and `timestep` from `env.ready_obs`;
- builds stacked observations with `GameSegment`;
- calls `policy.forward(stack_obs, action_mask, to_play, ready_env_id=..., timestep=...)`.

That is the clean parity route.

## Code Change

Updated the eval-only official Atari smoke:

- `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py`
  - `run_stock_evaluator=True` now uses `lzero.worker.MuZeroEvaluator`;
  - records positional or keyword `action_mask`, `to_play`, `ready_env_id`, and
    `timestep`;
  - labels `ding.worker.InteractionSerialEvaluator` as the generic path not used
    for MuZero parity.
- `src/curvyzero/infra/modal/lightzero_pong_checkpoint_probe.py`
  - `_to_plain` now serializes sets, needed because `MuZeroEvaluator` passes
    `ready_env_id` as a set.

## Smoke Result

Command run:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --run-stock-evaluator --no-allow-model-fallback --max-eval-steps 4 --step-detail-limit 4
```

Result: pass.

Key fields:

```text
ok: true
LightZero: 0.2.0
DI-engine: 0.5.3
checkpoint: iteration_1.pth.tar
strict_policy_model_load_ok: true
model_fallback_used: false
policy_could_act_in_real_env: true
manual_vs_stock.ok: true
manual_first32_actions: [0, 0, 0, 0]
stock_first32_actions: [0, 0, 0, 0]
stock_evaluator.path: lzero.worker.MuZeroEvaluator
stock_evaluator.recorded_call_count: 4
stock action_mask_values: [[1, 1, 1, 1, 1, 1]]
stock data shape: [1, 4, 64, 64]
stock to_play: [-1]
stock timestep: [0], [1], [2], [3]
artifact ref: training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/eval/iteration_1_tiny/lightzero_visual_pong_eval_smoke_20260509T194012Z.json
artifact sha256: 6b5b998542d44659774ab3f0b708b98938e3612cafc7357bb352849d9d74e99f
```

The subprocess env manager logged an `invalid load key, 'A'` reset warning
during cleanup after the MuZero evaluator completed and after the artifact was
written. It did not block the recorded evaluator parity pass.

## Config-Shape Caveat

The Hugging Face/OpenDILab pretrained Pong checkpoint shape issue is separate
from the action-mask API gap. If a checkpoint was trained with an older 96x96
downsample config while the current stock Atari config/eval path is 64x64, then
strict load/eval can fail before evaluator parity is meaningful.

That older config path would affect evaluator parity in this specific way:

- it changes the model/input contract and checkpoint loadability;
- it may require the matching older Atari config and wrapper observation shape;
- it does not change the evaluator API conclusion.

In other words: `InteractionSerialEvaluator` remains blocked for MuZero because
it does not collate/pass `action_mask`; `MuZeroEvaluator` is still the clean
stock evaluator path, but only after the checkpoint and config shapes match.

## Sources Checked

- Upstream `train_muzero` uses `MuZeroEvaluator`, not
  `InteractionSerialEvaluator`:
  https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/entry/train_muzero.py
- Upstream `MuZeroEvaluator` extracts `action_mask`, stacks observations, and
  calls `policy.forward(stack_obs, action_mask, to_play, ...)`:
  https://opendilab.github.io/LightZero/_modules/lzero/worker/muzero_evaluator.html
- Upstream `MuZeroPolicy._forward_eval` requires `action_mask`:
  https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/policy/muzero.py
- Upstream Atari env returns `observation`, `action_mask`, `to_play`, and
  `timestep`:
  https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/envs/atari_lightzero_env.py

## Exact Next Command

Use a matching checkpoint/config pair. For the current 64x64 local checkpoint,
the next safe parity command is:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --run-stock-evaluator --no-allow-model-fallback --max-eval-steps 32 --step-detail-limit 32
```

For the Hugging Face pretrained Pong checkpoint, do not run this command until
the matching older 96x96/downsample config path is selected or adapted; otherwise
the result tests shape compatibility, not the action-mask/evaluator API gap.
