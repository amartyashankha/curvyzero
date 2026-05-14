# Eval/GIF Cadence Pass-Through Patch Plan

Date: 2026-05-13

Scope: focused follow-up audit after the one-frame cadence patch. No source was
edited for this audit.

## Finding

Dirac's finding checks out.

The background eval/GIF config objects already carry cadence:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py::_background_eval_config_from_command` adds `decision_ms`, `decision_source_frames`, `source_physics_step_ms`, and `source_max_steps_semantics` at lines 5687-5695.
- `_background_gif_config_from_command` adds the same fields at lines 5772-5780.

But the checkpoint spawn helpers drop those fields:

- `_spawn_one_checkpoint_background_eval` starts at line 5950. Its call to `lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect.spawn` begins at line 5982 and passes `source_max_steps` at line 5994, but not `decision_ms`, `decision_source_frames`, `source_physics_step_ms`, or `source_max_steps_semantics`.
- `_spawn_one_checkpoint_background_gif` starts at line 6066. Its call to `lightzero_curvytron_visual_survival_checkpoint_selfplay_gif.spawn` begins at line 6131 and passes `source_max_steps` at line 6141, but not the cadence fields.

The remote functions also cannot receive the fields yet:

- `_run_checkpoint_eval_and_inspect` at line 6650 has no cadence parameters.
- `lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect` at line 8869 has no cadence parameters.
- `_run_checkpoint_selfplay_gif` at line 7575 has no cadence parameters.
- `lightzero_curvytron_visual_survival_checkpoint_selfplay_gif` at line 8926 has no cadence parameters.

The env builder then falls back to defaults:

- `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py::_make_policy_and_env` starts at line 686 and calls `_build_visual_survival_configs` with `decision_ms=DEFAULT_DECISION_MS` at line 720.
- `_build_visual_survival_configs` in the train module accepts only `decision_ms` and writes default `decision_source_frames`, `source_physics_step_ms`, and `source_max_steps_semantics` into `env_cfg` at lines 4602-4605.

This is currently masked because `DEFAULT_DECISION_MS` now equals one source
physics frame. It is still a real pass-through bug: explicit cadence in the
background config does not control the actual eval/GIF env.

## Smallest Patch

Patch only the existing argument chain. Do not introduce a cadence object yet.

1. Extend `_build_visual_survival_configs`.

Add keyword args with defaults:

```python
decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES,
source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS,
source_max_steps_semantics: str = "source_physics_steps",
```

Use those values in `env_cfg` instead of the hardcoded defaults:

```python
"decision_source_frames": int(decision_source_frames),
"source_physics_step_ms": float(source_physics_step_ms),
"source_max_steps_semantics": str(source_max_steps_semantics),
```

Keep existing callers unchanged; defaults preserve behavior.

2. Extend `lightzero_curvytron_visual_survival_eval.py`'s eval env path.

Add the same three fields plus `decision_ms` to:

- `_make_policy_and_env`;
- `_eval_checkpoint`;
- `_run_eval`;
- `curvytron_visual_survival_eval_cpu`;
- `curvytron_visual_survival_eval_gpu`;
- `curvytron_visual_survival_eval_gpu_cpu40`;
- `main` and its `remote_kwargs`, if direct eval CLI parity is wanted in the same patch.

Then pass them down to `_build_visual_survival_configs`.

Minimum defaults:

```python
decision_ms: float = DEFAULT_DECISION_MS,
decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES,
source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS,
source_max_steps_semantics: str = "source_physics_steps",
```

The eval module currently imports `DEFAULT_DECISION_MS` but not
`DEFAULT_DECISION_SOURCE_FRAMES` or `DEFAULT_SOURCE_PHYSICS_STEP_MS`; add those
to the import from the train module.

3. Extend checkpoint eval/inspection remote wrapper path in the train module.

Add the four cadence parameters to:

- `_run_checkpoint_eval_and_inspect`;
- `lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect`.

In `_run_checkpoint_eval_and_inspect`, pass them into `eval_mod._run_eval(...)`.

In `_spawn_one_checkpoint_background_eval`, add spawn args:

```python
decision_ms=float(config.get("decision_ms", DEFAULT_DECISION_MS)),
decision_source_frames=int(config.get("decision_source_frames", DEFAULT_DECISION_SOURCE_FRAMES)),
source_physics_step_ms=float(config.get("source_physics_step_ms", DEFAULT_SOURCE_PHYSICS_STEP_MS)),
source_max_steps_semantics=str(config.get("source_max_steps_semantics", "source_physics_steps")),
```

4. Extend checkpoint self-play GIF path.

Add the four cadence parameters to:

- `_run_checkpoint_selfplay_gif`;
- `_capture_checkpoint_selfplay_gif_variant`;
- `lightzero_curvytron_visual_survival_checkpoint_selfplay_gif`.

In `_capture_checkpoint_selfplay_gif_variant`, pass them into
`eval_mod._make_policy_and_env(...)`.

In `_spawn_one_checkpoint_background_gif`, normalize `gif_config` with parent
fallbacks before building the request:

```python
gif_config["decision_ms"] = float(gif_config.get("decision_ms", config.get("decision_ms", DEFAULT_DECISION_MS)))
gif_config["decision_source_frames"] = int(gif_config.get("decision_source_frames", config.get("decision_source_frames", DEFAULT_DECISION_SOURCE_FRAMES)))
gif_config["source_physics_step_ms"] = float(gif_config.get("source_physics_step_ms", config.get("source_physics_step_ms", DEFAULT_SOURCE_PHYSICS_STEP_MS)))
gif_config["source_max_steps_semantics"] = str(gif_config.get("source_max_steps_semantics", config.get("source_max_steps_semantics", "source_physics_steps")))
```

Then pass those values to
`lightzero_curvytron_visual_survival_checkpoint_selfplay_gif.spawn(...)`.

5. Poller parity is part of the same smallest fix.

The poller rebuilds the same background config, so add the four fields to:

- `_checkpoint_eval_poller_command` at line 6240;
- `lightzero_curvytron_visual_survival_checkpoint_eval_poller` at line 8982;
- local poller spawn at `main`, lines 10067-10102;
- `_spawn_two_seat_checkpoint_poller` only if two-seat cadence should be explicit now. Otherwise leave it on defaults and document that two-seat background eval remains default-cadence.

For the stock fixed-opponent launcher, pass `decision_ms`, default
`decision_source_frames`, default `source_physics_step_ms`, and
`source_max_steps_semantics="source_physics_steps"` into the poller spawn.

## Regression Tests

All test changes fit in
`tests/test_curvytron_live_checkpoint_eval_plumbing.py`.

1. Tighten `test_live_checkpoint_trigger_spawns_eval_and_selfplay_gif_without_volume_commit`.

Add non-default sentinel cadence values to the test config:

```python
"decision_ms": 10.0,
"decision_source_frames": 2,
"source_physics_step_ms": 5.0,
"source_max_steps_semantics": "source_physics_steps",
```

and the same values under `selfplay_gif`.

Assert:

```python
assert eval_call["decision_ms"] == 10.0
assert eval_call["decision_source_frames"] == 2
assert eval_call["source_physics_step_ms"] == 5.0
assert eval_call["source_max_steps_semantics"] == "source_physics_steps"
assert gif_call["decision_ms"] == 10.0
assert gif_call["decision_source_frames"] == 2
assert gif_call["source_physics_step_ms"] == 5.0
assert gif_call["source_max_steps_semantics"] == "source_physics_steps"
```

Also assert `request["config"]` and `request["selfplay_gif"]["config"]` still
contain those values, proving the request JSON and function call match.

2. Tighten `test_checkpoint_eval_poller_completes_eval_inspection_and_selfplay_gif_jobs`.

Update the `_checkpoint_eval_poller_command(...)` call to pass sentinel cadence
values. Assert the fake eval and GIF spawn calls include the same four fields.
This catches the poller path, not only the hook-trigger path.

3. Tighten `test_local_launcher_passes_gif_config_to_poller_and_prints_enabled`.

Assert `fake_poller.calls[0]` includes:

```python
decision_ms == train_mod.DEFAULT_DECISION_MS
decision_source_frames == train_mod.DEFAULT_DECISION_SOURCE_FRAMES
source_physics_step_ms == train_mod.DEFAULT_SOURCE_PHYSICS_STEP_MS
source_max_steps_semantics == "source_physics_steps"
```

This protects the local `main(... background_eval_launch_kind="poller")` spawn
site at lines 10067-10102.

4. Add one env-builder pass-through test.

Monkeypatch `train_mod._build_visual_survival_configs` as already done in this
file, call `eval_mod._make_policy_and_env(...)` with sentinel cadence, and
assert the fake builder receives those exact values. Stub enough policy/env
surface machinery to avoid real LightZero work, or keep this as a narrow
function-level test around `_make_policy_and_env` if the existing fake LightZero
module helper is sufficient.

Target assertion:

```python
assert build_calls[0]["decision_ms"] == 10.0
assert build_calls[0]["decision_source_frames"] == 2
assert build_calls[0]["source_physics_step_ms"] == 5.0
assert build_calls[0]["source_max_steps_semantics"] == "source_physics_steps"
```

If stubbing `_make_policy_and_env` is too much for this patch, extend
`test_modal_config_defaults_to_one_source_frame_per_policy_action` to call
`_build_visual_survival_configs` with explicit `decision_source_frames=2` and
`source_physics_step_ms=5.0`, then assert `env_cfg` and `surface` reflect them.
That is weaker than testing `eval_mod._make_policy_and_env`, but still prevents
the builder from hardcoding defaults forever.

## Risk If Left Unfixed

- Background eval and GIF jobs can silently run with default cadence while their
  request config says a different cadence was requested.
- Eval tables and GIF summaries become misleading because they report cadence
  from the actual env info, not from the dropped config. A mismatch will look
  like the checkpoint underperformed or behaved differently, when the evaluator
  simply used a different time base.
- Future non-default cadence experiments will produce incomparable `steps`,
  `source_max_steps`, GIF frame stride, timeout, and survival metrics.
- The bug is easy to miss because current one-frame defaults make the happy path
  pass; only a sentinel pass-through regression test catches it.

## Verification Command

After implementing the patch, run:

```bash
uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py -q
```
