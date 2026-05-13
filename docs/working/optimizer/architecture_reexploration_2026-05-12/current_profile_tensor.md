# Current Profile Tensor

Date: 2026-05-12

Purpose: keep the next Optimizer profiling wave structured. These rows are
measurement jobs for the trusted stock LightZero path, not learning claims.

## Current Truth

Use this path:

```text
stock LightZero train_muzero/profile
env_variant=source_state_fixed_opponent
opponent_policy_kind=frozen_lightzero_checkpoint
opponent_use_cuda=false
```

Do not use the old custom `two-seat-selfplay` path for new speed evidence.

## Generated Artifacts

Manifest id:

```text
opt-stock-frozen-profile-first-wave-20260512e
```

Files:

```text
artifacts/local/curvytron_optimizer_profile_manifests/opt-stock-frozen-profile-first-wave-20260512e.json
artifacts/local/curvytron_optimizer_profile_manifests/opt-stock-frozen-profile-first-wave-20260512e.commands.jsonl
artifacts/local/curvytron_optimizer_profile_manifests/opt-stock-frozen-profile-first-wave-20260512e.commands.sh
```

Builder:

```text
scripts/build_curvytron_optimizer_profile_manifest.py
```

Validation so far:

- `python3 -m py_compile scripts/build_curvytron_optimizer_profile_manifest.py`
- `uv run ruff check scripts/build_curvytron_optimizer_profile_manifest.py`
- Manifest generation completed with 17 rows.

Correction: the first `20260512a` manifest was too loose for validation because
it used very high train/env limits and no explicit learner-call stop. The six
brief `20260512a` validation wrappers were stopped before widening.

Correction: `20260512b` added the learner-call cap and completed useful runs,
but summary files did not reliably appear in the Modal volume for every row.

Correction: `20260512c` proved the six validation rows, but final volume commits
were too expensive and one commit reported an error. Do not make wide profile
grids depend on final profile volume commits.

Correction: `20260512d` launched profile rows with `--profile-spawn` from a
non-detached parent `modal run`. The parent app shut down and the child function
call returned an empty `RemoteError`.

Current direction: `20260512e` launches profile rows with both parent
`modal run --detach` and child `--profile-spawn`. Metrics are read from the
returned Modal function result. The main run volume summary remains best-effort
archival data, not the source of truth for profile tables.

The capped rows use:

```text
max_train_iter=64
max_env_step=65536
stop_after_learner_train_calls=10
save_ckpt_after_iter=9999
```

## First Wave Rows

| row | family | hardware | manager | collectors | episodes | batch | sims | death | render | reward | status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 01 | `anatomy_base` | `gpu-l4-t4` | `base` | 1 | 1 | 16 | 8 | `normal` | `browser_lines` | `sparse_outcome` | `complete` |
| 02 | `anatomy_base` | `gpu-l4-t4` | `base` | 1 | 1 | 16 | 8 | `nodeath` | `browser_lines` | `sparse_outcome` | `complete` |
| 03 | `anatomy_base` | `gpu-l4-t4` | `base` | 1 | 1 | 16 | 8 | `nodeath` | `body_circles_fast` | `sparse_outcome` | `complete` |
| 04 | `collector_width` | `gpu-l4-t4-cpu40` | `subprocess` | 8 | 8 | 16 | 8 | `normal` | `browser_lines` | `sparse_outcome` | `complete` |
| 05 | `collector_width` | `gpu-l4-t4-cpu40` | `subprocess` | 16 | 16 | 16 | 8 | `normal` | `browser_lines` | `sparse_outcome` | `complete` |
| 06 | `collector_width` | `gpu-l4-t4-cpu40` | `subprocess` | 32 | 32 | 16 | 8 | `normal` | `browser_lines` | `sparse_outcome` | `complete` |
| 07 | `collector_width` | `gpu-l4-t4-cpu40` | `subprocess` | 64 | 64 | 16 | 8 | `normal` | `browser_lines` | `sparse_outcome` | `complete` |
| 08 | `collector_width` | `gpu-l4-t4-cpu40` | `subprocess` | 96 | 96 | 16 | 8 | `normal` | `browser_lines` | `sparse_outcome` | `complete` |
| 09 | `sim_ladder` | `gpu-l4-t4-cpu40` | `subprocess` | 32 | 32 | 16 | 4 | `normal` | `browser_lines` | `sparse_outcome` | `complete` |
| 10 | `sim_ladder` | `gpu-l4-t4-cpu40` | `subprocess` | 32 | 32 | 16 | 16 | `normal` | `browser_lines` | `sparse_outcome` | `complete` |
| 11 | `sim_ladder` | `gpu-l4-t4-cpu40` | `subprocess` | 32 | 32 | 16 | 32 | `normal` | `browser_lines` | `sparse_outcome` | `complete` |
| 12 | `long_render_lens` | `gpu-l4-t4-cpu40` | `subprocess` | 16 | 16 | 16 | 8 | `nodeath` | `browser_lines` | `sparse_outcome` | `complete` |
| 13 | `long_render_lens` | `gpu-l4-t4-cpu40` | `subprocess` | 16 | 16 | 16 | 8 | `nodeath` | `body_circles_fast` | `sparse_outcome` | `complete` |
| 14 | `hardware_ladder` | `gpu-l4-t4-cpu40` | `subprocess` | 64 | 64 | 16 | 16 | `normal` | `browser_lines` | `sparse_outcome` | `complete` |
| 15 | `hardware_ladder` | `gpu-h100-cpu40` | `subprocess` | 64 | 64 | 16 | 16 | `normal` | `browser_lines` | `sparse_outcome` | `planned` |
| 16 | `hardware_ladder` | `gpu-h100-cpu40` | `subprocess` | 64 | 64 | 16 | 32 | `normal` | `browser_lines` | `sparse_outcome` | `planned` |
| 17 | `reward_lens` | `gpu-l4-t4-cpu40` | `subprocess` | 32 | 32 | 16 | 8 | `normal` | `browser_lines` | `dense_survival_plus_outcome` | `complete` |

## Launch Policy

Do not launch the whole tensor until a validation slice proves that command
surface, result refs, and summarization still work.

Validation slice:

- Row 01: base manager, normal death, browser render. This is the attribution
  row.
- Row 02: base manager, no death, browser render. This is the long-trajectory
  attribution row.
- Row 03: base manager, no death, fast render. This is the fast-render
  attribution row.
- Row 04: subprocess manager, 8 workers. This is the first scale row.
- Row 06: subprocess manager, 32 workers. This is the practical scale anchor.
- Row 12: subprocess manager, no death, browser render. This is the long
  trajectory render lens.

If those rows finish and summarize cleanly, launch rows 05-17 as the first
parallel wave. If any validation row fails at command or summary level, fix the
surface before widening.

Completed validation launch:

```text
rows: 01, 02, 03, 04, 06, 12
mode: non-detached Modal calls from the generated `20260512c` manifest commands
      with `--detach` removed
status: `20260512c` rows 01, 02, 03, 04, 06, 12 completed and summarized;
        results are in `profile_validation_results.md`
```

Current readback validation:

```text
row: 01 from `20260512d`
mode: `--profile-spawn` without parent `--detach`
status: failed with empty remote errors because the parent app stopped
```

Current readback validation:

```text
row: 01 from `20260512e`
mode: parent `modal run --detach` plus child `--profile-spawn`
source of truth: `modal.FunctionCall.from_id(function_call_id).get()`
status: passed
result path: artifacts/local/curvytron_optimizer_profile_results/opt-stock-frozen-profile-first-wave-20260512e/row_01_result.json
steps/wall: 69 steps in 12.93 sec, 5.34 steps/s
```

## How To Read Results

Base-manager profiles are the attribution lens: they expose env step, render,
stack, frozen-opponent, search, and learner buckets in one process.

Subprocess profiles are the scaling lens: they show end-to-end throughput under
many env workers, but detailed env/render/opponent time is hidden inside worker
processes unless sampled worker telemetry records it.

Plain Amdahl read:

- If row 01 says render/opponent/env dominates, bigger GPUs will not save the
  stock loop until that CPU work is reduced or parallelized.
- If rows 04-08 scale well, more CPU workers and actor fanout are the next big
  lever.
- If rows 09-11 make MCTS dominate and GPU use rises, search batching or MCTX
  becomes worth a focused experiment.
- If rows 12-13 diverge strongly, render fidelity is a real speed knob for long
  trajectories.
