# Prelaunch Validation Round 2 - 2026-05-13

Scope: local validation only. No Modal long runs and no large matrix launch.

## Commands Run

```bash
uv run pytest tests/test_curvyzero_source_state_visual_survival_lightzero_env.py tests/test_vector_multiplayer_env.py tests/test_vector_runtime.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_curvytron_gif_browser.py tests/test_eval_curves.py
```

Result: passed, `291 passed, 10 skipped`.

```bash
uv run pytest tests/test_vector_*.py tests/test_curvyzero_*lightzero*.py tests/test_curvytron_*.py tests/test_eval_curves.py tests/test_benchmark_debug_visual_lightzero_adapter.py tests/test_multiplayer_ego_lightzero_coach_smoke.py
```

Result: passed, `526 passed, 14 skipped`.

```bash
uv run python scripts/build_curvytron_stock_train_manifest.py --stdout-only
```

Result: expected fail/pass guard, exit code `1`. The generator refused to emit
the historical stock manifest by default and named the missing current
survivaldiag schema pieces: `survival_plus_bonus_no_outcome`,
`blank_canvas_noop`, separated seed/copy fields, render pairs, and explicit
stochasticity.

```bash
uv run python scripts/build_curvytron_stock_train_manifest.py --stdout-only --allow-historical-matrix --matrix-name stock-control-v1
```

Result: passed. Output remained `dry_run_only=true`,
`historical_only=true`, `current_launch_approved=false`,
`launches_modal=false`, with guard metadata marking this as a stale
reward/opponent/seed schema for historical audit only.

```bash
uv run python scripts/analyze_curvytron_eval_curves.py --help
```

Result: passed. CLI exposes local curve scoring with multi-metric support.

```bash
printf '%s\n' '{"rows":[{"short_name":"local-survivaldiag","attempt_id":"attempt-local","eval_checkpoints":[{"checkpoint":"iteration_0","seeds":2,"mean_steps":8,"outcome_histogram":{"loss":2},"action_summary":{"top_action_fraction":0.5,"collapsed":false},"mean_training_reward":0.8,"mean_bonus_pickup_count":0,"mean_bonus_reward":0,"ok_count":2,"failure_count":0},{"checkpoint":"iteration_10","seeds":2,"mean_steps":16,"outcome_histogram":{"cap":2},"action_summary":{"top_action_fraction":0.98,"collapsed":true},"mean_training_reward":1.7,"mean_bonus_pickup_count":1,"mean_bonus_reward":0.1,"ok_count":1,"failure_count":1}]}]}' > /private/tmp/curvytron_eval_curve_snapshot.json
uv run python scripts/analyze_curvytron_eval_curves.py /private/tmp/curvytron_eval_curve_snapshot.json --metric mean_survival,mean_training_reward,bonus_pickup_count,failure_rate --format markdown
```

Result: passed. The tool produced rows for `mean_survival`,
`mean_training_reward`, `bonus_pickup_count`, and `failure_rate`, with action
collapse and `has_failures` health surfaced from the status-style checkpoint
payload.

## Feature Evidence

- `survival_plus_bonus_no_outcome`: focused tests cover survival reward without
  catch, same-step bonus pickup reward, and terminal outcome exclusion.
- `blank_canvas_noop`: focused tests cover player 1 public lifecycle,
  observation/render hiding, no movement/trail/collision/bonus side effects,
  seeded player 1 body scrubbing, and ego wall death still working.
- Action repeat flags: focused tests cover one LightZero policy transition with
  multiple physical steps, reward accumulation, and Modal config pass-through
  for `policy_action_repeat_min`, `policy_action_repeat_max`, and
  `policy_action_repeat_extra_probability`.
- Eval/status rich export: local status rollup tests preserve reward,
  bonus, terminal-cause histogram, action histogram/entropy, failure rate, eval
  health, train action observability, poller, and GIF fields.
- Eval curve tooling: tests and CLI smoke cover status-style
  `eval_checkpoints`, multiple metrics, collapse flags, reward/survival readout,
  and eval health.
- Stale manifest guard: local CLI validation confirms the current generator is
  blocked by default and only emits historical review artifacts behind
  `--allow-historical-matrix`.

## Gate Read

Cleared locally by this round:

- local survival plus bonus reward behavior;
- exact reward variant/config support sizing path in tests;
- blank-canvas no-op behavior and render masking;
- stock action-repeat config/telemetry semantics;
- checkpoint/status/eval/GIF discovery surfaces covered by local tests;
- eval-curve tooling can read local survivaldiag-style snapshots;
- stale historical manifest generator is guarded from accidental default use.

Still not cleared by local validation alone:

- high-cap `source_max_steps=65536` runtime canary;
- superseded by round 3: the dry-run survivaldiag manifest shape is locally
  cleared, emits 50 executable review rows plus 10 gated specs, and remains
  launch-disabled;
- real upstream volume/status export from a fresh canary containing all rich
  fields;
- scripted/random/checkpoint opponent rows beyond their existing design notes
  and local plumbing tests.

Verdict: local feature validation is green. Round 3 cleared the local dry-run
manifest shape; the large survivaldiag launch remains blocked on live high-cap
and rich-status/readout gates.
