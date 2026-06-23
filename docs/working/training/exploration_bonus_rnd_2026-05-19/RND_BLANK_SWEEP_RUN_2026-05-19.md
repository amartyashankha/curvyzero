# RND Blank Sweep Run

Created: 2026-05-19.

## Purpose

Run the requested blank-canvas/no-tournament exploration-bonus sweep with GIFs
on. This is intentionally separate from the CZ26 tournament-grid builder.

Current status: historical launched diagnostic only. Do not use this as current
guidance for positive RND. Positive `rnd_replay_target_v0` is blocked pending
the intrinsic-normalization decision, so treat this weight ladder as invalidated
for recommendation purposes.

## Artifacts

- Manifest:
  `artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-sweep-20260519a/rnd-blank-sweep-20260519a.json`
- Submit dry run:
  `artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-sweep-20260519a/rnd-blank-sweep-20260519a.submit_dryrun.json`
- Submit launch:
  `artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-sweep-20260519a/rnd-blank-sweep-20260519a.submit_launch.json`
- Post-guard dry run:
  `artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-sweep-20260519a/rnd-blank-sweep-20260519a.submit_dryrun_after_guards.json`

## Launched Rows

All rows use:

- `env_variant=source_state_fixed_opponent`
- `reward_variant=survival_plus_bonus_no_outcome`
- `opponent_runtime_mode=blank_canvas_noop`
- `opponent_death_mode=normal`
- no `opponent_assignment_ref`
- `opponent_assignment_refresh_interval_train_iter=0`
- `own_checkpoint_opponent_refresh_enabled=false`
- `background_eval_enabled=true`
- `background_gif_enabled=true`
- `background_gif_max_steps=4096`
- `background_gif_frame_stride=4`
- tournament disabled in manifest metadata

Sweep:

| Row | Mode | Weight |
| --- | --- | ---: |
| r001 | `none` | 0.0 |
| r002 | `rnd_meter_v0` | 0.0 |
| r003 | `rnd_replay_target_v0` | 0.003 |
| r004 | `rnd_replay_target_v0` | 0.01 |
| r005 | `rnd_replay_target_v0` | 0.03 |
| r006 | `rnd_replay_target_v0` | 0.1 |
| r007 | `rnd_replay_target_v0` | 0.3 |
| r008 | `rnd_replay_target_v0` | 0.6 |
| r009 | `rnd_replay_target_v0` | 1.0 |

## Order Of Magnitude

Current `CurvyRNDRewardModel` normalizes RND prediction error to `[0, 1]` per
estimate batch before multiplying by `exploration_bonus_weight`.

For `survival_plus_bonus_no_outcome`, one ordinary alive reward is `1.0`, and a
same-step bonus catch adds `1.0`. Therefore:

- `0.003` is about 0.15% of the survival+bonus max step.
- `0.01` is about 0.5%.
- `0.03` is about 1.5%.
- `0.1` is about 5%.
- `0.3` is about 15%.
- `0.6` is about 30%.
- `1.0` is about 50%, and equal to one alive-reward unit.

The top rung is aggressive but still the same order of magnitude as one normal
reward component, not an unbounded scale jump.

Positive RND now also adjusts the LightZero target-support metadata by the
bounded intrinsic reward term. With the current default support cap this usually
does not increase the actual support size, but the requested reward/value support
scales include the intrinsic bound instead of silently ignoring it.

## Launch Proof

`submit_launch.json` records:

- `selected_row_count=9`
- `row_count=9`
- `assignment_write_count=0`
- `refresh_pointer_write_count=0`
- `training_candidate_refresh_config_record=null`
- every row has both train and poller function call IDs

This means the launch did not write opponent assignments, refresh pointers, or
training-candidate/tournament control config.

The launched run ids use the already-consumed `rnd-blank-current-*` namespace.
The manifest builder now defaults future run and attempt prefixes to the matrix
name so follow-up gates do not collide unless an operator explicitly reuses a
prefix.

The grouped submitter now refuses `--allow-launch` with a filtered subset
(`--limit`, `--row-id`, or `--row-kind`) unless `--allow-partial-launch` is
provided. This protects the exact failure mode where a stale `--limit 1` turns a
planned sweep into a single-row launch. Verified against this manifest:
`--allow-launch --limit 1` now raises `refusing partial launch: selected 1 of 9`.

## Fast Checkpoint Relaunch

The first live sweep used `save_ckpt_after_iter=10000`, which was too sparse for
the blank-canvas RND question. At the first status check, learners were already
around `5.8k-9.7k` train calls and still had only `iteration_0` checkpoints, so
the practical cadence was closer to ~20 minutes than the requested 5-10 minute
band.

Replacement matrix:

- manifest:
  `artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-sweep-fastckpt-20260519a/rnd-blank-sweep-fastckpt-20260519a.json`
- launch record:
  `artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-sweep-fastckpt-20260519a/rnd-blank-sweep-fastckpt-20260519a.submit_launch.json`
- rows: same 9-point ladder, `stock-w0`, `meter-w0`, then positive RND
  `0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0`
- `save_ckpt_after_iter=2500`
- `stop_after_learner_train_calls=80000`
- `max_train_iter=80000`
- `max_env_step=8000000`
- poller interval `60s`
- poller max runtime `18000s` (5 hours)
- GIFs on, tournament still off

Based on the first sweep's observed speed, `2500` learner calls should usually
land in the 5-10 minute checkpoint band, with the whole diagnostic run landing
in the few-hour range across faster and slower rows. The explicit learner-call
cap is the main run-length guard; `max_train_iter` and `max_env_step` remain as
secondary caps.

The old coarse sweep was cancelled by its recorded train and poller
`FunctionCall` IDs (`18/18` cancel requests accepted). This is intentionally
narrower than stopping the deployed trainer app.

Initial health check on the replacement sweep:

- all 9 train functions are `running`
- all 9 have `status_heartbeat_exists=true`
- all 9 are at stage `auto_resume_checked`
- `progress_latest.json` was not present yet at the first immediate poll, which
  is expected during startup before the first progress write

Follow-up cadence proof:

- all 9 rows have `progress_exists=true` and `iteration_0` checkpoints
- by the next poll, rows had reached roughly `2.0k-3.5k` learner train calls
- `iteration_2500` checkpoints already existed for `stock-w0`, `rnd-w0p01`,
  `rnd-w0p03`, `rnd-w0p1`, `rnd-w0p3`, and `rnd-w0p6`
- the remaining rows were slower but healthy, around `2.0k` learner calls
- background pollers were `running`
- eval manifests existed for all rows at `iteration_0`, and at least
  `stock-w0` and `rnd-w0p6` had `iteration_2500` evals
- GIF artifacts existed for all rows at `iteration_0`, and `rnd-w0p6` had an
  `iteration_2500` GIF artifact at the first curve-summary check

Long-sleep monitor update:

- active GIF browser URL:
  `https://modal-labs-shankha-dev--curvyzero-curvytron-gif-browser--f71ce8.modal.run/`
- the old `bada8e` browser URL returns 404; `f71ce8` is the current Modal web
  endpoint
- the browser was redeployed after adding
  `rnd-blank-sweep-fastckpt-20260519a-` to the current-batch prefix list, so the
  default page now lands on the fast-checkpoint sweep instead of only the old
  `rnd-blank-current-*` rows
- after a 30-minute sleep, all 9 train rows were still running with progress and
  heartbeats
- checkpoint spread was roughly `iteration_32500` through `iteration_52500`
- latest learner calls were roughly `33k` through `53k`
- checkpoint/eval/GIF artifacts were still flowing; latest eval/GIF checkpoints
  were roughly `32.5k-50k` depending on row
- no status `problem_count` was reported

Second long-sleep update:

- all 9 train rows were still `running`, with progress and heartbeats
- checkpoint spread advanced to roughly `iteration_40000` through
  `iteration_62500`
- latest learner calls were roughly `41k` through `64.6k`
- latest eval/GIF artifacts were still present and advancing; curve summary
  reported `17-26` eval points per row
- browser default page returned `200` and still resolved to
  `category=current`

Third long-sleep update:

- all 9 train rows were still `running`, with progress and heartbeats
- checkpoint spread advanced to roughly `iteration_47500` through
  `iteration_75000`
- latest learner calls were roughly `48.8k` through `76.2k`
- stock is close to the explicit `80k` learner-call cap
- curve summary reported `20-31` eval points per row and latest GIF checkpoints
  from `47.5k` through `75k`
- browser default page returned `200`

Cap-check update:

- stock completed at `79,999` learner calls with latest checkpoint
  `iteration_77500`
- stock poller status was `completed`, with eval/GIF artifacts through
  `iteration_77500`
- the other 8 rows were still `running`
- running checkpoint spread was roughly `iteration_52500` through
  `iteration_75000`
- running learner calls were roughly `53.8k` through `75.1k`
- browser default page returned `200`

## Current Remote Status

The older immediate-poll status where every row had only `iteration_0` is stale.
Later polls in this same document supersede it: the fast-checkpoint replacement
sweep reached trained checkpoints, evals, and GIFs well beyond `iteration_0`,
with stock completing around `79,999` learner calls and `iteration_77500`.

Use the long-sleep and cap-check sections above for the current run-history
read. The `iteration_0` artifacts still matter only as startup wiring proof, not
trained-policy evidence.

Live RND JSONL metrics were also checked directly from the runs volume:

- `meter-w0`: `estimate_calls` over 5k, `target_reward_changed=false`,
  target hash unchanged before/after train
- `rnd-w0p1`: `estimate_calls` over 5k, `target_reward_changed=true`,
  latest `last_target_reward_delta_abs_max` about `0.099`, matching weight `0.1`
  order of magnitude

The JSONL stream was valid, but the `rnd_reward_model_metrics_latest.json`
snapshot retrieved from the Modal volume was null-filled. The writer now writes
the latest snapshot directly instead of using temp-file replacement, and the
trainer-side metrics scanner falls back to the last valid JSONL event if the
latest snapshot is corrupt.

## Tests Run

Focused new/regression tests:

```text
uv run pytest tests/test_curvytron_rnd_blank_sweep_manifest.py
```

Result: `3 passed`.

After adding run-id collision coverage:

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_curvytron_rnd_blank_sweep_manifest.py -q
```

Result: `5 passed`.

Broader relevant suite:

```text
uv run pytest \
  tests/test_curvytron_rnd_blank_sweep_manifest.py \
  tests/test_exploration_bonus.py \
  tests/test_lightzero_config_builder.py \
  tests/test_reward_contracts.py \
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py \
  tests/test_curvytron_survivaldiag_submitter.py -q
```

Result after the submitter/RND/support/metrics-writer/metrics-scan guard
updates: `121 passed, 1 skipped`.

Additional compile check:

```text
python -m py_compile \
  scripts/submit_curvytron_survivaldiag_manifest.py \
  scripts/build_curvytron_rnd_blank_sweep_manifest.py \
  src/curvyzero/training/exploration_bonus.py \
  src/curvyzero/training/lightzero_config_builder.py
```

Result: passed.

## Known Risk

Positive `rnd_replay_target_v0` is currently blocked for recommendations. It
augments replay target rewards, lacks running RND normalization parity with the
OpenAI reference, and does not yet persist RND replay buffer contents across
resume. The current local model state exposes `state_dict`, and target-network
freezing is tested, but end-to-end resume of the reward-model sidecar still
needs a dedicated checkpoint round trip.

This sweep is one training seed across the full weight ladder. It is useful for
plumbing, gross scale, and first signal; it is not enough evidence to choose a
production exploration weight. Use `--replicas N` for seed robustness once the
first trained checkpoints and RND metrics look sane.
