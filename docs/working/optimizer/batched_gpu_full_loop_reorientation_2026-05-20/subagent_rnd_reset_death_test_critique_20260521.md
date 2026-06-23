# Subagent RND Reset Death Test Critique

Date: 2026-05-21

Scope: read-only critique of the RND/reset/death validation surface. I did not
edit production code, touch live training runs, or promote a Coach-facing speed
claim.

Focused local check:

```text
uv run pytest -q -p no:cacheprovider tests/test_exploration_bonus.py tests/test_vector_reset.py tests/test_vector_autoreset.py
```

Result:

```text
49 passed, 4 skipped in 0.18s
```

## Plain Read

The local unit contracts are directionally good, but they do not yet prove the
batched GPU/profile lane is safe to summarize as Coach training speed.

The current RND implementation exposes the right observability hooks for a
small proof gate: default `rnd_update_per_collect=100`, device placement in the
reward model, train/estimate counters, predictor/target hashes, raw MSE stats,
and meter-mode target reward deltas. The main correctness worry is not absence
of hooks; it is that existing profile rows often use no-death/no-RND shapes, and
positive RND still uses batch min-max novelty.

The reset/autoreset layer has stronger local evidence than the speed lane:
terminal data is copied before reset mutation, autoreset validates
`done == terminated OR truncated`, and scalar LightZero env tests show terminal
`final_observation` survives manual reset. The missing proof is stock-boundary
coverage where a batched/profile manager has mixed live and terminal rows in
the same collect step.

Bottom line: keep no-death/no-RND profile rows as optimizer/Amdahl probes only.
Before a Coach-facing speed claim, require the three smallest gates below.

## Source Anchors

Key implementation anchors reviewed:

| Surface | Anchors |
| --- | --- |
| RND cadence/config | `src/curvyzero/training/exploration_bonus.py:26`, `:426`, `:789`, `:798` |
| RND device/metrics/hash | `src/curvyzero/training/exploration_bonus.py:560`, `:652`, `:692`, `:800`, `:812`, `:832` |
| RND batch-normalized novelty | `src/curvyzero/training/exploration_bonus.py:821`, `:844`, `:858` |
| RND tests | `tests/test_exploration_bonus.py:261`, `:400`, `:455` |
| Reset snapshot-before-mutation | `src/curvyzero/env/vector_reset.py:135`, `:147`, `:178` |
| Autoreset final-observation contract | `src/curvyzero/env/vector_autoreset.py:40`, `:115`, `:229`, `:236`, `:314` |
| Reset/autoreset tests | `tests/test_vector_reset.py:171`, `tests/test_vector_autoreset.py:117`, `:256`, `:517` |
| Scalar env final observation tests | `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py:401`, `:449` |
| Optimizer speed warnings | `actual_training_speed_read_20260521.md`, `subagent_validation_gate_review_20260521.md`, `subagent_amdahl_sanity_20260521.md`, `next_experiment_grid.md` |

## What Is Already Covered

### RND Config And Local Model

- `src/curvyzero/training/exploration_bonus.py` defaults
  `RND_DEFAULT_UPDATE_PER_COLLECT` to `100` and carries it into
  LightZero reward-model config as `update_per_collect`.
- `CurvyRNDRewardModel` stores `self.device`, moves the predictor/target module
  to that device, and moves train/estimate tensors with `.to(self.device)`.
- `train_with_data()` increments `train_with_data_calls`, skips small buffers,
  performs exactly `update_per_collect` optimizer steps, records predictor and
  target hashes around each train step, and increments `train_cnt_rnd`.
- `estimate()` extracts latest policy gray64 frames, records raw MSE stats, then
  batch-normalizes novelty with `(mse - mse.min()) / (mse.max() - mse.min() +
  1e-6)`. Meter mode keeps `intrinsic_reward_weight=0.0`, so reward targets
  should remain unchanged even though RND is estimated.
- `tests/test_exploration_bonus.py` covers fail-closed config, latest-frame
  extraction shapes, unnormalized input rejection, predictor-change/target-freeze
  behavior, seed determinism, update cadence, small-buffer metrics, meter-mode
  target identity, and the `disable_cudnn` metric flag.

### Reset, Autoreset, Final Observation

- `vector_reset.reset_arrays()` snapshots selected terminal rows before copying
  reset template rows and clearing `reset_pending`, `done`, `terminated`,
  `truncated`, and `terminal_reason`.
- `vector_autoreset.plan_autoreset_rows()` defaults to `done` rows, validates
  `done == terminated OR truncated`, requires final observation/reward metadata
  for selected rows, and stores a copied `final_transition_snapshot`.
- `vector_autoreset.apply_autoreset_rows()` composes that plan with
  `reset_arrays()`, returning both the final transition snapshot and reset
  snapshot.
- `tests/test_vector_reset.py` and `tests/test_vector_autoreset.py` cover
  terminal snapshots, copied metadata, mutation isolation, skipped rows, invalid
  masks, explicit non-done autoreset reporting, and missing selected-row
  terminal metadata.
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py` verifies
  the scalar LightZero env attaches terminal `final_observation`, zero action
  masks at terminal, terminal death metadata, and that saved final observation
  survives a later manual reset.

### Optimizer Docs Already Warn Correctly

- `actual_training_speed_read_20260521.md` separates actual Coach run speed
  from profile-only optimizer speed and says actual Coach speedup is not proven.
- `next_experiment_grid.md`, `subagent_validation_gate_review_20260521.md`, and
  `subagent_amdahl_sanity_20260521.md` all converge on the same missing gates:
  normal-death/autoreset, final observation, RND latest-frame extraction, and
  speed-row semantic identity.
- The RND planning docs explicitly block positive `rnd_replay_target_v0` as a
  recommendation because batch min-max novelty is not the desired serious
  intrinsic-normalization contract.

## Correctness Risks

### 1. RND Update Cadence Can Be Misread

The implementation is clear locally: every `train_with_data()` call runs
`update_per_collect` optimizer steps after the buffer reaches `batch_size`.
That does not by itself prove the stock collect/replay cadence used by a
batched profile row. A row can have `rnd_update_per_collect=100` configured but
still be invalid if `train_with_data_skipped_small_buffer_count` is nonzero
after warmup, `estimate_calls` outruns training badly, or the row used a
different denominator than its no-RND anchor.

Minimum proof: report `rnd_batch_size`, `rnd_update_per_collect`,
`collect_data_calls`, `train_with_data_calls`, `train_cnt_rnd`,
`estimate_calls`, `estimate_cnt_rnd`, `train_cnt_per_estimate`, and
`train_with_data_skipped_small_buffer_count`. Reject throughput reads where the
buffer never warmed or the cadence fields are absent.

### 2. RND Device Behavior Is Observable But Not Yet A Promotion Gate

The model and tensors move to `self.device`, and metrics report the device.
That is enough for a local CPU/CUDA comparison, but not enough for a batched GPU
manager claim. CUDA RND shares the same accelerator with render, policy,
learner, and hash/metrics readbacks. Predictor/target state hashing also pulls
tensors to CPU, which can synchronize the device.

Minimum proof: every RND row must report RND device, `disable_cudnn`, max GPU
memory/utilization, RND phase timers split into collect/train/estimate/metrics
or equivalent, and a matched no-RND row on the same compute/death/sim/manager
shape. Do not compare CPU RND to CUDA RND unless the device difference is the
point of the row.

### 3. Batch-Normalized Novelty Blocks Positive RND Claims

`estimate()` batch-normalizes current-batch prediction error to `[0,1]`. That
is acceptable for a meter overhead gate because `weight=0.0` keeps rewards
unchanged. It is not a stable curiosity signal: a sample's intrinsic reward
depends on the other samples in the same learner batch.

Minimum proof: positive `rnd_replay_target_v0` remains blocked from
recommendations until a running/global normalization contract and resume story
are selected. For now, only trust `rnd_meter_v0` for overhead and plumbing.

### 4. Final Observation Can Be Correct Locally And Wrong At The Boundary

The scalar env and reset/autoreset modules copy terminal observations before
reset. The profile boundary can still be wrong if it materializes terminal
timesteps after autoreset, filters terminal rows out before final-observation
attachment, or lets post-reset latest frames leak into RND input.

Minimum proof: a mixed normal-death stock-boundary row must have at least one
terminal physical row and one live physical row in the same batch. Terminal rows
must emit `done=true`, `final_observation_present=true`, terminal action masks,
nonzero `terminal_row_count`, nonzero `autoreset_row_count`, and
`terminal_before_autoreset=true`; live rows must remain ready after terminal
rows reset.

### 5. Profile-No-Death Amdahl Rows Are Not Normal-Death Training Claims

No-death/no-RND rows are useful because they expose long-survival observation,
stack, and scalarization costs without reset churn. They are not a proof that
normal death, terminal rewards, final observations, autoreset, and RND latest
frames survive a stock trainer boundary.

Minimum proof: speed tables must label `death_mode`, `profile_only`,
`calls_train_muzero`, `stock_lightzero_integrated`, RND mode, and denominator
source. Coach-facing summaries should separate:

- no-death/no-RND Amdahl evidence;
- normal-death correctness evidence;
- RND meter overhead evidence;
- actual Coach training speed.

## Smallest Proof Gates

### Gate A: Stock-Boundary RND Meter Gate

Purpose: prove RND is correct and measurable through the stock reward-model
entrypoint before interpreting RND overhead.

Shape:

```text
mode=profile
entrypoint=train_muzero_with_reward_model
env_manager_type=curvyzero_batched_profile
exploration_bonus.mode=rnd_meter_v0
exploration_bonus.weight=0.0
feature_source=policy_gray64_latest/v0
matched no-RND anchor
no eval/GIF/tournament/checkpoint promotion
fresh scratch run ids
```

Pass criteria:

- `called_train_muzero_with_reward_model=true` or equivalent entrypoint proof.
- `input_shape=[1,64,64]` and `source_observation_shape=[4,64,64]`.
- A sampled checksum or explicit marker proves RND latest frames equal the
  latest channel of the manager policy stack for the same env ids/players.
- `collect_data_calls > 0`, `train_with_data_calls > 0`, `estimate_calls > 0`.
- `train_cnt_rnd > 0`, finite `last_train_loss`, and no post-warmup small-buffer
  skips for throughput rows.
- Predictor hash changes after train; target hash stays unchanged.
- `last_target_reward_changed=false`,
  `last_target_reward_delta_abs_mean=0.0`, and
  `last_target_reward_delta_abs_max=0.0`.
- RND device and phase timing are recorded.

Fail rule: if any reward target delta is nonzero, if predictor hash does not
change, or if latest-frame source is unproven, the row is an invalid RND gate
regardless of speed.

### Gate B: Mixed Normal-Death Autoreset Gate

Purpose: prove terminal/final-observation/autoreset semantics through the same
boundary used for speed rows.

Shape:

```text
mode=profile
entrypoint=train_muzero
env_manager_type=curvyzero_batched_profile
death_mode=normal
same observation backend as speed candidate
one tiny fixed horizon
mixed live and terminal physical rows in one batch
```

Pass criteria:

- `profile_only=true`, no live-run writes, no checkpoint promotion.
- `death_mode=normal`, not `profile_no_death`.
- At least one terminal and one live physical row in the same batch.
- Terminal rows carry `done`, `terminated`/`truncated`, terminal reason, death
  metadata, terminal action mask, and `final_observation`.
- Autoreset happens after terminal timestep materialization.
- Live rows remain policy-ready and row/player mapping remains row-major.
- Terminal rows are not passed as policy/search roots after zero-mask filtering.
- `final_observation` shape/dtype/range matches policy stack expectations, and
  copied terminal frames do not mutate after reset.

Fail rule: missing final observation, post-reset final frame leakage, row/player
misalignment, or terminal zero-mask roots reaching policy/search is a hard
correctness fail.

### Gate C: Speed-Claim Attestation Gate

Purpose: prevent fast but semantically ambiguous rows from becoming Coach
guidance.

Every summarized speed row must include, or link to, these fields:

- profile identity: `profile_only`, `calls_train_muzero`,
  `stock_lightzero_integrated`, `touches_live_runs`, git SHA, compute;
- backend identity: env manager type, renderer backend, observation surface,
  stack dtype/range, no-hidden-fallback flag;
- denominator identity: env/source steps, MCTS roots, sim count, learner calls,
  replay samples, warmup policy, denominator source;
- death/reset identity: death mode, terminal row count, autoreset count,
  final-observation count/bytes, terminal-before-autoreset proof;
- RND identity when present: mode, weight, feature source, update cadence,
  train/estimate counters, predictor/target hashes, reward-target delta proof;
- workload controls: eval/GIF/tournament/checkpoint sidecars on/off.

Fail rule: if a row lacks these identity fields, it can stay in the experiment
log but should not appear in a Coach-facing speed recommendation.

## Recommendation

Do not add more broad renderer microbenchmarks to answer this question. The
smallest useful next work is:

1. Run Gate B once without RND.
2. Run Gate A with `rnd_meter_v0` plus a matched no-RND anchor.
3. Add Gate C as a summary/parser guard before publishing speed tables.

Until those pass, phrase the current state as:

```text
Profile-only no-death batched GPU rows are promising optimizer evidence.
Actual Coach training speedup is not proven.
Normal-death and RND stock-boundary semantics remain promotion gates.
Positive RND is blocked on intrinsic normalization and resume semantics.
```
