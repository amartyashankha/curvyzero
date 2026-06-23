# Radical Search Validation Harness Plan - 2026-05-22

Scope: minimum validation harness before trusting any radical compact search
service, array-native CTree lane, or similar search rewrite. This is a
validation plan only. I did not touch live Coach runs.

Reviewed anchors:

- `radical_optimizer_trust_gates_20260522.md`
- `direct_ctree_promotion_gates_20260522.md`
- `subagent_validation_gap_review_20260522.md`
- `tests/test_lightzero_phase_profiler.py`
- `scripts/compare_curvytron_direct_ctree_stock.py`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `scripts/summarize_curvytron_optimizer_profile_results.py`

## Plain Rule

Do not trust a faster search lane until it proves two things separately:

1. It gives the same training data as the stock path for deterministic cases.
2. Its speed claim is measured against the same stock `train_muzero`
   denominator, with the same sidecars, death mode, RND mode, and observation
   contract.

Exact action parity on neutral MCTS rows is not the right gate. Neutral rows can
tie, and stock CTree can split visits differently without changing the training
meaning. Use exact tests for forced semantics and statistical tests for random
or tie-heavy behavior.

## Current Baseline

Already present in `tests/test_lightzero_phase_profiler.py`:

- direct hook install/restore and stock fallback behavior;
- all-actions-legal schema smoke;
- mixed-mask and single-legal raw visit contract;
- bad-mask fail-closed tests;
- all-actions-legal stochastic fast-path distribution smoke;
- compact summary fallback accounting.

Those tests are useful, but they validate the current profile hook. A radical
array-native or service lane needs a shared harness that checks the consumer
boundary too: what LightZero stores for replay and target building.

## Minimum P0 Harness

### P0. Candidate Contract Adapter

Create one small test adapter interface for each candidate lane:

```text
stock_collect(fixture) -> collect_output
candidate_collect(fixture) -> collect_output, self_audit
```

For a service lane, the candidate can be an in-process deterministic toy
service first. It does not need queues or Modal for the first gate. It does need
request ids, env ids, player ids, masks, actions, values, rewards, and visit
lists.

Exact checks:

- same outer env ids as stock;
- same field names as stock for a stock-compatible lane;
- explicit `replacement_trainer=true` if it is not stock-compatible;
- no hidden fallback when the row claims candidate speed;
- self-audit says implementation name, root count, sim count, mask mode,
  fallback count, and output row count.

### P0. Forced Mask And Raw Visit Contract

Use deterministic fake roots/model outputs with masks:

```text
all legal, clear preference
single legal action 0
single legal action 1
single legal action 2
mixed masks: [1,0,1], [0,1,0], [1,1,0]
```

Exact checks:

- chosen action is a full action id, not a legal-list index;
- chosen action is legal;
- illegal visit mass is zero;
- raw stock `visit_count_distributions` length/order match the legal action
  list;
- candidate values, predicted values, policy logits, searched values, and
  entropy match stock under the same fake outputs;
- fractional masks, empty masks, and mask-length mismatch fail closed before
  model/search.

This is the first gate because it catches the easiest silent target poison.

### P0. Replay/Target Consumer Canary

Feed stock and candidate collect outputs into the same small consumer path. Use
the real LightZero/GameSegment storage path if practical; otherwise use the
smallest adapter that calls the same `store_search_stats`-style contract.

Exact checks:

- action;
- raw visit counts;
- searched value/root value;
- predicted value and policy logits if carried;
- reward;
- done;
- `to_play`;
- action mask;
- env id/player id order;
- final observation metadata for terminal rows.

The test should fail if a collect dict has the right shape but trains different
targets.

### P0. Row/Column Sentinel

Use deliberately non-identity indices:

```text
env ids:        [17, 3, 42]
player ids:     [1, 0, 1]
latent paths:   [2, 0, 1]
batch indices:  [1, 2, 0]
last actions:   [2, 0, 1]
```

Make recurrent outputs encode `(path, batch, action)`.

Exact checks:

- candidate gathers the intended latent row;
- candidate passes the intended last action;
- backprop receives the intended reward/value/logit row;
- output returns to the original env/player order;
- service responses cannot be matched by arrival order alone; they must match
  request ids.

This is mandatory for GPU-latent, array-native, and service designs because
row swaps can look fast and still train garbage.

### P0. Normal Death And Autoreset Canary

Use a small batch with one live row and one row that dies.

Exact checks:

- terminal row is not searched as a live root after death;
- terminal reward, done flag, death player/cause, and `to_play` are correct;
- final observation is from before autoreset;
- autoreset observation is used only for the next episode;
- live rows remain live and keep their order;
- terminal rows do not create zero-mask searches unless the upstream contract
  explicitly filters and counts them.

No-death speed rows are not enough for trust. They are useful for throughput,
not for terminal semantics.

### P0. RND Meter Safety Canary

Run the same local fixture with RND off and with `rnd_meter_v0`.

Exact checks for meter mode:

- reward-model entrypoint is used;
- predictor hash changes after training;
- target hash stays frozen;
- target training rewards are unchanged;
- RND latest-frame source matches the same env/player row used for replay;
- terminal/autoreset latest-frame rows are labeled correctly;
- `collect_data_calls`, `train_with_data_calls`, `estimate_calls`,
  `train_cnt_rnd`, and `estimate_cnt_rnd` are positive after warmup.

Statistical checks:

- RND overhead versus matched no-RND row;
- predictor loss/novelty behavior.

Meter mode does not prove positive-RND learning. It only proves plumbing and
overhead without changing target rewards.

### P0. Profile Attestation Reject Gate

Any profile row used in a speed table must pass the summary attestation gate.

Exact required fields:

- `called_train_muzero=true`;
- `mode=profile`;
- backend name;
- fallback calls;
- output rows;
- env steps source;
- death mode;
- RND mode and cadence;
- observation contract;
- sidecar/eval/GIF/checkpoint state;
- root count and sim count;
- compile/service/array-native status if applicable.

For a direct or radical backend:

```text
fallback_calls == 0
candidate_calls > 0
output_rows > 0
```

Rows that fail this can stay in experiment logs, but not in Coach advice.

## P1 Harness

### P1. Root Noise And Stochastic Selection

Use many seeds. Do not require exact per-seed actions.

Statistical checks:

- empirical action frequencies close to stock;
- zero illegal actions;
- zero illegal visit mass;
- finite entropies;
- legal-action-only Dirichlet noise under partial masks.

### P1. Neutral MCTS Distribution Gate

For tie-heavy or near-neutral rows, compare distributions rather than exact
actions.

Statistical checks:

- action frequency over seeds;
- visit L1 threshold;
- searched value tolerance;
- no illegal actions;
- same predicted values/logits under same model.

Plain trap to avoid: exact action equality on tied rows will reject valid
implementations for the wrong reason.

### P1. Long-Trajectory Surface Canary

Run longer no-death local/profile rows only after P0 semantics pass.

Check:

- latest-frame stack order;
- row/player perspective;
- bonus symbol surface;
- trail/render cache labels;
- observation checksum contract;
- throughput after warmup.

This catches denominator shifts that short bad-policy rows hide.

### P1. Search Service Queue And Freshness Gate

Only for a real search service.

Check:

- offered roots;
- accepted roots;
- terminal roots;
- dropped roots;
- queue depth;
- wait time;
- recurrent batch size;
- model version per request;
- policy lag;
- replay row id;
- retry/crash behavior.

Exact where ids and counts define semantics. Statistical where queue timing is
noisy.

### P1. Matched Full-Loop A/B

After P0 passes, run matched profile-only rows:

```text
stock train_muzero
candidate train_muzero
same seed
same collector count
same sim count
same death mode
same RND mode
same checkpoint/eval/GIF sidecar settings
same warmup and measured window
```

Require at least:

- two no-RND repeats;
- one `rnd_meter_v0` repeat;
- one normal-death semantic row;
- fallback calls zero.

Throughput is statistical. The row is trusted only if the semantic gates are
exact and the speed win survives repeats.

## How To Compare Against Stock

Use stock in four layers:

1. Stock collect output: compare dict fields and forced actions.
2. Stock consumer output: compare what replay/target code actually stores.
3. Stock profile summary: compare semantic identity and attestation fields.
4. Stock full-loop timing: compare matched profile rows only after the first
   three layers pass.

Root noise should be off for exact gates:

```text
root_noise_weight=0.0
epsilon=0.0
deterministic collect where possible
clear visit/logit preference
```

Then turn root noise and stochastic selection on for P1 distribution checks.

## What Is Exact Versus Statistical

Exact:

- schema;
- env id/player id order;
- `to_play`;
- forced action masks;
- full action ids;
- illegal action count;
- illegal visit mass;
- raw legal-action visit-list length/order;
- same-model values/logits/rewards;
- terminal row accounting;
- final observation before autoreset;
- RND meter safety invariants;
- fallback count and attestation fields.

Statistical:

- neutral/tie-heavy action agreement;
- stochastic selection;
- root-noise visit distributions;
- throughput;
- queue timing;
- positive RND learning strength.

## Minimum Implementation Shape

Keep this small:

1. Add one shared fixture module for deterministic observations, masks,
   fake model outputs, terminal rows, and RND latest-frame sentinels.
2. Add one test module for the candidate contract adapter and consumer canary.
3. Extend the existing profile summarizer tests for radical backend
   attestation fields.
4. Reuse `compare_curvytron_direct_ctree_stock.py` for optional P1
   distribution checks, but do not let it replace the P0 consumer canary.

Suggested names:

```text
tests/search_validation_fixtures.py
tests/test_radical_search_validation_contract.py
```

## Promotion Decision

A radical search lane can be called "trusted enough to recommend to Coach" only
after:

- all P0 tests pass;
- no hidden fallback is possible in the recommended row;
- profile attestation passes;
- matched stock/candidate full-loop rows show stable speed;
- RND and normal death/autoreset are either tested or explicitly out of scope
  for that recommendation.

Until then, the honest label is:

```text
optimizer probe, not Coach-facing training default
```
