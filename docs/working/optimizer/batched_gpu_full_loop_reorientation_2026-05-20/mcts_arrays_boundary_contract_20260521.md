# MCTS Arrays Boundary Contract

Date: 2026-05-21

Status: profile-only facade and direct CTree arrays probes implemented and
smoke-tested. This is not trainer code and not Coach launch advice.

## Why This Exists

The latest profile rows say the public LightZero MCTS collect branch is the
current wall.

```text
H100 public MCTS collect sim8:    ~2572 roots/sec
H100 array ceiling recurrent_toy: ~8681 roots/sec
```

That ceiling row is not real MCTS, but it does enough real MuZero model work to
show that the public MCTS branch boundary has removable overhead. A real
replacement must keep compact arrays in and compact arrays out while preserving
MuZero semantics.

## Minimal Input Contract

```text
search_arrays(
  obs: float32 or uint8 tensor [N, C, H, W],
  action_mask: bool/int [N, A],     # 1 means legal
  to_play: int[N],                  # fixed-opponent lane is -1
  ready_env_id: int[N],
  mode: collect | eval,
  temperature: float,
  epsilon: float,
  rng_state,
  config
)
```

Config must include:

```text
num_simulations
discount_factor
pb_c_base
pb_c_init
root_dirichlet_alpha
root_noise_weight
value_delta_max
value/reward support transform settings
```

## Minimal Output Contract

```text
{
  action: int[N],
  visit_count_distributions: float[N, A],
  searched_value: float[N],
  predicted_value: float[N],
  predicted_policy_logits: float[N, A],
}
```

Output must be compact arrays. Do not fan out into one Python dict per root
inside the hot path unless a validation or compatibility edge explicitly needs
that.

## Semantics That Must Not Change

- PUCT selection with the same `pb_c_base`, `pb_c_init`, priors, visit counts,
  Q/value normalization, `value_delta_max`, and discount.
- Legal masks: action ids must map back to full action ids, not legal-list
  indexes.
- Root noise: collect mode uses Dirichlet root noise; eval mode does not.
- Value and reward support transforms before backup.
- Batched recurrent inference for leaf expansion.
- Visit-count policy target. Epsilon can change the executed action, but it
  must not rewrite the stored visit distribution.
- Fixed-opponent lane uses `to_play=-1` and non-board-game value semantics.
- Randomness must be controlled for root noise, epsilon, temperature sampling,
  reset seeds, learner-seat selection, opponent selection, and action-repeat
  stochasticity.
- Action masks are binary `0/1`. Stock LightZero legal-action construction uses
  equality with `1`, so fractional masks are not a valid stock parity case.

## Smallest Validation Gates

1. Mask/action parity: fixed logits and masks where each action is illegal in
   turn. Replacement and LightZero must never choose an illegal action.
2. Eval parity: no noise, deterministic temperature, small `N`, simulations
   `1`, `2`, and `8`; compare actions, visits, root values, and logits.
3. Collect RNG parity: fixed seeds, root noise on, temperature fixed; compare
   visit distributions and sampled actions.
4. Support-transform parity: synthetic value/reward supports must invert to the
   same scalar values LightZero backs up.
5. Replay target parity: one tiny env segment must store the same action,
   reward, done, `to_play`, root value, and child-visit segment.
6. Seat/player canary: player 0, player 1, and random learner-seat episodes
   preserve observation perspective, reward perspective, selected action, and
   opponent metadata.
7. Joint-action canary: centralized 9-action lane decodes scalar action and
   joint mask the same way as the current wrapper.

## Current Non-Negotiable Caveat

Do not build a faster policy sampler and call it MuZero. The speed probe can be
fast because it skips tree semantics. The production path only matters if it
keeps the training target semantics intact.

This boundary may only replace the hot path after fixed-seed parity against
stock LightZero passes. Until then it is profile-only and must not feed Coach
launch advice or production training defaults.

## Implemented Path So Far

The first implementation started with a profile-only arrays facade over stock
LightZero MCTS, not a new tree.

1. Add a stack probe beside the existing collect-forward probe that accepts the
   same pre-scalar `[B,2,4,64,64]` stack and action masks.
2. Call stock `policy.collect_mode.forward` or `policy.eval_mode.forward`
   internally so LightZero still owns MCTS semantics.
3. Decode the stock result into compact arrays:
   `action`, `visit_count_distributions`, `searched_value`,
   `predicted_value`, and `predicted_policy_logits` where available.
4. Compare this facade against the existing collect-forward probe on fixed
   masks and small simulation counts before any attempt to bypass
   `collect_mode.forward`.

This step is useful even though it is not faster yet. It creates the production
shape we want: compact arrays at the boundary, LightZero semantics preserved,
and a clean place to replace the internal stock call only after parity gates
exist.

## 2026-05-21 Direct CTree Arrays Result

The second profile-only probe now bypasses public
`policy.collect_mode.forward`. It still uses the real LightZero MuZero model and
real `policy._mcts_collect.search` / CTree MCTS, then packs compact arrays.

Same shape:

```text
B512 physical rows
1024 player-view roots per call
25 measured steps, 5 warmup steps
sim8
uint8 [B,2,4,64,64] stack
scalar timestep materialization off
```

| row | run id | scalar roots/sec | main read |
| --- | --- | ---: | --- |
| H100 stock facade | `ap-HJk70PQP2iLAvA7mxxn99u` | `2419.81` | public collect path plus compact decode |
| H100 direct CTree arrays | `ap-XEoAIwCpbbQTuFLmSnjvwY` | `2806.64` | real direct CTree path, but output assembly was huge |
| H100 direct CTree arrays fast path | `ap-XEB8GF9B2Gw5V600QVtu10` | `3859.44` | all-actions-legal vectorized output assembly |
| H100 direct CTree arrays, current host uint8 | `ap-DoCqvAulFMhZyoAcownQmn` | `5247.95` | matched current-image host input row |
| H100 direct CTree arrays, current pinned uint8 | `ap-APSw7b1ZSJjSSuPtGEHO3w` | `4678.23` | lower H2D, but slower total wall than current host uint8 in short row |
| H100 direct CTree arrays, resident reuse ceiling | `ap-KCtqhJDwTuLptLKd4XSv38` | `5820.96` | upper-bound only; reuses stale input |
| L4/T4 direct CTree arrays | `ap-5OB4ye6HKiGfPQ3UjP221v` | `1460.41` | cheaper GPU sanity row |

The first direct row found the next dumb bottleneck:

```text
H100 direct output assembly before fast path: 4.709s / 25 measured steps
H100 direct output assembly after fast path:  0.027s / 25 measured steps
```

Plain read:

```text
The direct CTree profile lane is a real speed win over the stock facade, but it
is not yet a trainer recommendation. The best short current H100 row is about
2.17x faster than the stock facade. After output assembly was fixed and input
modes were priced, the remaining wall is MCTS search/root prep/model/output and
ordinary observation/stack work around the real CTree call, not compact output
packing. Pinned input lowered H2D but is not yet a proven total-wall win.
```

Longer input-mode repeat:

| row | run id | scalar roots/sec | main read |
| --- | --- | ---: | --- |
| H100 direct CTree arrays, host uint8, 60/15 | `ap-QPLEHOs3dGrcs2tlRpbMge` | `4111.80` | fresh host stack copied each step |
| H100 direct CTree arrays, pinned uint8, 60/15 | `ap-5F1tMU2HiuHXDcu4O1tGkw` | `4513.15` | stable modest total-wall win; H2D fell about `1.21s -> 0.14s` |
| H100 direct CTree arrays, resident reuse, 60/15 | `ap-wsKyodSayU2KGsTgKKpAqc` | `5537.40` | stale-input upper bound only |

Plain correction: pinned input is now a likely low-risk transfer improvement in
the profile lane, but it is only about a 10% total-wall win in the longer row.
It does not change the main blocker: direct-vs-stock fixed-seed parity and
remaining search/root-prep/model/output accounting.

Fresh current-telemetry repeat, same H100 B512/A16/sim8 60/15 shape:

| row | scalar roots/sec | H2D | search | root prep | model total | model-output D2H | observation | main read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stock facade, `host_uint8` | `2473.11` | `1.04s` | stock wrapper | n/a | `0.96s` | n/a | `2.08s` | public compact-facade anchor |
| direct CTree, `host_uint8` | `4564.03` | `1.14s` | `6.26s` | `0.49s` | `0.83s` | `0.10s` | `2.23s` | best fresh direct row |
| direct CTree, `host_uint8_pinned` | `4113.52` | `0.05s` | `8.11s` | `0.53s` | `1.09s` | `0.12s` | `2.46s` | H2D win, total-wall loss in this refresh |
| direct CTree, `resident_torch_reuse` | `4884.69` | `0.00s` | `6.50s` | `0.46s` | `0.90s` | `0.13s` | `2.21s` | stale-input ceiling only |

Plain correction after the fresh telemetry row: pinned input is a useful
transfer probe, but not yet a total-wall recommendation. Resident reuse is also
not a big remaining ceiling in this shape. The direct CTree lane is still worth
validating because it is about `1.85x` faster than the stock facade, not
because input movement alone can deliver the next 2-3x.

Current accounting contract:

- `lightzero_mcts_arrays_boundary_root_prepare_sec` excludes model-output D2H.
- `lightzero_mcts_arrays_boundary_model_output_d2h_sec/bytes` price the root
  value, latent-state, and policy-logit transfer separately.
- `lightzero_consumer_collect_forward_non_model_sec` is only meaningful for
  rows that call public `collect_mode.forward`.
- Direct CTree rows use
  `lightzero_mcts_arrays_boundary_non_model_sec` /
  `lightzero_consumer_direct_boundary_non_model_sec` for their broader
  non-model boundary cost.
- Manifest row labels only describe the requested input mode. Runtime telemetry
  is the source of truth for whether resident input was fresh first-fill or
  stale reuse.

Caveat:

```text
The all-actions-legal fast path preserves distributional action sampling but
does not preserve exact stock RNG consumption order. That is acceptable for a
profile-only speed probe. Do not promote it into training until fixed-seed
stock-vs-direct parity gates are explicit.
```

## 2026-05-21 Parity Gate Update

New local tests now cover the first real parity checks:

```text
uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py
uv run --extra lightzero pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py -k real_policy_cpu
uv run pytest -q -p no:cacheprovider tests/test_multiplayer_source_state_target_rows.py
```

Results:

```text
79 passed, 2 warnings
6 passed, 76 deselected, 2 warnings
14 passed, 2 warnings
```

What is now covered:

- Stock-facade decoding maps legal-action-length visit counts back into the
  full action ids.
- Stock-facade decoding zeroes any illegal visit mass when a full action-space
  visit vector is returned.
- Stock-facade and direct CTree rows now count model-evaluation roots the same
  way: initial root batch plus recurrent simulation batches.
- Stock-facade compact decoding now carries predicted value and policy-logit
  arrays when LightZero returns them, including legal-action-length logits
  mapped back to full action ids. This makes the debug/parity surface closer to
  the direct CTree output.
- LightZero collect/direct boundary probes now reject fractional action masks
  up front so profile rows cannot silently diverge from stock legal-action
  semantics.
- Real LightZero CPU policy checks for simulations `1`, `2`, and `8` compare
  stock facade versus direct CTree arrays on searched values, normalized visit
  row shape, legal action outputs, and zero illegal action count.
- A biased-logit real-policy check makes action 1 the clear winner and verifies
  stock facade and direct CTree both choose action 1 as the top action for every
  root at sim8. This tells us the direct path is not losing the obvious policy
  preference.
- Single-legal-action real-policy rows now pass exactly: both paths choose the
  only legal action and produce the expected one-hot visit distribution.
- Masked biased-logit rows now pass: when the preferred action is illegal, both
  paths choose the best legal fallback and keep zero visit mass on illegal
  actions.
- The checked multiplayer target-row bridge accepts non-one-hot MCTS visit
  distributions as `policy_target`, preserves `root_value` and `policy_source`,
  and still checks action legality and reset-to-step alignment. This proves the
  compact arrays output can fit the repo-owned target-row contract.

What is not yet claimed:

```text
Exact action/visit parity is not proven.
Native LightZero GameSegment/full trainer integration is not proven.
```

In the real CTree probe, fixed Python/NumPy/Torch seeds did not make visit
distributions exactly identical between two separate stock/direct search calls.
Neutral-logit rows are especially tie-heavy; even with root noise disabled,
visit allocation can move by one simulation across actions. The deterministic
mask tests above make this much less scary: the direct path respects forced
legal actions, masked-out winners, and clear policy preferences. Treat exact
neutral visit/action parity as a remaining diagnostic gate, not as something
the speed probe already proves. The current direct CTree arrays path remains
profile-only.

2026-05-21 follow-up: exact neutral visit parity is probably the wrong
production gate. A local diagnostic showed fixed Python/NumPy/Torch seeds do
not make the stock LightZero CTree path repeat identical neutral/tie-heavy
visit allocations even stock-vs-stock. In plain language: the tie-heavy case is
not a stable ruler. It is still useful for finding obvious decode bugs, but it
should not be the thing that blocks or approves a production search boundary.

The meaningful replacement contract should be:

- exact parity for forced cases: single legal action, masked-out preferred
  action, illegal-action mass zeroing, action-mask id mapping, searched value
  shape/value sanity, output schema, root value/source metadata, and replay
  target row compatibility;
- clear-preference parity: when logits make one legal action obviously best,
  stock facade and direct CTree choose that action and do not put visit mass on
  illegal actions;
- stochastic/statistical parity for ordinary collect rows: compare distributions
  and training-facing metrics over many roots/seeds, not one exact tie-heavy
  visit vector;
- matched full-loop profile gate before any Coach advice: same observation
  contract, same RND mode, same reset/death/autoreset behavior, same
  checkpoint/eval/tournament compatibility story.

Until those gates are accepted deliberately, `direct_ctree_arrays` is still a
profile-only speed probe.

## 2026-05-21 Implementation Status

The first facade now exists as a profile-only probe:

```text
--hybrid-lightzero-mcts-arrays-boundary-probe
```

It uses the same pre-scalar `[B,2,4,64,64]` stack as the other hybrid LightZero
probes, calls stock `collect_mode.forward`, and reports compact action,
visit-distribution, and searched-value array shapes/bytes. It does not replace
MCTS internals yet.

The local project now has an optional LightZero dependency extra:

```text
uv run --extra lightzero python -c "import lzero, torch"
```

Verified imports:

```text
LightZero 0.2.0
torch 2.8.0
```

Modal smoke:

```text
run id: ap-Amg22e2oRyJHZMNqPy9god
backend: lightzero_mcts_arrays_boundary_consumer
semantics: stock_lightzero_mcts_arrays_facade
policy_device: cuda:0
action_shape: [8]
visit_shape: [8, 3]
searched_value_shape: [8]
compact_output_bytes: 192
public_output_bytes: 1086
illegal_action_count: 0
```

This proves the compact arrays facade can call real LightZero policy/search in
the Modal image and decode the public result into compact arrays. It does not
prove speed yet because it still calls the public LightZero collect path.

Medium same-shape rows:

```text
shape: B512 physical rows, 2 player roots each, sim8, 25 measured steps
L4/T4 run: ap-5COErgYQQR2Gb9IsShxOnY
H100 run:  ap-4KPxuHpOOw4AfgD7rrlqIu
```

| compute | scalar roots/sec | collect-forward sec | decode sec | H2D sec | model sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| L4/T4 | `1421.28` | `14.23s` | `0.69s` | `0.81s` | `1.16s` |
| H100 | `2319.65` | `8.24s` | `0.41s` | `0.53s` | `0.44s` |

Plain read:

```text
The facade is now proven at a useful profile shape. It is not faster than the
public collect path because it still calls that path internally. Its value is
that compact arrays are now the measured boundary, so the next replacement can
target the right shape without changing trainer defaults first.
```

Local validation:

```text
94 passed
focused ruff passed
py_compile passed
```

Next validation before promotion:

```text
fixed-seed parity against stock LightZero outputs
resident input/H2D split
```
