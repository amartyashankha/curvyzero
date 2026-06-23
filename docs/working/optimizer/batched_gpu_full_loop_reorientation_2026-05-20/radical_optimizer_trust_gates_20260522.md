# Radical Optimizer Trust Gates - 2026-05-22

Scope: validation critique for the radical optimizer lane:
`direct_ctree_gpu_latent`, `dense_torch_mcts_compile_spike`, RND meter mode,
array-native CTree, and possible future batched search/service designs. This
doc is a trust checklist, not a launch plan. I did not edit production code or
touch live runs.

## Core Rule

No optimizer claim is trusted unless it states which claim class it is making
and passes the gates for that class.

Claim classes:

1. Profile-side roots/sec claim:
   standalone or sidecar profile evidence only.
2. Stock-loop profile claim:
   bounded `mode=profile`, `called_train_muzero=true`, no live-run mutation.
3. Coach-facing recommendation:
   same training meaning as the trusted path, plus stable matched full-loop
   wins and semantic attestation.
4. Replacement-trainer architecture claim:
   custom collector/replay/search semantics are explicitly documented and
   tested as a new trainer, not described as stock LightZero.

If a row lacks semantic identity, it can stay in experiment logs but must not
be used as Coach advice.

## Cross-Lane Mandatory Gates

| Gate | Must prove | Exact or statistical |
| --- | --- | --- |
| Forced masks | single legal action, mixed legal action sets, full-action ids, raw legal-action visit lists, zero illegal visit mass | exact |
| Fail-closed masks | fractional masks reject; empty masks reject or are filtered upstream with explicit counts | exact |
| All-legal selector | deterministic clear-preference parity and stochastic distribution parity for all-actions-legal fast paths | mixed: exact for deterministic, statistical for stochastic |
| Search values | same-model predicted value/logits and searched/root values match expected transforms | exact or tight numeric tolerance |
| Row/order sentinels | row, player, env id, latent batch index, action id, reward/value/logit unpack order | exact |
| Replay/target canary | action, visit counts, searched value, reward, done, mask, `to_play`, and final-observation metadata survive consumer path | exact |
| Terminal/live batch | at least one terminal row and one live row in the same batch; terminal rows do not pollute live roots | exact |
| Long trajectories | long no-death trajectories do not change observation/search denominator story; renderer/cache/fallback labels are explicit | statistical throughput plus exact surface checks |
| RND meter | reward-model entrypoint, cadence, latest-frame source, predictor changed, target frozen, target rewards unchanged | exact for safety, statistical for overhead |
| Full-loop A/B | matched stock/candidate profile repeats with eval/GIF/checkpoint sidecars disabled or matched | statistical throughput |
| Attestation | backend, death/RND mode, env-step source, warmup, fallback calls, sidecars, commit/image, and consumer semantics present | exact schema gate |

## Direct CTree GPU-Latent

Current status: this is the best tactical stock-loop optimizer probe after the
all-actions-legal output fast path. It is still profile-only.

Already recorded as landed in the promotion docs:

- hook lifecycle and stock no-op behavior;
- train-hook mixed-mask and single-legal forced-mask tests;
- fail-closed fractional/zero mask and ready-env-id mismatch tests;
- all-actions-legal stochastic fast-path sampling smoke;
- direct-row attestation requiring direct calls, output rows, and zero fallback;
- matched C64/sim16/no-RND and `rnd_meter_v0` profile-loop wins in the current
  denominator.

Remaining gates before Coach-facing trust:

- Hook output to stock replay/target material:
  prove raw legal-action visit lists, actions, masks, rewards, searched values,
  predicted values/logits, `to_play`, done, and final-observation fields become
  the same replay/target material as stock.
- GPU-latent row/column sentinel:
  fake `batch_traverse` with non-identity latent path/batch/action indices and
  assert `batch_backpropagate` receives the expected reward/value/logit rows.
- Terminal/live normal-death row:
  candidate and stock must both show terminal rows filtered from policy roots
  while final observations and live rows remain correct.
- Full-loop repeat policy:
  require at least two matched no-RND repeats and at least one matched
  `rnd_meter_v0` repeat on the same shape before claiming stable speed.
- Fallback rule:
  `collect_search_backend_fallback_calls` must be `0` in any direct speed row.

Trusted phrasing until those pass:

```text
direct_ctree_gpu_latent output-fast is the current best profile-loop optimizer
probe. It is not yet a Coach training default.
```

## Dense Torch MCTS Compile Spike

Current status: profile-only architecture falsifier. It is not LightZero CTree
and should not be sold as training-compatible unless it later earns a separate
replay/search semantics contract.

Tests/profiles required before trusting any dense compile speed claim:

- Fixed-shape preconditions are attested:
  CUDA device, all roots active, all actions legal, fixed root count, fixed
  action count, fixed sim count, `root_noise_weight=0.0`, no exception-shaped
  action-input probing in the captured body.
- Compile status is explicit:
  `compile_enabled`, `compile_attempted`, `compile_status`, and
  `compile_reason` must be present. Fallback rows cannot be summarized as
  compiled rows.
- Dense semantics pass:
  nonzero reward/discount backup, value checksum, visit normalization, finite
  policy logits, zero illegal actions, and fractional masks reject before model
  inference.
- Forced-mask CPU/eager parity remains separate from compile:
  if compile requires all actions legal, mixed-mask semantics must either use
  the safe eager path or be explicitly out of scope.
- Root-noise gate:
  do not make root-noise claims until legal-action-only noise weighting is
  tested against direct CTree or stock. The first trusted compile spike should
  stay at `root_noise_weight=0.0`.
- Timing gate:
  include CUDA-event or explicit sync diagnostics so `search_update_sec`,
  recurrent inference, final readback, and graph replay are not misread.
- Falsifier:
  on the same H100 B512/A16 sim16 denominator, compiled dense must beat
  `direct_ctree_gpu_latent` by a meaningful margin after warmup. If it only wins
  sim8 or falls back to eager for sim16, keep it as a failed/partial falsifier.

Trusted phrasing:

```text
dense_torch_mcts_compile_spike can test whether fixed-shape GPU search is
viable. It is not a stock LightZero training recommendation.
```

## RND Meter Mode

Current status: `rnd_meter_v0` is useful for overhead and plumbing. It is not
positive-RND learning proof.

Must pass for every RND speed claim:

- Entry point:
  profile uses the reward-model path, not a no-RND trainer with RND metadata.
- Cadence:
  report `rnd_update_per_collect`, `rnd_batch_size`,
  `collect_data_calls`, `train_with_data_calls`, `estimate_calls`,
  `train_cnt_rnd`, `estimate_cnt_rnd`, and small-buffer skip counts.
- Warm buffer:
  reject throughput reads where RND training never warmed or skipped after
  warmup.
- Safety:
  predictor hash changes, target hash stays frozen, target reward delta mean
  and max are exactly zero in meter mode.
- Latest-frame source:
  sampled checksum or equivalent proves RND reads the latest policy stack frame
  for the same env ids/players, including terminal/autoreset rows.
- Device/timing:
  report RND device, collect/train/estimate/hash/metrics timers, and GPU
  contention. RND overhead must be compared against a matched no-RND anchor.
- Normalization warning:
  batch-min-max novelty blocks positive-RND claims. Any
  `rnd_replay_target_v0` or nonzero-weight claim needs a separate global/running
  normalization and resume/checkpoint contract.

Trusted phrasing:

```text
rnd_meter_v0 proves RND plumbing and overhead only. Positive RND remains a
separate normalization/cadence problem.
```

## Array-Native CTree

Current status: strongest conservative next implementation if the goal is to
preserve LightZero CTree semantics while removing Python/list fanout.

Required before trusting an array-native result:

- API contract:
  fixed `A=3` flat typed arrays for rewards, values, policy logits, and
  `to_play`; no Python nested lists in the per-simulation hot path.
- Local parity:
  compare stock/direct CTree/array-native over sim1, sim2, sim8, and sim16 with
  identical seeds, legal masks, values, policy logits, rewards, visit counts,
  searched values, and actions.
- Forced masks:
  single-legal and mixed-legal rows must be exact. Illegal visit mass must be
  zero. Fractional and empty masks fail closed.
- Root noise:
  legal-action Dirichlet noise must match stock/direct semantics under partial
  masks. Root-noise rows are a separate gate after no-noise parity.
- Row/column sentinels:
  typed buffers must carry sentinels that catch row/action/reward/value/logit
  swaps.
- CUDA canary:
  tiny Modal CUDA row with actual model outputs and fixed shape, then matched
  H100 sim8/sim16 profile rows against current direct hook.
- Full-loop A/B:
  no-RND and `rnd_meter_v0` stock-loop profile repeats against current direct
  output-fast, not only sidecar roots/sec.
- Fallback:
  if shape is not fixed `A=3`, masks are unsupported, or any API invariant
  fails, fall back to the known safe path and record the fallback.

Trusted phrasing:

```text
array-native CTree is trusted only if it is parity-equivalent to current CTree
and improves the stock-loop denominator, not just Cython microbenchmarks.
```

## Future Search Service / Batched Actor Service

Current status: plausible larger architecture, but it changes topology:
actors, active-root refill, queueing, weight freshness, search cadence, replay
cadence, and possibly trainer ownership.

Required before trusting a service claim:

- Semantic classification:
  either stock-compatible edge with identical replay/target fields, or explicit
  replacement-trainer contract. Do not blur the two.
- Deterministic toy service:
  fixed weights, fixed observations, fixed masks, deterministic forced cases,
  exact replay rows.
- Active-root accounting:
  report offered roots, accepted roots, active roots, terminal roots, refill
  count, dropped roots, queue depth, wait time, and recurrent batch size.
- Long-running current-env check:
  run enough trajectory length to expose long-trail/render/cache behavior,
  terminal/autoreset churn, and current opponent/current-policy row mixes.
- Weight freshness:
  report model version per search request, policy lag, learner update cadence,
  and whether stale searches are allowed or discarded.
- RND compatibility:
  prove RND feature source and reward-model cadence still align with replay
  rows under asynchronous service timing.
- Backpressure:
  show scheduler/queue overhead is smaller than search savings; otherwise the
  service is just moving the bottleneck.
- Failure isolation:
  service fallback, crash recovery, partial batch behavior, and deterministic
  replay of failed batches must be documented before any training claim.

Trusted phrasing:

```text
a search service can claim architecture headroom after toy semantics and
queue/root accounting pass. It cannot claim Coach speed until replay, RND,
death, weight freshness, and full-loop profiles pass.
```

## Long-Trajectory And Current-Env Gates

Short no-death fixed-opponent rows are optimizer probes. They are not enough for
current-environment claims.

Long-trajectory gates:

- Include at least one long no-death row, such as 512 or 1000 source steps, so
  renderer/cache/trail-history costs cannot hide behind short trajectories.
- Include a surface checksum/parity proof for the same observation contract:
  latest frame, stack FIFO, row/player perspective, bonus symbols, and terminal
  final observations if death is enabled.
- Record active trail count, render width, cache hit/fallback/full-rebuild
  counts, renderer mode, and whether persistent/incremental rendering is exact
  or approximate.
- Do not compare long-trajectory renderer speed against short search-boundary
  rows without labeling the denominator shift.

Current-env/current-policy gates:

- If the claim applies beyond `source_state_fixed_opponent`, run a current-lane
  canary with the actual current env/opponent mix and its `to_play` semantics.
- Prove row/player mapping for current-policy versus frozen rows, including
  absent/dead players and complete physical-row action commits.
- Prove action masks remain binary and ordered through current-policy payloads,
  not just fixed-opponent scalar envs.
- Prove checkpoint/opponent assignment metadata, current-policy freshness, and
  any current-vs-frozen mix labels are present in the profile summary.
- Do not reuse fixed-opponent `to_play=-1` proof for a two-seat/current-policy
  lane that needs player-id semantics.

## End-To-End Profile Attestation

Any row used in a recommendation must include or link:

- code/image identity: git SHA or artifact id, route, compute, seed;
- profile identity: `mode=profile`, `called_train_muzero`, profile-only/live
  mutation flags, auto-resume and volume-commit state;
- backend identity: env manager, observation backend, renderer modes, collect
  search backend, array/search implementation, compile status if applicable;
- denominator identity: env steps, env-step source, MCTS roots, sim count,
  learner calls, replay samples, warmup/measured steps, profiler stride;
- semantic identity: observation contract, stack dtype/range, row/player order,
  `to_play`, action-mask semantics, scalar materialization;
- death identity: no-death versus normal, terminal row count, autoreset count,
  final-observation presence/bytes, terminal-before-autoreset proof;
- RND identity: mode, weight, cadence, device, train/estimate counters,
  predictor/target hash results, reward-target delta;
- sidecars: eval, GIF, checkpoint, tournament, live publisher, and background
  workers enabled/disabled or matched between arms;
- direct/array/search self-audits: fallback calls, output rows, fast-path calls,
  compile status, service queue/root stats, or array-native API status.

Reject rows with missing attestation from Coach-facing summaries.

## Statistical Versus Exact

Exact gates:

- forced masks and legal full-action ids;
- illegal action and illegal visit mass;
- raw legal-action visit list length/order;
- output/replay/target schema;
- values/logits under the same model outputs;
- terminal/live row accounting and final observations;
- RND meter safety invariants;
- attestation field presence and zero fallback where required.

Statistical gates:

- stochastic all-legal action sampling;
- ordinary root-noise collect rows;
- neutral/tie-heavy action agreement;
- full-loop throughput;
- GPU contention/timing;
- positive RND learning behavior once a normalization contract exists.

Do not require exact action equality for neutral/tie-heavy stochastic rows. Do
require exact correctness for the parts that define training meaning.

## Minimal Next Trust Bundle

Before the next architecture experiment is allowed to become a recommendation,
the bundle should include:

1. Direct hook replay/target canary.
2. GPU-latent row/column sentinel.
3. Dense compile fixed-shape attestation plus sim16 falsifier.
4. RND meter matched no-RND/direct rows with cadence and latest-frame proof.
5. One long-trajectory row and one current-env/current-policy canary if the
   claim is broader than fixed-opponent no-death.
6. Summary attestation gate for every row included in a speed table.

No single roots/sec number, however pretty, should outrank this bundle.
