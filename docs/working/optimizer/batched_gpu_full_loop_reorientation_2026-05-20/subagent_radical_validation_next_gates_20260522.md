# Subagent Radical Validation Next Gates - 2026-05-22

Scope: ranked validation/falsifier ladder for radical optimizer lanes:
`mock_search_service`, `direct_ctree_gpu_latent`, `recurrent_toy`,
`dense_torch_mcts`, and any future compact batched search service or
array-native CTree rewrite. I did not touch live Coach runs.

Reviewed:

- `radical_optimizer_trust_gates_20260522.md`
- `tests/test_source_state_batched_observation_boundary_profile.py`
- `tests/test_lightzero_phase_profiler.py`
- `scripts/compare_curvytron_direct_ctree_stock.py`

## Plain Read

The current tests are useful, but they mostly prove profile probes and direct
hook output shape. They do not yet prove that a radical search lane trains the
same targets as stock LightZero.

Current coverage that already exists:

- train-facing direct hook install/restore and fallback;
- mixed-mask and single-legal full-action-id checks;
- raw legal-action visit-list contract for the direct hook;
- fractional/zero-mask fail-closed tests;
- stochastic all-actions-legal fast-path smoke;
- profile-only modes are labeled as not real CTree where appropriate;
- `mock_search_service` is clearly named as a ceiling, not MCTS;
- `recurrent_toy` proves repeated recurrent calls can be timed;
- `dense_torch_mcts` backs up nonzero reward/discount in a local unit test;
- compare script can run stock facade versus direct CTree variants over mask
  scenarios and fail on illegal actions, CUDA absence, or strict thresholds.

The missing trust boundary is replay/target material. A fast search row can have
the right collect dict keys and still feed different actions, visits, values,
`to_play`, done flags, or final observations into replay.

Update after the first durable H100 ceiling wave:

```text
mock_search_service sim16:       11648.29 roots/sec
direct_ctree_gpu_latent sim16:    5303.97 roots/sec
recurrent_toy sim16:              8512.57 roots/sec
```

The service-shaped ceiling is about `2.20x` above current direct, so it is worth
keeping alive. It is below the "obviously radical" `4x+` threshold, so the next
trust gate should include the broader native/vector buffer path, not just a
search-service wrapper.

## Ranked Ladder

### P0. Local Deterministic Contract Tests

These are cheap and should run before any new architecture claims speed.

1. Replay/target consumer canary.
   Feed stock and candidate collect outputs into the same smallest LightZero
   consumer path available. Compare stored action, visit counts, searched
   value, reward, done, action mask, `to_play`, env id, player id, and final
   observation metadata.

2. Row/column sentinel.
   Use non-identity env ids, player ids, latent path ids, batch ids, and action
   ids. Make fake recurrent outputs encode those ids. Assert the candidate
   gathers, backs up, and returns the intended rows.

3. Terminal/live/autoreset canary.
   One row dies, one row stays live. Exact checks: terminal row is not searched
   as a live root after death, final observation is pre-autoreset, autoreset
   observation is next-episode only, and live rows keep their order.

4. RND meter safety canary.
   With `rnd_meter_v0`, prove reward-model entrypoint, positive RND train and
   estimate counters after warmup, predictor hash changed, target hash frozen,
   target rewards unchanged, and latest-frame source matches the replay row.

5. Exact mask contract.
   Keep the existing forced-mask tests. Add any new candidate backend to the
   same mask harness before timing it.

Fail closed if:

- any candidate action is illegal;
- legal-list index is returned instead of full action id;
- visit list length/order differs from stock;
- any terminal row is searched after done;
- final observation is from after autoreset;
- RND target rewards change in meter mode;
- fallback is hidden or unlabeled.

### P1. Smallest Search-Service Worth-It Test

Status, 2026-05-22: implemented as a profile-only falsifier.

Current read:

```text
service_tax_probe now exists and can feed the compact replay proof.
H100 B512/A16 closed-profile rows:
  sim16 service_tax_probe: 2.10x direct
  sim32 service_tax_probe: 1.29x direct

This is useful but not enough for a 5-10x claim. Treat it as evidence that
wrapper-only compact topology is bounded; the next aggressive lane needs a
real fixed-shape/device-resident search body or MCTX/JAX scratch benchmark.
```

This is the first test that can tell whether a compact batched search service is
worth implementing.

Build a local or profile-only in-process `service_tax_probe`, not a real
distributed service:

```text
pack stock-shaped roots + masks + ids
send through one service-shaped function call
run batched initial/recurrent inference with the real model
return deterministic fake visits/actions/values in stock collect shape
record pack time, service call time, recurrent batch sizes, unpack time
```

It should include:

- B256 and B512 roots;
- sim8 and sim16;
- all-legal and mixed-mask rows;
- request ids and non-identity env/player ids;
- no network, no Modal queue, no multiprocessing at first.

Decision rule:

- If this in-process service-shaped probe cannot beat
  `direct_ctree_gpu_latent` by a large margin in roots/sec after warmup, a real
  service is not worth building yet. The real service will only add queueing,
  serialization, backpressure, and failure handling.
- If it beats direct by at least about `2x` on sim16 while preserving P0
  semantics, then build the next service prototype.
- If it beats direct by `4x+`, prioritize the service lane over more tiny
  direct-hook polish.

This test is deliberately not full MCTS. It answers the architectural question:
is the pack/batch/unpack shape cheap enough to support a radical search service?

Fail closed if:

- request ids are not preserved;
- mixed masks are ignored;
- recurrent batch size collapses to scalar calls;
- service overhead is close to or larger than saved search time;
- telemetry cannot separate pack, inference, search/update, and unpack.

### P2. Profile-Only Modal Rows

Only after P0 passes for the candidate boundary.

Run profile-only rows, not live Coach runs:

1. Stock baseline:
   stock `train_muzero`, same observation contract, same death/RND settings.

2. Current best tactical probe:
   `direct_ctree_gpu_latent + output-fast`.

3. Array ceiling:
   `recurrent_toy`, same B/sim shape, to estimate the real-model recurrent
   inference lower bound.

4. Service ceiling:
   `mock_search_service` or the new `service_tax_probe`, clearly labeled as
   not MCTS.

5. Dense search:
   `dense_torch_mcts` only as a replacement-search semantics experiment, not a
   stock-compatible claim.

Required row identity:

- `mode=profile`;
- `called_train_muzero=true` for stock-loop claims;
- backend/mode names;
- fallback calls;
- output rows;
- root count;
- sim count;
- warmup/measured windows;
- death mode;
- RND mode;
- observation contract;
- sidecars/eval/GIF/checkpoint state;
- compile/service status if relevant.

Fail closed if:

- a row claims direct/service speed with fallback calls > 0;
- compile row falls back but is summarized as compiled;
- `mock_search_service` is described as MCTS;
- sidecars differ across matched rows without being called out;
- warmup dominates the measurement.

### P3. Matched Full-Loop A/B

Only after local semantics and profile-only falsifiers pass.

Minimum matched A/B set:

- two no-RND repeats: stock versus candidate;
- one `rnd_meter_v0` repeat: stock versus candidate;
- one normal-death/autoreset repeat: stock versus candidate;
- same seed, collector count, sim count, learner cap, warmup, sidecars, and
  observation contract.

Exact checks:

- semantic attestation passes;
- zero hidden fallback;
- RND safety invariants pass in meter mode;
- normal death/autoreset counts are present and sane.

Statistical checks:

- throughput;
- GPU timing;
- stochastic action distribution;
- root-noise behavior.

Fail closed if:

- speed only exists in sidecar roots/sec and disappears in full-loop timing;
- RND rows skip training/estimate after warmup;
- normal-death row loses final observation or terminal ordering;
- candidate changes replay/target fields.

### P4. Coach-Facing Recommendation

Only after P0-P3.

The recommendation must say one of these plainly:

```text
stock-compatible optimizer path
replacement trainer/search architecture
profile-only ceiling, not training
```

Do not mix those labels. A replacement trainer can be worthwhile, but it must
be treated as a new trainer with its own learning proof.

## Exact Versus Statistical

Exact:

- masks;
- full action ids;
- illegal actions and illegal visit mass;
- raw visit-list length/order;
- env/player/request id order;
- same-model reward/value/logit rows;
- replay/target fields;
- terminal/final-observation/autoreset order;
- RND meter safety;
- attestation and fallback counts.

Statistical:

- neutral/tie-heavy MCTS action equality;
- root-noise visits;
- stochastic selection;
- throughput;
- queue timing;
- positive RND learning strength.

Avoid the exact-parity trap: neutral MCTS action equality is not the approval
gate. Use exact gates for the parts that define training meaning.

## Next Concrete Work

The smallest useful implementation next is the P1 `service_tax_probe` plus the
P0 replay/target consumer canary.

Order:

1. Add the replay/target consumer canary for the existing direct hook.
2. Add a shared deterministic fixture for stock/candidate collect output.
3. Add the in-process `service_tax_probe` using that fixture.
4. Run local deterministic tests.
5. If local tests pass, run profile-only Modal rows comparing stock, direct,
   recurrent_toy, and service_tax_probe at B256/B512 and sim8/sim16.

If the service-shaped probe cannot clear direct by a large margin, stop the
service lane and move to array-native CTree or another larger architecture.

Current adjustment:

```text
The mock service cleared direct by a meaningful margin but not by enough to be
the whole answer. Add the native/vector buffer falsifier next, and use the same
P0 replay/target canaries for any compact replay materialization.
```
