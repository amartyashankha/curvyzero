# Strategy Reset Critique, 2026-05-23c

Scope: bounded research/code/doc pass. I did not run Modal, touch live Coach
training, edit source code, checkpoints, evals, tournaments, GIFs, or volumes.

## Short Answer

The current bottleneck is not raw rendering. It is the collect/search/dataflow
shape:

```text
CPU env/compact rows
-> GPU model/root work
-> LightZero CTree or profile search service
-> CPU/list/object/search/replay materialization
-> replay/RND/sample edges
```

The next justified move is a profile-only compact search/dataflow slab behind
`CompactSearchServiceV1`, with either fixed-`A=3` array-native CTree or a real
fixed-shape compiled search backend. Keep MCTX as a side comparator. Do not
promote the current eager compact Torch service, and do not advise Coach to use
any of this until a capped stock-vs-candidate trainer smoke passes.

## 1. Current Bottleneck

Best evidence:

- Latest same-denominator H100 wave says direct CTree beats eager compact Torch:
  sim16 `5.47k` vs `4.05k` steps/sec; sim32 `3.14k` vs `2.67k`. Service-tax and
  mock still show headroom: sim16 `7.81k`/`7.46k`, sim32 `5.19k`/`9.17k`.
  Local source: `strategy_reorientation_20260523b.md:46-70`,
  `world_model.md:33-61`, `subagent_current_bottleneck_recritique_20260523b.md:109-167`.
- The direct CTree path still crosses GPU/CPU/list boundaries. The profile code
  builds CPU policy logits, legal action lists, Dirichlet noise lists, and
  `roots.prepare(...)`; its simulation loop calls CTree traverse, runs Torch
  recurrent inference, copies reward/value/policy to CPU NumPy, listifies, then
  calls CTree backprop. Code: `source_state_batched_observation_boundary_profile.py:5915-5979`
  and `source_state_batched_observation_boundary_profile.py:6449-6626`.
- The profile harness and compact slab are explicitly not Coach training:
  `HybridBatchedObservationProfileManager` is `profile_only=True`,
  `calls_train_muzero=False`, and not stock-integrated
  (`source_state_hybrid_observation_profile.py:581-588`). The trainer also blocks
  non-stock collect-search backends outside `mode="profile"`
  (`lightzero_curvyzero_stacked_debug_visual_survival_train.py:5522-5526`).

Plain read: there is real architecture headroom, but the current wall is a
search/control/materialization boundary, not a single render kernel.

## 2. Why Render Work Did Not Give A Huge Coach Speedup

Render work helped in narrow places, but it did not remove the whole hot
contract. The persistent compact render-state buffer removed one conversion leaf
yet partly replaced it with parent-side compact trail writes. In the A/B, sim16
improved versus copied production state, but sim32 regressed, and the borrowed
production-state path was still faster because it avoided the parent copy.
Source: `experiment_log.md:95-163`.

The same pattern appears in the wider batch rows: bigger batches help, but B2048
only modestly beats B1024 while production-to-compact, delta pack, H2D/update,
public packaging, and game mechanics grow. Source: `experiment_log.md:53-93`.

The important weak spot in older claims is denominator drift. Some earlier rows
made compact Torch or render deletion look like a bigger win, but the latest
profile labels and same-denominator wave say those are optimizer probes, not
Coach speed. Source: `experiment_log.md:5-40` versus
`subagent_current_bottleneck_recritique_20260523b.md:109-118`.

## 3. Big Move Actually Justified Next

Do this next, still profile-only:

```text
HybridCompactBatch
-> CompactRootBatchV1
-> CompactSearchServiceV1
-> better backend candidate
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
-> sample/RND/terminal validation
```

I would bias first toward fixed-`A=3` array-native CTree if it can be exposed
cleanly, because it preserves LightZero semantics while attacking the observed
CPU/list ABI tax. If that is capped or awkward, use a fixed-shape compiled
Torch/Triton/CUDA-graph search spike behind the same service contract. The
current eager compact Torch service is useful scaffolding, not the backend.
Sources: `strategy_reorientation_20260523b.md:142-173`,
`subagent_big_move_options_20260523b.md:56-113`,
`subagent_big_move_options_20260523b.md:169-219`.

Keep hardening the compact slab at the same time. The code already stages
selected actions, requires the next batch to apply them, and commits previous
search rows only then (`compact_rollout_slab.py:48-130`). The two-phase payload
gate also hides replay payloads until identities match
(`compact_search_service.py:163-225`). That is the right platform layer for any
faster backend.

## 4. Fast Falsifiers

Kill or demote the next backend if any of these fail:

- Same-denominator H100 profile-only grid: `B512/P2/A3`, sim16 and sim32,
  `root_noise_weight=0.0`, compact replay proof on, direct CTree baseline,
  service-tax/mock ceiling, and candidate backend.
- Candidate must beat direct CTree after warmup and required readback. For a
  compiled backend, use the existing stricter bar: at least `1.25x` faster at
  sim16 and not slower at sim32. For fixed-A3 CTree, use at least `15%` at sim16
  and no sim32 regression. Source: `subagent_big_move_options_20260523b.md:93-113`
  and `subagent_big_move_options_20260523b.md:201-219`.
- Selected action at record `k` must become the env `joint_action` for record
  `k+1`; compact rows must materialize to trusted target rows and learner-visible
  samples; terminal final observations and RND latest-frame identity must stay
  attached. Source: `subagent_validation_ladder_recritique_20260523b.md:46-136`.
- No Coach-facing claim until a capped full-loop smoke calls the real trainer
  entrypoint, has no fallback, and emits replay/sample/RND digests. Source:
  `subagent_validation_ladder_recritique_20260523b.md:121-136`.

If the backend fails while service-tax/mock remain well above direct CTree, stop
polishing wrappers and change the search strategy. If it passes, the next proof
is a tiny matched stock-vs-candidate full-loop smoke, not a big Coach launch.

## 5. External Systems Check

External systems support the direction, but with caveats:

- MCTX supports the idea of dense, batched, accelerator-native search. Its README
  describes JAX-native MCTS, JIT support, batch-parallel inputs, and MuZero/
  Gumbel MuZero policies that take `RootFnOutput` plus a `recurrent_fn`.
  Web source: <https://github.com/google-deepmind/mctx>. Local read: MCTX is a
  side comparator until there is a JAX-native model/search/replay decision
  (`subagent_mctx_comparator_status_20260523b.md:8-24`,
  `subagent_mctx_comparator_status_20260523b.md:163-240`).
- PufferLib supports the buffer/slab lesson, not a trainer replacement. Its
  vectorization docs emphasize async env collection, shared memory, a single
  shared buffer, and zero-copy batching. Web source:
  <https://pufferai.github.io/build/html/rst/landing.html#vectorization>. Local
  read: steal the static-buffer contract, not PPO/V-trace learning
  (`subagent_external_fast_rl_patterns_20260523b.md:54-108`).
- MiniZero supports many self-play workers with multiple MCTS instances and
  batch GPU inference. Web source: <https://github.com/rlglab/minizero>. This
  supports many roots/leaves in flight, not one scalar LightZero object path.
- OpenSpiel supports actor/learner/evaluator separation and warns against the
  simple Python path: its docs say Python AlphaZero has no inference batching and
  CPU-only inference/training, while C++/LibTorch uses threads, cache, batching,
  and GPU inference/training. Web source:
  <https://openspiel.readthedocs.io/en/latest/alpha_zero.html>.
- EfficientZero supports the conservative production shape: PyTorch plus
  C++/Cython CTree pieces, Ray distribution, CPU/GPU actors, and parallel MCTS
  knobs. Web source: <https://github.com/YeWR/EfficientZero>. This reinforces
  batched inference/search-service architecture, not full GPU tree magic.
- JAX docs reinforce the MCTX caveat: JIT caching depends on stable functions and
  static values; changing static arguments can recompile. Web source:
  <https://docs.jax.dev/en/latest/jit-compilation.html>.

## Contradictions And Weak Evidence

- The current docs already contain older rows where compact Torch looked faster
  than direct CTree. The latest same-denominator wave overturns that for the
  decision at hand. Use the newest no-noise direct/compact/service-tax/mock grid,
  not isolated smoke rows.
- "Env" time is a dangerous bucket name. Local notes warn it contains
  observation handoff, stack ownership, final-observation packing, and metadata
  packaging. Do not conclude pure CurvyTron mechanics are the wall without
  splitting that bucket. Source: `world_model.md:174-181`.
- MCTX rows are fast but use toy JAX dynamics and independent per-seat `A=3`
  roots. They do not prove LightZero target semantics, multiplayer adequacy,
  RND, learner samples, checkpoints, evals, tournaments, or Coach speed.
- Validation is much better than before, but mostly contract/profile validation.
  The trainer fence is intentional and correct. Treat it as a blocker to
  promotion, not an inconvenience.

## Final Recommendation

Take the big optimizer move, but keep the label honest:

```text
profile-only compact slab + CompactSearchServiceV1
+ fixed-A3/array-native CTree or fixed-shape compiled search
+ MCTX real-root side comparator
+ replay/RND/terminal/sample gates
```

The first win condition is not "faster roots/sec". It is:

```text
same-denominator faster than direct CTree
and selected action -> env step -> replay row -> learner sample stays correct
```

Until that passes through a capped trainer smoke, the only Coach recommendation
is: keep stock LightZero as the trusted lane.
