# Profile Next Steps

Date: 2026-05-09

Status: immediate optimizer checklist.

## Short Read

Do not add bigger training runs yet. First make the report surface stable enough
that repo-native and LightZero lanes can be compared without hiding different
denominators.

Current CurvyTron hook verdict: visual CurvyTron is the main optimizer bench.
It is not Atari/ALE and not a stock Atari ROM env. It should be a CurvyTron
visual adapter shaped for LightZero, but the current profiler target is only
`debug_visual_tensor` / `curvyzero_debug_occupancy_gray64/v0`: raw
`uint8[1,64,64]` CHW occupancy smoke frames, optionally normalized to
`float32[1,64,64]` CHW for LightZero-facing payloads. This is not
source-faithful visual truth. The source-backed scalar/ray profile remains
useful as a diagnostic sidecar, not the main training target.

Latest reorientation: Optimizer work is CurvyTron-only. Environment has a
narrow strict public scalar/ray slice, not full visual CurvyTron. Coach/Pong
work is historical context only unless explicitly reopened. Optimizer should
measure the CurvyTron visual adapter path directly and keep scalar/ray numbers
as side evidence.

## Report Contract Patches

- Keep the report small: run provenance, schema IDs, shapes/dtypes,
  denominators, timings, latency, integrity checks, artifact paths, and plain
  caveats.
- Keep canonical timing keys visible with `0` when the lane does not expose a
  bucket.
- Record denominators explicitly: env transitions, player ticks, ego decisions,
  policy rows, MCTS roots/simulations, rollout rows, learner samples, learner
  updates, completed games.
- Keep reward tensors, terminal/truncation counts, and completed games/minute
  as throughput/context. Do not treat returns, win rate, Elo, or best-checkpoint
  fields as optimizer evidence.

## Immediate Profile Sequence

1. Keep profiling the bounded visual smoke/profiler around
   `debug_visual_tensor`, not source-faithful visual truth. The first active
   source baseline after one-pass renderer vectorization is
   `B=32,T=64,stack+copy`, loop `0.0927s`, throughput `22087/s`, render
   `0.0391s`, stack+copy `0.0130s`, normalize `0.0056s`, env step `0.0174s`.
   Reset/startup advance is timed separately; policy/search/replay are not
   included. Do not keep optimizing this debug renderer unless whole-loop visual
   adapter timing still points there.
2. Record raw render, stack update/copy, dtype/range normalization,
   source-lifecycle startup advance, final-frame staging, and replay payload
   size separately.
3. Treat the smallest non-ALE LightZero visual adapter/config smoke as complete
   for no-train plumbing: local tests pass, and installed Modal runtime reset
   and step pass for `curvyzero_debug_visual_tensor_lightzero`.
4. Add per-stage timing for the visual train path: env step, render,
   stack/normalize, collector/preprocess, policy/search, replay push/sample,
   learner, evaluator, checkpoint, and GPU utilization.
5. Keep scalar/ray profiling only when it answers a visual-adapter question or
   checks source geometry. Do not optimize rays as the main path.
6. Keep process sharding as a later scout with p95/p99 latency and policy
   staleness, not just rows/sec.
7. Mirror the same metadata in any CurvyTron LightZero-shaped profile where
   LightZero exposes it: checkpoint id, config, wrapper boundary,
   collector/evaluator counts, search settings, target metadata, and plain
   caveats.
8. Add one CurvyTron LightZero-specific profile report before making
   distributed architecture claims: setup/import/env creation, collect, replay
   push, replay sample/target construction, learner update, GPU utilization,
   envstep/sec, train_iter/sec, and update count. Prefer a separate direct
   one-collect/sample/train harness with no evaluator.
9. For strict native CurvyTron trainer profiling, the first batch-array writer
   and body-cursor ray slice are wired and parity-tested. Treat that as a real
   speed cleanup inside the strict slice, not a reason to generalize beyond it.

Profiler status: the old Pong profiler is archived context only. The active
clean timing path is a CurvyTron visual direct harness.

## Concrete Microbench Matrix

- Source scalar scout:
  `uv run python scripts/benchmark_source_env.py --repeats 10 --js --js-repeats 2 --json`
- Trainer-shaped dry run:
  `uv run python scripts/repo_native_ppo_actor_loop_dry_run.py --batch-size 4 --rollout-steps 8 --artifact-root /private/tmp/curvy-repo-native-ppo-actor-loop-dry-run-smoke --format json`
- Policy/search CPU stand-in:
  `uv run python scripts/benchmark_policy_search_batch_standin.py --env-batch 64 --players 2 --obs-dim 106 --hidden-dim 128 --action-count 3 --simulations 4 --decision-batches 20 --warmup 2 --copy-mode copy --format json`
- Heavier search stand-in:
  same command with `--simulations 32`.
- Strict native vector trainer profile:
  `uv run python scripts/benchmark_vector_trainer_actor_loop_profile.py --batch-size 8 --rollout-steps 64 --artifact-root /private/tmp/curvy-vector-trainer-b8-t64-profile --format json`
- Optional learner smoke:
  `uv run python scripts/repo_native_ppo_learner_smoke.py --batch-size 4 --rollout-steps 8 --artifact-root /private/tmp/curvy-repo-native-ppo-on-policy-smoke --format json`
- CurvyTron debug visual adapter timing:
  `uv run python scripts/benchmark_debug_visual_lightzero_adapter.py --steps 64 --seed 5`

Preferred LightZero-shaped optimizer path before more remote runs: build a
CurvyTron debug visual one-collect/sample/train harness that skips evaluator
startup entirely. Do not run Pong for this lane.

## Next Script

Target script:

```text
scripts/benchmark_source_trainer_actor_loop_profile.py
```

Target command:

```text
uv run python scripts/benchmark_source_trainer_actor_loop_profile.py \
  --batch-size 2 --rollout-steps 2 --step-ms 16.6666667 \
  --artifact-root /private/tmp/curvy-source-trainer-b2-profile \
  --format json
```

The first version exists. It writes `profile_report.json` and one
trainer/replay-shaped `.npz`, then read-validates the file. It does not train,
score, promote a framework, or claim broad source fidelity.

Implementation guard: use
`src/curvyzero/env/source_trainer_adapter.py`. The current best local CurvyTron
profile mode is `source_world_bodies_circle_rays_v0`, not the older center-cell
approximation. It uses source world body circles plus avatar body metadata
sidecar and packs all-player trainer rows as `float32[B,2,106]`.

Short body/trail command:

```text
uv run python scripts/benchmark_source_trainer_actor_loop_profile.py \
  --batch-size 8 --rollout-steps 64 \
  --occupancy-policy source_world_bodies_circle_rays_v0 \
  --source-setup-mode controlled_trail \
  --source-drive-mode direct_step \
  --artifact-root /private/tmp/curvy-source-trainer-b8-t64-circle-rays \
  --format json
```

This deliberately profiles observation packing with nonempty source bodies. It
does not measure natural reset/spawn fidelity, browser-visible trail history,
bonus geometry, real policy/search, or learner work.

Latest short source profile refresh:

- `B=8,T=64`: loop `0.392s`, env step `0.011s`, source adapter `0.058s`,
  observation `0.318s`, ray cast `0.272s`, throughput `1306.6/s`.
- `B=16,T=64`: loop `0.783s`, env step `0.024s`, source adapter `0.120s`,
  observation `0.632s`, ray cast `0.540s`, throughput `1307.3/s`.
- `B=32,T=32`: loop `0.680s`, env step `0.019s`, source adapter `0.078s`,
  observation `0.577s`, ray cast `0.490s`, throughput `1506.7/s`.

Read: ray casting remains about `69-72%` of loop time, with total observation
packing larger still. Env step is not the current bottleneck.

Next local choices should now be stricter:

- validate circle-ray source body geometry against fidelity expectations and
  keep optimizing only if production-like profiles still point at ray casting;
- calibrated real policy/search timing on the same `[B,2,106]` rows;
- replay/learner handoff timing with real trainer payloads.

Do not keep polishing the center-cell no-train observation path.

Matched CPU policy/search overlay still says real policy/search timing is
needed before final Amdahl claims. Fake NumPy search at `32` simulations was
`0.036s` for `B=8,T=64` and `0.0146s` for `B=32,T=16`. This does not replace
real Mctx/LightZero timing.

Native vector corrected profile: `B=8,T=64` loop `0.259s`, public env.step
`0.252s`, throughput `1980/s`; `B=16,T=64` loop `0.582s`, step `0.572s`,
`1760/s`; `B=32,T=32` loop `0.640s`, step `0.635s`, `1600/s`; `B=128,T=64`
loop `6.846s`, step `6.830s`, `1197/s`. One-pass ray probes were
`0.0035s`, `0.0069s`, `0.0124s`, and `0.0561s` for `B=8/16/32/128`. The
optimizer read is unchanged: native vector helps remove adapter overhead, but
the strict public step is still observation/ray-bound, and larger CPU batch is
not automatically better.

Post batch-array writer profile with corrected phase probe: `B=8,T=64`
throughput `1993/s`, `B=16,T=64` `2375/s`, `B=32,T=64` `2241/s`,
`B=128,T=16` `2493/s`. Public env.step still dominates; the batch observation
probe is mostly ray casting. The `B=128` longer run hit a terminal row, so use
the `T=16` row only as a short speed probe. Next implementation bet is
preallocated/batched ray observation with scalar-row parity tests, not replay
or policy-row mapping.

Post body-cursor slice profile: `B=8,T=64` `3038/s`, `B=16,T=64` `2599/s`,
`B=32,T=64` `3108/s`, `B=128,T=16` `3399/s`. This confirms that unused native
body-buffer tail was a real ray-path cost. Next stop condition: do not keep
deepening CPU ray work blindly. First calibrate real model/search/Mctx on the
same faster `[B,2,106]` rows; continue ray batching only if search remains
small and public env.step is still the largest comparison-valid bucket.

Post-review patch status: the native vector profile report now has the schema,
status, debug-event mode, env-step timing notes, pre-step mask violation checks,
positive selected-action-weight checks, straight-action fallback fix,
nonnegative seed validation, and focused tests. Treat this as report
correctness cleanup, not a new throughput conclusion.

## LightZero Profiler Stop Rule

Do not run a full faithful-short training rung merely to profile phase shares.
Use the profiler hook cap:

- hook sanity: `profile_stop_after_learner_train_calls=1` to `2`;
- coarse phase shares: `profile_stop_after_learner_train_calls=5` to `20`;
- only scale after the profile is too noisy to rank buckets.

`max_env_step_override` is not enough by itself because LightZero can spend a
long time in learner updates after collection/eval. `max_train_iter_override`
also did not cap the actual hot learner loop in the `iter5` attempt. Treat
profiler-stopped runs as optimizer profile/control runs, not exact
reproductions and not coach learning evidence.

## Stop Conditions

- Stop env-only optimization if policy/search exceeds env plus obs plus reset in
  the first production-like report.
- Stop framework conclusions if denominators, observation/action/reward schema,
  or hidden work differ.
- Stop replay conclusions if the run writes only one chunk or uses debug
  observations/rewards.
- Stop learner conclusions unless coach has accepted the run as learning
  evidence.
