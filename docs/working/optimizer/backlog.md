# Optimizer Backlog

Date: 2026-05-09

## Next

- Keep the optimizer [world model](world_model_2026-05-09.md) current as other
  lanes produce evidence.
- Use [CurvyTron native LightZero profile](curvytron_native_lightzero_profile_2026-05-11.md)
  as the current optimizer timing source for the source-state visual
  `train_muzero` path. Old debug visual and scalar-ray profiles are historical
  or diagnostic unless reopened.
- Requested no-death long-survival rerun is now unblocked for optimizer timing.
  The old natural-bonus blocker is historical/resolved:
  [environment handoff: natural bonus runtime blocker](environment_handoff_bonus_runtime_blocker_2026-05-11.md).
  Current matched profile facts: c16/sim16 collected `3840` steps in `56.86s`;
  c32/sim16 collected `7680` steps in `74.76s`; both reached replay and `5`
  learner train calls. Keep death suppression labeled
  `profile_only_not_source_fidelity`.
- LightZero env manager optimization is now active: default
  `env_manager_type=subprocess`. Matched long no-death throughput:
  base c16/c32/c64 = `67.5`, `102.7`, `140.3` steps/s; subprocess
  c16/c32/c64 = `79.7`, `146.3`, `225.7` steps/s. Use `base` for detailed env
  timers only; use `subprocess` for Coach-facing training speed.
- Current simple speed knobs for native CurvyTron runs: `collector_env_num` and
  `n_episode` together (`32/32` as the conservative fast default after the
  fresh width sweep; `64/64` for throughput stress), sparse checkpointing, and
  `--env-telemetry-stride 50` or larger when dense per-step action JSONL is not
  needed. Do not treat sampled telemetry histograms as full action counts.
- Keep evaluator costs separated in reports. Stock LightZero in-loop eval is
  controlled by `--lightzero-eval-freq` and can be skipped in optimizer profile
  mode with `--skip-lightzero-eval-in-profile`; checkpoint eval/inspection/GIF
  is the separate spawned Modal path tied to checkpoint artifacts. Do not mix
  the two in Amdahl reads.
- Current next bottleneck branch: long-survival search plus collection. Fresh
  post-churn subprocess width sweep at sim16/source_max_steps=240 measured
  `152.35`, `192.46`, and `227.53` steps/s for `16/16`, `32/32`, and `64/64`.
  Wider collection still helps, but with diminishing returns. Fresh sim-budget
  contrast shows c64/sim4 at `344` steps/s, c64/sim16 at `221` steps/s, and
  c32/sim50 at `88` steps/s. Next serious branch is searched actor-chunk fanout
  from a frozen checkpoint; do not expect another easy single-process 10x from
  the current knobs.
- Keep the compact [runtime verdict](runtime_verdict_2026-05-10.md) current
  when source profile numbers, CPU/GPU boundary evidence, or full-GPU rewrite
  stance changes.
- Reorientation: visual CurvyTron is the main training path. The optimizer
  target is LightZero-style visual input and conv-stack plumbing, not the
  source-backed `[B,2,106]` scalar-ray path. Keep scalar-ray as a sidecar
  diagnostic and timing comparison only.
- Historical note: the old `debug_visual_tensor` /
  `curvyzero_debug_occupancy_gray64/v0` surface was a smoke target only. The
  active trainer/profile surface is now source-state visual stack
  `curvyzero_source_state_gray64_stack4_player_perspective/v1`.
- Current visual profiler now advances source lifecycle before timing the loop.
  Keep `startup_advance_ms`, source random mode, density, and stage-inclusion
  booleans in every readout. The old inactive one-pixel smoke numbers are
  deprecated.
- Active debug visual smoke read after one-pass renderer vectorization:
  `B=32,T=64,stack+copy` gives about `22.1k` transitions/s. This is good enough
  for debug adapter plumbing. Stop polishing debug pixels unless the real
  visual adapter whole-loop profile still points there.
- Next optimizer bottleneck work must move up the loop: non-ALE LightZero
  visual adapter smoke, then policy/search/replay/learner/eval timing. The
  debug visual renderer no longer deserves a standalone optimization lane.
- Non-ALE debug visual LightZero adapter smoke now passes locally and in the
  installed Modal runtime. A first bounded installed profile also reaches
  LightZero eval-mode MCTS/search, builds visual replay rows/samples, and runs
  `MuZeroPolicy.learn_mode.forward` for loss under a no-op optimizer, scheduler,
  and target-update patch. Next work is not another env smoke; it is the real
  LightZero collector/GameBuffer boundary with the same explicit
  no-learning/profiling labels.
- Two-seat CurvyTron LightZero-shaped samples now preserve
  `iteration/env_row_id/player_id/decision_index` metadata into the learner
  adapter. When that metadata is present, `_learn_mode_batches` trains
  `target_value` on remaining discounted survival return instead of the legacy
  `[0.0, immediate_reward]` placeholder, and two-seat samples request a bounded
  learner batch equal to the sampled row count instead of silently using the
  hard-coded policy config batch size of `2`.
- Direct local debug visual adapter timing now exists:
  `scripts/benchmark_debug_visual_lightzero_adapter.py --steps 512 --seed 5`
  measured about `7416` env-step transitions/s. This excludes trainer
  frame-stack consumption, policy/search, replay, learner, and eval, so do not
  optimize from this number alone.
- Current measured source-backed baseline should use
  `source_setup_mode=controlled_trail` when timing body/ray geometry. Default
  source setup can produce zero body circles and overstate ray-path throughput.
- Next full-loop measurement is the native source-state visual LightZero
  `train_muzero` path. Report env step, render, stack/normalize, policy/search,
  replay, reset, learner, checkpoint, and checkpoint/policy-version metadata
  when using actor chunks. The current stock loop is synchronous; actor-fleet
  freshness only applies to future split collection.
- Current post-reorientation order for CurvyTron optimizer work:
  1. Keep the native source-state `train_muzero` path as the active trainer and
     profile surface.
  2. Use subprocess env manager and wider collector batches first: `32/32`,
     then `64/64` sweeps.
  3. Add a profile-only scripted-survivor stress mode only if true
     long-survival timing is needed.
  4. Promote visual-fidelity claims only through Environment-owned evidence.
  5. Scout actor/search fanout only with explicit throughput, queue/transfer
     cost, replay age, and checkpoint-freshness metadata.
- `curvytron_vector_trainer_sample` exists in `mctx_synthetic_benchmark.py`
  and ran once on Modal. Keep it as the small bridge between native
  observation/ray timing and Modal GPU search timing; next use it for batch
  sweeps or a real learned-model/search swap, not another synthetic-only mode.
- Preserve the Environment boundary: strict `VectorTrainerEnv1v1NoBonus` is
  usable for speed plumbing, but source fidelity, broader lifecycle, 3P/4P,
  bonuses, visual rendering, and full LightZero adapter semantics remain
  Environment/RAM-owned.
- Preserve the Coach boundary: Pong is historical/control context only from the
  optimizer lane. Do not run or post-process Pong unless the user explicitly
  reopens that lane.
- Build CurvyTron-only LightZero-shaped timing next: native source-state
  `train_muzero` timing is now the active baseline. Next useful timing is either
  a long-survival profile-only stress run or a collect/search fanout probe.
  Preserve provenance/version metadata and keep it labeled setup/profile, not
  learning evidence.
- Keep old hook-stopped stock Pong profile numbers only as archived evidence for
  synchronous LightZero shape. Do not use them as the active optimizer path.
  Minimum CurvyTron timers are setup/import/config/env creation, collect/search,
  replay push/remove, replay sample/target construction, learner update,
  checkpoint save if included, artifact scan/Volume commit, envstep/sec,
  train_iter/sec, update count, checkpoint bytes, and GPU sampled utilization.
- Keep the one-Modal-function stance for Coach-facing stock-loop runs until a
  profile shows a concrete reason to split actor, learner, inference/search,
  replay, or eval. Candidate future splits need explicit latency,
  queue/Volume transfer cost, checkpoint-freshness metadata, and failure-mode
  measurements.
- After the first LightZero phase profile, choose the next optimization branch
  by evidence:
  - if checkpoint/eval/artifact work is material, fix cadence/layout and keep
    it as coarse separate jobs;
  - if collect/search dominates and GPU is underused, investigate in-container
    central inference/search with micro-batching before any Modal-level split;
  - if learner dominates while collection waits, investigate same-container
    actor/learner overlap with an in-memory bounded queue and explicit
    checkpoint-version metadata;
  - if replay sample/target construction is large, split target construction
    from learner timing and consider background/reanalyse work only if it uses
    otherwise idle compute;
  - if CPU env/preprocessing dominates, sweep `collector_env_num`/batch knobs
    and only then consider process sharding.
- If LightZero collect/search dominates, investigate a second-level hook inside
  `MuZeroMCTSCtree.search`: C++ tree traverse/backprop, latent-state gathering,
  H2D tensor creation, recurrent inference CUDA time, D2H detach-to-NumPy, and
  root inference in MuZero policy collect/eval. Do not add this before the
  coarse profiler proves search is worth the extra intrusion.
- If LightZero collect/search dominates, concrete next hook targets from the
  internals review are `MuZeroPolicy._forward_collect`, `_forward_eval`, runtime
  model `initial_inference`/`recurrent_inference`, `MuZeroMCTSCtree.search`,
  env-manager `step`, and `MuZeroGameBuffer.sample` helpers such as batch/target
  construction. Discover exact class owners inside the Modal image with
  `inspect` because the dependencies are not importable locally.
- For CurvyTron, keep the repo-native CPU source/trainer path as the current
  optimizer bench:
  `CurvyTronSourceEnv -> source_snapshot_to_vector_trainer_state ->
  observe_vector_1v1_egocentric_rays_v0 -> [B,2,106] obs + [B,2,3] masks`.
  `vector_runtime` may become the speed kernel, but raw kernel rows are not
  full training-loop throughput.
- Keep `scripts/benchmark_vector_trainer_actor_loop_profile.py` as the strict
  native vector comparison bench for `VectorTrainerEnv1v1NoBonus` only. It
  should continue reporting public `[B,2,106]` obs/masks, policy-row mapping,
  replay-v0 chunk timing, and caveats that source-backed JS/`CurvyTronSourceEnv`
  remains the oracle.
- Preserve the native reset guard: larger `B` warmup timer callback caps should
  stay scaled, and the `B=128` reset regression should remain in focused test
  runs.
- Keep the [MuZero loop bottleneck map](muzero_loop_bottleneck_map_2026-05-09.md)
  and [profile next steps](profile_next_steps_2026-05-09.md) current as new
  profile reports land.
- Keep optimizer-owned report output lean: canonical timers, latency summaries,
  denominator counts, integrity checks, checkpoint/version metadata for actor
  chunks, artifacts, and plain caveats.
- Sketch the first project-owned PPO/CleanRL-style CurvyTron runner contract:
  PettingZoo Parallel-shaped env boundary, rollout buffer fields, scorecard
  outputs, and profiling buckets.
- Summarize when debug events are allowed in benchmarks versus excluded from
  the training hot path.
- Track the handoff from debug obs/reward packing to trainer-facing
  obs/reward/replay contracts once the environment lane exposes them.
- Write stop criteria for switching from an owned PPO runner to LightZero/Mctx
  search work based on measured loop cost and coach-lane learnability evidence.
- Maintain `source_world_bodies_circle_rays_v0` as the current repo-native
  scalar-ray diagnostic baseline. It is source-backed and trainer-shaped
  (`[B,2,106]`), not Atari/ALE/LightZero and not the primary visual target.
  Keep phase timers and `source_body_trail`/sidecar metadata visible in every
  report. Score quality remains explicitly out of scope.
- Keep the CurvyTron interface stance explicit: primary optimizer work is the
  native source-state visual LightZero trainer; old non-ALE
  `debug_visual_tensor` smoke/profiler and source-backed `[B,2,106]` trainer
  rows are diagnostics. Those scalar rows are wrapper observations over source
  state, not native source objects.
- Keep `source_body_trail` counters required in every source/trainer report:
  body counts, body-circle counts, nonzero occupancy counts, and own/opponent
  occupied-cell counts. Default center-cell runs can still be empty-body runs.
- Use `source_setup_mode=controlled_trail` as the repeatable nonempty body/trail
  baseline. Keep the label explicit because it is not natural reset/spawn
  evidence.
- Do not continue broad center-cell optimization. Next CurvyTron branches are:
  validate/optimize circle-ray body geometry if fidelity and profiles justify
  it; add calibrated real policy/search timing on the same `[B,2,106]` rows;
  or time trainer replay/learner handoff.
- Batched two-ego observation writer is complete and worth keeping. It preserves
  scalar parity, row order, masks, rewards, `to_play`, and copied public arrays;
  focused tests, `ruff`, and `py_compile` pass. It cleared the controlled-trail
  stop rule on larger profiles. Next optimizer work should not keep polishing
  center-cell observation unless newer production-like timing says to.
- Add replay write timing for trainer payloads, not just debug replay-v0-shaped
  payloads.
- Modal disaggregation review is complete: keep hot loop one function; split
  coarse eval/probe/artifact jobs only. Next architecture experiments are a
  coarse split A/B and an in-container actor/search handoff microbench.
- Matched CPU policy/search overlay is complete for the current row shape. It
  did not displace observation/ray work, but it did show that larger policy
  batches matter. Next policy/search work should be calibrated real Mctx,
  LightZero, or learner timing, not another synthetic integration.
- Before a GPU raycaster rewrite, test the nearer-term architecture: source
  rows stacked into the native batch observer, CPU env/obs producers feeding
  GPU model/search, larger sync batches, and process sharding with p95/p99
  action latency and policy staleness recorded.
- Do not assume bigger native CPU batch is better. The corrected strict vector
  profile slowed from `1980/s` at `B=8,T=64` to `1197/s` at `B=128,T=64`;
  future batch sweeps need latency and policy freshness, not throughput alone.
- Do not claim the observation problem is solved after the wall-hit patch. It
  improved strict native `B=32,T=64` to `5046/s` and source-backed `B=8,T=64`
  to `1748/s`, but source-backed observation/ray time is still the dominant
  measured bucket.

## Later

- Compare CPU vector actor loop versus CPU env plus GPU model/search once real
  model/search timing replaces the NumPy stand-in.
- Add a concise CPU actor-pool versus bigger-sync-batch decision note after the
  single-container report exists.
- Write a Modal boundary note: whole-job sweeps only, no per-step/per-player
  remote calls.
- Capture stop criteria for GPU env, JAX env, C++/Rust, EnvPool-style backends,
  and distributed actors. Current verdict: no known fundamental blocker to full
  GPU env/obs/model/search, but it requires a new tensor runtime plus parity
  tests because the source env is a CPU object graph.

## Waiting On Other Lanes

- Environment fidelity: broader reset/timer/autoreset, final observation,
  trainer-facing observation/reward, and real rollout contracts.
- Training: LightZero replication/control status, checkpoint quality, and
  whether policy-only, LightZero, Mctx, or another learner path has useful
  learning evidence.
- Replay: production write timing and learner handoff, not just in-memory staging.
