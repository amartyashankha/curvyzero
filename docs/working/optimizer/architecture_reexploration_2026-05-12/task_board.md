# Architecture Re-Exploration Task Board

Date: 2026-05-12

## Now

- [x] Create a clean workspace for the architecture investigation.
- [x] Collect current stock LightZero dataflow facts.
- [x] Compare external full-system Zero frameworks.
- [x] Write a CurvyTron-specific bottleneck map.
- [x] Decide the next smallest architecture experiment.
- [x] Finish large-scale Zero architecture research.
- [x] Finish collect-only fanout prototype design.
- [x] Finish MCTX visual-root benchmark plan.
- [x] Generate the first stock-frozen profile tensor manifest.
- [x] Launch and summarize the profile tensor validation slice.
- [x] Fix or choose a low-overhead profile summary readback path before wide
  detached grids.
- [x] Run first function-readback profile wave: collector width, MCTS slope,
  long-render pair, and dense reward bookkeeping.
- [x] Run second function-readback profile wave: C128/C160 width, C32 long
  render, fixed-straight opponent ablation, CPU64, and C96 dense reward.

## Candidate Experiments

- [ ] Stock LightZero iteration anatomy profile with worker telemetry enabled.
- [x] Collector width ladder with worker telemetry: C16, C32, C64, C96.
- [x] Follow-up collector width ladder: C128/C160.
- [x] Follow-up render lens: C32 no-death browser vs fast.
- [x] Frozen-opponent cost ablation: C32 and C96 fixed-straight timing lenses.
- [ ] Coarse synchronous fanout sketch: N collect jobs from one frozen checkpoint,
  merge chunks, then learner step.
- [ ] MCTX visual-root benchmark over `[B, P, 4, 64, 64]` roots.
- [ ] MiniZero quick-run control, if setup cost is small enough.

## Active Second Wave

- `large_scale_zero_architectures.md`: what large actor/search/replay/learner
  systems actually separate, and what CurvyTron should copy.
- `collect_only_fanout_design.md`: the smallest repo-shaped way to fan out
  searched collection from one checkpoint without claiming learning. Done; it
  recommends a small repo-owned collect-only entrypoint using LightZero
  `collect_mode`, `GameSegment`, and `MuZeroGameBuffer` compatibility rather
  than fighting `train_muzero` into collect-only mode.
- `mctx_visual_root_benchmark_plan.md`: a concrete benchmark for batched
  CurvyTron visual roots through MCTX/JAX. Done; next action would be an
  isolated Modal script, not trainer integration.

## Current Near-Term Bet

Do not migrate frameworks yet. Keep stock LightZero as the control/proof path.
The immediate next work is the stock-frozen profile tensor in
`current_profile_tensor.md`. It answers the basic Amdahl question before we
touch architecture again: attribution, collector width, simulation cost, render
surface, and hardware shape.

Validation results are in `profile_validation_results.md`. They say the next
optimizer lane should stay focused on collection-side work: browser observation,
frozen-opponent CPU cost, and collector parallelism. The readback path is now
parent `modal run --detach` plus child `--profile-spawn`, with metrics returned
through `FunctionCall.get()` and no final profile volume commit.

First-wave function-readback results are in `profile_validation_results.md`.
Keep these as optimizer-owned profile runs, not Coach training runs.

Current read: use CPU fanout before chasing larger GPUs, with C96 as the
current default wide starting point. Rendering is important for long
trajectories, MCTS is not yet dominant at the tested sim counts, frozen
checkpoint opponent inference is a real fixed-opponent lane cost, and dense
reward bookkeeping is not a major cost.

After that, the next high-leverage architecture experiment is likely a
collect-only fanout ladder from one checkpoint, because it tests the broader
parallelism question directly:

```text
checkpoint K -> N searched collectors -> chunks -> merge/read -> learner work
```

Run it first as measurement only. Use `N=1,2,4,8` actor jobs, native LightZero
searched `GameSegment` chunks, strict manifests, and a merge/import smoke into
`MuZeroGameBuffer`. If chunk write/merge and later learner work are small, scale
self-play harder. If merge or learner becomes the limiter, more actors alone
will not help.

## Next Smallest Architecture Experiment

Build only the first proof of the fanout path:

1. Modal collect-only actor smoke:
   one frozen checkpoint, one actor, tiny episode count, writes one native
   LightZero searched `GameSegment` chunk plus manifest.
2. Modal merge/import smoke:
   read that one chunk, validate manifest compatibility, push into
   `MuZeroGameBuffer`, optionally sample one batch, and report shapes/timing.
3. Fanout ladder:
   run the same actor job at `N=1,2,4,8` from the same checkpoint and compare
   aggregate decisions/sec, chunk write/sec, merge/import time, and later
   learner work.

Do not train in step 1 or 2. Do not touch live Coach training defaults. Do not
use the old custom two-seat replay path for this proof.

Implementation sketch from read-only code review:

- Add isolated Modal module
  `src/curvyzero/infra/modal/lightzero_curvytron_collect_only_fanout.py`.
- Add small chunk helpers in
  `src/curvyzero/training/lightzero_collect_chunk_v0.py`.
- Expose actor, merge/import, and orchestration smoke functions.
- Start with `base` env manager and `n_episode=1-2`, then test subprocess.
- Chunk format is native same-image LightZero `GameSegment` cloudpickle plus a
  strict manifest. This is a smoke/profiling format, not a long-term replay
  storage decision.
- Merge/import should validate schema/checkpoint/config compatibility, push
  segments into `MuZeroGameBuffer`, and sample only if enough positions exist.

Top risk: LightZero collector, env manager, and replay-buffer constructors must
be verified inside Modal because local imports do not prove the installed remote
surface.

## Frozen Opponent Cost Note

Frozen checkpoint opponent inference is inside each env worker's `step()` path.
With `subprocess` collection, each worker owns its own checkpoint-backed policy
and calls it independently. That explains why fixed-straight timing rows are
much faster.

Safe future improvement: keep subprocess frozen opponents CPU-only, but consider
a batched or centralized frozen-opponent inference service if this fixed-opponent
lane remains important. Avoid simply turning on CUDA inside subprocess env
workers because forked CUDA contexts are unsafe.

## Current Hypotheses

- Short trajectories are not render-bound; collector width and setup overhead
  matter more.
- Long trajectories with `browser_lines` are render/observation-bound.
- Bigger GPUs do not help much while env/render/opponent worker CPU time is the
  main wall-clock limiter.
- Multi-GPU is probably not useful until search/model batches are large enough
  and the framework actually spreads the relevant work.
- The likely large win is not one magic kernel. It is better architecture:
  many actors/search workers, batched inference where possible, replay chunks,
  learner/checkpoint handoff, and sparse artifact work.

## Open Worries

- Frozen-opponent stock training is trusted plumbing, but not true current-policy
  simultaneous self-play.
- Native two-seat replay/target bridge is promising but still gated by Coach.
- If we build actor fanout around frozen opponents, it may speed proof/control
  runs before it solves final self-play.
- Framework migration might cost more than fixing the local architecture.
- MCTX can speed search if search becomes dominant, but it does not solve env,
  replay, learner, or checkpoint orchestration.
- EfficientZero-style Ray architecture is useful evidence that the missing
  piece is system decomposition, not a different reward knob.
