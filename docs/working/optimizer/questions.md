# Optimizer Questions

Date: 2026-05-09

## Open

- Does `source_world_bodies_circle_rays_v0` match Environment/RAM fidelity
  expectations closely enough for optimizer speed decisions, or does it need a
  visible-trail/bonus geometry upgrade first?
- If circle-ray observation remains the top bucket in a production-like loop,
  should the next speed path batch it further, move it into a vector runtime
  surface, or wait for real policy/search timing?
- Can source-backed circle-ray rows be stacked into the native batch observer
  without changing parity against the current scalar source observation?
- Is dense exact ray-circle batching memory-safe enough at realistic body
  counts, or does it need a compiled CPU kernel / broad phase?
- What minimum calibrated model/search timing should replace the NumPy stand-in:
  LightZero MCTS, project-owned Mctx, or a fixed proxy from measured runs?
- What batch sizes keep GPU/search useful without hurting p95/p99 action
  latency or policy freshness?
- Does the toy process-shard speedup transfer to source-backed CurvyTron with
  `[B,2,106]` observation packing, real policy/search, and replay/learner
  handoff, or does observation/raycast/IPC erase the gain?
- Does the strict native vector trainer path stay ray-bound after source
  adapter removal once calibrated policy/search is in the same report?
- When should debug event rows be sampled, disabled, or moved out of the hot
  path?
- What exact replay write path should be timed first: local `.npz`, Modal
  Volume, object storage, or learner-adjacent queue?
- What policy-staleness budget is acceptable for the first async or actor-pool
  experiment?
- What evidence should promote or demote the repo-native PPO/IPPO measurement
  hypothesis versus LightZero, Mctx, TorchRL, or Sample Factory?
- In the LightZero Pong control lane, what share of wall time is collector/env,
  MCTS/search, replay sample/target construction, learner update,
  checkpoint/eval, artifact scan/Volume commit, and setup?
- During LightZero collect/search/learn, is the `L4` GPU actually busy, or is
  the job mostly waiting on CPU env/search/replay/checkpoint work?
- Should any LightZero or CurvyTron training piece leave the single Modal
  container? Only answer after measuring the single-container loop and the
  added latency/staleness of any proposed split.
- What does the direct one-collect/sample/train LightZero harness show, and is
  a hook-stopped stock run still needed only to quantify startup/evaluator tax?
- After the strict native ray cleanup, does real Mctx/LightZero/model-search
  timing displace env/observation as the first optimizer bottleneck?
- Can the source-backed circle-ray profile be refreshed with current
  Environment/RAM caveats and still show ray/observation as the dominant
  oracle-adjacent cost?

## Answered For Now

- Is CurvyTron observation a screenshot? The primary target is visual, but the
  current visual surface is only debug occupancy:
  `curvyzero_debug_occupancy_gray64/v0` raw `uint8[1,64,64]` CHW, optionally
  normalized to `float32[1,64,64]` CHW for LightZero-facing payloads. It is not
  source-faithful visual truth.
- Is current CurvyTron optimizer work Atari/ALE? No. The primary visual hook is
  non-ALE `debug_visual_tensor` smoke/profiling. Source-backed trainer rows
  remain scalar-ray diagnostics: `CurvyTronSourceEnv` snapshots adapted into
  `float32[B,2,106]` ray/scalar observations plus `bool[B,2,3]` masks and
  replay-v0. ALE is only for official Atari Pong control.
- Is there a fundamental blocker to full GPU env/obs/model/search? No known
  fundamental blocker, but the current source env is a CPU object graph. A full
  GPU path means a new tensor runtime plus parity tests, not a local patch.
- Is LightZero all GPU? No. It mixes CPU subprocess envs, ALE/preprocessing,
  replay/checkpoint/eval/artifact work, CPU MCTS tree/control, GPU Torch
  model/learner calls, and host/device movement.
- Does current source profiling still point at ray observation? Yes. The latest
  refresh has ray casting about `69-72%` of loop time across `B=8,T=64`,
  `B=16,T=64`, and `B=32,T=32`; env step remains tiny.
- Does the latest local source refresh still agree? Yes, with the current
  controlled-trail source-backed circle-ray run at `B=8,T=64`: loop `0.471s`,
  observation packing `0.381s`, ray cast `0.322s`, env step `0.013s`, and
  throughput `1087/s`. That run wrote
  `/private/tmp/curvy-source-trainer-b8-t64-refresh-20260510/profile_report.json`.
- Does the strict native vector profile remove the bottleneck? No. It removes
  source-adapter cost, but corrected `VectorTrainerEnv1v1NoBonus` results are
  still dominated by public env.step/observation work: `1980/s` at `B=8,T=64`
  drops to `1197/s` at `B=128,T=64`, and one-pass ray probes scale from
  `0.0035s` at `B=8` to `0.0561s` at `B=128`.
- Is native vector `1v1/no_bonus` the environment oracle? No. It is strict
  optimizer plumbing evidence with public `[B,2,106]` rows, replay-v0 chunking,
  and a tiny policy/search stand-in. Source-backed JS/`CurvyTronSourceEnv`
  remains the oracle boundary.
- Is larger native CPU batch automatically better? No. The corrected profile
  got slower per transition by `B=128`; batch choice needs p50/p95/p99 latency,
  policy/search timing, and freshness alongside throughput.
- Is the native vector reset/warmup issue guarded? Yes for now:
  `VectorTrainerEnv1v1NoBonus.reset` scales the warmup timer callback cap for
  larger `B`, a `B=128` reset regression exists, and focused validation passed
  (`14 passed`).
- Is D2H action transfer the current Modal Mctx boundary concern? Not in the
  retimed synthetic boundary: steady selected-action D2H median is
  `0.0000471s`, while steady H2D is small but visible and Mctx search is about
  `0.0029s`. This still does not include CPU ray generation or source fidelity.
- Did the new Modal Mctx boundary check change that? Not materially. App
  `ap-EkNEv5A3xDRj7QxZbmeTFe`, synthetic `curvytron_trainer_flat
  B=64,P=2,obs=106,sim=8` on L4, measured steady Mctx `0.00245s`, steady H2D
  `0.00051s`, selected-action D2H `0.0000147s`. Still synthetic; still not
  real CurvyTron obs production or learned dynamics.
- Did feeding real strict native observations into Modal Mctx change the
  Amdahl read? No. App `ap-ZkCdPu0mPNrniXaQAgxDjv` added
  `curvytron_vector_trainer_sample`: real `VectorTrainerEnv1v1NoBonus`
  `[64,2,106]` observations/masks, two no-event straight rollout steps, then
  synthetic Mctx. Host env/obs setup was `0.206s`; steady Mctx was
  `0.00233s`, steady H2D `0.000536s`, action D2H `0.0000157s`. This still
  points at observation/env production before GPU search at the sampled shape.
- Can CurvyTron be hooked up today? Yes for no-train optimizer profiling and
  replay-shape plumbing. The current best local hook uses source body-circle
  rays plus avatar body sidecar metadata into `[B,2,106]` rows. Not yet for a
  final learning claim: browser-visible trail history, broad lifecycle/bonus coverage, real policy/search,
  learner handoff, and production replay still need work.
- Has source body-circle geometry replaced center-cell as the local optimizer
  baseline? Yes. `source_world_bodies_circle_rays_v0` is now the preferred
  local profiling mode. Center-cell is retained as older plumbing evidence, not
  the branch to keep polishing.
- Is the batched two-ego observation writer worth keeping? Yes. It cleared the
  local controlled-trail stop rule on larger profiles and preserves scalar
  parity/copy semantics. It is a modest speed patch, not a final production
  bottleneck answer.
- Does the matched CPU search stand-in change the next optimizer priority? Not
  yet. At `32` fake simulations and `1024` policy rows, the stand-in is
  `0.036s` at `B=8` and `0.0146s` at `B=32`, while source/trainer observation
  packing is still `0.327s` and `0.274s`. Treat this as a CPU proxy only.
- What did the latest Environment review change? It tightened the boundary:
  `VectorTrainerEnv1v1NoBonus` is a real strict `1v1/no_bonus/P=2` trainer
  slice with final observation/reward, terminal metadata, truncation,
  autoreset staging, and replay-v0 plumbing. It is not broad lifecycle, 3P/4P,
  bonuses, visual LightZero, or full CurvyTron.
- What did the latest Coach review change? It made official/control LightZero
  Atari Pong a useful training-pipeline reference, not a CurvyTron claim.
  Current speed evidence says evaluator/collector/env/MCTS dominate more than
  learner GPU bulk, so CurvyTron should also be profiled as a whole actor/search
  loop before deeper env-only optimization.
- Is `body_write_cursor` slicing safe? Safe for native vector state when the
  cursor is the authoritative live-prefix length. Importers/profilers must keep
  cursor and `body_active` consistent; source-adapter states currently fall
  back when no cursor exists.
- Can observation be optimized? Yes. The wall-hit/normalization vectorization
  pass improved strict native `B=32,T=64` from about `2985/s` to `5046/s` and
  source-backed `B=8,T=64` from `1087/s` to `1748/s`. That is not the end of
  the problem: source-backed observation is still the largest bucket, so the
  next target is source-row stacking and dense exact circle-ray work.

## Boundaries To Preserve

- Environment lane answers source behavior, vector parity, unsupported source
  semantics, and final observation contracts.
- Training lane answers eval gates, checkpoint quality, target correctness, and
  learner choice.
- Optimizer lane answers setup tradeoffs, measurement shape, bottleneck
  prioritization, and when a systems rewrite is justified.

## Useful References

- [Environment active lanes](../environment/active_lanes.md)
- [Optimizer runtime verdict](runtime_verdict_2026-05-10.md)
- [Self-play speed lane](../environment/selfplay_speed_lane_2026-05-09.md)
- [Training-loop bottlenecks and Amdahl's law](../../research/training_loop_bottlenecks_amdhals_law_2026-05-09.md)
- [Training state index](../training_state_index_2026-05-09.md)
