# Optimizer Profiling Log

Date: 2026-05-09

Status: compact log of known timing evidence. Numbers here are setup evidence,
not training-quality claims.

## 2026-05-10 Short Refresh

CurvyTron stacked debug visual survival profile now has an installed LightZero
policy/search plus no-step learn-mode rung. Modal app
`ap-8Y4Ezpvfx7B12WHdtyFeei` ran:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_profile \
  --seed 0 \
  --steps 4 \
  --num-simulations 2
```

Result: `ok=true`, elapsed `6.523788s`, LightZero `0.2.0`, DI-engine `0.5.3`,
torch `2.11.0`, env type
`curvyzero_stacked_debug_visual_survival_lightzero`, observation to policy
`float32[1,4,64,64]`, replay sample `float32[4,4,64,64]`, reward
`curvyzero_survival_time/v0`, no ALE, no `train_muzero`, no real collector or
GameBuffer, no optimizer step, no checkpoint/eval, no learning claim. It used
`MuZeroPolicy.eval_mode.forward` for search and
`MuZeroPolicy.learn_mode.forward` for forward/loss under a no-op patch:
optimizer, scheduler, and target updates were blocked,
`model_parameters_changed=false`, and `model_state_restored=true`.

| Bucket | Seconds |
| --- | ---: |
| LightZero setup | `6.415709` |
| policy/search, 4 eval/search calls | `0.038073` |
| env step + render + stack | `0.001278` |
| replay row build | `0.000184` |
| replay sample batch | `0.000077` |
| learner forward/loss | `0.067410` |

Read: this finally crosses from env-only smoke into CurvyTron visual
policy/search plumbing. It is still not a full training iteration: no real
LightZero collector, GameBuffer, optimizer step, checkpoint, eval, or
source-fidelity visual claim.

Visual debug tensor smoke is now a real active-source plumbing profile, not the
old inactive reset loop. The profiler uses seeded source random tapes, advances
source lifecycle to `startup_advance_ms=3000`, renders
`curvyzero_debug_occupancy_gray64/v0` raw `uint8[1,64,64]`, normalizes a
LightZero-facing `float32[1,64,64]` payload, and optionally updates a
`float32[4,64,64]` stack. It is still `debug_visual_tensor`, non-ALE, and not
source/browser/canvas fidelity.

`scripts/benchmark_curvytron_visual_observation.py --batch-size 32
--rollout-steps 64 --format json`:

| Shape | Loop | Env step | Body snapshot | Render | Normalize | Stack | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=32,T=64,stack+copy` | `0.0927s` | `0.0174s` | `0.0131s` | `0.0391s` | `0.0056s` | `0.0130s` | `22087/s` |
| `B=32,T=64,stack no-copy` | `0.0907s` | `0.0179s` | `0.0137s` | `0.0403s` | `0.0058s` | `0.0083s` | `22583/s` |
| `B=32,T=64,no stack` | `0.125s` | `0.0182s` | `0.0137s` | `0.0816s` | `0.0061s` | `0.0000s` | `16332/s` |
| `B=64,T=64,stack+copy` | `0.193s` | `0.0374s` | `0.0287s` | `0.0808s` | `0.0111s` | `0.0257s` | `21277/s` |

Read: a vectorized debug point-marking pass roughly halved render cost on the
matched active-source smoke (`0.0864s` -> `0.0391s` for `B=32,T=64,stack+copy`).
Render is still the biggest local bucket, but now it is close to env step plus
body snapshot. Normalize is small, and stack-copy costs roughly `0.0047s` over
`2048` frames on the matched `B=32,T=64` run. Reset/startup advance is timed separately
(`~0.07s` for `B=32`, `~0.14s` for `B=64`) and excluded from loop throughput.
Policy/search, replay, learner, and evaluator are explicitly not included.
Density is now nontrivial for a smoke (`mean_world_bodies_per_frame=6.55`,
`max=44`, `mean_nonzero_pixels_per_frame=4.48` at `B=32`), but this is still a
coarse occupancy marker, not a real visual renderer. Previous one-pixel
numbers came from a bad benchmark setup: `reset(warmup_ms=0)` scheduled source
start but the profiler did not drain the timer, so the world stayed inactive.

Latency scout with `--latency-samples` on `B=32,T=64,stack+copy` after the
vectorized renderer: render p50 `15.75us`, p95 `40.54us`, p99 `48.70us`; env
step p50 `4.25us`, p95 `29.71us`, p99 `38.28us`; body snapshot p50 `2.58us`,
p95 `25.83us`, p99 `31.98us`; stack update/copy p50 `6.08us`, p95 `9.19us`,
p99 `12.00us`. Render tails are now much less ugly, and stack-copy is not the
first debug-smoke optimization target.

Current controlled-trail source-backed scalar-ray sidecar refresh from this
session, using nonempty source body circles and no optional observation phase
timers:

| Shape | Loop | Adapter | Observation | Env step | Throughput | Mean body circles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=16,T=64` | `0.297s` | `0.138s` | `0.124s` | `0.028s` | `3448/s` | `22.7` |
| `B=32,T=64` | `0.555s` | `0.279s` | `0.217s` | `0.050s` | `3692/s` | `22.7` |
| `B=128,T=16` | `0.354s` | `0.148s` | `0.150s` | `0.052s` | `5786/s` | `6.7` |

Artifacts:
`/private/tmp/curvy-optimizer-source-controlled-b16-t64-notimers/profile_report.json`,
`/private/tmp/curvy-optimizer-source-controlled-b32-t64-notimers/profile_report.json`,
and
`/private/tmp/curvy-optimizer-source-controlled-b128-t16-notimers/profile_report.json`.

Same setup with phase timers enabled, `B=32,T=64`: loop `0.683s`, adapter
`0.322s`, observation `0.287s`, ray cast sub-timer `0.168s`, env step
`0.057s`, throughput `3000/s`. Phase timers add overhead; use them for bucket
shape, not exact throughput.

Default source setup is a bad ray/body timing baseline right now because it can
produce zero source body circles. A default `B=32,T=64` run measured `8523/s`
but reported `0` mean body circles, so it mostly timed wall/scalar work. Use
`source_setup_mode=controlled_trail` when the question is body/ray geometry.

Repo-native PPO actor shape smoke on the toy scalar env, `B=64,T=64`, ran in
`2.962s` loop time, `1383` env transitions/s, with observation packing
`2.485s`, env step `0.417s`, policy `0.036s`, and rollout write `0.078s`.
This is useful scalar trainer-shape plumbing only; it is not source-backed
CurvyTron and not the visual path.
The optional Torch learner smoke skipped locally because Torch is not installed
in this environment.

Plain read: source-backed scalar-ray CurvyTron can move to sidecar diagnostic
profiling now. The current measured local tax is adapter plus observation
production. The next experiment should add visual-path policy/search and
learner handoff before another scalar-ray env-only optimization pass.

Observation optimization pass 1, wall-hit/normalization vectorization:

- Code: `src/curvyzero/env/vector_trainer_observation.py` now computes wall
  intersections for all `24` ray directions in one NumPy call and normalizes
  hit-distance arrays without a Python list loop. Circle-hit semantics are
  unchanged.
- Guard: `tests/test_vector_trainer_observation.py` compares the new
  vectorized wall/normalization helpers against the old scalar helpers.
- Validation: `ruff` passed for the touched files; vector observation tests
  passed (`16 passed`); vector env/benchmark source/native tests passed
  (`23 passed`).

Strict native vector matrix after the patch:

| Shape | Loop | Public env.step | Throughput | Ray probe |
| --- | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.120s` | `0.112s` | `4269/s` | `0.0019s` |
| `B=16,T=64` | `0.238s` | `0.229s` | `4300/s` | `0.0026s` |
| `B=32,T=64` | `0.406s` | `0.394s` | `5046/s` | `0.0048s` |
| `B=128,T=16` | `0.446s` | `0.435s` | `4589/s` | `0.0164s` |

Source-backed circle-ray matrix after the patch:

| Shape | Loop | Env step | Adapter | Observation | Ray cast | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.293s` | `0.013s` | `0.065s` | `0.209s` | `0.158s` | `1748/s` |
| `B=16,T=64` | `0.626s` | `0.028s` | `0.140s` | `0.450s` | `0.341s` | `1636/s` |
| `B=32,T=32` | `0.565s` | `0.028s` | `0.105s` | `0.427s` | `0.315s` | `1812/s` |

Artifacts:
`/private/tmp/curvy-vector-trainer-b8-t64-wallvec-20260510/profile_report.json`,
`/private/tmp/curvy-vector-trainer-b16-t64-wallvec-20260510/profile_report.json`,
`/private/tmp/curvy-vector-trainer-b32-t64-wallvec-20260510/profile_report.json`,
`/private/tmp/curvy-vector-trainer-b128-t16-wallvec-20260510/profile_report.json`,
`/private/tmp/curvy-source-trainer-b8-t64-wallvec-20260510/profile_report.json`,
`/private/tmp/curvy-source-trainer-b16-t64-wallvec-20260510/profile_report.json`,
and
`/private/tmp/curvy-source-trainer-b32-t32-wallvec-20260510/profile_report.json`.

Plain read: yes, observation can be optimized. This pass roughly halves the
ray probe at `B=32,T=64` and improves strict native throughput from about
`2985/s` to `5046/s`. Source-backed `B=8,T=64` improved from `1087/s` to
`1748/s`. Observation/ray work is still the largest source-backed bucket, so
the next useful local optimization is more ray/body math batching or a
broad-phase candidate filter, not replay.

Current source-backed circle-ray path after source row stacking and source
`body_write_cursor` exposure:

- Code: `scripts/benchmark_source_trainer_actor_loop_profile.py` stacks
  source vector row states and calls
  `observe_vector_1v1_egocentric_rays_batch_arrays_v0` for the circle-ray
  path; `source_snapshot_to_vector_trainer_state` now exposes
  `body_write_cursor` so padded source rows do not scan beyond their live body
  prefix.
- Validation: source benchmark/source adapter/vector observation tests passed
  (`29 passed`); `ruff` passed for touched files.

| Shape | Loop | Env step | Adapter | Observation | Ray cast | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.252s` | `0.012s` | `0.060s` | `0.174s` | `0.144s` | `2035/s` |
| `B=16,T=64` | `0.467s` | `0.023s` | `0.114s` | `0.324s` | `0.277s` | `2194/s` |
| `B=32,T=32` | `0.403s` | `0.020s` | `0.082s` | `0.298s` | `0.255s` | `2540/s` |

Artifacts:
`/private/tmp/curvy-source-trainer-b8-t64-cursor-20260510/profile_report.json`,
`/private/tmp/curvy-source-trainer-b16-t64-cursor-20260510/profile_report.json`,
and
`/private/tmp/curvy-source-trainer-b32-t32-cursor-20260510/profile_report.json`.

Plain read: the adapter/cursor cleanup helps, but observation/ray casting is
still the largest current source-backed bucket. The next observation work is
the exact circle-ray kernel itself: dense/chunked batching or a compiled CPU
kernel, with scalar parity first.

Denominator note: these timing values are total seconds across the whole
profile chunk, not per-step seconds. For example, `B=16,T=64` does `1024` env
row-steps in `0.467s`; `0.324s` of that chunk is observation packing. That is
about `69%` of wall time and about `0.316 ms` observation work per env row-step,
not `0.324s` per step.

Short larger-batch probes on the same current source/cursor path:

| Shape | Loop | Env step | Adapter | Observation | Ray cast | Throughput | Replay file |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=64,T=16` | `0.147s` | `0.017s` | `0.063s` | `0.062s` | `0.024s` | `6985/s` | `996K` |
| `B=128,T=8` | `0.130s` | `0.016s` | `0.054s` | `0.058s` | `0.020s` | `7875/s` | `1.1M` |

Read: larger batches can raise short-chunk throughput, partly because body
counts are lower at short `T` and Python overhead is amortized. This is not yet
a training architecture verdict. Compare with p95/p99 action latency, policy
staleness, real policy/search time, replay ingestion, and longer rollouts
before choosing a production batch size.

Dense batch circle-ray patch, current source-backed actor chunks:

| Shape | Loop | Adapter | Observation | Ray cast | Env step | Throughput | Replay file |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=16,T=64` | `0.255s` | `0.120s` | `0.105s` | `0.057s` | `0.024s` | `4016/s` | `928K` |
| `B=32,T=64` | `0.466s` | `0.233s` | `0.182s` | `0.097s` | `0.045s` | `4392/s` | `1.8M` |
| `B=64,T=32` | `0.353s` | `0.164s` | `0.146s` | `0.063s` | `0.039s` | `5801/s` | `1.8M` |

Validation: combined vector/source/env/profile tests passed (`69 passed`);
`ruff` passed for touched observation/adapter/test files.

Plain read: observation work is still material, but after this patch the next
optimizer question moves up a level. Add real policy/search and learner handoff
before spending another round purely on env observation.

Local optimizer refresh after the latest environment/coach reorientation.
These runs enabled observation phase probes, so compare them as current
component checks rather than exact replacements for earlier no-probe rows.

Source-backed circle-ray matrix, controlled-trail setup, direct source step:

| Shape | Loop | Env step | Adapter | Observation | Ray cast | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.471s` | `0.013s` | `0.069s` | `0.381s` | `0.322s` | `1087/s` |
| `B=16,T=64` | `0.875s` | `0.026s` | `0.132s` | `0.708s` | `0.603s` | `1170/s` |
| `B=32,T=32` | `0.831s` | `0.024s` | `0.095s` | `0.706s` | `0.595s` | `1233/s` |

Artifacts:
`/private/tmp/curvy-source-trainer-b8-t64-refresh-20260510/profile_report.json`,
`/private/tmp/curvy-source-trainer-b16-t64-refresh-20260510/profile_report.json`,
and
`/private/tmp/curvy-source-trainer-b32-t32-refresh-20260510/profile_report.json`.
Replay write plus read-validate stayed tiny in these runs. This is
oracle-adjacent source plumbing, not full source fidelity or training.

Strict native vector matrix, `VectorTrainerEnv1v1NoBonus`, no-event:

| Shape | Loop | Public env.step | Policy map+forward | Replay record | Chunk build | Throughput | Ray probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.207s` | `0.200s` | `0.0037s` | `0.0010s` | `0.0081s` | `2470/s` | `0.0028s` |
| `B=16,T=64` | `0.405s` | `0.395s` | `0.0047s` | `0.0012s` | `0.0099s` | `2529/s` | `0.0063s` |
| `B=32,T=64` | `0.686s` | `0.675s` | `0.0057s` | `0.0014s` | `0.0177s` | `2985/s` | `0.0104s` |
| `B=128,T=16` | `0.610s` | `0.604s` | `0.0035s` | `0.0005s` | `0.0171s` | `3360/s` | `0.0306s` |

Artifacts:
`/private/tmp/curvy-vector-trainer-b8-t64-refresh-20260510/profile_report.json`,
`/private/tmp/curvy-vector-trainer-b16-t64-refresh-20260510/profile_report.json`,
`/private/tmp/curvy-vector-trainer-b32-t64-refresh-20260510/profile_report.json`,
and
`/private/tmp/curvy-vector-trainer-b128-t16-refresh-20260510/profile_report.json`.
Integrity checks passed on the inspected `B=32,T=64` row with zero
masked-action and NaN/Inf violations.

- CPU policy/search stand-in:
  `B=64,P=2,obs=106,hidden=128,sim=32,batches=20`, elapsed `0.041s`,
  recurrent fake-search loop `0.0396s`, `31k` env decisions/s. This is only a
  NumPy shape/copy scout.
- Modal Mctx synthetic boundary:
  app `ap-EkNEv5A3xDRj7QxZbmeTFe`, `curvytron_trainer_flat
  B=64,P=2,obs=106,sim=8,hidden=64,depth=8`. L4 GPU, steady Mctx median
  `2.454ms`, steady H2D median `0.511ms`, host obs setup `0.444ms`, selected
  action D2H median `0.0147ms`, compile+first `6.898s`.
- Modal Mctx native-observation boundary:
  app `ap-ZkCdPu0mPNrniXaQAgxDjv`, `curvytron_vector_trainer_sample
  B=64,P=2,obs=106,sim=8,hidden=64,depth=8`, no-event, two straight rollout
  steps from `VectorTrainerEnv1v1NoBonus`. Host observation setup `0.206s`
  total: env init `0.0084s`, reset `0.0608s`, rollout steps
  `[0.0665s, 0.0704s]`, policy-row mapping `0.000097s`. Steady Mctx median
  `2.330ms`, steady H2D median `0.536ms`, selected-action D2H median
  `0.0157ms`, compile+first `5.669s`. Output shape was real strict native
  `[64,2,106]` with `128` live roots and zero padded roots.

Plain read: the refreshed source-backed profile is still dominated by
observation/ray production. The synthetic GPU search boundary is small enough
that it does not justify a full GPU env rewrite by itself, but it also is not a
real learned model/search integration. The native-observation Modal run now
confirms that feeding real strict native observations into synthetic Mctx does
not make search the first measured tax at this shape; host env/obs setup still
dominates the sample.

Source-backed CurvyTron profile refresh from Averroes:

| Shape | Loop | Env step | Adapter | Observation | Ray cast | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.392s` | `0.011s` | `0.058s` | `0.318s` | `0.272s` | `1306.6/s` |
| `B=16,T=64` | `0.783s` | `0.024s` | `0.120s` | `0.632s` | `0.540s` | `1307.3/s` |
| `B=32,T=32` | `0.680s` | `0.019s` | `0.078s` | `0.577s` | `0.490s` | `1506.7/s` |

Read: observation/raycast still dominates the current source-backed profile;
ray casting is about `69-72%` of loop time. This is not source-fidelity,
MCTS/GPU, learner, or production replay evidence.

Native vector trainer profile, strict `1v1/no_bonus/P=2` plumbing only:

| Shape | Loop | Public env.step | Policy map+forward | Replay record | Chunk build | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.259s` | `0.252s` | `0.0041s` | `0.0010s` | `0.0054s` | `1980/s` |
| `B=16,T=64` | `0.582s` | `0.572s` | n/a | n/a | n/a | `1760/s` |
| `B=32,T=32` | `0.640s` | `0.635s` | n/a | n/a | n/a | `1600/s` |
| `B=128,T=64` | `6.846s` | `6.830s` | n/a | n/a | n/a | `1197/s` |

Separate observation phase probe, one pass over current state:
`ray_cast_sec` was `0.0035s`, `0.0069s`, `0.0124s`, and `0.0561s` for
`B=8/16/32/128`. Read: the native vector path removes source-adapter cost, but
the public step remains observation/ray-bound, and larger CPU batch is not
automatically better. Source-backed JS/`CurvyTronSourceEnv` remains the oracle;
this is strict native vector plumbing evidence only.

Post batch-array writer pass, same strict native plumbing surface:

| Shape | Loop | Public env.step | Throughput | Batch obs probe | Ray cast |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.257s` | `0.248s` | `1993/s` | `0.00323s` | `0.00303s` |
| `B=16,T=64` | `0.431s` | `0.424s` | `2375/s` | `0.00640s` | `0.00608s` |
| `B=32,T=64` | `0.914s` | `0.904s` | `2241/s` | `0.01293s` | `0.01237s` |
| `B=128,T=16` | `0.822s` | `0.816s` | `2493/s` | `0.04973s` | `0.04779s` |

Read: validating once and avoiding row dataclass construction helps, especially
past tiny batches, but the corrected batch observation probe is still mostly
ray casting. The `B=128` row is a short `T=16` run because the longer
straight-action run hit a terminal row; do not treat it as a clean `T=64`
steady-state comparison.

Post body-cursor slice, same strict native surface:

| Shape | Loop | Public env.step | Throughput | Batch obs probe | Ray cast |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.169s` | `0.162s` | `3038/s` | `0.00237s` | `0.00217s` |
| `B=16,T=64` | `0.394s` | `0.383s` | `2599/s` | `0.00468s` | `0.00432s` |
| `B=32,T=64` | `0.659s` | `0.649s` | `3108/s` | `0.00936s` | `0.00862s` |
| `B=128,T=16` | `0.602s` | `0.596s` | `3399/s` | `0.03297s` | `0.03105s` |

Read: slicing by `body_write_cursor[row]` was a real native-vector win and
confirms unused body-buffer tail was being paid for in the ray path. Source
adapter states fall back to full arrays because they currently do not emit a
cursor.

Toy bridge refresh: serial `22063.6` env steps/s; thread `18950.4` env steps/s
(`0.813x`, bad); process `51859.3` env steps/s (`3.579x`, `0.895`
efficiency). Caveat: toy object env, synthetic NumPy policy, local sharding;
not source fidelity, MCTS, Modal, or production throughput.

Retimed Modal L4 Mctx check after the D2H patch, app
`ap-u3YpTqQcqArxzFk5PI6ZbH`, `curvytron_trainer_flat
B=64,P=2,obs=106,sim=8,hidden=64,depth=8`: compile+first `5.0005s`, steady
Mctx median `0.002904s`, host obs setup `0.000622s`, steady H2D median
`0.000545s`, selected-action D2H samples `[0.01914, 0.0000471, 0.0000023]`
with median `0.0000471s`, and action-weights D2H median `0.0000055s`. Read:
the first action conversion has first-use/sync overhead; steady action-only D2H
is tiny. This still does not measure CPU ray generation or source fidelity.

## Known Runs

### Vector Batch Rows

Source: [self-play speed lane](../environment/selfplay_speed_lane_2026-05-09.md).

- `benchmark_vector_batch_rows.py`, `B=32/128`, debug-event and no-event, widened
  one-step fixture slice.
- At `B=128`, no-event step rows/sec were about `1.20M` for `P1_K4`, `488k`
  for `P2_K4`, and `445k` for `P3_K4`.
- Debug event emission was a large tax for multiplayer groups: about `49%` of
  the `P2_K4/B=128` step bucket and `32%` of `P3_K4/B=128`.
- Unknown: production rollout speed, because this is fixture-row timing without
  real reset/autoreset, trainer observations, replay write, or policy/search.

### Actor-Loop Bridge

- `benchmark_vector_actor_loop_bridge.py`, `B=32/128`, two-step rollout blocks,
  debug-event and no-event.
- At `B=128`, no-event env rows/sec were about `436k` for `P1_K4`, `260k` for
  `P2_K4`, and `218k` for `P3_K4`.
- No-event mode exposed synthetic policy/search as a large bucket: about `52%`
  for `P1_K4/B=128`, `50%` for `P2_K4/B=128`, and `47%` for `P3_K4/B=128`.
- In-memory replay staging was small in this scout, about `1%` to `2%`.
- Unknown: real learner/search/replay cost; the policy/search is synthetic and
  observations/rewards are debug payloads.

### No-Event Larger Shape

- User-provided `benchmark_vector_actor_loop_bridge.py` run with `B=2/32/128`,
  `rollout_steps=4`, `hidden_dim=16`, `simulations=2`, no-event.
- At `B=128`: `P1_K4` about `378k` env rows/sec, `P2_K4` about `227k`,
  `P3_K4` about `267k`.
- Actor-step p50 was about `0.333 ms`, `0.561 ms`, and `0.461 ms` respectively.
- Unknown: p95/p99 production latency with real model/search.

### Heavy Synthetic Search Stand-In

- `benchmark_vector_actor_loop_bridge.py` with `simulations=64`, `B=32`,
  no-event.
- P2/P3 shifted from env-step-led to synthetic policy/search-led.
- P2 synthetic policy/search rose to about `71%` of loop with actor p95
  `1.491 ms`; P3 rose to about `77%` with actor p95 `2.026 ms`.
- Unknown: whether real PPO/MCTS/Mctx lands closer to the light or heavy
  stand-in regime.

### Optimizer Tiny Actor-Loop Scouts

Run locally by optimizer on 2026-05-09. Scope: two P2 fixtures only
(`source_borderless_wrap_step`, `source_normal_wall_death_step`), debug
observations/rewards, synthetic NumPy policy/search, in-memory replay staging,
debug internal autoreset. These are setup measurements, not production speed.

Light synthetic search:

```text
uv run python scripts/benchmark_vector_actor_loop_bridge.py \
  scenarios/environment/source_borderless_wrap_step.json \
  scenarios/environment/source_normal_wall_death_step.json \
  --batch-sizes 32 128 \
  --event-modes debug-event no-event \
  --repeat 80 --warmup 10 --rollout-steps 3 \
  --hidden-dim 16 --simulations 1 --chunk-steps 16 --format json
```

Notable read:

- `P2_K4/B=32`: debug-event env step was about `57%` of loop; no-event total
  loop was about `1.34x` faster than debug-event.
- `P2_K4/B=128`: no-event env step was about `49%`, autoreset about `25%`,
  synthetic policy/search about `9%`, replay staging about `1.3%`.
- Debug event overhead was large: at `B=128`, no-event made env-step bucket
  about `2.04x` faster and total loop about `1.55x` faster.

Heavier synthetic search:

```text
same two fixtures, no-event, --batch-sizes 32 128, --repeat 60,
--rollout-steps 3, --hidden-dim 64, --simulations 16
```

Notable read:

- `P2_K4/B=32`: synthetic policy/search became the top bucket at about `48%`;
  env step dropped to about `26%`; actor-step p95 was about `0.824 ms`.
- `P2_K4/B=128`: synthetic policy/search was about `57%`; env step about
  `23%`; actor-step p95 was about `1.37 ms`.

Interpretation: even a fake heavier decision box quickly dominates this loop.
That supports measuring real PPO/MCTS/Mctx timing before optimizing the
environment in isolation.

Contract/sample guard run:

```text
uv run pytest tests/test_benchmark_vector_actor_loop_bridge.py \
  tests/test_debug_actor_loop_replay.py tests/test_trainer_replay_v0_builder.py -q
```

Result: `32 passed in 0.16s`.

Replay-v0-shaped debug sample:

```text
uv run python scripts/benchmark_vector_actor_loop_bridge.py \
  scenarios/environment/source_borderless_wrap_step.json \
  scenarios/environment/source_normal_wall_death_step.json \
  --sample-only --batch-sizes 4 --event-modes debug-event \
  --rollout-steps 2 --hidden-dim 32 --simulations 4 \
  --body-capacity 4 --player-count 2 \
  --sample-replay-v0-chunk /private/tmp/actor_bridge_replay_v0_debug.npz \
  --format json
```

Result after adding isolated sample-write telemetry: wrote a valid
replay-v0-shaped `.npz` with `file_bytes=16840`, `write_elapsed_sec=0.002764`,
`chunk_steps=2`, `batch_size=4`, and `obs_dim=9`. The report marks
`production_training_decision=blocked` because the payload uses debug
observation/reward schemas:

- observation schema: `curvyzero_debug_global_player_obs/v0`;
- reward schema: `curvyzero_debug_score_round_delta_death_penalty/v0`.

Interpretation: replay-v0 shape validation and local sample write telemetry are
solid, but this bridge is still not trainer payload and this one write is not a
production replay stream. The next useful measurement is the same loop with
`curvyzero_egocentric_rays/v0` (`obs_dim=106`), sparse trainer rewards, and
repeatable replay-write or learner-handoff timing.

### Repo-Native PPO Dry-Run Scaffold

Run locally by optimizer on 2026-05-09:

```text
uv run python scripts/repo_native_ppo_actor_loop_dry_run.py \
  --batch-size 4 --rollout-steps 8 \
  --artifact-root /private/tmp/curvy-repo-native-ppo-actor-loop-dry-run-smoke \
  --format json
```

Result: dry-run contract probe passed and wrote:

- report: `/private/tmp/curvy-repo-native-ppo-actor-loop-dry-run-smoke/report.json`;
- rollout buffer: `/private/tmp/curvy-repo-native-ppo-actor-loop-dry-run-smoke/rollout_buffer.npz`;
- rollout file size on disk: `63942` bytes;
- selected rollout arrays: `54880` bytes.

Important fields:

- observation schema: `curvyzero_egocentric_rays/v0`;
- observation shape: `[8,4,2,106]`;
- action mask shape: `[8,4,2,3]`;
- reward schema: `curvyzero_sparse_round_outcome/v0`;
- status: `ok`;
- caveat: not a PPO learner, not source fidelity, not vector-runtime
  throughput.

Timing read:

- wall time: `0.1099s`;
- observation packing: `0.1011s`, about `58%` of timed bucket total;
- artifact write: `0.0655s`, about `37%` of timed bucket total;
- env step: `0.00466s`, about `2.7%`;
- policy forward: `0.00060s`, about `0.3%`;
- action p95: `1.91 ms`;
- env transitions/sec: about `291`;
- ego decisions/sec: about `583`.

Interpretation: this is a useful sidecar shape check for `[B,P,106]` rows and
PPO-style rollout artifacts. It is not production speed and not the primary
visual CurvyTron path because it uses toy scalar env rows, a masked-uniform
policy, no learner update, and no source-faithful vector runtime. The high
observation-packing share says scalar diagnostics should split ray casting from
array row mapping before claiming simulator cost.

Follow-up instrumentation: the dry-run report now includes
`optimizer_profile_schema=curvyzero_optimizer_profile_report/v0`, run
provenance, loop shape, denominators, replay/rollout metadata,
policy/search metadata, learner/eval non-run markers, and synchronous
policy-staleness metadata. Focused test coverage:

```text
uv run pytest tests/test_repo_native_ppo_actor_loop_dry_run.py \
  tests/test_benchmark_vector_actor_loop_bridge.py \
  tests/test_debug_actor_loop_replay.py tests/test_trainer_replay_v0_builder.py -q
```

Result: `33 passed in 0.29s`.

### Repo-Native PPO Learner Smoke

Command:

```text
uv run python scripts/repo_native_ppo_learner_smoke.py \
  --batch-size 4 --rollout-steps 8 \
  --artifact-root /private/tmp/curvy-repo-native-ppo-on-policy-smoke-optimizer \
  --format json
```

Result: skipped locally with
`reason=torch is not importable; pyproject does not declare it as a dependency`.

Interpretation: the optional learner boundary exists, but this local uv
environment is NumPy-only. This is a setup/dependency fact, not a PPO quality
result and not a reason to retire the repo-native measurement hypothesis.

### Policy/Search Batch Stand-In

Command shape:

```text
uv run python scripts/benchmark_policy_search_batch_standin.py \
  --env-batch 64 --players 2 --obs-dim 106 --hidden-dim 128 \
  --action-count 3 --decision-batches 20 --warmup 2 \
  --copy-mode copy --format json
```

With `--simulations 4`: elapsed `0.006238s`, about `410k` policy rows/sec,
`recurrent_search_loop=0.005022s`, `root_model=0.000735s`, lower-bound hidden
tree bytes `327680`.

With `--simulations 32`: elapsed `0.039126s`, about `65k` policy rows/sec,
`recurrent_search_loop=0.037872s`, `root_model=0.000631s`, lower-bound hidden
tree bytes `2162688`.

Interpretation: this is CPU NumPy shape evidence only. It proves that a heavier
decision box can dominate the actor loop, so real PPO/MuZero/Mctx timing must
be included before env-only optimization. It does not prove real MCTS cost,
GPU/JAX behavior, target quality, or CurvyTron learning value.

### LightZero Profile Attempt Aborted

Optimizer launched this command on 2026-05-09:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train --compute gpu-l4-t4 --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-2048-profile-v0 \
  --progress-interval-sec 60 --max-env-step-override 2048 \
  --profile-phases --gpu-sample-interval-sec 10
```

Result: stopped manually after it was still running at learner iteration
`1300`. The local Modal process was killed and app
`ap-CLIw2m3bXwNbKVHItDQP33` was stopped. No final profile summary should be
treated as evidence from this attempt.

Interpretation: `max_env_step=2048` was still the wrong profiling control for
this LightZero loop. For pure optimizer timing, a few learner train calls are
enough if hooks are installed, but `max_train_iter_override` later proved it was
not a reliable tiny-run cap either. Future stock-loop profiles should use the
explicit learner-train hook stop and be labelled as profile/control runs, not
exact reproductions or coach learning evidence.

Second LightZero profile attempt also aborted:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train --compute gpu-l4-t4 --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-profiler-iter5-installed-0.2.0-s0-v0 \
  --progress-interval-sec 15 --max-train-iter-override 5 \
  --profile-phases --gpu-sample-interval-sec 2
```

Result: stopped manually; app `ap-xkTDXj5wNV8DiwVvLFpYJ9`. It did not produce a
final `summary.json`. Logs showed first evaluator/checkpoint activity about
`112s` after app start, first collect/iteration-0 around `4m12s`, and the
learner loop reached `Training Iteration 600` before stop. This was not a
signature-gate failure; `max_train_iter` simply did not act as the desired
inner-loop cap here.

Correction: the profiler now needs to stop from inside the patched
`BaseLearner.train` hook after N train calls. That should produce a final
summary while still timing startup/eval/collect/learner, instead of trusting
stock `train_muzero` loop arguments.

Cleaner next step: a direct one-collect/sample/train harness with no evaluator.
The hook-stopped stock run still includes startup/evaluator tax, so it is
accounting evidence, not the clean steady-state unit.

Local safety test added after the abort:
`tests/test_lightzero_phase_profiler.py` verifies the profiler stop hook raises
after `2` fake `BaseLearner.train` calls and restores the original method.
Focused validation: `29 passed`; `ruff` passed.

### Source Env Scout Refresh

Command:

```text
uv run python scripts/benchmark_source_env.py --repeats 5 --js --js-repeats 2 --json
```

Narrow 111-step 1v1/no-bonus wall-round-done lifecycle:

- Python source env: about `0.000927s/rollout`, about `119.8k` steps/sec.
- Persistent JS worker: about `0.00979s/rollout`, about `11.3k` steps/sec.

Interpretation: useful source-scout evidence for one lifecycle, not production
vector-loop throughput and not broad multiplayer/bonus coverage.

### Source-Stepped Trainer Profile Smoke

Command:

```text
uv run python scripts/benchmark_source_trainer_actor_loop_profile.py \
  --batch-size 2 --rollout-steps 2 \
  --artifact-root /private/tmp/curvy-source-trainer-b2-profile \
  --format plain
```

Result: wrote `profile_report.json` and a schema-read-validated replay-v0 chunk
under `/private/tmp/curvy-source-trainer-b2-profile`; reported about `590`
source env transitions/sec and about `1,179` ego decisions/sec on the tiny
local smoke.

Interpretation: this is the first source-stepped scalar-ray sidecar actor-loop
profile. It uses source-derived positions/headings/alive, but occupancy is
empty, so trail/body ray channels are not source-faithful. Treat it as plumbing
evidence for adapter, timing, policy row mapping, replay write/read schema, and
caveats. It is not visual-path or replay semantic validation.

Follow-up tiny sweeps:

- `B=8,T=2`: about `1,226` env transitions/sec and `2,451` ego decisions/sec;
  wall `0.0131s`, observation packing `0.0122s`, policy `0.00010s`, source step
  `0.00021s`.
- `B=2,T=32`: about `1,142` env transitions/sec and `2,285` ego decisions/sec;
  wall `0.0560s`, observation packing `0.0498s`, policy `0.00163s`, source step
  `0.00104s`.
- `B=2,T=32,hidden=256`: about `1,154` env transitions/sec and `2,309` ego
  decisions/sec; wall `0.0554s`, observation packing `0.0495s`, policy
  `0.00156s`, source step `0.00102s`.

Read: with empty occupancy and tiny NumPy policy, the Python trainer observation
helper dominates this profile. This still does not prove final production
observation cost because source trail/body occupancy is missing.

Center-cell source body occupancy mode:

- `B=2,T=32,occupancy=source_world_bodies_center_cell_v0`: about `1,252` env
  transitions/sec and `2,503` ego decisions/sec; wall `0.0615s`, loop
  `0.0511s`, observation packing `0.0461s`, source adapter `0.0020s`, env step
  `0.0011s`.
- `B=8,T=16,occupancy=source_world_bodies_center_cell_v0`: about `1,310` env
  transitions/sec and `2,621` ego decisions/sec; wall `0.1055s`, loop
  `0.0977s`, observation packing `0.0909s`, source adapter `0.0035s`, env step
  `0.0017s`.

Read: center-cell source bodies remove the all-empty occupancy caveat, but the
trainer rays are still grid-cell approximations rather than exact source circle
geometry, visible trail history, or own-body latency semantics. Observation
packing still dominates.

Cleanup validation after report-key purge:

- Focused optimizer/runtime tests:
  `uv run pytest tests/test_source_trainer_adapter.py tests/test_benchmark_source_trainer_actor_loop_profile.py tests/test_repo_native_ppo_actor_loop_dry_run.py tests/test_benchmark_vector_actor_loop_bridge.py tests/test_debug_actor_loop_replay.py tests/test_trainer_replay_v0_builder.py tests/test_source_env.py -q`
  -> `72 passed in 1.30s`.
- Dry-run report now uses `policy_search`, `latency_sec.policy_action`,
  `env_transitions_per_sec`, `ego_decisions_per_sec`, full `[T,B,P,...]` shape
  metadata, and `wall_elapsed_sec` that includes artifact write time.
- Source trainer profile now uses `occupancy_policy`,
  `occupancy_source_fields`, `approximate_fields`, `policy_search`,
  `latency_sec.policy_action`, and `artifacts.report_json`.
- Optional Torch PPO learner smoke still skips in local `uv`:
  `torch is not importable; pyproject does not declare it as a dependency`.

Cleaned-report source profile matrix:

- `B=8,T=32,occupancy=empty,h=64`: about `565` env transitions/sec and
  `1,129` ego decisions/sec; loop `0.453s`, observation packing `0.411s`,
  env step `0.018s`, policy `0.0046s`.
- `B=8,T=32,occupancy=center-cell,h=64`: about `586` env transitions/sec and
  `1,173` ego decisions/sec; loop `0.437s`, observation packing `0.369s`,
  env step `0.0167s`, policy `0.019s`.
- `B=8,T=32,occupancy=center-cell,h=256`: about `597` env transitions/sec and
  `1,194` ego decisions/sec; loop `0.429s`, observation packing `0.369s`,
  env step `0.0135s`, policy `0.0052s`.
- `B=2,T=256,occupancy=center-cell,h=64`: about `759` env transitions/sec and
  `1,518` ego decisions/sec; loop `0.675s`, observation packing `0.599s`,
  env step `0.0122s`, policy `0.0115s`, completed games/min about `178`.

Read: observation packing still dominates the scalar source-stepped profile.
These runs are cleaner measurement artifacts, not a final production bottleneck
verdict, because exact source trail/body observations and real policy/search
are still missing.

Exact local hookup scout requested by optimizer on 2026-05-09:

```text
uv run python scripts/benchmark_source_trainer_actor_loop_profile.py \
  --batch-size 2 --rollout-steps 8 \
  --occupancy-policy source_world_bodies_center_cell_v0 \
  --artifact-root /private/tmp/curvy-source-trainer-hookup-scout \
  --format json
```

Result: no-train profile passed. Wall elapsed `0.0453s`, loop elapsed
`0.0154s`, observation packing `0.0121s`, source adapter `0.00065s`, source env
step `0.00034s`, policy forward `0.00184s`, replay write/handoff `0.00316s`,
and replay read validation `0.00413s`. Throughput was about `1,037` env
transitions/sec and `2,075` ego decisions/sec on `16` env transitions and `32`
player ticks.

Shape/integrity: actions `[8,2,2]`, observations `[8,2,2,106]`,
`done == terminated OR truncated` failures `0`, masked action violations `0`,
NaN/Inf count `0`, replay schema read validation `true`, replay semantic
validation still not performed. Artifacts:
`/private/tmp/curvy-source-trainer-hookup-scout/profile_report.json` and
`/private/tmp/curvy-source-trainer-hookup-scout/replay_v0_chunk.npz`.

Interpretation: the scalar-ray sidecar is hookable today for small repo-native
`[B,P]` plumbing/profiling. It is still not the visual CurvyTron training path
or an honest source-faithful training loop: occupancy is source-world-body
center-cell raster, not exact source circle geometry, visible trail history, or
own-body latency semantics.

Larger local source/trainer matrix run on 2026-05-09:

```text
uv run python scripts/benchmark_source_trainer_actor_loop_profile.py \
  --batch-size 16 --rollout-steps 64 \
  --occupancy-policy source_world_bodies_center_cell_v0 \
  --artifact-root /private/tmp/curvy-source-trainer-b16-t64-scout \
  --format json
```

Result: `1024` env transitions and `2048` ego decisions, replay-v0 write/read
valid, no masked-action or terminal invariant failures. Loop elapsed `0.778s`,
about `1,315` env transitions/sec and `2,631` ego decisions/sec. Timings:
observation packing `0.733s`, source adapter `0.0257s`, env step `0.0129s`,
policy forward `0.0028s`, row compaction `0.0018s`, action scatter `0.0010s`,
replay write `0.0018s`, replay read validation `0.0022s`.

Read: the current source-stepped scalar-ray sidecar profile is overwhelmingly
observation-packing-led with the tiny NumPy policy and no search/learner. This
does not prove final production bottleneck for visual CurvyTron, but it gives a
concrete sidecar optimization target: split and improve
`observe_1v1_egocentric_rays_v0`/ray packing before touching source env step
speed in this no-train diagnostic lane.

Two extra comparison rows were run concurrently, so treat exact ratios as noisy:
empty occupancy (`/private/tmp/curvy-source-trainer-b16-t64-empty-scout`) and
center-cell occupancy with hidden dim `512`
(`/private/tmp/curvy-source-trainer-b16-t64-h512-scout`). Both still had
observation packing as the dominant bucket (`1.15s` and `1.13s` respectively),
with env step and policy remaining far smaller.

Empty-occupancy ray fast path patch:

- changed `_cast_rays` to skip occupancy owner grouping and trail nearest-cell
  searches when occupancy is all zero;
- added a contract test for no own/opponent trail hits on empty occupancy;
- focused tests: `28 passed`;
- worker timing after patch: empty `observe_1v1_egocentric_rays_v0` around
  `237us`; local `B=16,T=64` center-cell/warmup-0 profile
  `observation_packing_sec=0.6039`, `loop=0.6493`.

Read: useful first speedup for the current empty/no-body observation case. Do
not generalize it to source-faithful trail/body occupancy; the real trail/body
geometry path still needs its own measurement.

Ray-direction precompute patch:

- precomputed fixed ray-angle sin/cos once and replaced per-ray
  `math.radians`/`math.cos`/`math.sin` calls with one per-heading direction
  array;
- focused validation: `29 passed`; `ruff` and `py_compile` pass;
- fresh local command:

```text
uv run python scripts/benchmark_source_trainer_actor_loop_profile.py \
  --batch-size 16 --rollout-steps 64 \
  --occupancy-policy source_world_bodies_center_cell_v0 \
  --artifact-root /private/tmp/curvy-source-trainer-b16-t64-raydir-scout \
  --format json
```

Result: loop `0.641s`, observation packing `0.594s`, source adapter `0.0255s`,
env step `0.0130s`, policy forward `0.0041s`, throughput about `1,597` env
transitions/sec and `3,195` ego decisions/sec.

Warmup sanity check:

```text
uv run python scripts/benchmark_source_trainer_actor_loop_profile.py \
  --batch-size 8 --rollout-steps 32 --warmup-ms 2000 \
  --occupancy-policy source_world_bodies_center_cell_v0 \
  --artifact-root /private/tmp/curvy-source-trainer-b8-t32-warm-scout \
  --format json
```

Result: loop `0.159s`, observation packing `0.147s`, env step `0.0033s`.
A direct source-env check still showed `world_bodies_snapshot()` length `0`
after warmup and several steps. Interpretation: observation packing is a real
bottleneck for the current source/trainer plumbing profile, but the current
profile does not exercise source-faithful trail/body occupancy. Do not treat it
as final CurvyTron observation cost.

Source body/trail counter and controlled trail setup:

- `benchmark_source_trainer_actor_loop_profile.py` now reports
  `source_body_trail`: source world body count, adapted occupancy nonzero cells,
  per-player occupied-cell counts, and nonempty sample counts.
- `source_drive_mode` records whether the profile uses direct wrapper stepping
  or source frame timers.
- `source_setup_mode=controlled_trail` force-places players in safe lanes and
  fires the delayed source trail-start timer before profiling. This creates a
  short body/trail observation profile without running a long training job.

Default `B=2,T=256` center-cell run:

```text
uv run python scripts/benchmark_source_trainer_actor_loop_profile.py \
  --batch-size 2 --rollout-steps 256 \
  --occupancy-policy source_world_bodies_center_cell_v0 \
  --artifact-root /private/tmp/curvy-source-trainer-b2-t256-body-count \
  --format json
```

Result: only `2` of `1024` samples had nonempty body/occupancy data. Read:
those are death artifacts, not steady trail bodies. The default setup is still
mostly empty-body plumbing.

Controlled trail/body run:

```text
uv run python scripts/benchmark_source_trainer_actor_loop_profile.py \
  --batch-size 2 --rollout-steps 64 \
  --occupancy-policy source_world_bodies_center_cell_v0 \
  --source-setup-mode controlled_trail \
  --source-drive-mode direct_step \
  --artifact-root /private/tmp/curvy-source-trainer-b2-t64-controlled-trail \
  --format json
```

Result: all `256` samples had nonempty source body/occupancy data. Mean source
world bodies `22.67`, mean adapted occupied cells `20.38`, max occupied cells
`38`. Loop `0.351s`; observation packing `0.331s`; source adapter `0.0125s`;
env step `0.0034s`; policy `0.0017s`; throughput about `364` env transitions/s
and `729` ego decisions/s.

Interpretation: with nonempty approximate body/trail occupancy, observation
packing remains the dominant measured local bucket. The next useful optimizer
step is to split the observation bucket and optimize ray/body work. This is
still not a production training-loop verdict because occupancy is center-cell,
policy/search is fake, and learner updates are absent.

Observation phase profile before vectorized center hits:

```text
same controlled trail command with --profile-observation-phases,
artifact_root=/private/tmp/curvy-source-trainer-b2-t64-controlled-trail-phases
```

Result: observation packing `0.330s`; `ray_cast_sec=0.318s`.
Everything else was small by comparison: old scalar batch assembly about
`0.0025s`, scalar packing `0.0034s`, reward `0.0016s`. Read: the next
optimizer target was ray/body hit work, not copies, masks, rewards, or row
mapping.

Vectorized center-hit patch:

- precomputes own/opponent occupied-cell centers once per ego observation;
- replaces `_nearest_cell_hit`'s per-ray/per-cell Python loop with vectorized
  circle-intersection math over center arrays;
- focused validation still passed (`31 passed`, `ruff`, `py_compile`).

Clean post-patch controlled trail run:

```text
uv run python scripts/benchmark_source_trainer_actor_loop_profile.py \
  --batch-size 2 --rollout-steps 64 \
  --occupancy-policy source_world_bodies_center_cell_v0 \
  --source-setup-mode controlled_trail \
  --source-drive-mode direct_step \
  --artifact-root /private/tmp/curvy-source-trainer-b2-t64-controlled-trail-vector-hit-clean \
  --format json
```

Result: loop `0.243s`; observation packing `0.223s`; env step `0.0033s`;
policy `0.0016s`; throughput about `526` env transitions/sec and `1,053` ego
decisions/sec. The same run with phase timers showed `ray_cast_sec=0.220s`.

Second ray batching patch:

- batches all ray directions against each occupied-center array in one call;
- also batches opponent-head ray hits;
- focused validation still passed (`31 passed`, `ruff`, `py_compile`).

Clean post-batch run:

```text
uv run python scripts/benchmark_source_trainer_actor_loop_profile.py \
  --batch-size 2 --rollout-steps 64 \
  --occupancy-policy source_world_bodies_center_cell_v0 \
  --source-setup-mode controlled_trail \
  --source-drive-mode direct_step \
  --artifact-root /private/tmp/curvy-source-trainer-b2-t64-controlled-trail-raybatch-clean \
  --format json
```

Result: loop `0.105s`; observation packing `0.086s`; env step `0.0034s`;
policy `0.0015s`; throughput about `1,220` env transitions/sec and `2,441`
ego decisions/sec. The same run with phase timers showed `ray_cast_sec=0.078s`.

Read: on the current controlled nonempty center-cell body/trail profile, the
two ray patches improved loop wall time by roughly `3.35x` and observation
packing by roughly `3.86x` versus the first controlled trail measurement. It is
still not final CurvyTron production speed because the geometry is approximate
and real search/learner timing is absent.

Larger controlled-trail scale check:

```text
uv run python scripts/benchmark_source_trainer_actor_loop_profile.py \
  --batch-size 8 --rollout-steps 64 \
  --occupancy-policy source_world_bodies_center_cell_v0 \
  --source-setup-mode controlled_trail \
  --source-drive-mode direct_step \
  --artifact-root /private/tmp/curvy-source-trainer-b8-t64-controlled-trail-raybatch-clean \
  --format json
```

Result: all `1024` samples had nonempty body/occupancy data. Loop `0.432s`;
observation packing `0.363s`; source adapter `0.048s`; env step `0.012s`;
policy `0.004s`; throughput about `1,186` env transitions/sec and `2,372` ego
decisions/sec.

Phase run:

```text
same command with --profile-observation-phases,
artifact_root=/private/tmp/curvy-source-trainer-b8-t64-controlled-trail-raybatch-phases
```

Result: loop `0.411s`; observation packing `0.343s`; `ray_cast_sec=0.296s`;
`source_adapter_sec=0.049s`; old scalar batch assembly about `0.009s`;
`scalar_pack_sec=0.014s`. Read: at larger local batch, scalar ray casting still
dominates the no-train profile. The next useful local optimization was an
actual batched observation path or replacing the fake policy/search with
calibrated timing; not more env-step work.

Batched two-ego observation writer:

- `observe_1v1_egocentric_rays_v0` now validates once, shares the simple
  occupancy/head-cell ray context, computes both ego rows into preallocated
  arrays, and preserves scalar parity/copy semantics.
- Focused validation after the patch:
  `uv run pytest tests/test_source_trainer_adapter.py tests/test_trainer_replay_v0_builder.py tests/test_trainer_contract.py tests/test_benchmark_source_trainer_actor_loop_profile.py tests/test_lightzero_phase_profiler.py -q`
  -> `31 passed in 0.25s`; `ruff` and `py_compile` pass.
- `B=2,T=64` with phase timers, artifact
  `/private/tmp/curvy-source-trainer-b2-t64-batched-writer-phases-final`:
  loop `0.096s`, observation packing `0.0766s`, `ray_cast_sec=0.0688s`,
  throughput about `1,328` env transitions/sec.
- `B=8,T=64`, artifact
  `/private/tmp/curvy-source-trainer-b8-t64-batched-writer`: loop `0.357s`,
  observation packing `0.296s`, env step `0.011s`, source adapter `0.045s`,
  throughput about `1,433` env transitions/sec. Compared with the prior
  raybatch clean run, that is roughly `17%` lower loop time and `18%` lower
  observation packing time.
- Short larger matrix after patch:
  `B=16,T=16` loop `0.179s`, obs `0.154s`;
  `B=32,T=16` loop `0.328s`, obs `0.288s`;
  `B=64,T=8` loop `0.314s`, obs `0.280s`.

Read: the batched writer is a real modest scalar-ray sidecar win and should
stay. It does not define the main CurvyTron bottleneck: this no-train
source/trainer loop is still observation/ray dominated under center-cell trail
geometry, with no real policy/search/learner or visual-frame payload in the
measurement.

Matched CPU policy/search overlay:

- Method: keep source/trainer loop separate from
  `benchmark_policy_search_batch_standin.py`, matching `B`, `P=2`, `obs_dim=106`,
  and total policy rows. This keeps the report honest: real source stepping and
  approximate center-cell body occupancy on one side, fake CPU search on the
  other.
- `B=8,T=64` source phase artifact
  `/private/tmp/curvy-source-trainer-b8-t64-policy-search-calibration`: loop
  `0.399s`, observation packing `0.327s`, `ray_cast_sec=0.295s`,
  source adapter `0.054s`, env step `0.012s`.
- Matched stand-in, `B=8,T=64`, `simulations=4`: elapsed `0.0051s`;
  recurrent loop `0.0038s`.
- Matched stand-in, `B=8,T=64`, `simulations=32`: elapsed `0.0362s`;
  recurrent loop `0.0343s`.
- `B=32,T=16` source phase artifact
  `/private/tmp/curvy-source-trainer-b32-t16-policy-search-calibration`: loop
  `0.311s`, observation packing `0.274s`, `ray_cast_sec=0.247s`,
  source adapter `0.026s`, env step `0.0085s`.
- Matched stand-in, `B=32,T=16`, `simulations=32`: elapsed `0.0146s`;
  recurrent loop `0.0140s`.

Read: under this CPU proxy, fake search is not the current top bucket; source
observation/ray work is still much larger. Bigger policy batches help a lot:
the same `1024` policy rows with `32` fake simulations took `0.036s` at
`B=8,T=64` versus `0.0146s` at `B=32,T=16`. Do not treat this as a real
Mctx/LightZero/GPU result; use it only as an Amdahl guard against premature
env-step or replay polish.

Circle-ray source body profile update:

- Profiling mode: `source_world_bodies_circle_rays_v0`.
- Meaning: source world body circles plus avatar body metadata into sidecar
  all-player `float32[B,2,106]` diagnostic rows.
- Latest short refresh:
  - `B=8,T=64`: loop `0.392s`, env step `0.011s`, source adapter `0.058s`,
    obs `0.318s`, ray cast `0.272s`, throughput `1306.6/s`.
  - `B=16,T=64`: loop `0.783s`, env step `0.024s`, source adapter `0.120s`,
    obs `0.632s`, ray cast `0.540s`, throughput `1307.3/s`.
  - `B=32,T=32`: loop `0.680s`, env step `0.019s`, source adapter `0.078s`,
    obs `0.577s`, ray cast `0.490s`, throughput `1506.7/s`.

Read: ray casting remains the main measured sub-bucket at about `69-72%` of
loop time. Total observation packing is larger still; env step remains tiny.

### Native Vector Trainer Actor Loop

- Script:
  `scripts/benchmark_vector_trainer_actor_loop_profile.py`.
- Scope: strict `VectorTrainerEnv1v1NoBonus`, `P=2`, public
  `float32[B,2,106]` observations, `bool[B,2,3]` masks, policy-row mapping,
  tiny policy/search stand-in, and replay-v0 chunk construction.
- Corrected profile read: native vector removes source adapter cost but stays
  observation/ray-bound. Bigger CPU batch is not automatically better:
  throughput falls from `1980/s` at `B=8,T=64` to `1197/s` at `B=128,T=64`.
- Reset guard: `VectorTrainerEnv1v1NoBonus.reset` now scales the warmup timer
  callback cap for larger `B`; regression coverage includes `B=128` reset.
- Validation: `ruff` passed for the new script/tests/env files;
  `pytest tests/test_benchmark_vector_trainer_actor_loop_profile.py tests/test_vector_trainer_env.py -q`
  returned `14 passed`.
- Post-review report-shape patch: added `optimizer_profile_schema/status`,
  `run.debug_event_mode`, timing notes that `env_step_public` includes
  observation/mask/reward/done packing, pre-step-mask
  `masked_action_violations`, separate
  `selected_action_positive_weight_violations`, straight-action mixed fallback
  correction, nonnegative seed validation, and tests. This changes metadata
  correctness and interpretation guardrails, not the speed read above.
- Patch sanity run: `B=8,T=8`, no-event, straight, profile phases. Loop
  `0.031033s`, `env_step_public` `0.029856s`, policy row mapping `0.000365s`,
  policy forward `0.000291s`, recorder `0.000124s`,
  replay build/validate `0.002293s`, throughput `2062.3/s`,
  observation phase probe `0.002998s`, ray cast `0.002542s`. Artifact:
  `/private/tmp/curvy-vector-trainer-profile-sanity-fixed-metadata/profile_report.json`.
- Patch validation: `ruff` passed for the script/test;
  `pytest tests/test_benchmark_vector_trainer_actor_loop_profile.py -q`
  returned `3 passed`; `py_compile` passed.
- Batch-array observation writer patch: public `_observe_arrays` now calls
  `observe_vector_1v1_egocentric_rays_batch_arrays_v0`, which validates the
  state once per batch, returns `[B,2,106]`/mask arrays directly, and keeps the
  scalar ray kernel. The scalar row observer remains the parity oracle. Focused
  validation passed: `ruff` for touched files and
  `pytest tests/test_vector_trainer_observation.py tests/test_vector_trainer_env.py tests/test_benchmark_vector_trainer_actor_loop_profile.py -q`
  -> `31 passed`.
- Body-cursor slicing patch: ray observation now slices native body arrays by
  `body_write_cursor[row]` before `_trail_body_mask` / `_nearest_circle_hits`,
  with full-array fallback for source adapter states. Added a regression for
  active junk beyond the cursor. Focused validation: `ruff` passed and the same
  pytest target returned `32 passed`.
- Boundary: this is not source-fidelity evidence. The source-backed
  JS/`CurvyTronSourceEnv` path remains the environment oracle.

### CurvyTron Debug Visual Adapter Timing

- Script: `scripts/benchmark_debug_visual_lightzero_adapter.py`.
- Scope: direct local `CurvyZeroDebugVisualLightZeroEnv` reset/step timing.
  It emits LightZero-shaped `float32[1,64,64]` debug visual payloads with
  action space `3`.
- Non-scope: no policy/search, no replay, no learner, no trainer frame-stack
  consumption, no ALE, and no source-fidelity visual claim.
- Current longer smoke commands:
  `uv run python scripts/benchmark_debug_visual_lightzero_adapter.py --steps 2048 --seed 7 --action-policy fixed --fixed-action 1`
  and
  `uv run python scripts/benchmark_debug_visual_lightzero_adapter.py --steps 2048 --seed 7 --action-policy random`.
- Fixed-action result: `2048` transitions, `12` resets, `11` done rows,
  `env_step_total=0.3354s`, `loop_elapsed=0.3375s`, about `6107`
  env-step transitions/s and `6068` loop transitions/s.
- Random-action result: `2048` transitions, `11` resets, `10` done rows,
  `env_step_total=0.3549s`, `loop_elapsed=0.3588s`, about `5771`
  env-step transitions/s and `5709` loop transitions/s.
- Read: the debug visual adapter wrapper is not the obvious bottleneck by
  itself. The missing evidence is still the CurvyTron visual collect/search,
  replay/sample, and learner timing.

### Parallel Toy Bridge

- `benchmark_selfplay_parallel_bridge.py` on toy `CurvyTronEnv`, not source
  vector rows.
- Threads worsened latency. Process shards helped coarse actor throughput:
  about `1.95x` steady speedup with 2 workers and `3.62x` with 4 workers.
- Unknown: whether this transfers after real source-faithful vector reset,
  replay, and policy/search are in the loop.

### Source Env Scout

- `benchmark_source_env.py --repeats 20 --js --js-repeats 3`.
- Narrow 111-step 1v1/no-bonus wall-round-done lifecycle: Python source env
  about `0.001133s/rollout` and `97,998` steps/sec; persistent JS worker about
  `5,580` steps/sec.
- Unknown: full multiplayer, bonuses, broad lifecycle, and optimized fast-path
  throughput.

### Modal / Mctx Boundary

- Modal L4 JAX/Mctx dependency and synthetic search smokes pass.
- Recorded synthetic profile: about `4,456` decisions/sec and `17,825`
  simulations/sec for a small `B=8`, `num_simulations=4` shape.
- Actor-bridge sample boundary has run with fixture-seeded debug observations
  feeding live ego roots, with setup/H2D/compile/steady timing separated.
- Unknown: real model, real replay, real CurvyTron rollout, and D2H/action
  scatter cost in a production loop.

## Next Placeholders

- [x] First source-stepped scalar-ray sidecar actor-loop profile with empty
  occupancy caveat.
- [x] Strict native vector sidecar actor-loop profile with replay-v0 chunk and
  public `[B,2,106]` diagnostic rows.
- [ ] Source-faithful trainer observation profile with real trail/body
  occupancy.
- [ ] Replay-v0 production write timing: local `.npz` first, then Modal Volume
  or learner-adjacent path.
- [ ] PPO/CleanRL-style policy forward and update timing on the same ego-row
  shape.
- [ ] Real MCTS/Mctx timing on the same ego-row shape, with H2D and D2H split.
- [ ] Completed games/minute plus p50/p95/p99 action latency on a real
  reset/autoreset loop.
- [ ] Actor-pool/process-shard profile with policy staleness recorded.

## Guardrail

Do not turn any row/sec number above into a headline unless the report also
states fixture/source coverage, observation/reward schema, replay path,
policy/search implementation, debug mode, and latency percentiles.
