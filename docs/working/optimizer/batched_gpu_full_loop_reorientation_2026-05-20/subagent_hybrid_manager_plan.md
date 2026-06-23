# Subagent Hybrid Manager Plan

Date: 2026-05-20

Scope: profile-only prototype for preserving subprocess-style actor parallelism
while batching GPU observation render centrally. Do not use this as a Coach
training path, do not change trainer defaults, and do not touch live runs.

## Read Anchors

- Stock LightZero currently sees many scalar env wrappers returning `[4,64,64]`
  timesteps, while the vector batch is lost before observation/render:
  [world_model.md](/Users/shankha/curvy/docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/world_model.md:7).
- The expected big win is architectural: keep vector batches across
  env/observation/policy/search boundaries, not just faster scalar rendering:
  [world_model.md](/Users/shankha/curvy/docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/world_model.md:27).
- The current one-process batched GPU manager still emits scalar Python/NumPy
  timesteps, and external fast systems either keep the whole loop on accelerator
  or keep many workers alive while batching GPU-heavy pieces:
  [world_model.md](/Users/shankha/curvy/docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/world_model.md:58).
- The orchestration guardrail is explicit: profile-only run ids, no trainer
  defaults, no promotion, no live-run interference:
  [orchestration.md](/Users/shankha/curvy/docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/orchestration.md:23).
- Current pass gates already name the fragile semantics this prototype must not
  dodge: exact backend metadata, missing/extra action rejection, partial
  autoreset, terminal `final_observation`, RND latest-frame extraction, and
  matched full-loop accounting:
  [orchestration.md](/Users/shankha/curvy/docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/orchestration.md:73).
- The experiment grid already lists this as the next architecture row after
  one-process manager saturation:
  [next_experiment_grid.md](/Users/shankha/curvy/docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/next_experiment_grid.md:51).
- The host-overhead ladder is the reason to preserve subprocess-style actors:
  at C64, subprocess CPU-oracle was about `5.4x` faster than base for the same
  no-death env steps:
  [experiment_log.md](/Users/shankha/curvy/docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/experiment_log.md:133).
- The one-process manager zero-observation rows prove the batched manager
  boundary has headroom when pixels are removed:
  [experiment_log.md](/Users/shankha/curvy/docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/experiment_log.md:262).
- The existing mock collector is already the right semantic reference for a
  profile-only bridge: it keeps `profile_only=True`, materializes row/player
  scalar timesteps, and optionally runs RND hooks:
  [source_state_batched_observation_mock_collector.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_mock_collector.py:1).

## Smallest Useful Prototype

Build a profile-only "hybrid manager canary" that does not call stock
`train_muzero` at first. It should replay the collector-facing sequence of
`ready_obs -> action_by_env_id -> timestep_by_env_id`, but split ownership like
the architecture we actually want:

1. Parent/manager process owns the central observation service and the profile
   metric counters.
2. Actor subprocesses own compact CurvyTron row state and step physics/actions.
3. Actors send compact render requests to the parent after each env step.
4. Parent batches those requests by tick into one render-service call.
5. Parent scalarizes only at the LightZero-shaped boundary and returns
   `ready_obs`/timesteps to the collector-shaped harness.

This is deliberately smaller than a full DI-engine env manager. The point is to
measure the missing architecture shape before risking LightZero registry,
subprocess CUDA, or Coach defaults.

## Process Boundaries

Keep CUDA/JAX in exactly one process for the prototype.

- Parent process:
  - owns the render service and any JAX/Torch device context;
  - owns action routing, scalar env id to `(actor_id, row, player)` mapping, and
    profile counters;
  - owns the final scalar `BaseEnvTimestep`-like materialization, reusing the
    contracts in
    [source_state_batched_observation_mock_collector.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_mock_collector.py:662)
    and
    [source_state_batched_observation_mock_collector.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_mock_collector.py:826).
- Actor subprocess:
  - owns N rows of CPU source-state simulation;
  - receives scalar/player actions or compact joint row actions from parent;
  - steps source state only;
  - emits render requests plus reward/done/final-state metadata;
  - never imports or initializes JAX/CUDA.
- Transport:
  - start with `multiprocessing.Pipe` or queues for simplicity, but make bytes
    and pickle time first-class metrics;
  - use one actor batch message per parent step, not one IPC message per scalar
    player;
  - postpone shared memory until the payload canary says pickle is the wall.

The actor message should be compact source state, not `[4,64,64]` observations.
That is the entire experiment. If the first prototype ships image stacks over
IPC, it has already answered the wrong question.

## Payload Risk

A scalar float32 stack is `4 * 64 * 64 * 4 = 65,536` bytes before Python object
overhead. With two players, that is about `128 KiB` per physical row per step;
C512 would be roughly `64 MiB` per step before `info`, action masks, pickle
headers, and duplication. Sending rendered observations from actors to the
parent would almost certainly erase the subprocess win.

The render request payload must therefore be measured and capped:

- row id / actor row id: int32;
- player mask or both-player request: small int/bitmask;
- current tick and reset generation: int32/int64;
- compact geometry/source-state fields needed by `direct_gray64`;
- reward and done arrays: `float32[B, players]`, `bool[B]`;
- terminal compact source state only for rows that just died;
- no `info` blobs, no rendered frames, no per-player Python dicts across IPC.

The parent may still create scalar Python timestep objects after batching. That
is acceptable for the prototype because the current stock boundary already does
that. The metric must separate:

- actor step time;
- actor-to-parent serialize/send time;
- parent request gather time;
- central zero/real render time;
- parent scalarization time;
- parent-to-harness ready/timestep payload bytes;
- pickle proxy bytes/time for both directions.

Use the existing mock collector's payload fields as precedent: it already
records `pickle_sec`, `pickle_bytes_total`, and `pickle_bytes_per_row` for a
profile-only loop:
[source_state_batched_observation_mock_collector.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_mock_collector.py:549).

## Autoreset And Final Observation

The parent must preserve terminal semantics even when render is central:

- actor steps a row and reports `done[row]=true` plus terminal compact state;
- parent renders terminal `final_observation` from that terminal state before
  any reset state is rendered;
- parent attaches per-player `info["final_observation"]` and optional
  `final_reward` to the scalar timesteps for the env ids that acted;
- only after that does the actor reset the dead row, or the parent sends an
  explicit reset command for that row;
- the next `ready_obs` for that scalar env id must be from the reset state, not
  the terminal state.

This mirrors the current scalar bridge shape: actions are checked against the
current ready ids, the loop step is captured, terminal rows are reset, and the
timestep is still sourced from the pre-reset step:
[source_state_batched_observation_mock_collector.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_mock_collector.py:218).
The final-observation materializers already fail closed on terminal rows without
the required final arrays:
[source_state_batched_observation_mock_collector.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_mock_collector.py:752).

The first hybrid profile should run in no-death mode, but the implementation
plan must include this terminal handshake before any normal-death row is trusted.
Partial autoreset has to be row-local: resetting actor row `i` must not change
neighboring rows' observations, live ids, reset generations, or pending actions.

## RND Lane

Keep RND out of the first canary. The docs already say RND cadence is a
separate axis, and that RND should not be folded into renderer claims:
[orchestration.md](/Users/shankha/curvy/docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/orchestration.md:117).

For the second canary, add RND only in meter mode with `intrinsic_reward_weight=0`
and a real batch size greater than one. The experiment log shows batch size one
fails in reward-model training, while batch size two proves the plumbing:
[experiment_log.md](/Users/shankha/curvy/docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/experiment_log.md:57)
and
[experiment_log.md](/Users/shankha/curvy/docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/experiment_log.md:113).

RND must consume the same latest policy frame that the learner would see, after
parent-side scalarization. The existing profile loop treats this as a semantic
contract:
[source_state_batched_observation_mock_collector.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_mock_collector.py:621).
Do not run RND inside actor subprocesses; that would reintroduce model/device
work per actor and hide the boundary being measured.

## Toy Canary To Run First

Run a zero-observation hybrid actor canary before any real render canary.

Toy contract:

- `calls_train_muzero=false`;
- `profile_only=true`;
- `touches_live_runs=false`;
- actors step compact source state and return reward/done metadata;
- parent replaces observations with deterministic zero `[4,64,64]` stacks;
- parent still constructs scalar `ready_obs`, `action_mask`, `to_play`, reward,
  done, and `info`;
- no RND, no death, sim/search disabled or mocked;
- fixed action RNG and matched row counts.

Suggested first matrix:

| row | actors | rows/actor | player count | steps | observation | purpose |
|---|---:|---:|---:|---:|---|---|
| `hybrid-zeroobs-a1-r64-s256` | 1 | 64 | 2 | 256 | zeros | process overhead floor |
| `hybrid-zeroobs-a4-r16-s256` | 4 | 16 | 2 | 256 | zeros | actor parallelism without bigger total batch |
| `hybrid-zeroobs-a8-r16-s256` | 8 | 16 | 2 | 256 | zeros | subprocess scaling check |
| `hybrid-zeroobs-a8-r32-s256` | 8 | 32 | 2 | 256 | zeros | larger parent batch check |

Pass gates:

- env steps equal `actors * rows_per_actor * player_count * steps`;
- parent receives exactly one action per ready scalar env id;
- missing/extra action injection fails like the current scalar bridge:
  [source_state_batched_observation_mock_collector.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_mock_collector.py:242);
- actor payload bytes per physical row are at least an order of magnitude below
  rendered stack bytes;
- zero-observation throughput beats or at least approaches the one-process
  zero-observation manager at comparable total width before adding real render;
- no actor imports JAX/CUDA;
- output metadata records the exact observation backend as
  `hybrid_zero_observation_profile`, not `cpu_oracle` or scalar `jax_gpu`.

If this canary is slower than the one-process zero-observation manager, the
hybrid path is probably not worth real render work yet. The next fix would be
message shape/shared memory, not GPU rendering.

## Real Render Canary After Zero Obs

Only after the zero-observation canary passes, swap the parent observation
provider from zero stacks to central `direct_gray64`.

Keep the actor message unchanged except for compact render-state fields. The
parent should batch all ready rows into one central render call per harness step,
then scalarize. Compare:

- one-process zero observation;
- hybrid zero observation;
- one-process real central render;
- hybrid real central render;
- subprocess CPU-oracle anchor at the nearest collector width.

The first real-render result is useful only if the zero-observation hybrid row
already proved that actor subprocessing is not drowning in IPC. Otherwise the
real-render row will mix two unknowns.

## Recommendation

The smallest profile-only prototype is not a full new LightZero env manager. It
is a collector-shaped harness with real actor subprocesses, parent-owned
zero-observation first, and a central render-service seam that can be switched
to `direct_gray64` later.

This keeps the experiment honest:

- it preserves the subprocess parallelism that current stock rows benefit from;
- it avoids CUDA-in-child risk;
- it measures compact-state IPC before image-stack IPC can accidentally dominate;
- it keeps `final_observation`, partial autoreset, and RND as explicit gates;
- it gives a fast fail signal before spending more time on real batched render.
