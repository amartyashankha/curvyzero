# Hybrid Actor + Batched Observation Prototype Plan

Date: 2026-05-21

Status: planned. Profile-only. Do not touch live training, trainer defaults,
tournament defaults, checkpoint metadata, eval, or GIF jobs.

## Plain Goal

The current one-process batched GPU manager tops out around the current C512
plateau. Renderer work still matters, but C512 real render is already within
about `1.25x` of the stable zero-observation ceiling. That means a bigger win
needs a bigger shape:

```text
many CPU actors step CurvyTron compact state in parallel
-> one parent batches compact state
-> one GPU observation service renders [B,2,4,64,64]
-> scalar LightZero-shaped payload only at the outer edge
```

This prototype asks one question:

```text
Can actor parallelism plus central batched observation beat the one-process
zero-observation ceiling before we spend time on real render?
```

If the answer is no, the hybrid lane is not worth wiring into LightZero yet.

## Why This Is Different

The current batched GPU profile manager does this:

```text
one process owns all env rows
-> one process steps env
-> one process renders
-> one process builds scalar LightZero timesteps
```

That proves batching, but it gives up the subprocess actor parallelism that
made the CPU-oracle path fast.

The hybrid prototype keeps the two useful ideas separate:

- CPU actors are good at stepping independent CurvyTron rows in parallel.
- GPU is good at rendering/inference/search large batches.

## First Version

Do not call stock `train_muzero` yet.

First prototype:

```text
parent process
  -> starts N actor subprocesses
  -> sends action batches
  -> receives compact source state + reward + done + generation
  -> uses zero observation first
  -> materializes LightZero-shaped scalar obs/timestep dicts
  -> measures every boundary
```

Only after zero observation beats the one-process zero ceiling should the same
harness switch to direct GPU render.

## Code Ownership

Suggested new files:

- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `scripts/profile_hybrid_batched_observation_manager.py`
- `tests/test_source_state_hybrid_observation_profile.py`

Reuse existing seams:

- `VectorMultiplayerEnv` for source-state stepping.
- `SourceStateBatchedRenderRequest` and renderer protocol.
- `materialize_trainer_surface_env_id_timestep` only if it stays clean; do not
  force trainer defaults through this prototype.

## Metrics Required

Per measured step:

- actor env-step time;
- actor idle/wait time;
- parent send/receive time;
- compact payload bytes;
- parent gather/merge time;
- observation time: zero first, then real render if zero passes;
- stack update time;
- scalar timestep materialization time;
- ready obs count;
- timestep count;
- live physical row count;
- terminal/autoreset row count;
- total effective env steps/sec.

## Pass Criteria

Zero-observation hybrid passes only if:

- it is `profile_only=true` and `calls_train_muzero=false`;
- deterministic small-seed row/player/reward/done/reset checks match the
  one-process oracle;
- compact actor payload is much smaller than shipping rendered stacks;
- it beats the one-process C512 zero-observation ceiling (`~1805 steps/s`) by
  at least `20%`, or clearly scales upward with actor count while C768
  one-process does not.

Real-render hybrid is allowed only after zero-observation passes.

## Fail Criteria

Stop this lane if:

- actor IPC/gather time dominates before real render is enabled;
- hybrid zero observation cannot match one-process zero observation;
- correctness requires changing trainer defaults;
- it needs full rendered stacks crossing subprocess boundaries;
- it loses row/player/reset/terminal semantics.

## Relationship To Current Grid

The active saturation grid decides whether C768 is a real plateau:

- If C768 stays flat or worse, this hybrid prototype becomes the next serious
  architecture lane.
- If C512/C768 real rows unexpectedly approach zero rows and sim4 behaves well,
  do targeted renderer/stack cleanup first.
- If sim4 makes both real and zero much slower, search/policy scheduling needs
  equal priority with this actor work.

## Implementation Note

First scaffold landed as an in-process, profile-only zero-observation harness in
`source_state_hybrid_observation_profile.py`, with a JSON profile script and
focused tests. It keeps `profile_only=true`, `calls_train_muzero=false`, and
does not change trainer defaults; multiprocessing and real render remain future
gates.
