# Modal Training Execution Plan

Date: 2026-05-09

Status: simplification pass. Modal is the real execution target for training
and eval. Local runs are for tiny debug only.

Working-memory rule: the main thread plans, delegates, and decides. Workers do
the edits, runs, and docs. Keep this note as the Modal/Mctx reference for the
active backlog, not as a separate priority list.

## Current Truth / No More Pretending

- We validated stock LightZero CartPole MuZero progression.
- We validated an Mctx search benchmark.
- We validated a CEM-v2 Pong baseline.
- We validated a raster-only MLP Pong baseline.
- We have not run an actual project-owned MuZero/Mctx train loop for Pong.
- We have not run an actual project-owned MuZero/Mctx train loop for Curvy.
- CEM-v2 and the MLP are baselines and scaffolding only. They are not MuZero
  progress.
- The next main lane is LightZero-first: adapt dummy Pong as a custom env and
  run a capped LightZero MuZero trainer on Modal. Project-owned Mctx is fallback
  if LightZero fails or hides required telemetry/artifacts.

Prevention rules:

- Prove the target is scoreable before scaling it.
- Keep baselines separate from MuZero.
- Name the algorithm in every experiment title, command, and summary.
- Distinguish stock LightZero MuZero from project-owned MuZero/Mctx.
- Do not describe CEM, imitation, or MLP results as MuZero progress.

Architecture pointer: the high-level MuZero-on-Modal map now lives in
[`docs/design/muzero_modal_architecture.md`](../design/muzero_modal_architecture.md).
Its main decision is to start with one container owning env stepping, self-play,
Mctx search, replay sampling, training, checkpointing, and small eval probes;
Modal primitives coordinate coarse jobs and durable artifacts only.

This note is blunt on purpose. The current Modal direction has the right core
idea, but too many small wrappers and smoke variants are starting to hide the
simple pattern we need.

## Blunt Critique

What is good:

- The important rule is already documented: do not put Modal calls, Queues, or
  Dicts inside environment ticks, MCTS expansion, model inference batches, or
  optimizer steps.
- `curvyzero-runs` Volume works as durable remote storage for small artifacts.
- `run_management.py` has useful boring helpers for safe ids, refs, JSON
  manifests, hashes, and checkpoint pointers.
- `dummy_survival_train_attempt` and `dummy_pong_scoreboard_attempt` prove the
  right coarse-job shape: one Function owns a whole train or eval job, writes
  files to a Volume, commits, and returns compact refs.
- `dummy_pong_cem_train_attempt` now applies that shape to the CEM-v2 lag-1
  Pong baseline: one CPU Function writes summary, checkpoint, compact rows,
  manifests, and a latest checkpoint pointer to `curvyzero-runs`. This is not
  MuZero.
- `dummy_pong_imitation_train_attempt` applies the same one-Function,
  Volume-backed shape to the stack-2 `raster_only` MLP imitation baseline. It
  reads an existing replay ref, copies replay rows into the attempt, trains the
  NumPy learner, writes summary/checkpoint refs, and updates the latest
  checkpoint pointer. This is not MuZero.
- `mctx_dependency_smoke.py` is the right kind of dependency smoke: tiny,
  synthetic, and separate from training claims.

What is too complex:

- There are too many one-off Modal modules:
  `artifact_smoke`, `fidelity_smoke`, `dummy_survival`, `volume_dummy_survival`,
  `dummy_survival_train_attempt`, `dummy_line_duel`,
  `dummy_pong_scoreboard_attempt`, `mctx_dependency_smoke`, and `smoke`. Some
  are useful history; they should not become the architecture.
- We are duplicating image definitions, Volume setup, id validation, manifest
  writes, and run/attempt transitions across modules.
- Ephemeral artifact wrappers are now mostly obsolete. Once a task writes
  training outputs, it should write to the runs Volume.
- The docs mention deployed functions and `Function.from_name`, but the code is
  still mostly `modal run` entrypoints. That is fine for iteration, but serious
  launchers should target a deployed app.
- The current Pong training discussion has more checkpoint-vs-parent machinery
  than learning evidence. Wins against `track_ball` are the hard gate, but 0/64
  wins alone is too blunt to diagnose the learner. Every current Pong learner
  comparison also needs episode length, truncation rate, and shaped return proxy
  against `track_ball` so we can see survival/loss-delay movement before wins.

What is wrong or misleading:

- `smoke.run_tests` exists, but this lane must not use pytest. Keep it as a
  general infra smoke, not a training workflow step.
- `volume_dummy_survival` writes to a deterministic path based on seed and
  config. That is okay as an old smoke, but it is not retry-safe run
  management. Prefer the run/attempt layout.
- `dummy_survival_train_attempt` mirrors periodic checkpoints only after the
  local trainer returns. A long real trainer must checkpoint and commit inside
  the training loop.
- `mctx_dependency_smoke.py` and `mctx_gpu_dependency_smoke.py` exist and passed
  on Modal CPU plus an L4 GPU on 2026-05-09, so older "missing GPU dependency
  smoke" language is stale.
- A Modal Volume checkpoint is not a resume contract by itself. Resume needs
  model state, optimizer state, replay cursor/chunks, RNG state or deterministic
  cursors, completed steps, config, code/dependency fingerprint, and a clear
  latest pointer.

What is missing:

- One minimal training app that owns the default training/eval Functions.
- One shared image module or helper so CPU, JAX/Mctx GPU, and PyTorch GPU images
  are defined once.
- One cache Volume for model/dependency caches when using Hugging Face, torch,
  or JAX compile caches.
- A stronger Pong trainer. The minimal
  `curvyzero.infra.modal.dummy_pong_train_attempt` wrapper now writes replay,
  checkpoints, summaries, manifests, and a latest checkpoint pointer for remote
  reproduction of the current weak self-play loop.
- A real resume loop inside the training Function.
- A deployed app and local launcher using `modal.Function.from_name` for serious
  jobs.
- Eval fanout only after the single coarse scoreboard job is too slow.

## Minimal Pattern

Use one active training app:

```python
app = modal.App("curvyzero-training")
```

Use three image definitions:

```python
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy>=1.26")
    .env({"PYTHONPATH": "/repo/src"})
    .add_local_dir("src", remote_path="/repo/src", copy=True)
    .add_local_dir("scripts", remote_path="/repo/scripts", copy=True)
)

jax_mctx_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy>=1.26", "mctx==0.0.6", "jax[cuda12]==0.7.0")
    .env({
        "PYTHONPATH": "/repo/src",
        "XDG_CACHE_HOME": "/cache/xdg",
    })
    .add_local_dir("src", remote_path="/repo/src", copy=True)
)

torch_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy>=1.26", "torch==2.7.1")
    .env({
        "PYTHONPATH": "/repo/src",
        "HF_HOME": "/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/cache/huggingface/hub",
        "TRANSFORMERS_CACHE": "/cache/huggingface/transformers",
        "TORCH_HOME": "/cache/torch",
        "XDG_CACHE_HOME": "/cache/xdg",
    })
    .add_local_dir("src", remote_path="/repo/src", copy=True)
)
```

Use two Volumes:

```python
runs_volume = modal.Volume.from_name("curvyzero-runs", create_if_missing=True)
cache_volume = modal.Volume.from_name("curvyzero-cache", create_if_missing=True)

RUNS = Path("/runs")
CACHE = Path("/cache")
```

Use one run layout:

```text
/runs/training/<task_id>/<run_id>/
  run.json
  latest_attempt.json
  attempts/<attempt_id>/
    attempt.json
    train/
      summary.json
      metrics.jsonl
      replay/
      checkpoints/
      logs/
  checkpoints/
    step-000001000/
      checkpoint.*
      trainer_state.json
    latest.json
    best.json
  eval/<eval_id>/
    summary.json
    episodes.jsonl
```

Use only these Functions at first:

- `train_pong_attempt(config)`: one whole training attempt in one container.
  Replay build, self-play, model inference/search, replay sampling, optimizer
  updates, checkpointing, and health metrics stay inside this Function. It
  commits after material checkpoints and at the end.
- `score_pong_checkpoint(config)`: one coarse eval/scoreboard job. It reads
  checkpoint refs, writes eval artifacts, commits, and returns a compact table.
  For Pong, the compact table must include wins plus survival/loss-delay
  signals against `track_ball`: mean episode length, truncation rate, and shaped
  return proxy. Wins are the promotion gate; these metrics are required to
  diagnose learning while wins are still zero.
- `mctx_dependency_smoke`: tiny CPU dependency check only.
- `mctx_gpu_dependency_smoke`: tiny GPU dependency check only.
- `mctx_synthetic_benchmark(config)`: one GPU Function that measures JAX/Mctx
  compile time, steady-state search throughput, memory-sensitive shape fields,
  and output sanity for fixed synthetic search shapes.

Use `Function.map`/`starmap` only for coarse independent work:

- eval shards over seed ranges;
- checkpoint candidates in a sweep;
- replay conversion chunks;
- independent hyperparameter attempts.

Do not use `Queue`, `Dict`, or Function calls for action selection, MCTS nodes,
per-step replay writes, or per-batch gradient updates.

## Current Modal Truth

- Yes, the current Modal pattern is correct for now: simple coarse Functions,
  only needed local files in images, `curvyzero-runs` Volume for checkpoints and
  summaries, and no Queue/Dict/Function calls in the hot loop.
- Current Pong train/eval runs are actually on Modal when launched through
  `dummy_pong_train_attempt`, `dummy_pong_cem_train_attempt`, and
  `dummy_pong_imitation_train_attempt`, then scored through
  `dummy_pong_scoreboard_attempt`.
- Those Pong runs are baselines/scaffolds. They are not a project-owned
  MuZero/Mctx train loop.
- Current Pong train/eval runs are CPU NumPy jobs. They do not request Modal
  GPUs.
- Modal GPU is proven only for JAX/Mctx dependency smokes and the synthetic
  benchmark lane, not for the current Pong learner.
- Do not move the current NumPy Pong learner to GPU. It has no GPU work.
- The next Pong learner that uses JAX/Mctx search or GPU model training should
  use a single L4/T4 Modal GPU Function from the start.
- Do not judge the current Pong learner on `track_ball` wins alone while wins
  are still zero. Wins remain the hard gate, but the shaped-learning read is:
  longer episodes, more truncations, better loss-delay/shaped return proxy, and
  no collapse against `random_uniform`.
- Current Modal/Mctx support lane: benchmark the fixed-shape synthetic Mctx
  path only as fallback/comparison. The next main lane is LightZero custom
  dummy Pong config/train smoke. Do not connect the Mctx benchmark to
  environments, replay, or trainers until LightZero has either passed or failed
  clearly.

## Purge From The Docs

- Purge generation/promotion language that implies "new checkpoint beats parent"
  is enough. Parent-vs-child is secondary.
- Purge any framing that says 0/64 wins against `track_ball` by itself fully
  judges the learner. It fails the hard gate, but it does not tell us whether a
  shaped objective is moving in the right direction.
- Purge model-server, Queue, Dict, actor fleet, league, Elo, cluster, and
  multi-node plans from the active Pong path.
- Purge GPU scale language for the current NumPy Pong trainer.
- Purge old one-off smoke modules from the active architecture. Keep them as
  history or primitive checks only.

## Exact Pong Lane

These are CPU Modal baseline lanes, not MuZero lanes. Run them only if we are
explicitly staying in baseline land. The next main lane should otherwise be the
LightZero custom dummy Pong config/train smoke.

```text
dummy_pong_train_attempt
  policy: random_uniform or current selected checkpoint
  objective: current learner plus explicit survival/loss-delay shaped return
  artifacts: replay, checkpoints, metrics, latest pointer

dummy_pong_cem_train_attempt
  target: current scoreable lagged_track_ball_1 CEM-v2 baseline, not MuZero
  objective: score-primary CEM selection with survival/loss-delay as tie-break
  artifacts: summary, checkpoint, CEM rows, manifests, latest pointer

dummy_pong_imitation_train_attempt
  target: existing stack-2 lag-1 exact-trace replay
  objective: supervised MLP behavior cloning on raster_only frame-stack rows,
    not MuZero
  artifacts: copied replay refs, summary, checkpoint, manifests, latest pointer

dummy_pong_scoreboard_attempt
  opponents: random_uniform, track_ball, parent/previous, selected
  required rows: wins, losses, draws/truncations, mean episode length,
    truncation rate, shaped return proxy against track_ball
```

Stop if the shaped metrics are flat and random performance collapses. Continue
only if survival/loss-delay improves against `track_ball`; promote only after
actual `track_ball` wins appear.

## Function Shape

The training Function should look boring:

```python
@app.function(
    image=jax_mctx_image,
    gpu=["L4", "T4"],
    cpu=8,
    memory=32768,
    timeout=12 * 60 * 60,
    startup_timeout=20 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=0.0),
    single_use_containers=True,
    volumes={RUNS: runs_volume, CACHE: cache_volume},
)
def train_pong_attempt(config: dict) -> dict:
    runs_volume.reload()
    state = load_latest_state_or_start_new(config, RUNS)
    while not state.done:
        run_self_play_and_train_inside_this_container(state)
        if state.should_checkpoint:
            write_checkpoint_payload_first(state, RUNS)
            write_latest_pointer_last(state, RUNS)
            runs_volume.commit()
    write_summary(state, RUNS)
    runs_volume.commit()
    return compact_result_refs(state)
```

This is the design. Everything else is scaffolding.

## Volume Rules

- Write payloads first, pointer files last.
- Write immutable checkpoint/replay paths. Do not have two containers write the
  same file.
- Commit after real checkpoints and at job end.
- Call `runs_volume.reload()` at Function start when reading files that another
  Function may have committed.
- Keep replay chunks chunky. Avoid many tiny files.
- Store caches separately from run artifacts. A cache Volume can be deleted or
  rebuilt without losing experiment records.

Suggested cache mount policy:

```text
/cache/huggingface
/cache/huggingface/hub
/cache/huggingface/transformers
/cache/torch
/cache/xdg
/cache/jax
```

Only add a cache once the dependency uses it. Do not mount a model cache into a
NumPy-only CPU job.

## Scaling Path

1. CPU Modal for Pong replay/train/eval wrappers.
2. GPU Modal dependency smoke for JAX/Mctx.
3. GPU Modal synthetic Mctx benchmark with fixed shapes.
4. One-container GPU MuZero-style training attempt.
5. Eval fanout with `map`/`starmap`.
6. Multi-node only after one-container training is clearly bottlenecked and the
   checkpoint schema is stable.

Do not start with Modal clusters. Modal multi-node is powerful, but it adds
rank, network, fault-tolerance, and distributed-launcher concerns before we
have a single-container learner worth scaling.

## Local Examples Inspected

- `/Users/shankha/modal-examples/06_gpu_and_ml/long-training.py`: the useful
  pattern is checkpoint to Volume, resume from latest checkpoint, add retries,
  set timeout, and use `spawn(...).get()` for interruptible jobs.
- `/Users/shankha/modal-examples/14_clusters/simple_torch_cluster.py`: useful
  later for `torch.distributed.run` under `@modal.experimental.clustered`; not
  the next CurvyZero step.
- `/Users/shankha/modal-examples/06_gpu_and_ml/import_torch.py`: simple GPU
  torch image/function smoke.
- `/Users/shankha/modal-examples/06_gpu_and_ml/gpu_fallbacks.py`: GPU fallback
  lists are valid for cheap smokes.
- `/Users/shankha/modal-projects/flash-projects/lovable/scripts/train_open_model_sft_modal.py`:
  good cache-Volume pattern for Hugging Face and run outputs, plus `spawn` for
  launch. It is too heavyweight for CurvyZero Pong, but the storage split is
  right.

## Official Modal Sources

- Images and local code: https://modal.com/docs/guide/images
- CUDA on Modal: https://modal.com/docs/guide/cuda
- GPU selection: https://modal.com/docs/guide/gpu
- Volumes, commits, reloads, and file-count limits:
  https://modal.com/docs/guide/volumes
- Volume reference and concurrent-write warning:
  https://modal.com/docs/reference/modal.Volume
- Long resumable training example:
  https://modal.com/docs/examples/long-training
- Timeouts and `startup_timeout`: https://modal.com/docs/guide/timeouts
- Retries: https://modal.com/docs/guide/retries
- Deployed function lookup and `Function.from_name`:
  https://modal.com/docs/guide/trigger-deployed-functions
- Function `remote`, `spawn`, `map`, `starmap`, and `spawn_map`:
  https://modal.com/docs/reference/modal.Function
- Multi-node training: https://modal.com/docs/guide/multi-node-training
- Queues, for coarse active communication only:
  https://modal.com/docs/guide/queues

## Exact Next Commands

Do not run pytest for this lane.

First, prove the current dependency smoke after the main-thread image fix:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_dependency_smoke --kind cpu
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_gpu_dependency_smoke
```

Then keep the known durable CPU train-attempt smoke healthy:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_survival_train_attempt \
  --iterations 1 \
  --episodes-per-iter 2 \
  --seed 0 \
  --eval-episodes 2 \
  --checkpoint-every-iterations 1
```

Then run the minimal Pong train-attempt wrapper, with no new Modal primitives:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_train_attempt \
  --games 16 \
  --epochs 5 \
  --seed 0
```

This command proves Modal artifact discipline and remote reproduction for the
current dummy Pong self-play trainer. It does not prove that the objective is
correct or that the learned policy is strong.

The simplest Modal wrapper for the stack-2 raster-only MLP imitation baseline
is now `dummy_pong_imitation_train_attempt`. This is supervised baseline work,
not MuZero. Upload the existing local replay rows once, then train from the
Volume ref:

```sh
modal volume put curvyzero-runs \
  artifacts/local/dummy-pong-lag1-trace-replay-stack2-smoke-2026-05-09/replay_rows.jsonl \
  training/dummy-pong/manual-replays/lag1-trace-stack2/replay_rows.jsonl
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_imitation_train_attempt \
  --replay-path ref:training/dummy-pong/manual-replays/lag1-trace-stack2/replay_rows.jsonl \
  --seed 0 \
  --epochs 800 \
  --learning-rate 0.005 \
  --validation-fraction 0.2 \
  --class-weighting balanced \
  --feature-mode raster_only \
  --frame-stack 2 \
  --model-type mlp \
  --hidden-dim 128
```

This should write:

```text
training/dummy-pong/<run_id>/attempts/<attempt_id>/replay/replay_rows.jsonl
training/dummy-pong/<run_id>/attempts/<attempt_id>/train/summary.json
training/dummy-pong/<run_id>/attempts/<attempt_id>/train/checkpoint.npz
training/dummy-pong/<run_id>/checkpoints/latest.json
```

Then score the Modal checkpoint before comparing it to CEM-v2:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --checkpoints mlp_stack2=ref:training/dummy-pong/<run_id>/attempts/<attempt_id>/train/checkpoint.npz \
  --episodes 32 \
  --seed 9050913 \
  --split-id dummy_pong_lag1_raster_only_mlp_modal \
  --split-role monitor
```

The CEM-v2 lag-1 baseline now has the same Modal-backed pattern. This is
geometry-CEM baseline work, not MuZero:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_cem_train_attempt \
  --width 15 \
  --height 9 \
  --paddle-height 3 \
  --max-steps 120 \
  --generations 8 \
  --population-size 32 \
  --elite-count 8 \
  --eval-games 16 \
  --seed 8050913 \
  --opponent-weights lagged_track_ball_1=1.0,random_uniform=0.10,track_ball=0.10 \
  --target-opponent-id lagged_track_ball_1 \
  --loss-delay-weight 0.5 \
  --truncation-value 0.0
```

Observed 2026-05-09 Modal pass:
`ap-SzIu3KSSe7NRAq2Iqn33Yu`,
`pong-cem-20260509T045950Z-e8b06974a402`,
`attempt-20260509T045950Z-f16d342d760b`. The final eval scored 25/32
learner wins versus `lagged_track_ball_1`, 30/32 versus `random_uniform`, and
32/32 truncations versus default `track_ball`.

After it writes a checkpoint Volume ref, score it remotely:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --checkpoints latest=ref:training/dummy-pong/<run_id>/attempts/<attempt_id>/train/checkpoint.npz \
  --episodes 8 \
  --seed 331 \
  --split-id dummy_pong_selfplay_smoke_monitor \
  --split-role monitor
```

Then run the tiny synthetic Mctx runtime benchmark:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --batch-size 8 \
  --num-simulations 4 \
  --hidden-dim 32 \
  --max-depth 4 \
  --warmup-runs 1 \
  --steady-runs 2
```

Interpret the JSON as a runtime benchmark, not learning evidence. A clean run
has `ok: true`, a GPU JAX backend, package versions present, finite normalized
`action_weights`, compile-plus-first-run timing separated from steady-state
timing, and nonzero decisions/sec plus simulations/sec. Do not connect a real
environment, replay buffer, or trainer until this fixed-shape synthetic path is
boring.
