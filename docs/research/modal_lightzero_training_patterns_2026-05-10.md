# Modal LightZero Training Patterns - 2026-05-10

Scope: hard critique of the current LightZero/Pong Modal wrapper and adjacent
Modal patterns, with official Modal docs prioritized over local examples. No
pytest was run.

## Bottom Line

The right default remains boring: one Modal Function owns one whole training
attempt or one whole eval scorecard, shared artifacts live in a Volume, and
monitoring reads committed Volume refs. Do not put Modal between MuZero and
environment steps, MCTS, replay sampling, or model inference.

The current exact Pong wrapper is closer after the `Function.spawn` patch, but
it is still a launch/run-management wrapper, not yet Modal's full long-training
pattern. It launches train mode in the background, writes progress snapshots,
and uses one mounted Volume. It does not yet resume from a latest checkpoint, so
it should not add `modal.Retries` or `single_use_containers=True` as if retries
were safe. It also should make `modal run --detach` an explicit command
requirement for background train launches.

## Claims

- Long/reentrant training: Modal's official long-training example says the
  generic pattern is periodic Volume checkpoints, start by checking the latest
  checkpoint, and add retries. It also uses explicit timeout,
  `modal.Retries(initial_delay=0.0, max_retries=10)`,
  `single_use_containers=True`, and `spawn(...).get()` under
  `modal run --detach`.
- Background train launch: `Function.spawn` is the right primitive when the
  local caller should not block on the result. `spawn` returns a FunctionCall
  handle that can later be polled or waited on. For deployed services, a
  deployed `Function.from_name(...).spawn(...)` is the stable way to submit
  jobs from another Python process.
- Volumes: commits/reloads are explicit consistency boundaries. Background
  commits exist, but other containers only see committed data after they reload.
  Concurrent modification of the same files is unsafe; use immutable payloads
  plus small pointers.
- Eval jobs: eval should be a separate Modal Function over the same image and
  Volume, reading immutable checkpoint refs and writing its own eval refs. Use
  `map`, `starmap`, or multiple `spawn` calls only across independent
  checkpoint/opponent/seed rows.
- Sandboxes: useful for isolated, command-oriented environment reconstruction,
  browser/renderer probes, image inspection, and snapshotting a prepared build.
  They are not the training primitive and not a per-step MuZero environment
  transport.

## Non-claims

- This does not claim the current Pong policy is improving.
- This does not claim a spawned ephemeral train job is robust without
  `modal run --detach` or a deployed submitter.
- This does not claim retries are safe before LightZero resume is implemented
  and tested from committed checkpoint pointers.
- This does not claim Modal Sandboxes are faster than Functions for training.
- This does not recommend Sandboxes for per-step environment simulation.

## Current Wrapper Critique

Current file:

- `src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py`

What is good:

- CPU dry path and GPU train path are separate Modal Functions. The GPU train
  Function has a long timeout, mounted `curvyzero-runs` Volume, fixed image,
  CPU/memory/GPU requests, and one whole `train_muzero` call.
- Train mode now uses `train_fn.spawn(...)` and prints a compact launch record
  with `function_call_id` and `progress_ref`.
- Progress snapshots and final summaries are written to the Volume and
  explicitly committed.
- The wrapper keeps LightZero training inside one container. That is the
  crucial hot-loop boundary.

What is still suboptimal:

- It is fire-and-return spawn, not the official long-training
  `spawn(...).get()` under `modal run --detach`. Fire-and-return is acceptable
  only if the launch command is detached or the function is invoked from a
  deployed app.
- It has no reentrant resume from a committed latest checkpoint pointer. That
  makes omission of retries correct for now, but it also means this is not yet
  the resilient long-job pattern.
- It writes progress by scanning the experiment tree. That is practical, but
  progress is observational, not a trainer state contract. Checkpoint pointers
  should be first-class.
- It uses direct Volume writes under `/runs` during training. That is acceptable
  for a single writer and LightZero's native path behavior, but avoid concurrent
  writers to the same run tree.

Recommended next change: add a small launch mode flag rather than more
infrastructure. `--mode train` should either:

- require/document `modal run --detach` for background launch, or
- support `--wait true` that uses `spawn(...).get()` for users who want CLI logs
  and long-training-example semantics.

Do not add `Retries` until resume is real.

## Volumes, Checkpoints, Progress

Use this contract:

- Write checkpoint payloads immutably, e.g.
  `checkpoints/lightzero/iteration_1234.pth.tar`.
- Write metadata next to each payload with run id, attempt id, iteration,
  config ref, code/dependency marker, feature/action schema, and sha256.
- Update `checkpoints/latest.json` only after payload and metadata are present.
- Update `checkpoints/best.json` only after an eval job has proven the
  checkpoint under the project-owned scorecard.
- Commit after payload+metadata+pointer writes that another Function or local
  monitor must see.
- Readers call `Volume.reload()` before reading progress/checkpoints from a
  long-running container.
- Never have multiple containers update the same pointer or checkpoint file at
  once. Separate eval rows write separate eval dirs.

Progress snapshots are useful. They should stay small and best-effort. They are
not a substitute for resumable trainer state.

## Eval Pattern

Eval should be separate from trainer return:

- Same image family as training, unless eval intentionally needs a smaller CPU
  image.
- Same `curvyzero-runs` Volume.
- Input is a checkpoint ref plus hash/config expectation.
- Output is an eval manifest with policy action histograms, score return,
  survival steps, truncations, shaped diagnostic returns, opponent id, seed(s),
  and checkpoint hash.
- Parallelism is at eval-row granularity: one row per
  checkpoint/opponent/seed/config. Use `Function.map`/`starmap` when the caller
  waits for the panel; use `spawn`/`spawn_map` when the caller submits a panel
  and watches Volume refs.

Do not block an eval pipeline on the trainer returning a Python object. The
trainer's durable contract is the Volume.

## Deployed App vs Ephemeral Run

Use ephemeral `modal run --detach` while the wrapper is changing daily. It is
simple and keeps source iteration cheap.

Move to `modal deploy` plus `modal.Function.from_name(APP, FN).spawn(...)`
when:

- repeated train/eval submissions are cluttering ephemeral app history;
- a local script, dashboard, or queue-like submitter needs stable function
  names;
- launch latency and grouped observability matter more than ad hoc iteration.

Do not introduce a web endpoint just to submit Python-owned train/eval jobs.
The Modal Python SDK is a better fit for Python callers.

## Sandboxes

Recommendation: keep Sandboxes as an environment reconstruction and diagnostics
tool, not as the MuZero runtime substrate.

Useful Sandbox jobs for CurvyTron:

- Build or inspect a browser/JS environment with arbitrary OS/package steps.
- Run a headless renderer or dev server behind a tunnel for manual/automated
  diagnostics.
- Execute untrusted or generated environment probes in isolation.
- Inspect a registry image or source checkout and copy artifacts back.
- Snapshot a prepared filesystem containing ROMs/assets/browser deps/build
  outputs so later diagnostic Sandboxes start faster.
- Branch the same prepared environment into several probe variants.

Non-recommendation:

- Do not call a Sandbox per environment step, per MCTS expansion, per episode
  step batch, or per policy inference. `Sandbox.exec` is a subprocess/control
  API with stdout/stderr/stdin handles, not an in-process simulator call.
- Do not replace Function-based training with a long-lived Sandbox unless the
  job is fundamentally interactive/service-shaped.
- Do not rely on Memory Snapshots for core training reproducibility; they are
  alpha and short-retention. Filesystem snapshots are the practical one to
  consider.

Practical CurvyTron sandbox lane:

1. Keep canonical training/eval as Functions plus Volumes.
2. Add one disposable Sandbox probe only if a browser/source reconstruction
   needs dynamic commands, tunnels, or image inspection that Functions make
   awkward.
3. If setup is expensive and repeated, snapshot the prepared filesystem and
   record the snapshot image id in a Volume manifest.
4. Promote stable outputs back into normal Images, source files, and Volumes.
   Sandboxes are a lab bench, not the production training interface.

## Local Examples Used

- `/Users/shankha/modal-examples/06_gpu_and_ml/long-training.py`: official local
  copy of the long-training pattern; checkpoints on a Volume, retries, timeout,
  `single_use_containers=True`, and `spawn(...).get()` under detached run.
- `/Users/shankha/modal-examples/06_gpu_and_ml/unsloth_finetune.py`: one
  whole finetune Function with model/dataset/checkpoint Volumes, retries, and
  resume from latest `checkpoint-*`.
- `/Users/shankha/modal-examples/06_gpu_and_ml/yolo/finetune_yolo.py`: separate
  train and inference/eval-shaped functions over one Volume; uses `reload()`
  before training reads dataset state.
- `/Users/shankha/modal-examples/06_gpu_and_ml/dreambooth/diffusers_lora_finetune.py`:
  fine-tune writes model artifacts to a Volume and commits before downstream
  serving/inference reloads.
- `/Users/shankha/modal-examples/13_sandboxes/safe_code_execution.py`: simple
  Sandbox command execution across languages using `Sandbox.exec`.
- `/Users/shankha/modal-examples/13_sandboxes/jupyter_sandbox.py`: service-like
  Sandbox with encrypted tunnel and readiness polling.
- `/Users/shankha/modal-examples/13_sandboxes/opencode_server.py`: long-lived
  interactive service in a Sandbox with tunnel access.
- `/Users/shankha/modal-projects/flash-projects/lovable/scripts/inspect_lovable_sandbox_image.py`:
  disposable Sandbox for image inspection, command execution, artifact copyout,
  and optional Volume commit.
- `/Users/shankha/modal-projects/flash-projects/lovable/scripts/environment_execution_modal.py`:
  one-live-Sandbox-per-session pattern with explicit project sync; useful as a
  cautionary example for command/session tooling, not RL hot loops.

## Official Sources

- Modal long training: https://modal.com/docs/examples/long-training
- Modal Function reference: https://modal.com/docs/reference/modal.Function
- Modal Volumes: https://modal.com/docs/guide/volumes
- Modal timeouts: https://modal.com/docs/guide/timeouts
- Modal retries: https://modal.com/docs/guide/retries
- Modal `modal run --detach`: https://modal.com/docs/reference/cli/run
- Modal managing deployments: https://modal.com/docs/guide/managing-deployments
- Modal invoking deployed functions:
  https://modal.com/docs/guide/trigger-deployed-functions
- Modal scaling/map/spawn limits: https://modal.com/docs/guide/scale
- Modal Sandboxes: https://modal.com/docs/guide/sandboxes
- Modal Sandbox commands: https://modal.com/docs/guide/sandbox-spawn
- Modal Sandbox snapshots: https://modal.com/docs/guide/sandbox-snapshots
- Modal Sandbox reference: https://modal.com/docs/reference/modal.Sandbox
