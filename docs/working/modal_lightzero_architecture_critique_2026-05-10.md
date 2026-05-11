# Modal LightZero Architecture Critique - 2026-05-10

## Claim

The current LightZero/Pong Modal architecture is directionally correct for the
near-term control lane: one coarse GPU training Function owns a whole attempt,
artifacts are rooted in the `curvyzero-runs` Volume under attempt-specific refs,
long training is launched with `Function.spawn`, and eval is implemented as a
separate Modal Function that can load committed checkpoints while training
continues.

## Non-claim

This does not prove the policy is good, the frequent-checkpoint run will finish,
or the artifact contract is already clean enough for unattended train/eval
curves. It also does not prove a deployed service is needed before the current
manual control loop has produced a stronger same-run checkpoint curve.

## Evidence

- `src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py` mounts
  `curvyzero-runs` at `/runs`, changes the training working directory to
  `/runs`, and sets LightZero `exp_name` to a relative attempt train ref. This
  is the right workaround for DI-engine/LightZero writing `./<exp_name>` style
  checkpoint paths into the current directory.
- The train wrapper commits the Volume after the starting progress snapshot,
  after each progress watcher snapshot, after the final progress snapshot, and
  after `summary.json`, `attempt.json`, and `latest_attempt.json`.
- The local train entrypoint now uses `train_fn.spawn(...)` for `mode=train`.
  That fixes the earlier coupling between a long remote trainer and the local
  `modal run` process.
- `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py` is a separate app and
  Function. It reads checkpoint refs from the same Volume and writes immutable
  per-checkpoint eval JSON plus a manifest for multi-checkpoint runs.
- Eval already has a parallel path: multiple checkpoint refs or `--parallel`
  uses `eval_fn.map(...)`, then writes a manifest through a small separate
  Function.

## Critique

Volumes are mostly being used correctly, but the contract is still "LightZero
writes files, progress snapshots scan and commit what exists." That is good
enough for the current lane, but it is not yet a clean checkpoint publication
API. There is no small `checkpoints/latest.json` pointer produced immediately
after each checkpoint, and there is no checkpoint-ready manifest independent of
the expensive directory scan. Because Modal Volume visibility depends on
`commit()`, eval can only see new checkpoints after the progress watcher or final
summary commits them. With the default five-minute progress interval, a new
checkpoint may sit invisible to a separate eval job for up to roughly one
progress interval.

Train and eval are separated enough for now. Eval does not run inside the trainer
process, does not need trainer completion, and writes to attempt eval paths
instead of train paths. The remaining coupling is operational rather than
architectural: someone or something must poll progress/checkpoints and launch
eval at the right time. The current setup can evaluate while train continues as
long as the checkpoint file has been fully written and committed to the Volume.

Do not jump straight to deployed functions as the next architectural step.
Deployment can make invocation cleaner later, but it does not solve the main
loop problem. The next useful step is a small monitor/poller that watches the
Volume for newly committed `iteration_*.pth.tar` checkpoints, records which ones
have been evaled, and spawns eval jobs. This can start as a local script or Modal
Function. Only deploy it once the checkpoint/eval manifest contract is boring.

Eval is not forced to wait for trainer completion, and it is not inherently
serial: the parallel `eval_fn.map(...)` path exists. We may still be wasting time
in practice if operators run single-checkpoint high-detail evals manually, use
stock evaluator probes too often, or wait for run completion before launching the
curve. The cheap loop should be: low-detail strict eval every new checkpoint,
then high-detail or stock-evaluator parity only on interesting checkpoints.

## Prioritized Recommendations

1. Keep using `modal run --detach` plus `Function.spawn` for long training.
   Treat Volume artifacts, progress JSON, and Modal function-call id as the
   observable interface; do not wait on the trainer's return value for run
   management.

2. Add a tiny checkpoint discovery/eval monitor before deploying anything. It
   should call `volume.reload()`, read the latest progress snapshot, discover
   committed `iteration_*.pth.tar` files under the attempt's LightZero exp dir,
   skip already-evaled checkpoint refs, and spawn low-detail strict eval jobs.

3. Reduce progress latency for frequent-checkpoint runs. For checkpoint cadence
   `1000`, use a shorter progress interval such as 60 seconds during the signal
   curve phase so checkpoint publication is not delayed by the default 300
   seconds.

4. Write a cheap checkpoint-ready pointer or manifest. Best small shape:
   `train/checkpoints_seen.json` or `train/progress/checkpoints_latest.json`
   containing checkpoint ref, iteration, size, mtime, and timestamp after a
   Volume commit. This can be generated by the progress watcher without changing
   LightZero internals.

5. Make low-detail eval the default curve path. Use `--parallel` or multiple
   selected iterations for checkpoint batches, `--eval-pass low` for first pass,
   strict loading with `--allow-model-fallback false`, and reserve 512-step plus
   stock evaluator probes for the baseline and promising later checkpoints.

6. Keep eval artifacts immutable and manifest-led. The existing per-checkpoint
   output refs and parallel manifest are good; add a compact "curve summary"
   table that reports `steps_survived`, `delta_steps_survived` versus same-run
   `iteration_0`, return, stock return if run, action histogram, fallback status,
   and artifact ref.

7. Deployed functions are a second-order improvement. Deploy train/eval/monitor
   only after the monitor protocol is settled, mainly to make repeated launches
   less dependent on local source packaging and command spelling.

## Sandboxes: Use For X, Avoid For Y

Use Sandboxes for reconstruction and isolation work around the training lane,
not for the trainer hot loop. Modal Sandboxes are runtime-created containers for
arbitrary commands, isolated code, custom dependencies, and state snapshots.
They can use Images and Volumes like Functions, and Sandbox snapshots can save a
filesystem state for later reuse. That makes them useful around the edges of the
LightZero workflow.

Use Sandboxes for:

- Environment reconstruction probes: recreate a suspect LightZero/ALE/Gym import
  environment, run shell-level probes, inspect package state, and snapshot the
  filesystem once the environment is known-good.
- Browser or render isolation: run headless rendering, video/frame extraction,
  notebook/browser checks, or Gym/ALE display experiments without polluting the
  training Function image.
- Asset snapshotting: preserve ROM install state, generated debug frames,
  videos, repro scripts, or one-off dependency experiments as a filesystem
  snapshot or as files copied into the `curvyzero-runs` Volume.
- Untrusted or generated code checks: run generated scripts, temporary notebooks,
  or throwaway repro commands in an isolated container before turning them into
  committed Functions.
- Debug branch experiments: start from the same environment snapshot and try
  small dependency or wrapper changes independently.

Avoid Sandboxes for:

- The LightZero trainer hot loop. The current Function owns resource requests,
  timeout, source packaging, Volume commits, progress snapshots, and structured
  return artifacts more clearly.
- Routine checkpoint eval curves. The existing eval Functions already load
  checkpoints, write manifests, and support parallel `map`; moving this to
  Sandboxes would add process-management surface without improving the artifact
  contract.
- Durable run orchestration. Volumes and small JSON manifests are the durable
  coordination layer; Sandbox filesystem snapshots should not become the source
  of truth for checkpoints or eval results.
- Long jobs that need clean retries/resume. Functions plus Volume checkpoints are
  easier to reason about for preemption and reruns. Use Sandbox snapshots only
  to preserve or fork environment state, not to replace checkpointing.

Compared with the current Function+Volume pattern, Sandboxes are a better
scratchpad and snapshot tool. Functions are the better productized interface for
repeatable train/eval jobs. The practical split is: use Functions for
`train`, `eval`, and `monitor`; use Sandboxes to answer "can this environment or
render path be reconstructed exactly?" before promoting the answer back into an
Image, Function, or Volume artifact.

## Cheap Changes For Faster Iteration

- Launch the current frequent-checkpoint run with `--progress-interval-sec 60`
  on the next replacement attempt.
- Eval as soon as `iteration_1000` is committed; do not wait for final training
  completion.
- Batch same-run baseline plus new checkpoints in one parallel eval command.
- Keep `run_stock_evaluator=false` on routine low-detail checks.
- Use a stable eval id per curve, for example `live-ckpt1000-curve`, so all
  manifests are easy to find under one attempt eval root.
- Add a one-page command crib next to the active board once the first monitor
  loop has worked once.

## Code Edit Guidance

No code edit is required before the next run. The only tiny safe code change I
would consider next is adding a helper that extracts checkpoint refs from the
existing progress scan payload and writes a compact checkpoint-list JSON during
the progress watcher commit. A fuller change would be a separate monitor
Function, but that should be implemented after one manual poll/eval cycle
confirms the exact checkpoint paths for the current spawned run.

No pytest was run for this critique.
