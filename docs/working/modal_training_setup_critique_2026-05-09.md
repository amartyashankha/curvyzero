# Modal Training Setup Critique - 2026-05-09

Scope: blunt repo read of the Pong training lane, Modal wrappers, run
management, checkpoint durability, evals, and scaling choices. This is working
memory for the training coach, not a final design.

## Short Answer

The setup is directionally right for this stage. Pong is the right current toy
because it uses visual raster observations, fixed baselines, checkpoints, and a
scoreboard. Modal is being used in the right shape where it exists: whole jobs
write files to a Volume and return compact summaries.

The setup is not mature yet, but Pong now has CPU Modal train and eval paths.
The train wrapper writes replay, checkpoints, manifests, and a latest pointer
to `curvyzero-runs`. The scoreboard can run remotely, read checkpoints from
`curvyzero-runs`, and write eval artifacts back to the Volume.

Do not scale GPU or multi-node training yet. The current bottleneck is not
compute. The current bottleneck is learning. Wins against `track_ball` are the
hard gate, but 0/64 wins alone is not enough to diagnose learner changes. The
current shaped signal must include survival/loss delay: episode length,
truncation rate, and shaped return proxy against `track_ball`.

## LightZero-First Addendum

This note predates the accepted LightZero-first next step, so read the
recommendation this way: the Modal pattern is still right, but the next real
MuZero attempt should be a LightZero custom-env dummy Pong job, not another
NumPy Pong learner.

Use the existing Modal shape:

- one coarse Modal Function owns the whole LightZero train smoke;
- no Modal Function calls inside environment steps, MCTS/search, replay, or
  LightZero trainer internals;
- mount the existing `curvyzero-runs` Volume at `/runs`;
- reuse `src/curvyzero/infra/modal/run_management.py` for ids, manifests,
  refs, JSON, hashes, and commits;
- keep retries off until resume is truly idempotent.

Use a new `TASK_ID = "lightzero-dummy-pong"` so LightZero artifacts do not get
mixed with the older `dummy-pong` NumPy/CEM/imitation runs. The Volume root
should be:

```text
/runs/training/lightzero-dummy-pong/<run_id>/
  run.json
  latest_attempt.json
  attempts/<attempt_id>/
    attempt.json
    config.json
    command.json
    stdout_tail.txt
    stderr_tail.txt
    train/
      summary.json
      episodes.jsonl
      lightzero_artifacts_manifest.json
      lightzero_training_signals.json
  checkpoints/
    lightzero/
      iteration_*.pth.tar
      ckpt_best.pth.tar
      manifest.json
    latest.json
```

Do not use the existing `checkpoint_file_ref()` helper as-is for LightZero
payloads, because that helper names `checkpoint.npz`. Keep the pointer schema
and `latest.json`, but write LightZero checkpoint payloads under
`checkpoints/lightzero/*.pth.tar` and point at those refs.

The first implementation should write LightZero's native `exp_name` tree under
`/tmp/curvyzero-lightzero-dummy-pong/<run_id>/<attempt_id>/`, scan it after the
trainer returns, then mirror the useful files and manifests into `/runs`.
Directly training into the Volume is unnecessary for the smoke and risks many
small framework-owned files on the shared filesystem.

Minimum Modal image/function shape:

```python
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")
runs_volume = modal.Volume.from_name("curvyzero-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("LightZero==0.2.0", "numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)

@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=20 * 60)
def train_lightzero_dummy_pong_tiny(...):
    ...
```

The function must call LightZero's real `lzero.entry.train_muzero` on
`DummyPongLightZeroEnv`, with `to_play=-1`, one ego action, scripted opponent,
honest score reward, and sidecar telemetry for survival/loss-delay. A dry
config smoke or stock CartPole/Atari run is not evidence for this task.

Exact recommended train command shape after implementation:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0 \
  --opponent-policy random_uniform \
  --max-env-step 64 \
  --max-train-iter 2 \
  --num-simulations 2 \
  --batch-size 8 \
  --update-per-collect 1 \
  --n-evaluator-episode 1 \
  --run-id lightzero-dummy-pong-smoke-20260509-001
```

Run the config/import smoke first if the adapter is new. The train smoke is the
first command that should be called real dummy Pong MuZero progress.

## 1. Stage Pattern

Correct:

- Keep `src/curvyzero/` algorithm-neutral.
- Keep the first Pong learner simple and local.
- Use raster observations now, because visual input is the point of this lane.
- Evaluate learned policies against `random_uniform` and `track_ball`.
- Keep angle/contact probes as debug tools, not progress claims.
- Save checkpoints and run a boring scoreboard before talking about MuZero.

Not correct yet:

- This is not a MuZero training stack yet. It is a tiny NumPy supervised policy,
  a tiny value-target smoke, and evaluator plumbing.
- There is no actor/search/trainer loop for Pong.
- A local Pong checkpoint selection record now exists. The first imitation
  heldout check was mixed, so selection still should not be treated as a quality
  claim.
- There is not yet a standard heldout gate for every new Pong candidate.
- There is not yet a standard shaped-signal gate for zero-win candidates:
  episode length, truncation rate, and shaped return/loss-delay proxy against
  `track_ball`.

Coach read: the repo is doing the right small steps. The danger is naming the
current learner like it is more mature than it is.

## 2. Modal Pattern

What is good:

- `src/curvyzero/infra/modal/dummy_survival_train_attempt.py` and
  `src/curvyzero/infra/modal/dummy_pong_train_attempt.py` show the right
  coarse-job pattern.
- Each runs one whole training attempt in one Modal Function.
- Each mounts the `curvyzero-runs` Volume at `/runs`.
- Each writes run/attempt manifests, train outputs, checkpoints or checkpoint
  pointers, and compact return refs.
- Each commits the Volume after writing the attempt.
- They keep resume out of scope, which is honest.

What is missing:

- Pong has a CPU Modal scoreboard wrapper.
- Pong has a CPU Modal train wrapper that writes replay, checkpoints, manifests,
  summaries, and a latest checkpoint pointer to Volume.
- Pong eval/scoreboard artifacts can now be written to the Volume by the Modal
  wrapper.
- Dummy line duel and older one-off wrappers do not use the run/attempt Volume
  layout.
- Current Modal training does not run on GPUs.
- Current Pong training does not need GPUs because it is a NumPy CPU learner.
- Current Modal GPU coverage is JAX/Mctx dependency smokes plus the synthetic
  benchmark lane, not Pong training.

Official Modal docs still support this plan:

- Volumes are a good fit for write-once/read-many ML files such as checkpoints:
  https://modal.com/docs/guide/volumes
- Volume changes need commit/reload discipline, and concurrent writes to the
  same file are last-writer-wins:
  https://modal.com/docs/reference/modal.Volume
- GPUs are selected on a Function with `gpu=...`; one container can request
  multiple GPUs with strings like `H100:8`:
  https://modal.com/docs/guide/gpu
- Multi-node clusters exist, but they are beta and require cluster-specific
  code and scheduling:
  https://modal.com/docs/guide/multi-node-training

Coach read: the Modal design is right. Pong train/eval is now remote CPU
Modal. The next Modal step is not GPU training; it is one CPU Modal Pong lane
with explicit survival/loss-delay telemetry.

## 3. Checkpoints And Evals

What is now in place:

- Pong imitation training now supports `--checkpoint-every-epochs`.
- Periodic policy snapshots are written under
  `output_dir/checkpoints/epoch-000NNN/checkpoint.npz`.
- The root `checkpoint.npz` still stores the final policy.
- `summary.json` records `checkpoints.count`, `checkpoints.refs`, and
  `checkpoints.latest`.
- The Pong checkpoint scoreboard accepts labeled checkpoints such as
  `latest=...`, `previous=...`, or `epoch_003=...`.
- The scoreboard includes baseline sanity rows, learned-vs-baseline rows, and
  learned-vs-learned rows when multiple checkpoints are passed.
- A local Pong `selection_record.json` helper exists and records that heldout
  confirmation is required before a quality claim.
- The selection helper now treats the `track_ball` gate more honestly: after
  wins against `track_ball`, it prefers fewer losses to `track_ball`, then more
  truncations against `track_ball`, then random-opponent wins. It also records
  a pressure ranking.

What is still missing:

- No automatic `latest`, `previous`, or `selected_best` policy naming from a
  run manifest.
- No Pong `best.json` pointer.
- No standard heldout confirmation command/gate for each new Pong selection.
- Scoreboard split metadata is light: it records split id/role, but not a full
  seed-list hash yet.
- The current periodic checkpoint smoke is too tiny to judge quality.
- Scoreboard summaries need the shaped-learning readout for zero-win learners:
  mean episode length, truncation rate, and shaped return proxy against
  `track_ball`.

Fresh smoke:

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 2 \
  --seed 0 \
  --split-id dummy_pong_monitor_v0 \
  --split-role monitor \
  --checkpoint epoch_1=artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09/checkpoints/epoch-000001/checkpoint.npz \
  --checkpoint epoch_3=artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09/checkpoints/epoch-000003/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-periodic-checkpoint-scoreboard-smoke-2026-05-09
```

Result: the scoreboard can consume periodic checkpoints. It did not show useful
learning. Epoch 3 beat random 2/4 versus epoch 1's 1/4, but both checkpoints
still won 0/4 against `track_ball`, and epoch 1 versus epoch 3 tied. Treat this
as plumbing only.

Coach read: checkpoint/eval plumbing is good enough. Do not compare generations
as the main path. The next issue is policy improvement, measured by wins as the
hard gate and survival/loss delay as the zero-win diagnostic signal.

## 4. Useful Parallelism Now

Parallelize these now:

- Pong checkpoint scoreboards across checkpoints and seed splits.
- Pong replay generation shards.
- Small CPU sweeps over seeds, learning rate, epochs, and replay source.
- Artifact inspection and summary extraction.
- Modal CPU train/eval jobs for bounded learner comparisons.

Keep these inside one process/container for now:

- Environment stepping.
- Action choice.
- Replay sampling.
- Model update.
- Future MCTS search.

Coach read: fan out independent jobs. Do not split one tiny training loop across
remote calls.

## 5. What Not To Scale Yet

Do not scale these yet:

- Modal GPU Pong training.
- Multi-GPU training.
- Multi-node training.
- Modal retries for training attempts.
- Full resume.
- League/Elo/population machinery.
- Large replay storage designs beyond chunked files plus manifests.
- Mctx/JAX tied to real Pong before the synthetic benchmark is explicit.

Why: none of those solves the current failure. The current failure is that the
learned Pong policy does not beat `track_ball`, and we do not yet have enough
survival/loss-delay telemetry to tell whether learner changes are at least
moving toward that gate.

## Historical Next Three Moves

This list is superseded by the current Pong critique. Keep it as context only.

1. Decide whether to repair the crude Pong self-play trainer or switch to a
   simpler known baseline/curriculum, using survival/loss-delay metrics as the
   shaped signal while wins are zero.
2. Use the CPU Modal Pong scoreboard wrapper for remote eval when checkpoints
   are already in `curvyzero-runs`.
3. Modal is the serious-run target, but do not add GPU or large training runs
   until the learner path has a real reason to run.

## Files Inspected

- `src/curvyzero/training/dummy_pong_imitation_train.py`
- `scripts/train_dummy_pong_imitation.py`
- `src/curvyzero/training/dummy_pong_eval.py`
- `scripts/run_dummy_pong_checkpoint_scoreboard.py`
- `src/curvyzero/infra/modal/run_management.py`
- `src/curvyzero/infra/modal/dummy_survival_train_attempt.py`
- `src/curvyzero/infra/modal/volume_dummy_survival.py`
- `src/curvyzero/infra/modal/smoke.py`
- `docs/design/modal_training_run_management.md`
- `docs/design/training_eval_protocol.md`
- `docs/runbooks/training_smokes.md`
- `docs/working/training_coach_packet.md`
- `docs/working/pong_training_plan.md`
