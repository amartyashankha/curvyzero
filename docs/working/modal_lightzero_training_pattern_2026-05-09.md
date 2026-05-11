# Modal LightZero Training Pattern Critique - 2026-05-09

Scope: local read of the current Modal examples/docs and LightZero dummy Pong
wrappers. `/Users/shankha/curvy/modal-projects` was not present in this
workspace, and the local repo docs/examples were sufficient, so no web research
was needed.

## Plain Answer

The current pattern is mostly right for today.

We are using Modal in the right coarse shape: one Modal Function owns one whole
training or eval job, writes durable outputs to `curvyzero-runs`, commits the
Volume, and returns compact refs. We are not putting Modal in the hot loop for
environment steps, replay, MCTS, or gradient updates. That is the important
line to keep.

The image/dependency story is simple enough for the current LightZero Pong
stage. The wrappers use `debian_slim`, Python 3.11, pinned `LightZero==0.2.0`,
minimal extra packages, `PYTHONPATH=/repo/src`, and copied local `src/`.
That is a good smoke/train image. Do not make a shared mega-image yet.

Volumes are mostly used correctly for checkpoints and artifacts. The train
wrapper writes LightZero's framework-owned experiment tree under `/tmp`, then
scans and mirrors useful files into `/runs/training/lightzero-dummy-pong/...`.
That is better than letting LightZero spray many small files directly into the
shared Volume. The wrappers commit after writing summaries/checkpoints/evals.
The main gap is that LightZero checkpoint pointers are not yet as polished as
the project-owned `.npz` checkpoint layout: there is a copied
`checkpoints/lightzero/manifest.json`, but no canonical
`checkpoints/latest.json` / `best.json` pointer for LightZero runs yet.

We are not actually using Modal GPUs for LightZero dummy Pong training today.
That is okay. The current custom-env train wrappers set `policy.cuda=false` and
do not request `gpu=...`. The existing GPU proof is in the Mctx/JAX/smoke lane,
not in the LightZero Pong train lane. For these tiny tabular MLP runs, CPU is
the right default until the eval boundary is honest and the job is actually
compute-bound.

The smallest stable pattern now is:

1. Run one CPU whole-job LightZero train attempt on dummy Pong.
2. Write LightZero native outputs to `/tmp`.
3. Mirror checkpoint `.pth.tar` files, config, logs, episode sidecar, and
   summary JSON into `curvyzero-runs`.
4. Run independent eval as a separate whole-job Modal Function using Volume
   checkpoint refs.
5. Treat policy-head greedy eval as a loader/control result, not MuZero/MCTS
   policy quality.
6. Promote the passing one-call MCTS loader smoke into a real episode/opponent
   scorecard before scaling training.

## Current Pattern

Current train path:

- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
- `src/curvyzero/training/lightzero_dummy_pong_env.py`

What it does well:

- Uses one Modal Function per whole train attempt.
- Calls LightZero's real `lzero.entry.train_muzero`.
- Keeps LightZero on the custom `DummyPongLightZero-v0` env, not stock Atari
  Pong or CartPole.
- Keeps the first surface small: `dummy_pong_lag1`, `tabular_ego`,
  observation shape `10`, action size `3`, MLP model, one collector, one
  evaluator, tiny simulation/batch caps.
- Writes run and attempt manifests with `TASK_ID = "lightzero-dummy-pong"`.
- Writes `summary.json`, `episodes.jsonl`, config, command, log tails, training
  signals, and a LightZero artifact manifest.
- Mirrors LightZero `.pth.tar` checkpoints into
  `/runs/training/lightzero-dummy-pong/<run_id>/checkpoints/lightzero/`.
- Commits the Volume at the end.

Current eval/probe path:

- `src/curvyzero/infra/modal/lightzero_dummy_pong_checkpoint_probe.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_loader_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_policy_head_scoreboard_attempt.py`
- `docs/working/lightzero_checkpoint_loader_probe_2026-05-09.md`
- `docs/working/lightzero_pong_scorecard_plan_2026-05-09.md`

What it proves:

- LightZero checkpoints can be found, hashed, inspected, and loaded into the
  expected MLP config.
- The direct policy-head greedy scoreboard can run outside the trainer and
  writes eval artifacts to the Volume.
- A 512/8 `iteration_8` MCTS/eval-mode loader smoke now passes strict full
  model load and one eval-mode forward call.

What it does not prove yet:

- It does not prove a useful learned policy.
- The policy-head scoreboards are still constant-up controls.
- The MCTS/eval-mode path has passed one call, not a full episode scorecard
  across `random_uniform`, `lagged_track_ball_1`, and `track_ball`.

## Gaps

The main gap is not Modal. The main gap is the LightZero evaluation boundary.
Trainer-side LightZero evaluator rewards and env-side sidecar rows are useful
debug evidence, but they are not enough to claim checkpoint quality. The next
trustworthy artifact is a full CurvyZero-owned MCTS/eval-mode scorecard that
loads a mirrored `.pth.tar` from the Volume and runs many episodes against the
fixed Pong ladder.

The second gap is checkpoint pointer discipline for LightZero. The mirror
manifest lists copied checkpoints, but later jobs would be simpler if each run
also wrote:

```text
/runs/training/lightzero-dummy-pong/<run_id>/checkpoints/latest.json
/runs/training/lightzero-dummy-pong/<run_id>/checkpoints/best.json
```

Those pointer files should include the checkpoint ref, hash, source attempt,
iteration/name, config ref, feature/action schema, and eval status. Payload
files should stay immutable.

The third gap is resume/retry. Current LightZero wrappers correctly leave real
resume out of scope. Do not enable Modal retries as if training is reentrant
until the checkpoint contains enough state and the wrapper knows how to resume
from a committed pointer. For now, failed attempts should be recorded as failed
attempts, not retried into the same mutable outputs.

The fourth gap is dependency duplication. Several wrappers repeat the same
LightZero image, Volume, mount, and constants. This is tolerable today. Factor
only after the MCTS scorecard path settles; premature cleanup would mostly move
instability around.

The fifth gap is cache policy. There is no separate cache Volume for LightZero
now, and that is fine because current jobs install a pinned package and run
small CPU tasks. Add `curvyzero-cache` only when there is a real repeated cache
cost: Torch model downloads, JAX compilation cache, Hugging Face artifacts, or
large framework build output. Do not put caches in `curvyzero-runs`.

## Recommended Small Pattern

Keep this as the stable LightZero Pong pattern:

```python
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("LightZero==0.2.0", "numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)

runs_volume = modal.Volume.from_name("curvyzero-runs", create_if_missing=True)

@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=20 * 60)
def train_lightzero_dummy_pong_attempt(...):
    # whole LightZero train attempt happens inside this function
    # framework output goes to /tmp
    # useful artifacts are mirrored to /runs
    # Volume is committed once the attempt summary and checkpoint mirror exist
    ...
```

Use this command family for train attempts:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --opponent-policy random_uniform \
  --max-env-step 512 \
  --max-train-iter 8 \
  --num-simulations 4 \
  --batch-size 16 \
  --update-per-collect 1 \
  --n-evaluator-episode 4 \
  --seed 1
```

Use a separate Modal eval command for checkpoint scorecards. Keep scorecard
outputs under `eval/lightzero-dummy-pong/...` or under the training run's
`eval/<eval_id>/...`, but always include the checkpoint input refs and hashes.

## Scaling Knobs Later

Use these knobs in this order:

1. Eval honesty: full MCTS/eval-mode scorecard before more training scale.
2. More seeds: independent whole-job train attempts with separate run ids.
3. More eval episodes: independent scorecard jobs, not bigger trainers.
4. Larger caps: `max_env_step`, `max_train_iter`, `n_evaluator_episode`,
   `num_simulations`, `batch_size`, and `update_per_collect`.
5. GPU: add `gpu=["L4", "T4"]` or `gpu="L40S"` and set LightZero `cuda=true`
   only after CPU jobs are too slow and the checkpoint/eval path is trusted.
6. Parallel sweeps: use Modal `map`/`starmap` only for independent seeds,
   checkpoint evals, and config variants.
7. Cache Volume: add `/cache` only when dependency or compile caches are
   measured as a real cost.
8. Retries: add `modal.Retries` and `single_use_containers=True` only after
   resume from `checkpoints/latest.json` is idempotent.
9. Multi-node: defer until one-container GPU training is the bottleneck and
   checkpoint/replay schemas are stable.

## Next-Step Checklist

- [ ] Turn the passing MCTS/eval-mode loader smoke into a full Modal scorecard
      across `random_uniform`, `lagged_track_ball_1`, and `track_ball`.
- [ ] Keep direct policy-head greedy scorecards labeled as non-MCTS controls.
- [ ] Add LightZero `checkpoints/latest.json` and later `best.json` pointer
      files that reference immutable `.pth.tar` payloads.
- [ ] Include action histograms, survival steps, truncation rate, score return,
      and shaped loss-delay return in every compact scorecard row.
- [ ] Keep LightZero dummy Pong on CPU until the full scorecard is trustworthy.
- [ ] Factor shared LightZero Modal image/Volume constants only after train and
      eval wrappers stop changing daily.
- [ ] Add a cache Volume only if repeated runs show a real cache bottleneck.
- [ ] Do not add Modal retries until LightZero training resume is explicit and
      tested through a committed latest checkpoint pointer.

## Local Sources Read

- `docs/research/modal_training_patterns.md`
- `docs/research/modal_example_patterns.md`
- `docs/design/modal_training_run_management.md`
- `docs/working/modal_training_setup_critique_2026-05-09.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-implementation-inspection.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-longer-run.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-policy-head-scoreboard.md`
- `docs/working/lightzero_checkpoint_loader_probe_2026-05-09.md`
- `docs/working/lightzero_pong_scorecard_plan_2026-05-09.md`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_checkpoint_probe.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_loader_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_policy_head_scoreboard_attempt.py`
- `src/curvyzero/infra/modal/run_management.py`
