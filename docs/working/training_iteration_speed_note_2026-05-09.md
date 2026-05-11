# Training Iteration Speed Note - 2026-05-09

Scope: critique of the current installed-package LightZero Atari Pong training
loop speed and eval parallelism. This is a docs-only note. No code was edited
and no pytest was run.

## Current Speed Read

The latest faithful-short run asked for `8192` env steps, collected `14791`,
and took about `1326s` on a Modal L4. That is roughly:

- `11.2` collected env steps per second.
- `55.8` training seconds per requested `512` eval steps, if comparing only
  wall time to the eval cap size.
- `0.70` requested env steps per collected env step, because the stock collector
  overshot the requested cap in a final batch.

This is slow enough that every next rung should answer a narrow question. The
`32768` rung is a bounded scale check, not a proof that learning has started.

## Likely Bottlenecks

The train wrapper in
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py` keeps the
installed LightZero Atari config intact except for `exp_name` and the optional
`train_muzero(max_env_step=...)` override. The stock surface includes:

- `collector_env_num=8`
- `n_episode=8`
- `evaluator_env_num=3`
- `n_evaluator_episode=3`
- `num_simulations=50`
- `batch_size=256`
- `replay_ratio=0.25`
- `game_segment_length=400`
- `eval_freq=2000`
- CUDA enabled

The main bottleneck is not Modal startup or JSON writing. It is the real
LightZero training loop:

- Atari env collection uses subprocess envs, so each collect round has process
  and emulator cost.
- MuZero action selection is MCTS-heavy. `num_simulations=50` matters for both
  train-time collection and eval-time policy calls.
- The trainer does learner work after collection based on replay ratio and
  collected transitions. More collected steps can mean many learner iterations.
- Eval can interrupt training at `eval_freq=2000` learner iterations, and stock
  eval uses multiple evaluator envs/episodes.
- The requested env-step cap is not a hard stop at exactly that number. A collect
  batch can push beyond it, as the `8192 -> 14791` overshoot showed.

The progress watcher scans the artifact tree every interval. That is useful and
should stay at `120s` for the next rung. It is not the speed problem unless the
checkpoint count explodes again.

## What Not To Change For A Faithful Rung

Do not speed up the `32768` training rung by changing these knobs:

- `num_simulations`
- `collector_env_num`
- `evaluator_env_num`
- `n_evaluator_episode`
- `batch_size`
- `replay_ratio`
- `update_per_collect`
- `game_segment_length`
- `eval_freq`
- checkpoint cadence

Changing those would make the rung a different experiment. That can be useful
later, but it would no longer answer "does the installed stock setup behave
better with 4x more faithful-short env steps?"

The one accepted training change for this lane is still only
`max_env_step_override`. That is why `32768` is clean and a full `200000` run is
premature.

## Faster Eval That Stays Honest

Live or post-train eval can be faster without lying if it keeps these rules:

- Use strict checkpoint load: `--no-allow-model-fallback`.
- Keep manual `max_eval_steps` and env `--max-episode-steps` equal.
- Use periodic `iteration_*.pth.tar` files first, not `ckpt_best`.
- Read the manifest table first.
- Treat stock/manual action mismatch as a diagnostic, not as a reason to discard
  the manual return automatically.
- For cheap GPU eval, use `--compute gpu-l4-t4`. This only changes Modal
  placement and `policy.cuda=true`; it does not change training.
- While training is running, poll the checkpoint directory in the Modal Volume
  and skip checkpoints whose eval artifact directory already exists.

Recommended first-pass settings after `32768` finishes:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --compute gpu-l4-t4 \
  --parallel \
  --eval-pass low \
  --eval-id faithful-short-32768-periodic-low-stockeval-s0 \
  --checkpoint-refs '<ITERATION_0_REF>,<MID_ITERATION_REF>,<FINAL_ITERATION_REF>' \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-relpath \
  --low-detail-max-eval-steps 512 \
  --max-episode-steps 512 \
  --low-detail-step-detail-limit 8 \
  --no-allow-model-fallback \
  --run-stock-evaluator
```

This is still honest because it runs the real ALE-backed Pong env, uses the
policy eval path, keeps fallback off, and gives each checkpoint the same
512-step cap. It is faster because it stores only a few detailed steps, runs
selected checkpoints in parallel, and can place the policy model on a cheap
L4/T4 worker.

For the live loop details, including the checkpoint polling command and the
duplicate-skip rule, use
`docs/working/lightzero_live_gpu_eval_loop_2026-05-09.md`.

If the low pass is flat, do not run high detail. High detail should only be for
checkpoints with one of these signals:

- less-negative return than earlier checkpoints
- any positive Pong reward
- broader action support
- lower dominant-action collapse
- clean strict/no-fallback behavior worth inspecting

For high detail, keep the checkpoint list small and use:

- `--eval-pass high`
- `--high-detail-max-eval-steps 512`
- `--max-episode-steps 512`
- `--high-detail-step-detail-limit 8`

Longer eval caps such as `1024` can be honest, but they answer a different
question. Use them only after a 512-step pass shows a reason.

## Where Parallelism Helps

Parallelism helps most after training, across checkpoints. The eval wrapper in
`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py` uses
`lightzero_pong_eval_smoke.map(...)` when `--parallel` is set or more than one
checkpoint is selected. Each checkpoint gets its own Modal function call and
writes its own JSON artifact. A separate lightweight function writes the
manifest.

That means wall time should move closer to the slowest single checkpoint eval
instead of the sum of all selected evals, subject to Modal scheduling and Volume
I/O.

Good uses:

- `iteration_0`, one middle checkpoint, final checkpoint
- a small periodic curve if the `32768` run unexpectedly produces more
  `iteration_*.pth.tar` files
- high-detail reruns over only the one or two promising checkpoints

The best eval parallelism unit is checkpoint ref, not step. Splitting one
episode's 512 steps across workers would change the stateful episode and would
not be a valid Pong rollout.

## Where Parallelism Does Not Help Much

Parallel eval does not make the training loop itself faster. The training
wrapper calls one remote `train_muzero(...)` job. Inside that job, stock
LightZero already uses subprocess collection and its own learner/evaluator
flow. Adding more Modal workers around the same single training run would not
split the replay buffer, learner state, or env-step counter safely.

Parallelism also does not fix train-time eval cost unless the training code is
changed. The exact wrapper deliberately does not change stock evaluator counts
or eval cadence.

It also does not fix MCTS cost inside one policy action. A 512-step eval with
`num_simulations=50` still pays for 512 policy decisions per checkpoint.

## Before The 32768 Rung Finishes

Prepare these before the run ends so the readout is quick:

- Keep watching `train/progress/latest.json` for CUDA, artifact root, checkpoint
  count, and checkpoint bytes.
- Decide the eval checkpoint set as soon as filenames are visible:
  `iteration_0`, final, and one middle checkpoint only if it exists.
- Do not include `ckpt_best` in the first quality curve. It is useful for
  debugging but confuses the first periodic read.
- Keep a copy-ready eval command with `--max-episode-steps 512` matching
  `--low-detail-max-eval-steps 512`.
- Plan to fetch and summarize the manifest, not inspect every per-checkpoint
  JSON first.
- Keep the stop rule: if checkpoint count exceeds `10` or checkpoint bytes move
  toward `2 GiB`, write a manifest before pruning or rerunning anything.

## Bottom Line

The slow part is faithful stock-ish MuZero training, not the docs, summary JSON,
or Modal wrapper overhead. Do not tune training knobs for the `32768` rung if
the goal is a clean scale comparison.

Eval is different: checkpoint eval is safe to fan out. The honest fast path is
strict/no-fallback parallel eval over a tiny periodic checkpoint set, 512 manual
steps, matching `max_episode_steps=512`, small step detail, and manifest-first
readout.
