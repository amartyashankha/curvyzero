# LightZero Pong No-Signal Bug Hunt - 2026-05-09

Scope: official LightZero Atari Pong first, with custom dummy Pong mentioned
only where the same no-learning failure mode matters. This is a falsification
plan, not another reproduction log. No pytest, no training, and no eval jobs
were run for this note.

## Observed

The current official Atari Pong evidence is mechanical success and learning
failure.

- The installed `LightZero==0.2.0` 64x64 Atari path can train on Modal, write
  native checkpoints, strict-load periodic checkpoints, and run both manual and
  stock-evaluator eval paths.
- The latest near-upstream installed-package rung reached `8192` env steps with
  `num_simulations=25`, `batch_size=64`, `collector_env_num=4`,
  `game_segment_length=128`, and stock `update_per_collect=None`.
- Periodic checkpoints `iteration_0`, `100`, `500`, `900`, and `932` all
  collapsed to one action for the 256-step eval window and returned `-6`.
- `ckpt_best` is not credible learning evidence yet: manual eval saw diverse
  actions and return `0`, but stock evaluator still returned `-6` and
  manual/stock first actions did not match.
- There is no exact full reproduction yet. Current GitHub upstream,
  installed PyPI `LightZero==0.2.0`, and the older pretrained 96x96 surface are
  different targets. The current runs are still patched controls.

## Hypotheses And Falsifiers

### 1. Hybrid config fidelity is the main bug

The runs keep the official path, but still mix or shrink stock settings:
installed PyPI versus current GitHub, non-segment trainer versus segment
intuition, reduced env steps, reduced sims, reduced batch, reduced collectors,
short segments, one evaluator, short caps, and frequent checkpointing.

Would falsify it: choose one authority surface and show the same collapse under
a clearly labeled, nearly faithful profile whose only remaining deviations are
explicitly listed and small enough not to change learning semantics.

Cheapest next experiment: dry-only surface diff for `pypi-lightzero-0.2.0`
versus `github-main`, then one tiny "semantic-patch guard" launch that exits
before training if any non-output knob differs from the chosen source.

### 2. The learner is simply far below Pong scale

`8192` env steps is still tiny compared with installed-package stock
`200000` and current GitHub `500000`. The action collapse may be early MuZero
noise, not a mysterious plumbing failure.

Would falsify it: a budget ladder with the same chosen profile shows no
monotonic improvement in score, action dependence, root margins, or value
calibration as env steps increase by a meaningful factor.

Cheapest next experiment: run one bounded budget ladder only after accounting
is fixed: for example `8192 -> 32768` on the same source surface, evaluating
the same checkpoint indices and reporting action histograms, rewards, root
visit margins, replay size, and update count.

### 3. Replay/update cadence is overfitting a tiny replay pool

Restoring stock `update_per_collect=None` caused about `933` learner iterations
and roughly `934` checkpoint files from only `8192` env steps. That could mean
many updates against a small, early, correlated replay pool.

Would falsify it: logs show a healthy ratio of env steps, replay size,
sampled game segments, learner updates, target updates, and checkpoint cadence;
or a fixed-update control still collapses the same way with better accounting.

Cheapest next experiment: add accounting to one small train: env steps
collected, replay transitions/segments, learner updates, skipped updates,
target-network updates, checkpoint ids retained, and total artifact size.

### 4. Eval protocol mismatch is hiding or inventing signal

Periodic checkpoints match manual and stock first actions, but `ckpt_best`
does not. Manual eval uses a custom loop and frame stack; stock evaluator uses
LightZero's evaluator path. The disagreement matters most for `ckpt_best`.

Would falsify it: for the same checkpoint, reset seed, cap, frame-stack state,
and logged first observations, manual and stock paths match actions, rewards,
root visits, values, and logits on periodic checkpoints and `ckpt_best`.

Cheapest next experiment: eval-only parity on `iteration_932` and `ckpt_best`,
with one seed, one cap, no fallback, logged first raw observation hash, stacked
observation hash, first 64 actions, rewards, root visits, values, and logits.

### 5. `ckpt_best` is selected or saved under different semantics

`ckpt_best` is smaller/suspicious in prior notes, has different behavior from
periodic checkpoints, and may be selected by trainer-side evaluator state that
does not match the post-train eval harness.

Would falsify it: `ckpt_best` has the same state keys, model tensor shapes,
optimizer/model fields, source iteration metadata, and load behavior as a
periodic checkpoint; and it reproduces the trainer-side selection score under
the same evaluator config.

Cheapest next experiment: file/state diff `ckpt_best` versus `iteration_932`:
size, keys, tensor shapes, tensor norms, saved iteration, evaluator score
metadata, optimizer fields, model fields, and strict load path.

### 6. The 256-step eval cap explains too much of the flat score

The periodic rows all return `-6` because the policy loses six points inside
the short cap. That is useful for comparing collapse, but it is not stock Pong
scoring and cannot distinguish "bad quickly" from "bad over a full game."

Would falsify it: longer caps keep the same ordering and still show no
credible action/state dependence, so the cap only compresses the number and
does not change the conclusion.

Cheapest next experiment: eval `iteration_0`, one mid checkpoint,
`iteration_932`, and `ckpt_best` at 256 and 1024 steps, with the same seed and
same action/reward trace summary.

### 7. RNG/reset path is making the scorecard too narrow

One evaluator env, one evaluator episode, fixed seed behavior, and short caps
can accidentally test a tiny slice of Pong. Earlier dummy Pong work also found
seed handling could mislead sidecars.

Would falsify it: a small seed sweep shows the same collapse pattern across
reset seeds and env instances, with no hidden first-life/reset artifact.

Cheapest next experiment: eval-only seed sweep over three reset seeds for one
periodic checkpoint and `ckpt_best`, logging first observation hashes and
action/reward traces.

### 8. Wrapper patches are changing Atari semantics

The wrapper patches episode caps, eval frequency, checkpoint cadence, env
counts, segment length, update settings, and sometimes source/version targets.
Even when each patch is reasonable for a smoke, the combination may no longer
be the stock training problem.

Would falsify it: a patch manifest proves only intended knobs changed, and a
no-training env canary shows action meanings, preprocessing, reward clipping,
episode-life behavior, frame stack, and max-step behavior match the selected
source surface.

Cheapest next experiment: dry/runtime canary that prints original config,
patched config, wrapper chain, action meanings, reset observation shape,
policy observation shape, reward clipping flag, episode-life flag, and
effective episode cap.

### 9. Model/checkpoint shape drift is still present somewhere

The current 64x64 periodic checkpoints strict-load, but the pretrained 96x96
lane fails strict load with downsample keys. This proves there are multiple
model surfaces in circulation.

Would falsify it: every training, eval, and checkpoint artifact in a lane
records the same model surface: source id, observation shape, downsample flag,
action space, state-dict key set, and config module.

Cheapest next experiment: add a model-surface block to eval summaries and run
it on one current periodic checkpoint plus the blocked pretrained checkpoint,
expecting current to pass and pretrained to fail with an explicit label.

### 10. Library version drift changes the target recipe

The docs already found at least one important drift: current GitHub upstream
uses `500000` env steps while installed `LightZero==0.2.0` reports `200000`.
DI-engine, gym/gymnasium, ale-py, AutoROM, and torch versions are also part of
the runtime contract.

Would falsify it: a pinned source identity and package manifest reproduce the
same config surface and checkpoint load behavior every time; no run blends
GitHub prose, PyPI imports, and old pretrained checkpoint cards.

Cheapest next experiment: write a source-lock manifest before any train:
LightZero source/version/commit, DI-engine version, Atari config path, config
hash, ALE action meanings, ROM install note, and full package versions.

### 11. Custom dummy Pong has a separate objective/target problem

Dummy Pong failures should not be used as proof that official Atari is broken.
Still, they rhyme: sparse reward, weak targets, action collapse, low sim
counts, and eval reconstruction footguns can all produce no signal.

Would falsify it: the dummy lane learns under a fixed horizon with explicit
reward/target telemetry and independent scorecards, while official Atari still
collapses under faithful stock settings; or vice versa.

Cheapest next experiment: keep dummy and official reports separate, but require
both to emit the same minimal learning contract: horizon, reward used for
training, eval reward, replay/update counts, checkpoint id, action histogram,
root visits, and reset seed.

## Next Three Experiments

1. **Checkpoint/eval parity and `ckpt_best` file diff.** Best value per time.
   No training. It directly tests whether the only apparently good checkpoint
   is real, mis-saved, or being evaluated through mismatched protocols.

2. **Dry source-surface and wrapper canary.** Cheap and durable. Pick
   `pypi-lightzero-0.2.0` or `github-main`, print the exact unpatched and
   patched config surfaces, and fail if semantic knobs drift unexpectedly.

3. **One accounted bounded train rung.** Do this only after the first two.
   Keep the profile explicit, retain few checkpoints, and log env steps,
   replay size, learner updates, target updates, checkpoint ids, root margins,
   and action/reward traces. The goal is not a big exact reproduction; it is to
   falsify "tiny replay plus opaque update cadence caused arbitrary collapse."

## Non-Goals For The Next Loop

- Do not treat `ckpt_best` manual return `0` as learning until parity explains
  the stock/manual disagreement.
- Do not compare trainer `-21` and capped eval `-6` as the same metric.
- Do not reopen action-mapping as a primary theory unless the runtime canary
  contradicts ALE action meanings.
- Do not run a larger expensive train without checkpoint retention and
  update/replay accounting.
- Do not merge official Atari Pong, custom dummy Pong, and CurvyTron claims.
