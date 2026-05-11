# LightZero Pong Bug Hunt Refresh - 2026-05-09

Scope: broad refresh for why Atari Pong checkpoints still show no learning
signal. I read the current handoff/state docs, setup audit, red-team note,
collapse investigation, exact reproduction wrapper, eval wrapper, and nearby
bug-hunt notes. No pytest. No code changes.

## Current Known Facts

- Official Atari Pong mechanics work on Modal: ALE Pong starts, LightZero
  trains, native checkpoints are written, periodic checkpoints strict-load, and
  eval can call `MuZeroPolicy.eval_mode.forward`.
- The latest selected periodic checkpoint curve is still bad. Checkpoints
  `0`, `100`, `500`, `900`, and `932` each chose one action for all `512`
  eval steps, returned `-13`, saw no positive rewards, matched manual and
  stock evaluator action prefixes, and used no fallback.
- `ckpt_best` is not trusted. The state diff says it looks reset-like:
  `last_iter=0`, `last_step=0`, empty optimizer state, norm counters `0`, and
  running means `0`. `iteration_932` has trained counters and optimizer state.
- The exact reproduction wrapper is now much cleaner than older train smokes.
  It imports installed `LightZero==0.2.0`, uses
  `zoo.atari.config.atari_muzero_config`, calls `lzero.entry.train_muzero`,
  mutates only `exp_name` in exact mode, and blocks CPU training.
- The faithful-short exact wrapper is not exact full upstream. It keeps the
  installed config but passes a shorter `max_env_step` to `train_muzero`.
- Older tiny/train wrappers intentionally changed many semantic knobs:
  env-step budget, train-iter cap, collector/evaluator counts, sims, batch,
  `update_per_collect`, episode caps, segment length, eval frequency, and
  checkpoint cadence.
- The installed package target is not current GitHub upstream. Installed
  `LightZero==0.2.0` reports `max_env_step=200000` for the non-segment 64x64
  config. Current GitHub prose/config references use about `500000`. The older
  pretrained checkpoint surface is a separate 96x96/downsample surface.
- Action mapping is no longer a leading suspect. ALE Pong action `5` is
  `LEFTFIRE`, and the eval wrapper now logs runtime action meanings.
- Observation shape is probably okay for the current 64x64 path. Raw env reset
  is one grayscale frame, and the manual eval stack becomes `[4,64,64]`.
  Periodic checkpoints matched the stock evaluator action prefix.
- The eval cap explains repeated `-6` or `-13` numbers. It does not explain
  why checkpoints choose one action everywhere.
- Stock `update_per_collect=None` restored replay-ratio behavior, but it caused
  a huge learner/checkpoint burst from the 8192-step run. That needs accounting,
  not guesswork.
- Custom dummy Pong is a separate lane. It has a similar action-collapse smell,
  but its likely blockers are root-visit target quality, sparse reward, and
  custom wrapper choices. Do not merge that with official Atari evidence.

## Likely Bugs Or Concrete Setup Risks

### 1. `ckpt_best` save/selection is broken or misleading

This is the sharpest concrete bug candidate. The file exists, but the state
metadata looks like initialization while periodic checkpoints look trained.
Manual eval once made `ckpt_best` look better, but stock eval did not agree.

Falsifying check: diff `ckpt_best` against `iteration_932` and the trainer log:
file size, top-level keys, model tensor norms, optimizer state, `last_iter`,
`last_step`, running norm stats, source iteration, and best-score metadata.
Then eval both checkpoints with the same manual and stock evaluator settings.

Pass condition: `ckpt_best` points to a real trained source iteration and
reproduces the trainer's best score under the same evaluator.

Fail condition: `ckpt_best` remains reset-like or cannot reproduce the best
selection. Then stop using it as quality evidence.

### 2. The run is still far below Pong scale, even when it is "near-upstream"

8192 env steps is tiny beside installed stock `200000`. Earlier 4096/sim10 was
much smaller and more patched. A deterministic bad action after a few thousand
steps can be early noisy MuZero, not a deep Atari finding.

Falsifying check: one accounted budget ladder after the current faithful-short
run completes. Keep one chosen source surface, one eval protocol, and compare
checkpoint curves at larger budgets with replay/update accounting.

Pass condition: action dependence, root margins, value calibration, or rewards
improve with budget.

Fail condition: larger, accounted rungs still produce one-action policies with
no positive rewards and healthy replay/update counts.

### 3. Replay/update/checkpoint accounting is opaque

Restoring stock `update_per_collect=None` produced about 934 checkpoints and a
large artifact burst from only 8192 env steps. That may be correct LightZero
replay-ratio behavior, or it may be a bad small-replay over-update setup. We
cannot tell from returns alone.

Falsifying check: for the live faithful-short run and the next bounded rung,
record env steps, replay transitions or segments, learner updates, target model
updates, skipped updates, checkpoint count, checkpoint cadence, and total
artifact bytes.

Pass condition: the train has enough diverse replay before many updates, and
checkpoint retention is intentional.

Fail condition: hundreds of updates recycle tiny early replay or fill the
Volume with near-duplicate checkpoints.

### 4. Eval defaults can still hide problems if strict mode is not enforced

The eval wrapper default is `allow_model_fallback=True`. Serious recent evals
used no fallback, but the default remains a footgun. A fallback eval would test
the raw model head, not the intended MuZero eval path.

Falsifying check: every checkpoint-curve manifest must say
`model_fallback_used=false`, strict load true, `stock_evaluator.path` is
`lzero.worker.MuZeroEvaluator` when stock parity is requested, and manual/stock
prefix match status is recorded.

Pass condition: all promoted quality rows are strict no-fallback rows.

Fail condition: any quality table includes fallback rows or generic
`ding.worker.InteractionSerialEvaluator` rows.

### 5. Source/config target soup remains dangerous

There are three different authority surfaces in circulation:
installed 0.2.0 non-segment 64x64, current GitHub config, and older pretrained
96x96/downsample. Mixing prose from one with checkpoints/config from another
can make a false "no learning" or false "pretrained broken" claim.

Falsifying check: each train/eval artifact gets a source-lock block:
LightZero version/source, DI-engine version, config module, config hash or
surface, observation shape, downsample flag, action space, and checkpoint key
surface.

Pass condition: one run lineage uses one source surface end to end.

Fail condition: train, eval, and checkpoint refs disagree on source/version,
shape, or model keys.

## Lower Suspects

- Action id mapping: low. Runtime action meanings and stock/manual parity make
  this unlikely.
- GPU/CPU mistake: low for promoted train rows. The exact wrapper blocks CPU
  train, and train summaries include CUDA probes. Eval on CPU is okay if the
  checkpoint strict-loads.
- Observation stack: medium-low. Manual stacking is a possible measurement gap,
  but periodic checkpoint stock/manual parity lowers this. Keep hashes/logs.
- Eval temperature: low. LightZero eval mode is deterministic by design.
- Seed/reset narrowness: medium. One seed and one evaluator episode can hide
  variance. It does not explain identical one-action collapse by itself.
- Episode caps: high for interpreting the numeric return, low for explaining
  action collapse. Caps make `-6` and `-13` smoke scores, not full Pong scores.
- Custom dummy Pong objective: separate. It may have target/reward bugs, but it
  should not be used to explain official Atari until reports stay separated.

## Falsifying Checks

1. `ckpt_best` forensic diff:
   compare `ckpt_best.pth.tar` with `iteration_932.pth.tar` on metadata,
   optimizer state, norm stats, model tensor norms, and source iteration.

2. Manual-vs-stock eval parity on both `iteration_932` and `ckpt_best`:
   same seed, same cap, no fallback, first raw observation hash, first stacked
   observation hash, first 64 actions, rewards, root visits, values, and logits.

3. Source-lock manifest:
   emit package versions, config module, config surface, observation shape,
   action space, downsample, ROM/action meanings, and checkpoint state surface.

4. Replay/update accounting:
   env steps, replay size, sampled segments, learner updates, target updates,
   skipped updates, checkpoint ids retained, and artifact bytes.

5. Eval-cap ladder:
   same checkpoint at 256, 512, and 1024 steps. This separates "bad quickly"
   from full-episode score claims.

6. Small reset-seed sweep:
   one periodic checkpoint plus `ckpt_best`, at three reset seeds, with action
   histograms and first observation hashes.

7. Runtime Atari canary:
   wrapper chain, action meanings, reward clipping, episode-life setting,
   frame-stack behavior, and effective max-step cap from the actual Modal image.

## Next Actions

1. Wait for the live faithful-short Modal run summary before making any new
   quality claim. Classify it as faithful-short, not exact full reproduction.

2. Run or inspect the `ckpt_best` forensic diff first. If it is reset-like,
   delete it from learning summaries and use periodic checkpoints only.

3. Run eval-only parity for `iteration_932` and `ckpt_best` with strict
   no-fallback, stock evaluator requested, and observation/action/root logs.

4. Add or require an accounting block in the next train summary before any
   larger run: replay size, learner updates, target updates, checkpoint count,
   retained checkpoint ids, and artifact bytes.

5. Only after those checks, spend on the next bounded scale rung. Use one
   source surface, recorded eval-wave rules, checkpoint retention, and a curve that
   reports action entropy, positive rewards, root margins, and capped returns.

## Short Verdict

The top current suspect is not action mapping or a CPU/GPU typo. It is a
cluster: a suspicious `ckpt_best`, too-small/off-recipe training, and weak
replay/update accounting. The exact wrapper is finally close enough to ask a
cleaner question, but the answer is not in yet.
