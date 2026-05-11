# LightZero Signal Debug Audit - 2026-05-10

## Replication Audit Update

Claim: the official/control runs are close to the installed `LightZero==0.2.0`
Atari Pong training surface on env/model/search/replay shape: `PongNoFrameskip-v4`,
`atari_lightzero`, conv `[4, 64, 64]`, six actions, `collector_env_num=8`,
`evaluator_env_num=3`, `num_simulations=50`, `batch_size=256`,
`game_segment_length=400`, `update_per_collect=None`, and
`replay_ratio=0.25`. The main intentional training deviation is the
faithful-short `train_muzero.max_env_step` override (`32768` or `65536`
instead of the installed package's `200000`) plus checkpoint cadence.

Non-claim: this does not prove the setup is an exact current-GitHub upstream
replication, a solved-Pong result, or stable return learning. Current local
GitHub checkout `/tmp/lightzero-src` still shows the same broad Atari Pong
recipe but `max_env_step=int(5e5)`, so "stock LightZero" must stay labeled as
either installed-package stock (`200000`) or current-GitHub stock (`500000`).

Plain diagnosis: the survival bumps are real weak-policy telemetry, not stable
Pong learning. They mean some checkpoints delay losing points longer under the
2048-step microscope, but they still usually lose all 21 points and almost
never win points. That pattern matches undertrained sparse-reward Pong and
action/search instability better than a simple action-map, reward-clipping,
frame-stack, sticky-action, or dummy-env mixup.

Concrete evidence:

- The trainer wrapper imports `zoo.atari.config.atari_muzero_config`, reads the
  package `max_env_step`, deep-copies `main_config` / `create_config`, and calls
  `lzero.entry.train_muzero`; the validator expects the installed package
  surface listed above.
- The local source checkout's Atari wrapper uses Gymnasium
  `PongNoFrameskip-v4`, `NoopResetWrapper`, `MaxAndSkipWrapper` with
  `frame_skip`, no sticky-action wrapper, grayscale/scale/warp, and optional
  reward clipping. Collector env cfg sets `episode_life=True` and
  `clip_rewards=True`; evaluator cfg sets `episode_life=False` and
  `clip_rewards=False`.
- The 2048-step eval manifests report strict checkpoint load and no model
  fallback. Action meanings from direct Gym are
  `NOOP/FIRE/RIGHT/LEFT/RIGHTFIRE/LEFTFIRE`; action-space size is `6`.
- Seed 0 detached `iteration_5000` reached `1001/2048` versus same-run
  baseline `761/2048`, but manual and stock returns both stayed `-21`,
  positive rewards stayed `0`, and later checkpoints fell back toward baseline.
- Seed 2 L4 `iteration_10000` reached `882/2048` versus `762/2048`, but stock
  return stayed `-21`; latest `iteration_16829` had only one manual positive
  reward and stock return still `-21`.
- Fresh stock-telemetry validation on seed 3 used strict no-fallback stock
  evaluator, `2048` manual/env caps, and `update_per_collect=None` via
  `--update-per-collect -1` for same-run `iteration_0` and `iteration_16000`.
  Manifest:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s3/attempts/train-faithful-short-installed-0.2.0-s3-65536-ckpt1000-spawn-relpath/eval/stocktelemetry-seed3-2048-iteration0-vs-16000/manifest_custom_steps2048_seed3_20260510T041525Z.json`.
  Local summary:
  `artifacts/local/lightzero-eval-manifests/stocktelemetry-seed3-2048-iteration0-vs-16000-summary.tsv`.
  This manifest proves the new stock-side fields are present on actual
  LightZero Pong checkpoints: table fields include `stock_steps_survived`,
  `stock_episode_length`, stock nonzero/positive/negative reward counts, and
  `stock_action_histogram`; nested `stock_evaluator.stock_rollout` includes
  `actions` and full `step_records`.
- Plain stock/manual comparison for that run: `iteration_0` manual survived
  `762/2048` with return `-21`, while stock survived `758/2048` with stock
  return `-21`; `iteration_16000` manual survived `893/2048` with return
  `-21`, while stock survived `1515/2048` with stock return `-17`, stock
  reward counts `25` nonzero / `4` positive / `21` negative, and stock action
  histogram `{"0":271,"1":775,"2":156,"3":61,"4":156,"5":96}`. Both rows have
  `stock_manual_match=false`, so the stock survival improvement is real
  stock-path telemetry, not confirmation that the manual episode matched it.

Setup mistakes / risks found:

1. **Not enough training for a stock-return expectation.** `32768` is about
   `16%` of installed-package stock `200000`; `65536` is about `33%`. Against
   current-GitHub / published-card `500000`, they are about `7%` and `13%`.
   That is enough to see noisy survival behavior, not enough to expect direct
   stable `+20` Pong.
2. **Eval config drift remains.** The eval helper rebuilds the stock config via
   the tiny-train helper, then patches `collect_max_episode_steps` /
   `eval_max_episode_steps`, `eval_freq=1`, `save_ckpt_after_iter=1`,
   `max_train_iter=1`, and by default `update_per_collect=1`. Most of this is
   eval plumbing, and `MuZeroPolicy.eval_mode` should not use learner update
   count, but it is still not a clean stock-eval surface. Use
   `--update-per-collect -1` for strict eval reruns and/or add a dedicated eval
   config path that keeps stock `None`.
3. **Single-episode capped eval is a microscope, not official score.** The
   useful 2048 cap prevents 512-step saturation, but it is still one seeded
   episode. Keep survival-first deltas, but do not treat one stock return row
   as a full evaluator result.
4. **Manual and stock eval are not bit-identical on many non-collapsed rows.**
   Keep both, but lead with stock return for return claims and new
   `stock_rollout` telemetry for stock survival/reward/action claims when it is
   present. Use manual traces/action histograms as diagnostics and for older
   artifacts that predate stock-side telemetry.
5. **Current-GitHub versus installed-package stock must not be blended.** The
   installed Modal package surface captured in manifests says `max_env_step=200000`;
   `/tmp/lightzero-src` says `500000`. Both are primary-ish local sources, but
   they are different reproduction targets.

## Update-Per-Collect Drift Rerun

Claim: rerunning the requested strict 2048-step stock-evaluator evals with
`--update-per-collect -1` keeps the same broad interpretation. The rows still
strict-load with no fallback, the later checkpoints still show weak/non-solved
signal versus same-run `iteration_0`, and manual/stock mismatch remains the
dominant eval caveat.

Non-claim: this does not prove the eval helper is bit-identical to the
trainer-side evaluator, and it does not prove solved Pong. The rerun changes
some magnitudes enough that single-episode manual survival should stay
diagnostic rather than a quality claim.

Corrected seed 3 L4 run:
`lz-visual-pong-exact-installed-0.2.0-s3`,
attempt
`train-faithful-short-installed-0.2.0-s3-65536-ckpt1000-spawn-relpath`,
eval id
`drift-s3-l4-stockupc-none-0-16000-17010-stock2048-seed3-corrected`.
Remote manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s3/attempts/train-faithful-short-installed-0.2.0-s3-65536-ckpt1000-spawn-relpath/eval/drift-s3-l4-stockupc-none-0-16000-17010-stock2048-seed3-corrected/manifest_custom_steps2048_seed3_20260510T041154Z.json`.
Local summary:
`artifacts/local/lightzero-eval-manifests/drift-s3-l4-stockupc-none-0-16000-17010-stock2048-seed3-corrected/summary_baseline_deltas.tsv`.

- Baseline `iteration_0`: `762/2048`, manual return `-21`, stock return
  `-21`, positive rewards `0`, dominant action `1` share `0.884514`.
- `iteration_16000`: `1172/2048`, `+410` manual survival, manual return
  `-19`, stock return `-20`, positive rewards `2`, dominant action `1` share
  `0.482082`.
- Latest `iteration_17010`: `847/2048`, `+85` manual survival, manual return
  `-21`, stock return `-18`, positive rewards `0`, dominant action `1` share
  `0.502952`.

Compared with the prior default-UPC seed 3 compact summary, `iteration_16000`
manual survival drops materially (`1605 -> 1172`) and stock return weakens
(`-18 -> -20`), but it remains above baseline. Latest `iteration_17010` manual
survival drops materially (`1236 -> 847`) while stock return improves
(`-21 -> -18`). That is material single-episode eval sensitivity in magnitude
and manual/stock disagreement, not a clean reversal to "no signal."

Repeat A seed 0 run:
`lz-visual-pong-exact-installed-0.2.0-s0`,
attempt
`train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath`,
eval id
`drift-repeatA-stockupc-none-0-9559-stock2048-seed0`.
Remote manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath/eval/drift-repeatA-stockupc-none-0-9559-stock2048-seed0/manifest_custom_steps2048_seed0_20260510T040400Z.json`.
Local summary:
`artifacts/local/lightzero-eval-manifests/drift-repeatA-stockupc-none-0-9559-stock2048-seed0/summary_baseline_deltas.tsv`.

- Baseline `iteration_0`: `761/2048`, manual return `-21`, stock return
  `-21`, positive rewards `0`, action `2` collapse.
- Latest `iteration_9559`: `909/2048`, `+148` manual survival, manual return
  `-21`, stock return `-18`, positive rewards `0`, dominant action `1` share
  `0.338834`.

Compared with the prior default-UPC repeat A compact summary, latest
`iteration_9559` nudges down (`973 -> 909` manual survival and `-17 -> -18`
stock return), but the same sign remains: above-baseline manual survival and
above-baseline stock return, with no manual return or positive-reward
improvement.

Artifact hygiene: an earlier seed 3 drift command omitted `--seed 3`, produced
`drift-s3-l4-stockupc-none-0-16000-17010-stock2048-seed3` with manifest seed
`0`, and is not used for the requested seed 3 comparison.

Things checked that do **not** currently look like the simple bug:

- Action repeat / frame skip: stock Atari wrapper applies `MaxAndSkipWrapper`
  using config `frame_skip`; no local override was found in official/control
  training.
- Frame stack: training/evaluator assemble four one-channel frames for the
  conv model; manual eval repeats the initial frame then rolls the stack, which
  matches the LightZero `GameSegment` pattern.
- Reward clipping: collector clipped/evaluator unclipped is stock LightZero
  Atari behavior; the flat `-21` stock return is not explained by accidental
  survival shaping.
- Sticky actions: no sticky-action wrapper appears in the inspected LightZero
  Atari path.
- Run mixing: the main summaries compare later checkpoints to the same
  attempt's own `iteration_0`; strict load is true and fallback false on the
  cited rows.

## Claim

The official/control lane is using LightZero's installed Atari Pong MuZero
training path, not the custom dummy Pong path. The active train wrapper imports
`zoo.atari.config.atari_muzero_config`, deep-copies its `main_config` and
`create_config`, validates the stock surface, and calls
`lzero.entry.train_muzero`.

The strongest current signal is survival-only: the old seed-0
`iteration_9092` survived `821/2048` versus `761/2048` for its same-run
`iteration_0`, but both rows still finished at manual return `-21`, stock
return `-21`, `21` negative/nonzero rewards, and `0` positive rewards.

Manual eval and the stock evaluator are not currently the same rollout driver.
Manual eval is a hand-rolled single-env rollout; the stock probe calls
`lzero.worker.MuZeroEvaluator` through a DI-engine env manager. Rows with
`stock_manual_match=false` prove the first recorded stock actions diverged from
the manual actions, so manual return/positive-reward counts and stock return
must not be treated as interchangeable.

## Non-Claim

This does not prove solved Pong, exact upstream reproduction, or CurvyTron
readiness.

This does not prove point-scoring behavior. The best longer-cap rows delay
losses, but they still do not win points.

This does not prove the eval code is bit-identical to LightZero's trainer-side
evaluator. It is strict enough to compare checkpoints within one recorded eval
wave, but the helper rebuilds a stock-ish config and applies eval-time patches.

This does not prove that stock evaluator return validates older manual
survival rows. As of the 2026-05-10 instrumentation patch, new stock-evaluator
artifacts record stock-path step/reward/action telemetry, but historical rows
that only have `stock_return` still cannot validate manual survival.

## Code-Grounded Read

Training path:

- `src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py` imports
  `zoo.atari.config.atari_muzero_config`, then calls
  `lzero.entry.train_muzero([main_config, create_config], seed=..., max_env_step=...)`.
- The stock surface validator expects `PongNoFrameskip-v4`,
  `env_type=atari_lightzero`, `model_type=conv`, observation shape
  `[4, 64, 64]`, action space `6`, `collector_env_num=8`,
  `evaluator_env_num=3`, `num_simulations=50`, `batch_size=256`,
  `game_segment_length=400`, `replay_ratio=0.25`,
  `update_per_collect=None`, and no collect/eval episode caps.
- The normal official/control runs intentionally changed only artifact path,
  total training budget via `max_env_step_override`, and sometimes checkpoint
  cadence via `save_ckpt_after_iter_override`. Those make the run
  "faithful-short", not exact upstream. They do not switch envs, action space,
  reward shaping, collector counts, batch size, simulations, replay ratio, or
  model shape.
- The optional survival-shaped env is guarded: it only activates when
  `--survival-reward-per-step` is positive and requires `survival-shaped` in
  both run and attempt ids. The official/control rows discussed here are not
  shaped rows.

Eval path:

- `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py` loads checkpoints
  with `_find_state_dict`, then `_load_state_dict_probe`, and raises unless the
  policy model strict-load succeeds.
- `scripts/lightzero_live_eval_queue.py` prints eval commands with
  `--no-allow-model-fallback` and `--run-stock-evaluator`. The local manifests
  report `strict_load=true` and `fallback_used=false` for the relevant rows.
- The evaluator also runs `lzero.worker.MuZeroEvaluator` for a stock return
  probe, so reports should keep both manual return and stock return.
- Manual eval creates one evaluator env from `get_vec_env_setting`, seeds it,
  resets it, maintains a frame stack, then loops until `done` or
  `max_eval_steps`. On each step it calls
  `MuZeroPolicy.eval_mode.forward` directly with the stacked observation,
  `action_mask`, `to_play`, and `ready_env_id`; it records selected actions,
  rewards, done info, `steps_run`, reward timing, and action histograms.
- Stock eval compiles the same patched config, forces one evaluator env and one
  evaluator episode, creates an env manager, wraps `policy.eval_mode` to record
  the first 32 calls, wraps the evaluator env factory to record the actual
  `env.step(action)` sequence under `lzero.worker.MuZeroEvaluator`, and then
  calls `MuZeroEvaluator`. New artifacts keep `stock_return` from
  `eval_episode_return_mean` and add `stock_rollout` telemetry:
  `steps_run` / `episode_length`, total reward, nonzero / positive / negative
  reward counts, terminal and truncation fields when exposed by `info`, full
  action histogram, action list, nonzero reward steps, and raw step records.
- The parity check only compares the manual action list against the first
  recorded stock evaluator actions. It is a divergence detector, not a proof
  that later stock survival/reward timing matches manual timing.
- Caveat: eval currently rebuilds configs through
  `_patched_stock_atari_pong_configs` from the tiny-train helper. That helper
  adds `collect_max_episode_steps` / `eval_max_episode_steps`, changes
  `policy.eval_freq` to `1`, sets checkpoint-save cadence to `1`, and by
  default sets `policy.update_per_collect=1`. These are mostly eval plumbing,
  but `update_per_collect=1` differs from the training wrapper's stock
  `None`. Treat this as config-drift risk until a strict eval helper reuses
  the exact wrapper surface or passes `--update-per-collect -1`.

Manual/stock mismatch examples:

- Repeat A latest `iteration_9559`: manual survived `973/2048` with manual
  return `-21`, `21` negative rewards, and `0` positive rewards; stock return
  was `-17`. The first 32 actions diverged, so the stock `-17` is a separate
  stock-evaluator episode result, not confirmation of the manual 973-step
  survival rollout.
- Seed 2 latest `iteration_16829`: manual survived `840/2048` with manual
  return `-20`, `21` negative rewards, and `1` positive reward; stock return
  was `-21`. Again, first-32 action parity failed.
- Earlier matched rows, such as repeat A `iteration_0` and `iteration_1000`,
  had matching first recorded actions and matching manual/stock return
  `-21`, which supports the diagnosis that the later disagreement comes from
  path/action divergence rather than from a simple summarizer bug.

Metric rule until parity is fixed:

- For official LightZero quality, lead with `stock_return` because it is the
  actual `MuZeroEvaluator` output.
- For loss-delay/survival diagnostics, prefer `stock_steps_survived` /
  `stock_episode_length`, `stock_*_reward_count`, and
  `stock_action_histogram` when present. Keep manual `steps_survived` as a
  labeled fallback for older artifacts and for direct path-divergence diagnosis.
- Do not claim "stock-confirmed survival" from historical artifacts that lack
  `stock_rollout`, and do not treat manual reward counts as comparable to
  stock reward counts when `stock_manual_match=false`.

## Most Likely Explanation

1. **Undertraining / sparse reward is the lead explanation.** The faithful
   short runs stop at `32768` or `65536` env steps versus the installed stock
   `200000` trainer budget. A Pong agent can learn to delay losing a point
   before it learns to win points. That matches `+60` or `+180` survival steps
   with unchanged `-21` return.

2. **Policy quality is weak and sometimes action-collapsed.** Several rows have
   dominant action shares near or at `1.0`. Later survival rows are less
   collapsed than the baseline in some cases, but still dominated by one action
   family and still score no positive rewards.

3. **The 512-step cap hid survival differences.** The old `iteration_9092`
   looked like return improvement at `512/512`; under `2048`, it becomes a
   clearer survival-only improvement with no return improvement. Use 2048 or
   full-episode caps for survival claims.

4. **MCTS/eval settings can amplify collapse but are not proven broken.**
   Eval uses deterministic MuZero eval mode with `num_simulations=50`. That is
   the stock setting, but deterministic eval over weak roots can produce
   near-single-action policies. This explains ugly action histograms better
   than an action-map bug.

5. **Action mapping is not the lead suspect.** The manifest reports direct Gym
   action meanings for `PongNoFrameskip-v4` as `NOOP`, `FIRE`, `RIGHT`, `LEFT`,
   `RIGHTFIRE`, `LEFTFIRE`; action space size is `6`; the collapsed action
   changes across runs. That pattern looks like weak policy/MCTS, not a fixed
   inverted-action bug.

6. **Reward handling is probably okay for these rows.** Manual and stock
   returns agree on the longer-cap key rows: all negative rewards, no positive
   rewards, terminal `eval_episode_return=-21`. There is no evidence that the
   official/control rows accidentally received survival shaping.

7. **Replay/self-play settings are stock-ish, not obviously broken.** The
   training wrapper validates stock `replay_ratio=0.25`,
   `game_segment_length=400`, `collector_env_num=8`, and `n_episode=8`.
   The likely issue is budget/sparse learning, not an accidental custom replay
   regime.

## Prioritized Bug / Hypothesis List

1. **Eval config drift from training surface.** Highest small bug risk. Re-run
   one existing checkpoint pair with `--update-per-collect -1` so eval keeps
   `policy.update_per_collect=None`, or add a strict eval path that imports the
   exact reproduction wrapper's stock surface instead of the tiny helper.

2. **Manual/stock evaluator non-parity.** Current stock probe records only
   return plus a 32-action prefix, while manual eval records the survival
   episode. Instrument the stock evaluator env path to emit stock-path
   `steps_survived`, reward timing, positive reward count, terminal info, and
   full action histogram, or drive manual metrics through the same
   `MuZeroEvaluator` env manager. Then compare stock survival to stock return.

3. **Undertrained official Pong.** Continue same-run 2048/full-episode eval at
   later checkpoints around `9000`, `16000`, `32000`, and final. Do not infer
   failure from `iteration_1000` or `iteration_3000` alone.

4. **Deterministic eval over weak MCTS roots.** For the same checkpoint and
   same first observations, sweep `num_simulations` and log visit counts,
   selected action, and value. If actions diversify or survival changes, the
   collapse is partly eval/search sensitivity.

5. **Metric cap confusion.** Stop leading with 512-step saturated survival.
   Use 2048 or full-episode reads whenever a row hits `512/512`.

6. **Action-map guardrail.** Keep this low priority unless a fixed-seed
   scripted baseline or direct ALE trace shows action effects are wrong.

## Minimal Tooling Fix

Implemented stock-path metric capture in `lightzero_pong_eval_smoke.py`: the
env function passed into `_run_stock_evaluator_probe` is wrapped with a small
recorder that logs every stock-evaluator `step(action)` reward/done/info and
action for the one evaluator episode. New artifacts persist
`stock_rollout.steps_run`, `stock_rollout.episode_length`,
`stock_rollout.reward_histogram`, nonzero / positive / negative reward counts,
terminal/truncation fields when exposed, nonzero reward steps, action lists,
step records, and `stock_rollout.action_histogram`. The manifest table and
`scripts/summarize_lightzero_pong_eval_manifest.py` now render the key
`stock_*` columns when present.

That is smaller and cleaner than trying to infer survival from
`eval_episode_return_mean`, and it makes repeat A `iteration_9559` and seed 2
`iteration_16829` directly comparable on the same stock evaluator path.

## Next Smallest Experiment

Re-run one already-evaluated pair under the same 2048-step strict
stock-evaluator recipe, but pass `--update-per-collect -1`. Good candidates:
seed 0 detached `iteration_0/5000/9086` or seed 2
`iteration_0/10000/16829`.

Why this is the smallest useful test: it does not train anything, does not
change checkpoints, and directly tests whether the eval helper's
`update_per_collect=1` config drift is influencing action selection or returns.

Disproof target: if `iteration_9092` still shows about `+60` survival steps and
still has `-21` manual/stock return with `0` positive rewards, the survival-only
read is not caused by that eval config drift. Then the next likely story is
undertraining/weak policy, not a loader/eval bug.

Active board pointer:
[docs/working/training_coach_active_board_2026-05-10.md](training_coach_active_board_2026-05-10.md).

No pytest was run.
