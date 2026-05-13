# Training Coach Active Board - 2026-05-10

Short current board. This is the working-memory page; detailed evidence lives
in linked docs.

## Start Here

North Star: [coach_north_star_2026-05-10.md](coach_north_star_2026-05-10.md).

Next reward gate: first CurvyTron runs should keep trainer reward and eval
metrics separate. The active two-seat default is now:

- `scaled_tiny_survival_plus_outcome`: `+0.01` while a seat is alive after a
  policy decision, plus immediate same-step bonus pickup helper `+0.05` per
  bonus caught by that player, plus sparse terminal outcome `(+1/-1/0) *
  0.01 * episode_step_count`.

Survival length is always eval/telemetry, not the trainer reward by itself.
Promotion needs heldout eval and telemetry that separate trainer reward, sparse
outcome, survival length, timeout/truncation, action behavior, and terminal
causes.

Implementation guardrail: the main CurvyTron learning gate is actual
current-policy self-play. Fixed-opponent, turn-commit, and centralized
joint-action stock LightZero runs are controls/profile paths unless they also
prove honest two-player current-policy self-play. The active full self-play
lane is the canonical Coach launcher
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
with `--mode two-seat-selfplay`.

Eval guardrail: survival progress is measured from episode length. Eval may
need `model_reward_variant` only to reconstruct a checkpoint's LightZero
support/model shape; that is not the eval score.

## Current State

Updated 2026-05-12 after optimizer fast-direct correction.

- Launch hold: do not start new overnight CurvyTron jobs until the user gives
  the exact launch recipe. The code is launch-ready, but no new overnight run
  should be started from this board alone.
- Pre-overnight cleanup passed local compile/ruff and Modal smoke coverage for
  the corrected two-seat path. Current reward/no-op meaning:
  skipped policy chances send NOOP and stay out of replay/reward targets;
  bonus pickup reward is immediate on the exact pickup step. Because skipped
  physical ticks can hide terminal/bonus credit from replay, policy no-op skip
  is no longer a safe default for the first serious runs.
- Default two-seat training horizon and background survival-eval cap are both
  `65,536` steps. GIF max steps stay short by default because GIFs are visual
  samples, not the survival metric.
- Optimizer recommendation now supersedes the earlier B32/browser-lines canary:
  run an aggressive approximation-heavy matrix. Main surface is
  `fast_gray64_direct`, mostly L4/T4, B64, sim8, collect64, updates4,
  accumulated replay, learner sample 256, normal death, CurvyZero background
  eval/GIF on. Browser-lines is only one or two small sentinels and should not
  gate the fast-direct matrix. Historical browser-lines timing canaries: B32
  finished 4 iterations in `626s` with `6,734` fresh rows; B64 finished 4
  iterations in `1,183s` with `12,590` fresh rows. Those browser-lines numbers
  do not override the fast-direct recommendation. Optimizer matrix:
  [optimizer recommendations](optimizer/coach_next_training_run_recommendations_2026-05-12.md).
- Live-run guardrail: optimizer can read status/progress/logs/checkpoints from
  overnight runs, but should not mutate them. Early read-only fast-direct
  progress says the speed bottleneck has shifted toward policy/search/MCTS, not
  rendering, in rows that are actually collecting.
- Run naming rule: every serious run starts with a readable purpose prefix and
  names the important variant knobs. Use names like
  `curvy2seat-selfplay-baseline-noskip-b32-sim8-20260512` or
  `curvy2seat-selfplay-variant-obsnoise10-b32-sim8-20260512`, not bare seed or
  vague run ids.
- Action-collapse rule: a deterministic greedy GIF choosing one action is a
  warning, not proof of training collapse. The overnight blocker is collapse in
  fresh policy-decision histograms from trainer progress. Physical action
  histograms are secondary because no-op skips intentionally execute NOOP.

- Canonical CurvyTron Coach launcher is now:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  with `--mode two-seat-selfplay`.
- Default Coach path is current-policy two-seat self-play on GPU L4/T4. One
  live LightZero MuZero policy chooses both players' actions from the same
  pre-step state; the CurvyTron env advances once with the joint action; learner
  updates mutate that same policy for later collection.
- Reset starts are varied by generated reset seeds. A run can keep one top-level
  seed for reproducibility, but training calls `reset(seed=None)` /
  `autoreset_done_rows(seed=None)` so every env restart gets a fresh per-row
  derived seed. Replay rows and step records log `reset_seed` for audit.
  Source-default natural bonus spawning is on by default; no-bonus is only a
  controlled ablation. Default stochasticity is now visual-only: the legacy
  `policy_action_repeat_*` flags mean policy no-op skips, not held actions, and
  default to off (`min=1`, `max=1`, `extra_probability=0.0`). Add visual
  Gaussian noise `0.10`, keep random no-op/drop off, and use no warmup schedule.
- Default two-seat trainer reward is shaped but explicit: the reward float that
  replay/learner consume is tiny survival helper plus immediate same-step bonus
  pickup helper plus scaled sparse terminal outcome. The bonus reward is not an
  end-of-game sum; it is only attached to the player row for the step where that
  player catches the bonus. The row also logs dense helper, bonus pickup count,
  raw sparse outcome, terminal outcome, episode step count, and return-target
  discount.
- The older `lightzero_curvytron_two_seat_train_smoke.py` Modal wrapper has
  been deleted. Historical commands must be translated to the canonical
  launcher before use.
- Tiny canonical wrapup smoke passed:
  `curvytron-canonical-two-seat-wrapup-smoke-20260512` /
  `wrapup-smoke`. It ran on Modal GPU, used `cuda:0`, collected both seats,
  changed model weights, wrote `iteration_0` and `iteration_1`, and wrote
  progress/summary/checkpoint artifacts.
- Canonical observability smoke passed:
  `curvytron-canonical-two-seat-observability-smoke-20260512` / `obs-smoke`.
  It kept stock LightZero in-training eval off, spawned CurvyZero checkpoint
  eval/inspection and GIF jobs, completed 2 eval/inspection jobs and 2 GIF jobs,
  and loaded two-seat checkpoints by inferring the 601-wide support heads.
- Background checkpoint eval, inspection, and GIF spawning are on by default.
  The loader now infers the checkpoint model support-head size before loading,
  so two-seat checkpoints with the LightZero Atari-style 601-wide heads can be
  inspected. The eval is still a fixed-opponent survival read, so use it as
  observability, not as proof of two-seat self-play strength.

- Pong replication has passed the basic learning-signal gate. Stock-like visual
  LightZero Pong learned to survive longer across several same-run checkpoint
  curves.
- Active Modal Pong training jobs have been stopped. Checkpoints and eval
  artifacts remain on the Modal Volume.
- Pong is now a control/reference lane, not the main training lane.
- CurvyTron is the active Coach lane. The source-state turn-commit wrapper,
  `env_variant=source_state_turn_commit`, is useful as a stock LightZero
  plumbing smoke/profile path only. It is not a trainable/default path and not
  learning-quality current-policy self-play.
- Turn-commit shape: player 0's scalar step records a private pending action,
  does not advance physics, and gets reward `0`; player 1's scalar step commits
  the full joint action, advances the real source tick, and gets the survival
  reward. Stock LightZero stores both scalar steps as normal GameSegment
  transitions, so value targets may give player 0 states credit for player 1
  survival. Treat this as a reward-credit risk.
- The old separate two-seat Modal wrapper has been deleted. The canonical
  launcher above owns the two-seat self-play entrypoint.
- Historical two-seat repeat/dropout plumbing smoke passed on Modal before the
  no-op-skip fix:
  `curvytron-two-seat-repeat-smoke-s6101-20260511` /
  `repeat-smoke-s6101`. It ran on GPU, used one live LightZero policy for both
  seats, changed model weights, saved `iteration_0` and `iteration_1`, and
  logged the old per-seat repeat behavior (`16` active seat rows, `8` fresh
  decisions, `8` reused actions). Do not use this as current behavior evidence:
  the active implementation now sends NOOP on skipped policy chances and keeps
  skipped ticks out of replay/reward targets.
- Four detached long two-seat CurvyTron runs were launched after that smoke:
  `clean-detached-s6301`, `repeat-mild-detached-s6302`,
  `repeat-strong-detached-s6303`, and `obsnoise002-detached-s6304`. All use
  current-policy two-seat collection, accumulated replay, survival reward,
  `batch_size=32`, `collect_steps_per_iteration=128`,
  `updates_per_iteration=8`, `num_simulations=4`, `max_ticks=16384`, and
  checkpoint every `100` iterations. Active containers were visible in
  `modal container list` after launch.
- CurvyTron native LightZero trainer now exposes `source_state_turn_commit` and
  `source_state_joint_action` in addition to fixed-opponent controls. Treat
  fixed/frozen opponent, turn-commit, and centralized joint-action runs as
  controls. None of those are true current-policy competitive self-play.
- Source-state turn-commit plumbing smoke passed after cleanup:
  `curvytron-source-state-turncommit-smoke-s20260511b` /
  `profile-smoke-sim2-c2-steps64-20260511b`. It used stock LightZero
  `train_muzero`, GPU model/search, MCTS, replay sample, one learner step, and
  copied `iteration_0`.
- Target/replay audit smoke then confirmed the blocker:
  `curvytron-source-state-turncommit-audit-smoke-s20260511c` /
  `profile-audit-smoke-sim2-c2-steps64-20260511c`. LightZero GameSegments
  stored fake pending rows with alternating rewards like `0,1,0,1`, and sampled
  value targets backed those commit rewards through pending rows. `mode=train`
  is blocked for `source_state_turn_commit` until reward credit is redesigned.
- That smoke wrote env-step telemetry: `36` scalar rows, `18` pending rows,
  `18` physical-commit rows, balanced acting-player rows (`18/18`), natural
  source mechanics, and no action collapse in sampled rows.
- Env-wrapper cleanup now covers the safe plumbing issues from the audit:
  source-state env knobs, metadata, telemetry, stack schema hash, and
  non-mutating render. This does not fix reward credit.
- Joint-action smoke passed on Modal:
  `curvytron-source-state-joint-action-smoke-20260511a` /
  `smoke-sim2-c2-steps64-20260511a`. It used stock `train_muzero`, saved
  `iteration_0` through `iteration_6` plus `ckpt_best`, collected `53`
  physical rows, had no action-collapse warning, and used all 9 scalar joint
  actions (`0:11, 1:4, 2:1, 3:4, 4:11, 5:8, 6:4, 7:7, 8:3`).
- That smoke is a stock-path control, not a learning claim. It uses
  `curvyzero_all_players_alive_diagnostic/v0`: reward `+1` if both players are
  alive after the tick, else `0`. Stock LightZero targets here are still the
  configured discounted scalar reward, not the two-seat `gamma=1`
  winner/loser return schema.
- Sparse/dense fixed-opponent reward variant smokes passed on Modal under
  `curvytron-reward-variant-smoke-20260511`. Both called stock
  `lzero.entry.train_muzero`, passed readiness, wrote checkpoints, and had no
  action-collapse warning. Sparse produced trainer reward only from terminal
  outcome (`trainer_reward_sum=3.0` over `35` telemetry rows); dense produced
  survival helper plus terminal outcome (`trainer_reward_sum=38.0` over `35`
  rows). Both logged survival separately.
- Default reward dry smoke passed under
  `curvytron-reward-default-smoke-20260511` / `auto-default-sparse-dry`.
  Omitting `--reward-variant` now resolves to `sparse_outcome`, not dense.
- Stop-cap fix: `stop_after_learner_train_calls=0` now means "no learner-call
  cap." The old default `1` silently made "long" runs end after one learner
  train call unless every launch overrode it correctly.
- Tiny wait-mode canary passed after the stop-cap fix:
  `curvytron-reward-stopcap-smoke-wait2-20260511` /
  `default-nocap-maxenv64-wait2`. It called stock `train_muzero`, wrote
  `summary.json`, `status_heartbeat.json`, `env_steps.jsonl`, and action
  observability. It was only a launch/artifact canary, not a learning run.
- Eval-cadence fix: `lightzero_eval_freq=0` now means "skip stock LightZero
  eval during the training run" by setting the internal eval interval beyond
  `max_train_iter`. Background CurvyZero checkpoint survival eval, inspection,
  and GIF spawning are separate from stock LightZero eval and are on by default.
- Eval harness fix: standalone CurvyTron survival eval now passes the new
  `lightzero_eval_freq` argument into the shared config builder. The earlier
  `steps=0` eval rows in the `waitlong` runs were setup failures, not survival
  measurements.
- Stopped the older eval-heavy fixed-opponent comparison apps:
  `curvytron-reward-compare-sparse-sim16-waitlong-20260511` and
  `curvytron-reward-compare-dense-sim16-waitlong-20260511`.
- Stopped the corrected fixed-opponent `cleanlong` pair after deciding that
  fixed-opponent controls are not the main gate. They are useful controls, but
  they do not test two-player current-policy self-play.
- Active CurvyTron training lane is now the two-seat current-policy self-play
  path. This is not stock `train_muzero`; it is the small custom adapter where
  one live LightZero MuZero policy chooses both players' actions before the env
  advances, and learner updates mutate that same policy for later collection.
  Treat it as the full current-policy self-play lane for now, with the caveat
  that it is custom code.
- Live14 self-play runs launched with frequent progress writes and no automatic
  eval: `clean`, `explore`, `repeat`, and `strong`. Latest checks show all four
  moving, no problems, model weights changing, all three actions used by both
  players, and mean completed episode length still around `10-11` steps. This
  is a moving baseline, not yet a learning claim.
- Modal app list at 2026-05-11 19:13 EDT shows the active CurvyZero detached
  training apps are two-seat self-play trainer jobs: the older `s6301`-`s6304`
  batch plus the `live14` clean/explore/repeat/strong batch. The fixed-opponent
  `curvyzero-lightzero-curvytron-visual-survival-train` apps from this cleanup
  pass are stopped.

## Pong Facts

Final cleanup evals confirmed that several Pong runs learned to survive longer.
The curves are noisy, so compare each run to its own `iteration_0`.

| run | checkpoints | mean survival steps | mean game score |
| --- | --- | --- | --- |
| `s114` L4/T4 stock64 | `0/13159` | `759.625 -> 1014.75` | `-21 -> -19.875` |
| `s120` L4/T4 stock64 | `0/14012` | `759.625 -> 1125.88` | `-21 -> -19.75` |
| `s121` L4/T4 stock64 | `0/17013` | `871.125 -> 759.625` | `-20.875 -> -21` |
| `s142` H100 stock64 repeat | `0/15000/20000/26000/26806` | `759.625 -> 873 -> 1214.62 -> 910.5 -> 1308.88` | `-21 -> ? -> ? -> ? -> -18.5` |
| `s122` H100 stock64 | `0/26000/26672` | `759.625 -> 2395.12 -> 2970.62` | `-21 -> -14.25 -> -12.375` |
| `s123` exact control | `0/20000/30000` | `759.625 -> 1147.25 -> 2008.12` | `-21 -> -19.5 -> -10.75` |
| `s113` exact control | `0/20000/30000` | `759.625 -> 862.125 -> 1481.75` | `-21 -> -20.625 -> -18.25` |

Common eval sampler seed for this cleanup wave:
`202605111551`.

## CurvyTron Facts

Current native trainer path:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

Despite the stale filename, the native path is source-state visual CurvyTron.
There are now three important variants:

- `env_variant=source_state_fixed_opponent`: fixed/frozen opponent control, not
  self-play.
- `env_variant=source_state_turn_commit`: stock LightZero `train_muzero` with a
  lightweight turn-commit env wrapper. This is a native plumbing smoke/profile,
  not a learning-quality self-play claim. The trainer now blocks
  `mode=train` for this variant after the target audit.
- `env_variant=source_state_joint_action`: stock LightZero control candidate
  where one scalar action indexes the centralized pair
  `(player_0_action, player_1_action)`. This is one real source tick per
  transition and avoids turn-commit pending rows, but it is centralized control,
  not true competitive self-play.
- All three use non-ALE visual stack `[4,64,64]`.

Optimizer smoke evidence:

- `opt-native-train-smoke-c16-s1121-wait`
- `train-smoke-c16-sim16-sparse-wait`
- `called_train_muzero=true`
- `ok=true`
- checkpoints copied for `iteration_0`, `iteration_35`, and `ckpt_best`

This is setup evidence only. It is not a CurvyTron learning claim.

Reward-credit caveat for turn-commit: the first scalar step is bookkeeping and
gets zero reward because physics has not advanced yet. The second scalar step
commits the real tick and gets the physical-step survival reward. Because stock
LightZero stores both scalar steps as normal transitions, value targets can
credit the pending/player0 state for player1 survival. Use turn-commit for
stock plumbing smoke/profile only; do not call it trainable or current-policy
self-play.

Turing recommendation, candidate/control only until tested:
`source_state_joint_action`, a 9-action centralized joint-action wrapper. One
LightZero scalar action maps to `(p0_action, p1_action)`, one real CurvyTron
tick, one reward, `to_play=-1`, and `action_space_size=9`. Loud caveat:
centralized control, not true competitive self-play.

Implementation note: the first diagnostic scalar is `+1` only while both
players are alive after the real tick, otherwise `0`. That is a single control
reward for the centralized policy. It is not per-player reward, not zero-sum,
and not a competitive self-play objective. Do not apply two-seat
winner/loser-return shaping to this wrapper unless it grows an honest per-player
target surface.

Current launcher ref:
[canonical two-seat handoff](training/curvytron_canonical_two_seat_handoff_2026-05-12.md).
Older cleanup notes that predate the canonical launcher were moved to
`training/archive_2026-05-12_two_seat_purge/`.

Reward-shaping note: the current two-seat reward keeps the LightZero contract
simple. The collector writes one reward float per replay row; the row metadata
stores the pieces. This is close to standard LightZero env behavior and avoids
a new replay API. Use eval and telemetry to decide: trainer reward, sparse
outcome, survival length, timeout/truncation, action behavior, and terminal
causes must stay separate. The old full-scale `+1 per step + T winner / -T
loser` idea is too large for default training; the current version uses the
same shape but scales it down to `0.01`.

Target/support note: any value/reward support or discount tuning is a separate
training-config ablation. Do not hide it inside reward shaping. If support size
changes to fit a reward stream, report it next to the reward variant.
Any reward variant or target profile saved with a model/checkpoint is for
reconstructing the model and target shape needed to load that checkpoint. It
does not define the eval score.

Two-seat debug return schema: `terminal_winner_keeps_survival_loser_zero/v0`
uses raw finite survival count with `gamma=1.0`; the decisive winner keeps its
episode survival return and the loser trajectory is zeroed. This is a debug
target path, not the current stock LightZero joint-action scalar reward.

The active two-seat path exposes legacy `policy_action_repeat_*` knobs whose
current meaning is policy no-op skipping. Skipped no-op ticks are not learner
rows. Robustness variants may set these knobs, but learning claims should say
whether the run used them.

Historical launched two-seat run refs before the latest no-op-skip and bonus
pickup reward cleanup:

| run | attempt | function call | variant |
| --- | --- | --- | --- |
| `curvytron-two-seat-selfplay-live14-clean-observable-20260511` | `live14-clean-progress1` | `fc-01KRCM72SVS61ZM51WEKYSN3G0` | clean self-play, temp `1.0`, epsilon `0.25` |
| `curvytron-two-seat-selfplay-live14-explore-observable-20260511` | `live14-explore-progress1` | `fc-01KRCM72TBPF684GRWVBXM5BG5` | higher exploration, temp `2.0`, epsilon `0.50` |
| `curvytron-two-seat-selfplay-live14-repeat-observable-20260511` | `live14-repeat-progress1` | `fc-01KRCM72SP02E32QNTQK033B4A` | repeat max `3`, extra probability ramps to `0.10` over `1000` iterations |
| `curvytron-two-seat-selfplay-live14-strong-sim8-20260511` | `live14-strong-sim8-progress1` | `fc-01KRCMQDT933KQXQ7RB1XCFFVZ` | stronger search/update run, temp `1.5`, epsilon `0.25`, sims `8`, updates `8` |

These are not current launch guidance. Historical live14 shared knobs for
`clean`/`explore`/`repeat`: GPU L4/T4,
`batch_size=16`, `collect_steps_per_iteration=64`,
`updates_per_iteration=4`, `num_simulations=4`, accumulated replay,
`learner_sample_size=128`, `max_ticks=16384`,
`checkpoint_every_iterations=500` as an explicit run override, progress every
iteration, initial checkpoint saved. Current canonical checkpoint cadence
defaults to `100` iterations.

Live14 `strong` uses the same main shape but raises search/update pressure:
`num_simulations=8`, `updates_per_iteration=8`, and
`learner_sample_size=256`.

Live14 early progress, 2026-05-11 23:06 UTC:

| run | iteration | total steps | mean completed episode steps | max completed episode steps | action read |
| --- | ---: | ---: | ---: | ---: | --- |
| `live14-clean` | `45` | `2880` | `10.73` | `26` | balanced, no collapse |
| `live14-explore` | `49` | `3136` | `10.87` | `28` | balanced, no collapse |
| `live14-repeat` | `49` | `3136` | `10.74` | `32` | balanced, no collapse |

Live14 later progress, 2026-05-11 23:11 UTC:

| run | iteration | total steps | mean completed episode steps | max completed episode steps | action read |
| --- | ---: | ---: | ---: | ---: | --- |
| `live14-clean` | `63` | `4032` | `10.45` | `33` | balanced, no collapse |
| `live14-explore` | `78` | `4992` | `10.75` | `25` | balanced, no collapse |
| `live14-repeat` | `79` | `5056` | `10.53` | `31` | balanced, no collapse |
| `live14-strong` | `16` | `1024` | `10.97` | `27` | balanced, no collapse |

At this speed, checkpoint `iteration_500` is expected roughly around the
75-90 minute mark after launch. Until then, progress rows are the main signal.

The older `live13` launch used `progress_every_iterations=50`, which made it
hard to tell quickly whether the first iterations were moving. Those apps were
stopped and superseded by `live14`.

The earlier non-detached `s6201`-`s6204` launch attempt printed refs but did
not show active containers or progress directories. Treat those refs as
superseded unless later Volume evidence appears.

Check at 2026-05-11 16:36 EDT:

- Detached `s6301`-`s6304` apps are active in Modal.
- Progress files exist for all four runs.
- `progress_latest.json` is still the `start` row for all four runs.
- `iteration_0`, `latest.pth.tar`, and `ckpt_best.pth.tar` exist for all four.
- No iteration-25 progress row or iteration-100 checkpoint yet.

## CurvyTron Auto Resume

Implemented 2026-05-11 in:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

Facts:

- Resume is automatic. There is no resume flag.
- At startup, the trainer scans the current run directory for
  `iteration_*.pth.tar` checkpoints.
- It checks the current attempt, older attempts under the same run, and the
  run-level `checkpoints/lightzero` mirror.
- It picks the highest numbered iteration checkpoint and sets
  `policy.learn.learner.hook.load_ckpt_before_run`.
- LightZero/DI-engine then restores learner `last_iter` and `last_step` when
  present, plus MuZero model, target model, and optimizer state.
- If a matching CurvyZero sidecar exists, the trainer also restores safe state:
  collector progress, evaluator progress, policy helper state when visible, and
  Python/NumPy/Torch random state.
- Replay GameSegments are not restored yet. Raw LightZero GameSegment objects
  include state that did not reload cleanly through this sidecar path.
- Live environment manager internals are not restored.
- Sidecars use `iteration_N.resume_state.pkl`. The checkpoint-name parser uses
  plain string checks, not regex.

Validation:

- Dry smoke `autoresume-dry-smoke-20260511a` found
  `iteration_35.pth.tar` and wrote it into `load_ckpt_before_run`.
- Train smoke `autoresume-train-smoke-20260511a` loaded the same checkpoint,
  called `train_muzero`, saved `iteration_35/36/37`, and had no action
  collapse in telemetry (`0:14, 1:15, 2:16`).
- Two-phase sidecar smoke
  `curvytron-autoresume-sidecar-smoke-20260511c` passed:
  phase 2 found `iteration_2.pth.tar`, found
  `iteration_2.resume_state.pkl`, resumed, collected new env rows, saved four
  checkpoints, and had no action collapse (`0:6, 1:6, 2:8`).
- This validates resume mechanics only. It is not a CurvyTron learning claim.

## Reporting Rules

- Report trainer reward, sparse outcome, and survival length side by side; do
  not let dense/helper reward stand in for true outcome.
- Treat eval/progress survival as episode length. Checkpoint reward variant or
  target profile only tells loaders how to rebuild checkpoint shape.
- Keep LightZero's stock evaluator separate from the CurvyZero checkpoint
  survival harness. `lightzero_eval_freq` controls the in-training stock
  evaluator; `background_eval_*` and the standalone eval module control
  checkpoint survival eval.
- Compare each run to its own `iteration_0`.
- Do not call early flat Pong reads failures before enough horizon.
- Do not mix Pong and CurvyTron claims.
- Do not report a training claim without checkpoint refs, eval settings,
  action histogram or collapse check, and a plain non-claim.
- Treat seeds as reproducibility tools, not a thing to overfit.

## Eval Cadence

- Current default for long CurvyTron training runs: stock LightZero eval off
  unless explicitly requested; CurvyZero checkpoint eval, inspection, and GIF
  spawning on; checkpoint cadence `100` iterations by default.
- If checkpoint cadence is made much faster for debugging, remember that each
  checkpoint can spawn CurvyZero observability work.

## Current Gates

- Keep CurvyTron as the active training lane.
- Use CurvyTron checkpoint eval/inspection artifacts for trainer reward, sparse
  outcome, survival length, timeout/truncation, action behavior, and death
  causes.
- Treat no survival growth in CurvyTron as an early stop-and-debug signal, but
  promote only after eval shows sparse outcome progress or an explicitly
  accepted tie-break inside a sparse-outcome-equivalent band.

## Links

- Pong monitor:
  [lightzero_pong_replication_monitor_2026-05-11.md](lightzero_pong_replication_monitor_2026-05-11.md).
- Pong stock64 comparison:
  [training/pong_stock64_signal_comparison_2026-05-11.md](training/pong_stock64_signal_comparison_2026-05-11.md).
- Training knowledge index:
  [training/coach_training_knowledge_index_2026-05-11.md](training/coach_training_knowledge_index_2026-05-11.md).
- CurvyTron eval/inspection handoff:
  [training/curvytron_checkpoint_eval_inspection_handoff_2026-05-11.md](training/curvytron_checkpoint_eval_inspection_handoff_2026-05-11.md).
- Optimizer Coach handoff:
  [optimizer/coach_handoff_2026-05-11.md](optimizer/coach_handoff_2026-05-11.md).
