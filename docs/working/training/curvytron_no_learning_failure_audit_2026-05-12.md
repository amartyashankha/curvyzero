# CurvyTron No-Learning Failure Audit - 2026-05-12

Purpose: explain why the May 12 CurvyTron training waves did not show useful
survival learning, and separate artifact facts from likely causes.

No code was edited for this audit. The active training Modal apps were stopped
earlier; artifacts remain in the `curvyzero-runs` Modal Volume.

## Short Read

The main issue is probably not "MuZero cannot learn this game." The main issue
is that the CurvyTron path we scaled was not the same trusted path as stock
Pong.

Stock Pong learning evidence came from LightZero's normal `train_muzero` loop:
collector, GameSegment, GameBuffer, learner, checkpoints, and eval.

The CurvyTron two-seat path uses LightZero policy/search and
`learn_mode.forward`, but it owns collection, replay rows, targets, and
checkpointing itself. That can change weights without proving the training
contract is correct.

Two concrete red flags showed up:

- Pure current-policy self-play produced almost no competitive terminal signal
  in representative latest progress files. Hundreds of episodes completed, but
  `sparse_outcome_reward_sum` and `terminal_outcome_reward_sum` were often
  `0.0`; the learner mostly saw the small alive reward.
- The two-seat path passes CurvyTron player ids as `to_play=0/1`, while the
  repo's LightZero env notes say non-board-game CurvyTron/Pong-like rows should
  use `to_play=-1` unless we deliberately implement and test board-game player
  semantics.

## What Actually Ran

Main CurvyTron wave:

- Script: `scripts/launch_curvytron_overnight40_20260512.zsh`
- Path:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode two-seat-selfplay`
- 40 rows, mostly `fast_gray64_direct`, normal death, accumulated replay,
  `B64/sim8/collect64/updates4/sample256`.
- Checkpoints every `25` or `50` iterations depending on the final script/docs
  state; artifacts are still in the Volume.

Frozen-opponent wave:

- Script: `scripts/launch_curvytron_mixpast_20260512.zsh`
- Same two-seat path, with a fraction of env rows using a frozen older
  checkpoint as one opponent.
- Replay rows from frozen-controlled seats were excluded by design.

## Artifact Snapshot

Representative progress curves from `progress.jsonl` did not show sustained
survival improvement. Mean completed episode length stayed around the random
early-life regime.

| run | latest iter | first mean steps | best mean steps | latest mean steps | latest max steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| overnight40 row 01 default | 1060 | 14.50 | 15.75 | 14.90 | 55 |
| overnight40 row 04 default seed | 780 | 14.66 | 16.32 | 13.39 | 43 |
| overnight40 row 18 lr=1e-4 | 780 | 14.95 | 15.98 | 14.73 | 41 |
| overnight40 row 23 terminal-only | 770 | 15.34 | 15.61 | 14.63 | 38 |
| overnight40 row 25 survival-only | 700 | 15.75 | 16.19 | 15.38 | 46 |
| overnight40 row 39 browser sentinel | 420 | 14.13 | 17.55 | 14.86 | 34 |
| mixpast row 04 frozen p25 | 310 | 13.85 | 15.19 | 12.82 | 32 |

Representative latest reward components:

| run | completed episodes | sparse outcome sum | terminal outcome sum | alive reward sum | bonus reward sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| overnight40 row 01 default | 290 | 0.0 | 0.0 | 78.88 | 0.0 |
| overnight40 row 04 default seed | 335 | 0.0 | 0.0 | 78.48 | 0.0 |
| overnight40 row 18 lr=1e-4 | 295 | 0.0 | 0.0 | 78.87 | 0.0 |
| overnight40 row 25 survival-only | 280 | 0.0 | 0.0 | 79.00 | 0.0 |
| overnight40 row 39 browser sentinel | 72 | 0.0 | 0.0 | 19.74 | 0.0 |
| mixpast row 04 frozen p25 | 336 | 21.0 | 4.18 | 69.06 | 0.0 |

Plain read: the frozen-opponent run created some win/loss signal, but it still
did not improve survival. The pure same-policy self-play rows mostly trained on
"stay alive for one more step" and did not get useful competitive outcome.

Training collection actions were not fully collapsed in these progress files.
For example row 01 latest action counts were varied enough that the top action
was about `0.64` overall. Greedy GIF collapse is still useful to inspect, but it
is not the main explanation by itself.

## Why This Differs From Pong

The Pong lane eventually showed real survival movement once the stock runs had
enough horizon. The Pong audit records stock64 examples such as `s114`, `s120`,
`s121`, `s122`, `s113`, and `s123` improving survival on later checkpoints.

CurvyTron did not use that same trusted loop. The current two-seat result
payload explicitly says:

- `called_train_muzero: false`
- no LightZero collector
- no LightZero upstream GameBuffer target builder
- local two-seat replay adapter plus direct `MuZeroPolicy.learn_mode.forward`

The direct learner adapter builds a shape-compatible batch by hand:

- repeats the same sampled action across unroll steps;
- repeats the immediate reward across unroll steps;
- repeats the same root policy across unroll steps;
- builds value targets from local return metadata;
- sets masks and weights locally.

This may be right enough for a smoke, but it is not proven equivalent to
LightZero's native `MuZeroGameBuffer.sample(...)` target construction.

## Likely Failure Causes

1. The custom two-seat training contract is not proven.

   This is the largest issue. We treated the path as if it were "basically
   LightZero training," but it is a repo-owned collector/replay/target adapter.
   A flat curve may mean the adapter is wrong, not that CurvyTron is too hard.

2. Pure same-policy self-play likely removed the terminal signal.

   With one shared policy controlling both symmetric seats, many episodes appear
   to end with no winner or zero sparse outcome. That gives both seats almost
   the same alive reward and very little reason to improve competitively.

3. `to_play` may be wrong for this environment shape.

   The code passes CurvyTron player ids into LightZero as `to_play=0/1`. Existing
   env notes say CurvyTron/Pong-like scalar ego rows should use `to_play=-1`.
   If LightZero treats `to_play` as board-game player identity, this can silently
   affect value/policy semantics.

4. Replay is small relative to the data rate.

   With `B64`, two seats, and `collect64`, one iteration can create about
   `8192` replay rows. The default `max_replay_rows=65536` keeps about eight
   iterations. That is a very young window for long-horizon credit assignment,
   especially if terminal events are sparse or episodes span many iterations.

5. Reward scale and value support need a real audit.

   Defaults use `alive=0.01`, terminal outcome per step `0.01`, and
   `return_target_discount=1.0`. If agents ever survive much longer, value
   targets can grow large. That may be okay, but it must be checked against the
   LightZero support/value transform actually used by the policy.

6. Some stochastic variants were diagnostic, not clean learning runs.

   The repeat/no-op skip path can advance physical ticks without learner replay
   rows for those ticks. New code now blocks this for optimizer training, but
   earlier diagnostic rows should not be read as learning evidence.

## What Not To Conclude

- Do not conclude that CurvyTron cannot be learned.
- Do not conclude that LightZero cannot learn visual games; stock Pong did show
  survival learning.
- Do not conclude that greedy GIF one-action behavior proves training collapse;
  collect-mode action logs can be more varied than greedy eval clips.
- Do not treat `ckpt_best` from the custom path as a real metric-best
  checkpoint. In this path it can be a copy of the latest saved checkpoint.

## Next Gate

Before another large CurvyTron run, pass one of these two gates:

1. **Stock-loop gate:** make CurvyTron look like a normal scalar-action
   LightZero env row and call stock `train_muzero`. The env can hide opponent
   or simultaneous mechanics internally, but LightZero should own collector,
   GameBuffer, learner, and target construction.

2. **Native-buffer bridge gate:** keep the custom simultaneous collector, but
   convert per-seat trajectories into native-compatible GameSegments, push them
   through LightZero's GameBuffer, and compare sampled target batches against
   the current hand-built adapter.

Related cleanup note:
[curvytron_train_muzero_reconciliation_2026-05-12.md](curvytron_train_muzero_reconciliation_2026-05-12.md)
explains how stock `train_muzero` paths still exist, why turn-commit was blocked
for training, and why the custom two-seat path should be treated as a collector
prototype until native replay semantics are restored.

Either way, also verify:

- `to_play=-1` for non-board-game rows unless a tested board-game contract says
  otherwise;
- sparse outcome / terminal reward is nonzero when someone dies alone;
- pure self-play is not stuck in symmetric draw-only data;
- reset randomization creates genuinely different starts;
- replay age, target min/max, and terminal outcome rates are reported every run.

## Useful Pointers

- Current launcher doc:
  `docs/working/training/curvytron_overnight40_launch_2026-05-12.md`
- Frozen-opponent doc:
  `docs/working/training/curvytron_frozen_checkpoint_mix_plan_2026-05-12.md`
- Reward contract:
  `docs/working/training/curvytron_two_seat_reward_contract_2026-05-12.md`
- Custom-vs-stock autopsy:
  `docs/working/training/archive_2026-05-12_two_seat_purge/custom_vs_stock_muzero_contract_autopsy_2026-05-11.md`
- Pong audit:
  `docs/working/training/pong_replication_failure_audit_2026-05-11.md`
