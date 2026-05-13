# Things We Know Were Wrong

Purpose: one blunt list of mistakes so they do not get repeated.

## 1. Calling The Custom Two-Seat Path "Canonical"

Wrong because: it made an experimental adapter sound like the trusted trainer.

Reality: it solved action collection, but it bypassed stock `train_muzero`,
stock collector, native `GameSegment`, and `MuZeroGameBuffer`.

Fix: rename and document it as `custom_two_seat_adapter` or equivalent until it
feeds native replay or passes target parity.

## 2. Treating "Weights Changed" As Learning Evidence

Wrong because: a broken target builder can still update weights.

Reality: learning evidence needs a checkpoint curve with survival, outcome,
reward components, action distribution, and clear opponent source.

Fix: learning gates now require curves and trainer-path metadata.

## 3. Letting "Self-Play" Mean Too Many Different Things

Wrong because: fixed opponent, frozen opponent, recent checkpoint opponent,
same-policy two-seat, and centralized joint-action are different experiments.

Reality: each can be useful, but each proves a different thing.

Fix: every doc/run must name the opponent source and action semantics.

## 4. Dismissing Fixed/Frozen Too Strongly

Wrong because: fixed/frozen does not prove live two-seat self-play, but it may
be the cleanest stock-loop training route.

Reality: recent frozen checkpoints can act like a practical opponent curriculum
while keeping LightZero replay and target construction intact.

Fix: revive it as a stock-loop control/curriculum candidate, with honest labels
and opponent panels.

## 5. Scaling Before Replay/Target Semantics Were Proven

Wrong because: the highest-risk code was not Modal or GPU setup. It was the
custom learner data contract.

Reality: simultaneous action collection changed the replay problem. The target
builder needed parity tests before a large run.

Fix: native replay bridge or target parity gate before scaling two-seat again.

## 6. Leaving Stale Front Doors In Docs And Defaults

Wrong because: old handoffs and default CLI settings continued to point people
back to the failed lane.

Reality: cleanup must happen at the repo entry points, not only in a new audit.

Fix: patch docs/defaults/scripts that still say two-seat is the main trainer.

## 7. Letting Defaults Point At The Risky Path

Wrong because: a no-argument or casual Modal run can still route into
`two-seat-selfplay` and call it canonical.

Reality: defaults are part of the architecture. A dangerous default can undo a
good postmortem.

Fix: make custom two-seat explicit opt-in, or change the default to a dry/status
mode or a stock-loop control.

## 8. Hiding The GPU/CPU Split Behind "Use More GPU"

Wrong because: some slow parts are not neural-network compute.

Reality: model inference and learner work can use GPU, but env/render/replay
and much search bookkeeping remain CPU-visible.

Fix: every throughput doc should say which bucket is slow before recommending
more GPU.

## 9. Forgetting The Pong Lesson

Wrong because: Pong already showed that custom/debug training lanes can look
alive without giving credible learning evidence.

Reality: the credible Pong signal came from the stock/near-stock LightZero path
plus strict eval and survival-first curves after enough horizon.

Fix: use Pong as the process template for CurvyTron: stay close to stock
training, prove strict checkpoint eval, track survival/outcome curves, and do
not scale custom learner contracts before parity tests.

## 10. Passing Public Player Ids As Non-Board-Game `to_play`

Wrong because: the custom two-seat path appears to pass CurvyTron player ids
`0/1` into LightZero policy/replay fields where non-board-game MuZero rows are
expected to use a neutral `to_play` value, usually `-1`.

Reality: this is another reason the custom adapter may silently differ from the
stock Pong/Atari-style path. It is not the only problem, but it is a sharp
code-level red flag.

Fix: any future two-seat/native bridge work must assert the `to_play` contract
on tiny traces before training. Stock fixed/frozen and centralized-joint-action
controls should record their `to_play` semantics in run metadata.

## 11. Expecting Running Modal Pollers To Pick Up Local Edits

Wrong because: detached Modal jobs run a packaged code snapshot. Local edits do
not magically update old pollers or old spawned workers.

Reality: current-code GIF/eval changes need a redeployed poller, a separate
backfill job, or a fresh run. Old two-seat runs also do not have a trustworthy
resume path.

Fix: put code-version assumptions in run metadata and never rely on old
detached jobs picking up new local code.
