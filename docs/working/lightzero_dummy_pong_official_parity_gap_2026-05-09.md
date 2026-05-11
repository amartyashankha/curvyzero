# LightZero Dummy Pong Official Parity Gap - 2026-05-09

Scope: compare the project custom dummy Pong LightZero setup against official
Atari Pong and official sparse board-game LightZero patterns. No training and
no pytest were run for this note.

## Short Answer

Action mapping is no longer a leading suspect. In custom dummy Pong,
`ACTION_LABELS = ("up", "stay", "down")`, action id `2` maps to `down`, and
all three actions are legal.

The immediate mismatch is target quality. The new oracle result shows that, in
down-needed states, train-time MCTS with `num_simulations=2` can produce root
visits `[1, 1, 0]`. That means the policy target gives zero mass to the winning
action. If the learner is trained on that row, it is being taught not to choose
`down`.

So the smallest change that brings custom Pong closer to normal LightZero is:
raise train-time MCTS simulations and fix the real compiled support scale.
If only one change can go first, raise train-time simulations first, because it
directly fixes wrong policy targets. But the next serious custom Pong run
should do both, because `support_scale=300` is also badly mismatched to
`-1/0/+1` rewards.

## Closest Official Patterns

### Official Atari Pong

Official Atari Pong is a visual benchmark setup:

| Surface | Official Atari Pong | Custom dummy Pong |
| --- | ---: | ---: |
| Observation | stacked image frames | `10` tabular floats, or one flat `9x15` raster |
| Model | conv MuZero | MLP MuZero |
| Actions | Atari Pong action space, usually `6` | `3`: up, stay, down |
| Train-time search | commonly `50` simulations | often `2`, sometimes `8` |
| Batch/update scale | large replay and batch settings | tiny diagnostic runs |
| Episode/data scale | long Atari runs | 64 to 4096 step diagnostic caps |

Official Atari Pong is not just "Pong with LightZero." It gets far better
search targets, visual history, conv capacity, and much more data.

### Official Sparse Board Games

Official TicTacToe/Connect4/Gomoku bot-mode configs are closer to dummy Pong's
sparse terminal reward:

| Surface | Official sparse board games | Custom dummy Pong |
| --- | ---: | ---: |
| Reward | delayed terminal outcome | delayed terminal score |
| Opponent | hidden bot step can be inside env | hidden scripted/checkpoint opponent |
| `td_steps` | reaches final outcome | sometimes patched, not original default |
| Discount | `1` | sometimes patched to `1` |
| Search | normal configs use many more sims | `2` in key failed custom runs |
| Updates | much larger batch/update volume | tiny diagnostic volume |
| Legal actions | action mask changes | all three actions legal |

The useful borrow is final-outcome target hygiene plus enough search to make
the policy target useful. The bad borrow would be pretending dummy Pong is a
turn-based board game with varied legal actions.

## Ranked Gaps

### 1. Train-Time MCTS Target Quality

This is now the top gap.

The new oracle says down-needed states can receive root visits `[1, 1, 0]`
with `num_simulations=2`. The winning action gets zero visits, so the
normalized policy target has zero probability on the action that solves the
state.

That explains the no-learning shape better than an action-map bug:

- trainer-side exploration can physically play `down`;
- baseline policies can emit `down`;
- independent scorecards can count `down`;
- learned MCTS rows still collapse to no `down`;
- now we have direct evidence that low-sim search can create wrong labels.

This ranks above support scale because it corrupts the policy target at the
root. A perfect value/reward support cannot teach `down` from a policy target
that assigns `down` zero mass in the decisive state.

Smallest change:

- stop using `num_simulations=2` for custom Pong learning;
- use at least `8` as the small diagnostic floor;
- prefer `16` for the next serious target-quality check if cost is acceptable;
- keep eval simulations matched or explicitly record train/eval sim mismatch.

### 2. `support_scale=300` Mismatch

This is the second-ranked gap and still a real config bug candidate.

Dummy Pong rewards and values are small: terminal `+1/-1`, nonterminal `0`,
and timeout `0`. Existing notes indicate our requested reward/value support
ranges can appear in patch surfaces while the compiled LightZero MuZero model
still uses `support_scale=300`.

That is far from normal for a tiny sparse toy. It can dilute gradients and make
value/reward calibration harder. It is probably not the only cause of zero
`down`, but it can make the low-sim target problem worse by keeping roots weak
and nearly tied.

Smallest change:

- expose and patch the actual compiled `policy.model.support_scale`;
- set it to a small value for dummy Pong, for example `1` if only final
  win/loss value is represented, or a small safety range like `5`;
- log the compiled policy config, not just the requested patch surface.

### 3. Official Search Scale

Official working configs do not rely on two simulations for the main policy
target. Atari and board-game references commonly use far more search.

This is closely related to gap 1, but it is worth separating:

- gap 1 is the direct oracle failure: `[1, 1, 0]` gives the winning action
  zero target mass;
- this gap is the parity argument: official setups avoid that regime by using
  substantially more search.

Smallest change:

- make `num_simulations=8` the minimum for custom Pong train attempts;
- reserve `2` for import/config smokes only;
- treat `16` as the first serious diagnostic setting;
- do not compare a `2`-sim checkpoint to official LightZero quality.

### 4. Observation And Model Difference

Custom `tabular_ego` is probably Markov enough for the toy: it includes paddle
positions, ball relative position, ball velocity, ball row, and step fraction.
But it is not official visual Pong.

The `raster_flat` path is weaker than both: one unstacked flat grid has no
velocity and is still fed to an MLP, not a conv model.

Smallest change:

- keep `tabular_ego` for debugging target quality;
- do not use `raster_flat` as a visual Pong claim until it has stacked frames
  or velocity channels;
- postpone conv dummy Pong until low-sim target quality and support scale are
  fixed.

### 5. Sparse Final-Outcome Settings

Board-game-like settings matter: `td_steps` should reach the outcome and
`discount_factor` should be `1` for final-outcome learning. Some custom runs
already patched this, so it is no longer the freshest root cause.

Smallest change:

- keep explicit `pong_episode_max_steps`;
- keep `td_steps` aligned to that horizon for sparse terminal reward tests;
- keep `discount_factor=1.0`;
- keep `num_unroll_steps` explicit and logged.

### 6. Replay And Update Volume

Official configs use much more data and update volume. But prior custom runs
already showed that more updates or simple exploration alone did not fix heldout
quality.

This ranks below target quality because scaling bad targets can make collapse
more confident.

Smallest change:

- do not scale update count before fixing train-time root targets;
- after sims/support are fixed, use modest larger batch/update settings as a
  second step;
- keep independent heldout MCTS scorecards as the quality gate.

### 7. Collector/Evaluator And Telemetry Differences

Trainer-side sidecar rows are not final checkpoint quality. Independent MCTS
scorecards remain the authority.

Smallest change:

- keep seed histograms;
- split collector rows from evaluator rows if possible;
- always score `iteration_0`, final, and `ckpt_best` independently.

### 8. Action Mapping, Legal Actions, And Truncation

This is now low suspicion.

Action id `2` is `down`, `down` is legal, the wrapper action mask is all ones,
and baseline policies can produce `down`. Terminal and truncation handling
still need to be logged carefully, but they do not explain `[1, 1, 0]` on a
down-needed state.

Smallest change:

- keep action-map assertions in debug summaries;
- keep terminal vs truncation counts in scorecards;
- do not spend the next run on another action inversion check unless a new
  contradiction appears.

## Recommended Smallest Run Change

Use both changes in the next meaningful custom Pong run:

1. Increase train-time `num_simulations` from `2` to `8` or `16`.
2. Patch the actual compiled `policy.model.support_scale` to a small dummy
   Pong value and log it.

If forced to choose only one, choose more train-time MCTS simulations first.
The oracle showed a direct label-quality failure: the winning action can get
zero target mass. Support scale is important, but it cannot repair a policy
target that never names the winning action.

The smallest normal-LightZero-aligned custom Pong configuration is:

```text
feature_mode=tabular_ego
num_simulations=8 minimum, 16 preferred for diagnosis
support_scale=1 or small range such as 5, verified in compiled config
td_steps=pong_episode_max_steps
discount_factor=1.0
num_unroll_steps explicit
reward stays sparse +1/-1/0
independent MCTS scorecard after training
```

Do not claim official Atari parity from this. It is a custom sparse-control
diagnostic that borrows the most relevant LightZero habits: enough search to
make policy targets sane, value support matched to reward scale, and
final-outcome target settings.
