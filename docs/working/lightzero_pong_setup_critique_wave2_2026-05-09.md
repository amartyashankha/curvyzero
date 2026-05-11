# LightZero Pong Setup Critique Wave 2 - 2026-05-09

Scope: external-pattern critique against official LightZero/MuZero patterns.
Read local docs/source first, then checked primary sources only. No pytest was
run.

Primary web sources checked:

- LightZero quick start:
  https://opendilab.github.io/LightZero/tutorials/installation/installation_and_quickstart.html
- LightZero upstream Atari Pong MuZero config:
  https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_config.py
- LightZero upstream Atari Pong segment MuZero config:
  https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_segment_config.py
- LightZero upstream TicTacToe bot-mode MuZero config:
  https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/board_games/tictactoe/config/tictactoe_muzero_bot_mode_config.py
- MuZero paper page:
  https://www.nature.com/articles/s41586-020-03051-4

## Short Read

Our current custom dummy Pong lane is real LightZero MuZero plumbing, but it is
not close to the official working Pong setup. It is also only partly close to
the official sparse board-game setup.

Official Atari Pong is visual: stacked frames, conv model, Atari action space,
large replay/update budget, many simulations, and long training. Official
board games are sparse: final-outcome targets, discount `1`, legal-action
masks, board tensors, and much larger update volume. Our custom lane is a
small MLP on a 10-float ego feature row, with a hidden scripted opponent,
tiny caps, and repeated independent scorecard collapse.

The practical conclusion is now stricter: do not run the same sparse tabular
config longer, do not repeat higher update/replay alone, and do not repeat
simple random-warmup/epsilon collection under the same sparse target. The next
dummy Pong run should make the target scoreable through reset/opponent
curriculum while keeping `env.step()` reward sparse.

## Closest Official Patterns

### Stock Atari Pong

Official LightZero Pong uses `PongNoFrameskip-v4`, `atari_lightzero`,
`train_muzero` or `train_muzero_segment`, conv MuZero, and observation shape
`(4, 64, 64)`. Upstream defaults are much larger than our smokes:

| Surface | Official Atari Pong | Our custom dummy Pong |
| --- | ---: | ---: |
| Observation | `4x64x64` grayscale frames | `10` tabular floats, or flat raster smoke |
| Model | conv MuZero | MLP MuZero |
| Actions | Atari action space, `6` for Pong | `3`: up/stay/down |
| Collector | `8` envs, `n_episode=8` or `num_segments=8` | usually `1` env, `n_episode=1/2/4` |
| Search | `num_simulations=50` | mostly `8`; some tiny `2/4` |
| Batch | `256` | `32/64` in sparse probes |
| Replay/update | `replay_ratio=0.25`, `update_per_collect=None` | explicit `update_per_collect=8/25` in sparse probes |
| Train budget | `500000` env steps | `1024/2048` sparse probes |
| Segment path | `game_segment_length=20`, train start after `2000` env steps | episode collector; segment not replicated |

What this means: our tabular dummy Pong is not a stock Pong replication. It is
a custom control toy. The official visual Pong smoke proves the ALE/ROM/train
path can run and load checkpoints, but not that our dummy Pong learner should
work under tiny MLP caps.

### Sparse Board Games

Official TicTacToe/Connect4 bot-mode patterns are useful because they show how
LightZero handles sparse delayed outcomes:

| Surface | Official sparse board games | Our sparse dummy Pong |
| --- | ---: | ---: |
| Env type | `board_games` | `not_board_games` |
| Action type | `varied_action_space` | `fixed_action_space` |
| Legal actions | changing action mask | all three actions legal |
| Target style | final outcome, large `td_steps` | patched to `td_steps=120` in sparse probes |
| Discount | `1` | patched to `1` in sparse probes |
| Updates | `update_per_collect=50` | `8`, then `25` |
| Batch | `256` | `32`, then `64` |
| Budget | `200000` TicTacToe, `500000` Connect4 | `1024/2048` |

The useful borrow is final-outcome target hygiene: fixed horizon, `td_steps`
to outcome, discount `1`, and small reward/value supports. The bad borrow
would be switching to `board_games` or `varied_action_space`; our env hides the
opponent inside `step()` and always has three legal actions.

## Main Gaps

### 1. Observation/model mismatch

Our tabular lane is not visual Pong. It removes frame history and uses one
hand-built ego row. That can be a good toy, but it cannot validate the official
Atari visual pattern. The `raster_flat` smoke is also not official visual Pong:
it flattens a tiny raster into an MLP rather than using a conv stack.

Practical risk: the model may not get enough temporal or spatial signal to
make search roots state-sensitive. The repeated one-action eval collapse fits
that risk.

### 2. Tiny caps versus official scale

The sparse ladder did the right controlled tests, but it is still tiny. Rung 1
only doubled to `2048/32`, and UPC25 raised update pressure without changing
the objective. Official working configs use far more collection, replay, and
search.

This does not mean "just train longer." The pure 2x run failed, UPC25 failed,
and higher eval sims only changed the collapsed action. It means the current
tiny setup has not earned a scale-up.

### 3. Replay/update ratio is still not official-like

Official Atari uses `update_per_collect=None` with `replay_ratio=0.25`.
Official board games use `update_per_collect=50` and `batch_size=256`. Our
latest UPC25 run used more updates, but still with one collector env, tiny
batches, small data diversity, and the same sparse objective.

Practical risk: extra learner updates may overfit weak trajectories instead of
creating a better search policy. The UPC25 heldout curve already showed this:
final did not cleanly beat initialization, and `ckpt_best` was all-up.

### 4. Simple exploration is now a failed fix

The code now exposes `random_collect_episode_num`, epsilon collect, epsilon
schedule, and fixed temperature. The follow-up UPC25 epsilon-collect run used
`random_collect_episode_num=8`, epsilon collect `0.75 -> 0.25`, and
`fixed_temperature_value=0.5`. It diversified trainer-side actions to
`[288, 74, 64]`, but heldout `iteration_50` still collapsed to
`[1158, 0, 50]`, raw score stayed `-0.25` versus both
`lagged_track_ball_1` and `random_uniform`, and `ckpt_best` stayed all-up
`[806, 0, 0]`.

Practical read: exploration knobs can change collection behavior, but they did
not create a better heldout controller under the same sparse target. Stop
recommending simple exploration-only repeats as the next fix.

### 5. Reward targets are only half replicated

We correctly kept environment reward honest: terminal score `+1/-1`, zero
otherwise, timeout `0`. The sparse probe also replicated board-game-ish
`td_steps=120` and `discount_factor=1`. But survival/loss-delay is still only
telemetry, not a training target in LightZero.

Practical risk: the learner gets too little early score pressure before it can
discover ball-contact geometry. The best next curriculum is not reward shaping;
it is a scoreable reset/opponent distribution that preserves sparse score
reward.

This partly matches official MuZero/LightZero patterns and partly conflicts
with them:

- Match: LightZero already supports bot/opponent modes and custom env configs;
  official sparse board games train from final outcomes with `discount_factor=1`
  and large `td_steps`, not dense per-step shaping.
- Match: MuZero trains from trajectories produced by the environment and learns
  reward, value, and policy targets from those trajectories, so changing the
  data distribution is a valid curriculum mechanism.
- Conflict: official Atari Pong is a benchmark setup, not a hand-biased
  contact curriculum. A curriculum run must be labeled as curriculum, not as
  stock Pong replication.
- Conflict: biased near-contact starts can make the eval lie. Any curriculum
  must keep a normal-reset heldout scorecard against `random_uniform`,
  `lagged_track_ball_1`, and `track_ball`.

### 6. Policy eval API is better, but still needs discipline

The direct policy-head path is only a loader/action smoke. The independent MCTS
scorecard with strict `MuZeroPolicy` loading is the quality gate. That part is
now much more trustworthy.

Remaining eval gap: deterministic eval over weak roots can create action
histograms that look worse or different from collect mode. We should keep
first-N root diagnostics: logits, visits, selected action, tie count, and visit
margin.

### 7. Checkpoint curves are not healthy

The sparse ladder has exactly the curve shape we should reject:
`iteration_0` is often no worse than final, final collapses, and `ckpt_best`
can be more collapsed than final. Rung 1 did not improve heldout survival,
shaped return, raw score, or action entropy. UPC25 also failed the heldout
curve test.

## What Has Been Replicated

- Official visual Atari Pong can run a tiny train/load smoke on Modal.
- Official sparse TicTacToe and Connect4 bot-mode smokes can run on Modal.
- Custom dummy Pong LightZero training, checkpoint mirroring, strict MCTS
  loading, scorecards, seed telemetry, horizon split, and frozen-checkpoint
  opponent plumbing all work mechanically.
- Sparse target knobs are exposed: horizon, `td_steps`, discount, unroll steps,
  reward/value supports, update count, batch size, episode count.

## What Has Not Been Replicated

- No stock Atari Pong quality run or checkpoint curve has been replicated.
- No official visual observation/model pattern has been replicated for dummy
  Pong.
- No official-scale replay/search/update setup has been replicated.
- No healthy dummy Pong heldout checkpoint curve has been produced.
- No sparse dummy Pong run has shown non-degenerate heldout action entropy with
  better survival and score.
- UPC25 plus simple random warmup / epsilon collect did not improve heldout,
  despite improving trainer-side action diversity.
- No LightZero run has beaten the project-owned CEM-v2 geometry baseline.
- No project-owned Pong or Curvy MuZero trainer has run.

## Ranked Next 3 Experiments

### 1. Scoreable contact/angle curriculum

Run one LightZero MuZero dummy Pong curriculum that changes reset/opponent
distribution toward scoreable paddle-contact states while preserving sparse
environment reward:

- keep `env.step()` reward as `+1/-1/0`
- keep normal heldout resets separate
- use `lagged_track_ball_1` or another proven scoreable opponent/reset slice
- prefer states with imminent or learnable paddle contact and scoring pressure
- report train action diversity, but gate only on heldout MCTS scorecards

Official-pattern read: this matches LightZero custom-env/bot-mode flexibility
and board-game final-outcome target logic. It conflicts with stock Atari Pong
only if it is mislabeled as benchmark replication.

Falsified if heldout normal-reset scorecards still do not beat initialization
on `lagged_track_ball_1` or `random_uniform`, still collapse to one action, or
show contact/angle improvements that do not move raw sparse score.

### 2. Temporary auxiliary target, only if curriculum data is still too sparse

If the scoreable curriculum still produces too few useful targets, add a
clearly labeled temporary auxiliary target such as loss-delay or contact-choice
labels. Do not change promoted `env.step()` reward and do not use auxiliary
metrics as the quality claim.

Falsified if auxiliary metrics improve but heldout raw score, survival, and
action entropy do not improve against `random_uniform` and
`lagged_track_ball_1`.

### 3. Visual/raster model alignment probe

Stop mixing visual claims with tabular checkpoints. Run a small raster-only
lane with explicit `feature_mode=raster_flat`, matched checkpoint loading, and
either a conv model if feasible or a documented MLP raster baseline if not.
The goal is not quality first; it is to test whether richer observation removes
weak-root action collapse.

Falsified if first-N roots remain near-uniform, heldout eval still collapses to
one action, and survival/score do not beat the tabular sparse initialization
control.

## Bottom Line

The current setup has replicated LightZero mechanics, not LightZero Pong
success. The strongest practical gap is not one missing flag. It is the
combination of tiny data, weak sparse targets, non-official observation/model
shape, and deterministic eval over weak roots. Simple exploration/data
distribution knobs have now failed under UPC25. The next useful run is a
scoreable sparse curriculum through reset/opponent distribution, then the same
heldout scorecard discipline.
