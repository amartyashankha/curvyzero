# LightZero Reproduction Parity Plan - 2026-05-09

Scope: reproduction/parity research only. No code changes beyond this note. No
pytest. I read the local working/experiment docs first, then the local
LightZero source snapshot at `/tmp/lightzero-src`.

## Short Answer

The closest already-working LightZero control for our Pong/CurvyTron problem is
stock visual Atari Pong:

- `zoo.atari.config.atari_muzero_config`
- `PongNoFrameskip-v4`
- `lzero.entry.train_muzero`
- conv visual MuZero, four 64x64 grayscale frames, six Atari actions

That is the only official control that exercises the visual Atari/Pong stack:
ALE/Gym wrappers, ROMs, OpenCV preprocessing, frame stacking, conv model, and
Pong rewards.

The closest sparse/delayed-reward controls are TicTacToe and Connect4 bot mode,
but they are secondary. They prove LightZero can run sparse terminal-outcome
MuZero on Modal. They do not prove visual Pong or our custom Pong is learnable.

Our custom Pong runs are currently below the scale where I would expect a clean
MuZero signal. The new oracle fact makes this sharper: in states where `down`
immediately wins, a two-simulation root gave visits `[1, 1, 0]`. That means the
target generator assigned zero policy-target mass to the winning action. This is
not a meaningful failure of Pong learning; it is a weak-search target failure.

## What Counts As A Control

Use three control classes, in this order.

| Control | What it proves | What it does not prove |
| --- | --- | --- |
| Stock Atari Pong | The official visual Pong path can create envs, train, load checkpoints, and eval through LightZero. | Our custom Pong wrapper is correct. |
| CartPole | The simplest LightZero MuZero trainer can run on our Modal image. | Sparse reward, visual control, Pong dynamics. |
| TicTacToe / Connect4 bot mode | Sparse terminal-outcome LightZero examples can run. | Pixel/raster Pong, frame history, continuous-ish paddle timing. |

Do not use board games as the primary parity answer. They are useful for
terminal reward and final-outcome target settings only.

## Official Default Scale

These are the local LightZero defaults from `/tmp/lightzero-src`:

| Example | Sims | Collection | Updates | Batch | Train budget |
| --- | ---: | --- | --- | ---: | ---: |
| CartPole MuZero | 25 | `collector_env_num=8`, `n_episode=8` | `update_per_collect=100` | 256 | `1e5` env steps |
| Atari Pong MuZero | 50 | `collector_env_num=8`, `n_episode=8` | `update_per_collect=None`, `replay_ratio=0.25` | 256 | `5e5` env steps |
| Atari segment MuZero | 50 | `num_segments=8`, `game_segment_length=20` | `update_per_collect=None`, `replay_ratio=0.25`, starts after 2000 env steps | 256 | `5e5` env steps |
| TicTacToe bot MuZero | 25 | `collector_env_num=8`, `n_episode=8` | `update_per_collect=50` | 256 | `2e5` env steps |
| Connect4 bot MuZero | 50 | `collector_env_num=8`, `n_episode=8` | `update_per_collect=50` | 256 | `5e5` env steps |

Tiny local smokes with `num_simulations=2`, `batch_size=4/8`, one collector,
and `update_per_collect=1` are valid execution tests. They are not valid
learning-signal tests.

Plain rule: `2` simulations is a smoke setting. `8` simulations is still a
debug setting. `25` simulations is the first "normal enough" LightZero MuZero
setting for small controls. `50` is the official Atari/Connect4 default.

## Custom Pong Implication

LightZero MuZero trains the policy head toward MCTS root visit distributions,
not toward the action that was finally executed by temperature, epsilon, or
random warmup.

So if a custom Pong root has visits `[1, 1, 0]` when `down` wins immediately,
then the stored target policy says:

```text
up: some mass
stay: some mass
down: zero mass
```

More replay of that exact target will not teach `down`. Epsilon can execute
`down`, and random warmup can visit `down` states, but neither one turns the
root policy target into an imitation label.

This is the current parity diagnosis: our custom Pong setup is close enough to
exercise LightZero plumbing, but not close enough to judge learning until root
targets are generated at a normal simulation/update scale and checked against
known scoreable states.

## Minimum Reproduction Ladder

Do not make this a campaign. Run the smallest ladder that can separate
"framework works", "official Pong can produce a weak signal", and "custom Pong
targets are sane".

### Rung 0: Mechanical Smokes

Status: mostly already done.

Keep these as historical controls, not quality gates:

- stock CartPole tiny train;
- stock TicTacToe tiny train;
- stock Connect4 tiny train;
- stock Atari Pong env reset/step;
- stock Atari Pong tiny train/load/eval;
- custom Pong config/import/train/checkpoint/eval plumbing.

Expected signal: imports, reset/step, checkpoint writes, strict load, eval path,
artifact refs. No policy-quality claim.

### Rung 1: Official Atari Pong Runtime Control

Goal: prove stock visual Pong can run beyond one or two toy checkpoints, still
bounded.

Minimum run:

```text
source: zoo.atari.config.atari_muzero_config
entry: train_muzero
env_id: PongNoFrameskip-v4
compute: cheap GPU if available
collector_env_num: 1
n_episode: 1
evaluator_env_num: 1
num_simulations: 25 first, 50 only if 25 is mechanically clean
batch_size: 64 minimum; 256 is official
updates: prefer replay_ratio=0.25 / update_per_collect=None if wrapper supports it
max_env_step: 4096 minimum for a tiny curve, not 128/512
max_train_iter: enough to emit at least 3 post-init checkpoints
eval cap: at least 256 real Atari steps
```

Expected minimum signal:

- no fallback actions;
- action histogram is not constant across all checkpoints;
- later checkpoint differs from `iteration_0` on logits/visits/actions;
- nonzero Atari rewards are observed under the eval cap;
- do not require positive Pong return yet.

Stop if this cannot run at `25` simulations. That would mean our official
control is still only an execution smoke.

### Rung 2: Official Sparse Control At Normal Search

Goal: keep one cheap sparse terminal-reward sanity check at a normal search
scale.

Use TicTacToe bot mode first, because it has an existing tiny test-shaped path.

Minimum run:

```text
source: zoo.board_games.tictactoe.config.tictactoe_muzero_bot_mode_config
entry: train_muzero
num_simulations: 25
update_per_collect: 50
batch_size: 64 minimum; 256 is official
collector_env_num: 1 or 2
n_episode: match collector count
td_steps: 9
discount_factor: 1
reward/value support: (-10, 11, 1)
max_env_step: 512 or 1024 for bounded control
```

Expected minimum signal:

- complete run with checkpoints;
- terminal rewards appear;
- policy/value losses are finite;
- checkpoint curve exists.

This does not need to become strong TicTacToe. It only anchors that sparse
terminal targets work when search/update scale is not toy-small.

### Rung 3: Custom Pong Root-Target Parity Probe

Goal: decide whether custom Pong targets are sane before another training run.

Use a fixed set of scoreable custom Pong states, including states where the
oracle says `down` immediately wins.

For each state, run LightZero MCTS with the current custom wrapper at:

```text
num_simulations: 2, 8, 16, 25, 50
decision modes: eval/no-noise and collect/noise if available
```

Record:

- oracle-best action;
- policy logits;
- root visits;
- selected action;
- top-1 minus top-2 visit margin;
- exact tie rate;
- searched value;
- whether the winning action gets nonzero visit mass.

Minimum pass:

- at `25` simulations, known immediate-win actions get nonzero visit mass;
- at `25` or `50`, most known immediate-win states prefer the oracle action or
  at least rank it top two;
- exact max ties are not the dominant explanation;
- the result is state-dependent, not one action everywhere.

Minimum fail:

- `down` immediate-win states still look like `[1, 1, 0]`, `[many, many, 0]`,
  or any zero-mass winning action at `25/50` simulations;
- higher simulations only make a wrong single action more confident;
- values remain flat across known win/loss states.

If this fails, do not scale custom Pong training. The issue is target quality,
value calibration, support scale, observation/model capacity, or wrapper
semantics, not lack of a longer run.

### Rung 4: Custom Pong Minimum Train Signal

Run this only after Rung 3 passes.

Minimum custom Pong train:

```text
feature_mode: tabular_ego first
opponent/reset: fixed scoreable fixture or contact_pressure, not broad random-only
pong_episode_max_steps: explicit, fixed, e.g. 120
num_simulations: 25
collector_env_num: 2 minimum
n_episode: 2 or 4
batch_size: 64 minimum
update_per_collect: 25 minimum; 50 is closer to sparse controls
td_steps: final-outcome horizon for sparse reward, or clearly documented shorter value target
discount_factor: 1 for true sparse terminal outcome
reward/value support: narrow and explicit for `-1/0/+1`, e.g. `(-5, 6, 1)`
max_env_step: 2048 minimum for first rung, 4096 only if the first curve moves
```

Expected minimum signal:

- checkpoint curve includes `iteration_0`, midpoint, final, and `ckpt_best`;
- held-out scoreable-state MCTS root targets improve, not only trainer returns;
- action entropy stays non-degenerate;
- known oracle-best `down` states get visit mass and sometimes selected actions;
- independent scorecards improve on at least two checkpoints, not only
  `ckpt_best`.

Stop if the checkpoint curve is flat or if held-out MCTS still assigns zero
mass to known winning actions.

## How To Know Custom Pong Is Close Enough

Custom Pong is close enough to official LightZero controls when all of these
are true:

- The wrapper contract matches non-board LightZero usage:
  fixed action space, all-ones action mask, `to_play=-1`, ego reward sign,
  explicit horizon, explicit feature mode.
- A known-state oracle agrees with MCTS root targets at normal search scale:
  `25` or `50` simulations, not `2`.
- The training run uses update pressure in the same rough family as official
  controls: at least `update_per_collect=25/50` or replay-ratio accounting, not
  `1`.
- Scorecards are provenance-locked: checkpoint SHA, feature mode, horizon,
  reset profile, opponent, support ranges, state key, and simulation count are
  recorded together.
- Improvement appears in independent eval and root diagnostics, not only in
  trainer-side exploratory actions.

Custom Pong is not close enough when:

- a winning action gets zero root visits at `25/50` simulations;
- the setup still uses `num_simulations=2` as target generation for learning
  claims;
- train actions diversify but stored visit targets stay collapsed;
- evaluation horizon or feature mode can silently differ from training;
- single-frame raster is used without velocity/history and then compared to
  official Atari frame-stack behavior.

## Recommended Next Step

Do not run another longer custom Pong sparse campaign yet.

Run one bounded root-target parity probe on scoreable custom Pong states at
`2/8/16/25/50` simulations. Treat `25` as the first real threshold. If the
oracle-winning actions still get zero or near-zero visit mass there, fix target
quality before training. If they become state-dependent at `25/50`, then run
the custom Pong minimum train rung with `num_simulations=25` and
`update_per_collect>=25`.

In parallel, keep the official Atari Pong control separate. Its job is to prove
that our Modal/LightZero visual Pong path can produce checkpoint curves under
normal-ish MCTS settings. It should not be used to excuse or condemn the custom
Pong wrapper until the custom root targets pass the oracle check.

## Sources Used

Local docs:

- `docs/working/lightzero_official_visual_pong_pattern_2026-05-09.md`
- `docs/working/lightzero_official_example_pattern_choice_2026-05-09.md`
- `docs/working/lightzero_source_setup_audit_2026-05-09.md`
- `docs/working/lightzero_muzero_target_semantics_2026-05-09.md`
- `docs/working/lightzero_pong_sparse_training_scale_ladder_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_root_cause_red_team_2026-05-09.md`
- `docs/working/training_state_index_2026-05-09.md`
- official control experiment notes under `docs/experiments/2026-05-09-modal-lightzero-*`

Local LightZero source:

- `/tmp/lightzero-src/zoo/atari/config/atari_muzero_config.py`
- `/tmp/lightzero-src/zoo/atari/config/atari_muzero_segment_config.py`
- `/tmp/lightzero-src/zoo/classic_control/cartpole/config/cartpole_muzero_config.py`
- `/tmp/lightzero-src/zoo/board_games/tictactoe/config/tictactoe_muzero_bot_mode_config.py`
- `/tmp/lightzero-src/zoo/board_games/connect4/config/connect4_muzero_bot_mode_config.py`

No new web lookup was needed for this pass because the local docs and local
LightZero source already contained the relevant primary configuration facts.
