# LightZero Official Atari Settings Audit - 2026-05-09

Scope: official LightZero Atari/Pong MuZero settings versus our GPU512,
GPU1024, and GPU2048 official Pong controls. No code changes, no training, no
pytest.

## Short Answer

Our GPU512/1024/2048 controls prove that the stock LightZero Atari Pong path
runs on Modal with a real ALE Pong env, CUDA, checkpoints, and no eval fallback.
They do not use LightZero's recommended Atari training scale.

The official stock Atari Pong MuZero example is roughly a `500,000` env-step
run with `8` collector envs, `3` evaluator envs, `50` MCTS simulations,
`batch_size=256`, four-frame visual input, `game_segment_length=400`,
`learning_rate=0.2`, `target_update_freq=100`, `eval_freq=2000`, and a
`1,000,000` transition replay buffer.

Our controls used `512`, `1024`, or `2048` env steps, one collector env, one
evaluator env, `2` simulations, `batch_size=8`, `update_per_collect=1`,
`game_segment_length=16`, 256-step train/eval episode caps for the larger two
rungs, `eval_freq=1`, and the same four-frame conv model. The run called
GPU2048 is only `0.41%` of the official `500,000` env-step budget. It is a
smoke test, not a real Atari learning run.

Plain conclusion: GPU2048 may look like "twice GPU1024", but it is still too
small and too patched to expect stable Pong signal. Worse, several patched
knobs change the meaning of the run: tiny search, tiny batch, only one env,
very short game segments, forced one update per collect, one-episode evals, and
fewer learner updates than the official target-network update period.

## Primary Upstream Evidence

Official quick start says Pong MuZero is run with:

```text
python3 -u zoo/atari/config/atari_muzero_config.py
```

Source: https://opendilab.github.io/LightZero/tutorials/installation/installation_and_quickstart.html

Upstream `zoo/atari/config/atari_muzero_config.py` defaults to:

| Setting | Official value |
| --- | ---: |
| env | `PongNoFrameskip-v4` |
| env type | `atari_lightzero` |
| entry | `train_muzero` |
| collector envs | `8` |
| collect episodes per cycle | `8` |
| evaluator envs | `3` |
| evaluator episodes | `3` |
| num simulations | `50` |
| update per collect | `None` |
| replay ratio | `0.25` |
| batch size | `256` |
| max env steps | `500,000` |
| frame stack | `4` |
| observation | `(4, 64, 64)`, grayscale |
| model | conv MuZero |
| CUDA | `true` |
| game segment length | `400` |
| learning rate | `0.2` |
| target update freq | `100` |
| eval freq | `2000` |
| replay buffer size | `1,000,000` |

Source: https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_config.py

Upstream `zoo/atari/config/atari_muzero_segment_config.py` is also an official
Atari/Pong path. It keeps `8` collector envs, `3` evaluator envs, `50`
simulations, `batch_size=256`, `max_env_step=500,000`, and `learning_rate=0.2`,
but uses `train_muzero_segment`, `num_segments=8`, `game_segment_length=20`,
and `train_start_after_envsteps=2000`.

Source: https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_segment_config.py

LightZero's config docs define the relevant knobs in the expected way:
collector/evaluator env counts control parallel collection/evaluation,
`game_segment_length` is the collection segment length, `update_per_collect`
is model updates after collection, `num_simulations` is MCTS simulations,
`eval_freq` is evaluation frequency, `replay_buffer_size` is replay capacity,
and `target_update_freq` is how often the target network is updated.

Source: https://opendilab.github.io/LightZero/tutorials/config/config.html

The published OpenDILab Hugging Face model card for
`PongNoFrameskip-v4-MuZero` is another primary-ish upstream artifact. It trains
the agent for `500,000` steps and reports mean reward `20.4 +/- 0.49`. Its
configuration is the heavier agent-style Pong config: `8` collectors, `3`
evaluators, `game_segment_length=400`, `update_per_collect=1000`,
`batch_size=256`, `learning_rate=0.2`, `target_update_freq=100`,
`num_simulations=50`, `eval_freq=2000`, and `replay_buffer_size=1,000,000`.

Source: https://huggingface.co/OpenDILabCommunity/PongNoFrameskip-v4-MuZero

## Local Evidence Read

Local source and docs reviewed:

- `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py`
- `src/curvyzero/infra/modal/lightzero_pong_dry_config_smoke.py`
- `docs/experiments/2026-05-09-modal-lightzero-pong-gpu512-control.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-gpu1024-control.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-gpu2048-control.md`
- `docs/working/lightzero_official_visual_pong_pattern_2026-05-09.md`
- local upstream checkout at `/tmp/lightzero-src`

The Modal wrapper patches the official non-segment
`zoo.atari.config.atari_muzero_config` before calling `lzero.entry.train_muzero`.
The GPU control commands passed `--batch-size 8`, `--game-segment-length 16`,
and did not pass `--num-simulations`, so they inherited the wrapper default
`num_simulations=2`. They also inherited `collector_env_num=1`,
`evaluator_env_num=1`, and `update_per_collect=1`.

## Official Versus Our Controls

| Setting | Official `atari_muzero_config` | Our GPU512 | Our GPU1024 | Our GPU2048 |
| --- | ---: | ---: | ---: | ---: |
| max env steps | `500,000` | `512` | `1024` | `2048` |
| share of official env steps | `100%` | `0.10%` | `0.20%` | `0.41%` |
| max train iter requested | effectively huge | `4` | `8` | `16` |
| checkpoint reached | not comparable | `iteration_4` | `iteration_4` | `iteration_8` |
| collector envs | `8` | `1` | `1` | `1` |
| evaluator envs | `3` | `1` | `1` | `1` |
| evaluator episodes | `3` | `1` | `1` | `1` |
| collect episodes per cycle | `8` | `1` | `1` | `1` |
| num simulations | `50` | `2` | `2` | `2` |
| update per collect | `None` with `replay_ratio=0.25` | `1` | `1` | `1` |
| batch size | `256` | `8` | `8` | `8` |
| replay buffer size | `1,000,000` | unchanged | unchanged | unchanged |
| game segment length | `400` | `16` | `16` | `16` |
| frame stack | `4` | `4` | `4` | `4` |
| observation | `(4,64,64)` | same | same | same |
| learning rate | `0.2` | `0.2` | `0.2` | `0.2` |
| target update freq | `100` | `100` | `100` | `100` |
| eval freq | `2000` | `1` | `1` | `1` |
| train/eval episode cap | not capped in config | `128` eval cap recorded | `256` | `256` |

The frame stack, visual model, action space, learning rate, target update
frequency, replay buffer capacity, Atari env id, and CUDA path are still close
to official. Most other scale knobs are not.

## Control Results

GPU512:

- trained on an L4;
- reached `iteration_4`;
- final eval used all six actions;
- 128-step eval return was `-2.0`;
- 256-step baseline eval return was later measured as `-5.0`;
- no fallback was used.

GPU1024:

- trained on an L4;
- requested `max_train_iter=8`, but env-step cap stopped at `iteration_4`;
- final 256-step eval used all six actions;
- got one positive Pong reward;
- returned `-3.0`;
- no fallback was used.

GPU2048:

- trained on an L4;
- requested `max_train_iter=16`, but env-step cap stopped at `iteration_8`;
- sampled eval checkpoints used only actions `0`, `1`, and `2`;
- got no positive rewards in the fixed 256-step evals;
- `iteration_0`, `iteration_4`, and `iteration_8` all returned `-6.0`;
- no fallback was used.

Plain read: GPU1024 had a small real signal, but GPU2048 did not confirm it.
That is normal at this size. It is a noisy, undertrained control surface.

## Why 2048 Env Steps Is Too Small

`2048` env steps is less than half of one percent of the official `500,000`
step Atari Pong budget. The published model-card path also trains at `500,000`
steps before claiming strong Pong reward.

The control also generated very little learner work. GPU2048 reached
`iteration_8`. With `target_update_freq=100`, the official target-network
period is much longer than the whole run. So a setting that matters in real
training barely activates here.

The search budget is also tiny. Official Atari uses `50` MCTS simulations; our
controls used `2`. That is `4%` of the official search depth per action.
Weak/tied roots and unstable action choice are expected.

The learner batch is tiny. Official Atari uses `batch_size=256`; our controls
used `8`. Keeping `learning_rate=0.2` while shrinking the batch by `32x` makes
updates noisier than the official tuning target.

The replay data is tiny. The replay buffer can hold `1,000,000` transitions,
but a GPU2048 run can only put a few thousand steps into it. A huge buffer
capacity does not help when the buffer is barely populated.

The evaluator is tiny. Official Atari evaluates with `3` evaluator envs and
`3` episodes. Our controls used one evaluator env and one episode, then
separately ran fixed 256-step no-fallback evals. One lucky or unlucky point can
move the whole readout.

The episode cap is short. Pong can be long, and the official config does not
use the 256-step cap we used for these controls. Our eval windows are useful
for quick comparisons, but they are not full-game quality estimates.

## Why 2048 May Also Be Misconfigured

The run is not only small; it changes the balance of the official recipe.

`update_per_collect=1` is a big semantic change. In the official config,
`update_per_collect=None` and `replay_ratio=0.25` let LightZero compute updates
from collected data. In the heavier agent config, Pong uses
`update_per_collect=1000`. Our forced `1` update per collect is a smoke choice,
not an Atari recommendation.

`game_segment_length=16` is much shorter than the official non-segment value
`400`. It is even shorter than many real Pong point sequences. Short segments
are fine for proving the trainer runs, but they may starve MuZero of useful
long-horizon targets.

`eval_freq=1` plus `save_ckpt_after_iter=1` makes sense for checkpoint
diagnostics, but it is not official training cadence. It can make a tiny run
look like a checkpoint curve even though almost no learning happened between
checkpoints.

One collector env removes the official parallel data shape. Official Atari
collects from `8` envs and `8` episodes per collection cycle. One env is more
sensitive to seed, early point timing, and accidental action bias.

The unchanged `learning_rate=0.2` and `target_update_freq=100` are official
only in the context of much larger batches, many more updates, and a much
larger replay stream. Leaving them unchanged in an 8-update smoke is not wrong
mechanically, but it is not a faithful quality setting.

## Expected Scale Before Seeing Signal

For infrastructure signal, our current controls are enough:

- env creation works;
- ROM image works;
- GPU/CUDA path works;
- checkpoints are mirrored;
- `MuZeroPolicy.eval_mode.forward` works without fallback.

For policy-learning signal, expect at least thousands to tens of thousands of
env steps before the curve becomes interpretable, and treat anything below the
official segment warmup of `2000` env steps as mostly plumbing. The official
recommended scale is `500,000` env steps. A serious intermediate rung should
move multiple knobs toward official shape, not only double `max_env_step`.

The clean next interpretation ladder is:

| Rung | What it can prove |
| --- | --- |
| `512` to `2048` tiny controls | plumbing, checkpoint loading, no-fallback eval, rough action diversity |
| `8192` to `20000` steps, repeated seeds | whether tiny positive rewards repeat or vanish |
| larger search, e.g. `16` or `25` sims | whether MCTS roots stop being weak/tied |
| larger batch, e.g. `64` or `128` | whether learner noise drops |
| several collector/evaluator envs | whether seed/point variance drops |
| official-like `50` sims, `batch_size=256`, many envs | actual Atari comparison |
| `500,000` env steps | comparable to LightZero's stock Pong recommendation |

## Bottom Line

GPU2048 failing to improve over GPU1024 is not strong evidence that official
LightZero Atari Pong is broken. It is stronger evidence that our control is
still a smoke-sized, heavily patched training run.

The biggest differences from official are scale and balance: `2048` steps
instead of `500,000`, `2` simulations instead of `50`, one env instead of
`8/3`, batch `8` instead of `256`, one update per collect instead of official
auto replay-ratio behavior, `16`-step segments instead of `400`, and an eval
cadence built for diagnostics rather than quality. Read GPU2048 as a useful
negative control, not as a failed official reproduction.
