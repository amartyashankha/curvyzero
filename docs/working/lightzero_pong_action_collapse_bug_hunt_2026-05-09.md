# LightZero Pong Action Collapse Bug Hunt - 2026-05-09

Scope: side-lane bug hunt for why the latest LightZero dummy Pong checkpoint
chooses action index `2` zero times under independent MCTS eval and does not
improve. No pytest. I only wrote this doc.

## Short Read

The simple action mapping is not inverted or masked:

- `ACTION_LABELS = ("up", "stay", "down")`.
- `PongEnv._move_paddle()` uses `delta = action - 1`.
- Local smoke: `[(0, "up", -1), (1, "stay", 0), (2, "down", 1)]`.
- Baseline eval can emit `down`; local random-vs-random smoke had
  `{"random_uniform": [22, 23, 31]}`.

The current strongest read is that the loaded LightZero `model` really has a
bad action prior under our eval adapter. Diagnostics on real eval states showed
the 4096/64 `iteration_64:model` direct policy head picked `[48, 0, 0]`, and
MCTS samples picked `[8, 8, 0]`. Full independent MCTS scorecards then also
picked zero `down`.

So this is probably not "down is illegal". It is either:

1. a real learned-policy collapse, helped by weak/sparse training signal and
   misleading trainer-side telemetry; or
2. a still-unproven LightZero eval/load protocol mismatch that makes the
   independent adapter use a different eval model or sign convention than the
   trainer.

## Evidence From Latest Runs

Post deep-seed-fix 1024/16 run:

- Train run: `lz-dpong-20260509T154530Z-b049f29edb64`.
- Checkpoint: `iteration_16.pth.tar`.
- Trainer side seed diversity passed: 131 unique seeds in 148 rows.
- Trainer `player_0` actions: `[999, 853, 91]`; training did physically use
  action `2`.
- Paired independent MCTS learned rows:
  - vs `lagged_track_ball_1`: `[3140, 1854, 0]`
  - vs `random_uniform`: `[270, 539, 0]`
  - vs `track_ball`: `[3875, 2388, 0]`
- Player0-only independent MCTS learned rows:
  - vs `lagged_track_ball_1`: `[877, 549, 0]`
  - vs `random_uniform`: `[138, 261, 0]`
  - vs `track_ball`: `[2338, 1263, 0]`

Earlier 4096/64 run:

- Trainer-side sidecar said `535 / 43`, but 513 of 578 rows used
  `episode_seed=2`; those rows were `505 / 8`.
- Independent MCTS on `iteration_64:model` matched horizon `4096` and still
  chose zero `down`:
  - vs `random_uniform`: `[290, 219, 0]`
  - vs `lagged_track_ball_1`: `[6475, 2045, 0]`
  - vs `track_ball`: `[3335, 1284, 0]`

## Bug-Suspect Checklist

### 1. Learned policy prior collapsed away from `down`

This is the strongest concrete suspect.

The checkpoint diagnostics say `iteration_64:model` direct policy head chose
`up` on all 48 sampled real states. The mean top-1 margin was not just a tiny
tie; docs recorded about `0.1403`. MCTS then often moved from pure `up` to
`up+stay`, but still never to `down`.

Immediate check:

- Log first-N independent MCTS rows with observation, policy logits, root visit
  counts, selected action, seat, and seed.
- Specifically inspect states where `ball_dy_from_ego_center > 0`; a sane
  vertical policy should sometimes choose `2` there.

### 2. Trainer-side telemetry is not final checkpoint quality

This already explained the biggest "trainer good, scorecard bad" gap.

The sidecar mixes LightZero training/eval activity and was previously dominated
by one seed. The post deep-seed-fix run fixed seed diversity, but the sidecar is
still not a held-out final-checkpoint scorecard.

Immediate check:

- Add row source to sidecar telemetry: collector vs evaluator.
- Add policy/checkpoint identity if LightZero exposes it.
- Report final held-out evaluator rows separately from all env-side rows.

### 3. Official LightZero load path may still differ from adapter load path

Current adapter builds `MuZeroPolicy(cfg.policy)`, strict-loads
`checkpoint["model"]` into `policy._model`, and calls
`policy.eval_mode.forward(...)`.

This probably works: loader smokes show eval output has trained-looking logits,
and `target_model` controls can produce `down`, which argues action extraction
is not broken. Still, the safest falsifier is to compare against LightZero's
own load method.

Immediate check:

- Compare these on the same observations:
  - `policy._model.load_state_dict(checkpoint["model"])`
  - `policy.learn_mode.load_state_dict(full_checkpoint)`
- Confirm `policy.eval_mode` / `_eval_model` sees the loaded tensors in both
  cases.

### 4. `model` vs `target_model` key

Main scorecards load `model`. That is likely right for LightZero eval.

But checkpoint diagnostics showed `target_model` is very different. It also
produced some MCTS `down` in a small control even though its policy head looked
untrained/tied. That makes it a useful control, not a better main choice.

Immediate check:

- Run a small `target_model` scorecard vs `random_uniform`, labeled
  `target_model_control`.
- Do not replace `model` unless official LightZero loading proves that eval
  uses another key.

### 5. Seat/perspective is not the main cause, but still worth logging

Training defaults to `ego_agent=player_0`. Paired independent eval also seats
the checkpoint as `player_1`.

However player0-only MCTS still had zero `down`, so paired-seat transfer is not
hiding a good training-seat policy. The observation is mostly ego-oriented
(`ball_dx_forward`, `ball_vx_forward`, role-based paddle y), but absolute
`ego_paddle_x` and `opponent_paddle_x` leak the physical side.

Immediate check:

- Keep reporting player0-only and paired-seat scorecards separately.
- Add action histograms by seat, not only by policy.

### 6. `to_play=-1`, reward sign, and action mask look low suspicion

For this wrapper, LightZero controls one ego paddle and the opponent is folded
into the env transition. The reward returned to LightZero is raw ego reward:
`+1` ego score, `-1` ego miss, `0` otherwise. Shaped loss-delay return is only
telemetry.

The MCTS adapter calls eval with:

- data shape `[1, 10]`
- `action_mask = [[1, 1, 1]]`
- `to_play = [-1]`
- `ready_env_id = [0]`

All three actions are legal in dummy Pong. Baseline rows prove the scorecard can
count action `2`.

Immediate check:

- Run one small control with `to_play=[0]` only as a diagnostic, not as a main
  scorecard, and compare logits/visit counts/action distribution.
- If it changes action `2` materially, inspect LightZero non-board-game value
  sign handling.

### 7. Horizon/config mismatch was real, but not current root cause

Earlier scorecards silently used 64/120 horizon values for a 512-step training
run. That was fixed: LightZero checkpoint eval now uses
`PongConfig(max_steps=lightzero_max_env_step)`.

Corrected 512/8 and 4096/64 scorecards still had zero `down`, so horizon
mismatch is not the current explanation. It remains a footgun because Modal
scoreboard defaults can still be `64` unless the caller passes the training
horizon.

Immediate check:

- Every scorecard summary should assert:
  - `config.max_steps == lightzero_eval_config.max_env_step`
  - `max_env_step` equals the train command for that checkpoint.

### 8. Opponent policy mismatch is real distribution shift

Training is against `random_uniform` by default. Independent eval also tests
`lagged_track_ball_1` and `track_ball`. Losing to `track_ball` alone would not
prove a bug.

But the checkpoint also loses or underperforms vs `random_uniform`, and
player0-only random rows still choose zero `down`. So opponent distribution
shift is not enough to explain the collapse.

Immediate check:

- Train one tiny deterministic-opponent control, e.g. vs `lagged_track_ball_1`,
  and score against the same opponent first.
- If `down` appears there, random opponent stochasticity and sparse reward are
  likely hurting MuZero's learned dynamics/value.

## Files Inspected

- `src/curvyzero/training/dummy_pong.py`
- `src/curvyzero/training/dummy_pong_eval.py`
- `src/curvyzero/training/lightzero_dummy_pong_env.py`
- `src/curvyzero/training/lightzero_dummy_pong_features.py`
- `src/curvyzero/training/lightzero_dummy_pong_policy.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_scoreboard_attempt.py`
- `scripts/run_dummy_pong_lightzero_mcts_checkpoint_scoreboard.py`
- Prior working/experiment notes for the 512/8, 4096/64, and 1024/16 runs.

## Local Smokes Run

No pytest.

```text
Action mapping:
[(0, 'up', -1), (1, 'stay', 0), (2, 'down', 1)]

Reset ego perspective sample:
{'player_0': (6, 1, 2, 1, 13), 'player_1': (6, -1, 2, 13, 1)}

Baseline-only eval horizon and random action histogram:
max_steps = 120
random_uniform = [22, 23, 31]
```

## Next Checks To Run

1. Add first-N MCTS debug rows for a scorecard: observation row, logits,
   visit counts, selected action, seed, seat, and opponent.
2. Compare adapter load against official LightZero `learn_mode.load_state_dict`
   on the same checkpoint and observations.
3. Run `model` vs `target_model` controls on a tiny random-opponent scorecard.
4. Run a seed-2 reproduction scorecard and a fresh held-out scorecard for the
   same checkpoint to isolate memorized seed behavior.
5. Run a deterministic-opponent training control before scaling the same
   random-opponent setup again.
