# LightZero Pong Training Config Bug Investigation - 2026-05-09

Scope: LightZero dummy Pong train/config/eval path. No pytest. I made only small
direct fixes in the LightZero Pong wrapper/evaluator.

## Executive Read

The up-only checkpoint behavior is not an action-index inversion. The action
schema is consistent end to end: `0=up`, `1=stay`, `2=down`.

The strongest concrete bug found was a scorecard config mismatch. The completed
512/8 MCTS scorecard was labeled for the 512/8 run, but its pulled summary says
the LightZero checkpoint reconstruction used `max_env_step=64`, while the
independent Pong game used `PongConfig.max_steps=120`. That directly changes
the `step / max_steps` observation feature and can change policy behavior. It
does not fully explain up-only by itself, because MCTS still chose almost all
up, but it makes that scorecard not a faithful 512-step checkpoint eval.

The training setup is single-agent-vs-scripted-opponent play. LightZero controls
one ego paddle and the wrapper supplies the opponent action from
`random_uniform`, `track_ball`, or `lagged_track_ball_1`. This is not self-play.

## Code Edits

- `src/curvyzero/training/lightzero_dummy_pong_env.py`
  - Fixed `random_action()` to use a persistent per-episode RNG instead of
    reseeding from `self._episode_seed` on every call.
  - Added `timestep` to the base LightZero observation dict, so it is no
    longer only supplied by the scaled train wrapper monkey patch.
- `src/curvyzero/training/dummy_pong_eval.py`
  - When scoring LightZero policy-head or MCTS checkpoints, the independent
    eval now constructs `PongConfig(max_steps=lightzero_max_env_step)`.
    Baseline-only eval keeps the historical default `PongConfig()` horizon
    of 120.

These are contract fixes. They do not claim to make the existing checkpoint
learned or non-up-only.

## Findings By Question

1. Action mapping is consistent.

- `ACTION_LABELS = ("up", "stay", "down")`.
- `PongEnv._move_paddle()` uses `delta = action - 1`, so action `0` decreases
  paddle y, `1` stays, and `2` increases paddle y.
- Fixed baseline policies return the same ids: track-ball returns `0` when the
  ball is above the paddle center and `2` when below.
- LightZero env action space is `Discrete(3)`, action mask is all ones, and
  policy-head/MCTS adapters return integer ids directly into the same schema.

2. Observations are mostly encoded correctly, but horizon normalization was
wrong in the completed scorecard.

- Training wrapper and LightZero checkpoint adapters both use
  `encode_tabular_ego_observation()` for `tabular_ego`.
- Ego perspective is encoded in `PongEnv.observation()` with `forward = 1` for
  `player_0` and `-1` for `player_1`; `ball_dx_forward` and
  `ball_vx_forward` are ego-frame values.
- The wrapper trains/evals ego `player_0` by default, so the current real run
  did not exercise the `player_1` LightZero ego path.
- The bad part: the pulled full MCTS summary at
  `eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-20260509T150000Z-20260509T150243Z/summary.json`
  recorded checkpoint reconstruction `max_env_step=64`, but summary
  `config.max_steps=120`. The source training run was `max_env_step=512`.
  This means the `step` feature was normalized differently from training.
- Patch made: future LightZero scorecards use `lightzero_max_env_step` for the
  independent `PongConfig.max_steps` too. A faithful rerun of the 512/8
  checkpoint still must pass `--max-env-step 512`.

3. LightZero is training against the intended scripted opponent and raw reward.

- The wrapper takes only the ego action from LightZero.
- The opponent action comes from `opponent_policy`; the 512/8 run config used
  `random_uniform`.
- Env reward returned to LightZero is raw terminal score reward:
  `+1` ego score, `-1` opponent score, `0` otherwise.
- `shaped_loss_delay_return` is telemetry only in `info`/summaries; it is not
  returned as the env reward.
- Plainly: this is fixed-opponent play, not self-play.

4. Tiny configs are too small, but there were also wire/config bugs.

- The 64/2 smoke and 512/8 run are both tiny for MuZero. Weak/tied logits can
  easily make policy-head argmax pick action `0`.
- The MCTS scorecard also remained effectively up-only, so policy-head argmax
  tie-breaking is a symptom, not the whole root cause.
- Real wire bugs found:
  - `random_action()` reseeded every call.
  - `timestep` existed only through `lightzero_dummy_pong_train_attempt`'s
    monkey patch.
  - The completed MCTS scorecard reconstructed/evaluated a 512-step checkpoint
    with 64/120 horizon values.

5. `to_play=-1` is right for this setup.

This is the non-board-game/single-agent-control LightZero path. There is no
alternating two-player search state exposed to LightZero; the scripted opponent
is folded into the environment transition. `to_play=-1` is consistent with the
existing docs and adapters.

6. Wrapper contract status.

- `reset()` returns `observation`, `action_mask`, `to_play`, and now `timestep`.
- `step()` returns `BaseEnvTimestep(obs, reward, done, info)`.
- `done` is `terminated or truncated`.
- `info["curvyzero_pong"]` includes terminated/truncated, winner, final rewards,
  action counts, action trace, trace hash, and score/survival telemetry.
- All three actions are always legal, so all-ones `action_mask` is correct.

## Ruled Out

- No action-id inversion found.
- No reward-shaping leak found.
- No action-mask restriction found.
- Loader strictness is not the current blocker for `iteration_8`; the MCTS
  loader used `res_connection_in_dynamics_true` and strict load passed.
- Direct policy-head greedy scorecard is not MCTS, but the later full MCTS
  scorecard also chose almost all up: combined LightZero histogram was
  `[2060, 7, 0]` in the existing summary.

## Verification

- Pulled the existing MCTS summary from the Modal volume and verified:
  - `lightzero_eval_config.max_env_step: 64`
  - checkpoint metadata `config_surface.max_env_step: 64`
  - independent eval `config.max_steps: 120`
  - `num_simulations: 8`
- Local action mapping smoke printed:
  - `0 up -1`
  - `1 stay 0`
  - `2 down 1`
- Baseline-only `run_dummy_pong_eval(episodes=1, seed=0)` still reports
  `config.max_steps == 120`.
- `py_compile` passed for the inspected training/env/eval/scorecard modules.

## Next Runs

1. Rerun the MCTS scorecard for the 512/8 `iteration_8` checkpoint with
   `--max-env-step 512` and record the summary's `lightzero_eval_config` and
   `config.max_steps`; both should be 512 after the patch.
2. Run a real-observation logits/action grid for `iteration_0`, `iteration_8`,
   and `ckpt_best` with `max_env_step=512`.
3. If MCTS remains up-only after the faithful 512 rerun, treat this as a
   learning-signal/config-size problem next: inspect replay size, policy/value
   loss movement, target reward/value stats, root visit entropy, and checkpoint
   tensor deltas.
