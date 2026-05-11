# 2026-05-09 LightZero dummy Pong contact-pressure curriculum smoke

## Question

Can the next dummy Pong experiment be made explicit and scoreable by changing
only the custom dummy Pong reset distribution toward near-contact/scoring
pressure states, while preserving the real sparse environment reward?

## Setup

- Algorithm: LightZero MuZero.
- Environment: project-owned custom dummy Pong, not stock Atari Pong.
- Curriculum knob: `pong_reset_profile=contact_pressure`.
- Pressure target: training uses `pong_reset_pressure_agent=ego`, resolved by
  the LightZero wrapper to the learner's `ego_agent`; the matching independent
  scorecard used `pong_reset_pressure_agent=player_0` with
  `--no-paired-seats`.
- Opponent: `lagged_track_ball_1`.
- Feature mode: `tabular_ego`.
- Reward: unchanged sparse env reward only: ego score `+1`, opponent score
  `-1`, non-score step `0`. No survival reward was added to `env.step()`.

The new reset profile starts episodes with the ball a few cells from a
selected paddle and moving toward it. It records reset metadata in `info`; it
does not alter paddle-hit/scoring reward.

## Commands

Compile/import smoke:

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong.py \
  src/curvyzero/training/lightzero_dummy_pong_env.py \
  src/curvyzero/training/dummy_pong_eval.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_scoreboard_attempt.py \
  scripts/run_dummy_pong_lightzero_mcts_checkpoint_scoreboard.py
```

Tiny Modal train:

```sh
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 31 \
  --opponent-policy lagged_track_ball_1 \
  --ego-agent player_0 \
  --max-env-step 64 \
  --pong-episode-max-steps 64 \
  --pong-reset-profile contact_pressure \
  --pong-reset-pressure-agent ego \
  --max-train-iter 2 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --num-simulations 2 \
  --batch-size 8 \
  --update-per-collect 1 \
  --n-episode 1 \
  --game-segment-length 32 \
  --td-steps 64 \
  --num-unroll-steps 5 \
  --discount-factor 1.0 \
  --reward-support-min -1 \
  --reward-support-max 1 \
  --reward-support-delta 1 \
  --value-support-min -1 \
  --value-support-max 1 \
  --value-support-delta 0.01
```

Matching tiny MCTS scorecard:

```sh
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt \
  --checkpoints lightzero:iter2=ref:training/lightzero-dummy-pong/lz-dpong-20260509T174107Z-5ff5f902f8ed/checkpoints/lightzero/iteration_2.pth.tar \
  --episodes 4 \
  --seed 41 \
  --split-id dummy_pong_contact_pressure_curriculum_smoke \
  --split-role monitor \
  --run-id lz-dpong-20260509T174107Z-5ff5f902f8ed \
  --attempt-id attempt-20260509T174107Z-2f29dc98a231 \
  --eval-id mcts-scoreboard-contact-pressure-smoke-iter2 \
  --max-env-step 64 \
  --num-simulations 2 \
  --feature-mode tabular_ego \
  --pong-reset-profile contact_pressure \
  --pong-reset-pressure-agent player_0 \
  --no-paired-seats
```

## Results

- Compile/import smoke passed.
- Modal train passed:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-bNRz3Mtil6apjX5w6tNZxa`.
- Modal MCTS scorecard passed:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-XRyCAYWAN7F3ptvRAKRC0x`.
- Train run:
  `lz-dpong-20260509T174107Z-5ff5f902f8ed`,
  attempt `attempt-20260509T174107Z-2f29dc98a231`.
- Train summary:
  `training/lightzero-dummy-pong/lz-dpong-20260509T174107Z-5ff5f902f8ed/attempts/attempt-20260509T174107Z-2f29dc98a231/train/summary.json`.
- Eval summary:
  `training/lightzero-dummy-pong/lz-dpong-20260509T174107Z-5ff5f902f8ed/attempts/attempt-20260509T174107Z-2f29dc98a231/eval/mcts-scoreboard-contact-pressure-smoke-iter2/summary.json`.

Trainer-side env telemetry was tiny but non-flat:

- Episodes: 4.
- Wins/losses/timeouts: 2 / 1 / 1.
- Raw score return mean/std: `0.25` / `0.8292`.
- Shaped loss-delay telemetry mean/std: `0.3027` / `0.7510`.
- Survival steps mean/median/p90/std: `36.25` / `32.5` / `56.2` / `17.81`.
- Learner actions: up `81`, stay `46`, down `18`.

Independent held-out MCTS scorecard did not pass policy-quality criteria:

- `lightzero_iter2` versus `lagged_track_ball_1`: learned wins `1/4`, raw
  mean `-0.5`, shaped mean `-0.4727`, survival mean `7.5`, actions
  `[19,11,0]`.
- `lightzero_iter2` versus `random_uniform`: learned wins `1/4`, raw mean
  `-0.5`, shaped mean `-0.4746`, survival mean `7.0`, actions `[20,8,0]`.
- `lightzero_iter2` versus `track_ball`: learned wins `0/4`, raw mean `-0.75`,
  shaped mean `-0.7227`, survival mean `19.5`, actions `[56,22,0]`.
- Baseline sanity row `track_ball` versus `track_ball` still truncated `4/4`
  at 64 steps under this reset profile.

## Interpretation

This is a mechanical implementation pass, not a policy-quality win.

The custom contact-pressure reset distribution is now an explicit opt-in
dummy Pong curriculum. It is clearly separate from stock Atari Pong benchmark
replication and does not change the sparse reward contract.

The tiny training-side telemetry shows the curriculum can expose score-bearing
episodes quickly. The independent scorecard still shows a collapsed learned
checkpoint with zero `down` actions, so this exact 64-step/2-iteration smoke is
a stop signal for quality claims.

Post-smoke scoreability probe:
`docs/experiments/2026-05-09-dummy-pong-contact-pressure-scoreability-probe.md`
sampled 64 real contact-pressure reset states, swept `up/stay/down`, and kept
the sparse env reward unchanged. Every reset/opponent group was action-sensitive
by contact angle, score, or survival. The scoreability split matters:
`lagged_track_ball_1` was scoreable in 46/64 groups and `stay` in 59/64, while
default `track_ball` was scoreable in 0/64. This supports only a narrow
lagged-opponent curriculum diagnostic, not a `track_ball` score target.

## Artifacts

- `src/curvyzero/training/dummy_pong.py`
- `src/curvyzero/training/lightzero_dummy_pong_env.py`
- `src/curvyzero/training/dummy_pong_eval.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_scoreboard_attempt.py`
- `scripts/run_dummy_pong_lightzero_mcts_checkpoint_scoreboard.py`
- `scripts/probe_dummy_pong_contact_pressure_scoreability.py`

## Follow-ups

- Go only to a modest next curriculum rung if it keeps the same explicit
  labels: custom dummy Pong, `contact_pressure`, sparse env reward unchanged.
- Do not claim stock Atari Pong progress from this lane.
- Next useful rung should score `iteration_0`, `iteration_2`, and final under
  the same contact-pressure eval, and should require nonzero `down` use plus
  held-out raw/shaped improvement before scaling.
