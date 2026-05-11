# CurvyTron No-Learning Investigation - 2026-05-11

Purpose: keep the current failure picture clear. The goal is not to explain
everything at once. The goal is to stop scaling runs that are likely measuring
the wrong thing or training through a broken contract.

## Current Read

- Live11 two-seat runs are not showing sustained survival improvement.
- They do update model weights.
- They do not show a hard action collapse under the current 95% top-action check.
- Survival means are mostly flat around the current trainer timestep scale.
- This is not enough evidence to keep blindly scaling.

## Measurement Correction

Old CurvyTron step counts are not comparable unless the timestep is the same.

Simple scripted baselines on the trainer surface:

| decision interval | straight mean steps | random mean steps | read |
| ---: | ---: | ---: | --- |
| 300 ms | about 9 | about 8 | current live11 scale |
| 50 ms | about 43 | about 40 | same behavior, finer timestep |
| 16 ms | about 134 | about 132 | same behavior, much finer timestep |

So a `170 step` result may just mean a smaller decision interval, not a better
policy. Always record `decision_ms`, `max_ticks`, eval path, opponent, and
checkpoint id with survival curves.

## Concrete Suspects

1. Player observation ambiguity. Fixed for the next run.

The two-seat path was giving both players the same global visual frame. The
current fix remaps the raw player pixels into a player perspective before
normalization, without changing the model input shape. For each policy row:
controlled player pixels become "self" values and the other player's pixels
become "other" values. Do not treat the old `to_play=player_id` custom-path
note as native-compatible; use `to_play=-1` for single-agent/bot-style rows,
and only use LightZero board-game ids `1/2` in a tested board-game contract.

Local probe passed: player-frame delta is nonzero after reset and after one
step.

2. Return target boundary. Fixed for the next run.

Survival return targets were likely grouped by outer training iteration. If one
episode spanned two iterations, the target could become "survive this short
chunk" instead of "survive the episode." Replay rows now preserve
`episode_id`, samples carry episode ids, and the return lookup groups by
`episode_id, env_row, player` when that metadata exists.

Local proof passed: one episode split across iterations now returns
`[4, 3, 2, 1]` instead of being cut at the iteration boundary.

3. Eval/train mismatch.

Some older evals used single-ego or frozen-opponent surfaces. The current
two-seat path is current-policy self-play. Do not merge these into one learning
claim.

4. Noisy action runs may train mixed signals.

If action noise changes the environment action but the policy target still comes
from the pre-noise action, noisy variants can become confusing. Treat those
variants as secondary until the clean path learns.

## Useful Positive Evidence

The trainer-surface baseline tool now includes simple observation-based
policies. A tiny smoke showed `ray_clearance` and `wall_avoid` surviving longer
than random/straight. This is not a final result, but it suggests the observation
interface contains usable survival information.

Runbook:

```sh
uv run python -m curvyzero.training.curvytron_baseline_eval \
  --episodes 64 \
  --batch-size 64 \
  --max-steps 2048 \
  --policy-kinds straight,left,right,random_legal,wall_avoid,ray_clearance \
  --observation-summary-dir artifacts/local/curvytron_learnability_probe \
  --observation-summary-limit 8
```

## Next Gates

- Monitor the corrected live12 run.
- Confirm progress rows show no problems, model changes, non-collapsed actions,
  and rising mean/max completed episode steps.
- Compare against scripted baselines using the same `decision_ms`.
- Only then launch another larger batch.

## Corrected Live Run

```text
run_id: curvytron-two-seat-selfplay-live12-playerpersp-episodefix-clean-long-20260511
attempt_id: live12-playerpersp-episodefix-clean-long-20260511
function_call_id: fc-01KRBMJJ0N2ZK3550F3TDVDBMC
compute: gpu-l4-t4
batch_size: 32
collect_steps_per_iteration: 128
outer_iterations: 100000
updates_per_iteration: 8
num_simulations: 4
replay_scope: accumulated
learner_sample_size: 512
max_replay_rows: 65536
max_ticks: 16384
alive_reward: 0.01
dead_reward: -1.0
checkpoint_every_iterations: 100
progress_every_iterations: 25
```

Initial status check found `progress_latest.json` and `iteration_0`. No learning
claim yet.

## Non-Claims

- We have not yet proven stable CurvyTron learning.
- We have not yet proven that more training time alone solves the issue.
- We should not treat fixed-opponent spikes as proof of general self-play
  progress.
