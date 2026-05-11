# LightZero Pong Survival And Reward Audit - 2026-05-09

Purpose: make the Pong scoring rule explicit. Pong progress is not just
win/loss. Survival length is a required early signal, especially before a
policy wins points.

## Plain Rule

There are two different things:

1. Training reward: the reward the environment gives to MuZero.
2. Survival telemetry: the scorecard signal that tells us whether a weak
   policy is losing later, reaching rallies, or collapsing.

Do not mix them.

## What Is Implemented

Custom dummy Pong keeps `env.step()` reward sparse:

```text
ego scores:       +1
opponent scores:  -1
no score:          0
timeout:           0 plus truncated=true
```

Source: `src/curvyzero/training/dummy_pong.py`.

LightZero dummy Pong uses that same sparse reward. The wrapper records episode
telemetry separately:

- `steps`
- `max_steps`
- `truncated`
- `score_return`
- `survival_fraction`
- `shaped_loss_delay_return`
- action counts and terminal metadata

Source: `src/curvyzero/training/lightzero_dummy_pong_env.py`.

The dummy Pong eval/scoreboard code already summarizes survival:

- mean, median, p90, min, max, and std survival steps
- truncation rate
- score-return stats by policy
- shaped loss-delay stats by policy
- action histograms

Sources:

- `src/curvyzero/training/dummy_pong_eval.py`
- `scripts/run_dummy_pong_lightzero_mcts_checkpoint_scoreboard.py`
- `scripts/summarize_lightzero_pong_scorecards.py`

Official Atari Pong eval also reports survival length as `steps_survived`
alongside manual return, stock return, reward counts, action collapse, and
entropy.

Sources:

- `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py`
- `scripts/summarize_lightzero_pong_eval_manifest.py`

## Shaped Loss-Delay Telemetry

The current dummy Pong shaped readout is:

```text
survival_fraction = episode_steps / max_steps

win:     +1.0
loss:    -1.0 + 0.5 * survival_fraction
timeout:  0.0
```

This is telemetry and tie-break context. It is not the default LightZero
training reward.

## Reward Shaping Status

Default LightZero dummy Pong training reward shaping does not exist. The
training reward is still sparse `+1/-1/0`.

2026-05-10 update for official Atari Pong: there is now an opt-in shaped
ablation hook in the exact LightZero wrapper. It is separate from stock/control
Pong and only activates when `--survival-reward-per-step` is positive. The run
and attempt ids must contain `survival-shaped`. See
`docs/working/pong_survival_reward_shaping_2026-05-10.md`.

One separate lookahead replay tool has an optional loss-delay target:

- CLI: `scripts/build_dummy_pong_lookahead_replay.py --loss-delay-alpha`
- Code: `src/curvyzero/training/dummy_pong_lookahead_replay.py`

That is a labeled training-label ablation for imitation/relabeling, not a
change to `PongEnv.rewards` and not the default MuZero reward head target.

## Separate Shaped-Objective Run Path

There is now a separate custom dummy Pong shaped-objective ablation path. It is
not stock LightZero Atari Pong, not the LightZero dummy Pong MuZero control, and
not comparable as sparse-reward Pong.

Run label:

```text
custom_dummy_pong_shaped_objective_ablation / loss_delay alpha=0.5
```

Training reward for this labeled path:

```text
true score_return is still logged separately:
  win:      +1.0
  loss:     -1.0
  timeout:   0.0

training_return for reward_mode=loss_delay:
  win:      +1.0
  loss:     -1.0 + alpha * (episode_steps / max_steps)
  timeout:   truncation_bonus
```

The first bounded Modal smoke was launched and completed:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.dummy_pong_survival_curriculum_train_attempt \
  --run-id pong-survival-shaped-loss-delay-alpha0.5-smoke8192-s0 \
  --attempt-id survival-shaped-loss-delay-alpha0.5-smoke8192-s0 \
  --epochs 8 \
  --games-per-epoch 8 \
  --eval-games 4 \
  --max-steps 120 \
  --survival-weight 0.5 \
  --truncation-bonus 0.0 \
  --reward-mode loss_delay \
  --seed 0
```

Refs:

- Modal app: `ap-att7Gn5sZMB5uYkVoCSF1F`
- Summary:
  `training/dummy-pong-survival-shaped/pong-survival-shaped-loss-delay-alpha0.5-smoke8192-s0/attempts/survival-shaped-loss-delay-alpha0.5-smoke8192-s0/train/summary.json`
- Checkpoint:
  `training/dummy-pong-survival-shaped/pong-survival-shaped-loss-delay-alpha0.5-smoke8192-s0/checkpoints/iteration-000008/checkpoint.npz`

Warning: report this as a shaped-objective ablation only. Keep
`mean_score_return`, wins/losses/timeouts, survival, action histogram, and
`mean_training_return` side by side. Do not compare it as stock LightZero Atari
Pong or stock sparse-reward dummy Pong.

## Reporting Requirement

Every Pong report must include:

- wins/losses/timeouts
- score return
- survival mean, median, p90, and std
- truncation rate
- shaped loss-delay return mean and std
- action histogram and entropy
- checkpoint refs and eval split

Never summarize a Pong checkpoint as only `0/N wins` or only return. A policy
that loses at step 100 is different from a policy that loses at step 8, even if
both have zero wins.

## Consolidated References

- `docs/working/pong_reward_shaping_research_2026-05-09.md` - current reward
  shaping recommendation.
- `docs/working/pong_survival_target_recovery_2026-05-09.md` - historical
  recovery note; useful for the optional lookahead loss-delay target, not the
  active default plan.
- `docs/experiments/2026-05-09-dummy-pong-scoreboard-telemetry-patch-smoke.md`
  - smoke proving scorecard survival/shaped telemetry fields.
- `docs/working/lightzero_dummy_pong_scorecard_summary_automation_2026-05-09.md`
  - compact comparison table with score, shaped return, survival, actions, and
  entropy.

## Next Implementation Step

Keep training reward unchanged. The next exact step is to make checkpoint
promotion and live-eval summaries rank or at least display rows with survival
first-class beside score:

```text
primary claim: heldout score/win improvement
early signal: survival mean/p90/std and shaped loss-delay
reject: worse score, timeout farming, or action collapse
```

If we later train on survival shaping, create a separately named shaped-objective
run and keep heldout eval unshaped.
