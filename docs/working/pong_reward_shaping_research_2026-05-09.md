# Pong Reward Shaping Research - 2026-05-09

Owner: reward-shaping research lane  
Scope: Pong-like delayed/sparse games under MuZero/AlphaZero-style training  
Question: when is survival-time reward shaping useful or harmful, and how
should we use survival here?

## Recommendation

Keep the default training environment reward sparse and true:

```text
ego scores:       +1
opponent scores:  -1
otherwise:         0
timeout:           0 with truncation telemetry
```

Use survival time, loss-delay return, timeout rate, p90 steps, action entropy,
and terminal cause as telemetry, curriculum controls, replay/seed diagnostics,
and at most bounded checkpoint-selection tie-breakers. Do not train the
MuZero/LightZero reward head on survival shaping unless we explicitly label the
run as a different objective and can justify why the shaped reward will not
change the policy we actually want.

This matches the current project preference: true sparse env reward first;
survival/loss-delay as measurement unless there is a strong reason to train on
shaping.

Short version for current work:

- Survival-first reporting is required. Survival-as-training-reward is not the
  default.
- Official Atari Pong, custom dummy Pong, and CurvyTron are separate lanes.
  Share the reporting discipline; do not share quality claims.
- In official Atari Pong, compare same-run checkpoints and report
  `iteration_0` deltas. `8192 final` versus `32768 iteration_0` is not a
  regression comparison.
- In custom dummy Pong, keep sparse true score as the promotion metric and use
  survival/loss-delay to explain weak-policy progress or failure.
- In CurvyTron, inherit the telemetry pattern, not Pong's result claims.

## Why

AlphaZero-style board-game training uses game outcome, not hand-shaped
positional rewards. MuZero generalizes this by learning a model that predicts
reward, policy, and value for planning; in Atari, that reward is the observed
game reward stream. For Pong-like tasks, the clean analogue is score reward:
point won, point lost, or no score.

Potential-based reward shaping is the main safe exception. Ng, Harada, and
Russell show that policy invariance is preserved for shaping rewards of the
form `F(s, s') = gamma * Phi(s') - Phi(s)` under the right MDP assumptions.
A flat per-step survival bonus, or a terminal "lose later is less bad" reward,
does not obviously have that form. It changes preferences between scoring now,
taking a risky winning shot, keeping a rally alive, or farming a timeout.

Pong is especially sensitive because survival is correlated with competence
early in training but is not the objective. In a competitive setting, paying
both sides to survive can reward longer rallies instead of winning. ALE's
multi-agent Pong documentation is a useful warning: score is `+1/-1`, and a
serve timer exists specifically to prevent indefinite stalling. If we make
longevity a reward, we create our own stalling incentive.

## When Survival Shaping Helps

Survival is useful when it is a progress signal, not the optimized objective.
In sparse games where almost every early episode is a loss, "lost at step 40"
is more informative than "lost at step 8". It helps answer whether the policy
is reaching the ball, delaying first contact failure, or generating longer
state sequences for value/dynamics learning before win rate moves.

Practical good uses:

- Eval telemetry: mean/median/p90 survival, truncation rate, fast-loss buckets,
  shaped loss-delay readouts, and variance.
- Debugging: detect whether a checkpoint is improving contact/rally length
  while true score remains sparse.
- Curriculum: choose easier starts, weaker opponents, contact-pressure seeds,
  or horizon caps based on fast losses versus long losses.
- Replay prioritization: sample rare scoring events and informative long-loss
  states without changing `env.step()` rewards.
- Selection tie-breaker: among candidates with statistically similar true score
  and win rate, prefer better survival/loss-delay only if timeout/stall metrics
  do not worsen.

## When Survival Shaping Hurts

Survival shaping is harmful when optimization pressure can turn "not losing
yet" into "avoid finishing the point." That can produce passive controllers,
long rallies with no scoring pressure, timeout farming, or policies that look
better on shaped return while true score gets worse.

It is especially risky in MuZero-style training because reward is part of the
learned planning model. If the reward head learns survival bonus, MCTS plans
toward survival bonus. This is not just a logging change; it changes the
objective being searched.

Local evidence points the same way. Current dummy Pong code keeps
`PongEnv.rewards` sparse (`+1/-1/0`) and records survival/loss-delay separately
in eval summaries. Recent LightZero dummy Pong runs show that shaped
loss-delay/survival can move before true score, but checkpoint curves can be
mixed: improvement against `random_uniform` did not reliably transfer to
scripted opponents, and action collapse remained visible. That makes survival
valuable as instrumentation, not enough as a promotion metric.

## Project Rule

Use four separate lanes:

| Lane | Definition | Allowed use |
| --- | --- | --- |
| Env reward | `+1/-1/0` score-delta stream | Training reward, replay reward, MuZero reward target, true return |
| True eval | Win rate, score margin/return, opponent wins, heldout split | Promotion and claims of learning |
| Survival telemetry | Steps, p90 steps, truncation rate, terminal cause, shaped loss-delay readout | Debugging, charts, early warning, curriculum |
| Shaped ablation | Any trained loss-delay/survival target | Explicit experiment only, labeled objective change |

Promotion rule:

```text
primary: true heldout score / win rate
secondary: survival or shaped loss-delay only inside a true-score tie band
reject: higher timeout rate, worse true score, passive action collapse, or
        improvement only in trainer-side telemetry
```

Training-on-shaping rule:

Only train on survival shaping if all of these are true:

1. Sparse reward plus better scale/search/replay/curriculum has been tried or
   is clearly blocked.
2. The run is tagged as a shaped-objective ablation, not the main Pong metric.
3. The shaped term is bounded, does not reward timeouts as wins, and cannot
   dominate true score.
4. Evaluation remains on unshaped heldout score/win metrics.
5. We track anti-stall diagnostics: timeout rate, action entropy, score
   attempts, and survival improvements without score regression.

## Concrete Use Here

For the current Pong/MuZero lane:

- Keep `PongEnv` and LightZero wrapper rewards sparse.
- Keep MuZero reward/value targets aligned to the real score stream.
- Continue reporting `survival_steps` and `shaped_loss_delay_return` in
  scorecards, but name them telemetry.
- Put survival next to score in every readout: same-run baseline, true score,
  steps survived, reward counts/timing, positive rewards, action collapse, and
  checkpoint/manifest refs.
- Use survival to build curriculum buckets: fast losses, long losses, wins,
  timeouts, opponent type, and reset-state class.
- Use survival as a tie-breaker only when true heldout score is effectively
  tied and timeout/action-collapse checks pass.
- Do not promote a checkpoint because shaped return improved if win rate or
  score return regressed.

## Current Next Actions

1. Keep official Atari live eval sparse-reward and survival-first in reporting.
2. Keep custom dummy Pong target-sidecar/support-scale debugging separate from
   official Atari claims.
3. Mark any future survival-trained run as a shaped-objective ablation before
   launch, not after results arrive.

## Sources Used

Primary/public sources:

- Silver et al., "Mastering the game of Go without human knowledge" (AlphaGo
  Zero, outcome-style self-play value target):
  https://www.nature.com/articles/nature24270
- Silver et al., "Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm" (AlphaZero, game outcome targets):
  https://arxiv.org/abs/1712.01815
- Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with
  a Learned Model" (MuZero predicts reward, policy, value):
  https://arxiv.org/abs/1911.08265 and
  https://www.nature.com/articles/s41586-020-03051-4
- Ng, Harada, and Russell, "Policy Invariance Under Reward Transformations:
  Theory and Application to Reward Shaping" (potential-based shaping):
  https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html
- ALE multi-agent Pong documentation (`+1/-1` scoring and serve timer to
  prevent indefinite stalling):
  https://ale.farama.org/multi-agent-environments/pong/
- OpenAI, "Faulty reward functions in the wild" (proxy reward failure example):
  https://openai.com/index/faulty-reward-functions
- Google DeepMind, "Specification gaming: the flip side of AI ingenuity"
  (reward shaping can change the optimal policy if not potential-based):
  https://deepmind.google/blog/article/Specification-gaming-the-flip-side-of-AI-ingenuity

Local docs/source reviewed:

- `docs/research/reward_shaping_for_pong_curvy_muzero.md`
- `docs/research/pong_reward_design.md`
- `docs/working/pong_survival_target_recovery_2026-05-09.md`
- `docs/working/lightzero_muzero_target_semantics_2026-05-09.md`
- `docs/working/lightzero_pong_sparse_training_scale_ladder_2026-05-09.md`
- `docs/experiments/2026-05-09-lightzero-muzero-dummy-pong-checkpoint-curve.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-lag1-shaped-knob-run.md`
- `src/curvyzero/training/dummy_pong.py`
- `src/curvyzero/training/dummy_pong_eval.py`
- `src/curvyzero/training/lightzero_dummy_pong_env.py`
- `src/curvyzero/training/dummy_pong_lookahead_replay.py`
