# RL Debugging: CartPole Vs Pong MuZero - 2026-05-09

Goal: collect outside evidence for why a simple MuZero-style control can work
while a simple Pong-style task fails, then map that evidence to our current
Pong runs. Sources are primary docs/papers where possible. No blogs were used.
No pytest was run.

## Short Read

The outside sources point to the same failure surfaces we are already seeing:

- MuZero is not just "MCTS plus a network." It trains reward, value, and policy
  targets, then uses search visit counts for action selection. If the reward is
  sparse, the value/reward target scale is weak, or search roots are tied, the
  policy can look alive during collection but fail in deterministic eval.
- AlphaZero/MuZero-style systems intentionally use stochastic self-play or
  collect behavior, then more deterministic evaluation. This can hide action
  collapse unless we log collect and eval action distributions separately.
- LightZero exposes the important knobs directly: `update_per_collect`,
  `replay_ratio`, `random_collect_episode_num`, `eps`, `num_simulations`,
  `game_segment_length`, `td_steps`, reward/value support, collect/eval episode
  caps, and env wrapper contracts.
- RL results have high variance. A working tiny CartPole run proves the stock
  path can train, not that the custom Pong protocol is sound.

For our Pong evidence, the main current diagnosis is not "LightZero cannot run."
CartPole and Pong differ on reward density, horizon semantics, opponent
stochasticity, action count, custom env/eval plumbing, update ratio, and
deterministic MCTS tie behavior. The sharpest symptom is action collapse in
independent MCTS eval: recent trusted Pong checkpoints choose zero `down`
actions in every checkpoint-vs-baseline row.

## Primary Source Anchors

- [MuZero, Schrittwieser et al. 2019/2020](https://arxiv.org/abs/1911.08265):
  MuZero learns the reward, policy, and value used for planning. The paper also
  separates Atari and board-game regimes, which matters because our dummy Pong
  is closer to an Atari/control task than a clean board-game self-play task.
- [AlphaZero, Silver et al. 2017](https://arxiv.org/abs/1712.01815):
  search returns a policy from root visit counts; moves can be selected
  proportionally or greedily from those counts; training starts from self-play
  and root exploration noise is used for exploration.
- [LightZero config docs](https://opendilab.github.io/LightZero/tutorials/config/config.html):
  LightZero says to start custom work from a nearby zoo config and tune the
  frequently changed fields, including collector/evaluator counts,
  collect/eval episode caps, reward/value support, `game_segment_length`,
  `update_per_collect`, `random_collect_episode_num`, `eps`,
  `num_simulations`, `n_episode`, and `max_env_step`.
- [LightZero custom env docs](https://opendilab.github.io/LightZero/tutorials/envs/customize_envs.html):
  non-board env observations still need `observation`, `action_mask`, and
  `to_play=-1`; discrete non-board envs usually return an all-ones action mask.
- [LightZero MuZero policy source/docs](https://www.aidoczh.com/lightzero/_modules/lzero/policy/muzero.html):
  collect mode samples from MCTS visit distributions with collect temperature
  and optional epsilon; eval mode chooses the highest-value action rather than
  sampling. The source comments also warn that larger `update_per_collect` means
  more off-policy training and should depend on episode length/reuse.
- Local LightZero source:
  `/tmp/lightzero-src/lzero/agent/config/muzero/gym_cartpole_v0.py` uses a
  small MLP, 2 actions, `num_simulations=25`, `update_per_collect=100`, and
  `game_segment_length=50`.
- Local LightZero source:
  `/tmp/lightzero-src/lzero/agent/config/muzero/gym_pongnoframeskip_v4.py` uses
  an Atari conv setup, 6 actions, `num_simulations=50`,
  `update_per_collect=1000`, `game_segment_length=400`,
  `random_collect_episode_num=0`, and `eps_greedy_exploration_in_collect=False`.
- [Fedus et al. 2020, Revisiting Fundamentals of Experience Replay](https://proceedings.mlr.press/v119/fedus20a.html):
  replay capacity and replay ratio matter materially in off-policy deep RL.
- [Henderson et al. 2018, Deep Reinforcement Learning That Matters](https://arxiv.org/abs/1709.06560):
  non-determinism and high variance make RL results easy to misread without
  careful seeds and reporting.
- [Agarwal et al. 2021, Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264):
  small-run point estimates are unreliable; use interval-style thinking and
  multiple seeds/runs before claiming progress.

## Debug Signals From The Sources

Reward sparsity:
CartPole has dense alive-step reward. Our Pong reward is mostly `0` until score
or miss, with truncation returning `0`. MuZero has a reward head, but sparse
terminal reward still means few useful reward targets and harder credit
assignment.

Exploration:
AlphaZero and LightZero use exploration at the root/collection side. LightZero
also offers `random_collect_episode_num` and epsilon-greedy collect. If these are
off or not logged, a sparse Pong learner may only see narrow data.

Action collapse:
A policy can emit varied actions during collect because collect samples from
visit counts, while eval is deterministic. Our current Pong MCTS eval rows show
the failure plainly: zero `down` actions for `iter0`, `iter16`, and `best`
against every baseline row in the trusted checkpoint-curve scorecard.

Deterministic eval vs stochastic collect:
LightZero collect and eval are different modes. This explains why train-side
telemetry can look varied while an independent scorecard collapses.

MCTS visit ties:
With low `num_simulations`, three legal actions, and weak logits, many roots can
tie or nearly tie. Our debug rows showed tiny logits and visits like `[2,3,3]`
or `[3,3,2]`; deterministic eval tie-breaking can then dominate behavior.

Replay/update ratio:
LightZero's stock MuZero CartPole config uses `update_per_collect=100`; stock
Atari Pong uses `1000`. Our custom Pong comparison note shows main Pong attempts
used `update_per_collect=1`. That may be far too little learner work per collect
for sparse rewards and small runs.

Value/reward target scale:
LightZero uses categorical reward/value supports and scalar transforms. If the
support, loss weights, or terminal reward scale are mismatched, the value/reward
heads can be weak even when the code runs.

Episode horizon:
Our Pong wrapper has used `max_env_step` both as trainer budget and Pong episode
cap/normalization denominator. Changing training budget can accidentally change
the task, the `step/max_steps` observation feature, and truncation frequency.

Seed/diversity:
Earlier Pong sidecars were misleading because seeds repeated heavily. The
post-deep-seed-fix run improved diversity, but independent MCTS still failed.
That means seed plumbing was a real bug, but not the whole cause.

Overfitting to opponent:
Default `track_ball` is a survival/tie wall in current geometry. The scoreable
target became `lagged_track_ball_1`, while default `track_ball` remains a
survival diagnostic. A model can improve against random or lag-1 and still not
make scoring pressure against default `track_ball`.

Train/eval mismatch:
The trusted Pong read must come from independent final-checkpoint scorecards,
not only trainer sidecar wins. We have already seen trainer sidecars look strong
while final checkpoints fail against random/scripted opponents.

## Comparison To Our Pong Evidence

Working CartPole lane:

- Stock-ish LightZero path, stock env wrapper, stock trainer/evaluator.
- Dense reward every alive step.
- Small MLP observation and 2 actions.
- Progression config used `update_per_collect=4` in our tiny Modal run and the
  stock source config uses `100`.
- Final evaluator reward/checkpoints were enough for a plumbing/progression
  signal.

Failing custom Pong lane:

- Custom LightZero env around a single ego paddle plus scripted opponent.
- Sparse terminal reward: nonterminal `0`, score `+1`, miss `-1`, truncation `0`.
- Three actions: `0=up`, `1=stay`, `2=down`.
- Main attempts used `update_per_collect=1`.
- `max_env_step` also changes Pong horizon and normalized time feature.
- Seed plumbing needed fixes; after fixes, quality still failed.
- Independent MCTS eval is the quality gate, and the trusted checkpoint curve
  shows no consistent checkpoint improvement.
- The current strongest concrete symptom: all three trusted LightZero
  checkpoints have zero `down` actions in every checkpoint-vs-baseline row.

This makes the next question narrower: are the roots weak because the model did
not learn useful reward/value/policy targets, because collect data is too narrow,
because update/replay ratio is too low, or because deterministic eval is exposing
ties that collect sampling hid?

## Recommended Diagnostic Ladder

1. Env contract:
   Confirm `observation`, all-ones `action_mask`, `to_play=-1`, reward sign,
   terminated/truncated split, and action ids on a fixed seed/action trace.

2. Baseline floor:
   Score `random_uniform`, `track_ball`, and `lagged_track_ball_1` with paired
   seats. Keep default `track_ball` as survival/tie telemetry, not the only gate.

3. Collect-vs-eval same observations:
   On saved observations, call collect mode and eval mode from the same
   checkpoint. Log logits, visit counts, visit entropy, selected action, and
   whether the root had ties.

4. Simulation sweep:
   Run first-N debug rows at `num_simulations={2,8,16,25,50}`. If action collapse
   disappears only with more simulations, the current eval is too tie-prone. If
   it persists, the learned model/search signal is weak.

5. Exploration audit:
   Log `random_collect_episode_num`, epsilon schedule, collect temperature, root
   noise, and collect action histograms. Verify all three actions appear before
   learning and during early collect.

6. Update/replay audit:
   Sweep `update_per_collect` upward before scaling env steps. Try values closer
   to the stock examples, while holding seeds/horizon constant.

7. Target audit:
   Log reward targets, value targets, predicted reward/value, support ranges,
   reward/value/policy losses, and mask coverage. Watch for all-zero reward
   batches and value targets compressed near zero.

8. Horizon isolation:
   Separate trainer budget from Pong `max_steps`. Keep observation normalization
   fixed while changing training length.

9. Opponent/curriculum check:
   Train/eval separately against `random_uniform`, `lagged_track_ball_1`, and
   default `track_ball`. Do not mix them until each row is interpretable.

10. Seed and overfit check:
    Require unique seed counts, top-seed share, paired-seat rows, fixed monitor
    seeds, and held-out seeds before claiming progress.

Stop the ladder early if a step produces a clear bug. In this task, a compact
debug row with logits, visits, action, seed, opponent, and outcome is more useful
than another long training run.
