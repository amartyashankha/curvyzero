# Survival Signal Web Research, 2026-05-16

Goal: explain why CurvyTron MuZero/LightZero survival may not clearly improve
even though the checkpoint -> tournament -> assignment feedback loop works.

Active run context:

- Batch manifest:
  `artifacts/local/curvytron_tonight18_manifests/curvy-r18fresh-allv2-20260516a/curvy-r18fresh-allv2-20260516a.json`
- Tournament / Elo:
  `curvy-r18fresh-live-bounded-dsf1-20260516b` /
  `elo-r18fresh-live-bounded-dsf1-20260516b`
- Matrix: 18 rows = 3 reward variants x 3 opponent recipes x 2 noise modes.
- Stock LightZero path: rows call `lzero.entry.train_muzero`, but the env is
  `source_state_fixed_opponent`. This is checkpoint/fixed-opponent training, not
  exact current-policy two-seat self-play.

## Strongest Read

The feedback loop can be correct while the learning target is still too noisy,
stale, or poorly scaled. The highest-risk local issues are:

1. **Replay/update ratio may be much larger than intended.** LightZero sets
   `update_per_collect = collected_transitions * replay_ratio` when
   `update_per_collect=None`; its docs/source warn that larger
   `update_per_collect` is more off-policy. Our manifest uses
   `collector_env_num=256`, `n_episode=256`, `batch_size=32`, and does not
   explicitly patch `update_per_collect`, so the actual learner updates per
   collection must be measured from run summaries before trusting curves.
2. **Opponent distribution is nonstationary.** Assignment refresh runs every
   2000 train iterations, while replay can still contain trajectories from older
   opponent pools. Multi-agent replay literature treats changing co-players as a
   real replay-staleness problem, not just noise.
3. **Dense survival rewards are probably saturating value/reward supports.**
   The launcher sets `discount_factor=1.0` for source-state fixed-opponent
   runs, keeps stock `td_steps` unless an explicit target-horizon knob exists,
   and caps model support scale at `300`. But `source_max_steps=1,048,576`, and
   `survival_plus_bonus_plus_outcome` can create terminal rewards scaled by
   episode length. If targets saturate, the value head cannot distinguish
   "survived 400" from "survived 40,000" cleanly.
4. **Action collapse is a first-class failure, not a cosmetic warning.** The
   local stagnation note reports collapse flags in most rows. LightZero exposes
   policy entropy, target policy entropy, reward/value losses, predicted rewards,
   predicted values, and MCTS visit entropy-like fields; those should be plotted
   against survival before changing algorithms.
5. **Tournament Elo and survival are different objectives.** The tournament
   measures bounded checkpoint-vs-checkpoint strength. Background eval samples a
   small seed set and recipe context. A checkpoint can gain Elo by killing weak
   opponents sooner while not improving absolute survival.

## What The Sources Imply

- MuZero is trained to predict the planning-relevant quantities: reward, policy,
  and value. Debugging should therefore inspect all three target/loss streams,
  not only episode return or Elo.
- AlphaZero/AlphaGo Zero-style loops rely on self-play/eval/checkpoint ladders,
  not final-checkpoint faith. Local evidence where mid-run checkpoints beat final
  checkpoints is compatible with instability, overfitting, stale replay, or
  reward/support mismatch.
- LightZero's MuZero loop is collect -> push `GameSegment`s -> sample replay ->
  train. If the replay ratio is high or the opponent pool changes under the
  replay buffer, the learner can chase a mixture of old and new game dynamics.
- Reward shaping is not automatically objective-preserving. Dense survival and
  bonus rewards are useful diagnostics, but they should be separated from the
  tournament objective unless the intended objective is explicitly survival-plus-
  bonus rather than winning.

## Diagnostics To Run Before More Scaling

1. **Dump actual LightZero learning ratios per row.** From each `summary.json` /
   `phase_profile`, report collector calls, game segments pushed, env steps
   collected, replay sample calls, learner train calls, and effective
   `learner_train_calls / env_steps_collected`. Also record compiled
   `update_per_collect`, `replay_ratio`, `td_steps`, `num_unroll_steps`,
   `discount_factor`, support ranges, and support sizes.
2. **Check target saturation.** For each reward variant, sample replay targets
   and compute min/median/p95/max for reward and value targets before categorical
   support transform. Count values outside support. If nonzero for dense rows,
   treat current curves as suspect.
3. **Plot action health per checkpoint.** For train collect, background eval,
   and tournament, plot action histogram, MCTS visit entropy, policy entropy,
   target policy entropy, illegal-action fallback count, and survival. Collapse
   should be marked as a blocker on promotion.
4. **Tag every replay/eval row with opponent assignment generation.** Compare
   survival against the launch assignment, current assignment, and each refreshed
   assignment separately. Do not aggregate across changing opponent pools until
   the per-pool curves are visible.
5. **Split metrics by objective.** For every checkpoint ladder, report survival
   mean/median/p90/censored count, sparse win/loss/draw, dense survival reward,
   bonus reward, terminal outcome reward, and Elo. Treat disagreement as signal.
6. **Verify eval/tournament parity.** On the same checkpoint and same fixed
   opponent snapshot, run background eval and tournament eval with matched seat,
   action mode, observation surface, max steps, MCTS simulations, and seed list.

## Recommended Next Experiments

Run these as small controlled ladders, not another 18-row sweep.

1. **Frozen no-refresh control.** One reward variant, one immutable opponent
   assignment, no assignment refresh, no action-repeat/straight-override noise,
   fixed eval seeds, 3 seeds. This answers whether the policy can improve against
   a stationary pool.
2. **Replay-ratio ladder.** Same control with explicit `update_per_collect` or
   replay ratio settings, e.g. very low / medium / current-equivalent. Promote
   the setting that improves survival without action collapse.
3. **Reward/support ladder.** Use a short horizon such as `source_max_steps=4096`
   or scale alive reward by horizon so value targets fit inside support. Compare:
   sparse outcome, scaled dense survival no outcome, and dense survival plus
   unscaled terminal outcome. Avoid episode-length-scaled terminal outcome until
   support saturation is falsified.
4. **Single-opponent curriculum control.** Train against only blank canvas, only
   wall-avoidant immortal, and one frozen mortal checkpoint. This isolates
   whether collapse comes from opponent mixture variance or from the core
   reward/search setup.
5. **Best-vs-latest checkpoint gate.** Every 10k checkpoint should be evaluated
   on the same fixed grid. Keep the best-survival checkpoint for promotion; do
   not assume latest is best.
6. **Tournament objective audit.** Rerate a small set of checkpoints with both
   win/Elo and survival reported per game. If Elo winners are not survival
   winners, maintain two leaderboards or choose the promotion objective
   explicitly.

## Source Notes

Primary/web sources:

- MuZero paper: the algorithm combines tree search with a learned model and
  learns reward, policy, and value predictions for planning.
  <https://arxiv.org/abs/1911.08265>
- AlphaZero paper: tabula-rasa self-play from random play is the core training
  regime. <https://arxiv.org/abs/1712.01815>
- AlphaGo Zero Nature page: documents self-play, empirical evaluation, and
  tournament game artifacts. <https://www.nature.com/articles/nature24270>
- LightZero config docs: `main_config`/`policy` hold environment and learning
  settings; `collector_env_num`, `num_simulations`, `update_per_collect`,
  `batch_size`, `max_env_step`, `td_steps`, and `num_unroll_steps` are expected
  tuning knobs. <https://opendilab.github.io/LightZero/tutorials/config/config.html>
- LightZero MuZero agent source: creates `MuZeroGameBuffer`, collector,
  evaluator, and samples replay after each collect; `update_per_collect=None`
  falls back to collected transitions times `replay_ratio`.
  <https://opendilab.github.io/LightZero/_modules/lzero/agent/muzero.html>
- LightZero MuZero policy source: logs policy/value/reward losses and entropy
  style statistics, uses categorical value/reward support transforms, and has
  root Dirichlet noise / temperature / epsilon collection controls.
  <https://github.com/opendilab/LightZero/blob/main/lzero/policy/muzero.py>
- DI-engine buffer docs: replay buffer/PER are explicit data-pipeline choices,
  not invisible implementation details.
  <https://opendilab.github.io/DI-engine/04_best_practice/buffer.html>
- OpenSpiel AlphaZero docs: actors, learner, FIFO replay, checkpoints, evaluator
  games, and analysis plots for losses, value accuracy, evaluation results,
  actor speed, and game lengths.
  <https://openspiel.readthedocs.io/en/latest/alpha_zero.html>
- MiniZero official repo: its training architecture separates server,
  self-play workers, optimization worker, data storage, and keeps optimization
  steps proportional to collected self-play to avoid overfitting.
  <https://github.com/rlglab/minizero>
- Multi-agent replay nonstationarity: changing co-players can make replay data
  obsolete; suggested fixes include decaying obsolete data or conditioning on a
  replay-age fingerprint.
  <https://arxiv.org/abs/1702.08887>
- Replay fundamentals: replay capacity and replay ratio are independently
  important enough to study directly.
  <https://arxiv.org/abs/2007.06700>
- Reward shaping invariance: only specific reward transformations preserve the
  optimal policy in general; arbitrary dense shaping can change the learned
  objective. <https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf>
- EfficientZero: MuZero-derived visual RL adds value-prefix, consistency, and
  off-policy correction to improve sample efficiency, reinforcing that stale
  replay/targets are not a minor detail.
  <https://arxiv.org/abs/2111.00210>

Local sources:

- `survival_stagnation_investigation_2026-05-16.md`: current evidence summary
  and collapse observation.
- `run_reward_control_design.md`: current reward variants and warning against
  silently mixing reward recipes in replay.
- `caveats.md`: head-to-head strength is not the same as survival improvement.
- `docs/design/training/lightzero_stock_loop_contract.md`: current
  fixed/frozen-opponent LightZero claim labels.
- `docs/design/training/curvytron_learning_gates.md`: required curves for a
  learning claim.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`:
  active launcher, target support cap, assignment refresh, and stock
  `train_muzero` integration.
