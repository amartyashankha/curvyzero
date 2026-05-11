# LightZero Pong Sparse Training Scale Ladder - 2026-05-09

Role: train-longer-under-sparse-settings design critic. No code. No pytest.

## Read

Sparse reward probably needs more training, but "just run longer" is not yet
the best explanation.

Current evidence:

- Official sparse TicTacToe bot-mode ran end to end on Modal. That proves the
  LightZero/Modal sparse terminal path can execute; it does not prove dummy
  Pong should learn under tiny settings.
- The sparse discrepancy audit says dummy Pong is closer to board-game bot mode
  than CartPole: hidden opponent, sparse terminal outcome, `to_play=-1`, and
  final-outcome targets. The important borrowed knobs are explicit horizon,
  `td_steps` to the outcome, `discount_factor=1`, sane support ranges, and much
  more collection/update volume.
- The horizon-fixed probe worked mechanically: `max_env_step=1024` stayed a
  training budget and `pong_episode_max_steps=120` was the episode horizon. It
  still produced all-up MCTS eval actions, so horizon hygiene alone did not fix
  learning.
- The sparse-settings probe also worked mechanically:
  `td_steps=120`, `discount_factor=1.0`, `num_unroll_steps=5`, and support
  ranges `[-5, 6]` ran cleanly. But the checkpoint was not a useful controller:
  heldout MCTS was weak, chose no `down`, and first-N rows had tied visits such
  as `[2, 3, 3]`.
- The train-longer-vs-bug note says action collapse is best explained by
  deterministic eval over weak/tied roots, while poor learning is best
  explained by underpowered/bad sparse config. A hard action-map bug is lower
  probability.

So the right answer is: test longer, but only as one rung in a ladder that can
falsify it quickly.

## Invariants For Every Rung

Keep these fixed unless the rung explicitly changes them:

```text
env dummy_pong_lag1
feature_mode tabular_ego
opponent_policy lagged_track_ball_1
pong_episode_max_steps 120
game_segment_length 50 for tiny probes, 120 for serious update/replay probe
td_steps 120
num_unroll_steps 5
discount_factor 1.0
reward/value support [-5, 6, 1]
seed family: train seed 9/10/11; heldout scorecard seed 1701
heldout opponents: lagged_track_ball_1, random_uniform, track_ball
independent scorecard horizon: 120
paired-seat scorecards only
```

Do not use trainer-side reward as the quality claim. Use trainer telemetry to
debug data collection, then gate on independent heldout MCTS scorecards and
first-N root diagnostics.

## Ladder

| Rung | Question | Train shape | Eval shape | Decision |
| --- | --- | --- | --- | --- |
| 0. Reproduce sparse tiny probe | Is the sparse-settings lane stable across one more seed? | `1024` env steps, `16` train iters, `num_simulations=8`, `batch_size=32`, `update_per_collect=8`, `n_episode=2` | Score `iteration_0`, `iteration_16`, and `ckpt_best` at 8 sims; first-N debug rows | Go only if the run is mechanically clean and not worse than the existing sparse probe. |
| 1. Pure 2x budget | Does same config improve with only more experience? | Same as rung 0 except `max_env_step=2048`, `max_train_iter=32` | Score `0/16/32/best` at 8 sims | If curves improve, longer is plausible. If flat/collapsed, stop treating length alone as the lever. |
| 2. Higher eval/search simulations | Is collapse mostly low-simulation tie behavior? | No new train required first; use rung 1 checkpoints | Same checkpoint and same first-N observations at `num_simulations=8,16,32,64`; then heldout scorecard at best sim count | Go to train-time sim increase only if ties/action collapse improve with more sims. |
| 3. Higher train-time simulations | Does better search during collection improve the learned roots? | `2048/32`, `num_simulations=16` or `25`, keep `batch_size=32`, `update_per_collect=8`, `n_episode=2` | Score `0/16/32/best` at 16 or 32 sims; first-N debug rows | Go only if root separation and heldout score improve, not just wall time. |
| 4. Higher update/replay | Is the learner starved on sparse terminal examples? | `2048/32`, best sim count from rung 2/3, `n_episode=4`, `batch_size=64` or `128`, `update_per_collect=25` then `50` if stable, `game_segment_length=120` | Score all checkpoints; compare to rung 1 at the same env-step budget | Go if heldout score improves at same env steps. If not, suspect objective/representation/curriculum. |
| 5. Checkpoint-curve eval | Is there a real learning curve rather than a lucky final checkpoint? | Use the best rung so far, then optionally `4096/64` only after gates pass | Score every mirrored checkpoint on the same heldout split; report curve table | Scale beyond `4096/64` only if the curve is directionally monotone and non-degenerate. |

## Concrete Run Commands

Rung 0, one-seed sparse-settings repro:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 1024 --pong-episode-max-steps 120 --max-train-iter 16 --num-simulations 8 --batch-size 32 --update-per-collect 8 --n-evaluator-episode 4 --collector-env-num 1 --evaluator-env-num 1 --n-episode 2 --game-segment-length 50 --td-steps 120 --num-unroll-steps 5 --discount-factor 1.0 --reward-support-min -5 --reward-support-max 6 --reward-support-delta 1 --value-support-min -5 --value-support-max 6 --value-support-delta 1 --seed 10 --run-id lz-dpong-sparse-h120-lag1-s10 --attempt-id train-1024x16-sparse-h120
```

Rung 1, pure 2x budget:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 2048 --pong-episode-max-steps 120 --max-train-iter 32 --num-simulations 8 --batch-size 32 --update-per-collect 8 --n-evaluator-episode 4 --collector-env-num 1 --evaluator-env-num 1 --n-episode 2 --game-segment-length 50 --td-steps 120 --num-unroll-steps 5 --discount-factor 1.0 --reward-support-min -5 --reward-support-max 6 --reward-support-delta 1 --value-support-min -5 --value-support-max 6 --value-support-delta 1 --seed 10 --run-id lz-dpong-sparse-h120-lag1-s10-2x --attempt-id train-2048x32-sparse-h120
```

Rung 3, higher train-time search:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 2048 --pong-episode-max-steps 120 --max-train-iter 32 --num-simulations 16 --batch-size 32 --update-per-collect 8 --n-evaluator-episode 4 --collector-env-num 1 --evaluator-env-num 1 --n-episode 2 --game-segment-length 50 --td-steps 120 --num-unroll-steps 5 --discount-factor 1.0 --reward-support-min -5 --reward-support-max 6 --reward-support-delta 1 --value-support-min -5 --value-support-max 6 --value-support-delta 1 --seed 10 --run-id lz-dpong-sparse-h120-lag1-s10-sim16 --attempt-id train-2048x32-sim16-sparse-h120
```

Rung 4, higher update/replay:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 2048 --pong-episode-max-steps 120 --max-train-iter 32 --num-simulations 16 --batch-size 64 --update-per-collect 25 --n-evaluator-episode 4 --collector-env-num 1 --evaluator-env-num 1 --n-episode 4 --game-segment-length 120 --td-steps 120 --num-unroll-steps 5 --discount-factor 1.0 --reward-support-min -5 --reward-support-max 6 --reward-support-delta 1 --value-support-min -5 --value-support-max 6 --value-support-delta 1 --seed 10 --run-id lz-dpong-sparse-h120-lag1-s10-upc25 --attempt-id train-2048x32-sim16-upc25-sparse-h120
```

Scorecard template after each train:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:<label>=ref:<checkpoint-ref> --episodes 8 --seed 1701 --split-id dummy_pong_sparse_h120_lag1_ladder_v0 --eval-id <eval-id> --max-env-step 120 --num-simulations <8|16|32|64> --run-id <run-id> --attempt-id <attempt-id>
```

Use `--episodes 4` only for cheap smoke checks. Use `--episodes 8` or `16`
before making a scale decision.

## Stop/Go Criteria

### Survival

Go:

- Trainer-side survival mean rises versus the sparse probe baseline
  (`22.83`) without seed dominance.
- Fresh recorded eval-wave survival mean improves on `random_uniform` and
  `lagged_track_ball_1`; target is `>=30` at 120-step horizon or an absolute
  improvement of `>=8` steps over the rung-0 multi-start scorecard.
- P90 survival reaches `>=60` against at least one non-`track_ball` opponent
  without all results being timeouts.

Stop:

- Survival stays near first-contact loss (`median <=10`) across rungs 0 and 1.
- Only trainer survival improves while heldout scorecard survival is flat.
- Survival improves only by collapsing to one action.

### Shaped Return

Go:

- Heldout shaped loss-delay return improves by `>=0.10` absolute versus
  rung 0 on `random_uniform` or `lagged_track_ball_1`.
- At least one of those two opponents reaches nonnegative shaped return.
- The checkpoint curve shows shaped return improving before or with raw score.

Stop:

- Shaped return is flat or worse from `1024/16` to `2048/32`.
- Shaped return improves in training but not in independent heldout eval.
- Shaped return is the only positive signal while raw score, survival, and
  action/root diagnostics remain degenerate.

### Raw Score / Heldout Scorecards

Go:

- Against `random_uniform`: score return is `>=0.0` on the heldout scorecard.
- Against `lagged_track_ball_1`: score return is `>=0.0`, or improves by
  `>=0.25` absolute versus rung 0.
- Against `track_ball`: do not require wins yet, but require survival/shaped
  return not to regress versus rung 0.
- Improvement appears in at least two checkpoints, not only `ckpt_best`.

Stop:

- The 2x run has the same or worse score than the tiny sparse probe on
  `random_uniform` and `lagged_track_ball_1`.
- Claimed progress comes only from trainer/evaluator returns, not independent
  checkpoint scorecards.
- `ckpt_best` improves but adjacent iteration checkpoints do not, suggesting
  scorecard noise or checkpoint-selection artifact.

### Action Entropy

Use normalized entropy over `[up, stay, down]`, `H(action) / log(3)`.

Go:

- Train action entropy is `>=0.55`.
- Heldout MCTS action entropy is `>=0.35` aggregated over
  `random_uniform + lagged_track_ball_1`.
- No action exceeds `85%` of heldout actions unless first-N diagnostics show
  state-dependent, untied root preference.
- All three actions appear in heldout eval by rung 3. A temporary missing
  action is acceptable in rung 0 only if visits/logits are tied rather than
  dead.

Stop:

- Any serious rung has a learned heldout histogram like `[all, 0, 0]`,
  `[0, all, 0]`, or `[0, 0, all]`.
- The missing action is the same across `8/16/32/64` scorecard simulations.
- Collect actions are diverse but eval actions collapse and first-N roots are
  still tied.

### Visit Ties And Root Strength

For first-N MCTS rows, log observation, logits, visits, selected action, and
whether the geometry calls for up/stay/down.

Go:

- At `num_simulations=16` or higher, tied max visits occur in `<25%` of first-N
  rows.
- Median top-1 minus top-2 visit gap is `>=2` at 16 sims, or `>=4` at 32 sims.
- Selected actions change with ball-paddle geometry rather than row order.
- Increasing simulations reduces ties and improves heldout score or survival.

Stop:

- Rows keep showing ties like `[2, 3, 3]` or equivalent at higher simulations.
- Deterministic argmax tie order explains most selected actions.
- More simulations only makes the same bad single action more certain.

### Checkpoint Curves

Go:

- `iteration_0 -> mid -> final` shows directionally better heldout shaped
  return, survival, and raw score.
- Action entropy stays non-degenerate while score improves.
- Root ties fall over time.
- The same qualitative curve appears on at least two train seeds.

Stop:

- No checkpoint curve: only final checkpoint eval is available.
- The curve is noisy with no positive slope from `0` to final.
- Best checkpoint selection disagrees with heldout scorecard quality.

## Interpretation Rules

Evidence for "needs longer":

- Rung 1 improves over rung 0 with no config changes except budget.
- Checkpoint curves improve smoothly.
- Action entropy remains healthy.
- Visit ties decrease or do not dominate selected actions.
- Heldout scorecards improve on `random_uniform` and `lagged_track_ball_1`.

Evidence for "bug/config still wrong":

- Rung 1 is flat or worse.
- Higher simulations do not reduce tie-driven collapse.
- Higher update/replay improves learner losses but not heldout scorecards.
- Train telemetry improves while independent checkpoint eval does not.
- Action collapse persists across seeds and scorecard simulation counts.

If rungs 0-4 fail these gates, stop sparse scaling and switch lanes: reward
curriculum, dense auxiliary target, representation audit, or wrapper trace
parity. A longer sparse run is only justified after this ladder shows that time
is the bottleneck rather than the excuse.

## 2026-05-09 Follow-Up Decision

UPC25 plus a smallest collection-distribution change was tested after this
ladder: random warmup / collect-time epsilon with true sparse `env.step`
reward preserved. It diversified trainer-side actions but failed the heldout
gate:

- Train Modal: `ap-MYxTxehyWrDFkygrQTGXEk`
- Scorecard Modal: `ap-jnREG1t3V0jyw8dsOqtbUB`
- Trainer actions: `[288,74,64]` for `[up, stay, down]`
- Heldout `iteration_50` aggregate learned actions across baseline rows:
  `[1158,0,50]`
- Heldout `iteration_50` raw score versus `lagged_track_ball_1` and
  `random_uniform`: `-0.25` and `-0.25`
- `ckpt_best`: all-up `[806,0,0]`

Read:
simple exploration/data-distribution alone did not fix sparse tabular Pong.
Do not repeat same-sparse-target runs with only random warmup or epsilon
changed. The next smallest learning move is a scoreable contact/angle
curriculum that changes reset/opponent distribution toward paddle-contact and
scoring-pressure states while preserving the true sparse reward. A shaped
auxiliary target is lower priority unless it is explicitly labeled as a
temporary target and kept out of `env.step()` reward and promotion metrics.

Implementation update: the first such curriculum is now an opt-in custom dummy
Pong reset profile, `pong_reset_profile=contact_pressure`; it is not stock
Atari Pong replication. Tiny Modal train `ap-bNRz3Mtil6apjX5w6tNZxa` and
matching MCTS scorecard `ap-XRyCAYWAN7F3ptvRAKRC0x` passed mechanically with
true sparse reward preserved. Held-out `iteration_2` still failed quality
criteria and used zero down actions, so the next rung must compare
`iteration_0`/final and require held-out raw/shaped/action improvement before
scaling.
