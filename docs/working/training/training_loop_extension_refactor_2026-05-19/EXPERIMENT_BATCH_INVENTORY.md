# Experiment Batch Inventory

Last updated: 2026-05-19

This is a side-lane inventory of recent local experiment manifests. It is read-only evidence for planning the next experiment batch; it is not the main refactor lane.

## Plain Summary

Recent 18-row batches mostly followed the same matrix:

- 3 reward variants;
- 3 opponent recipes;
- 2 noise modes.

Most training-scale knobs were fixed. The meaningful changes across batches were opponent source/control mode, initial policy, assignment refresh/static mixture mode, mortality/immortality recipe, and render/trail mode.

Later `cz26-full-20260517a` is not an 18-row matrix. It is a larger 136-run grid that fixed the main training scale and varied reward alpha, opponent recipe, noise, and leaderboard immortality.

## Fixed In Inspected 18-Row Batches

These appeared fixed in the inspected row manifests:

- `batch_size=32`;
- `collector_env_num=256`;
- `n_episode=256`;
- `evaluator_env_num=1`;
- `n_evaluator_episode=1`;
- `num_simulations=8`;
- `max_env_step=30000000`;
- `max_train_iter=300000`;
- `save_ckpt_after_iter=10000`;
- `commit_on_checkpoint=true`;
- `lightzero_eval_freq=0`;
- `skip_lightzero_eval_in_profile=true`.

Evidence examples:

- `artifacts/local/curvytron_tonight18_manifests/curvy-v2refresh18p-20260514b/curvy-v2refresh18p-20260514b.rows.jsonl`
- `artifacts/local/curvytron_tonight18_manifests/curvy-r18fresh-allv2-20260516a/curvy-r18fresh-allv2-20260516a.rows.jsonl`

## Varied Axes

Reward variants:

- `sparse_outcome`;
- `survival_plus_bonus_no_outcome`;
- `survival_plus_bonus_plus_outcome`.

Noise modes:

- clean;
- `straight_override_p10_repeat_p10`, meaning 0.1 straight override and 0.1 extra action-repeat probability.

Opponent recipes:

- Earlier v2 examples used recipes such as `blank5-wall5-rank2_25-rank1_65`, `blank20-wall5-rank1_75`, and `blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35`.
- R18 fresh/current examples used recipes such as `blank10-wall10-rank2_25-rank1_55`, `blank20-wall5-rank1_70-rank1imm5`, and `blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5`.

Control mode:

- scratch/no initial policy;
- bootstrap from leaderboard or curated checkpoint refs;
- assignment-backed opponent refs;
- inline static mixture specs;
- refresh-backed opponent assignment.

Refresh cadence:

- none/null for static assignment/control variants;
- 50 for `v2refresh18p`;
- 2000 for `v2real18`, R18 fresh, and latest-style variants.

Render/observation:

- `source_state_bonus_render_mode=simple_symbols` was fixed in inspected manifests.
- Trail mode changed from earlier `body_circles_fast` to later/current `browser_lines`.
- Later/current docs and rows point to `learner_seat_mode=random_per_episode`; older row artifacts often do not include that field, so exact historical mode is not recoverable from rows alone.

Mortality/immortality:

- Earlier v2 rows generally made only the wall-avoidant slice immortal, often around 5%.
- Later R18 recipes made blank and wall slices immortal, with optional `rank1imm5`.
- Later recipe total immortal fractions were roughly 20%, 25%, or 30%, depending on the recipe.

## Later Batch: `cz26-full-20260517a`

Local submit artifacts indicate 136 runs were submitted/spawned.

Fixed:

- L4/T4 CPU40;
- `collector_env_num=256`;
- `n_episode=256`;
- `batch_size=64`;
- `num_simulations=8`;
- `max_train_iter=300000`;
- `max_env_step=30000000`;
- `save_ckpt_after_iter=10000`;
- opponent assignment refresh every 2000;
- `learner_seat_mode=random_per_episode`;
- `browser_lines + simple_symbols`;
- shared rank-1 R18 fresh `iteration_180000` initial checkpoint.

Varied:

- Grid A: 4 recipes x 4 reward alphas x 3 noise settings x 2 leaderboard-immortality settings = 96 runs.
- Grid B: 10 recipes x `out50` x 2 noise settings x 2 immortality settings = 40 runs.

Evidence examples:

- `artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.json`
- `artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.rows.jsonl`
- `artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.submit_launch.json`
- `docs/working/training/r18fresh_postmortem_2026-05-16/EXPERIMENT_LOG.md`
- `docs/working/training/r18fresh_postmortem_2026-05-16/CZ26_BATCH_RATIONALE_REORIENTATION_2026-05-18.md`

## Canary / Proof Runs

`cz26c-e2e-20260516a`:

- one canary run;
- fixed L4/T4 CPU40, `collector_env_num=256`, `batch_size=64`, `num_simulations=8`, `max_train_iter=2500`, checkpoint every 100, refresh every 100;
- random learner seat;
- seeded from rank-1 R18 fresh checkpoint;
- row labels: `out100`, `n0`, `imm0`, `b20w05r1`.

`cz26c-e2eproof-20260516b` / `cz26c-e2eproof-20260516c`:

- one run each;
- tiny proof shape: `collector_env_num=8`, `batch_size=4`, `num_simulations=2`, `n_episode=8`, refresh every 5;
- GIF/background eval off;
- difference: `20260516b` used `max_train_iter=2000`, `20260516c` used `20000`.

## Optimizer/Profile Batch

`opt-current-hw-batch-20260516a` was an 8-run optimizer/profile batch, not the main training matrix.

Fixed:

- stock source-state fixed-opponent path;
- sparse outcome reward;
- fixed-straight opponent;
- CPU oracle observations;
- `num_simulations=8`;
- `source_max_steps=512`;
- `max_train_iter=96`;
- no-death profile;
- no eval/GIF/checkpoint expectation.

Varied:

- compute class: L4/T4 CPU40 vs H100 CPU40;
- collector count: 128 vs 256;
- learner batch size: 32 vs 64.

## Tournament/Eval Settings Seen Locally

Local bounded/latest tournament artifacts show:

- `policy_mode=eval`;
- `num_simulations=8`;
- `games_per_pair=21`;
- `max_steps=1048576`;
- `policy_batch_size=8`;
- balanced/random seating in current control docs;
- adaptive bounded pairing in recent artifacts, with examples like `pairs_per_round=300` and `active_pool_limit=100`.

Evidence examples:

- `artifacts/local/curvytron_no_tournament_control_20260516/source/r18fresh_bounded_latest.json`
- `artifacts/local/curvytron_v2real18_live/rating_config_after_seed67.json`
- `artifacts/local/cz26_analysis_2026-05-18/cz26_tournament_debug_summary.json`

## Important Caveats

- Opponent recipe percentages are collector-env or episode-reset control, not guaranteed learner mini-batch composition. Learner `batch_size` is replay sampling.
- Some older manifests do not record `learner_seat_mode`, so do not infer historical perspective behavior from absence.
- Static/control rows may use inline `opponent_mixture_spec` instead of an assignment preview by design.
- This pass used local docs/artifacts only; it did not refresh Modal or remote state.
