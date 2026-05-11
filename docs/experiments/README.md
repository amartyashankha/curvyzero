# Experiments

Experiments should be reproducible enough that another agent can rerun or critique them later.

## Experiment Log Format

```md
# YYYY-MM-DD short-name

## Question

## Setup

## Command

## Results

## Interpretation

## Artifacts

## Follow-ups
```

## Early Experiment Backlog

- [2026-05-08 toy baseline smoke](2026-05-08-toy-baseline-smoke.md) - first
  local training-adjacent harness and structured artifact smoke.
- [2026-05-08 dummy survival smoke](2026-05-08-dummy-survival-smoke.md) - first
  single-player dummy training loop with checkpoint and metrics artifacts.
- [2026-05-09 dummy survival eval smoke](2026-05-09-dummy-survival-eval-smoke.md) -
  first fixed random/scripted EVAL1 baseline table for dummy survival.
- [2026-05-09 dummy survival checkpoint eval smoke](2026-05-09-dummy-survival-checkpoint-eval-smoke.md) -
  first learned-checkpoint comparison against dummy survival baselines.
- [2026-05-09 dummy survival learning curve local](2026-05-09-dummy-survival-learning-curve-local.md) -
  two small 20x50 local training runs checking whether more dummy survival data
  improves learned-checkpoint eval.
- [2026-05-09 dummy survival checkpoint sweep smoke](2026-05-09-dummy-survival-checkpoint-sweep-smoke.md) -
  periodic checkpoint selection smoke for dummy survival.
- [2026-05-09 dummy survival safety planner smoke](2026-05-09-dummy-survival-safety-planner-smoke.md) -
  first positive learned-checkpoint signal after adding a planner safety mask.
- [2026-05-09 dummy survival safety epsilon smoke](2026-05-09-dummy-survival-safety-epsilon-smoke.md) -
  minimal positive-clearance epsilon collection filter; later checkpoint
  degradation remained mixed and the best learned checkpoint regressed.
- [2026-05-09 dummy survival planner-only eval smoke](2026-05-09-dummy-survival-planner-only-eval-smoke.md) -
  added `untrained_model_same_planner` to the survival eval defaults and
  confirmed the planner prior alone can solve the tiny monitor split.
- [2026-05-09 dummy survival eval protocol fields smoke](2026-05-09-dummy-survival-eval-protocol-fields-smoke.md) -
  added split metadata plus selected/latest checkpoint and
  `selection_record.json` artifacts to survival eval/sweeps.
- [2026-05-09 dummy survival selection heldout smoke](2026-05-09-dummy-survival-selection-heldout-smoke.md) -
  added heldout confirmation from a selection record; selected beat latest but
  tied planner-only, so the claim stayed inconclusive.
- [2026-05-09 dummy survival degradation diagnosis](2026-05-09-dummy-survival-degradation-diagnosis.md) -
  diagnosed later checkpoint degradation as learned negative/unknown values
  overriding the planner safety prior.
- [2026-05-09 dummy survival unknown next value](2026-05-09-dummy-survival-unknown-next-value.md) -
  tried making unknown next states look bad; learned checkpoints still lost to
  planner-only, so survival should stay diagnostic for now.
- [2026-05-08 Modal dummy survival smoke](2026-05-08-modal-dummy-survival-smoke.md) -
  CPU Modal wrapper for the dummy survival training loop.
- [2026-05-09 Modal Volume dummy survival smoke](2026-05-09-modal-volume-dummy-survival-smoke.md) -
  first durable `curvyzero-runs` Volume artifact smoke for dummy survival
  training outputs.
- [2026-05-08 dummy line duel smoke](2026-05-08-dummy-line-duel-smoke.md) - first
  two-player dummy training loop with ego replay rows and checkpoint artifacts.
- [2026-05-09 dummy line duel eval smoke](2026-05-09-dummy-line-duel-eval-smoke.md) -
  first fixed random/scripted EVAL2 baseline matrix for Tiny Line Duel.
- [2026-05-09 dummy line duel checkpoint eval smoke](2026-05-09-dummy-line-duel-checkpoint-eval-smoke.md) -
  first learned-checkpoint comparison against Tiny Line Duel baselines.
- [2026-05-09 dummy line duel eval protocol fields smoke](2026-05-09-dummy-line-duel-eval-protocol-fields-smoke.md) -
  added split metadata and paired-seat group summaries to Tiny Line Duel eval.
- [2026-05-09 dummy pong eval smoke](2026-05-09-dummy-pong-eval-smoke.md) -
  first fixed random/scripted baseline matrix for a tiny Pong-like two-player
  toy environment.
- [2026-05-09 dummy pong observability smoke](2026-05-09-dummy-pong-observability-smoke.md) -
  compact deterministic Pong game/step/raster-frame traces for future visual
  observation debugging.
- [2026-05-09 dummy pong imitation replay smoke](2026-05-09-dummy-pong-imitation-replay-smoke.md) -
  first learner-ready Pong replay rows from `track_ball` targets over raster
  observations.
- [2026-05-09 dummy pong imitation replay v0](2026-05-09-dummy-pong-imitation-replay-v0.md) -
  larger 32-game Pong imitation replay; useful for copying `track_ball`, but
  all games timed out with zero score reward.
- [2026-05-09 dummy pong artifact inspector smoke](2026-05-09-dummy-pong-artifact-inspector-smoke.md) -
  compact inspection command for Pong replay/trace directories; confirmed the
  v0 replay has valid raster rows but no reward signal.
- [2026-05-09 dummy pong imitation train smoke](2026-05-09-dummy-pong-imitation-train-smoke.md) -
  first tiny supervised raster checkpoint that copies `track_ball`; source
  games all timed out with zero score reward, so this is not reward learning.
- [2026-05-09 dummy pong periodic policy checkpoints smoke](2026-05-09-dummy-pong-periodic-policy-checkpoints-smoke.md) -
  added `--checkpoint-every-epochs` for the Pong imitation trainer and verified
  reloadable `checkpoints/epoch-000NNN/checkpoint.npz` policy snapshots.
- [2026-05-09 dummy pong periodic checkpoint scoreboard smoke](2026-05-09-dummy-pong-periodic-checkpoint-scoreboard-smoke.md) -
  verified the Pong scoreboard can load periodic epoch checkpoints from one
  attempt; epoch 3 did slightly better against random, but both scored 0 wins
  against `track_ball`.
- [2026-05-09 dummy pong selection record smoke](2026-05-09-dummy-pong-selection-record-smoke.md) -
  added a local `selection_record.json` creator for Pong scoreboard summaries;
  selected `epoch_3` on the tiny monitor smoke by the existing eval rule, with
  heldout confirmation still required for quality claims.
- [2026-05-09 dummy pong selection record v2 smoke](2026-05-09-dummy-pong-selection-record-v2-smoke.md) -
  changed selector tie-breaks so zero-win `track_ball` rows prefer fewer losses
  and more truncations instead of accidentally rewarding faster losses.
- [2026-05-09 dummy pong imitation v0 periodic e1000 scoreboard](2026-05-09-dummy-pong-imitation-v0-periodic-e1000-scoreboard.md) -
  longer raster imitation run with epoch 250/500/750/1000 snapshots; epoch 1000
  beat random 42/64 and earlier checkpoints, but still won 0/64 against
  `track_ball`; the selection record picked epoch 1000 and marked heldout
  confirmation pending.
- [2026-05-09 dummy pong imitation v0 heldout scoreboard](2026-05-09-dummy-pong-imitation-v0-heldout-scoreboard.md) -
  heldout confirmation for selected epoch 1000; it beat previous 50/64 and
  survived longer against `track_ball`, but still won 0/64 against it.
- [2026-05-09 dummy pong checkpoint eval smoke](2026-05-09-dummy-pong-checkpoint-eval-smoke.md) -
  added `learned:<checkpoint.npz>` Pong eval support and confirmed the
  supervised raster checkpoint behaves like `track_ball` against `track_ball`
  in a tiny environment smoke.
- [2026-05-09 dummy pong checkpoint eval e32](2026-05-09-dummy-pong-checkpoint-eval-e32.md) -
  larger learned-checkpoint eval; learned beat random 43/64 but stayed well
  below scripted `track_ball`.
- [2026-05-09 dummy pong scoring replay smoke](2026-05-09-dummy-pong-scoring-replay-smoke.md) -
  first learner-ready Pong scoring replay from `track_ball` versus
  `random_uniform` in both seats, with nonzero score-delta rewards.
- [2026-05-09 dummy pong scoring replay all-ego smoke](2026-05-09-dummy-pong-scoring-replay-all-ego-smoke.md) -
  scoring replay option with both ego policies, giving positive and negative
  terminal reward rows.
- [2026-05-09 dummy pong scoring imitation train eval](2026-05-09-dummy-pong-scoring-imitation-train-eval.md) -
  tiny supervised raster learner trained from score-bearing scoring replay rows;
  learned beat random 44/64 but still lost 0/64 to scripted `track_ball`.
- [2026-05-09 dummy pong scoring all-ego imitation train eval](2026-05-09-dummy-pong-scoring-all-ego-imitation-train-eval.md) -
  all-ego scoring replay was useful for reward labels but worse for action
  cloning, because it mixed random and scripted action targets.
- [2026-05-09 dummy pong scoring all-ego value train smoke](2026-05-09-dummy-pong-scoring-all-ego-value-train-smoke.md) -
  first score-delta return backup plus tiny raster value regressor smoke; this
  proves value-target plumbing, not policy improvement or angle-control play.
- [2026-05-09 dummy pong checkpoint scoreboard smoke](2026-05-09-dummy-pong-checkpoint-scoreboard-smoke.md) -
  added the smallest Pong checkpoint scoreboard command; same-checkpoint
  latest/previous labels only prove learned-vs-learned plumbing.
- [2026-05-09 dummy pong checkpoint scoreboard distinct smoke](2026-05-09-dummy-pong-checkpoint-scoreboard-distinct-smoke.md) -
  compared the three existing learned Pong policy checkpoints; all beat random
  more often than not, but none beat scripted `track_ball`.
- [2026-05-09 dummy pong paddle angle smoke](2026-05-09-dummy-pong-paddle-angle-smoke.md) -
  confirmed top, center, and bottom paddle contacts produce different outgoing
  `ball_vy` values, making off-center returns the next mini North Star for
  beating `track_ball`.
- [2026-05-09 dummy pong angle control probe](2026-05-09-dummy-pong-angle-control-probe.md) -
  added a focused `angle_control` scripted probe; it creates off-center returns
  and beats random, but still only truncates against `track_ball`.
- [2026-05-09 dummy pong contact outcomes smoke](2026-05-09-dummy-pong-contact-outcomes-smoke.md) -
  first controlled near-contact dataset probe; top/center/bottom contacts
  changed outgoing `ball_vy`, but short score-delta returns stayed flat against
  same-state `track_ball`.
- [2026-05-09 dummy pong contact-pressure scoreability probe](2026-05-09-dummy-pong-contact-pressure-scoreability-probe.md) -
  sampled real `pong_reset_profile=contact_pressure` starts under sparse env
  reward; all 192 reset/opponent groups were action-sensitive, `lagged_track_ball_1`
  was scoreable in 46/64 groups, and default `track_ball` was scoreable in 0/64.
- [2026-05-09 dummy pong contact outcomes width9 h48](2026-05-09-dummy-pong-contact-outcomes-width9-h48.md) -
  narrowed the geometry to width 9 for 64 states over horizon 48; contact
  choices still changed outgoing `ball_vy`, but all score-delta returns tied at
  `0.0`.
- [2026-05-09 dummy pong lookahead relabel smoke](2026-05-09-dummy-pong-lookahead-relabel-smoke.md) -
  added short score-delta lookahead replay labels; angle-tie targets produced
  some `track_ball` pressure but still 0 learned wins.
- [2026-05-09 dummy pong lookahead angle-tie g32 h32](2026-05-09-dummy-pong-lookahead-angle-tie-g32-h32.md) -
  larger angle-tie lookahead attempt; it produced 442 non-`track_ball` labels,
  but all lookahead checkpoints still won 0/64 against `track_ball` and the
  selector kept imitation epoch 1000.
- [2026-05-09 dummy pong loss-delay lookahead smoke](2026-05-09-dummy-pong-loss-delay-lookahead-smoke.md) -
  first `--loss-delay-alpha 0.05` lookahead smoke; it produced 0
  non-`track_ball` targets and 0/32 learned wins against `track_ball`, with
  5/32 truncations.
- [2026-05-09 dummy pong lookahead depth-2 strict smoke](2026-05-09-dummy-pong-lookahead-depth2-strict-smoke.md) -
  added `--ego-sequence-depth 2`; strict smoke found 10 avoided-loss sequence
  labels, but 0 non-`track_ball` targets, so no training run.
- [2026-05-09 dummy pong self-play smoke](2026-05-09-dummy-pong-selfplay-smoke.md) -
  first correct loop shape for Pong: self-play replay, score-plus-longevity
  shaped training return, tiny policy/value update, loadable checkpoints, and
  scoreboard. Superseded as a next path by the gen2 failure and critique wave;
  it proves plumbing, not policy quality.
- [2026-05-09 dummy pong self-play generation 2 smoke](2026-05-09-dummy-pong-selfplay-gen2-smoke.md) -
  manual second generation from `selfplay50`; one checkpoint improved slightly
  against random, but every gen2 checkpoint lost to its parent and won 0 games
  against `track_ball`. Do not promote or scale it; use it as negative evidence
  for the repair-vs-baseline decision.
- [2026-05-08 Modal dummy line duel smoke](2026-05-08-modal-dummy-line-duel-smoke.md) -
  CPU Modal wrapper for the two-player dummy line-duel training loop.
- [2026-05-09 Modal run management helper smoke](2026-05-09-modal-run-management-helper-smoke.md) -
  local compile/import smoke for reusable Modal run/attempt/checkpoint helper
  refs and pointer manifest shapes.
- [2026-05-09 Modal dummy Pong scoreboard attempt import smoke](2026-05-09-modal-dummy-pong-scoreboard-attempt-import-smoke.md) -
  local compile/import smoke for the CPU Modal Pong checkpoint-scoreboard
  wrapper.
- [2026-05-09 Modal dummy Pong scoreboard attempt remote smoke](2026-05-09-modal-dummy-pong-scoreboard-attempt-remote-smoke.md) -
  first real CPU Modal Pong scoreboard run using `curvyzero-runs` checkpoint
  refs and Volume eval outputs.
- [2026-05-09 Modal Mctx dependency smoke](2026-05-09-modal-mctx-dependency-smoke.md) -
  CPU and cheap-GPU Modal smoke for pinned JAX/Mctx imports plus one tiny
  synthetic `gumbel_muzero_policy` search. Passed on CPU and an L4 GPU; this
  proves runtime viability, not training quality.
- [2026-05-09 Modal LightZero CartPole tiny train smoke](2026-05-09-modal-lightzero-cartpole-tiny-train-smoke.md) -
  stock LightZero CartPole MuZero dry config patch plus one brutally capped
  CPU trainer call. It returned a `MuZeroPolicy`; this proves the stock
  entrypoint can start and stop, not policy quality.
- [2026-05-09 Modal LightZero Pong dry config smoke](2026-05-09-modal-lightzero-pong-dry-config-smoke.md) -
  stock LightZero Atari Pong MuZero config capture plus tiny CPU dry patches.
  It does not instantiate Atari or call the trainer; real Pong should wait on
  an explicit ALE/Gym/ROM environment smoke.
- [2026-05-09 Modal LightZero Pong env smoke](2026-05-09-modal-lightzero-pong-env-smoke.md) -
  no-train stock Pong create/reset/step probe. After adding OpenCV headless,
  the LightZero/DI-engine env path reaches ALE and then fails at the explicit
  missing-ROM gate, so ROM approval/prep comes before stock Pong training.
- [2026-05-09 Modal LightZero Pong scale128 control](2026-05-09-modal-lightzero-pong-scale128-control.md) -
  official Atari Pong CPU scale rung after the train/load/eval mechanics gates:
  cap widened to 128 env/episode steps with `num_simulations=2`; training and
  eval completed, but both old and new `iteration_1` checkpoints stayed
  all-action-0 with 128-step return `-2.0`, so there was no policy-quality
  improvement.
- [2026-05-09 Modal LightZero Pong GPU512 control](2026-05-09-modal-lightzero-pong-gpu512-control.md) -
  official Atari Pong cheap-GPU rung on an L4. It produced checkpoints through
  `iteration_4`; final eval used all six Atari actions with no fallback, but
  the 128-step return still matched the action-0 baselines at `-2.0`.
- [2026-05-09 Modal LightZero Pong 4096 sim10](2026-05-09-modal-lightzero-pong-4096-sim10.md) -
  official Atari Pong staged L4 rung with `max_env_step=4096`,
  `num_simulations=10`, `batch_size=32`, and `collector_env_num=2`. Train/eval
  mechanics passed with strict no-fallback eval and checkpoints through
  `iteration_8`, but returns stayed `-6.0` for `iteration_0`, `iteration_4`,
  and `iteration_8`; by `iteration_4` the policy collapsed to action `5`, so
  this is an infrastructure pass and signal fail, not a sim25 gate. Follow-up
  eval parity promotes the manual eval as valid: obs shapes are correct,
  action `5` is ALE `LEFTFIRE`, the first 32 steps are all action `5`, and
  roots are action-5 dominated with high logits. The stock DI-engine evaluator
  path is blocked by missing `action_mask`; the remaining official issue is
  training/setup quality, not action mapping. See
  [official Atari collapse investigation](../working/lightzero_official_atari_collapse_investigation_2026-05-09.md):
  primary suspect is undertrained/off-recipe early learning, `-6` is mostly the
  256-step eval cap, and the next official step is either fix stock evaluator
  collation/API plumbing or train closer to the full official recipe.
- [2026-05-09 Modal LightZero Pong 8192 sim25](2026-05-09-modal-lightzero-pong-8192-sim25.md) -
  installed `LightZero==0.2.0` Atari Pong rung on L4, closer to stock but still
  far below the exact 200k/50-sim recipe. It completed and mirrored
  `iteration_0..932`, but selected strict no-fallback evals still collapsed to
  one action and capped return `-6`. This is a control run, not a learning
  result.
- [2026-05-09 Modal LightZero Pong 8192 sim25 parallel 512](2026-05-09-modal-lightzero-pong-8192-sim25-parallel512.md) -
  parallel 512-step eval for selected periodic checkpoints
  `0,100,500,900,932` from the same run. Every row used one action for all
  `512` steps, returned `-13`, had no positive rewards, matched stock/manual,
  and used no fallback. Raising the cap from `256` to `512` did not reveal
  hidden progress.
- [2026-05-09 Modal LightZero exact faithful-short 8192 relpath](2026-05-09-modal-lightzero-exact-faithful-short-8192-relpath.md) -
  installed `LightZero==0.2.0` exact-wrapper rehearsal on L4. This is
  faithful-short, not exact reproduction: only `exp_name` and
  `train_muzero.max_env_step 200000 -> 8192` were patched. Train finished with
  correct Volume-root checkpoints and no alternate roots. Corrected 512-step
  eval of `iteration_0` and final `iteration_3697` strict-loaded with no
  fallback. Manual telemetry returned `-13` for both with no positive rewards;
  stock `MuZeroEvaluator` code-path returns were `-13` and `-8`, but that eval
  still used tiny wrapper defaults such as `num_simulations=2`, so it is useful
  evidence rather than final scoring. Corrected stock-ish eval completed on
  app `ap-81xAvfiyvnU8flV3eElPSH`, eval id
  `faithful-short-periodic-stockish512-stockeval-s0-8192-relpath`. It used
  `num_simulations=50`, `evaluator_env_num=3`, `collector_env_num=8`,
  `batch_size=256`, `game_segment_length=400`, `max_env_step=200000`,
  `max_train_iter=1`, `update_per_collect=1`, CPU eval wrapper, and 512 cap.
  `iteration_0` scored manual/stock `-13/-13`; final `iteration_3697` scored
  manual/stock `-5/-8` with one positive reward. This is the first weak
  less-bad final signal, but it is one seed/two checkpoints and manual-stock
  mismatch remains. Not solved and not exact reproduction.
- [2026-05-09 Modal LightZero Pong checkpoint state diff](2026-05-09-modal-lightzero-pong-checkpoint-state-diff.md) -
  compared `iteration_932` with `ckpt_best` from the 8192/sim25 run. The model
  keys and shapes match, but `ckpt_best` has `last_iter=0`, `last_step=0`, no
  optimizer state, and reset-looking norm stats. Do not treat `ckpt_best` as
  quality evidence for that run.
- [2026-05-09 Modal LightZero official Connect4 sparse smoke](2026-05-09-modal-lightzero-official-connect4-sparse-smoke.md) -
  second official sparse board-game MuZero bot-mode sanity check. It passed one
  tiny Modal CPU train after mounting the local official `zoo` snapshot because
  `LightZero==0.2.0` did not ship Connect4; useful for sparse delayed reward
  plumbing, not a Pong proxy.
- [2026-05-09 LightZero dummy Pong post-deep-seed-fix run](2026-05-09-lightzero-dummy-pong-post-deep-seed-fix-run.md) -
  reran the 1024-step / 16-iteration CPU trust check after the authoritative
  dynamic-seed fix; train seed diversity passed, but strict independent MCTS
  scoring still lost to random and `track_ball`.
- [2026-05-09 LightZero dummy Pong lag1 shaped knob run](2026-05-09-lightzero-dummy-pong-lag1-shaped-knob-run.md) -
  data-scale sanity run using the exposed trainer knobs against
  `lagged_track_ball_1`, plus a small paired-seat MCTS scorecard. Telemetry
  stayed negative and this is not full two-policy self-play.
- [2026-05-09 LightZero dummy Pong sparse UPC25 sim8 run](2026-05-09-lightzero-dummy-pong-sparse-upc25-sim8-run.md) -
  higher update/replay sparse fixed-horizon probe after rung0/rung1 pure
  length failed; heldout MCTS still lost to lagged/random, `ckpt_best` was
  all-up, and final `iteration_50` was nearly all-up `[1156,0,74]` across
  learned baseline rows.
- [2026-05-09 LightZero dummy Pong contact-pressure curriculum smoke](2026-05-09-lightzero-dummy-pong-contact-pressure-curriculum-smoke.md) -
  implemented the explicit opt-in custom dummy Pong `contact_pressure` reset
  curriculum and ran a tiny Modal train plus matching MCTS scorecard. Env reward
  stayed sparse `+1/-1/0`; the implementation is a mechanical pass, but
  held-out `iteration_2` still collapsed with zero `down` actions and is not a
  policy-quality win.
- [2026-05-09 LightZero dummy Pong contact-pressure modest rung](2026-05-09-lightzero-dummy-pong-contact-pressure-modest-rung.md) -
  ran the one allowed same-curriculum contact-pressure diagnostic rung against
  `lagged_track_ball_1`. Trainer-side used all three actions, but held-out MCTS
  checkpoint rows still had `down=0`; final `iteration_3` collapsed versus the
  scoreable lagged target, so the rung stopped and the default-reset scorecard
  was skipped by the stop rule.
- [2026-05-09 LightZero dummy Pong target replay telemetry](2026-05-09-lightzero-dummy-pong-target-replay-telemetry.md) -
  custom dummy Pong target-sidecar smoke. The completed run wrote
  `target_replay_steps.jsonl` and `target_replay_summary.json`, separating
  executed action from target mass. It does not yet label oracle-winning action
  per row. The related support-scale patch is opt-in and custom dummy Pong
  only; planned proof fields are `patched_config.surface.*` and
  `compiled_config.policy_model_cfg.*`. Official Atari is untouched.
- [2026-05-09 Modal Mctx synthetic benchmark](2026-05-09-modal-mctx-synthetic-benchmark.md) -
  first small GPU Mctx benchmark on Modal L4. It passed with finite normalized
  action weights and measured compile/steady search times.
- [2026-05-09 Modal dummy Pong train attempt smoke](2026-05-09-modal-dummy-pong-train-attempt-smoke.md) -
  first CPU Modal Pong replay-plus-train attempt using `curvyzero-runs`. It
  saved replay rows, a train summary, and a checkpoint; this proves remote
  plumbing, not policy quality.
- [2026-05-09 Modal dummy Pong train to scoreboard smoke](2026-05-09-modal-dummy-pong-train-to-scoreboard-smoke.md) -
  loaded the Modal-trained checkpoint by Volume ref and scored it remotely.
  This proves the remote train-to-eval path; the one-game checkpoint was weak.
- [2026-05-09 Modal Pong self-play repair run](2026-05-09-modal-pong-selfplay-repair-run.md) -
  first small real Modal Pong self-play repair run with four checkpoints.
  Epoch 50 beat random 39/64, but every checkpoint still won 0/64 against
  `track_ball`; run a small sweep, then change learner/curriculum if this holds.
- [2026-05-09 Modal Pong parallel sweep](2026-05-09-modal-pong-parallel-sweep.md) -
  three cheap parallel self-play variants with periodic checkpoints; every
  checkpoint still won 0/64 against `track_ball`, so the next step should be a
  learner/curriculum change rather than more blind self-play scaling.
- [2026-05-09 Modal Pong self-play 512-game feasibility](2026-05-09-modal-pong-selfplay-512g-feasibility.md) -
  one fresh-data/undertraining probe with 512 random self-play games and
  e25/e50/e75 checkpoint scoreboards; the best row did not beat repair ckpt25
  on mean steps, truncation rate, or shaped loss-delay proxy.
- [2026-05-09 dummy pong survival curriculum smoke](2026-05-09-dummy-pong-survival-curriculum-smoke.md) -
  added a tiny on-policy raster trainer whose objective and artifacts make
  survival/loss-delay visible alongside wins. The smoke checkpoint beat random
  6/8 and still won 0/8 versus `track_ball`, but forced 2/8 truncations. Keep
  it as a fallback shape; the old self-play longer-run feasibility call is now
  closed.
- [2026-05-09 dummy pong geometry CEM smoke](2026-05-09-dummy-pong-geometry-cem-smoke.md) -
  tiny CEM/random-search baseline over the geometry slice of the raster linear
  policy; its checkpoint loads in the existing scoreboard, beats random 16/16,
  and survives to full truncation 16/16 against `track_ball` without winning.
- [2026-05-09 dummy pong track_ball beatability probe](2026-05-09-dummy-pong-track-ball-beatability-probe.md) -
  exact bounded search over all normal reset states and both ego seats for
  `PongConfig(width=15,height=9,paddle_height=3,max_steps=120)`; found 0/40
  reset/player cases where any legal ego action sequence can score against
  deterministic `track_ball`.
- [2026-05-09 dummy pong target ladder probe](2026-05-09-dummy-pong-target-ladder-probe.md) -
  compact exact sweep over weaker target policies, paddle/geometry tweaks, and
  biased starts; recommends default geometry plus `lagged_track_ball_1`, which
  is scoreable in 40/40 normal reset/player cases with median 19.0 winning
  steps.
- [2026-05-09 dummy pong CEM-v2 lagged-track-ball-1 monitor](2026-05-09-dummy-pong-cem-v2-lagged-track-ball-1-monitor.md) -
  ran the score-primary CEM-v2 monitor against `lagged_track_ball_1`; the
  checkpoint scoreboard confirmed real score pressure with 53/64 learned wins
  versus lag-1, 60/64 versus random, and 64/64 truncation ties versus default
  `track_ball`.
- [2026-05-09 Modal dummy Pong CEM-v2 train attempt](2026-05-09-modal-dummy-pong-cem-v2-train-attempt.md) -
  modalized the CEM-v2 lag-1 lane as one CPU train-attempt Function writing
  summary, checkpoint, rows, manifests, and a latest checkpoint pointer to
  `curvyzero-runs`; the existing Modal scoreboard loaded the checkpoint by
  Volume ref and reproduced the 53/64 learned wins versus lag-1.
- [2026-05-09 dummy pong lag-1 trace visual-policy smoke](2026-05-09-dummy-pong-lag1-trace-visual-policy-smoke.md) -
  first post-CEM learned visual-policy lane: exact lag-1 winning traces become
  raster supervised rows, then a checkpoint is scored versus lag-1, random, and
  default `track_ball`.
- [2026-05-09 dummy pong lag-1 trace visual-policy balanced smoke](2026-05-09-dummy-pong-lag1-trace-visual-policy-balanced-smoke.md) -
  added `--class-weighting balanced` to the raster imitation trainer; it fixed
  the near-all-`up` supervised action skew but only moved lag-1 wins from 5/16
  to 6/16 and worsened default `track_ball` survival.
- [2026-05-09 dummy pong lag-1 trace raster-only policy smoke](2026-05-09-dummy-pong-lag1-trace-raster-only-policy-smoke.md) -
  added the truthful `--feature-mode raster_only` ablation; checkpoints record
  a 675-wide one-hot raster feature axis with no decoded geometry suffix, score
  6/16 versus `lagged_track_ball_1`, 10/16 versus random, and 0/16 versus
  default `track_ball` with 2/16 truncations.
- [2026-05-09 dummy pong lag-1 trace visual-policy replay augmentation smoke](2026-05-09-dummy-pong-lag1-trace-visual-policy-replay-augmentation-smoke.md) -
  added data-side vertical mirroring and per-action oversampling options to the
  lag-1 replay builder; mirroring nudged lag-1 wins from 5/16 to 6/16 on the
  same seed, while full oversampling balanced predictions but did not improve
  gameplay.
- [2026-05-09 dummy pong lag-1 frame-stack visual-policy smoke](2026-05-09-dummy-pong-lag1-frame-stack-visual-policy-smoke.md) -
  added a `--frame-stack` raster history path through lag-1 trace replay,
  imitation checkpoints, and eval; a 2-frame stack lowered label accuracy
  versus stack 1 but nudged closed-loop behavior from 11/32 to 13/32 wins
  against `lagged_track_ball_1`, 12/32 to 15/32 versus random, and 4 to 6
  truncations versus default `track_ball`.
- [2026-05-09 Modal dummy Pong raster-only MLP train attempt](2026-05-09-modal-dummy-pong-raster-only-mlp-train-attempt.md) -
  ran the stack-2 `raster_only` MLP through the new Modal train-attempt wrapper
  and scored the returned Volume checkpoint with the Modal scoreboard:
  49/64 wins versus `lagged_track_ball_1`, 34/64 versus random, and 25/64
  truncations versus default `track_ball`.
- [2026-05-09 dummy pong lag-1 DAgger visual-policy smoke](2026-05-09-dummy-pong-lag1-dagger-visual-policy-smoke.md) -
  added the smallest closed-loop relabeling path: roll out a visual checkpoint,
  exact-label visited lag-1 states with the current opponent memory, append
  rows, retrain, and score; the first 22-row aggregation proved plumbing but
  stayed at 5/16 wins versus lag-1. A broader follow-up appended 1,200 labeled
  closed-loop rows from both seats, epsilon behavior, and two checkpoints, but
  only matched the mirror-only 6/16 lag-1 row in the balanced diagnostic while
  worsening random sanity.
- CurvyTron reference setup smoke.
- Deterministic simulator unit/golden tests.
- Random-agent throughput benchmark.
- Vectorized environment equivalence benchmark.
- PPO baseline learnability.
- Modal CPU image smoke.
- Modal GPU device smoke for JAX and PyTorch.
- Mctx synthetic benchmark across batch sizes and simulation counts.
Training coach runs should use a dated experiment log. The active lane list lives in [Training Experiment Backlog](../working/training_experiment_backlog.md).
