# Training State Index - 2026-05-09

Compact map for the current training docs. Use this to choose the right source;
do not copy experiment logs into this file.

Current correction: for today's Coach lane, read
`docs/working/coach_north_star_2026-05-10.md` first. This index contains older
05-09 lineage and should not override the north-star rules: Pong is the control
lane, stock steps survived is the first eval signal, CurvyTron reward is
survival-only, and fixed/frozen opponents are not current-policy self-play.

Active board for today:

- `docs/working/training_coach_active_board_2026-05-10.md` - short live board
  for active lanes, run ids, eval trigger, tooling tasks, and stop/go rules.
- `docs/working/coach_north_star_2026-05-10.md` - current compact truth for
  Pong control-lane reads, CurvyTron survival target, eval seeds, Modal
  artifacts, and fixed/frozen-opponent caveats.
- `docs/working/coach_reoriented_priorities_2026-05-10.md` - corrected
  plain-language priorities: Atari-style means LightZero-compatible visual env
  shape, not literal ALE; Pong is a control lane; CurvyTron is survival-first.
- `docs/working/pong_lane_reconciliation_2026-05-10.md` - plain-language
  decision note: official/control LightZero Pong is the primary
  reproduction/control lane; custom dummy Pong is only a bridge/debug lane.

## Current Clean Worldview

- Rabbit-hole prevention rule: define the lane before using its results.
  `Atari-style` means LightZero-compatible visual env shape, not literal ALE.
  `Official Atari Pong` means the ALE-backed LightZero Pong control lane.
  `Custom dummy Pong` means bridge/debug/telemetry unless a run explicitly says
  it is testing visual frame-stack discipline. `CurvyTron adapter` means a
  custom visual LightZero env, likely without ALE.
- Keep three lanes separate. Official/control LightZero Pong is the priority
  reproduction/control lane. Custom dummy Pong is a small bridge/debug lane,
  not a competing quality lane. CurvyTron is the project-owned target game.
  Do not mix their scores or claims.
- Corrected term: `Atari-style` means a LightZero-compatible visual env shape,
  not literal ALE. ALE is only the Atari emulator for real Atari ROMs. A
  CurvyTron adapter should stay non-ALE and provide stacked image frames,
  discrete ego actions, reward/done/info, reset/seed, `action_mask`,
  `to_play=-1`, and full joint-action logging.
- Current priority order: keep official/control LightZero Pong as the visual
  control lane and read it by stock steps survived versus same-run
  `iteration_0`; keep custom dummy Pong bridge/debug only; continue CurvyTron
  visual survival work with clear fixed/frozen-opponent labels.
- Read Pong checkpoints with survival-first scoring: same-run baseline, stock
  steps survived, stock episode length, eval cap, stock reward counts,
  stock return, manual/raw telemetry, action collapse, then notes. Survival
  means steps survived. Fraction means steps/max eval steps and is secondary.
  Never report only win/loss or return.
- `8192` final versus `32768 iteration_0` is not a regression comparison. The
  former is a completed run's later checkpoint. The latter is the active run's
  starting checkpoint. The old relpath `32768` run now has a later same-run
  normal checkpoint, `iteration_9092`, and it has been evaluated against
  `iteration_0` with strict no-fallback stock-ish settings.
- Same-run `32768` result:
  manifest
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/eval/live-32768-iteration0-vs-9092-stockish512-stockeval-s0/manifest_custom_steps512_seed0_20260510T003157Z.json`.
  Both checkpoints survived `512/512`, so `delta_steps_survived=0`.
  `iteration_9092` improved manual return `-13 -> -8`, stock return
  `-13 -> -10`, and nonzero negative rewards `13 -> 8`; positive rewards
  stayed `0`. Plain read: weak but real same-run improvement on losses and
  return, not solved Pong.
- Current main LightZero/Modal visual Pong signal-curve run:
  Modal app `ap-w73lIzdU6eUthaoGFfb6jy`, attempt
  `train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath`.
  This detached relaunch replaced the killed frequent-checkpoint attempt
  `train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-relpath`, which
  stopped when the local Modal client disconnected before learner iteration
  `1000`. The detached launch used `--detach`, so local client disconnect
  should not stop training. Expected checkpoints are `iteration_0`, then every
  `1000` learner iterations, plus the final checkpoint. Non-claim: there is no
  new quality result until strict same-run eval of a later detached-run
  checkpoint.
- Docs must stay ahead of memory. Every run report must say the plain claim and
  the plain non-claim before anyone treats it as evidence.
- Minimum run-doc wording:
  `Claim: ...`
  `Non-claim: ...`
  If those two lines are missing, the run is not ready to cite in a decision.
- Training rewards stay sparse unless a run is explicitly labeled as a shaped
  objective ablation. Survival and shaped loss-delay are required telemetry,
  not the default MuZero/LightZero reward target.
- A separate custom dummy Pong shaped-objective ablation smoke now exists:
  `pong-survival-shaped-loss-delay-alpha0.5-smoke8192-s0`, attempt
  `survival-shaped-loss-delay-alpha0.5-smoke8192-s0`, Modal app
  `ap-att7Gn5sZMB5uYkVoCSF1F`. It used `reward_mode=loss_delay`,
  `alpha/survival_weight=0.5`, and `truncation_bonus=0.0`; the training return
  was `win=+1`, `loss=-1 + alpha * survival_fraction`, `timeout=0`, while true
  score stayed logged separately. Treat it only as a shaped-objective ablation,
  not stock Atari or sparse-reward dummy Pong.
- Completed analysis of that shaped smoke: final eval improved steps survived
  but not convincing Pong skill. Against `random_uniform`, raw score stayed
  `0.25` while mean steps rose `12.125 -> 20.375`; against
  `weak_track_ball`, raw score moved `-0.75 -> -0.625` and steps
  `17.625 -> 34.375`; against `track_ball`, raw score stayed `-0.875` and
  steps rose `27.5 -> 34.375`. Final shaped eval returns were `0.273958`,
  `-0.548438`, and `-0.794271`. Final greedy eval collapsed completely to
  action `down` for all three opponents, so treat this as loss-delay plumbing
  evidence, not policy quality.
- A larger same-lane shaped run also completed:
  `pong-survival-shaped-loss-delay-alpha0.5-epochs24-s0`, attempt
  `survival-shaped-loss-delay-alpha0.5-epochs24-s0`, Modal app
  `ap-f5ftgocWh7HFoEPdwhEdFi`, checkpoint
  `training/dummy-pong-survival-shaped/pong-survival-shaped-loss-delay-alpha0.5-epochs24-s0/checkpoints/iteration-000024/checkpoint.npz`,
  checkpoint sha256
  `387f47c799e76230d9caf2ccbe716f87a5b44167cfe5b159e359fb338116e9c2`.
  It used the same `reward_mode=loss_delay`, `alpha/survival_weight=0.5`,
  and `truncation_bonus=0.0`. Final eval kept the same raw scores as the
  8-epoch smoke (`0.25`, `-0.625`, `-0.875`) but mean steps were lower:
  `14.875`, `19.6875`, and `27.5`. Final greedy eval collapsed to action
  `up` for all opponents. Read: artifact lineage and shaped-objective
  plumbing, not a policy-quality improvement.
- Future short LightZero wrapper runs can lower checkpoint spacing with
  `--save-ckpt-after-iter-override`; leaving it unset keeps the stock cadence.
- `ckpt_best` is not quality evidence unless checkpoint state proves it is a
  real trained checkpoint. Prefer normal `iteration_*.pth.tar` checkpoints for
  live quality reads.

## Current Next Actions

1. Poll the detached frequent-checkpoint run
   `train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath`
   for the first normal checkpoint after `iteration_0`.
2. Eval that later checkpoint once with strict load, no fallback, same
   stock-ish 512-step settings, and baseline deltas versus same-run
   `iteration_0`.
3. Report survival-first rows: stock return, manual return, steps survived,
   survival fraction, reward counts/timing, positive rewards, action collapse,
   CPU/GPU placement, checkpoint ref, and manifest ref. Put
   `delta_steps_survived` before any fraction delta.
4. For custom dummy Pong, do not launch a quality run. If keeping it warm, use
   it only for bridge/debug work and inspect target sidecars plus compiled
   support-scale proof fields before any longer same-config run.
5. For CurvyTron, prototype the LightZero-compatible visual adapter: stacked
   image frames, discrete ego action, reward/done/info, reset/seed,
   `action_mask`, `to_play=-1`, and full joint-action logging. Do not turn the
   next CurvyTron step into another RL theory or framework-choice rabbit hole.

## 1. Current Truth And Next Lanes

Start here:

- `docs/working/training_coach_handoff_2026-05-09.md` - restart packet and
  current truth. Treat as the highest-level state summary after memory wipe.
- `docs/working/coach_optimizer_reorientation_2026-05-09.md` - optimizer
  correction for the coach worldview: LightZero remains a serious
  replication/control/bridge obligation. Historical note: this doc emphasized
  a parallel repo-native simultaneous `[B, P]` lane; the current next CurvyTron
  step is the LightZero-compatible visual adapter prototype above.
- `docs/working/shared_training_reporting_contract_2026-05-09.md` - shared
  metadata/reporting contract for LightZero and repo-native PPO reports.
- `docs/working/lightzero_live_eval_notes_2026-05-09.md` - how to eval
  checkpoints from the Modal Volume while training is still running.
- `docs/working/training_experiment_backlog.md` - active lane ledger and run
  lineage. It is long; use the top sections first.
- `docs/experiments/README.md` - dated evidence index. Use it to find full
  experiment records.

Current read:

- Two separate Pong lanes are active under the LightZero umbrella. Official
  Atari Pong is stock LightZero reproduction/control work using ALE because it
  is a real Atari ROM. Custom dummy Pong is our small controllable bridge/debug
  lane toward CurvyTron. Do not mix their results.
- Repo-native PPO/CleanRL-style actor-loop work is a third architecture lane
  for CurvyTron shape and timing. It is a parallel architecture probe, not a
  replacement for LightZero replication.
  The actor-loop dry run is `scripts/repo_native_ppo_actor_loop_dry_run.py`.
  The promoted on-policy learner smoke is
  `scripts/repo_native_ppo_learner_smoke.py`: optional Torch, same tiny
  actor-critic collects and is updated, `[T,B,P]` rollout arrays are
  preserved, one masked PPO update runs, artifacts are written, and the local
  smoke reported `masked_action_violations=0`. It is a no-quality boundary
  smoke, not a learning result and not a LightZero replacement.
- Framework reliability deep dive:
  `docs/working/rl_framework_reliability_deep_dive_2026-05-09.md`. Stars are
  not the decision rule. Historical layered approach: repo-native PPO baseline,
  PettingZoo-style API as adapter, LightZero as contained MuZero control, and
  Mctx later only if we own the MuZero trainer/search stack. Current next
  CurvyTron task is narrower: build the LightZero-compatible adapter prototype.
- Wrapper/self-play clarification: source timing is not fatal to the LightZero
  lane. A single-ego wrapper can advance the game with a wrapper joint
  action/control snapshot while LightZero controls one seat. True live all-player
  self-play still needs explicit opponent/checkpoint/seat/metadata handling.
  LightZero/MuZero supports self-play; the open question is API and game-shape
  fit for our CurvyTron path.
- Optimizer correction: do not let LightZero become the CurvyTron architecture
  by default, and do not demote it to "just a reference." Keep LightZero as a
  serious replication/control lane for stock Pong-like MuZero behavior, bridge
  tests, target audits, checkpoint/eval plumbing, and comparison. Older
  architecture probes remain evidence, but the current CurvyTron next task is
  the visual adapter prototype with stacked frames, discrete ego actions,
  `action_mask`, `to_play=-1`, and full joint-action logging. This is not a
  final framework decision. Reports must emit comparable profile metadata,
  contracts, timing buckets, throughput, latency, checkpoint ids, seed/reset
  details, and explicit non-claims.
- Setup fidelity verdict: official Atari work has partially followed the
  LightZero path as an official-path smoke, not an exact upstream reproduction.
  No exact full LightZero train has run yet. Exact installed-package dry repro
  now passes for installed `LightZero==0.2.0`: app
  `ap-Xz1gqGamx5CX0tCZfKknk8`, artifact
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/dry-exact-installed-0.2.0-stock-surface-v2/dry_exact_summary.json`.
  It mutates only `exp_name`; stock surface is 200k env steps, 50 sims,
  8 collectors, 3 evaluators, batch 256, segment 400, and no episode caps.
  Wrapper hardening smokes also passed the faithful-short dry path and CPU
  train guard. After the first GPU faithful-short attempt exposed an artifact
  path mismatch, the wrapper was patched to set `exp_name` as a relative Volume
  ref and to `chdir` to `/runs` in train mode. Relative-exp faithful-short dry
  app `ap-BJS7mWOdsqqzbA6DafF82z`, attempt
  `dry-faithful-short-installed-0.2.0-8192-relative-exp`, summary
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/dry-faithful-short-installed-0.2.0-8192-relative-exp/dry_exact_summary.json`,
  sha256 `bbff8da5746cb8999f03e8e28f4e26ba5ba05b361dd8e17be3d49de7afd791b8`,
  ok `true`;
  CPU guard app `ap-OaX6ZKqwp3JwMHzwXEsKCy`, attempt
  `train-cpu-guard-installed-0.2.0`, failed by design with
  `call_policy=blocked_cpu_train_before_train_muzero`, summary
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-cpu-guard-installed-0.2.0/train/summary.json`,
  sha256 `168722d09056fe705208a5d857f21a7b9278061c6624c1fc50d7b0a032b236b8`.
  The first GPU faithful-short attempt `ap-lDxY0C7O0GGDwu3jjxuMaI`,
  `train-faithful-short-installed-0.2.0-s0-8192`, was killed after
  `Training Iteration 400` because checkpoints were logging to `.//runs/...`
  while progress snapshots under `/runs/...` saw no checkpoints; make no
  quality or eval claim. The relpath GPU faithful-short train then completed:
  app `ap-ipdfYJmWQitQtIBxrKU2E9`, attempt
  `train-faithful-short-installed-0.2.0-s0-8192-relpath`, summary
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/train/summary.json`,
  sha256 `c97dc26094462ec17d1dd970370d86e392433a8059aed9b1eaea1e5614ed2a06`.
  Train ok `true`, GPU L4, torch CUDA true, actual max env step `8192`,
  collector overshot to `14791` env steps in one batch, remote elapsed about
  `1326s`, and checkpoint bytes were `256,613,692`. Checkpoints landed under
  the intended Volume root only: `ckpt_best`, `iteration_0`, and final
  `iteration_3697`. This is faithful-short, not exact reproduction; patches
  were `exp_name` for Modal artifacts and `train_muzero.max_env_step 200000 ->
  8192`.
  Corrected eval app `ap-ov622Yu6wEnN74V2Laf8HG`, manifest
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/eval/faithful-short-periodic-custom512-stockeval-s0-8192-relpath/manifest_custom_steps512_seed0_20260509T232832Z.json`.
  Eval used `iteration_0` and `iteration_3697` only, strict load true,
  fallback false, and 512 manual steps for both. The earlier low eval was
  invalid because manual `max_episode_steps` stayed `64` while stock used
  `512`. The corrected stock-ish eval completed: app
  `ap-81xAvfiyvnU8flV3eElPSH`, eval id
  `faithful-short-periodic-stockish512-stockeval-s0-8192-relpath`. It used
  `num_simulations=50`, `evaluator_env_num=3`, `collector_env_num=8`,
  `batch_size=256`, `game_segment_length=400`, `max_env_step=200000`,
  `max_train_iter=1`, `update_per_collect=1`, CUDA false because the eval
  wrapper ran on CPU, manual cap `512`, and `max_episode_steps=512`.
  `iteration_0`: manual return `-13`, stock return `-13`, `512` steps, `13`
  nonzero rewards, `0` positive rewards, dominant action `0` share `0.521484`,
  entropy `0.805545`, manual/stock match false. `iteration_3697`: manual
  return `-5`, stock return `-8`, `512` steps, `7` nonzero rewards, `1`
  positive reward, dominant action `0` share `0.714844`, entropy `0.644585`,
  manual/stock match false. Plain read: first weak signal that final is less
  bad than init under stock-ish eval, but one seed/two checkpoints and
  manual-stock mismatch remain. Not solved, not exact reproduction.
  Current GitHub config says about `500000` env steps, while installed
  `LightZero==0.2.0` in our Modal image says `200000`. The latest installed
  0.2.0 near-upstream rung
  reached `8192` env steps with `num_simulations=25`, still far below the full
  recipe. Its selected strict no-fallback stock `MuZeroEvaluator`/manual parity
  eval curve still did not show credible learning on periodic checkpoints. The
  latest parallel 512-step eval of periodic checkpoints
  `0,100,500,900,932` kept the same conclusion: every checkpoint chose one
  action for all `512` steps, returned `-13`, had no positive rewards, matched
  stock/manual, and used no fallback.
  This failed signal does not indict LightZero. See
  `docs/working/lightzero_setup_fidelity_audit_2026-05-09.md` and
  `docs/working/training_setup_red_team_2026-05-09.md`.
- Replaced frequent-checkpoint Modal rung:
  `train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-relpath`, app
  `ap-7Wd3QGsjT0RucAc2DC26mS`, launched around `2026-05-10T00:24:21Z`.
  Modal reported `Running (1/1 containers active)` after object creation and
  worker assignment, then stopped before the first useful periodic checkpoint.
  It has been replaced by detached attempt
  `train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath`,
  app `ap-w73lIzdU6eUthaoGFfb6jy`, which is now the current main
  LightZero/Modal visual Pong signal-curve run. Both attempts are the same
  installed `LightZero==0.2.0` official Atari Pong control lane with stock
  reward/control semantics: no episode caps, no collector/evaluator count
  changes, no `update_per_collect` change, and no `max_train_iter` cap. The
  deliberate rehearsal patches are only
  `train_muzero.max_env_step 200000 -> 32768`, relative `exp_name` for Modal
  Volume artifacts, and
  `policy.learn.learner.hook.save_ckpt_after_iter 10000 -> 1000` for observability.
  Current detached progress snapshots should update every `120` seconds under
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath/train/progress/latest.json`.
  Expected normal checkpoint cadence is startup `iteration_0`, then roughly
  every `1000` learner iterations (`iteration_1000`, `iteration_2000`, ...),
  plus the after-run final checkpoint. With `replay_ratio=0.25` and a
  `32768` env-step cap, expect the final useful checkpoint around
  `iteration_8200` to `iteration_9500`, allowing for collector overshoot. Do
  not make a learning/no-learning claim until later detached-run checkpoints are
  strict-loaded and evaluated against detached-run `iteration_0`.
  Launch note: first app `ap-15WcpcwIbSoNSZkx0kUxJf` failed before training with
  `KeyError('learn')` in the new cadence override path. The wrapper was patched
  to create `policy.learn.learner.hook` when setting
  `save_ckpt_after_iter`, matching the older tiny-train override pattern; the
  replacement app above started training but then stopped early. Failure read:
  progress artifact
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-relpath/train/progress/progress_failed_20260510T002955Z.json`
  reports `phase=failed`, timestamp `2026-05-10T00:29:55.477415Z`,
  `train_elapsed_sec=340.736058747`, `actual_save_ckpt_after_iter=1000`,
  `checkpoint_count=2`, and newest checkpoints only
  `ckpt/iteration_0.pth.tar` plus `ckpt/ckpt_best.pth.tar`. The saved
  `formatted_total_config.py` confirms `save_ckpt_after_iter=1000`,
  `max_env_step=32768`, `collector_env_num=8`, `evaluator_env_num=3`,
  `num_simulations=50`, `batch_size=256`, `game_segment_length=400`, and
  stock episode caps `108000`. Learner log reached `Training Iteration 400`
  at `2026-05-10 00:29:50`; there is no `iteration_1000.pth.tar` and no
  training summary/final checkpoint visible under the attempt root. Modal app
  logs give the plain cause: `Stopping app - local client disconnected. Use
  modal run --detach to keep apps running even if your local client
  disconnects.` The later `KeyboardInterrupt`/broken-pipe traces are shutdown
  fallout from subprocess env workers, not evidence of a model-training
  exception.
  Claim: the frequent-checkpoint attempt stopped before its first useful
  periodic checkpoint, so no strict same-run later-checkpoint eval was possible.
  Non-claim: this is not a Pong learning result, not evidence that the
  `1000`-iteration checkpoint cadence failed, and not evidence that LightZero
  cannot train Pong.
- Previous active Modal rung at prior handoff:
  `train-faithful-short-installed-0.2.0-s0-32768-relpath`, app
  `ap-xiGLACKHPZLvL1eYgygqvm`. This is a bounded 32768-step
  faithful-short scale/accounting probe, not a learning claim. Recent local
  log poll: learner reached iteration `3000`; the first collect batch reported
  `14822` environment steps, `16` episodes, average episode length about
  `926`, and train-time reward mean about `-20.3`. Earlier polls showed only
  `iteration_0.pth.tar` and `ckpt_best.pth.tar`, but the later normal
  checkpoint `iteration_9092.pth.tar` is now visible and evaluated. The strict
  no-fallback same-run eval manifest is
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/eval/live-32768-iteration0-vs-9092-stockish512-stockeval-s0/manifest_custom_steps512_seed0_20260510T003157Z.json`.
  Baseline-delta read: both `iteration_0` and `iteration_9092` survived
  `512/512`, so `delta_steps_survived=0`; `iteration_9092` improved manual
  return `-13 -> -8`, stock return `-13 -> -10`, and nonzero negative rewards
  `13 -> 8`, with positive rewards still `0`. CPU live eval of the earlier
  `iteration_0`-only pass completed with strict load true, fallback false,
  manual/raw return `-13`, stock return `-12`, `512` episode steps survived,
  `13` nonzero rewards, `0` positive rewards, and survived longer than
  `iteration_0`: baseline/not applicable. GPU proof eval app
  `ap-3icJTrptdJEw38GZAoK5wx` also completed on compute `gpu-l4-t4`: strict
  load true, fallback false, CUDA true on NVIDIA L4, manual/stock return
  `-13/-13`, `512` episode steps survived, `13` nonzero rewards, `0` positive
  rewards, action `2` for all steps, verdict `collapsed_action`. Survival time
  is first-class for this lane: every later eval must report episode steps
  survived, nonzero reward count, positive reward count, raw return, and
  whether it survived longer than same-run `iteration_0`. Treat the
  `iteration_9092` result as weak but real same-run improvement on losses and
  return, not a solve and not a comparison against the earlier `8192` final
  checkpoint. Use
  `docs/working/lightzero_next_scale_run_plan_2026-05-09.md` for stop rules
  and `docs/working/lightzero_live_eval_notes_2026-05-09.md` for live eval.
  For future short rehearsals, the train wrapper now supports
  `--save-ckpt-after-iter-override` so a run can save more often than stock.
- Latest orientation docs: `docs/working/pong_two_lane_worldview_2026-05-09.md`
  and `docs/working/pong_official_vs_custom_source_map_2026-05-09.md`.
  Recommendation: keep both lanes; merge only reporting/eval discipline
  (plain lane labels, run/checkpoint refs, independent eval, action/survival
  telemetry); do not claim custom dummy Pong or `raster_flat` single-frame MLP
  results are visual Atari parity.
- Transfer doc for CurvyTron:
  `docs/working/training_lessons_for_curvytron_2026-05-09.md`. CurvyTron
  should inherit the lane discipline, target-sidecar observability, survival
  telemetry, visual-history requirement, support-scale proof, and Modal
  checkpoint/eval artifact pattern; it should not inherit Pong result claims.
- Process correction: `docs/working/training_process_critique_2026-05-09.md`
  is the current lane-discipline note. Official Atari reproduction and custom
  root-target parity dominate; checkpoint archaeology is support work only.
- Visual bridge decision: keep `tabular_ego` as the custom debug/learning lane.
  Retire `raster_flat` to smoke-only. The next honest visual bridge is a
  separate `raster_stack4_ego` path with frame history before any CNN or
  Atari-parity claim.
- Stock visual Atari Pong now passes train/checkpoint-load/eval mechanics on
  Modal. The staged official Atari 4096/sim10 L4 run also completed with
  strict no-fallback eval and checkpoints through `iteration_8`, but the eval
  curve stayed at return `-6` for `iteration_0`, `iteration_4`, and
  `iteration_8`; by `iteration_4` the policy collapsed to action `5`. This is
  an infrastructure pass and a signal fail, not solved Pong. The follow-up
  official eval parity probe promotes the manual eval result as valid:
  strict-load/no-fallback is real, raw/policy observation shapes are correct,
  action `5` is ALE `LEFTFIRE`, the first 32 eval steps are all action `5`,
  and roots are dominated by action `5` with high logits. The stock evaluator
  gap for matching 64x64 checkpoints is now fixed: a tiny Modal smoke uses
  LightZero `MuZeroEvaluator` and matched the manual action sequence. The
  remaining official issue is training/setup quality, not action mapping. The
  follow-up installed `LightZero==0.2.0` 64x64 non-segment Atari rung did climb
  to `8192/sim25` on L4 and completed mechanically: dry app
  `ap-VasQbApDzGd18EaB38hM59`, train app `ap-qnwMaN8FlOUNJwLNo1mZKs`, run
  `lz-visual-pong-8192-sim25-s0`, attempt
  `train-8192-sim25-b64-env4-auto`, `ckpt_best` plus `iteration_0` through
  `iteration_932` mirrored. The promoted strict no-fallback eval curve ran for
  `iteration_0`, `iteration_100`, `iteration_500`, `iteration_900`,
  `iteration_932`, and `ckpt_best`. Periodic checkpoints strict-loaded, and
  manual/stock first-32 actions matched; every periodic checkpoint collapsed to
  one action and returned `-6`. `ckpt_best` manual eval reached `0` with
  diverse actions, but manual/stock first-32 did not match and stock return
  stayed `-6`, so treat `ckpt_best` as a parity warning, not quality evidence.
  The follow-up checkpoint state diff made this sharper: `ckpt_best` has the
  same model key set and shapes as `iteration_932`, but is smaller and
  reset-looking (`last_iter=0`, `last_step=0`, empty optimizer state, norm
  counters `0`, running means `0`, running vars/weights `1`). Treat it as a
  bad best-checkpoint artifact for this run, not a learned policy.
  The later parallel 512-step eval for periodic checkpoints
  `0,100,500,900,932` confirmed the 256-step read: each row stayed on one
  action for all `512` steps, returned `-13`, saw `13` negative rewards and no
  positive rewards, matched stock/manual, and used no fallback.
  Adoption risk: restoring stock `update_per_collect=None`
  caused a huge learner/checkpoint burst, around 934 checkpoint files / roughly
  90 GB, so bounded replication needs explicit update and checkpoint
  accounting discipline before any larger train. Do not reopen action-mapping
  rabbit holes. Pretrained OpenDILab Pong strict eval remains separately
  blocked by a model-surface mismatch: strict load still sees unexpected
  `representation_network.downsample_net.conv2.weight`, so no eval has run.
  Treat this as the older 96x96/downsample checkpoint versus current 64x64
  stock config/eval path until a matching config/checkpoint pair is found.
  Collapse investigation:
  `docs/working/lightzero_official_atari_collapse_investigation_2026-05-09.md`.
  Current read: the near-upstream installed-0.2.0 run still did not show
  credible learning. The next LightZero question is why the best-checkpoint
  save path produced a reset-looking file, and whether to spend on an exact
  upstream/pretrained config path.
- LightZero dummy Pong train, checkpoint mirroring, independent MCTS
  scorecards, and frozen-checkpoint opponent plumbing run on Modal.
- Dummy Pong checkpoints still do not show reliable held-out improvement.
- Current risk: custom dummy Pong may be too far from stock LightZero examples
  and assumptions, so discrepancy mapping is now active.
- Target-semantics correction: LightZero MuZero trains policy logits toward the
  MCTS root visit distribution, not toward the exploratory action that was
  finally executed. Exploration can execute `down` without making `down` the
  policy target. Custom dummy Pong target audit: action id `2` (`down`) is
  legal and correct in states that need down, but with `sims=2` the MCTS root
  visits can be `[1,1,0]`, so the stored policy target can omit the winning
  `down` action. Target replay telemetry is now validated on custom dummy Pong
  Modal collection; see
  `docs/experiments/2026-05-09-lightzero-dummy-pong-target-replay-telemetry.md`.
  The completed safe smoke `ap-rdvkRpLGRYedx39SggsVvm` wrote
  `target_replay_steps.jsonl` and `target_replay_summary.json` sidecars with
  16 rows / 1 episode / 1 collect call. The target-sidecar read in
  `docs/working/lightzero_dummy_pong_target_sidecar_read_2026-05-09.md`
  confirms the sidecars now separate executed action from target mass:
  executed `down` appears on 9/16 rows, while target mass for `down` is nonzero
  on 12/16 rows and zero on 4/16 rows. The sidecars do not yet label the
  oracle-winning action per row. Safe telemetry-smoke config is
  `game_segment_length >= 16` and `batch_size >= 2`; shorter segment or
  batch-1 variants hit replay sampling or learner-shape failures. For custom
  dummy Pong, the live blocker is root visit/target quality plus action
  collapse, not another longer same-config run.
- Do not run the same dummy Pong sparse config longer. The first explicit
  custom dummy Pong `contact_pressure` reset curriculum is implemented and has
  one tiny Modal train plus matching MCTS scorecard; it is a mechanical pass,
  not a policy-quality win. The one allowed modest same-curriculum
  contact-pressure rung also ran and stopped: trainer-side used all three
  actions, but held-out MCTS rows for `iteration_0`, `iteration_3`, and
  `ckpt_best` all had `down=0`, with final `iteration_3` collapsed versus the
  scoreable lagged target. Next learning moves are strict eval/accounting for
  the official Atari 8192/sim25 checkpoint burst, raster-flat debugging,
  discrepancy mapping, a bug/objective investigation for dummy Pong action
  collapse, and broad bug hunt. Simple
  exploration/data-distribution under the same sparse target is closed as the
  next fix.
- CartPole and official board-game smokes are controls only.

## Active Parallel Wave

Current coordination map; owners are conceptual lanes, not agent names:

- Custom dummy Pong action-collapse/root-target parity - owner: dummy Pong
  diagnostics.
  Active blocker. Sparse/UPC/simple-exploration/contact-pressure modest rungs
  are stopped unless this root-cause lane changes the premise. Check root visit
  targets and value/support scale before any new same-config run.
- Official Atari meaningful scale - owner: official Atari scale. Active blocker.
  Stock Atari train/checkpoint-load/eval mechanics work, and the official
  4096/sim10 L4 rung proved the widened wrapper can run the more expensive
  shape: `num_simulations=10`, `batch_size=32`, `collector_env_num=2`,
  `update_per_collect=2`, `game_segment_length=64`, and 512-step episode caps.
  It produced checkpoints through `iteration_8` and strict no-fallback evals
  for `iteration_0`, `iteration_4`, and `iteration_8`, but all three stayed at
  return `-6`; `iteration_4` and `iteration_8` were all action `5`. Treat this
  as infrastructure pass / signal fail. Manual eval parity is now validated:
  observation shapes are correct, action `5` is ALE `LEFTFIRE`, the first 32
  eval steps are all action `5`, and root logits/visits are genuinely
  action-5 dominated. The matching-checkpoint stock evaluator path now uses
  LightZero `MuZeroEvaluator` and a tiny Modal smoke matched manual actions, so
  this is no longer an action mapping or evaluator-collation investigation.
  The later installed-0.2.0 8192/sim25 non-segment Atari run completed on L4:
  dry app `ap-VasQbApDzGd18EaB38hM59`, train app
  `ap-qnwMaN8FlOUNJwLNo1mZKs`, `ckpt_best` plus `iteration_0..932`. Its strict
  no-fallback eval curve now exists for `iteration_0`, `iteration_100`,
  `iteration_500`, `iteration_900`, `iteration_932`, and `ckpt_best`.
  Periodic checkpoints passed manual/stock first-32 parity, collapsed to one
  action, and returned `-6`. `ckpt_best` manual eval reached `0` with diverse
  actions, but stock return stayed `-6` and manual/stock first-32 did not
  match. A checkpoint state diff shows `ckpt_best` is reset-looking:
  `last_iter=0`, `last_step=0`, empty optimizer state, norm counters `0`,
  running means `0`, and running vars/weights `1`, while `iteration_932` has
  trained counters and nonzero running stats. Current meaning: mechanics and
  selected parity are real, but no credible learning signal. The trainer-side
  `-21` and capped eval `-6` should
  not be compared as if they were the same score. Next official step: explain
  the trainer/evaluator/best-selection disagreement or invest in exact
  upstream/pretrained config, and keep update/checkpoint accounting discipline
  because stock `update_per_collect=None` produced a surprisingly large
  learner/checkpoint burst. Eval batching exists now; use it for every future
  checkpoint curve through the existing parallel checkpoint-eval wrapper in
  `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py`, paired with
  checkpoint retention rules. Parallel eval smoke passed on app
  `ap-GroNH8bnBAadark30VLY51` with manifest
  `training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_curve_parallel_smoke/manifest_low_steps32_seed0_20260509T202709Z.json`;
  it proves batching workflow only, not 32-step return quality. Pretrained
  OpenDILab strict eval remains
  blocked separately by older 96x96/downsample checkpoint surface versus the
  current 64x64 config: strict load still reports unexpected
  `representation_network.downsample_net.conv2.weight`, so no eval has run.
- Stock-vs-custom discrepancy mapping - owner: training diagnostics. Active
  risk. Official Atari Pong is stock reproduction/control; custom dummy Pong is
  the CurvyTron bridge. Map where custom env/action/replay/target behavior
  diverges from stock LightZero before assuming more scale will transfer. Start
  with `docs/working/pong_two_lane_worldview_2026-05-09.md` and
  `docs/working/pong_official_vs_custom_source_map_2026-05-09.md`.
- Docs/backlog hygiene - owner: training coordination. Keep stopped dummy Pong
  sparse/UPC/exploration/contact-pressure results visible without reopening
  them as default next runs.

## 2. Official Controls

Use these to understand what LightZero/Modal can execute, not to claim Pong or
CurvyTron policy quality:

- `docs/experiments/2026-05-09-modal-lightzero-cartpole-tiny-train-smoke.md`
  - stock CartPole train entrypoint control.
- `docs/experiments/2026-05-09-modal-lightzero-official-sparse-example-sanity.md`
  - official sparse TicTacToe bot-mode control.
- `docs/experiments/2026-05-09-modal-lightzero-official-connect4-sparse-smoke.md`
  - official sparse Connect4 bot-mode control.
- `docs/working/lightzero_official_example_pattern_choice_2026-05-09.md` and
  `docs/working/lightzero_cartpole_pong_discrepancy_audit_2026-05-09.md` -
  pattern analysis for why board-game controls are useful but not Pong proof.

## 3. Stock Visual Atari Pong

This is official LightZero Atari Pong, not the custom dummy Pong task:

- Plain-language terms: `official Atari Pong` means LightZero's built-in Pong
  example using the Atari emulator. `ALE` is the Atari emulator used by
  Gym/LightZero. A `checkpoint` is a saved model file. This lane is a sanity
  check for LightZero's normal visual training path, not dummy Pong or
  CurvyTron.
- `docs/experiments/2026-05-09-modal-lightzero-pong-env-smoke.md` - ALE/Gym
  env gate and ROM surface.
- `docs/experiments/2026-05-09-modal-lightzero-pong-tiny-train-smoke.md` -
  capped visual Atari Pong train smoke passed on Modal.
- `docs/experiments/2026-05-09-modal-lightzero-pong-checkpoint-load-smoke.md`
  - mirrored visual checkpoint strict-load and forward probe passed.
- `docs/working/lightzero_official_visual_pong_pattern_2026-05-09.md` - working
  synthesis for the official visual path.

Latest eval read: `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py` has
telemetry/detail caps and passed on Modal `ap-S0HADSUdYxYsy6y1yGj4mP`. It ran
64 real ALE eval steps through `MuZeroPolicy.eval_mode.forward`, no fallback,
`num_simulations=2`; actions `{0:64}`, rewards `{-1.0:1,0.0:63}`, total reward
`-1`, nonzero reward at step 60, done at step 63 with
`TimeLimit.truncated:true`, raw obs `[1,64,64]`, policy stack `[4,64,64]`.
Artifact:
`training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/eval/iteration_1_tiny/lightzero_visual_pong_eval_smoke_20260509T173516Z.json`.
This answers basic official visual eval mechanics, not policy quality.

Latest scale read: the CPU scale128 official Atari rung widened wrapper caps to
`max_env_step=128`, `max_train_iter=2`, and 128-step collect/eval caps while
staying on stock `zoo.atari.config.atari_muzero_config` with
`num_simulations=2`. Train app `ap-qoTln2RP7Ly65hjCK3V4On` completed, but
LightZero still saved only `iteration_0` and `iteration_1`. New eval app
`ap-xARflZIivWVe3TdtHD4vEL` and old-baseline eval app
`ap-D0OopWDZJD8K191krNFBBC` both produced the same 128-step result:
actions `{0:128}`, return `-2.0`, rewards at steps `60` and `95`, terminal
`TimeLimit.truncated:true`. This is real post-64-step Atari reward signal, but
no improvement over the all-action-0 tiny checkpoint. See
`docs/experiments/2026-05-09-modal-lightzero-pong-scale128-control.md`.

Next gate: do not rerun scale128 for quality. Either raise `max_env_step` enough
to actually produce a later official Atari checkpoint, or add a clean GPU
option before increasing MCTS/update cost.

Latest GPU/later-checkpoint read: `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`
now names the cheap GPU resource as `["L4", "T4"]` and records runtime CUDA
telemetry. The official Atari GPU512 rung ran on Modal `ap-NcECoDQrcIfrbpqmBRODbP`,
run `lz-visual-pong-20260509T180945Z-29b83d6ee638`, attempt
`attempt-20260509T180945Z-dc971b1ec0ff`, with actual GPU `NVIDIA L4`,
`policy.cuda=true`, `max_env_step=512`, `max_train_iter=4`, 128-step episode
caps, `num_simulations=2`, and `batch_size=8`. It completed and mirrored
`iteration_0` through `iteration_4`. Final checkpoint eval on Modal
`ap-AUNehXPKKdkXbPOCW5WM7B` used `iteration_4` for 128 real ALE steps with no
fallback: action histogram `{0:21,1:24,2:22,3:25,4:22,5:14}`, return `-2.0`,
rewards at steps `60` and `95`, terminal `TimeLimit.truncated:true`. This
clears the wrapper/GPU/cap blocker and breaks all-action-0, but still does not
beat the 128-step action-0 baseline. See
`docs/experiments/2026-05-09-modal-lightzero-pong-gpu512-control.md`.

Prior GPU1024 official Atari read: the cheap-GPU L4 control with
`max_env_step=1024`, `max_train_iter=8`, `batch_size=8`,
`max_episode_steps=256`, `game_segment_length=16`, and `num_simulations=2`
produced checkpoints through `iteration_4`. The final checkpoint eval at
256 real ALE steps improved same-cap return from the GPU512 baseline `-5` to
`-3` and recorded one `+1` reward. This is the first small positive signal in
the official visual reproduction lane. It is still not solved Atari Pong and
must not be mixed with custom dummy Pong.

GPU2048 official Atari read: the follow-up cheap-GPU L4 official Atari
run completed on L4, produced checkpoints through `iteration_8`, and the
same-cap final eval return stayed `-6`. It did not reinforce GPU1024's small
`-3` signal, but the audit conclusion is that GPU2048 is smoke-scale rather
than a failed official reproduction. The biggest mismatches versus official
Atari settings are `max_env_step=2048` vs about `500000`,
`num_simulations=2` vs `50`, one env, `batch_size=8`,
`game_segment_length=16`, and `update_per_collect=1`. Stop doing naive
same-shape official Atari scale; the next reproduction should move toward the
official shape, especially MCTS simulations and env/update scale, not just
double the same wrapper caps.

Latest official Atari 4096/sim10 read:
`docs/experiments/2026-05-09-modal-lightzero-pong-4096-sim10.md`. The staged
L4 run used `max_env_step=4096`, `num_simulations=10`, `batch_size=32`,
`collector_env_num=2`, `update_per_collect=2`, `game_segment_length=64`, and
`max_episode_steps=512`. Train completed on an L4 with CUDA and mirrored
`ckpt_best` plus `iteration_0` through `iteration_8`. Strict no-fallback evals
for `iteration_0`, `iteration_4`, and `iteration_8` all returned `-6` over the
256-step cap with no positive rewards. `iteration_0` used `{0:238,1:18}`;
`iteration_4` and `iteration_8` collapsed to `{5:256}`. This is official Atari
Pong only, an infrastructure pass, and a signal fail. Do not run sim25/sim50
from this result. The eval parity probe says the manual eval path is valid:
strict-load/no-fallback is real, obs shapes are correct, action `5` is ALE
`LEFTFIRE`, the first 32 steps are all action `5`, and action `5` dominates
the roots with high logits. The matching-checkpoint stock evaluator path now
uses LightZero `MuZeroEvaluator`; a tiny Modal smoke fixed the missing
`action_mask`/collation gap and matched manual actions. Next official Atari
work is train closer to the full official recipe or use a matching pretrained
checkpoint/config pair; do not do more action mapping or evaluator plumbing.
Pretrained OpenDILab strict eval remains separately blocked by the older
96x96/downsample checkpoint surface versus the current 64x64 config: strict
load still reports unexpected
`representation_network.downsample_net.conv2.weight`, so no eval has run.

Official Atari collapse investigation:
`docs/working/lightzero_official_atari_collapse_investigation_2026-05-09.md`.
Primary suspect is an undertrained, noisy early learner on an off-recipe config:
only `iteration_8`, 4096 env steps, 10 sims, batch 32, UPC 2, and 2 collectors.
The flat `-6` is mostly a 256-step eval-cap artifact; it is not a full Atari
Pong episode score. Manual eval parity is now accepted as the official read:
strict checkpoint load, no fallback, correct observation shapes, and real
action-5 root/action collapse. The matching-checkpoint evaluator path now uses
LightZero `MuZeroEvaluator` and matched the manual action sequence. The
remaining official question is training/setup quality plus bounded
update/checkpoint accounting.

Latest official Atari 8192/sim25 read:
`docs/experiments/2026-05-09-modal-lightzero-pong-8192-sim25.md`. This is the
installed `LightZero==0.2.0` 64x64 non-segment Atari lane, not GitHub-main
exact upstream, not the 96x96 pretrained lane, not custom dummy Pong, and not
CurvyTron. Dry config validation passed on Modal
`ap-VasQbApDzGd18EaB38hM59`. Train completed on L4 app
`ap-qnwMaN8FlOUNJwLNo1mZKs`, run `lz-visual-pong-8192-sim25-s0`, attempt
`train-8192-sim25-b64-env4-auto`, with `max_env_step=8192`,
`num_simulations=25`, `collector_env_num=4`, `batch_size=64`,
`game_segment_length=128`, `max_episode_steps=1024`, and stock
`update_per_collect=None` restored through the wrapper sentinel. It mirrored
`ckpt_best` plus `iteration_0` through `iteration_932`. The strict no-fallback
stock `MuZeroEvaluator`/manual parity eval curve ran for `iteration_0`,
`iteration_100`, `iteration_500`, `iteration_900`, `iteration_932`, and
`ckpt_best`. Periodic checkpoints strict-loaded, manual/stock first-32 actions
matched, each collapsed to one action, and each returned `-6`. `ckpt_best`
manual eval reached `0` with diverse actions, but manual/stock first-32 did
not match and the stock evaluator return stayed `-6`; this is a parity warning,
not quality evidence. Do not compare the trainer-side `-21` headline with the
capped strict-eval `-6`; they are different score surfaces. Adoption risk:
stock `update_per_collect=None` caused a much larger learner/checkpoint burst
than the `max_train_iter=64` label suggested, including roughly 934 checkpoint
files and about 90 GB of artifacts. Bounded replication now needs explicit
accounting for learner updates, checkpoint cadence, retained checkpoint count,
and eval selection before another train. The iteration loop bottleneck is now
obvious enough to design around: eval selected checkpoints in embarrassingly
parallel jobs via the existing `lightzero_pong_eval_smoke.py` parallel
checkpoint-eval wrapper, and retain only the checkpoints needed for audit,
curve, and rollback. Use this batched eval path for every future checkpoint
curve.
Current next step is not sim50 or another larger train: decide whether to
investigate why trainer/evaluator/best selection and periodic checkpoints
disagree, or invest in an exact upstream/pretrained config path.

## 4. Dummy Pong Raster

This is custom dummy Pong with `feature_mode=raster_flat`, separate from stock
Atari and separate from tabular sparse runs:

- `docs/experiments/2026-05-09-lightzero-dummy-pong-raster-flat-smoke.md` -
  train plus matching independent raster MCTS scorecard passed mechanically.
- `docs/working/lightzero_pong_action_collapse_bug_hunt_2026-05-09.md` and
  `docs/working/lightzero_pong_broad_bug_hunt_2026-05-09.md` - bug-hunt
  context for action collapse and eval/training adapter checks.

Current read: raster-flat mechanics work, but the scorecard was not a quality
win. Learned actions were `[424,146,0]` with zero `down`, and held-out rows
were bad versus random and `track_ball`. `tabular_ego` likely already contains
enough core Pong state for this toy task; single-frame `raster_flat` is weaker
because it lacks velocity/history.

## 5. Dummy Pong Sparse Failures And Stop Signals

These are custom dummy Pong `tabular_ego` sparse fixed-horizon runs:

- `docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-settings-probe.md`
  - sparse knobs exposed and first fixed-horizon probe.
- `docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-scale-ladder-rung0-rung1.md`
  - rung 0 and pure 2x rung 1; final/best checkpoints collapsed all-up.
- `docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-upc25-sim8-run.md`
  - higher update/replay at the same horizon; held-out scorecard still failed.
- `docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-upc25-epscollect-run.md`
  - UPC25 plus random warmup/epsilon/temperature improved train action
  diversity but still failed held-out.
- `docs/working/lightzero_pong_sparse_training_scale_ladder_2026-05-09.md` -
  ladder plan plus stop criteria.

Stop signals:

- Pure same-config 2x did not improve held-out survival, shaped return, raw
  score, or action entropy.
- Higher update/replay alone did not fix the sparse lane.
- Random warmup/epsilon/temperature improved train action diversity, but did
  not fix held-out quality.
- More eval simulations alone are not the fix; 16+ removed ties but collapsed
  action choice.
- Simple exploration/data-distribution under the same sparse target is closed
  as the next fix.
- Support/value scale is suspicious: requested small support ranges may be
  recorded in summaries while the compiled LightZero policy may still use
  `support_scale=300`. The support-scale patch status is opt-in and custom
  dummy Pong only; official Atari is untouched. Proof fields planned for the
  next custom run are `patched_config.surface.*` and
  `compiled_config.policy_model_cfg.*`. Verify those compiled-config fields
  before trusting support ablations.
- The next sparse move has begun as a scoreable contact/angle curriculum via
  reset distribution: `pong_reset_profile=contact_pressure`. It is opt-in
  custom dummy Pong curriculum, not stock Atari Pong replication, and true env
  reward stays sparse `+1/-1/0`. Tiny train `ap-bNRz3Mtil6apjX5w6tNZxa`
  passed, but matching MCTS scorecard `ap-XRyCAYWAN7F3ptvRAKRC0x` showed
  `iteration_2` still had zero down actions and failed held-out
  lagged/random/track rows. Continue only with the same survival/shaped/raw/action
  telemetry and stricter go criteria.
- Contact-pressure scoreability probe:
  `docs/experiments/2026-05-09-dummy-pong-contact-pressure-scoreability-probe.md`.
  Real contact-pressure starts are action-sensitive under sparse reward
  (192/192 reset/opponent groups), but scoreability is opponent-dependent:
  `lagged_track_ball_1` 46/64, `stay` 59/64, default `track_ball` 0/64. Go
  only for a narrow lagged/simple-opponent curriculum diagnostic; stop using
  `track_ball` as the scoreable contact-pressure target.
- Modest contact-pressure diagnostic rung:
  `docs/experiments/2026-05-09-lightzero-dummy-pong-contact-pressure-modest-rung.md`.
  Train app `ap-Zr829nRQJqi3WqnTUEwHwr`; scorecard app
  `ap-r5iWQT58qLeLGLIDQ4kDUM`; run
  `lz-dpong-20260509T175407Z-77159cc3a6b4`. The trainer-side row used all
  three actions, but held-out learned MCTS rows all had zero `down`. On the
  scoreable `lagged_track_ball_1` target, `ckpt_best` improved over init
  (`score 0.0625`, `shaped 0.0981`, survival mean `24.625`), but final
  `iteration_3` collapsed (`1/16` wins, score `-0.625`, `down=0`). The stop
  rule triggered; no default-reset scorecard and no campaign.
- Custom dummy Pong target-audit facts: action id `2` (`down`) is legal and
  correct in down-needed states. The failure is not that down is illegal. With
  `sims=2`, MCTS root visits can be `[1,1,0]`, so the policy target can omit
  the winning `down` action. Target replay telemetry sidecars now write in a
  completed custom dummy Pong Modal smoke: `ap-rdvkRpLGRYedx39SggsVvm` wrote
  16 rows / 1 episode / 1 collect call in `target_replay_steps.jsonl` plus
  `target_replay_summary.json` with `game_segment_length=16` and
  `batch_size=2`. Old runs did not persist replay `child_visit`/action
  segments. The next custom Pong sidecar should add or join row-level
  oracle-winning-action labels and include the opt-in support-scale proof
  fields: `patched_config.surface.*` and
  `compiled_config.policy_model_cfg.*`. Do not start another training campaign
  until the emitted targets and compiled support scale are inspected.

## 6. Self-Play And Frozen Checkpoint

The current working bridge is learner ego versus an env-owned frozen
checkpoint opponent. It is not live current-policy two-seat self-play:

- Source timing mechanics are not the blocker by themselves. A single-ego
  wrapper can advance the game by assembling a wrapper `joint_action`/control
  snapshot from the LightZero-controlled seat plus scripted, frozen-checkpoint,
  or other opponent seats. What is not solved is true live all-player self-play:
  current-policy opponents, checkpoint lineage, seat metadata, replay ownership,
  and eval separation must be explicit. LightZero/MuZero has self-play support;
  our question is whether its APIs fit this game shape cleanly.

- `docs/working/lightzero_pong_frozen_checkpoint_selfplay_plan_2026-05-09.md`
  - design, telemetry contract, and latest synthesis.
- `docs/working/lightzero_pong_live_selfplay_feasibility_2026-05-09.md` -
  boundary between frozen-checkpoint bridge and true live self-play.
- `docs/experiments/2026-05-09-lightzero-dummy-pong-frozen-checkpoint-selfplay-smoke.md`
  - first frozen `iteration_0` opponent smoke.
- `docs/experiments/2026-05-09-lightzero-dummy-pong-frozen-checkpoint-selfplay-iter16.md`
  - later frozen `iteration_16` opponent smoke plus tiny parent comparison.

Current read: frozen-checkpoint plumbing and telemetry work. The smokes are not
policy-quality wins.

## 7. Eval And Telemetry Rules

Stable protocol sources:

- `docs/design/training_eval_protocol.md` - stable eval protocol.
- `docs/working/lightzero_pong_survival_reward_audit_2026-05-09.md` -
  compact current audit. Training reward is sparse `+1/-1/0`; survival length
  and shaped loss-delay are required telemetry, not the default MuZero reward.
- `docs/experiments/2026-05-09-dummy-pong-scoreboard-telemetry-patch-smoke.md`
  - survival and shaped loss-delay telemetry patch smoke.
- `docs/working/lightzero_dummy_pong_scorecard_summary_automation_2026-05-09.md`
  - compact command for comparing the recent sparse, UPC25, epscollect,
  contact-pressure, and raster dummy Pong scorecards when their summary refs
  are locally mounted or fetched. The automation exists, but local Modal Volume
  access is still incomplete; do not assume every referenced summary is locally
  readable yet.
- `docs/working/lightzero_trainer_scorecard_mismatch_2026-05-09.md` and
  `docs/working/lightzero_pong_checkpoint_diagnostics_2026-05-09.md` -
  why trainer-side telemetry is not final held-out checkpoint quality.
- `docs/research/reward_shaping_for_pong_curvy_muzero.md` - reward-shaping
  rule: keep env reward true, use loss-delay shaping only as telemetry or a
  clearly labeled temporary target.

Every Pong report must include wins/losses/timeouts, score return, survival
mean/median/p90, survival variance or standard deviation, truncation rate,
shaped loss-delay return, shaped-return variance or standard deviation, action
histograms/entropy, seed information, checkpoint refs, eval split, and artifact
refs. Never reduce a Pong checkpoint to only win/loss or return; survival
length is the cleaner early signal when policies still lose.

## 8. Open Decisions

- Official visual Pong: basic train/checkpoint/load/eval mechanics are answered.
  CPU scale128 found no improvement over all-action-0. GPU512 on a cheap L4
  produced later checkpoints and broke all-action-0. GPU1024 then produced a
  small same-cap `-3` signal, but GPU2048 completed through `iteration_8` and
  same-cap eval stayed `-6`. Do not make a continuing scale claim; this is a
  stop signal for naive same-shape official Atari scale, not a failed official
  reproduction. The reproduction gap is still mostly settings scale:
  2048 vs 500k env steps, 2 vs 50 simulations, one env, batch 8, segment
  length 16, and `update_per_collect=1`.
- Dummy Pong visual bridge: do not scale `raster_flat`; it is smoke-only after
  showing mechanics without reliable `down`. Keep `tabular_ego` for debug and
  learning. Build a separate `raster_stack4_ego` lane with frame history before
  any CNN or Atari-parity claim.
- Dummy Pong sparse: choose one objective/curriculum or data-distribution
  change; do not rerun same-config sparse length, update/replay-only, or simple
  exploration/data-distribution probes under the same sparse target.
- Plain current next step: scale/eval the official/control LightZero Pong lane
  because the same-run `iteration_9092` versus `iteration_0` result improved
  losses and returns, even though it did not solve Pong. Keep custom dummy Pong
  bridge/debug only; shaped reward plumbing works, but policy quality is
  bad/collapsed. For CurvyTron, build the adapter prototype instead of opening
  another RL theory or framework-choice thread.
- Self-play: keep frozen-checkpoint bridge as the immediate staged path; design
  true current-policy simultaneous self-play separately. Do not treat
  simultaneous play as fatal to LightZero; use a single-ego joint-action wrapper
  for bridge work, but require explicit opponent/checkpoint/metadata handling
  before calling anything live simultaneous self-play.
- Repo-native runner: the optional-Torch no-quality PPO on-policy smoke now
  crosses the collection/update boundary with the same tiny actor-critic,
  preserved `[T,B,P]` rollout rows, written metrics/checkpoint/report
  artifacts, and `masked_action_violations=0`. Next slice: add the shared
  reporting contract and scorecard/full-loop profile around this shape while
  keeping simultaneous `[B, P]` state, masks, checkpoints, and explicit
  non-claims. Treat this as an architecture probe; do not call it the final
  framework decision.
- Project-owned Mctx: keep as a later search box or fallback/comparison if
  LightZero cannot expose required custom-env telemetry or artifact control.
