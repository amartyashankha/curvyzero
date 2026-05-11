# Training Coach Handoff - 2026-05-09

This is the compact memory-wipe packet. Read this first before continuing the
training lane.

## 2026-05-10 Bullshit-Purge Update

Read this update before older sections. It overrides older wording that makes
CurvyTron sound like it must use ALE, or makes custom dummy Pong sound like a
quality lane.

- `Atari-style` means LightZero-compatible visual env shape: stacked image
  frames, discrete actions, conv model path, reward/done/info, reset/seed, and
  eval/checkpoint discipline. It does not mean literal ALE.
- `ALE` is only the Atari emulator for stock Atari ROM control runs. Official
  Atari Pong uses ALE. CurvyTron should not need ALE.
- Official Atari Pong is the primary LightZero reproduction/control lane.
- Custom dummy Pong is bridge/debug/telemetry only unless a future run
  deliberately moves it toward visual frame-stack discipline. Do not compare
  custom dummy Pong scores to official Atari Pong scores.
- CurvyTron should use a custom visual LightZero-style env as the main target:
  stacked frames, discrete ego action, reward/done/info, `action_mask`,
  `to_play=-1`, and full wrapper action logging. This is not ALE; it is
  CurvyTron pixels shaped for LightZero.
- Older repo-native `[B, P]` notes remain architecture-probe history. The
  repo-native scalar/ray contract is now a diagnostic sidecar. The current
  no-regret CurvyTron task is the visual LightZero-compatible adapter boundary.
- Every run doc must include two plain lines near the top: `Claim: ...` and
  `Non-claim: ...`. If those lines are missing, do not cite the run as decision
  evidence yet.

## Session Reconstruction

Scope of this reconstruction: local project docs plus local Codex session
metadata/messages for this repo. No raw transcript dump is copied here.

The repeated user directive was: act as the training coach. Build a mature
training stack for MuZero/Mew Zero-style training, with Modal as the serious
run target and CurvyTron as the long-term game. The environment-fidelity agent
can keep reconstructing CurvyTron in parallel; the coach must make training
real: pick the smallest useful games, run toy smokes, learn the Modal patterns,
compare libraries, produce checkpoints, evaluate honestly, and move toward
superhuman self-play.

MuZero/LightZero data comes from repeated environment interaction. In
principle we can scale data with more actors, episodes, and environment steps.
Do not frame curriculum as a normal MuZero requirement. A custom dummy Pong
curriculum is only a bounded diagnostic for a toy sparse-reward failure, and it
must be judged by normal heldout scorecards.
Self-play is now an active lane, not a later idea. The current dummy Pong lane
is not final multiplayer self-play. It is a staged sanity lane: learner ego
versus a scripted opponent, then learner ego versus a frozen checkpoint
opponent. True current-policy two-seat play is a separate feasibility/design
lane; do not pretend the current single-ego Pong setup is already that.
Simultaneous play is not fatal: a single-ego wrapper can step the real
simultaneous game with joint actions while LightZero controls one seat. True
live simultaneous self-play needs explicit opponent policy, checkpoint lineage,
seat metadata, replay ownership, and eval separation. LightZero/MuZero does
support self-play; the open issue is whether the API and game shape fit our
CurvyTron path cleanly.

The user is frustrated because the work repeatedly lost the thread:

- We did many useful things, but the core claim stayed blurry: project-owned
  Pong/Curvy MuZero training has not run, while a tiny LightZero MuZero custom
  dummy Pong smoke has now run.
- We let baseline work, Modal plumbing, supervised policies, CEM, and Mctx
  search benchmarks sound like MuZero training progress.
- We collapsed Pong reads back to win fractions after being told many times to
  include survival/loss-delay telemetry.
- We did not keep the main thread clean enough; workers could have done more of
  the research, edits, and runs.
- We relied too much on short-term chat memory instead of writing the state,
  glossary, run lineage, and corrections into docs.
- We sometimes used unclear language and invented abstractions when the user
  wanted simple, blunt truth.

## Coach Operating Rules

- Main thread decides and plans. Workers do edits, runs, and research.
- Treat docs as working memory. Every parallel worker must update docs or
  return doc-ready facts with artifact/checkpoint refs.
- No run result is complete without same-run baseline, survival-first metrics,
  action histogram, checkpoint refs, and plain claim / plain non-claim.
- Keep custom dummy Pong in the debug lane. It must not distract from official
  LightZero Pong reproduction/control.
- Eval checkpoints as they appear, and compare only to the same-run
  `iteration_0` or another documented same-run baseline.

What went wrong in the process:

- The coach optimized for visible motion instead of the next falsifiable
  training claim.
- The run lineage was not explained clearly enough: separate runs, continued
  checkpoints, changed targets, changed policy classes, changed features, and
  changed Modal wrappers got mixed together in summaries.
- The eval story drifted. Score wins matter, but early Pong learners need
  survival steps, truncation rate, shaped loss-delay return, and variance to
  show whether anything is improving before wins move.
- Modal was treated both as infrastructure research and as proof of training.
  Modal is where serious jobs run; it is not itself a learning result.
- Worker outputs were not always promoted into durable docs, so old questions
  returned after memory pressure.

## Plain Truth

- We did not train project-owned Pong MuZero yet.
- We did not train project-owned Curvy MuZero yet.
- CartPole is infrastructure control only.
- Optimizer correction:
  `docs/working/coach_optimizer_reorientation_2026-05-09.md`. LightZero is a
  serious replication/control/bridge obligation, not the default CurvyTron
  architecture. Historical repo-native simultaneous `[B, P]` notes are an
  architecture probe, not a final framework decision. Current CurvyTron
  prep should stay repo-native and scalar/ray first: `float32[106]`,
  `action_mask`, `to_play=-1`, one ego action, named opponent policy, final
  observation/reward metadata, and full wrapper action logging. Visual
  LightZero-compatible frames are later adapter work, not the current optimizer
  target.
  LightZero remains a serious MuZero reproduction/control lane until credible
  reproduction or a clear blocker. All lanes must emit comparable profile
  metadata, contracts, timing buckets, throughput, latency, checkpoint ids,
  seed/reset details, and explicit non-claims.
- Repo-native PPO/CleanRL-style actor-loop work does not replace LightZero.
  The actor dry-run is `scripts/repo_native_ppo_actor_loop_dry_run.py`. The
  promoted on-policy learner smoke is
  `scripts/repo_native_ppo_learner_smoke.py`: optional Torch, same tiny
  actor-critic collects and is updated, preserves `[T,B,P]` arrays, runs one
  masked PPO update, writes metrics/checkpoint/report artifacts, and the local
  smoke reported `masked_action_violations=0`. It is a no-quality boundary
  smoke, not a learning result and not a LightZero replacement.
- Framework reliability deep dive:
  `docs/working/rl_framework_reliability_deep_dive_2026-05-09.md`. High-level
  stance: stars are not the decision rule. Keep a layered setup: repo-native
  PPO as the first learnability baseline, PettingZoo-style API as adapter,
  LightZero as contained MuZero control, and Mctx later only if we decide to
  own the MuZero trainer/search stack.
- There are two separate LightZero Pong lanes. Official Atari Pong is stock
  LightZero/ALE reproduction and control work; ALE is the Atari emulator for
  this stock ROM lane. Custom dummy Pong is our small controllable bridge/debug
  lane toward CurvyTron. Do not mix their results, and do not compare their
  scores.
- Setup fidelity verdict: official Atari work has only partially followed the
  LightZero setup. It is an official-path smoke, not exact upstream. Exact
  installed-package dry repro now passes for installed `LightZero==0.2.0`: app
  `ap-Xz1gqGamx5CX0tCZfKknk8`, artifact
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/dry-exact-installed-0.2.0-stock-surface-v2/dry_exact_summary.json`.
  It mutates only `exp_name`; stock surface is 200k env steps, 50 sims,
  8 collectors, 3 evaluators, batch 256, segment 400, and no episode caps.
  Current GitHub uses about `500000` env steps; installed `LightZero==0.2.0`
  in our Modal image exposes `200000`. The exact-wrapper faithful-short
  relpath run completed on L4: app `ap-ipdfYJmWQitQtIBxrKU2E9`, run
  `lz-visual-pong-exact-installed-0.2.0-s0`, attempt
  `train-faithful-short-installed-0.2.0-s0-8192-relpath`, summary
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/train/summary.json`,
  sha256 `c97dc26094462ec17d1dd970370d86e392433a8059aed9b1eaea1e5614ed2a06`.
  This is faithful-short, not exact reproduction: patches were `exp_name` and
  `train_muzero.max_env_step 200000 -> 8192`. Train ok `true`, torch CUDA
  true, actual max env step `8192`, collector overshot to `14791`, remote
  elapsed about `1326s`, checkpoints were `ckpt_best`, `iteration_0`, and final
  `iteration_3697`, checkpoint bytes were `256,613,692`, and there were no
  alternate roots. Corrected eval app `ap-ov622Yu6wEnN74V2Laf8HG`, manifest
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-8192-relpath/eval/faithful-short-periodic-custom512-stockeval-s0-8192-relpath/manifest_custom_steps512_seed0_20260509T232832Z.json`.
  `iteration_0` and `iteration_3697` both strict-loaded, used no fallback, and
  ran 512 manual steps. The earlier low eval was invalid because manual
  `max_episode_steps` stayed `64` while stock used `512`. The corrected
  stock-ish eval completed: app `ap-81xAvfiyvnU8flV3eElPSH`, eval id
  `faithful-short-periodic-stockish512-stockeval-s0-8192-relpath`. It used
  stock-ish wrapper knobs because the eval wrapper compiles a policy config:
  `num_simulations=50`, `evaluator_env_num=3`, `collector_env_num=8`,
  `batch_size=256`, `game_segment_length=400`, `max_env_step=200000`,
  `max_train_iter=1`, `update_per_collect=1`, CUDA false because the eval
  wrapper ran on CPU, manual cap `512`, `max_episode_steps=512`, strict load
  true, and fallback false. Result table: `iteration_0` manual return `-13`,
  stock return `-13`, `512` steps, `13` nonzero rewards, `0` positive rewards,
  dominant action `0` share `0.521484`, entropy `0.805545`, manual/stock match
  false. `iteration_3697` manual return `-5`, stock return `-8`, `512` steps,
  `7` nonzero rewards, `1` positive reward, dominant action `0` share
  `0.714844`, entropy `0.644585`, manual/stock match false. Plain read: this
  is the first weak signal that final is less bad than init under stock-ish
  eval, but it is one seed/two checkpoints and the manual-stock mismatch
  remains. Not solved, not exact reproduction.
  The previous installed-package 8192/sim25 rung reached
  `8192` env steps with `num_simulations=25`, still far from the full recipe.
  Its selected strict no-fallback stock `MuZeroEvaluator`/manual parity eval
  curve still did not show credible learning. Checkpoint-state diff now says
  `ckpt_best` is reset-looking, not a quality checkpoint: `last_iter=0`,
  `last_step=0`, optimizer state count `0`, norm counters `0`, running means
  `0`, and vars/weights `1`; `iteration_932` has `last_iter=932`,
  `last_step=3728`, optimizer state count `97`, norm counter `5592`, and
  nonzero running stats. Model keys/shapes match. Diff artifact:
  `training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_state_diff/iteration_932_vs_ckpt_best_20260509T223817Z.json`;
  sha256 `8bfe73bcacf4f4fa72f0cb96dc5838f098b75c40ad574e790582e184371a2fbf`;
  Modal app `ap-yIGfkon1zNYV11hsjyxWO6`. Therefore the failed learning signal
  is now stronger after the parallel 512-step periodic eval: checkpoints
  `0,100,500,900,932` each chose one action for all `512` steps, returned
  `-13`, had no positive rewards, matched stock/manual, and used no fallback.
  It still cannot indict LightZero yet. Link the setup fidelity audit and
  red-team note before making claims:
  `docs/working/lightzero_setup_fidelity_audit_2026-05-09.md` and
  `docs/working/training_setup_red_team_2026-05-09.md`.
- Active next rung:
  `train-faithful-short-installed-0.2.0-s0-32768-relpath`, Modal app
  `ap-xiGLACKHPZLvL1eYgygqvm`. It is a bounded scale/accounting probe while
  eval parity and reproduction assumptions are reviewed. It is not a learning
  claim. A fresh checkpoint poll at `2026-05-10T00:04:24Z` still showed only
  normal checkpoint `iteration_0.pth.tar` plus `ckpt_best.pth.tar`; no later
  normal checkpoint exists yet. This is expected because the installed stock
  cadence saves every `10000` learner iterations plus a final after-run save.
  The next action is to eval the later same-run periodic or final checkpoint
  when it appears, with strict load, no fallback, manual cap `512`, matching
  `--max-episode-steps 512`, stock evaluator enabled, and stock-ish settings
  instead of tiny wrapper defaults: `num_simulations=50`, `evaluator_env_num=3`,
  `collector_env_num=8`, `batch_size=256`, `game_segment_length=400`, and
  `max_env_step=200000`. Summarize baseline deltas against same-run
  `iteration_0`, with `delta_steps_survived` before fraction deltas. For
  future short rehearsals, the wrapper now supports
  `--save-ckpt-after-iter-override`; unset keeps stock cadence.
- Read `docs/working/pong_two_lane_worldview_2026-05-09.md` and
  `docs/working/pong_official_vs_custom_source_map_2026-05-09.md` for the
  current two-lane worldview and source discrepancy map. Recommendation: keep
  both lanes; merge only reporting/eval discipline; do not claim custom dummy
  Pong or `raster_flat` single-frame MLP results are visual Atari parity.
- Read `docs/working/training_lessons_for_curvytron_2026-05-09.md` for the
  current transfer rules. CurvyTron should inherit lane separation,
  target-sidecar observability, survival telemetry, visual-history discipline,
  support-scale proof, and durable Modal artifact patterns, not Pong result
  claims.
- Read `docs/working/training_process_critique_2026-05-09.md` for the current
  process correction: official Atari reproduction and custom root-target parity
  dominate; checkpoint archaeology supports decisions but must not become the
  main lane.
- Visual bridge decision: keep `tabular_ego` as the debug/learning lane, retire
  `raster_flat` to smoke-only, and make the next honest visual bridge a
  separate `raster_stack4_ego` lane with frame history before any CNN or
  Atari-parity claim.
- Current risk: custom dummy Pong may be too far from stock LightZero examples
  and assumptions, so stock-vs-custom discrepancy mapping is now active.
- The real MuZero trainer that has already run is stock LightZero CartPole on
  Modal plus LightZero custom dummy Pong runs on Modal.
- The LightZero dummy Pong run is real MuZero training, but tiny smoke-scale.
  Its policy is not yet proven good.
- LightZero MuZero trains policy logits toward the MCTS root visit
  distribution, not toward the exploratory action finally executed in the env.
  Exploration can execute `down` without making `down` the policy target.
- Custom dummy Pong target audit: action id `2` (`down`) is legal and correct
  in states that need down. The problem is that with `sims=2`, MCTS root visits
  can be `[1,1,0]`, so the policy target can omit the winning `down`. Old
  custom runs did not persist replay `child_visit`/action segments. Target
  replay telemetry is now validated on custom dummy Pong Modal collection; see
  `docs/experiments/2026-05-09-lightzero-dummy-pong-target-replay-telemetry.md`.
  The completed safe smoke `ap-rdvkRpLGRYedx39SggsVvm` used
  `game_segment_length=16` and `batch_size=2`, wrote 16 rows / 1 episode /
  1 collect call to `target_replay_steps.jsonl` and
  `target_replay_summary.json`, and avoided the earlier replay sampling error.
  Safe telemetry-smoke config is `game_segment_length >= 16` and
  `batch_size >= 2`.
- CEM-v2 is a strong geometry baseline.
- Stack-2 raster-only MLP is a useful supervised visual baseline.
- Mctx has only been benchmarked as search. It is not wired into training yet.
  Keep it as a later search box or fallback/comparison; it does not replace the
  repo-owned env, replay, learner, checkpoint, eval, and Modal job contracts.

## Latest Result Synthesis

- CartPole is infrastructure control only.
- 2026-05-09 update: official sparse/delayed TicTacToe and Connect4 bot-mode
  MuZero smokes both passed on Modal. They are useful controls only.
- 2026-05-09 newest stock visual Pong result: official visual Atari Pong
  passes mechanically on Modal through train wrapper
  `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`.
  Dry/import/config passed `ap-zJylQsu1IPoOIbtlhDlO2P`; capped trainer passed
  `ap-MbyIGvX6R815WMZcYzcAyu`. Artifacts are under
  `training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/...`.
  Checkpoint-load probe wrapper
  `src/curvyzero/infra/modal/lightzero_pong_checkpoint_probe.py` passed on
  Modal `ap-YqqMmryhwgtKdFTbStnDKJ`: strict load true for both direct
  `MuZeroModel` and the compiled `MuZeroPolicy` model, cheap forward true,
  input shape `[1,4,64,64]`, logits shape `[1,6]`, value shape `[1,601]`.
  Probe artifact:
  `training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/probe/lightzero_visual_pong_checkpoint_load_20260509T172430Z.json`.
  Eval-only wrapper
  `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py` has telemetry/detail
  caps and passed the official visual eval mechanics on Modal
  `ap-S0HADSUdYxYsy6y1yGj4mP` against checkpoint
  `training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/iteration_1.pth.tar`;
  earlier mismatch/shorter runs `ap-oZ9TMQ03OEEVVMRfNoYPOh` and
  `ap-zRGq8FFHsLhoKC4Ovf37aa` are historical. The policy ran 64 real ALE eval
  steps through `MuZeroPolicy.eval_mode.forward`, no fallback, with
  `num_simulations=2`. Actions were `{0:64}`, rewards were `{-1.0:1,0.0:63}`,
  total reward `-1`, nonzero reward appeared at step 60, done at step 63, and
  terminal info had `TimeLimit.truncated:true`. Raw env obs was `[1,64,64]`
  and policy stack was `[4,64,64]`. Artifact:
  `training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/eval/iteration_1_tiny/lightzero_visual_pong_eval_smoke_20260509T173516Z.json`.
  This proves basic stock visual Atari Pong train/checkpoint/load/eval
  mechanics, not policy quality. The next tiny official Atari CPU scale rung
  also ran on Modal: wrapper caps were widened to allow `max_env_step=128`,
  `max_train_iter=2`, and 128-step collect/eval caps while staying on the
  stock `zoo.atari.config.atari_muzero_config` path with `num_simulations=2`
  and CPU. Train app `ap-qoTln2RP7Ly65hjCK3V4On` completed in
  `19.894407s` remote wall time, but LightZero still only logged
  `Training Iteration [0]` and saved `iteration_0` plus `iteration_1`, not an
  `iteration_2`. New run:
  `training/lightzero-official-visual-pong/lz-visual-pong-20260509T174318Z-0536e5b37ea7/...`.
  New eval app `ap-xARflZIivWVe3TdtHD4vEL` loaded the new `iteration_1` and
  ran 128 real ALE steps with no fallback: actions `{0:128}`, rewards
  `{-1.0:2,0.0:126}`, total return `-2.0`, nonzero rewards at steps `60` and
  `95`, and terminal step `127` with `TimeLimit.truncated:true`. Fair baseline
  app `ap-D0OopWDZJD8K191krNFBBC` evaluated the old tiny `iteration_1` under
  the same 128-step cap and matched exactly: `{0:128}`, return `-2.0`, rewards
  at steps `60` and `95`. Result: real post-64-step Atari reward signal, no
  improvement over the all-action-0 tiny checkpoint. See
  `docs/experiments/2026-05-09-modal-lightzero-pong-scale128-control.md`.
- Dummy Pong `raster_flat` result: train plus matching independent MCTS
  scorecard works mechanically on Modal. This is not a quality win: eval
  aggregate actions were `[424,146,0]`, zero down, and bad versus random/track.
  Keep it separate from stock Atari Pong and the tabular sparse ladder.
  `tabular_ego` likely has enough core toy Pong state; single-frame
  `raster_flat` is weak because it lacks velocity/history.
- The board-game smokes are not evidence that Pong or CurvyTron works, because
  Pong/CurvyTron need visual/raster input and step-based control.
- Closest pattern to borrow: board-game bot mode final-outcome targets:
  `td_steps` to horizon/outcome, `discount_factor=1`, small reward/value
  supports, more update/replay than our smokes.
- Sparse-settings knobs are now exposed and ran in a fixed-horizon Pong probe,
  but the probe still has no clean signal.
- Tabular sparse ladder stop signal: pure 2x did not improve held-out
  survival, shaped return, raw score, or action entropy. `iteration_32` stayed
  all-up with normalized action entropy `0.0`, so do not run the same config
  longer.
- UPC25 higher update/replay at fixed sparse horizon ran on Modal: train
  `ap-3cwlhH1XUZkwZuIFQHFqLA`, scorecard `ap-jIpZP07OHI5Y3dYjpWcW6C`.
  Trainer-side looked mildly alive, but heldout did not pass. Final iteration
  aggregate actions were `[1156,0,74]`; `ckpt_best` all-up `[806,0,0]`.
  Stop repeating longer same-config, eval-sim-only, and update/replay-only
  probes. Next dummy Pong learning move is objective/curriculum or
  data-distribution change.
- Follow-up data-distribution probe also failed as a fix: UPC25 plus random
  warmup / epsilon collection ran on Modal (`ap-MYxTxehyWrDFkygrQTGXEk`) and
  scorecarded on Modal (`ap-jnREG1t3V0jyw8dsOqtbUB`). It preserved true sparse
  env reward and only changed collection exploration. Trainer-side actions
  diversified to `[288,74,64]`, but heldout `iteration_50` stayed collapsed
  across baseline rows with aggregate actions `[1158,0,50]`, raw score `-0.25`
  versus both `lagged_track_ball_1` and `random_uniform`, and `ckpt_best`
  all-up `[806,0,0]`. Stop simple exploration-only changes under the same
  sparse target. The next dummy Pong choice is a scoreable contact/angle
  curriculum, preferably reset/opponent distribution toward paddle-contact and
  scoring-pressure states while keeping `env.step` reward sparse.
- Contact/angle curriculum implementation update: the smallest reset
  curriculum now exists as an explicit opt-in custom dummy Pong mode, not a
  stock Atari Pong benchmark claim. Knobs: `pong_reset_profile=contact_pressure`
  and `pong_reset_pressure_agent=ego` for LightZero training; matching
  player-0-only MCTS eval uses `pong_reset_pressure_agent=player_0`. The real
  env reward remains sparse `+1/-1/0`; no survival reward was added to
  `env.step()`. Tiny Modal train passed `ap-bNRz3Mtil6apjX5w6tNZxa`; matching
  MCTS scorecard passed `ap-XRyCAYWAN7F3ptvRAKRC0x`. Trainer-side was non-flat
  over 4 episodes (2 wins, 1 loss, 1 timeout, raw mean `0.25`, shaped telemetry
  mean `0.3027`, learner actions up/stay/down `[81,46,18]`), but held-out
  `iteration_2` was not a quality win: versus lagged/random/track it got
  wins `1/4`, `1/4`, `0/4`, raw means `-0.5`, `-0.5`, `-0.75`, and action
  histograms `[19,11,0]`, `[20,8,0]`, `[56,22,0]`. Next go/stop: go only to a
  modest same-curriculum rung if it scores `iteration_0`/final and requires
  nonzero down plus held-out raw/shaped improvement; stop any claim that this
  proves policy quality or stock Pong progress.
- Contact-pressure scoreability probe update: local sparse-reward probe
  `scripts/probe_dummy_pong_contact_pressure_scoreability.py` sampled 64 real
  contact-pressure reset states and swept `up/stay/down` against
  `track_ball`, `lagged_track_ball_1`, and `stay`. It found 192/192
  reset/opponent groups action-sensitive and 188/192 with score-return spread.
  Scoreability was opponent-dependent: `lagged_track_ball_1` 46/64, `stay`
  59/64, default `track_ball` 0/64. Go only for a narrow lagged/simple-opponent
  diagnostic curriculum; do not train or claim a `track_ball` score target from
  this reset profile.
- Modest contact-pressure rung update: the one allowed same-curriculum
  diagnostic ran on Modal and stopped. Train app
  `ap-Zr829nRQJqi3WqnTUEwHwr`, scorecard app
  `ap-r5iWQT58qLeLGLIDQ4kDUM`, run
  `lz-dpong-20260509T175407Z-77159cc3a6b4`, attempt
  `attempt-20260509T175407Z-8105d62c1e00`. It kept custom dummy Pong
  `contact_pressure`, sparse env reward, `lagged_track_ball_1`, and 64-step
  horizon. Trainer-side used all three actions (`up=135 stay=74 down=31`), but
  held-out MCTS for `iteration_0`, `iteration_3`, and `ckpt_best` still had
  `down=0` in every learned row. Versus the scoreable lagged target,
  `iteration_0` was score/shaped/survival `-0.3750`/`-0.3433`/`15.875`,
  final `iteration_3` collapsed to `-0.6250`/`-0.5879`/`21.75`, and `ckpt_best`
  was only a weak non-final improvement at `0.0625`/`0.0981`/`24.625`. The
  default-reset scorecard was skipped because the matching held-out stop rule
  had already triggered. Do not launch a longer contact-pressure campaign from
  this evidence.
- Latest worker correction: the custom dummy Pong blocker is root visit/target
  quality and action collapse. Longer same-config runs, simple exploration, and
  trainer-side action diversity are not enough if MCTS root visit targets still
  collapse. Also verify support/value scale: summaries may record requested
  small support ranges while the compiled LightZero policy may still use
  `support_scale=300`. The support-scale patch is opt-in and custom dummy Pong
  only; official Atari is untouched. The planned proof fields are
  `patched_config.surface.*` and `compiled_config.policy_model_cfg.*`.
- Custom dummy Pong target audit update: `down` is not an illegal-action bug.
  Action id `2` is legal and correct in down-needed states, but low-sim MCTS
  can still write root visits like `[1,1,0]`; that means the policy target can
  train away from the winning `down` action. Target sidecars now separate the
  executed action from target mass, but do not yet label the oracle-winning
  action per row. The next custom run must add or join oracle labels and record
  the support-scale proof fields before it can answer whether training data is
  wrong.
- Better-shaped lag1 run did not show a robust survival gain. Independent
  eval collapsed to down-only.
- Tiny lagged-opponent run showed wins over random and lagged opponents, but
  independent eval collapsed to stay-only. That is not clean progress.
- Current Pong sparse-settings probe still has no clean signal: survival and
  shaped score did not improve robustly, and eval roots/action choices are
  weak or tied.
- MCTS sim sweep diagnostic: 8 simulations had ties; 16+ removed ties but
  collapsed fully to down. Eval tie-breaking was a symptom, not the root fix.
- Frozen-checkpoint opponent path works as staged self-play plumbing. It proves
  a live learner can train against an env-owned frozen LightZero checkpoint,
  and a tiny scorecard can compare the new checkpoint with its parent. It is
  not a policy-quality win. This does not mean simultaneous games are fatal to
  LightZero: a single-ego joint-action wrapper is a valid bridge. The missing
  piece for true live simultaneous self-play is explicit opponent/checkpoint
  and metadata handling, not generic MuZero self-play support.
- Active next lanes: official visual Pong now has a stop signal on naive
  same-shape scale and on the first more official-shaped staged rung. GPU1024
  on L4 gave a small same-cap `-3` signal, GPU2048 completed through
  `iteration_8` and same-cap eval stayed `-6`, and the official Atari
  4096/sim10 L4 run also completed with strict no-fallback eval but stayed
  `-6` for `iteration_0`, `iteration_4`, and `iteration_8`. By `iteration_4`
  it collapsed to action `5`. The official eval parity probe validates the
  manual eval path: strict-load/no-fallback is real, raw/policy observation
  shapes are correct, action `5` is ALE `LEFTFIRE`, the first 32 steps are all
  action `5`, and roots are action-5 dominated with high logits. Treat
  4096/sim10 as infrastructure pass and signal fail; the later 8192/sim25 rung
  is now also an evaled signal fail, not a solved-policy result. Do not reopen
  action-mapping rabbit holes. Collapse investigation:
  `docs/working/lightzero_official_atari_collapse_investigation_2026-05-09.md`.
  Primary suspect is an undertrained/off-recipe early learner; the flat `-6` is
  mostly the 256-step eval cap, not full Pong score. The matching-checkpoint
  stock evaluator path now uses LightZero `MuZeroEvaluator`; a tiny Modal smoke
  fixed the missing `action_mask`/collation gap and matched manual actions.
  The follow-up installed `LightZero==0.2.0` 64x64 non-segment Atari
  `8192/sim25` rung then completed on L4: dry app
  `ap-VasQbApDzGd18EaB38hM59`, train app `ap-qnwMaN8FlOUNJwLNo1mZKs`, run
  `lz-visual-pong-8192-sim25-s0`, attempt
  `train-8192-sim25-b64-env4-auto`, with `ckpt_best` plus `iteration_0`
  through `iteration_932` mirrored. The promoted strict no-fallback eval curve
  ran for `iteration_0`, `iteration_100`, `iteration_500`, `iteration_900`,
  `iteration_932`, and `ckpt_best`. Periodic checkpoints strict-loaded, and
  manual/stock first-32 actions matched; every periodic checkpoint collapsed to
  one action and returned `-6`. `ckpt_best` manual eval reached `0` with
  diverse actions, but manual/stock first-32 did not match and stock return
  stayed `-6`. Checkpoint diff shows `ckpt_best` is reset-looking while
  `iteration_932` has trained counters, optimizer state, and nonzero running
  stats; same model keys/shapes. Treat `ckpt_best` as unusable for quality
  until the best-save path is explained. The major
  adoption risk is now accounting: stock `update_per_collect=None` produced a
  huge learner/checkpoint burst, roughly 934 checkpoint files / 90 GB, so
  bounded replication needs explicit learner update counts, checkpoint
  cadence/retention, eval-selection discipline, and embarrassingly parallel
  eval using the existing
  `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py` parallel
  checkpoint-eval wrapper.
  Trainer-side `-21` versus capped strict-eval `-6` are not comparable score
  surfaces. Next official Atari work is: ignore `ckpt_best` for quality until
  the best-save path is explained; run exact stock reproduction or periodic
  checkpoint evals with the parallel wrapper; do not claim learning from
  `ckpt_best`.
  Pretrained OpenDILab
  Pong strict eval is separately blocked by model-surface mismatch: strict
  load still reports unexpected
  `representation_network.downsample_net.conv2.weight`, so no eval has run.
  Treat this as older 96x96/downsample checkpoint versus current 64x64 stock
  config until a matching checkpoint/config pair is available.
  Custom dummy Pong remains separate: add/join oracle labels to target sidecars
  and fix or prove the opt-in custom support-scale patch using
  `patched_config.surface.*` and `compiled_config.policy_model_cfg.*`.

## Active Parallel Wave

Current lanes to keep mapped, not expanded here:

- Action-collapse root cause.
- LightZero target semantics.
- Support/value scale audit.
- Official Atari scale.
  Current read:
  `docs/experiments/2026-05-09-modal-lightzero-pong-8192-sim25.md` - installed
  `LightZero==0.2.0` 64x64 non-segment Atari rung. Dry app
  `ap-VasQbApDzGd18EaB38hM59`; train app `ap-qnwMaN8FlOUNJwLNo1mZKs`;
  L4 train completed and mirrored `ckpt_best` plus `iteration_0..932`.
  Strict no-fallback eval now exists for `iteration_0`, `iteration_100`,
  `iteration_500`, `iteration_900`, `iteration_932`, and `ckpt_best`. Periodic
  checkpoints matched manual/stock first-32, collapsed to one action, and
  returned `-6`; `ckpt_best` manual eval reached `0` with diverse actions but
  failed first-32 parity and stock stayed `-6`. Checkpoint diff now shows
  `ckpt_best` is reset-looking, so do not use it as quality evidence or claim
  learning from it. No credible learning signal. Before any larger train,
  ignore `ckpt_best` until the best-save path is explained, or switch to exact
  upstream/pretrained config, with update/checkpoint accounting kept explicit.
  Prior read:
  `docs/experiments/2026-05-09-modal-lightzero-pong-4096-sim10.md` - staged
  4096 env-step L4 rung with `num_simulations=10`, `batch_size=32`, and
  `collector_env_num=2` ran and evaled with no fallback, but the curve stayed
  at `-6` and collapsed to action `5`.
- Stock-vs-custom discrepancy mapping.
- Two-lane worldview/source-map docs:
  `docs/working/pong_two_lane_worldview_2026-05-09.md` and
  `docs/working/pong_official_vs_custom_source_map_2026-05-09.md`.
- Contact-pressure oracle.
- Modal scale plan.

## Why The User Was Right To Be Annoyed

I let scaffolding sound like training progress. That is not acceptable. A
baseline that writes checkpoints is not MuZero. A search benchmark is not a
trainer. A supervised Pong policy is not self-play MuZero. Future summaries
must name the actual algorithm every time.

## Non-Negotiable Rules

- Put new knowledge in docs before relying on memory.
- Keep the main thread for planning, delegation, and decisions.
- Use workers for research, edits, runs, and doc updates.
- Do not run pytest in this lane unless explicitly redirected.
- Every experiment title and summary must say the algorithm:
  `LightZero MuZero`, `project-owned MuZero/Mctx`, `CEM`, `supervised MLP`,
  `NumPy staged Pong`, or `Mctx benchmark`.
- Do not call CEM, imitation, supervised MLP, or synthetic Mctx benchmark
  "MuZero progress."
- Use simple language. No invented jargon, no vague "MuZero-shaped" claims, no
  unexplained run names.
- Explain run lineage when reporting results: same run or separate run, parent
  checkpoint if any, target/opponent, feature mode, architecture, reward/target,
  Modal wrapper, and checkpoint source.
- Workers may research, edit docs, run commands, and perform scoped
  implementation. The main thread decides, delegates, follows up, and
  synthesizes.
- Before scaling a target, prove it is scoreable or clearly label it as a
  survival/tie floor.

## Operating Pattern

Use this loop:

1. Re-read this handoff and the active docs.
2. List the active lanes and the newest user requests.
3. Name the next decision in plain language.
4. Delegate narrow research, edit, run, or implementation lanes to workers.
5. Use follow-ups on workers as new facts arrive; do not restart the same
   question from scratch.
6. Keep the main thread focused on planning, orchestration, high-level
   reorientation, blockers, and handoff updates.
7. Make workers write findings into docs, not only chat.
8. Run small proof commands before scaling.
9. After every run, update the experiment log and the active backlog.
10. Before memory wipe or a long pause, update this handoff and end with the
   shorthand.

When the user sounds angry, do not argue with the tone. Extract the correction,
write it into docs, and change the work pattern. The repeated corrections are:
keep the docs current, keep the main thread clean, use parallel workers, stop
confusing baselines with MuZero, never forget survival-shaped Pong telemetry,
track all open tasks, and push toward actual Modal-backed MuZero training.

Current process failure to avoid: doing many useful side experiments while the
main claim stays false. Baselines are useful only because they give comparison
rows for the real trainer.

## Reward And Eval Rule

Current consolidated audit:
`docs/working/lightzero_pong_survival_reward_audit_2026-05-09.md`.

Keep the environment reward honest:

```text
ego scores:       +1
opponent scores:  -1
no score event:    0
```

For weak early Pong learners, report shaped loss-delay telemetry:

```text
steps_survived = episode_steps
survival_fraction = steps_survived / max_steps

win:     +1.0
loss:    -1.0 + 0.5 * survival_fraction
timeout:  0.0
```

This shaped value is a diagnosis signal and checkpoint-selection tie-breaker
only after true score is tied. Do not train MuZero reward heads on this shaped
quantity by default. A shaped training target is allowed only as an explicitly
labeled temporary ablation, never as the promoted metric and never by changing
`env.step()` reward. Never reduce a Pong read to only `0/N wins`. Always report:

- wins/losses;
- mean, median, and p90 survival steps;
- truncation rate;
- shaped loss-delay return;
- score return;
- variance or standard deviation of survival and shaped return.

Variance is not the environment reward. It can be used as a bounded early
exploration or checkpoint-selection signal when two candidates have similar
mean score/survival. The reason is practical: a policy with rare long rallies,
rare scores, or a wider survival tail may be closer to discovering a useful
behavior. Keep that bonus small and temporary so it does not reward random,
unstable, or stalling behavior.

Implementation note from this pass:
`src/curvyzero/training/dummy_pong_eval.py` and
`scripts/run_dummy_pong_checkpoint_scoreboard.py` now expose the required
survival and shaped loss-delay telemetry in eval summaries and scoreboard rows.
The smoke is recorded in
`docs/experiments/2026-05-09-dummy-pong-scoreboard-telemetry-patch-smoke.md`.
Do not let future reports say only `0/N wins`; include the survival and shaped
return fields.

Reward-shaping research update:
`docs/research/reward_shaping_for_pong_curvy_muzero.md` now records the active
recommendation. Keep `env.step` reward true to the game (`+1/-1/0`). Do not
add a positive per-step survival reward inside the environment. Use shaped
loss-delay only as telemetry or a temporary, clearly labeled target/tie-breaker,
and avoid incentives that teach stalling.

Latest shaped-objective smoke:
`docs/experiments/2026-05-10-dummy-pong-survival-shaped-loss-delay-smoke.md`.
It was custom dummy Pong only. It improved steps survived in final eval, but
raw score barely moved and greedy eval collapsed to action `down`. Treat it as
plumbing evidence for loss-delay shaping and artifact writing, not policy
quality.

## Pong Baselines To Keep

- `random_uniform`: sanity floor.
- default `track_ball`: survival/tie floor. It may be impossible to score
  against from normal resets in the current default geometry, but tying or
  surviving against it is still useful.
- `lagged_track_ball_1`: scoreable target ladder. Current CEM-v2 Modal
  baseline scored 53/64 against it.
- CEM-v2: geometry baseline, not visual, not MuZero.
- Stack-2 raster-only MLP: supervised visual baseline, not MuZero.

Trackball is not "optimal Pong." It is a deterministic script in this toy
geometry. Treat it as a baseline to survive against and eventually beat through
scoreable variants, not as proof that the game is solved.

## Long-Term Want

The long-term goal is not "have a bunch of Pong baselines." The goal is a
robust CurvyTron training stack:

- faithful enough CurvyTron environment and training interface;
- visual observations and clean replay rows;
- MuZero-family self-play with a real model, search, replay, updates,
  checkpointing, and evaluation;
- Modal whole-job execution with durable artifacts;
- simple baseline ladders that prove progress instead of hiding failure;
- later robustness work such as sticky actions, frozen controls, action noise,
  image noise, and domain randomization when the core loop is honest.

## Actual Next Lane

LightZero-first is still the active path because CartPole is infrastructure
control only, official sparse/delayed TicTacToe and Connect4 bot-mode MuZero
smokes passed on Modal as useful controls only, and LightZero custom dummy Pong
can train, mirror checkpoints, and run independent scorecards. The board-game
smokes are not evidence that Pong/CurvyTron works, because those need
visual/raster input and step-based control. Do not label any current
dummy Pong checkpoint as progress. The latest shaped lag1 run,
lagged-opponent run, and frozen-checkpoint opponent smokes all point to the same
blocker: the plumbing works better than the policy.

Plain current read: official Atari 8192/sim25 completed mechanically on
installed `LightZero==0.2.0` and now has a selected strict no-fallback eval
curve, but still shows no credible learning. Dry app
`ap-VasQbApDzGd18EaB38hM59`; train app
`ap-qnwMaN8FlOUNJwLNo1mZKs`; L4 run `lz-visual-pong-8192-sim25-s0` mirrored
`ckpt_best` plus `iteration_0..932`. The 8192/sim25 curve covered
`iteration_0`, `iteration_100`, `iteration_500`, `iteration_900`,
`iteration_932`, and `ckpt_best`: periodic checkpoints strict-loaded, matched
manual/stock first-32, collapsed to one action, and returned `-6`. `ckpt_best`
manual eval reached `0` with diverse actions, but stock return stayed `-6` and
manual/stock first-32 did not match. Checkpoint diff shows `ckpt_best` is
reset-looking: last iter/step, optimizer state count, norm counters, and
running means are zero, while `iteration_932` has trained counters and nonzero
running stats. Same model keys/shapes. Do not claim learning from `ckpt_best`.
The matching-checkpoint stock evaluator path now uses
LightZero `MuZeroEvaluator`; periodic parity says action mapping/evaluator
collation are not the main issue. Before another larger train, decide whether
to run exact stock reproduction or periodic checkpoint evals with the parallel
wrapper; first explain the best-save path if anyone wants to use `ckpt_best`.
Do not compare
the trainer-side `-21` headline with capped strict-eval `-6`; they are
different score surfaces. Keep update/checkpoint accounting explicit: stock
`update_per_collect=None` produced roughly 934 checkpoint files / 90 GB. The
iteration-loop bottleneck critique is now part of the official read: use the
existing `lightzero_pong_eval_smoke.py` parallel checkpoint-eval wrapper by
default, pair it with checkpoint retention, and avoid serial checkpoint
archaeology after a burst. Parallel eval smoke passed on app
`ap-GroNH8bnBAadark30VLY51`; manifest
`training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_curve_parallel_smoke/manifest_low_steps32_seed0_20260509T202709Z.json`.
It proves batching workflow only; 32-step returns are not quality. Be explicit
that this is not a LightZero or exact-upstream failure: GitHub current is about
`500000` env steps, installed `LightZero==0.2.0` is `200000`, and `8192` is
still far below both.
Pretrained OpenDILab Pong strict eval remains blocked separately by the older
96x96/downsample checkpoint surface versus the current 64x64 config: strict
load still reports unexpected
`representation_network.downsample_net.conv2.weight`, so there is no eval.
GPU2048 remains a smoke-scale audit point, not a failed official reproduction.
For custom dummy Pong, add or join oracle-winning-action labels to the target
sidecars, inspect why low-sim root visits can omit legal winning `down`, and
fix/prove the opt-in custom support-scale patch. Official Atari is untouched by
that support-scale patch.

The better-shaped lag1 probe used the exposed training knobs and collected
varied train actions, but independent MCTS eval did not show robust survival or
score improvement. It collapsed to all `down` actions. The tiny lagged-opponent
probe beat random and lagged rows on the small scorecard, but collapsed to all
`stay` actions, so treat it as a cheap completion/signal smoke, not a clean
learning result. Frozen-checkpoint opponent runs against `iteration_0` and
`iteration_16` prove staged self-play plumbing: a live learner can collect
against an env-owned frozen LightZero checkpoint, and independent eval can
compare the new checkpoint to the parent. They do not prove policy quality.

Current next blockers, in order:

1. Do not make the next Pong run just longer.
2. Official visual Pong eval-only smoke is done: wrapper
   `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py`, passing Modal
   `ap-S0HADSUdYxYsy6y1yGj4mP`; earlier mismatch/shorter runs
   `ap-oZ9TMQ03OEEVVMRfNoYPOh` and `ap-zRGq8FFHsLhoKC4Ovf37aa` are historical.
   It loaded
   `training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/iteration_1.pth.tar`
   and acted in a real ALE-backed LightZero Pong env through
   `MuZeroPolicy.eval_mode.forward` with no fallback and `num_simulations=2`.
   It ran 64 steps: actions `{0:64}`, rewards `{-1.0:1,0.0:63}`, total reward
   `-1`, nonzero reward at step 60, done at step 63 with
   `TimeLimit.truncated:true`; raw env obs `[1,64,64]` was stacked to policy
   input `[4,64,64]`. This answers basic official visual eval mechanics, not
   policy quality. Follow-up scale128 control also ran: it reached 128 real ALE
   eval steps and two true `-1` rewards, but both old and new `iteration_1`
   checkpoints stayed all-action-0 with return `-2.0`. See
   `docs/experiments/2026-05-09-modal-lightzero-pong-scale128-control.md`.
   Follow-up GPU512 control also ran after adding explicit cheap-GPU telemetry
   to `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`.
   Train app `ap-NcECoDQrcIfrbpqmBRODbP`, run
   `lz-visual-pong-20260509T180945Z-29b83d6ee638`, attempt
   `attempt-20260509T180945Z-dc971b1ec0ff`, used actual `NVIDIA L4` with
   `policy.cuda=true`, `max_env_step=512`, `max_train_iter=4`, 128-step
   episode caps, `num_simulations=2`, and `batch_size=8`. It completed and
   mirrored `iteration_0` through `iteration_4`. Final eval app
   `ap-AUNehXPKKdkXbPOCW5WM7B` loaded `iteration_4` and ran 128 real ALE steps
   with no fallback: actions `{0:21,1:24,2:22,3:25,4:22,5:14}`, return
   `-2.0`, rewards at steps `60` and `95`, terminal
   `TimeLimit.truncated:true`. This clears wrapper/GPU/cap blockers and breaks
   all-action-0, but is not a quality win because the return matches the
   action-0 baseline. See
   `docs/experiments/2026-05-09-modal-lightzero-pong-gpu512-control.md`.
   Follow-up GPU1024 on L4 was the first small official Atari signal: it produced
   checkpoints through `iteration_4`; final eval at 256 real ALE steps improved
   same-cap return from the GPU512 baseline `-5` to `-3` and saw one `+1`
   reward. This is a small official visual reproduction signal, not solved
   Pong and not custom dummy Pong progress.
   Follow-up GPU2048 on L4: train
   completed, checkpoints reached `iteration_8`, and same-cap eval stayed at
   return `-6`. It did not reinforce GPU1024's small `-3` signal, but this is
   smoke-scale rather than a failed official reproduction. The official Atari
   settings audit names the biggest gaps as `max_env_step=2048` vs about
   `500000`, `num_simulations=2` vs `50`, one env, `batch_size=8`,
   `game_segment_length=16`, and `update_per_collect=1`. Stop naive
   same-shape official Atari scale; the next reproduction should move toward
   official settings, especially simulations and env/update scale, instead of
   only doubling the same wrapper caps.
   Follow-up official Atari 4096/sim10 on L4: train completed with
   `num_simulations=10`, `batch_size=32`, `collector_env_num=2`,
   `update_per_collect=2`, and `game_segment_length=64`; checkpoints reached
   `iteration_8`; strict no-fallback evals for `iteration_0`, `iteration_4`,
   and `iteration_8` all returned `-6`. `iteration_4` and `iteration_8`
   collapsed to action `5` for all 256 eval steps. This is infrastructure
   pass and signal fail. The manual eval parity probe is valid: strict-load and
   no-fallback held, observation shapes are correct, action `5` is ALE
   `LEFTFIRE`, the first 32 eval steps are all action `5`, and roots are
   action-5 dominated with high logits. The stock DI-engine evaluator path is
   now fixed for matching 64x64 checkpoints by using LightZero
   `MuZeroEvaluator`; the tiny Modal smoke matched manual actions. Do not do
   more action mapping or evaluator collation work.
   Follow-up installed `LightZero==0.2.0` 64x64 non-segment Atari 8192/sim25
   on L4 is now the promoted latest official result: dry app
   `ap-VasQbApDzGd18EaB38hM59`, train app `ap-qnwMaN8FlOUNJwLNo1mZKs`, run
   `lz-visual-pong-8192-sim25-s0`, attempt
   `train-8192-sim25-b64-env4-auto`. It completed with
   `num_simulations=25`, `batch_size=64`, `collector_env_num=4`,
   `update_per_collect=None`, and `game_segment_length=128`; checkpoints
   reached `iteration_932` plus `ckpt_best`. Strict no-fallback eval ran for
   `iteration_0`, `iteration_100`, `iteration_500`, `iteration_900`,
   `iteration_932`, and `ckpt_best`. Periodic checkpoints strict-loaded,
   matched manual/stock first-32, collapsed to one action, and returned `-6`.
   `ckpt_best` manual eval reached `0` with diverse actions, but stock return
   stayed `-6` and first-32 parity failed. Checkpoint-state diff shows
   `ckpt_best` is reset-looking (`last_iter=0`, `last_step=0`, optimizer state
   count `0`, norm counters `0`, running means `0`, vars/weights `1`) while
   `iteration_932` has `last_iter=932`, `last_step=3728`, optimizer state count
   `97`, norm counter `5592`, and nonzero running stats. Same model
   keys/shapes. It is not quality evidence.
   Trainer-side `-21` versus capped strict-eval `-6` are not comparable score
   surfaces.
   Adoption risk: stock auto-update accounting caused a huge learner/checkpoint
   burst, roughly 934 checkpoint files / 90 GB. Next: ignore `ckpt_best` for
   quality until the best-save path is explained; run exact stock reproduction
   or periodic checkpoint evals with the parallel wrapper; do not claim
   learning from `ckpt_best`. Any next train needs explicit checkpoint
   retention, not a post-hoc serial pass over hundreds of artifacts.
   Pretrained OpenDILab
   Pong strict eval remains separately blocked by the older 96x96/downsample
   checkpoint surface versus the current 64x64 config: strict load still
   reports unexpected `representation_network.downsample_net.conv2.weight`,
   so there is no eval.
3. Visual bridge decision: keep `tabular_ego` as the dummy Pong debug/learning
   lane. Retire `raster_flat` to smoke-only: it trained and scorecarded
   mechanically on Modal, but eval actions were `[424,146,0]` with zero down
   and bad random/track rows. The next honest visual bridge is separate
   `raster_stack4_ego` with frame history before any CNN/Atari-parity claim.
4. Change the sparse objective/curriculum before more tabular time. Pure 2x
   did not improve held-out survival, shaped return, raw score, or action
   entropy; `iteration_32` stayed all-up with entropy `0.0`.
5. Sweep MCTS simulations as a diagnostic. Current read: 8-sim ties were only
   a symptom; 16+ sims removed ties but collapsed fully to down.
6. Do a broad bug hunt across observation encoding, target construction,
   reward/support config, action selection, and eval adapter behavior.
   Include the support/value-scale mismatch suspicion before trusting support
   ablations.
7. Keep survival, shaped return, score return, truncations, action histograms,
   seed histograms, checkpoint refs, and artifact refs in the readout.
8. Scale only after the small sparse-settings probe stops collapsing and
   held-out scorecards show real movement.

Implementation status: train knobs are exposed in the Modal config smoke, tiny
train smoke, and scaled train attempt entrypoints. The config records
`action_type='fixed_action_space'` explicitly. Frozen checkpoint opponent
support and telemetry exist for staged self-play smokes. No pytest in this
lane unless explicitly redirected.

Current library direction:
LightZero-first remains accepted. LightZero is the only complete MuZero trainer
we have already run on Modal, and it has now run on the custom dummy Pong env at
tiny smoke scale. The core plan is simple: get a trusted LightZero whole-job
Modal run, measure checkpoint curves honestly, scale actors/episodes/steps,
then move to frozen-checkpoint self-play and later full multiplayer. The
frozen checkpoint opponent is the immediate practical self-play step because
most support already exists. Do not switch to a project-owned Mctx trainer
unless the LightZero checkpoint/eval path fails or hides required
telemetry/artifacts.

Scorecard summary automation exists in
`docs/working/lightzero_dummy_pong_scorecard_summary_automation_2026-05-09.md`,
but local Modal Volume access is incomplete. Use it when refs are mounted or
fetched; do not pretend missing local summaries were checked.

Fresh current state:

- Tiny LightZero MuZero dummy Pong train passed on Modal:
  `ok: true`, `called_train_muzero: true`, `algorithm: LightZero MuZero`.
- Run id: `lz-dpong-20260509T141607Z-3696aa333028`.
- Attempt id: `attempt-20260509T141607Z-98662e4917b4`.
- Env-side telemetry has 5 terminal rows: 4 wins, 1 loss, 0 truncations
  against `random_uniform`; this is trainer-side smoke telemetry, not an
  independent checkpoint scorecard.
- Mirrored LightZero checkpoints exist:
  `ckpt_best.pth.tar`, `iteration_0.pth.tar`, and `iteration_2.pth.tar`.
- A main-thread 512/8 LightZero MuZero dummy Pong run also completed on Modal.
  Run id: `lz-dpong-20260509T144635Z-eb5a0ed35de0`; attempt id:
  `attempt-20260509T144635Z-ece79bad80d0`; Modal URL:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-4XcfDjKPeDhMG2uI93QcYN`.
  Summary ref:
  `training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/attempts/attempt-20260509T144635Z-ece79bad80d0/train/summary.json`.
- The 512/8 config was `max_env_step=512`, `max_train_iter=8`,
  `num_simulations=4`, `batch_size=16`, `n_evaluator_episode=4`, `seed=1`.
  Trainer-side telemetry reported 42 episodes, 37 wins, 5 losses, no
  timeouts, mean survival 9.0476, p90 survival 8.0, shaped mean 0.7633,
  score mean 0.7619, and checkpoint iterations `[0, 8]`.
- A CurvyZero-owned direct policy-head greedy scoreboard can now score the
  LightZero checkpoint. It is a narrow inference probe, not LightZero MCTS eval.
- `ckpt_best.pth.tar` is a Torch checkpoint with keys `last_iter`,
  `last_step`, `model`, `optimizer`, and `target_model`; its policy head shape
  is `(3, 32)`.
- Earlier checkpoint probes passed policy-head access while strict full model
  load failed on stale dynamics/config variants. That is now superseded for the
  512/8 `iteration_8` checkpoint by the passing MCTS loader smoke below.
- Probe ref:
  `training/lightzero-dummy-pong/lz-dpong-20260509T141607Z-3696aa333028/attempts/attempt-20260509T141607Z-98662e4917b4/probe/lightzero_checkpoint_probe_20260509T143137Z.json`.
- Policy-head scoreboard refs:
  `eval/lightzero-dummy-pong/policy-head-scoreboard-20260509T143955Z/summary.json`
  and
  `eval/lightzero-dummy-pong/policy-head-scoreboard-20260509T143955Z/episodes.jsonl`.
- Scoreboard rows: versus `random_uniform`, LightZero won 21/40 with mean
  score return 0.05, shaped mean 0.081875, mean steps 13.775, p90 20.1.
  Versus `lagged_track_ball_1`, LightZero won 16/40, lagged won 20/40, 4
  truncations, mean score -0.1, shaped mean -0.0764583, mean steps 21.95, p90
  29.1. Versus `track_ball`, LightZero won 0/40, track_ball won 38/40, 2
  truncations, mean score -0.95, shaped mean -0.8736458, mean steps 24.325,
  p90 41.0.
- Crucial action-histogram read: raw matchups show `lightzero_best` used
  constant up in every LightZero row: `[N, 0, 0]`. Do not treat the 21/40 row
  versus `random_uniform` as learning proof.
- A policy-head scoreboard for the 512/8 run also completed:
  `eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-20260509T144736Z/summary.json`
  and
  `eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-20260509T144736Z/episodes.jsonl`.
  It scored `ckpt_best` and `iteration_8`; both are still constant-up.
- 512/8 policy-head read: `best` vs random was 30/64 wins, random 34/64,
  action `[864, 0, 0]`, mean steps 13.5, shaped mean -0.03047. `iteration_8`
  vs random was 31/64 wins, random 33/64, action `[787, 0, 0]`, mean steps
  12.2969, shaped mean -0.00117.
- 512/8 policy-head read versus lagged and track: `best` vs
  `lagged_track_ball_1` was 21/64 wins, lagged 35/64, 8 truncations, action
  `[1595, 0, 0]`, shaped mean -0.1919. `iteration_8` vs lagged was 29/64 wins,
  lagged 35/64, action `[677, 0, 0]`, shaped mean -0.0691. `best` vs
  `track_ball` was 0/64 wins, track 60/64, 4 truncations, action
  `[1510, 0, 0]`, shaped mean -0.8704. `iteration_8` vs track was 0/64 wins,
  track 53/64, 11 truncations, action `[2217, 0, 0]`, mean steps 34.6406,
  shaped mean -0.7697.
- A strict-config direct policy-head rerun also completed after the loader fix:
  eval id `policy-head-scoreboard-512x8-strictcfg`, Modal URL
  `https://modal.com/apps/modal-labs/shankha-dev/ap-Q7sPmscebJQWisowuweBxV`.
  Refs:
  `eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-strictcfg-20260509T145841Z/summary.json`
  and
  `eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-strictcfg-20260509T145841Z/episodes.jsonl`.
  `load_state_dict` is now strict true through the direct policy-head path;
  the split residual dynamics config fix is working. Result is still
  constant-up: vs lagged `[590,0,0]`, 13 wins vs 17 lagged wins, 2 truncs,
  mean steps 18.4375, shaped -0.0987; vs random `[388,0,0]`, 12 wins vs 20
  random wins, mean steps 12.125, shaped -0.2134; vs track `[968,0,0]`,
  0 wins vs 28 track wins, 4 truncs, mean steps 30.25, shaped -0.8115.
- `scoreboard_rows` now include `action_histogram_by_policy`.
- Main decision: trainer-side env/evaluator wins are not a policy-quality
  proof. Independent checkpoint eval says the exported/reconstructed greedy
  policy-head is constant-up. The old missing-`cfg.policy.device` MCTS loader
  failure is stale; device/action-mask issues were fixed.
- MCTS loader smoke on the 512/8 `iteration_8` now passes. Artifact ref:
  `training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/attempts/attempt-20260509T144635Z-ece79bad80d0/probe/lightzero_mcts_loader_smoke_20260509T145607Z.json`.
  Details: `ok: true`, `mcts_eval_status: ok`,
  `strict_full_model_load_ok: true`, strict-load variant
  `res_connection_in_dynamics_true`. Call shape was `data [1,10]`,
  `action_mask [[1,1,1]]`, `to_play [-1]`, `ready_env_id [0]`.
  `MuZeroPolicy.eval_mode.forward` returned action `0`,
  `visit_count_distributions [2,1,1]`,
  `predicted_policy_logits [0.0170983, 0.00644484, 0.0132326]`,
  predicted value about `0.0000259`, and searched value about `0.000114`.
  This was the loader proof; it is now superseded by the full MCTS scorecard
  below.
- Full LightZero MCTS/eval-mode scorecard ran on Modal:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-Ou59sqrdljB295FFBpyIUP`.
  Summary ref:
  `eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-20260509T150000Z-20260509T150243Z/summary.json`.
  Episodes ref:
  `eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-20260509T150000Z-20260509T150243Z/episodes.jsonl`.
  Checkpoint: 512/8 `iteration_8.pth.tar`; 16 episodes per seating;
  `num_simulations=8`; strict full model load OK with
  `strict_full_model_load_variant=res_connection_in_dynamics_true`.
- MCTS rows: vs `lagged_track_ball_1`, LightZero won 13, opponent won 15,
  mean survival 25.09, LightZero shaped -0.0397, LightZero reward -0.0625,
  actions `[801,2,0]`. Vs `random_uniform`, LightZero won 17, opponent won 15,
  mean survival 13.84, LightZero shaped 0.0953, LightZero reward 0.0625,
  actions `[443,0,0]`. Vs `track_ball`, LightZero won 0, opponent won 30,
  mean survival 25.66, LightZero shaped -0.8618, LightZero reward -0.9375,
  actions `[816,5,0]`.
- Read: MCTS eval-mode is no longer just a loader smoke. Full episode eval
  works. But checkpoint behavior is still effectively up-only: combined
  LightZero action histogram `[2060,7,0]`; it never chose down in this
  scorecard. The next blocker is policy quality/training signal, not
  checkpoint loading.
- Config bug fixes landed: `DummyPongLightZeroEnv.random_action()` now keeps a
  persistent per-episode RNG, the base observation includes `timestep`, and
  LightZero scorecards construct `PongConfig(max_steps=lightzero_max_env_step)`
  for LightZero checkpoint eval. Baseline-only eval remains on the default
  120-step horizon.
- Corrected 512-step MCTS/eval-mode scorecard ran on Modal:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-zUwRanuyB0OHCA8NdpOHVQ`.
  Summary ref:
  `eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-maxstep512-20260509T151544Z/summary.json`.
  Config check: `config.max_steps=512`,
  `lightzero_eval_config.max_env_step=512`, `num_simulations=8`.
- Corrected rows: vs `lagged_track_ball_1`, LightZero won 14 and lagged won
  14, mean steps 74.44, actions `[2373,9,0]`; vs `random_uniform`, LightZero
  won 14 and random won 18, mean steps 12.81, actions `[407,3,0]`; vs
  `track_ball`, LightZero won 0 and track won 29, mean steps 64.88, actions
  `[2069,7,0]`.
- Read: horizon mismatch was real and fixed for future LightZero checkpoint
  evals, but it is not the root cause of up-only behavior. Across the corrected
  LightZero rows the aggregate action histogram is `[4849,19,0]`: still
  effectively up-only, with 0 down actions.
- Longer LightZero dummy Pong CPU train also succeeded after the scaled wrapper
  caps were intentionally changed to `8192/64`: Modal URL
  `https://modal.com/apps/modal-labs/shankha-dev/ap-Uj3XVYgEnzr9oSVan3NILH`,
  run `lz-dpong-20260509T151212Z-b95b61de2eb0`, attempt
  `attempt-20260509T151212Z-8b9db08f8fcb`, train summary
  `training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/attempts/attempt-20260509T151212Z-8b9db08f8fcb/train/summary.json`.
  `iteration_64` ref:
  `training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/checkpoints/lightzero/iteration_64.pth.tar`,
  sha256 `11a0cc80f797ce8e63150e0a6018efc163b7858bed9efd92b77dda8cadaf95e4`.
- Long-train trainer-side sidecar telemetry: 578 rows, 535 wins, 43 losses, no
  timeouts, survival mean 18.18, median/p90 19/19, max 52, score mean 0.8512,
  shaped mean 0.8513, player_0 actions `[9539,800,170]`. This is not a clean
  held-out final-checkpoint scorecard. Of the 578 rows, 513 used
  `episode_seed=2`; those rows were 505/8. The 65 non-seed-2 rows were 30/35.
  Treat the 535/43 headline as mostly repeated seed-2 sidecar telemetry.
- Post-train independent MCTS scorecard: Modal URL
  `https://modal.com/apps/modal-labs/shankha-dev/ap-G8BlfW9uUBtT7jTKxgtx0U`,
  summary
  `training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/attempts/attempt-20260509T151212Z-8b9db08f8fcb/eval/mcts-scoreboard-4096x64-iter64-maxstep4096/summary.json`.
  It scored `iteration_64` with `max_env_step=4096` and `num_simulations=8`.
  Vs random: LZ 13, random 19, mean steps 15.91, shaped -0.1862, actions
  `[290,219,0]`. Vs lagged: LZ 11, lagged 19, mean steps 266.25, shaped
  -0.2492, actions `[6475,2045,0]`. Vs track: LZ 0, track 31, mean steps
  144.34, shaped -0.9668, actions `[3335,1284,0]`.
- Read: longer training moved the independent policy from almost pure up to
  up+stay, but still 0 down and it does not beat random or scripted baselines.
  The seed-2 sidecar skew makes that independent fresh paired scorecard failure
  much less mysterious. Current read: no reliable baseline improvement yet.
  Infrastructure works; after the later deeper seed fix, the live blocker is
  policy learning/control signal.
- Seed handling instrumentation and the first dynamic-seed patch exposed a
  real LightZero env-manager issue. The deeper seed fix then passed the seed
  diversity gate on the next 1024/16 run: top seed was only 2/148 rows and
  `seed_dominance_warning=false`.
- Reset/randomization critique lives at
  `docs/working/lightzero_pong_reset_randomization_critique_2026-05-09.md`.
  Current read after the deeper seed fix: do not change reset profiles to hide
  the current failure. Canonical Pong stays fixed for this trust check; mild
  paddle-y jitter is only a later named profile.
- Seating/perspective note lives at
  `docs/working/lightzero_pong_seating_perspective_2026-05-09.md`. Current
  read: LightZero training controls `player_0` by default; paired eval tests
  checkpoint-as-`player_0` and checkpoint-as-`player_1`. The latest
  player0-only MCTS control after the deeper seed fix did not rescue the
  checkpoint: vs lagged `16-15` with one truncation, vs random `14-18`, vs
  track `0-29` with 3 truncations. Learned actions were `[3353,2073,0]`.
  Seat pairing was not hiding a good training-seat policy.
- Post-seed-fix run plan lives at
  `docs/working/lightzero_pong_post_seed_fix_run_plan_2026-05-09.md`. That
  first post-seed-fix run is now historical; it found the env-manager seed bug.
  The current evidence is the deeper-seed-fix experiment:
  `docs/experiments/2026-05-09-lightzero-dummy-pong-post-deep-seed-fix-run.md`.
  Train Modal:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-zYhiKPXQKKijsOXbN29aiK`.
  Paired scorecard Modal:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-GdYm0zsa8ifnsch1rUU9Vh`.
  Player0-only Modal:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-FriEt4BiggV7aYFoLqYgFq`.
  Run `lz-dpong-20260509T154530Z-b049f29edb64`, attempt
  `attempt-20260509T154530Z-ca60e4962603`. Train seed diversity passed:
  131 unique seeds, top seed 2/148 rows, `seed_dominance_warning=false`.
  Independent paired MCTS still failed: random `27-37`, lagged `27-33` with
  4 truncations, track `0-59` with 5 truncations. Learned paired actions were
  `[7285,4781,0]`. Player0-only did not rescue it: lagged `16-15`, random
  `14-18`, track `0-29`; learned actions `[3353,2073,0]`.
- Performance/Amdahl critique is active. Current hypothesis: do not jump to
  GPU, vectorization, or larger Modal jobs until measurement shows whether the
  bottleneck is Python env steps, LightZero MCTS/search, trainer loop, replay
  staging, or Modal startup. See
  `docs/research/training_loop_bottlenecks_amdhals_law_2026-05-09.md`.
- Current truth after the deeper seed fix and source/config audits: Modal
  training, checkpoint mirroring, independent eval, strict checkpoint load, and
  seed diversity work for this modest run. There is still no reliable baseline
  improvement, and current checkpoints should not be called progress. The live
  blocker is likely learner starvation/exploration/data volume, not seed
  dominance, checkpoint loading, horizon mismatch, or seat pairing.
- Self-play/data correction: MuZero data is generated by repeated environment
  interaction, so data volume can be scaled with more actors, episodes, and
  steps. Self-play is active now, but current LightZero dummy Pong is not final
  full self-play. It is learner ego versus a scripted opponent, usually
  `random_uniform`, or next learner ego versus a frozen checkpoint opponent.
  The frozen checkpoint opponent is the immediate practical step because most
  support already exists. Keep telemetry for survival steps, shaped score,
  action counts, wins, and opponent type. True current-policy two-seat play is
  a separate feasibility/design lane; do not pretend the current single-ego
  Pong setup is already that. Do not let this distract from the current better
  Pong run and official sanity lanes.
- Frozen-checkpoint self-play scout result lives at
  `docs/working/lightzero_pong_frozen_checkpoint_selfplay_plan_2026-05-09.md`.
  Recommendation: keep LightZero as the learner, add a checkpoint-backed
  opponent to `DummyPongLightZeroEnv`, load the frozen older checkpoint once
  in-container, and use low-simulation MCTS eval-mode as the first real
  opponent path. Direct policy-head greedy is only a dry smoke or fallback. A
  self-play implementation worker is investigating or implementing this minimal
  support in parallel.
- Active lanes right now: trusted LightZero whole-job Modal runs, honest
  checkpoint curves with survival steps and shaped score, learner-starvation
  probes, then frozen-checkpoint Pong wiring. Performance stance is scale after
  trust: scale actors/episodes/steps only after held-out scorecards are honest.

Decision note for confused future readers:
`docs/decisions/0005-main-pong-repository-library-choice.md` now records
LightZero-first as accepted for the next attempt. Read
`docs/research/lightzero_feature_fit_for_curvyzero.md` as the skeptical feature
fit audit: stochasticity, visual input, custom losses, architecture flexibility,
simultaneous-game limits, checkpoint/eval hooks, Modal fit, and exact failure
gates. Read
`docs/research/muzero_framework_vs_project_owned.md` as the actionable
LightZero-first adapter plan and implementation checklist: exact DI-engine env
interface, dummy Pong wrapper shape, visual/raster gate, architecture
flexibility, reward/telemetry rules, algorithm variants, scorecard telemetry,
Modal artifact mirror, first command, and Mctx fallback gates.

Repository/library split after optimizer correction:

- LightZero remains useful for stock controls, custom dummy Pong bridge tests,
  target audits, checkpoint/eval plumbing, and comparison runs. It remains a
  serious MuZero replication/control lane until credible reproduction or a clear
  blocker.
- Current CurvyTron next task: build a custom visual LightZero adapter
  prototype. It should expose stacked frames, discrete ego actions,
  reward/done/info, `action_mask`, `to_play=-1`, and full joint-action logging.
  It should not use ALE.
- Historical repo-native simultaneous `[B, P]` work remains useful architecture
  probe history: environment rows, player rows, batched policy/search,
  `joint_action [B, P]`, replay/rollout schema, scorecards, and profiling. Do
  not treat it as the current next task or final framework decision.
- Shared reporting contract:
  `docs/working/shared_training_reporting_contract_2026-05-09.md`. Both lanes
  should emit comparable profile metadata, contracts, timing buckets,
  throughput, latency, checkpoint ids, seed/reset details, and explicit
  non-claims.
- Mctx is the search library for the fallback project-owned trainer candidate.
  It gives us MuZero, Gumbel MuZero, and Stochastic MuZero search calls, but it
  does not provide the environment, replay buffer, model, optimizer, checkpoint
  format, or Modal run management.
- A project-owned trainer means writing the loop around the repo contracts:
  env adapter, model or policy, replay or rollout buffer, update loop,
  checkpoint, Modal job, scorecard, and profiler. Mctx can plug into the
  policy/search box later if search is worth the cost.
- It is now fair to say the tiny dummy Pong run used LightZero's trainer. Do
  not say the learned checkpoint is independently validated until the CurvyZero
  scoreboard can load and score the `.pth.tar`. Do not say "the main run is
  project-owned Mctx" unless LightZero fails and we explicitly switch.

Shortest already-working real MuZero replication:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke \
  --mode progression
```

LightZero-first sequence status:

```text
1. DONE: Run the LightZero feature-fit checklist from
   docs/research/lightzero_feature_fit_for_curvyzero.md.
2. DONE: Add DummyPongLightZeroEnv:
   BaseEnv reset/step, observation/action spaces, to_play=-1, action_mask=ones.
3. DONE: Add LightZero dummy Pong Modal smoke:
   patch CartPole MuZero config to env type dummy_pong_lightzero,
   model_type=mlp, observation_shape=10, action_space_size=3.
4. DONE: First command, config/import only:
   uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
     --env dummy_pong_lag1 \
     --feature-mode tabular_ego \
     --seed 0
5. DONE: First tiny trainer command:
   uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke \
     --mode progression \
     --env dummy_pong_lag1 \
     --feature-mode tabular_ego \
     --seed 0 \
     --opponent-policy random_uniform \
     --max-env-step 64 \
     --max-train-iter 2 \
     --num-simulations 2 \
     --batch-size 8 \
     --update-per-collect 1 \
     --n-evaluator-episode 1
6. DONE: Mirror LightZero logs/checkpoints plus env-side Pong telemetry into
   curvyzero-runs.
7. DONE: add a direct policy-head `.pth.tar` scoring path and run independent
   CurvyZero Pong scoreboards. This is greedy policy-head only.
8. DONE: expose `action_histogram_by_policy` in summary rows.
9. DONE: full LightZero MCTS/eval-mode scorecard ran across episodes/opponents
   for 512/8 `iteration_8`. It works as eval plumbing, but the policy is still
   effectively up-only.
10. DONE: fixed the small env/helper/config issues and reran MCTS with matching
   `max_env_step=512`. Treat the current 512/8 checkpoint as weak because the
   corrected MCTS action hist is `[4849,19,0]` and still never chooses down.
11. DONE: longer LightZero CPU train succeeded with wrapper caps `8192/64`;
   post-train `iteration_64` MCTS scorecard at `max_env_step=4096` moved from
   almost pure up to up+stay, but still 0 down and below random/scripted rows.
12. DONE: seed handling instrumentation and the deeper dynamic-seed fix landed.
   Configs record the setting, summaries include seed histogram/dominance
   checks, and the latest 1024/16 run passed seed diversity with 131 unique
   seeds and top seed 2/148 rows.
13. DONE/FAILED: post-deep-seed-fix 1024/16 CPU train and held-out independent
   MCTS scorecard ran. Seed trust passed, but policy quality failed. Paired
   scorecard: random `27-37`, lagged `27-33`, track `0-59`; learned actions
   `[7285,4781,0]`.
14. DONE/FAILED: player0-only MCTS control ran and did not rescue the
   checkpoint: lagged `16-15`, random `14-18`, track `0-29`; learned actions
   `[3353,2073,0]`.
15. DONE/FAILED: better-shaped lag1 knob run completed, but independent MCTS
   eval did not show robust survival gain and collapsed to down-only.
16. DONE/FAILED: tiny lagged-opponent run completed and showed wins over
   random/lagged rows, but independent eval collapsed to stay-only. Not clean
   progress.
17. DONE/PLUMBING ONLY: frozen-checkpoint opponent smokes ran against
   `iteration_0` and `iteration_16`. The path works for staged self-play
   plumbing and parent/new-checkpoint comparison, but it is not a policy-quality
   win.
18. NEXT: do not just run dummy Pong longer, and do not continue naive
   same-shape official Atari scale. GPU2048 is smoke-scale, not a failed
   official reproduction; next official Atari work should close the settings
   gap, especially 2 -> 50 simulations and env/update scale, before making a
   quality claim. Active next lanes are custom dummy Pong target-sidecar
   inspection/root-target audit, stock-vs-custom discrepancy mapping,
   smoke-only `raster_flat`, future `raster_stack4_ego` visual bridge design,
   and broad bug hunt. True current-policy two-seat play is still a separate
   feasibility/design lane.
19. Fallback to project-owned Mctx only if LightZero cannot call a real trainer
   while preserving metadata, artifacts, and Pong telemetry.
```

Modal implementation correction from Worker B:

```text
Use one whole-job Modal Function.
Mount curvyzero-runs at /runs.
Use task root training/lightzero-dummy-pong/<run_id>/.
Keep LightZero checkpoints as .pth.tar under checkpoints/lightzero/.
Do not use checkpoint_file_ref() for LightZero payloads because it assumes
checkpoint.npz.
Do not add GPUs, retries, or multi-node until the custom-env CPU smoke works.
```

Fallback project-owned lane if LightZero fails:

```text
one Modal Function
dummy Pong tabular or raster observations
model with representation/dynamics/prediction
Mctx search
environment-interaction replay
training update
checkpoint
scoreboard eval against random, track_ball, lagged_track_ball_1, parent, older
checkpoints
```

If any of those pieces are missing, say "not MuZero yet."

Use baselines as comparison rows, not as substitutes for this lane. Do not make
manual promotions or generations the core plan.

## Modal Rule

Use Modal for whole jobs and durable artifacts. Do not call Modal for env
steps, MCTS nodes, or per-action inference. Use Volumes for checkpoints,
replay chunks, metrics, summaries, and pointer files.

## Active Research Threads

- LightZero choice review: why LightZero first, what alternatives matter, and
  whether the decision still holds.
- Stochastic MuZero: whether Curvytron pickups, boosts, hazards, or noise make
  deterministic MuZero insufficient. Likely later branch unless the core env is
  actually stochastic from the agent's view.
- Robustness randomization: later training may add action noise, frozen
  controls, sticky actions, image noise, or domain randomization so policies do
  not overfit. This creates stochastic training data, but it may not require
  Stochastic MuZero. First try deterministic MuZero with explicit random seeds
  logged in env wrappers, train/eval scorecards, and replay metadata. Escalate
  to Stochastic MuZero only if the model must plan over real chance branches.
- Active delegated lanes:
  - LightZero/repo choice review: external MuZero repo fit, Modal friction, and
    why LightZero was chosen first.
  - LightZero feature-fit audit completed in
    `docs/research/lightzero_feature_fit_for_curvyzero.md`: LightZero is worth
    trying first because it is a complete trainer, but the first smoke must
    expose risks around stochasticity, custom losses, visual/raster input,
    custom architectures, simultaneous-game wrappers, scorecard telemetry,
    checkpoints/eval hooks, and Modal artifact mapping. Mctx remains
    fallback/comparison only if LightZero loses those fields or flexibility.
  - Stochastic MuZero decision note: deterministic first, chance nodes later
    only when the game has real unknown chance events.
  - Robustness/noise literature: sticky actions, frozen controls, image noise,
    action noise, domain randomization, and how much noise is too much.
  - Implementation support audit completed in
    `docs/research/muzero_repo_baseline_options.md`: LightZero is the least
    risky external trainer reference and the only stock trainer already proven
    on Modal; Mctx is the best search dependency for a project-owned candidate
    because it exposes MuZero, Gumbel MuZero, and Stochastic MuZero search
    without owning env/replay/training; sticky actions/action noise/observation
    transforms still need explicit wrappers and metadata.
  - Framework-vs-project-owned critique completed in
    `docs/research/muzero_framework_vs_project_owned.md`: lane A now wins the
    immediate direction. Try LightZero custom dummy Pong first; use
    project-owned Mctx only as fallback if the adapter or artifact/telemetry
    contract fails.
  See `docs/research/stochastic_muzero.md`,
  `docs/research/lightzero_feature_fit_for_curvyzero.md`,
  `docs/research/robustness_randomization_for_muzero.md`,
  `docs/research/muzero_repo_baseline_options.md`, and
  `docs/research/muzero_framework_vs_project_owned.md`.

## Shorthand

Truth: CartPole is infrastructure control only. Official sparse/delayed
TicTacToe and Connect4 bot-mode MuZero smokes passed on Modal. They are useful
controls only, not evidence that Pong/CurvyTron works, because Pong/CurvyTron
need visual/raster input and step-based control. Custom dummy Pong LightZero
MuZero ran at tiny, 512/8, longer CPU scale, and
post-deep-seed-fix 1024/16, but current Pong probes still have no clean signal.
Stock official visual Atari Pong now passes mechanically on Modal through
`src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`: dry/import/config
`ap-zJylQsu1IPoOIbtlhDlO2P`, capped trainer
`ap-MbyIGvX6R815WMZcYzcAyu`, artifacts under
`training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/...`.
Checkpoint-load probe
`src/curvyzero/infra/modal/lightzero_pong_checkpoint_probe.py` passed on
Modal `ap-YqqMmryhwgtKdFTbStnDKJ`: strict load true for direct `MuZeroModel`
and compiled `MuZeroPolicy` model, cheap forward true, input `[1,4,64,64]`,
logits `[1,6]`, value `[1,601]`; artifact
`training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/probe/lightzero_visual_pong_checkpoint_load_20260509T172430Z.json`.
Eval-only wrapper
`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py` passed on Modal
`ap-S0HADSUdYxYsy6y1yGj4mP`; earlier mismatch/shorter runs
`ap-oZ9TMQ03OEEVVMRfNoYPOh` and `ap-zRGq8FFHsLhoKC4Ovf37aa` are historical. It loaded
`training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/iteration_1.pth.tar`
and acted in a real ALE-backed LightZero Pong env through
`MuZeroPolicy.eval_mode.forward` with no fallback and `num_simulations=2`. It
ran 64 steps: actions `{0:64}`, rewards `{-1.0:1,0.0:63}`, total reward `-1`,
nonzero reward at step 60, done at step 63 with `TimeLimit.truncated:true`;
raw env obs `[1,64,64]` was stacked to policy input `[4,64,64]`. This proves
train/checkpoint/load/eval mechanics, not policy quality. Follow-up scale128
control reached 128 real ALE eval steps and two true `-1` rewards, but both old
and new `iteration_1` checkpoints stayed all-action-0 with return `-2.0`.
Dummy Pong `raster_flat` train plus matching independent MCTS scorecard works
mechanically on Modal, but it is not a quality win: eval aggregate actions
were `[424,146,0]`, zero down, and bad versus random/track. It is now
smoke-only. Keep `tabular_ego` as the debug/learning lane; the next honest
visual bridge is separate `raster_stack4_ego` with frame history.
Project-owned Pong/Curvy MuZero has not run.
Modal training, checkpoint mirroring, strict checkpoint load, independent MCTS
eval, and seed diversity now work for the modest trust run. The learned policy
has not shown reliable baseline improvement; do not call current checkpoints
progress. The deeper seed fix passed seed trust: 131 unique seeds, top seed
2/148 rows,
`seed_dominance_warning=false`. Independent MCTS still failed:
paired rows were random `27-37`, lagged `27-33`, track `0-59`, with learned
actions `[7285,4781,0]`. Player0-only did not rescue it: lagged `16-15`,
random `14-18`, track `0-29`, actions `[3353,2073,0]`. Current blocker is
policy learning/control signal and eval action collapse. Latest runs sharpen
that: the better-shaped lag1 run collapsed down-only in eval, the tiny
lagged-opponent run collapsed stay-only in eval, frozen-checkpoint smokes
proved plumbing only, and the sparse-settings probe did not robustly improve
survival/shaped score. Pure 2x tabular sparse ladder also failed held-out
survival/shaped/raw/action entropy, with `iteration_32` all-up entropy `0.0`.
Eval roots/action choices remain weak or tied. Active next lanes are official
LightZero Atari evaluator collation/API plumbing or fuller-recipe training,
custom dummy Pong target-sidecar oracle-label join, opt-in custom support-scale
proof/fix, stock-vs-custom discrepancy mapping, smoke-only `raster_flat`,
future `raster_stack4_ego` visual bridge design, and broad bug hunt. Keep the
custom lane separate from the official Atari
reproduction lane; the support-scale patch is custom dummy Pong only and
official Atari is untouched. CurvyTron transfer rules now live in
`docs/working/training_lessons_for_curvytron_2026-05-09.md`.
MuZero/LightZero can scale training data by running more environment actors,
episodes, and steps, but only after the run is trusted. Current dummy Pong is
not final multiplayer self-play; it is learner ego versus scripted or frozen
opponent. Self-play is active now: frozen checkpoint opponent first because
support mostly exists; true current-policy two-seat play is separate design
work. Report survival steps, shaped score, action counts, wins, and opponent
type every time. Core plan: raster support, sparse ladder, search/debug
checks, honest checkpoint curves, scale actors/steps only after trust, and keep
frozen-checkpoint play clearly labeled as staged plumbing until policy quality
improves.
Performance/Amdahl critique says measure Python env steps, LightZero MCTS,
trainer loop, replay staging, and Modal startup before jumping to
GPU/vectorization. Mctx is fallback/search dependency if LightZero fights the
interface. Randomness for robustness is a wrapper/training-data question first,
not automatically a new algorithm. Also: current LightZero Pong is
learner-vs-scripted, not final self-play. The active staged self-play step is
learner-vs-frozen-checkpoint; full current-policy two-seat play and
multiplayer/joint-action search are separate feasibility/design work.
Wrapper note: source timing is not fatal. A single-ego wrapper can step the env
by constructing a wrapper joint action/control snapshot while LightZero controls
one seat. Only call it true live simultaneous self-play after opponent policy,
checkpoint lineage, seat metadata, replay rows, and eval separation are
explicit.

## Short Restart Mnemonic

Real MuZero next. Modal whole-job. Pong telemetry, not just wins. Docs before
memory. Workers do work; main thread decides. Simple words; no pretending.
