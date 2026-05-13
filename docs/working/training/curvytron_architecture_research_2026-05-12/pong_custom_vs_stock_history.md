# Pong Custom Vs Stock History

Purpose: reconstruct the Pong history as an analogy for CurvyTron. The main
lesson is blunt: the custom paths taught us plumbing and diagnostics, but the
credible learning signal arrived when we moved back toward stock LightZero
Pong and evaluated survival curves late enough.

## Bottom Line

Custom dummy Pong was not a failed proof that MuZero cannot learn. It was mostly
a sequence of custom environments, custom objectives, custom scoreboards, and
custom replay/eval bridges. Those paths often passed infrastructure gates, but
they were weak or inconclusive as policy-quality evidence.

The stock or near-stock LightZero Atari Pong path eventually did show signal.
It used `PongNoFrameskip-v4`, LightZero's Atari MuZero config,
`lzero.entry.train_muzero`, strict checkpoint loading, stock-only evaluator
passes where possible, and survival-first checkpoint curves. Early tiny rungs
looked flat; later stock64 curves moved.

The CurvyTron analogy is direct: do not scale a custom collector/target/replay
adapter as if it were the stock MuZero loop. Keep the custom layer only where
CurvyTron truly needs it, and route everything else through stock LightZero or a
native-compatible bridge.

## What The Custom Pong Paths Proved

The first Modal dummy Pong train/eval path proved artifact discipline, not
policy quality. Its own wrapper says it proves Modal Volume discipline and remote
reproduction, and does not prove the self-play objective or a strong policy
(`src/curvyzero/infra/modal/dummy_pong_train_attempt.py:10`,
`src/curvyzero/infra/modal/dummy_pong_train_attempt.py:76`). The one-game smoke
scoreboard passed but was explicitly poor quality (`docs/experiments/2026-05-09-modal-dummy-pong-train-to-scoreboard-smoke.md:39`,
`docs/experiments/2026-05-09-modal-dummy-pong-train-to-scoreboard-smoke.md:44`).

The repaired NumPy self-play run still failed the real `track_ball` gate. Across
epochs 25, 50, 75, and 100 it got 0/64 learned wins against `track_ball`
(`docs/experiments/2026-05-09-modal-pong-selfplay-repair-run.md:54`). The
corrected survival/loss-delay audit showed some pressure differences, but the
interpretation stayed weak: no checkpoint won against `track_ball`, and blind
scaling was not justified (`docs/experiments/2026-05-09-modal-pong-selfplay-repair-run.md:100`,
`docs/experiments/2026-05-09-modal-pong-selfplay-repair-run.md:109`).

The parallel self-play sweep did not rescue that learner. All variants still had
0 learned wins against `track_ball`, and even after measuring mean steps,
truncations, and shaped loss delay, the sweep failed to beat the earlier repair
run (`docs/experiments/2026-05-09-modal-pong-parallel-sweep.md:134`,
`docs/experiments/2026-05-09-modal-pong-parallel-sweep.md:147`). The follow-up
was to stop blind self-play scaling for that learner
(`docs/experiments/2026-05-09-modal-pong-parallel-sweep.md:161`).

The CEM-v2 dummy path was the one custom positive, but it was not a stock Pong or
MuZero proof. It searched a compact geometry-only policy
(`src/curvyzero/training/dummy_pong_cem_train.py:1`,
`src/curvyzero/training/dummy_pong_cem_train.py:12`) and produced real score
pressure against the easier `lagged_track_ball_1` target: 53/64 heldout
scoreboard wins (`docs/experiments/2026-05-09-dummy-pong-cem-v2-lagged-track-ball-1-monitor.md:63`,
`docs/experiments/2026-05-09-dummy-pong-cem-v2-lagged-track-ball-1-monitor.md:74`).
It still tied/truncated against default `track_ball`, which the doc labels a
survival/tie diagnostic rather than a hard win gate
(`docs/experiments/2026-05-09-dummy-pong-cem-v2-lagged-track-ball-1-monitor.md:85`).

## What The Custom LightZero Dummy Pong Path Changed

The LightZero dummy Pong adapter was explicitly custom: one LightZero ego paddle,
a scripted or checkpoint opponent supplied by the wrapper, and telemetry in
`info` (`src/curvyzero/training/lightzero_dummy_pong_env.py:1`,
`src/curvyzero/training/lightzero_dummy_pong_env.py:67`). The config smoke says
it does not train; it only verifies that the custom env can be imported, reset,
stepped, and targeted by a tiny config
(`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:1`,
`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:11`).

The initial implementation note called it a staged sanity wrapper, not final
multiplayer self-play: learner ego versus scripted or frozen opponent, with
survival and shaped loss-delay required in reports
(`docs/experiments/2026-05-09-lightzero-dummy-pong-implementation-inspection.md:6`,
`docs/experiments/2026-05-09-lightzero-dummy-pong-implementation-inspection.md:26`).
The same note recorded that the first train smoke called `train_muzero`, but at
tiny CPU custom-env scale: observation shape 10, MLP model, action space 3,
`max_env_step=64`, `num_simulations=2`, `batch_size=8`
(`docs/experiments/2026-05-09-lightzero-dummy-pong-implementation-inspection.md:115`,
`docs/experiments/2026-05-09-lightzero-dummy-pong-implementation-inspection.md:126`).

Independent eval exposed the first big trap: policy-head evaluation was constant
`up`, not meaningful learned Pong. The bug audit says the policy-head scoreboard
was a negative loader/action-collapse canary only
(`docs/experiments/2026-05-09-lightzero-dummy-pong-bug-telemetry-audit.md:7`,
`docs/experiments/2026-05-09-lightzero-dummy-pong-bug-telemetry-audit.md:25`).
MCTS loader smoke later passed, so checkpoint loading was no longer the main
blocker (`docs/experiments/2026-05-09-lightzero-dummy-pong-bug-telemetry-audit.md:91`,
`docs/experiments/2026-05-09-lightzero-dummy-pong-bug-telemetry-audit.md:106`).

Full MCTS scorecards then made the quality failure sharper. The 512/8 MCTS
scorecard worked outside the training loop, but the checkpoint was effectively
up-only and never chose down (`docs/experiments/2026-05-09-lightzero-dummy-pong-mcts-scorecard.md:35`,
`docs/experiments/2026-05-09-lightzero-dummy-pong-mcts-scorecard.md:39`). A
corrected 512-step rerun still showed 99.61 percent action 0 and zero down moves
(`docs/experiments/2026-05-09-lightzero-dummy-pong-mcts-scorecard.md:102`).
Longer training moved to up+stay but still did not beat random or scripted
baselines (`docs/experiments/2026-05-09-lightzero-dummy-pong-mcts-scorecard.md:156`).

The seed fix removed one plumbing excuse but not the policy problem. After the
deep seed fix, the run had real seed diversity
(`docs/experiments/2026-05-09-lightzero-dummy-pong-post-deep-seed-fix-run.md:75`),
but independent MCTS still lost to random, lagged, and `track_ball`, and chose
zero action-index-2 moves (`docs/experiments/2026-05-09-lightzero-dummy-pong-post-deep-seed-fix-run.md:163`,
`docs/experiments/2026-05-09-lightzero-dummy-pong-post-deep-seed-fix-run.md:173`).

Training against `lagged_track_ball_1` produced better small-horizon win counts,
but it collapsed to all `stay` actions and still lost decisively to `track_ball`
(`docs/experiments/2026-05-09-lightzero-dummy-pong-lagged-opponent-smoke.md:164`,
`docs/experiments/2026-05-09-lightzero-dummy-pong-lagged-opponent-smoke.md:173`).
The sparse scale ladder was explicitly custom tabular dummy Pong, not visual
training, and pure 2x budget did not rescue it: final/best checkpoints still
chose all-up and action entropy collapsed
(`docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-scale-ladder-rung0-rung1.md:8`,
`docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-scale-ladder-rung0-rung1.md:237`).

Plain read: the custom LightZero dummy path got us loaders, MCTS scorecards,
seed audits, action histograms, survival curves, and failure visibility. It did
not give us a trusted learning architecture.

## What The Stock And Near-Stock Pong Path Changed Less

The stock-control Pong wrappers stayed close to official LightZero Atari. The
tiny train wrapper says it is stock ALE `PongNoFrameskip-v4` through
`zoo.atari.config.atari_muzero_config` and `train_muzero`, while warning that
the tiny caps are infrastructure only (`src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py:1`,
`src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py:15`).

The exact wrapper was even narrower: installed `LightZero==0.2.0`, official
Atari config, `lzero.entry.train_muzero`, stock `max_env_step=200000` in exact
mode, and only `exp_name` patched for Volume artifact location. Shortened runs
were labeled faithful-short, and survival shaping required explicit shaped run
ids (`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:1`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:7`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:20`).

The eval wrapper loaded real ALE Pong checkpoints with strict/no-fallback
options and could run the stock `lzero.worker.MuZeroEvaluator`
(`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:1`,
`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:99`). Separate Agent96
eval kept the older 96x96 surface separate from stock64 so numbers were not mixed
(`src/curvyzero/infra/modal/lightzero_pong_muzero_agent96_eval.py:1`).

The early official Atari rungs were not enough. GPU1024 showed a small real
signal, but not solved policy (`docs/experiments/2026-05-09-modal-lightzero-pong-gpu1024-control.md:132`).
GPU2048 did not strengthen that signal, though the wrapper still trained,
mirrored checkpoints, and evaluated real ALE Pong without fallback
(`docs/experiments/2026-05-09-modal-lightzero-pong-gpu2048-control.md:121`,
`docs/experiments/2026-05-09-modal-lightzero-pong-gpu2048-control.md:131`).
4096/sim10 was an infrastructure pass and quality fail, with action collapse
under strict eval (`docs/experiments/2026-05-09-modal-lightzero-pong-4096-sim10.md:206`,
`docs/experiments/2026-05-09-modal-lightzero-pong-4096-sim10.md:215`).
8192/sim25 exposed a stock-accounting hazard: restoring auto update semantics
produced about 90 GB of checkpoints, and the periodic checkpoints still did not
improve in the 256-step eval (`docs/experiments/2026-05-09-modal-lightzero-pong-8192-sim25.md:186`,
`docs/experiments/2026-05-09-modal-lightzero-pong-8192-sim25.md:262`).

The near-stock faithful-short run was useful but not a final proof. It used the
installed package config and stock trainer path, but shortened `max_env_step` to
8192 and patched `exp_name` (`docs/experiments/2026-05-09-modal-lightzero-exact-faithful-short-8192-relpath.md:11`).
Its interpretation was a first weak signal, not solved Pong
(`docs/experiments/2026-05-09-modal-lightzero-exact-faithful-short-8192-relpath.md:141`,
`docs/experiments/2026-05-09-modal-lightzero-exact-faithful-short-8192-relpath.md:154`).

The important shift came from survival-first curves over later checkpoints. The
failure audit says the stock/near-stock visual Pong lane did work once checkpoint
horizon was long enough, and that the earlier flat reads were premature
(`docs/working/training/pong_replication_failure_audit_2026-05-11.md:32`).
It records later positive rows, including seed 1 reaching the 2048-step cap and
Wave11 runs showing real late survival gains
(`docs/working/training/pong_replication_failure_audit_2026-05-11.md:18`,
`docs/working/training/pong_replication_failure_audit_2026-05-11.md:20`).

The stock64 comparison made the same point more cleanly: all checked runs used
installed LightZero 0.2.0 stock64 Atari Pong, sparse stock reward, no reward
shaping, 50 simulations, 8 collectors, and 3 evaluators
(`docs/working/training/pong_stock64_signal_comparison_2026-05-11.md:13`). Later
evals showed `s114`, `s120`, `s121`, `s122`, `s142`, `s113`, and `s123`
improving survival (`docs/working/training/pong_stock64_signal_comparison_2026-05-11.md:19`,
`docs/working/training/pong_stock64_signal_comparison_2026-05-11.md:64`).
The conclusion was not solved Pong; it was survival learning first, with score
lagging behind survival (`docs/working/training/pong_stock64_signal_comparison_2026-05-11.md:70`).

Wave11 is the clearest historical analogy. It was stock visual Pong evaluator
data, not CurvyTron, and its normal seeds showed strong late survival curves:
`s73`, `s74`, and `s76` moved substantially, with `s74` reaching the 2048 cap at
three late checkpoints (`docs/experiments/2026-05-10-lightzero-wave11-pong-survival-curves.md:3`,
`docs/experiments/2026-05-10-lightzero-wave11-pong-survival-curves.md:30`,
`docs/experiments/2026-05-10-lightzero-wave11-pong-survival-curves.md:38`).

## What We Changed Too Much

In custom Pong, we changed the environment, observation surface, opponent model,
action count, reward interpretation, evaluator path, checkpoint loader, and often
the learning objective. Some changes were reasonable diagnostics, but together
they made a flat or collapsed result hard to interpret.

The biggest changes:

- Replaced stock Atari visual Pong with dummy `tabular_ego` or raster/geometry
  feature surfaces (`docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-scale-ladder-rung0-rung1.md:8`,
  `src/curvyzero/training/lightzero_dummy_pong_env.py:97`).
- Replaced the normal ALE action space with 3 dummy actions and often a scripted
  opponent supplied inside the wrapper (`src/curvyzero/training/lightzero_dummy_pong_env.py:171`,
  `src/curvyzero/training/lightzero_dummy_pong_env.py:192`).
- Let trainer-side telemetry look positive before independent checkpoint eval
  proved the exported policy was degenerate
  (`docs/experiments/2026-05-09-lightzero-dummy-pong-bug-telemetry-audit.md:122`,
  `docs/experiments/2026-05-09-lightzero-dummy-pong-bug-telemetry-audit.md:135`).
- Treated small score/win rows as meaningful before action histograms, survival,
  and strict MCTS scorecards were present
  (`docs/experiments/2026-05-09-lightzero-dummy-pong-bug-telemetry-audit.md:145`,
  `docs/experiments/2026-05-09-lightzero-dummy-pong-mcts-scorecard.md:39`).
- Tried scale or more simulations as fixes before the training/eval contract was
  trusted (`docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-scale-ladder-rung0-rung1.md:12`,
  `docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-scale-ladder-rung0-rung1.md:243`).

The stock Pong path changed less of the learning contract. It still had wrapper
and cap mistakes, but the core collector, GameSegment/GameBuffer, learner, and
evaluator path stayed LightZero-owned.

## Lessons For CurvyTron

1. Keep Pong as a control, not as a guarantee. Pong proves the broad LightZero
   stack can carry survival signal; it does not prove CurvyTron two-seat
   self-play is solved (`docs/working/training/curvytron_vs_pong_architecture_reflection_2026-05-10.md:115`).

2. Prefer stock `train_muzero` for the first credible CurvyTron learning gates.
   The stock dataflow is env reset, collector policy/search, env step,
   GameSegment, MuZeroGameBuffer, learner, checkpoints, and evaluator
   (`docs/working/training/curvytron_architecture_research_2026-05-12/stock_lightzero_dataflow.md:28`).
   Calling `collect_mode.forward` or `learn_mode.forward` directly is not the
   whole loop (`docs/working/training/curvytron_architecture_research_2026-05-12/stock_lightzero_dataflow.md:43`).

3. Treat the current custom two-seat CurvyTron path as a collector prototype
   unless it feeds native-compatible replay or has parity tests. The May 12
   reconciliation says the scaled path owned collection, replay rows, targets,
   and checkpointing; it used LightZero policy/search but did not call
   `train_muzero` or use LightZero's collector/GameBuffer
   (`docs/working/training/curvytron_train_muzero_reconciliation_2026-05-12.md:21`,
   `docs/working/training/curvytron_train_muzero_reconciliation_2026-05-12.md:116`).

4. Do not confuse shape compatibility with semantic compatibility. The custom
   MuZero autopsy warns that hand-built targets can have the right shape but the
   wrong meaning, and calls out repeated rewards/policies, local replay, direct
   `learn_mode.forward`, and `to_play` risks
   (`docs/working/training/archive_2026-05-12_two_seat_purge/custom_vs_stock_muzero_contract_autopsy_2026-05-11.md:23`,
   `docs/working/training/archive_2026-05-12_two_seat_purge/custom_vs_stock_muzero_contract_autopsy_2026-05-11.md:39`,
   `docs/working/training/archive_2026-05-12_two_seat_purge/custom_vs_stock_muzero_contract_autopsy_2026-05-11.md:40`).

5. Evaluate survival curves against same-run `iteration_0` on held-out seed
   panels. The Pong mistake was judging too early and from score alone; the
   stock64 notes say survival moved before score got good
   (`docs/working/training/pong_stock64_signal_comparison_2026-05-11.md:84`,
   `docs/working/training/pong_stock64_signal_comparison_2026-05-11.md:87`).

6. Report action histograms, strict-load status, eval caps, seed panels,
   opponent/frozen-checkpoint lineage, replay age, and target ranges on every
   learning claim. The custom Pong history shows that hidden action collapse can
   make a scoreboard look merely weak when the real failure is degenerate policy
   behavior (`docs/experiments/2026-05-09-lightzero-dummy-pong-bug-telemetry-audit.md:145`).

7. For CurvyTron specifically, use one of two gates before another large run:
   stock-loop CurvyTron with fixed/frozen or centralized joint-action control, or
   a native-buffer bridge that converts two-seat trajectories into LightZero
   GameSegments and samples through MuZeroGameBuffer
   (`docs/working/training/curvytron_no_learning_failure_audit_2026-05-12.md:165`,
   `docs/working/training/curvytron_train_muzero_reconciliation_2026-05-12.md:189`).

Short version: custom Pong helped us learn what to measure. Stock Pong showed
that the trusted LightZero loop can learn. CurvyTron should borrow the trusted
loop first and customize only the simultaneous-action boundary that stock
LightZero cannot represent directly.
