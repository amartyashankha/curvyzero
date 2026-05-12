# MuZero Training Footguns - 2026-05-11

Purpose: explain, in plain language, why the stock installed LightZero Pong
path is now producing useful survival signal while several custom or
custom-ish paths stayed flat. This is not a new experiment log. It is a
checklist for avoiding silent MuZero training contract drift.

No pytest was run for this note. I only read code and docs.

## Short Read

The credible Pong signal came from the boring path:

- installed `LightZero==0.2.0`;
- stock Atari Pong config;
- stock `lzero.entry.train_muzero`;
- stock collector, MCTS/search, GameBuffer/replay, learner, and evaluator
  contracts;
- strict checkpoint loading;
- same-run `iteration_0` vs later checkpoint curves;
- stock-only survival evals over seed panels.

The custom paths often changed several things at once: env API, reward, frame
stack shape, replay rows, target construction, checkpoint format, eval loop,
seed/reset policy, or the meaning of self-play. Some of those changes are
reasonable diagnostics. They are footguns when we read their output as normal
MuZero learning evidence.

## Current Evidence Snapshot

Stock installed LightZero Pong now has a real survival curve, not solved Pong.
The replication monitor calls out run `s122`: mean survival moved
`761.5 -> 934.5 -> 988.562 -> 1295.06` over checkpoints
`0/7000/10000/12000`, while mean score was still mostly losing
`-21 -> -20.25 -> -19.75 -> -17.9375`
(`docs/working/lightzero_pong_replication_monitor_2026-05-11.md:77`,
`docs/working/lightzero_pong_replication_monitor_2026-05-11.md:79`).
That matters because score alone would hide early survival learning
(`docs/working/lightzero_pong_replication_monitor_2026-05-11.md:117`,
`docs/working/lightzero_pong_replication_monitor_2026-05-11.md:127`).

Earlier reconciliation found the same pattern at smaller scale: one normal
LightZero Atari Pong seed had a strong later checkpoint, but this was not a
stable cross-seed solved-Pong claim
(`docs/working/lightzero_signal_reconciliation_2026-05-10.md:21`,
`docs/working/lightzero_signal_reconciliation_2026-05-10.md:23`,
`docs/working/lightzero_signal_reconciliation_2026-05-10.md:84`).

Two follow-up notes sharpen the boundary. The failure audit says the real proof
lane is stock visual Pong survival against the same run's `iteration_0`, with
score secondary until survival moves
(`docs/working/training/pong_replication_failure_audit_2026-05-11.md:9`,
`docs/working/training/pong_replication_failure_audit_2026-05-11.md:10`).
The stock64 comparison says `s114`, `s120`, `s121`, `s122`, and `s142` all use
the same installed LightZero stock64 Atari Pong surface, so the split between
flat and improving rows is not explained by obvious config drift
(`docs/working/training/pong_stock64_signal_comparison_2026-05-11.md:13`,
`docs/working/training/pong_stock64_signal_comparison_2026-05-11.md:17`).

## What Should Have Worked But Did Not Yet?

These are the uncomfortable rows: they are not custom trainer excuses. The
latest eval pass corrected the earlier pessimistic read: several rows that
looked flat at `7k-11k` improved later.

- `s114`, `s120`, and `s121` should have been useful stock64 Pong controls.
  They now show later survival gains: `s114` reached `1612.12` mean steps by
  `13k`, `s120` reached `961.375` by `14k`, and `s121` reached `1579.38` by
  `17k`. The footgun was judging too early, not a proven broken stock path.
- `s122` did work on survival, but only as one H100/100k seed so far. It does
  not prove H100 caused learning, because L4/T4 rows also improved later.
  `s122` is now close to the `2048` eval cap, so the next footgun is capped
  evals hiding further survival progress.
- `s142` is the clean repeat of `s122`'s H100/100k surface. It is weaker than
  `s122` so far, but it improved by `12k-15k`, so it supports the “delayed
  survival signal” read.
- Some spawned/no-root rows, such as `s111`, `s112`, and `s130`-`s133`, did not
  produce a policy verdict because roots or progress were not visible. That is
  a Modal launch / detached call / Volume visibility footgun, not a MuZero
  learning result
  (`docs/working/training/pong_replication_failure_audit_2026-05-11.md:23`,
  `src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:2216`,
  `src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:2241`).
- Eval speed is a decision-loop footgun. A serious curve needs strict stock
  eval over seed panels, but doing checkpoint-by-checkpoint, seed-by-seed evals
  serially makes us react too late. The queue now groups checkpoints, maps
  seeds inside each Modal call, and can execute many local launches in parallel
  (`scripts/lightzero_live_eval_queue.py:391`,
  `scripts/lightzero_live_eval_queue.py:398`,
  `scripts/lightzero_live_eval_queue.py:539`,
  `scripts/lightzero_live_eval_queue.py:549`,
  `scripts/lightzero_live_eval_queue.py:692`,
  `scripts/lightzero_live_eval_queue.py:694`).

Current plain-English read: `s122` is a real positive survival row; `s114`,
`s120`, and `s121` are real stock-lane disappointments so far; `s142` is the
next clean check; launch roots and slow eval can erase days without proving
anything about the policy.

## The Good Pong Spine

The exact reproduction wrapper is intentionally narrow. It imports
`zoo.atari.config.atari_muzero_config` and calls
`lzero.entry.train_muzero` (`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:7`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:9`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:1723`).

The exact surface guard expects:

- env: `PongNoFrameskip-v4`, `atari_lightzero`;
- model: conv, `[4,64,64]`, action space `6`;
- collector envs: `8`;
- evaluator envs: `3`;
- `num_simulations=50`;
- `batch_size=256`;
- `update_per_collect=None`;
- `replay_ratio=0.25`;
- `game_segment_length=400`;
- `replay_buffer_size=1000000`;
- `frame_stack_num=4`.

Refs: `src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:915`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:925`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:928`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:935`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:936`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:939`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:943`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:944`.

The wrapper's own note says the exact mode mutates only `exp_name`, keeps stock
checkpoint cadence unless explicitly overridden, does not cap episodes, does
not reduce collector/evaluator counts, and does not change
`update_per_collect` (`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:1766`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:1772`).

## Footgun List

### 1. Collector/Search/Replay/Learner Contract

Footgun: calling `MuZeroPolicy.collect_mode.forward` and
`MuZeroPolicy.learn_mode.forward` directly is not the same as running
`train_muzero`.

Why it matters: normal LightZero training does more than select an action and
run a loss. It packages game segments, stores search visit targets, samples
from GameBuffer, updates priorities, calls learner hooks, saves checkpoints,
and runs evaluator cadence. A direct local adapter can update weights while
skipping important lifecycle pieces.

Good path: stock Pong calls `train_muzero` with the installed config
(`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:1578`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:1723`).

Custom two-seat path: it says `called_train_muzero: False`, says it does not
use the LightZero collector, and says its replay is a local
`learn_mode.forward` adapter, not LightZero's upstream GameBuffer target
builder (`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:845`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:852`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:854`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:857`).

Docs reached the same conclusion: native uses GameBuffer and BaseLearner;
custom stores plain replay rows and directly calls `learn_mode.forward`
(`docs/working/training/custom_vs_native_lightzero_diff_2026-05-10.md:46`,
`docs/working/training/custom_vs_native_lightzero_diff_2026-05-10.md:47`,
`docs/working/training/custom_vs_native_lightzero_diff_2026-05-10.md:59`).

Guardrail: if a path does not call `train_muzero`, label it as an actor/learner
smoke. Before trusting learning, either feed native-compatible `GameSegment`
objects into `MuZeroGameBuffer`, or write an explicit repo-owned learner
contract.

### 2. Search Target Meaning

Footgun: the action the actor executes is not enough. MuZero learns policy
targets from search visit distributions.

Why it matters: a run can have plausible actions but weak or wrong
`child_visit_segment` targets. Dummy Pong docs found that actual policy targets
live in `GameSegment.child_visit_segment`, and the old wrapper did not mirror
the replay buffer or GameSegment objects
(`docs/working/lightzero_dummy_pong_target_policy_persistence_audit_2026-05-09.md:15`,
`docs/working/lightzero_dummy_pong_target_policy_persistence_audit_2026-05-09.md:23`,
`docs/working/lightzero_dummy_pong_target_policy_persistence_audit_2026-05-09.md:27`).

Code later mirrored compact GameSegment target rows by patching
`MuZeroCollector.collect` and reading `child_visit_segment`
(`src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py:394`,
`src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py:428`,
`src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py:450`).

Guardrail: every custom trainer needs a target audit: executed action,
legal mask, visit distribution, root value, reward, done, and the exact learner
target arrays.

### 3. Replay Scope And Sample Size

Footgun: small per-iteration replay can look like training but teach almost
nothing.

Why it matters: the two-seat smoke can sample from `current_iteration` or
`accumulated`, but the default wrapper still starts at `current_iteration`
(`src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py:105`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:75`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:415`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:428`).
The audit notes that tiny settings can train on only two rows, and that this is
a flat-learning risk if treated as a real run
(`docs/working/training/curvytron_two_seat_bug_audit_2026-05-10.md:16`,
`docs/working/training/curvytron_two_seat_bug_audit_2026-05-10.md:98`).

Guardrail: report rows available, rows sampled, replay age, episode ids, and
update count. Use accumulated replay for any run meant to resemble training.

### 4. Frame Stack Shape

Footgun: `[4,64,64]` can mean either four environment frames or one custom
already-stacked feature. Those are not interchangeable.

Good stock Pong: model and env both use `[4,64,64]` with `frame_stack_num=4`
(`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:925`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:944`).

CurvyTron native visual survival: the model also sees `[4,64,64]`, but the
custom env says `frame_stack_num=1` and `image_channel=4`
(`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1199`,
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1200`,
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1227`,
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1230`).

Two-seat path: the outer batch is `[B,2,4,64,64]`, then each active player row
is fed to LightZero as `[4,64,64]`
(`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2246`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2255`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:1318`).

Manual Pong eval had to rebuild frame stacks around the env output
(`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:862`,
`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:877`,
`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:1112`,
`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:1149`).

Guardrail: every checkpoint artifact should name `observation_shape`,
`frame_stack_num`, `image_channel`, and whether stacking was done by the env,
wrapper, or eval code.

### 5. Env API Shape

Footgun: LightZero's native env API is one scalar action per env row. CurvyTron
physics wants simultaneous `joint_action[B,P]`.

Good Pong-like path: policy emits one action, env owns the rest of the world.

Two-seat path: it builds one policy row per active player, selects actions,
maps rows back to `joint_action`, and only then steps the env
(`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:986`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:1069`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:1193`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:1247`).

Optimizer docs say the custom collector is useful, but the replay should become
native-compatible one `GameSegment` per seat perspective before we trust the
learner (`docs/working/optimizer/full_training_loop_worldview_2026-05-11.md:111`,
`docs/working/optimizer/full_training_loop_worldview_2026-05-11.md:118`,
`docs/working/optimizer/full_training_loop_worldview_2026-05-11.md:123`,
`docs/working/optimizer/full_training_loop_worldview_2026-05-11.md:127`).

Guardrail: either hide opponent/world action inside `env.step(action)` for a
Pong-like run, or loudly label a custom simultaneous actor and prove its replay
conversion.

### 6. Self-Play Meaning

Footgun: "self-play" can mean several different things.

Stock Atari Pong under LightZero is single-agent Atari training against the ALE
game. It is not two live policies controlling two seats.

CurvyTron native visual survival explicitly records `current_policy_self_play`
and `trusted_current_policy_self_play` claims, often false or caveated
(`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:677`,
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:680`,
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:867`,
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:870`).

Two-seat current-policy smoke uses the same live policy object for both seats,
but optimizer docs warn this is independent seat search, not joint-action MCTS
(`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:1247`,
`docs/working/optimizer/distributed_muzero_architecture_research_2026-05-11.md:100`,
`docs/working/optimizer/distributed_muzero_architecture_research_2026-05-11.md:103`).

Guardrail: every run should say one of: ALE single-agent, fixed opponent,
frozen/checkpoint-lagged opponent, same-policy two-seat independent search, or
joint-action MCTS. Do not let all of these collapse into "self-play."

### 7. Eval Meaning

Footgun: manual eval, stock LightZero evaluator, scorecard sidecars, and
trainer-side `ckpt_best` are different measurements.

The current serious eval helper uses strict checkpoint loading, stock
`MuZeroEvaluator`, no model fallback, and stock-only mode for survival curves
(`scripts/lightzero_live_eval_queue.py:222`,
`scripts/lightzero_live_eval_queue.py:223`,
`scripts/lightzero_live_eval_queue.py:648`,
`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:1463`,
`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:1544`,
`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:1880`).

The eval helper also warns when `max_episode_steps` and `max_eval_steps` differ
(`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:282`).

Reconciliation found that shorter eval caps hid survival differences and that
old eval defaults accidentally used `update_per_collect=1` where stock training
used `None` (`docs/working/lightzero_signal_reconciliation_2026-05-10.md:53`,
`docs/working/lightzero_signal_reconciliation_2026-05-10.md:58`).

Decision-loop footgun: strict eval is necessary, but a slow serial eval loop
can leave us arguing from stale or missing curves. The queue helper was built
to group pending checkpoints, keep the strict stock-eval contract, and launch
groups in parallel when `--execute` is used
(`scripts/lightzero_live_eval_queue.py:638`,
`scripts/lightzero_live_eval_queue.py:648`,
`scripts/lightzero_live_eval_queue.py:681`,
`scripts/lightzero_live_eval_queue.py:694`).

Guardrail: learning claims should use same-run `iteration_0` and later
checkpoints, strict load, no fallback, stock evaluator when available, matching
episode/eval caps, and recorded seed panels.

### 8. Reward And Survival Metrics

Footgun: survival can be a training reward, an eval metric, both, or neither.

Stock Pong exact mode rejects survival reward shaping by default and labels it
as a separate shaped-objective ablation when enabled
(`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:956`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:1777`).
The reconciliation note also says normal positive runs did not use survival
reward shaping (`docs/working/lightzero_signal_reconciliation_2026-05-10.md:77`).

Dummy Pong uses the raw ego Pong reward from the environment step
(`src/curvyzero/training/lightzero_dummy_pong_env.py:210`), while earlier docs
warned that survival/loss-delay telemetry was not necessarily being trained
(`docs/working/lightzero_pong_broad_bug_hunt_2026-05-09.md:85`).

CurvyTron native visual survival is explicitly survival-only with no terminal
bonus or loser penalty
(`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:885`,
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:886`).
Two-seat custom reward is per-player alive-after-step reward
(`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:1208`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2279`).

Guardrail: each artifact should name the training reward schema and the eval
metric schema. Never compare a shaped survival run to an unshaped score run as
if they trained the same objective.

### 9. Seeding And Random Starts

Footgun: a fixed reset seed or one eval seed can erase or invent a curve.

The live eval queue defaults to multi-seed panels for Pong curves
(`scripts/lightzero_live_eval_queue.py:24`,
`scripts/lightzero_live_eval_queue.py:507`), and prints the sampler seed so the
panel can be replayed (`scripts/lightzero_live_eval_queue.py:617`,
`scripts/lightzero_live_eval_queue.py:618`).

Dummy Pong has a custom dynamic seed policy that can override
`env.seed(..., dynamic_seed=False)` depending on config
(`src/curvyzero/training/lightzero_dummy_pong_env.py:83`,
`src/curvyzero/training/lightzero_dummy_pong_env.py:151`,
`src/curvyzero/training/lightzero_dummy_pong_env.py:239`,
`src/curvyzero/training/lightzero_dummy_pong_env.py:302`).
The config smoke has explicit seed-policy checks because this was a known
footgun (`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:860`,
`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:876`).

CurvyTron no-learning notes warn old step counts are not comparable unless
`decision_ms`, `max_ticks`, eval path, opponent, and checkpoint id are recorded
(`docs/working/training/curvytron_no_learning_investigation_2026-05-11.md:17`,
`docs/working/training/curvytron_no_learning_investigation_2026-05-11.md:29`).

Guardrail: report train seed, eval seeds, reset profile, dynamic seed policy,
decision interval, max ticks, and same-run baseline.

### 10. Checkpoint Loading And Format

Footgun: "checkpoint loaded" is not enough. Which state dict key, surface, and
policy API loaded it?

Pong eval strict-loads the model state and fails the run if strict loading does
not pass (`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:1065`,
`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:1544`).

Two-seat custom checkpoints save local `model`, optional `target_model`, and
optional optimizer state, then copy `ckpt_best` and `latest`
(`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:1892`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:1904`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:1915`).
That can load mechanically, but it is not the same artifact lifecycle as
LightZero's learner saving through `train_muzero`
(`docs/working/training/custom_vs_native_lightzero_diff_2026-05-10.md:52`).

Guardrail: every eval row should include checkpoint ref, checkpoint SHA,
selected state key, strict load result, model surface, and whether fallback was
used.

### 11. Horizon, Budget, And Naming

Footgun: `max_env_step` can mean total training budget, episode horizon, or
scorecard horizon, depending on the wrapper.

Stock Pong exact mode passes `max_env_step` to `train_muzero`; it does not set
episode caps (`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:1578`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:1772`).

Dummy Pong used `resolve_pong_episode_max_steps`; if
`pong_episode_max_steps` is omitted, it falls back to `max_env_step`
(`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:269`,
`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:274`).
Docs called this out as a core no-signal footgun because scaling the training
budget silently changes the task
(`docs/working/lightzero_pong_broad_bug_hunt_2026-05-09.md:17`,
`docs/working/lightzero_pong_broad_bug_hunt_2026-05-09.md:46`).

Guardrail: separate names: `train_max_env_steps`, `episode_max_steps`, and
`eval_cap_steps`.

### 12. Support Scale And Target Range

Footgun: LightZero's categorical support must match the reward/value scale.

Dummy Pong added custom support-scale plumbing and validation because
LightZero v0.2.0 exposes one `model.support_scale` and coherent reward/value
ranges must agree (`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:187`,
`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:230`,
`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:240`,
`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:495`).

Guardrail: include reward range, value range, support scale, and compiled-model
support values in every custom config summary.

## Path Diff Summary

### Stock Installed Pong

Use this as the known-good reference:

- calls stock `train_muzero`;
- stock Atari env, stock collector, stock GameBuffer, stock learner;
- `[4,64,64]` with stock `frame_stack_num=4`;
- `update_per_collect=None`, `replay_ratio=0.25`;
- unshaped Pong reward by default;
- strict stock-only eval over seed panels;
- same-run checkpoint curves.

Refs:
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:915`,
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py:1578`,
`scripts/lightzero_live_eval_queue.py:648`,
`docs/working/lightzero_pong_replication_monitor_2026-05-11.md:79`.

### Dummy Pong Custom Env

Useful as plumbing, not stock-Pong proof:

- starts from CartPole MuZero config and replaces the env with custom dummy
  Pong (`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:418`);
- tiny defaults: collector env `1`, simulations `2`, update per collect `1`
  (`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:38`,
  `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:41`,
  `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:43`);
- custom feature modes, opponent policies, dynamic seeds, support-scale
  patches, and horizon fallback;
- trains raw ego reward, while survival was often telemetry.

Main footgun: it can call `train_muzero`, but the env/objective/config are far
from stock Atari Pong.

### CurvyTron Native Visual Survival

Custom-ish but closer to Pong than two-seat:

- calls `train_muzero`
  (`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:789`);
- custom visual survival env and survival-only reward;
- tiny defaults compared with stock Pong: `max_env_step=8192`,
  `max_train_iter=64`, collector env `1`, simulations `8`
  (`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:103`,
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:104`,
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:106`,
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:110`);
- explicitly says `debug_fidelity_only` and `learning_proof=False`
  (`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:882`,
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:883`).

Main footgun: because it calls `train_muzero`, it can look native, but reward,
env, scale, frame-stack semantics, and opponent/self-play claims are custom.

### CurvyTron Two-Seat Custom Trainer

Useful for mechanics, not a normal LightZero trainer:

- does not call `train_muzero`;
- one live policy object chooses both seats;
- local replay rows feed direct `learn_mode.forward`;
- can have valid metadata-backed survival returns, but it remains outside
  LightZero's collector/GameBuffer path;
- not joint-action MCTS.

Refs:
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:845`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:857`,
`docs/working/training/curvytron_two_seat_bug_audit_2026-05-10.md:38`,
`docs/working/training/curvytron_two_seat_bug_audit_2026-05-10.md:98`.

## Minimum Contract Before Believing A MuZero Curve

For any future run, write these into the summary before scaling:

- trainer entrypoint: stock `train_muzero` or custom actor/learner;
- env API: one scalar action, sequential commit, independent seat rows, or
  joint action;
- observation contract: shape, dtype, stack owner, player perspective;
- search contract: simulations, legal mask, `to_play`, root noise/eval mode,
  visit targets;
- replay contract: GameBuffer/GameSegment or custom rows, sampling scope,
  replay age, priorities, target builder;
- learner contract: BaseLearner train loop or direct `learn_mode.forward`,
  optimizer step count, target-network update count;
- reward contract: training reward schema and eval metric schema;
- seed contract: train seed, reset policy, eval seed panel, decision interval;
- checkpoint contract: file ref, SHA, state key, strict load, fallback status;
- eval contract: stock evaluator/manual evaluator, cap, opponent, same-run
  `iteration_0`, later checkpoints.

The optimizer docs say the full system is actor/search/replay/learner/checkpoint
publisher/evaluator, not just one local loop
(`docs/working/optimizer/full_training_loop_worldview_2026-05-11.md:18`,
`docs/working/optimizer/full_training_loop_worldview_2026-05-11.md:25`,
`docs/working/optimizer/distributed_muzero_architecture_research_2026-05-11.md:81`).
That is the mental model to keep.
