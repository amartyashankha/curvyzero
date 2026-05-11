# Coach North Star - 2026-05-10

This is the short memory page. Read this before launching or reporting new
training work.

## North Star

The real project goal is CurvyTron training.

Pong is a control path. We use it to prove that the Modal + LightZero + eval
stack can learn from visual input and produce useful checkpoints. Pong is not
the final target.

Keep the proof lanes separate: stock visual Pong survival-time eval remains the
control lane. CurvyTron scalar/ray work is not a substitute for that proof.

ALE means Arcade Learning Environment: the Atari emulator/API used for Atari
ROMs. Official/control Pong is the only current ALE path because it is real
Atari ROM Pong. CurvyTron should not use ALE. A later CurvyTron visual wrapper
means CurvyTron pixels shaped for LightZero, not Atari emulation.

## Main Signal

Survival time is the first signal.

For Pong reports:

- Lead with stock steps survived versus same-run `iteration_0`.
- Then report stock episode length, stock return, and stock reward counts.
- A `-21` to `-20` return change is weak. It is not the main proof.
- The Pong eval signal leads with survival steps. Return/score can annotate the
  table, but it should not lead the claim.
- The current Pong signal is whether stock evaluator steps survived improve.
  Score/return comes after that, not first.
- The useful question is: does the policy survive longer, and does that persist
  at later checkpoints?
- The gate is sustained survival improvement over training checkpoints, not one
  lucky seed or one lucky eval episode.
- Avoid fixed-seed obsession. Use reproducible random eval panels: record the
  sampler seed and exact sampled seeds, then replay that panel only when needed.

For CurvyTron:

- Reward and eval signal are steps survived.
- The first reward should be survival time only.
- Reward the controlled player for staying alive longer.
- End the episode when the controlled player dies.
- Do not start with a separate win/score reward unless later game design needs
  it.

## Current Read

- Old normal Pong seed `1` showed strong late survival improvement.
- Old normal Pong seed `3` showed a later survival bump, then partial fallback.
- Current seed `13`, `18`, and `19` show survival bumps, but they are not
  stable yet.
- repeatB seed `1` is flat so far.
- Early Wave11 normal evals at `iteration_1000` for seeds `70`-`74` did not
  show a survival gain. Across eval seeds `100`-`115`, stock steps survived
  stayed around a mean of `760`, so this is not proof of learning.
- Wave11 final stock-only `rand16-a` artifacts for seeds `70`-`74` are complete:
  `16` eval seeds per checkpoint, serious `50`-simulation eval, and `2048` cap.
  Survival-first read versus same-run `iteration_0`: s70 is `+36` at
  `iteration_7000`; s71 is flat; s72 is `+59.625` at `iteration_7000`; s73 is
  `+173.125` at `iteration_7000`; s74 is flat. This means `3/5` runs show later
  survival lift and `2/5` are flat. The signal is encouraging, but not enough
  to declare stable proof. Score/return remains secondary.
- Wave11 late stock-only `rand16-b` artifacts are complete for s70, s73, and
  s76. Survival-first read versus same-run `iteration_0`: s70 is `+29.1875` at
  `iteration_7000` but returns to `+0` at `iteration_10000`; s73 strengthens
  from `+188.125` at `iteration_7000` to `+274.938` at `iteration_9000`; s76
  improves from `+87.625` at `iteration_7000` to `+159.562` at
  `iteration_12000`. This makes s73 the clearest current normal proof-lane
  signal, adds a positive long-run s76 signal, and keeps s70 marked unstable.
  Score/return remains secondary.
- Wave11 late stock-only `rand16-c` artifacts are complete for s73 and s76.
  Survival-first read versus same-run `iteration_0`: s73 is strong at
  `iteration_9000` (`+268.875`) but falls back near baseline at
  `iteration_11000` (`+3.75`); s76 improves again from `iteration_12000`
  (`+205.438`) to `iteration_15000` (`+363.562`). Current plain read: s76 is
  the cleanest improving curve; s73 proved it can improve but is not monotonic;
  s70 is unstable. Score/return remains secondary.
- Wave11 s76 late stock-only `rand16-d` is complete through `iteration_18000`.
  Survival-first read: `iteration_0` mean stock steps `760.562`,
  `iteration_15000` mean `1072.44` (`+311.875`), and `iteration_18000` mean
  `1093.19` (`+332.625`). This keeps s76 positive. It is still not a full
  stable-learning claim because the curve should be read across more
  checkpoints and more runs.
- Full-curve Pong evals launched for normal s70, s71, s72, s73, s74, and s76,
  plus shaped side runs s80, s81, and s82. They use stock-only serious eval,
  `2048` step cap, `50` search simulations per action, `8` random eval starts
  per checkpoint, and selected checkpoints spread across the visible trajectory.
  s75 only had `iteration_0`, so there was no useful curve to launch yet.
- Wave11 full-curve `rand8-e` artifacts are now read. Normal proof-lane
  survival-first read: s73 reaches mean stock steps `2048` at `iteration_18000`
  and latest is `1823.38`; s74 reaches `2048` at `iteration_30000`, `37000`,
  and latest `37542`; s76 peaks at `1905` at `iteration_40000` and latest is
  `1786.25`; s71 turns positive late with latest/best `1351.12`. s70 and s72
  remain unstable because they fall back near baseline. Plain read: the normal
  Pong proof lane now has multiple late stock-evaluator survival curves, not
  just one favorable run. Still keep the claim survival-first and do not use
  shaped runs as proof.
- Shaped Pong runs are side-lane telemetry only.

Plain read: Pong now has multiple strong late normal-run survival curves. Keep
checking robustness, but the stock visual Pong control lane is much healthier
than the early reads suggested.

CurvyTron waited runs s40/s41/s42 completed with live checkpoint publishing and
strict evals. The eval curves are flat/noisy, not a clean learning signal:
s40 mean steps start `199.00` and end `199.38`; s41 starts `206.25` and ends
`211.38`; s42 starts `199.00` and ends `194.88`. These runs are useful plumbing
proof only. They are still learned ego versus a fixed straight opponent, so do
not use them as evidence that multiplayer/self-play CurvyTron is solved.

Plain self-play status: the current control Pong lane is Atari single-agent
learning, not two learned agents training each other. The current CurvyTron
debug visual lane is one learned ego player against either a fixed straight
opponent or one frozen LightZero checkpoint opponent. True
current-policy-versus-current-policy CurvyTron training is still an open
design gap.

Update: `SnapshotBackedOpponentPolicy` now exists as the bounded bridge for
learner-vs-frozen-checkpoint CurvyTron opponents.
`LightZeroCheckpointOpponentProvider` also now exists as the matching frozen
checkpoint utility for the current CurvyTron debug visual survival LightZero
surface: conv MuZero, `[4,64,64]`, `A=3`. Blunt status: provider implemented,
minimal trainer flag wired. The current default visual trainer is still learned
ego vs fixed straight opponent, but it can now be configured for a frozen
LightZero checkpoint opponent. The s40/s41/s42 fixed-opponent curves are
flat/noisy, so do not spend more effort on long fixed-opponent launches as the
main path.

Real-checkpoint opponent smoke passed on Modal: s42 `iteration_293.pth.tar`
loaded from `curvyzero-runs`, the frozen LightZero provider chose opponent
action `1`, and the wrapper built joint action `[[0,1]]` with no validation
problems. Summary artifact:
`training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/attempts/snapshot-opponent-wrapper-smoke-s42-iteration293-20260510/eval/snapshot_opponent_wrapper_smoke/summary.json`.
This proves the checkpoint can drive an opponent through the two-player wrapper.
It still does not mean the trainer is wired.

Current-policy self-play reality check: not available in the current
simultaneous-action CurvyTron `train_muzero` lane without a custom collector or
two-seat trainer contract. The exact blocker is that LightZero calls this env
with one ego action, while CurvyTron needs a joint action for both players. The
small useful bridge is staged frozen-checkpoint refresh, documented in
`docs/working/training/curvytron_current_policy_selfplay_reality_2026-05-10.md`
and generated by
`src/curvyzero/training/curvytron_frozen_opponent_refresh_plan.py`.

Frozen s42 matched-opponent CurvyTron eval curves are now read for s44-s47
against the same frozen s42 `iteration_293` opponent family. Mean steps by
iteration:

| run | matched-opponent mean steps |
| --- | --- |
| s44 | `0=612.000`, `8=711.375`, `20=671.500`, `30=538.125`, `40=780.500`, `50=630.500`, `60=739.625`, `63=693.125` |
| s45 | `0=426.500`, `12=195.000`, `28=251.750`, `44=178.250`, `60=217.000`, `76=346.500`, `92=164.875`, `104=162.125` |
| s46 | `0=428.250`, `24=166.250`, `48=166.000`, `72=415.875`, `96=560.125`, `128=165.750`, `160=469.250`, `191=297.250` |
| s47 | `0=635.125`, `40=662.625`, `80=532.875`, `120=490.250`, `160=564.875`, `200=679.500`, `256=625.750`, `310=559.375` |

Plain CurvyTron read: matched-opponent survival is much higher than the
fixed-straight baseline, but unstable. s44 improves, s45 worsens, s46 has a
middle bump, and s47 is high/noisy. This is still learner-vs-frozen-checkpoint,
not live current-policy self-play.

Next CurvyTron runs launched from the best matched-opponent bumps:

- s90: learner vs frozen s44 `iteration_40`, `32768` env steps, wait app
  `ap-qp6LOTNqLH87vW2GFlg06x`.
- s91: learner vs frozen s46 `iteration_96`, `65536` env steps, wait app
  `ap-EkCmrjfsXfEuLBy5DATVRP`.
- s92: learner vs frozen s47 `iteration_200`, `65536` env steps, wait app
  `ap-X7cYjhPJd7y5zVZUqFhUMx`.

These are staged refresh runs, not self-play. When checkpoints arrive,
evaluate fixed-baseline and matched-frozen-opponent survival curves.
Earlier spawned function call ids did not leave visible artifacts when polled;
do not count them as live jobs.

Attached refresh runs completed:

- s90: ok, checkpoints through `iteration_175`, ego actions balanced.
- s91: ok, checkpoints through at least `iteration_328`, ego actions balanced.
- s92: ok, checkpoints through `iteration_434`, ego actions balanced.

Refresh eval curves are now read for all three, both fixed-baseline and
matched frozen-opponent. Survival steps are the main metric. Mean steps by
iteration:

| run | fixed-straight mean steps | matched frozen-opponent mean steps |
| --- | --- | --- |
| s90 | `0=167.000`, `24=123.250`, `48=167.750`, `72=122.250`, `96=116.125`, `128=120.500`, `160=125.500`, `175=113.875` | `0=659.500`, `24=245.750`, `48=456.500`, `72=289.750`, `96=187.125`, `128=269.000`, `160=268.250`, `175=253.500` |
| s91 | `0=132.125`, `64=108.125`, `128=132.250`, `192=162.625`, `256=125.000`, `320=169.750`, `328=167.875` | `0=181.000`, `64=314.750`, `128=333.375`, `192=365.250`, `256=232.125`, `320=213.625`, `328=464.750` |
| s92 | `0=132.125`, `64=167.000`, `128=114.750`, `192=158.875`, `256=167.000`, `320=167.000`, `384=140.375`, `434=162.000` | `0=503.125`, `64=358.750`, `128=331.375`, `192=491.125`, `256=565.500`, `320=579.375`, `384=589.000`, `434=541.750` |

Plain refresh read: s90 degraded. s91 fixed-baseline is mostly flat/noisy, but
matched-opponent survival has useful noisy improvement. s92 fixed-baseline is
flat, while matched-opponent survival is the best signal so far, especially
`iteration_256` through `iteration_384`. This is still narrow
learner-vs-frozen-checkpoint evidence, not true self-play. Next move: staged
refresh from s92 `iteration_384` or true two-seat/current-policy self-play
implementation, preferably both in parallel.

## What Must Pass Before Moving The Claim

Scalar/ray CurvyTron survival wrapper work can proceed as no-regret prep, but
it is a contract check only. The current contract target is repo-native
single-ego rows: `float32[106]` rays/scalars, `action_mask`, `to_play=-1`, one
ego action, named opponent policy, reward/done/info, reset/seed,
final-observation/reward metadata, and full joint-action logging. Do not call
an adapter smoke a full training loop.

The old "wait for at least two normal Pong survival curves before starting
visual CurvyTron plumbing" gate is now satisfied. This only unblocks CurvyTron
plumbing and signal-gathering, not a CurvyTron quality claim. Visual CurvyTron
uses the wrapper-owned debug stack `(4,64,64)` with LightZero/env-manager frame
stacking disabled (`frame_stack_num=1`) unless a separate schema changes that.

Optimizer correction: the next CurvyTron training blocker is not the scalar
wrapper. It is visual `[4,64,64]` stacking plus a bounded
collect/search/replay/sample/learner profile.

Do not claim the whole stack is solid until:

- at least two normal runs have later stock survival gains;
- the gains are visible in stock evaluator fields;
- the result does not depend on shaped reward;
- stock versus survival-shaped runs are compared without letting shaping hide
  broken learning;
- artifacts are fetched and summarized;
- run id, attempt id, checkpoint id, seed, eval seed, eval cap, and compute are
  recorded.

## Immediate Plan

1. Keep current Pong runs alive.
2. Evaluate later checkpoints beyond the completed Wave11 `7000`/late read; do
   not treat flat `1000` rows as the final read. For Wave11, treat s73 and s76
   as encouraging normal proof-lane signals, keep s70 marked unstable, and do
   not declare stable proof from one or two favorable curves.
3. Use `--group-size 4 --max-parallel-launches 64` for normal multi-seed eval
   waves. Use `--group-size 1` only when fastest first-checkpoint signal matters
   more than repeated Modal startup.
4. For robustness checks on selected checkpoints, sample a fresh pseudo-random
   eval seed set, record the generator seed and exact list, and pass it through
   `--eval-seeds` so each checkpoint gets multiple independent starts.
5. Use `gpu-l4-t4-cpu40` for eval; Modal rejected `cpu=64`, so 40 CPUs is the
   valid high-CPU function size here.
6. Summarize survival first.
7. Start no-regret scalar/ray CurvyTron contract checks in parallel, but do not
   present them as training proof.
8. Keep CurvyTron reward simple: survival time.
9. For self-play prep, use the frozen-checkpoint bridge before live current
   policy self-play. The no-train real-checkpoint wrapper smoke has passed; the
   next concrete run is the profile-sized frozen-opponent trainer command in
   "Frozen Opponent Bridge".
10. Do not launch more long fixed-straight-opponent CurvyTron runs unless the
    question is specifically about plumbing or speed. The next useful CurvyTron
    training step is profiling frozen-checkpoint opponents, then moving toward
    current-policy self-play.

## Eval Tooling Notes

Keep eval readouts survival-first and easy to scan.

- Pong and CurvyTron Modal eval summaries should lead with a compact TSV-style
  aggregate survival table, then per-seed rows, then the JSON summary. The
  first table is checkpoint -> seed count, ok count, mean/median/min/max steps
  survived, and mean score/return as a secondary field.
- Root manifests should record `output_refs`; manifest writer responses should
  echo `manifest_ref`.
- Per-checkpoint artifacts stay full-detail. Root manifests can stay slim for
  queue-scale runs.
- The Pong live eval queue should list the checkpoint directory first, apply
  selected-iteration filters, and skip the eval-root volume read when no
  checkpoints remain.
- Embarrassingly parallel default is safe for independent checkpoint/seed evals:
  use Modal `Function.starmap` inside each eval command and queue fan-out with
  `--group-size 4 --max-parallel-launches 64`.
- Default eval seed panels are now varied but reproducible: `8` sampled seeds
  from sampler seed `20260510`, unless the caller passes an explicit
  `--eval-seeds` list. Record the sampler seed and exact list with any claim.
- Do not optimize the story around one fixed seed. Prefer reproducible random
  panels and preserve the seed list for replay/debug.
- CurvyTron visual survival eval must pass the fixed-straight opponent defaults
  into `_build_visual_survival_configs`. s44/s45 eval jobs exposed this: they
  failed before env reset with missing `opponent_policy_kind`,
  `opponent_checkpoint`, `opponent_snapshot_ref`, and
  `opponent_checkpoint_state_key` arguments. The eval harness fix is to keep
  the fixed-opponent eval surface explicit, not to infer self-play.
- CurvyTron visual survival eval now has a `gpu-l4-t4-cpu40` option, matching
  the Pong high-CPU eval pattern. Use it for broad checkpoint/seed curves. The
  `--summary-only` output should print the survival aggregate table and
  per-seed table, then stop; it should not dump a large JSON blob.
- Exact blocker for more parallelism: duplicate filtering still needs an
  eval-root listing unless the caller intentionally passes
  `--skip-eval-root-listing` for a known-new eval id.

## Frozen Opponent Bridge

Current status:

- Implemented provider: yes.
- Modal/runtime smoke for a real volume checkpoint: passed.
- Minimal trainer flag: added, not yet profile-run.
- Design only: no, the provider utility and trainer flag are real.
- Remaining limitation: the trainer flag is a single-ego source-opponent hook,
  not full `MetadataOnlyMultiplayerEgoWrapper` ownership.

Exact result from the local no-checkpoint wrapper smoke:

```text
ok: true
reset_observation_shape: [1, 2, 4, 64, 64]
opponent_provider_observation_shape: [4, 64, 64]
joint_action: [[0, 1]]
provider_kind: fake_visual_checkpoint
```

This proves the wrapper path and row shape locally. It does not load a real
LightZero checkpoint.

Exact result from the real-checkpoint Modal smoke:

```text
ok: true
provider_kind: lightzero_checkpoint
checkpoint: s42 iteration_293.pth.tar
opponent action: 1
joint_action: [[0, 1]]
validation_problems: []
summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/attempts/snapshot-opponent-wrapper-smoke-s42-iteration293-20260510/eval/snapshot_opponent_wrapper_smoke/summary.json
```

Exact next Modal command:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode profile \
  --compute cpu \
  --seed 43 \
  --run-id curvytron-visual-survival-debug-lz-frozen-s42-iter293-profile \
  --attempt-id profile-frozen-opponent-s42-iter293-20260510 \
  --max-env-step 64 \
  --max-train-iter 2 \
  --source-max-steps 128 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --n-episode 1 \
  --num-simulations 2 \
  --batch-size 4 \
  --save-ckpt-after-iter 1 \
  --stop-after-learner-train-calls 1 \
  --opponent-policy-kind frozen_lightzero_checkpoint \
  --opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar \
  --checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar \
  --snapshot-ref curvytron_visual_survival_s42_iteration_293
```

After that passes:

1. Confirm env telemetry includes `opponent_policy_kind=frozen_lightzero_checkpoint`
   and `opponent_policy_sidecar`.
2. If the profile collects and reaches the learner stop, then launch a small
   bounded train run. Do not return to long fixed-opponent runs.

## CurvyTron Visual Survival Urgent Wave

Do not duplicate the already-running CurvyTron debug visual survival runs:

```text
s10/s11: 32768x128
s12: 65536x256
```

Additional launches recorded for the credit-constrained Hail Mary wave:

```text
s14 variant:
  run_id: curvytron-visual-survival-debug-lz-s14-sim16-32k
  attempt_id: train-gpu-l4t4-survival-debug-32768x128-s14-sim16-20260510
  Modal app: ap-nYLT88ZGlSxGNxo8CywJKA
  function_call_id: fc-01KR95MRWF547QDNDZAC4XWMYG
  config: 32768 max_env_step, 128 max_train_iter, source_max_steps 1024,
    num_simulations 16, batch_size 32, save_ckpt_after_iter 4

s13 Hail Mary:
  run_id: curvytron-visual-survival-debug-lz-s13-hailmary131k
  attempt_id: train-gpu-l4t4-survival-debug-131072x512-s13-hailmary-20260510
  Modal app: ap-grD5fS18CXOzi5g0d6KrdA
  function_call_id: fc-01KR95MS06C0W8HNHMRSSZYRZY
  config: 131072 max_env_step, 512 max_train_iter, source_max_steps 2048,
    num_simulations 8, batch_size 32, save_ckpt_after_iter 8

s30 live-publish follow-up:
  run_id: curvytron-visual-survival-debug-lz-s30-livepublish-32768
  attempt_id: train-gpu-l4t4-survival-debug-livepublish-32768x128-s30-20260510
  Modal app: ap-o06ImkzrSwI8SGU1hsrvw9
  function_call_id: fc-01KR9MZBQ4FN83T39QFE3R7N3Y
  config: 32768 max_env_step, 128 max_train_iter, source_max_steps 1024,
    collector_env_num 4, num_simulations 8, batch_size 32,
    save_ckpt_after_iter 4

Corrected Modal rule:
  Ephemeral `modal run` plus `.spawn` is not durable for long training here:
  s30/s31/s32 function calls terminated when the local entrypoint app stopped.
  Use `--wait-for-train` for current long runs, or deploy a stable Modal app
  before relying on true background `.spawn`.

Waited replacement runs:
  s40:
    run_id: curvytron-visual-survival-debug-lz-s40-wait-livepublish-32768
    attempt_id: train-gpu-l4t4-survival-debug-wait-livepublish-32768x128-s40-20260510
    Modal app: ap-mt6JBIqU0Qnru3XEgIKO14
    config: 32768 max_env_step, 128 max_train_iter, source_max_steps 1024,
      collector_env_num 4, num_simulations 8, batch_size 32,
      save_ckpt_after_iter 4
  s41:
    run_id: curvytron-visual-survival-debug-lz-s41-wait-livepublish-sim16-32768
    attempt_id: train-gpu-l4t4-survival-debug-wait-livepublish-32768x128-s41-sim16-20260510
    Modal app: ap-IYivO65mexWBf30BtTrS5z
    config: 32768 max_env_step, 128 max_train_iter, source_max_steps 1024,
      collector_env_num 4, num_simulations 16, batch_size 32,
      save_ckpt_after_iter 4
  s42:
    run_id: curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536
    attempt_id: train-gpu-l4t4-survival-debug-wait-livepublish-65536x256-s42-20260510
    Modal app: ap-NrAlFxNrpBW5dhw5tM5IF2
    config: 65536 max_env_step, 256 max_train_iter, source_max_steps 1024,
      collector_env_num 4, num_simulations 8, batch_size 32,
      save_ckpt_after_iter 8
```

Read these as trainer plumbing and signal-gathering only. Before any quality
language, fetch the summaries and action telemetry; survival reward, `[4,64,64]`
input, checkpoints, env horizon, action collapse, and eval/summary paths are
the gates.

## Seed Policy

Seeds are a diagnostic and reproducibility tool, not the goal. Use them to make
runs repeatable, to catch sensitivity, and to avoid confusing two launches with
the same start.

Training should usually use varied/random starts. Repeating the same seed should
intentionally start the same way. Do not launch "different" runs without
changing `--seed`, `run_id`, and `attempt_id`.

Each eval wave should use a fresh pseudo-random eval seed set, with the
generator seed and exact list recorded for replay. Do not promote a result
because one seed or one small reused seed list looks good.

Fixed eval seed lists are for replay and debugging only. New claim-seeking eval
waves should not reuse a standing panel.

TODO: run diverse training configs, but judge them by survival trend over
checkpoints, not by one lucky seed.

## Eval Status Update - 2026-05-10 20:45 EDT

Mean steps survived is the lead metric. These rows use checkpoint curves, not a
single lucky checkpoint.

| lane | run | eval | starts | baseline | best | latest | read |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Pong normal | s70 | rand8-e stock | 8 | 761.750 | 1044.500 @ 13000 | 761.750 @ 17582 | transient, flat latest |
| Pong normal | s71 | rand8-e stock | 8 | 759.625 | 1351.125 @ 18287 | 1351.125 @ 18287 | late positive |
| Pong normal | s72 | rand8-e stock | 8 | 759.500 | 816.500 @ 7000 | 759.500 @ 8000 | flat latest |
| Pong normal | s73 | rand8-e stock | 8 | 761.000 | 2048.000 @ 18000 | 1823.375 @ 18098 | real signal |
| Pong normal | s74 | rand8-e stock | 8 | 761.625 | 2048.000 @ 37542 | 2048.000 @ 37542 | real signal, cap |
| Pong normal | s76 | rand8-e stock | 8 | 761.125 | 1905.000 @ 40000 | 1786.250 @ 53704 | real signal |
| CurvyTron old anonymous | s93 | fixed-straight | 16 | 143.688 | 148.500 @ 256 | 131.375 @ 584 | flat/down |
| CurvyTron old anonymous | s93 | matched frozen s92 iter384 | 16 | 306.750 | 309.375 @ 128 | 173.625 @ 584 | decays |
| CurvyTron player-aware | s100 | fixed-straight | 64 | 178.531 | 178.531 @ 0 | 171.906 @ 520 | flat/down |
| CurvyTron player-aware | s101 | fixed-straight | 64 | 154.812 | 171.375 @ 1071 | 171.375 @ 1071 | small lift, weak |
| CurvyTron player-aware sim8 | s102 | fixed-straight | 64 | 178.531 | 178.625 @ 384 | 174.719 @ 540 | flat |

Plain read:

- Pong has real normal-env signal in s73, s74, and s76. s71 is also positive.
- Pong s70 and s72 are transient or flat by the latest checkpoint.
- CurvyTron s93 did not rescue the old anonymous lane.
- CurvyTron s100 proves the player-aware observation works as plumbing, but
  fixed-straight training is still flat.
- CurvyTron s101 has a small late lift, but it is still a fixed-opponent control
  and not a clean learning result.
- CurvyTron s102 with more search per move is flat.

Eval status:

- No more Pong eval is needed for this table.
- CurvyTron s93 fixed, s93 matched, s100 fixed64, s101 fixed64, and s102
  fixed64 have combined manifests and are table-ready.
- The next useful CurvyTron eval is not another fixed-straight proof run. The
  useful missing panel is held-out/frozen-opponent coverage for candidate
  checkpoints and then true two-seat current-policy self-play once available.

## Stop Doing

- Do not report return before survival.
- Do not treat `-20` instead of `-21` as the main result.
- Do not mix shaped Pong with normal Pong proof.
- Do not mix custom dummy Pong with the control path.
- Do not wait on one slow eval when more checkpoints can be evaluated in
  parallel.
- Do not wait on one eval seed when robustness needs multiple starts; use
  checkpoint/seed fan-out.
- Do not leave new findings only in chat.
