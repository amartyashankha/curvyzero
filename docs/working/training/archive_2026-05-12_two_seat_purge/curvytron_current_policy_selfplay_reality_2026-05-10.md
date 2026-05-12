# CurvyTron Current-Policy Self-Play Reality - 2026-05-10

## Current Truth Check

- Main CurvyTron LightZero `train_muzero`: not self-play. It is single-ego
  training with either a fixed-straight opponent or a frozen checkpoint
  opponent.
- CurvyTron local NumPy smoke: true bounded two-seat current-policy smoke. One
  mutable current policy controls both seats, records both seats, then updates
  that same object. It is not LightZero or MuZero.
- CurvyTron two-seat LightZero smoke: true bounded two-seat current-policy smoke
  with one live `MuZeroPolicy` object controlling both seats and optional real
  optimizer updates. It is not `train_muzero`, not the LightZero collector, and
  not a full training loop.
- Pong self-play replay/train scripts: true tiny two-seat staged self-play for
  dummy Pong, but not LightZero and not full online self-play.
- LightZero dummy Pong env: fixed-opponent/control lane. LightZero controls one
  ego paddle; the env supplies the other paddle from a scripted or frozen
  checkpoint policy.

## Brutal Answer

No: the current CurvyTron LightZero `train_muzero` lane does not already do
current-policy self-play for simultaneous-action CurvyTron without a custom
collector or a new two-seat trainer contract.

What exists now:

- `fixed_straight`: LightZero chooses one ego action. The env fills the
  opponent with action `1`.
- `frozen_lightzero_checkpoint`: LightZero chooses one ego action. The env
  fills the opponent by running one frozen checkpoint provider.
- The live learner policy is not used for the opponent seat.
- s100 player-aware fixed-opponent eval was flat over 64 random starts:
  iterations `0,128,256,384,512,520` produced mean steps roughly
  `178.5,164.6,170.2,160.9,160.4,171.9`. Player-aware observation alone did
  not make fixed-opponent training learn.

Do not call either mode self-play.

Latest status:

- Fixed-opponent CurvyTron controls are flat or weak. Keep them as controls
  only, not self-play evidence.
- The iterative two-seat current-policy run
  `curvytron-two-seat-iterative-b8-s7-4x32-u2-sim2` passed with 4
  collect/update rounds, 2048 replay rows, and 5 checkpoints.
- The same run shows no learning signal yet over 32 random starts:
  `iteration_0..4` mean steps were `181.28125`, `174.0625`, `170.71875`,
  `174.125`, and `170.71875`.
- Two current blockers before scaling more runs:
  `target_value` is immediate reward rather than discounted survival return,
  and learner batch sizing may be capped at 2 rows because the LightZero
  profile hard-sets `policy.batch_size=2` while `_learn_mode_batches` slices
  samples to that size.
- Still missing before "real full training loop": rerun after the discounted
  survival-return target fix and learner batch-size fix, then promote only if
  checkpoints improve against same-run `iteration_0` on random-start eval
  panels.

## Exact Reason

The current training entrypoint is `lzero.entry.train_muzero` through:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

The env contract used by that path is single-ego:

```text
reset(seed) -> observation dict for one ego row
step(action) -> one scalar ego action
```

CurvyTron needs simultaneous actions:

```text
step(joint_action[player_0, player_1])
```

The current adapter bridges that mismatch by letting LightZero choose
`player_0` and by filling `player_1` inside the env. That makes fixed or frozen
opponents possible, but it means the env does not receive:

- the live collector policy object;
- the live learner weights;
- a learner update event;
- both players' observation rows before stepping.

So the env cannot ask "the current policy" for the opponent action. Real
current-policy self-play needs one of these larger changes:

- a custom collector that runs the current policy for both seats before
  `env.step(joint_action)`;
- a real two-seat LightZero env/trainer surface that records both policy
  actions in replay;
- a separate actor/learner loop with explicit weight refresh.

Smallest honest implementation path:

1. Add a two-seat collector/adapter that sees both players' observation rows
   and legal masks before `env.step`.
2. Run the current live policy weights for both seats on each decision.
3. Build `joint_action[B,P]` outside the env and step the public multiplayer
   env with that joint action.
4. Record both seats' actions, legal masks, search metadata, and policy weight
   revision in replay.
5. Define and log the actor weight refresh cadence.

That is the point where the run can be called current-policy two-seat self-play.
Adding another opponent kind inside the existing single-ego env would not be
enough.

## Smallest Useful Bridge Now

Use staged frozen-checkpoint refresh.

Plain rule:

1. Train stage `N` as learner versus frozen checkpoint `N-1`.
2. Publish checkpoints during the run.
3. Pick a named checkpoint from stage `N`, for example `ckpt_best.pth.tar`.
4. Use that checkpoint as the frozen opponent for stage `N+1`.
5. Evaluate both fixed-baseline and matched-frozen-opponent curves.

This is not current-policy self-play. It is a checkpoint-pool bridge that can
produce useful training data now while preserving exact labels.

Helper implemented:

```text
src/curvyzero/training/curvytron_frozen_opponent_refresh_plan.py
```

Example plan command:

```bash
PYTHONPATH=src python3 -m curvyzero.training.curvytron_frozen_opponent_refresh_plan \
  --initial-opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar \
  --initial-snapshot-ref curvytron_visual_survival_s42_iteration_293 \
  --stage-count 2 \
  --seed-start 60 \
  --max-env-step 4096 \
  --max-train-iter 32 \
  --eval-selected-iterations 0,8,16,24,32
```

The helper prints JSON with:

- staged train commands using `--opponent-policy-kind frozen_lightzero_checkpoint`;
- fixed-baseline eval commands;
- matched-frozen-opponent eval commands;
- the next stage's checkpoint ref;
- `current_policy_self_play: false`;
- the exact blocker text.

If a staged frozen-opponent run is needed next, use the already-cleared short
waited train gate against the s42 `iteration_293` checkpoint:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 44 \
  --run-id curvytron-visual-survival-debug-lz-s44-frozen-s42-iter293-smoke2048 \
  --attempt-id train-gpu-l4t4-frozen-s42-iter293-smoke2048-s44-20260510 \
  --max-env-step 2048 \
  --max-train-iter 16 \
  --source-max-steps 256 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --n-episode 1 \
  --num-simulations 4 \
  --batch-size 8 \
  --save-ckpt-after-iter 1 \
  --opponent-policy-kind frozen_lightzero_checkpoint \
  --opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar \
  --checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar \
  --snapshot-ref curvytron_visual_survival_s42_iteration_293 \
  --wait-for-train
```

Then eval the new `0,8,16` checkpoints against fixed-baseline first. If that
curve is sane, run the matched frozen-opponent eval from the snapshot-opponent
handoff. This staged run is still not self-play.

## Worker E Current Artifact

Small local train smoke now exists:

```text
src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py
```

Run:

```bash
uv run python -m curvyzero.training.curvytron_current_policy_selfplay_smoke \
  --seed 0 \
  --batch-size 1 \
  --steps 4 \
  --learner-updates 1
```

What it proves:

- builds `float32[B,P,4,64,64]` source-state gray64 stacked observations;
- uses one shared mutable current-policy object to choose actions for both
  player seats;
- maps policy rows back to `joint_action[B,P]`;
- calls `VectorMultiplayerEnv.step(joint_action)`;
- records two-seat replay rows for both players;
- uses steps-survived reward: `1.0` while the player is alive after the step,
  else `0.0`;
- samples the local replay rows;
- runs a bounded learner update on the same shared policy object.
- in the iterative Modal smoke, repeats collect/update for 4 rounds and saves
  5 checkpoints.

Current learning read:

```text
run_id: curvytron-two-seat-iterative-b8-s7-4x32-u2-sim2
status: passed
replay_rows: 2048
eval_random_starts: 32
mean_steps_by_checkpoint:
  iteration_0: 181.28125
  iteration_1: 174.0625
  iteration_2: 170.71875
  iteration_3: 174.125
  iteration_4: 170.71875
learning_signal: none yet
suspected_cause: target_value is immediate reward, not discounted survival return
second_blocker: policy.batch_size=2 plus _learn_mode_batches slicing may limit learning to 2 replay rows
blocker_status: target fix and learner batch-size fix are both required before scaling
```

Observed smoke result:

```text
ok: true
observation_shape: [1, 2, 4, 64, 64]
joint_action_schema_id: curvyzero_external_joint_action_player_major/v0
steps_survived: 4
replay.row_count: 8
replay.sample.players: [0, 1]
learner.status: updated
learner.model_parameters_changed: true
```

Honest limits:

- no `train_muzero`;
- no LightZero collector;
- no LightZero MCTS;
- no distributed actor weight refresh;
- included learner is a tiny local numpy policy/value update, not MuZero;
- visuals are source-state gray64, not browser pixel fidelity.

Allowed label:

```text
bounded local current-policy two-seat train smoke
```

Do not call this full LightZero current-policy self-play training.

## Worker F Current Artifact

Tiny LightZero two-seat smoke now exists:

```text
src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py
```

What is implemented:

- builds the same LightZero `MuZeroPolicy` setup used by
  `_build_lightzero_policy` in
  `curvyzero_stacked_debug_visual_survival_profile.py`;
- reuses the two-seat visual stack and row mapping from
  `curvytron_current_policy_selfplay_smoke.py`;
- renders `obs[B,P,4,64,64]`;
- flattens active player rows;
- asks the same live LightZero policy object for both player seats with
  `MuZeroPolicy.eval_mode.forward`;
- rebuilds `joint_action[B,P]`;
- steps `VectorMultiplayerEnv`;
- writes replay rows with both players' `observation`, `action_mask`, `action`,
  `action_weights`, `root_value`, and steps-survived reward;
- samples replay rows and tries `policy.learn_mode.forward` behind the existing
  no-op optimizer-step guard.

Local verification:

```bash
python3 -m py_compile src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py
```

Result: pass.

```bash
uv run python -m curvyzero.training.curvytron_two_seat_lightzero_train_smoke \
  --seed 0 \
  --batch-size 1 \
  --steps 4 \
  --num-simulations 2 \
  --learner-updates 1 \
  --allow-missing-lightzero
```

Result: command exits cleanly, but local `lightzero_policy_status` is
`blocked`. LightZero is not installed in this local runtime, so the local run
does not reach policy inference, replay collection, or learner forward.

Smallest remaining blocker:

```text
Full LightZero training is still blocked on a real two-seat collector/trainer
contract: `train_muzero` still does not collect both current-policy seats,
record both seats through LightZero replay, or refresh distributed actor
weights.
```

Do not patch `CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv.step`
to call the current policy for the opponent. That env receives only one ego
action from `train_muzero`, so it still cannot see the live policy object or
the learner weight revision for both seats.

Modal verification with installed LightZero:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_two_seat_train_smoke \
  --seed 0 \
  --batch-size 1 \
  --steps 4 \
  --num-simulations 2 \
  --learner-updates 1 \
  --output summary
```

Result:

```text
Modal app: ap-jGJMCz977NA7xgWmhG62Vu
ok: true
lightzero_policy_status: ok
steps_survived: 4
problems: []
replay.row_count: 8
replay.sample.players: [0, 1]
replay.sample.observation_batch_shape: [8, 4, 64, 64]
replay.sample.next_observation_batch_shape: [8, 4, 64, 64]
replay.sample.policy_batch_shape: [8, 3]
replay.sample.reward_sum: 8.0
learner_forward.status: run
learner_forward.ok: true
learner_forward.api: MuZeroPolicy.learn_mode.forward
learner_forward.blocker: null
```

This proves the bounded smoke can run one installed LightZero `MuZeroPolicy`
object for both CurvyTron seats, step the public multiplayer env with
`joint_action[B,P]`, sample both-player replay rows, and call
`learn_mode.forward`. It still does not call `train_muzero` and is not full
LightZero current-policy self-play training.

## Worker G Real-Step Bounded Train Smoke

The two-seat helper now keeps the old no-op smoke path, and also has a real
optimizer path:

```text
--allow-optimizer-step
```

Verification:

```bash
uv run python -m py_compile \
  src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py \
  src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py
```

Result: pass.

Tiny Modal real-step command:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_two_seat_train_smoke \
  --seed 0 \
  --batch-size 1 \
  --steps 8 \
  --num-simulations 2 \
  --learner-updates 1 \
  --allow-optimizer-step \
  --run-id curvytron-two-seat-realtrain-smoke-s0-20260510 \
  --attempt-id realtrain-steps8-updates1-20260510 \
  --output summary
```

Result:

```text
Modal app: ap-dohm6MpfKlDoZkU3LH3Ixx
ok: true
mode: bounded_two_seat_lightzero_collect_replay_real_train_smoke
lightzero_policy_status: ok
steps_survived: 8
problems: []
replay.row_count: 16
replay.sample.players: [0, 1]
learner_forward.status: updated
learner_forward.ok: true
learner_forward.optimizer_step: allowed
learner_forward.model_hash_before: d944a3b73e3e14af
learner_forward.model_hash_after: ffc34a228971d6b1
learner_forward.model_parameters_changed: true
```

Checkpoint refs written to the `curvyzero-runs` Modal Volume:

```text
training/lightzero-curvytron-visual-survival/curvytron-two-seat-realtrain-smoke-s0-20260510/checkpoints/lightzero/iteration_0.pth.tar
training/lightzero-curvytron-visual-survival/curvytron-two-seat-realtrain-smoke-s0-20260510/checkpoints/lightzero/iteration_1.pth.tar
training/lightzero-curvytron-visual-survival/curvytron-two-seat-realtrain-smoke-s0-20260510/checkpoints/lightzero/latest.pth.tar
training/lightzero-curvytron-visual-survival/curvytron-two-seat-realtrain-smoke-s0-20260510/checkpoints/lightzero/ckpt_best.pth.tar
```

Next eval pattern:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --run-id curvytron-two-seat-realtrain-smoke-s0-20260510 \
  --attempt-id realtrain-steps8-updates1-20260510 \
  --checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-two-seat-realtrain-smoke-s0-20260510/checkpoints/lightzero/iteration_0.pth.tar \
  --checkpoint-refs training/lightzero-curvytron-visual-survival/curvytron-two-seat-realtrain-smoke-s0-20260510/checkpoints/lightzero/iteration_0.pth.tar,training/lightzero-curvytron-visual-survival/curvytron-two-seat-realtrain-smoke-s0-20260510/checkpoints/lightzero/latest.pth.tar \
  --output summary
```

Plain read: this is now a real bounded two-seat train smoke. It still is not
`train_muzero`, not the LightZero collector, and not distributed current-policy
self-play.

Batchfix update: the learner batch next-observation bug is fixed. The Modal
two-seat current-policy smoke now reaches real optimizer updates. A strict-load
eval of the fixed checkpoint pair returned the same `176.5` mean steps for
`iteration_0` and `iteration_2`, so this is a pipeline-health pass only. Do not
claim learning from it.

Current blocker: full-loop integration, not the tiny helper shape. The helper
can autoreset rows that need reset. Update, 2026-05-10: the bounded two-seat
LightZero smoke now has a smallest-clean iterative local loop:
collect current-policy rows for both seats, build/sample replay for that
iteration, run one or more learner updates on the same live policy object, then
checkpoint by outer iteration. Later collection in the same process therefore
uses the refreshed policy object for both seats.

This changes the local smoke boundary, not the full training claim. It is still
not `train_muzero`, not LightZero's collector, and not distributed actor weight
refresh. The remaining larger gap is a production two-seat collector/trainer
contract plus eval cadence that proves policy progress beyond this bounded
local loop.

## Claim Boundary

Update, 2026-05-11:

```text
The custom two-seat LightZero path is now a real bounded current-policy
self-play path:
- flatten live seats
- run one shared current policy/search call for the active seats
- rebuild joint actions for both players
- step the CurvyTron env once
- add replay rows
- update the same policy object

This is enough to run learning probes. It is still not stock LightZero
train_muzero, not a distributed actor/learner setup, and not a final CurvyTron
training architecture.
```

Allowed claim now:

```text
learner versus refreshed frozen checkpoint opponent
current-policy joint-action smoke
bounded two-seat installed-LightZero collect/replay/learn-forward smoke
bounded two-seat installed-LightZero real optimizer-step smoke
bounded two-seat iterative collect/replay/update/checkpoint smoke
bounded two-seat current-policy self-play training probe
```

Forbidden claim:

```text
stock LightZero train_muzero current-policy self-play
distributed learner/actor weight refresh is implemented
final CurvyTron training architecture is settled
```
