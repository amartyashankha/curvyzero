# CurvyTron Snapshot Opponent Interface - 2026-05-10

Status: implemented as a narrow code seam plus a frozen LightZero checkpoint
provider utility. It is wired at the `MetadataOnlyMultiplayerEgoWrapper` level.
A Modal/runtime smoke entrypoint now exists for a real checkpoint from the
`curvyzero-runs` volume. A minimal trainer flag also exists now:
`opponent_policy_kind=frozen_lightzero_checkpoint`. This is still not full
current-policy self-play.

## What Changed

- `src/curvyzero/training/multiplayer_opponent_policy.py` now has
  `SnapshotBackedOpponentPolicy`.
- The policy delegates each live opponent seat to a
  `SnapshotOpponentActionProvider`.
- The provider receives the opponent row observation, legal mask, env row,
  player id, deterministic action seed, `snapshot_ref`, and optional
  `checkpoint_ref`.
- The policy validates that provider actions are legal before they enter the
  wrapper joint action.
- The sidecar records `snapshot_ref`, `checkpoint_ref`, `model_id`,
  `provider_id`, and `provider_version`.
- `src/curvyzero/env/multiplayer_ego_wrapper.py` now passes the public
  observation tensor into opponent policies and copies opponent metadata into
  the wrapper action sidecar.
- `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py` now has
  `LightZeroCheckpointOpponentProvider` and
  `snapshot_backed_lightzero_checkpoint_opponent_policy(...)`.
- The same file now has
  `build_lightzero_checkpoint_multiplayer_ego_wrapper(...)`. This is the
  smallest wrapper-level wiring path for a frozen LightZero checkpoint opponent.
- `src/curvyzero/training/lightzero_checkpoint_opponent_wrapper_smoke.py` now
  has a no-train smoke. It proves the wrapper can pass `[B,P,4,64,64]`
  observations to a snapshot-backed opponent and record the opponent metadata.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  now has `--mode opponent-smoke`. This runs the wrapper smoke inside the
  LightZero Modal image, resolves a checkpoint ref from the Modal Volume, and
  writes `eval/snapshot_opponent_wrapper_smoke/summary.json`.
- The same trainer module now accepts
  `--opponent-policy-kind frozen_lightzero_checkpoint` plus
  `--opponent-checkpoint-ref`, `--checkpoint-ref`, and `--snapshot-ref`.
- `CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv` now has the small
  env hook that uses the frozen checkpoint provider to select the source
  opponent action from the current `[4,64,64]` stack.

Existing fixed-action and seeded-random opponents still work. They ignore the
new observation argument.

## Frozen LightZero Provider

Implemented provider status: usable as a small utility, wrapper-wired, and
narrowly trainer-wired through the single-ego stacked debug visual survival env
hook. It is not wired as a true two-seat trainer/replay contract.

The provider:

- lazily loads a LightZero MuZero checkpoint;
- matches the current CurvyTron debug visual survival model surface:
  conv MuZero, observation `[4,64,64]`, action space `3`;
- extracts a strict model state dict from common LightZero checkpoint payload
  keys;
- calls `policy.eval_mode.forward(...)` with the provided legal mask;
- returns `OpponentActionChoice`;
- computes `action_logp` when LightZero exposes a visit-count distribution;
- raises on illegal model actions by default, with an opt-in `first_legal`
  fallback for exploratory runs.

Minimal use:

```python
from curvyzero.training.lightzero_checkpoint_opponent_provider import (
    snapshot_backed_lightzero_checkpoint_opponent_policy,
)

opponent_policy = snapshot_backed_lightzero_checkpoint_opponent_policy(
    checkpoint_path="/runs/training/lightzero-curvytron-visual-survival/.../iteration_32.pth.tar",
    snapshot_ref="curvytron_visual_survival_s30_iteration_32",
    checkpoint_ref="training/lightzero-curvytron-visual-survival/.../iteration_32.pth.tar",
    seed=0,
    num_simulations=8,
    batch_size=16,
)
```

Pass that policy to `MetadataOnlyMultiplayerEgoWrapper` when the wrapped env
observations are real learned visual rows shaped `[B,P,4,64,64]`.

Wrapper helper:

```python
from curvyzero.training.lightzero_checkpoint_opponent_provider import (
    build_lightzero_checkpoint_multiplayer_ego_wrapper,
)

wrapper = build_lightzero_checkpoint_multiplayer_ego_wrapper(
    env,
    checkpoint_path="/runs/training/lightzero-curvytron-visual-survival/.../iteration_32.pth.tar",
    snapshot_ref="curvytron_visual_survival_s30_iteration_32",
    checkpoint_ref="training/lightzero-curvytron-visual-survival/.../iteration_32.pth.tar",
    ego_player_id=0,
)
```

Smoke command without LightZero dependencies:

```bash
PYTHONPATH=src python -m curvyzero.training.lightzero_checkpoint_opponent_wrapper_smoke
```

This uses a fake provider. It only proves wrapper wiring, visual row shape,
legal action placement, and sidecar metadata.

Smoke command with a real frozen LightZero checkpoint:

```bash
PYTHONPATH=src python -m curvyzero.training.lightzero_checkpoint_opponent_wrapper_smoke \
  --checkpoint-path /runs/training/lightzero-curvytron-visual-survival/<run>/checkpoints/lightzero/iteration_32.pth.tar \
  --checkpoint-ref training/lightzero-curvytron-visual-survival/<run>/checkpoints/lightzero/iteration_32.pth.tar \
  --snapshot-ref curvytron_visual_survival_<run>_iteration_32
```

Run the real-checkpoint command inside the same runtime that has LightZero,
DI-engine, torch, and the checkpoint file.

Modal/runtime smoke with a real checkpoint from `curvyzero-runs`:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode opponent-smoke \
  --run-id curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536 \
  --attempt-id snapshot-opponent-wrapper-smoke-s42-iteration293-20260510 \
  --opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar \
  --checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar \
  --snapshot-ref curvytron_visual_survival_s42_iteration_293 \
  --seed 42 \
  --num-simulations 8 \
  --batch-size 16
```

Exact Modal result from this command: pass. The smoke loaded the s42
`iteration_293.pth.tar` checkpoint from `curvyzero-runs`, called the frozen
LightZero opponent provider, selected opponent action `1`, and built joint
action `[[0,1]]`. Validation problems were empty. Checkpoint file summary:
96,189,939 bytes, sha256
`83abafcb6df8fc10d6056261fcba55f7bb82fa93bce7a1fc2bf65688a468c5d6`.

Summary artifact:
`training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/attempts/snapshot-opponent-wrapper-smoke-s42-iteration293-20260510/eval/snapshot_opponent_wrapper_smoke/summary.json`

Local verification done after adding the Modal entrypoint:

```text
python3 -m py_compile ...lightzero_curvyzero_stacked_debug_visual_survival_train.py ...lightzero_checkpoint_opponent_wrapper_smoke.py
PYTHONPATH=src python3 -m curvyzero.training.lightzero_checkpoint_opponent_wrapper_smoke
```

Exact local smoke result: `ok=true`, reset observation shape `[1,2,4,64,64]`,
provider row shape `[4,64,64]`, joint action `[[0,1]]`, opponent metadata
present. This local run used the fake provider; the Modal command above is the
real-checkpoint gate.

## Smallest Correct Frozen-Opponent Path

Use frozen/snapshot opponents before live current-policy opponents.

1. Keep LightZero controlling one ego seat.
2. Load a frozen checkpoint outside the env.
3. Wrap the loaded model in a `SnapshotOpponentActionProvider`.
4. Pass `SnapshotBackedOpponentPolicy(provider=..., snapshot_ref=..., checkpoint_ref=...)`
   into `MetadataOnlyMultiplayerEgoWrapper`.
5. Let the wrapper build the full `[B,P]` joint action.
6. Record the sidecar. Do not call this full current-policy self-play.

This is the bridge from learner-vs-scripted to learner-vs-frozen-checkpoint.
True current-policy-versus-current-policy needs a separate snapshot refresh
rule and actor/learner handoff.

## Smallest True Two-Seat Self-Play Path

The smallest honest current-policy path is not another env flag. It is a new
collector/adapter path that owns both seats before stepping:

1. Build a two-seat collector around the existing `[B,P,...]` public env or
   `MetadataOnlyMultiplayerEgoWrapper` action-map helpers.
2. On every environment decision, gather both player observation rows and legal
   masks.
3. Run the same live MuZero policy weights for both seats, or two explicit
   live policy handles if Coach wants asymmetric seats.
4. Construct `joint_action[B,P]` outside the env and call
   `env.step(joint_action)`.
5. Write both player actions, legal masks, policy version/weight revision, and
   search metadata into replay.
6. Refresh actor weights from the learner on a named cadence and record that
   cadence in the run summary.

Until that exists, `opponent_policy_kind=frozen_lightzero_checkpoint` remains
learner-vs-frozen-checkpoint only.

## Trainer Wiring Read

Decision: minimal trainer flag wired, but only as a narrow first bridge.

Reason: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
still targets `CurvyZeroStackedDebugVisualSurvivalLightZeroEnv`, a single-ego
LightZero env. The new flag changes only how that env chooses the source
opponent action. It does not yet route training through
`MetadataOnlyMultiplayerEgoWrapper`, and it does not make a full `[B,P]`
trainer replay contract.

Current-policy self-play is not a small change in this path.

What happens today:

- Default train: LightZero chooses the ego action. The env fills the opponent
  with fixed straight.
- Frozen-checkpoint train: LightZero chooses the ego action. The env fills the
  opponent by loading one frozen LightZero checkpoint and running inference on
  the opponent row.
- Neither mode gives the env the live collector policy object or the learner's
  changing weights.

Blocker: LightZero `train_muzero` calls this env with only the ego action in
`env.step(action)`. The collector policy and learner live outside the env. The
env cannot ask the current policy for the opponent action unless we change the
collector/env contract or move to a real two-seat wrapper path.

Small code fix made on 2026-05-10: train summaries, env info, and env-step
telemetry now write:

```text
opponent_training_relation: learner_vs_fixed_straight | learner_vs_frozen_lightzero_checkpoint
current_policy_self_play: false
current_policy_self_play_blocker: LightZero train_muzero calls this env with only the ego action...
```

This is a labeling and observability fix. It is not self-play.

Smallest true self-play design:

1. Use a two-seat env/action contract where the trainer can choose both player
   actions, or patch the collector to run the current policy for both seats
   before stepping.
2. Give each seat its own observation row and legal mask.
3. Record both policy actions in replay.
4. Define how often actor weights refresh from the learner.
5. Only then call the run current-policy self-play.

Exact next command: profile-sized trainer gate against the frozen s42
checkpoint. This should prove the trainer can collect through the frozen
opponent env without launching another long fixed-opponent run.

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

Profile gate result: passed.

```text
ok: true
called_train_muzero: true
env steps collected: 46
learner train calls: 1
row_count: 91
ego action histogram: {"0": 42, "1": 33, "2": 16}
opponent action histogram: {"0": 80, "1": 11, "2": 0}
mirrored checkpoint: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-frozen-s42-iter293-profile/checkpoints/lightzero/iteration_0.pth.tar
summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-frozen-s42-iter293-profile/attempts/profile-frozen-opponent-s42-iter293-20260510/train/summary.json
```

Plain read: the frozen-opponent trainer hook reached real `train_muzero`,
collected env rows with non-degenerate ego actions and non-fixed frozen
opponent actions, made one learner update, and published the first checkpoint.
This clears the gate for a short waited train.

After the short frozen-checkpoint train/eval read, the next real design step is
to replace this single-ego source-opponent hook with a two-seat env that uses
`MetadataOnlyMultiplayerEgoWrapper` as the joint-action owner.

## Next Short Train+Eval Plan

Gate status: profile passed. Short waited trains are now the active next
step. Do not launch them as background spawns; keep the Modal app alive with
`--wait-for-train`.

Purpose: a small waited learner-vs-frozen-checkpoint train against the s42
`iteration_293` opponent checkpoint, with frequent checkpoint publishing, then
a compact survival eval curve over a few newly produced checkpoints. This is
learner-vs-frozen-checkpoint, not self-play.

Short waited train command:

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

Expected train artifacts:

```text
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s44-frozen-s42-iter293-smoke2048/attempts/train-gpu-l4t4-frozen-s42-iter293-smoke2048-s44-20260510/train/summary.json
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s44-frozen-s42-iter293-smoke2048/attempts/train-gpu-l4t4-frozen-s42-iter293-smoke2048-s44-20260510/train/action_observability.json
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s44-frozen-s42-iter293-smoke2048/attempts/train-gpu-l4t4-frozen-s42-iter293-smoke2048-s44-20260510/train/live_checkpoint_publish.json
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s44-frozen-s42-iter293-smoke2048/checkpoints/lightzero/iteration_0.pth.tar
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s44-frozen-s42-iter293-smoke2048/checkpoints/lightzero/iteration_8.pth.tar
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s44-frozen-s42-iter293-smoke2048/checkpoints/lightzero/iteration_16.pth.tar
```

Expected summary fields to sanity-check before eval:

```text
ok: true
called_train_muzero: true
opponent_policy_kind: frozen_lightzero_checkpoint
opponent_checkpoint_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar
snapshot_ref/opponent_snapshot_ref: curvytron_visual_survival_s42_iteration_293
checkpoint_mirror.count: >0
problems: []
```

Small survival eval curve command:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id curvytron-visual-survival-debug-lz-s44-frozen-s42-iter293-smoke2048 \
  --attempt-id train-gpu-l4t4-frozen-s42-iter293-smoke2048-s44-20260510 \
  --eval-id checkpoint_curve_frozen_s42_iter293_smoke_20260510 \
  --selected-iterations 0,8,16 \
  --seed 546745683 \
  --eval-seeds 546745683,1247268015,823376496 \
  --max-eval-steps 256 \
  --source-max-steps 256 \
  --num-simulations 4 \
  --batch-size 8 \
  --parallel \
  --summary-only \
  --quiet-framework-logs
```

Expected eval artifacts:

```text
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s44-frozen-s42-iter293-smoke2048/attempts/train-gpu-l4t4-frozen-s42-iter293-smoke2048-s44-20260510/eval/checkpoint_curve_frozen_s42_iter293_smoke_20260510/manifest_steps256_seeds546745683-1247268015-823376496_<UTCSTAMP>.json
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s44-frozen-s42-iter293-smoke2048/attempts/train-gpu-l4t4-frozen-s42-iter293-smoke2048-s44-20260510/eval/checkpoint_curve_frozen_s42_iter293_smoke_20260510/iteration_0_steps256_seed546745683/curvytron_visual_survival_eval_iteration_0_steps256_seed546745683_<UTCSTAMP>.json
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s44-frozen-s42-iter293-smoke2048/attempts/train-gpu-l4t4-frozen-s42-iter293-smoke2048-s44-20260510/eval/checkpoint_curve_frozen_s42_iter293_smoke_20260510/iteration_8_steps256_seed546745683/curvytron_visual_survival_eval_iteration_8_steps256_seed546745683_<UTCSTAMP>.json
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s44-frozen-s42-iter293-smoke2048/attempts/train-gpu-l4t4-frozen-s42-iter293-smoke2048-s44-20260510/eval/checkpoint_curve_frozen_s42_iter293_smoke_20260510/iteration_16_steps256_seed546745683/curvytron_visual_survival_eval_iteration_16_steps256_seed546745683_<UTCSTAMP>.json
```

If `iteration_16.pth.tar` is not present after the train, keep the same eval
command shape but replace `--selected-iterations 0,8,16` with the visible
latest three checkpoints from:

```bash
uv run --extra modal modal volume ls curvyzero-runs training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s44-frozen-s42-iter293-smoke2048/checkpoints/lightzero
```

## Delegation Notes

Checkpoint-loader worker:

- Use `LightZeroCheckpointOpponentProvider`; do not put checkpoint loading
  inside the vector env.
- Provider input should be one opponent observation row plus `legal_action_mask`.
- Provider output should be `OpponentActionChoice(action_id=..., action_logp=...)`
  when policy logits are available, or an integer action id when only argmax is
  exposed.
- Keep checkpoint loading outside `MetadataOnlyMultiplayerEgoWrapper`.

Replay worker:

- Confirm that wrapper `action_sidecar["opponent_policy_metadata"]` is preserved
  in the replay path used for training artifacts.
- If row-level opponent sidecars need checkpoint refs directly, extend
  `src/curvyzero/training/multiplayer_replay_v0.py` deliberately instead of
  relying only on batch metadata.

Coach worker:

- Treat this as implemented plumbing for snapshot-backed opponents.
- Label default runs learner-vs-fixed-straight. Label runs with
  `opponent_policy_kind=frozen_lightzero_checkpoint` as
  learner-vs-frozen-checkpoint, not self-play.
- Do not touch official Atari Pong eval tooling for this task.

## Current-Policy Self-Play Reality Check

Short answer: no, this LightZero `train_muzero` lane does not already give
simultaneous-action CurvyTron current-policy self-play without a custom
collector or a two-seat trainer contract.

Reason: this lane calls the CurvyTron LightZero env with one scalar ego action.
The env fills the opponent internally. That allows `fixed_straight` and
`frozen_lightzero_checkpoint`, but the env does not receive the live collector
policy, current learner weights, learner update events, or both players'
observation rows before `env.step`.

Small staged bridge now available:

```text
src/curvyzero/training/curvytron_frozen_opponent_refresh_plan.py
```

This helper prints a JSON plan for repeated learner-vs-frozen-checkpoint
stages. It also emits fixed-baseline eval commands, matched-frozen-opponent
eval commands, next-stage checkpoint refs, `current_policy_self_play: false`,
and the exact blocker text. This is checkpoint-pool refresh, not self-play.

Details:

```text
docs/working/training/curvytron_current_policy_selfplay_reality_2026-05-10.md
```
