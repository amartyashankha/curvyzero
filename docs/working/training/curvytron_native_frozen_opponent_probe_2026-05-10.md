# CurvyTron Native Frozen Opponent Probe - 2026-05-10

Question: can native LightZero `train_muzero` run CurvyTron single-ego training
against an env-owned frozen checkpoint opponent soon?

## Answer

Yes, mechanically. The shortest path is already present: use the native
CurvyTron visual survival trainer with
`opponent_policy_kind=frozen_lightzero_checkpoint`.

This is not current-policy two-seat self-play. It is:

```text
LightZero train_muzero learner controls player_0.
CurvyTron env owns player_1.
player_1 action comes from a frozen LightZero checkpoint provider.
training_relation: learner_vs_frozen_lightzero_checkpoint
```

## Existing Wiring

Native trainer entrypoint:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

It imports and calls:

```text
lzero.entry.train_muzero(...)
```

Registered env:

```text
env.type: curvyzero_stacked_debug_visual_survival_lightzero
env.import_names:
  curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env
env_id: CurvyZeroStackedDebugVisualSurvivalLightZero-v0
observation: [4,64,64]
action_space_size: 3
```

Frozen opponent code:

```text
src/curvyzero/training/lightzero_checkpoint_opponent_provider.py
  LightZeroCheckpointOpponentProvider
  snapshot_backed_lightzero_checkpoint_opponent_policy(...)

src/curvyzero/training/multiplayer_opponent_policy.py
  SnapshotBackedOpponentPolicy

src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_smoke.py
  opponent_policy_kind=frozen_lightzero_checkpoint
  _build_frozen_lightzero_opponent_policy(...)
  _opponent_action(...) calls the snapshot-backed provider
```

The Modal trainer already threads these config fields into the env:

```text
opponent_checkpoint_path
opponent_checkpoint_ref
opponent_snapshot_ref
opponent_checkpoint_state_key
opponent_policy_seed
opponent_num_simulations
opponent_batch_size
opponent_use_cuda
```

## Evidence In Docs

Existing docs say this path has run before:

```text
docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md
```

Recorded runs `s44` and `s45` used native `train_muzero` against frozen s42
`iteration_293`. They completed, published checkpoints, and telemetry showed
non-fixed frozen opponent action rows.

The learning signal was weak under the then-current fixed-straight eval
harness. Treat those runs as plumbing proof, not policy-progress proof.

## Blockers

No core implementation blocker for a near-term native run.

Exact blockers before calling it a useful training result:

```text
1. Compatible checkpoint selection.
   Need a real CurvyTron visual survival LightZero checkpoint ref in the
   mounted runs volume, with model shape [4,64,64] and action space 3.

2. Strict checkpoint load must pass.
   Provider uses strict model state loading. If payload keys differ, pass
   --state-key / opponent_checkpoint_state_key or inspect the checkpoint.

3. Eval must match the claim.
   Prior curves evaluated learner checkpoints mostly against fixed-straight,
   not necessarily against the same frozen checkpoint opponent. A useful result
   needs explicit frozen-opponent eval or clearly labeled fixed-opponent eval.

4. This does not solve current-policy self-play.
   Stock train_muzero gives env.step one ego action. The env cannot query the
   live collector policy for player_1 without a collector/two-seat action
   contract change.
```

## Shortest Path

Run a waited native profile first:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode profile \
  --compute cpu \
  --wait-for-train \
  --seed 0 \
  --run-id curvytron-native-frozen-opponent-profile-s0 \
  --attempt-id profile-native-frozen-opponent-20260510 \
  --max-env-step 64 \
  --max-train-iter 4 \
  --source-max-steps 32 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --n-episode 1 \
  --num-simulations 2 \
  --batch-size 4 \
  --save-ckpt-after-iter 1 \
  --stop-after-learner-train-calls 1 \
  --opponent-policy-kind frozen_lightzero_checkpoint \
  --opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/<run>/checkpoints/lightzero/<checkpoint>.pth.tar \
  --snapshot-ref curvytron_native_frozen_opponent_probe
```

Pass criteria:

```text
ok: true
called_train_muzero: true
problems: []
surface.opponent_policy_kind: frozen_lightzero_checkpoint
surface.opponent_training_relation: learner_vs_frozen_lightzero_checkpoint
action_observability.row_count > 0
opponent checkpoint metadata populated in env-step telemetry
phase_profile.counts.learner_train_calls >= 1
```

Then run a short waited train with the same flags, save frequent checkpoints,
and evaluate with labels that state whether the eval opponent is fixed-straight
or the same frozen checkpoint.

## Feasibility Call

Feasible: yes.

Main blocker: no missing code path; the practical blocker is proving a
compatible frozen checkpoint loads strictly and then evaluating against the
right opponent so the result is not mislabeled.

No pytest run; this was code/docs inspection only.
