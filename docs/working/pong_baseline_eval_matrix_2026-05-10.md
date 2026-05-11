# Pong Baseline Eval Matrix - 2026-05-10

## Claim

Pong checkpoint evaluation should compare checkpoints against all simple
baselines that exist for the same environment family, not only against adjacent
checkpoints from the same run.

For this repo, there are two separate Pong lanes:

- **Official/control LightZero Atari Pong**: ALE-backed `PongNoFrameskip-v4`,
  visual observations, 6 Atari actions, LightZero stock Atari config/evaluator
  machinery.
- **Custom dummy Pong**: project-owned `dummy_pong_v0`, 2-player paddle task,
  3 actions (`up`, `stay`, `down`), tabular/raster features, scripted opponent
  policies.

Do not mix baselines across these lanes. A dummy `track_ball` baseline is not an
Atari Pong baseline, and a LightZero Atari random/no-op policy is not a dummy
Pong opponent.

## Non-claim

This does not claim that current checkpoints are good, that custom dummy Pong is
evidence for official Atari Pong quality, or that same-run checkpoint deltas are
enough. Same-run deltas are useful for regression detection, but they are not a
baseline eval matrix.

## Existing Support

### Custom Dummy Pong

Existing simple baselines:

| Baseline | Applies to | Existing support |
| --- | --- | --- |
| `random_uniform` | custom dummy Pong only | Implemented in `src/curvyzero/training/dummy_pong_eval.py`; uniform random over `up/stay/down`. |
| `lagged_track_ball_1` | custom dummy Pong only | Implemented in `dummy_pong_eval.py` and available as a LightZero dummy opponent in `lightzero_dummy_pong_env.py`; tracks the previous ball row. |
| `track_ball` | custom dummy Pong only | Implemented in `dummy_pong_eval.py` and available as a LightZero dummy opponent; tracks the current ball row. |
| `stay` | custom dummy Pong only | Implemented as an extended eval baseline and available via the LightZero MCTS checkpoint scoreboard's `--baseline-policy stay`; not included in the default baseline set. |
| learned `.npz` checkpoints | custom dummy Pong only | `scripts/run_dummy_pong_checkpoint_scoreboard.py` wraps `run_dummy_pong_eval()` and compares checkpoint(s) against the default baseline set plus checkpoint peers. |
| LightZero dummy checkpoints | custom dummy Pong only | `scripts/run_dummy_pong_lightzero_checkpoint_scoreboard.py` and `scripts/run_dummy_pong_lightzero_mcts_checkpoint_scoreboard.py` load LightZero dummy checkpoints and evaluate against dummy baselines. |
| frozen checkpoint opponents | custom dummy Pong LightZero wrapper | `DummyPongLightZeroEnv` supports `lightzero_policy_head_checkpoint` and `lightzero_mcts_checkpoint` opponent policies. |

The default custom dummy eval matrix is already broad: `run_dummy_pong_eval()`
builds baseline-vs-baseline rows, checkpoint-vs-baseline rows, and
checkpoint-vs-checkpoint rows. It also supports paired seating by default.

Tiny command run during this scout:

```bash
uv run python scripts/run_dummy_pong_eval.py --episodes 1 --seed 0 --output-dir /private/tmp/curvy-pong-baseline-eval-smoke-20260510
```

Result: passed. It wrote `summary.json` and `episodes.jsonl`, and included
default baseline rows for `random_uniform`, `lagged_track_ball_1`, and
`track_ball`.

### Official/Control LightZero Atari Pong

Existing support:

| Capability | Applies to | Existing support |
| --- | --- | --- |
| Stock LightZero/ALE env | official Atari Pong only | `src/curvyzero/infra/modal/lightzero_pong_env_smoke.py` and `lightzero_pong_tiny_train_smoke.py` use `PongNoFrameskip-v4`. |
| Training checkpoints | official Atari Pong only | `lightzero_pong_tiny_train_smoke.py` writes LightZero checkpoints under `training/lightzero-official-visual-pong/...`. |
| Manual checkpoint eval | official Atari Pong only | `lightzero_pong_eval_smoke.py` loads a checkpoint and steps the real ALE-backed LightZero env. |
| Simple policy baseline scorecard | official Atari Pong only | `lightzero_pong_simple_baseline_scorecard.py` runs the same ALE-backed LightZero env/config with random legal, no-op, and fixed action/action-meaning policies. It does not import dummy Pong policies. |
| Stock evaluator parity | official Atari Pong only | `lightzero_pong_eval_smoke.py --run-stock-evaluator` uses `lzero.worker.MuZeroEvaluator`; docs note the generic DI-engine evaluator is the wrong MuZero path because it does not pass the action mask. |
| Prior checkpoint curve | official Atari Pong only | `lightzero_pong_eval_smoke.py` accepts `--selected-iterations`, `--checkpoint-ref-template`, `--checkpoint-refs`, `--parallel`, and writes a manifest/table. |
| Public pretrained 96x96 probe/eval | official Atari Pong only | `lightzero_pong_pretrained96_checkpoint_probe.py` and `lightzero_pong_pretrained96_eval_smoke.py`; this is older 96x96/downsample compatibility work, not the current 64x64 tiny-run scorecard. |

Feasible right now for official/control Atari Pong:

- **Stock evaluator**: yes, via `lightzero_pong_eval_smoke.py
  --run-stock-evaluator`, preferably strict/no fallback and with
  `--max-episode-steps` equal to the eval cap.
- **Prior checkpoints**: yes, via selected iterations or explicit checkpoint refs.
- **Same-env random/no-op/simple action policies**: now supported by
  `src/curvyzero/infra/modal/lightzero_pong_simple_baseline_scorecard.py`. The
  scorecard discovers ALE action meanings, evaluates `random_legal`, `noop`,
  and every fixed Atari action, and records return, steps survived / episode
  length, action histograms, and action meanings.
- **Track-ball**: not found for official Atari Pong. Existing `track_ball` uses
  privileged dummy Pong state and 3 dummy actions. It should not be used against
  `PongNoFrameskip-v4`.

## Missing Support

### Custom Dummy Pong

Missing or partial:

- `stay` is present but not part of the default custom baseline set. It can be
  requested for LightZero MCTS checkpoint scoreboards with `--baseline-policy
  stay`, but the plain checkpoint scoreboard defaults to `random_uniform`,
  `lagged_track_ball_1`, and `track_ball`.
- `AngleControlPolicy` exists in `dummy_pong_eval.py`, but `_make_policy()` does
  not expose it in `EXTENDED_POLICY_NAMES`; angle-control is currently a
  diagnostic/probe path, not a scoreboard baseline.
- No explicit `noop`/`no-op` name exists for dummy Pong. The semantic equivalent
  is `stay`.

### Official/Control LightZero Atari Pong

Missing or partial:

- Official simple baseline rows now exist, but they are baseline-only return
  distributions in the official ALE env. There is still no unified manifest that
  joins checkpoint rows and simple baseline rows into one table.
- No official Atari `track_ball` policy was found.
- No checkpoint-vs-baseline table for official Atari analogous to the dummy
  Pong `scoreboard_rows`.
- No unified manifest that puts `iteration_0`, later checkpoints, random/no-op
  returns, and stock evaluator returns in one official-only table.

## Minimal Next Commands

### Fastest Custom Dummy Pong Path

Use the existing MCTS checkpoint scoreboard and keep baseline rows in the same
custom env:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt \
  --checkpoints lightzero:iterN=ref:training/lightzero-dummy-pong/<RUN_ID>/checkpoints/lightzero/iteration_<N>.pth.tar \
  --episodes 32 \
  --seed 1701 \
  --split-id dummy_pong_baseline_matrix_fixed_v0 \
  --eval-id mcts-scoreboard-baselines-iterN \
  --max-env-step 512 \
  --num-simulations 8 \
  --baseline-policy random_uniform \
  --baseline-policy lagged_track_ball_1 \
  --baseline-policy track_ball \
  --baseline-policy stay \
  --run-id <RUN_ID> \
  --attempt-id <ATTEMPT_ID>
```

If evaluating local `.npz` policies instead of LightZero checkpoints:

```bash
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --checkpoint latest=artifacts/local/<RUN>/checkpoint.npz \
  --episodes 64 \
  --seed 1701 \
  --output-dir artifacts/local/<RUN>/scoreboard-baselines \
  --split-id dummy_pong_baseline_matrix_fixed_v0 \
  --split-role fixed_eval
```

### Fastest Official Atari Pong Path

Run the official simple baseline scorecard first. This produces official-only
baseline rows in the same ALE-backed LightZero Atari Pong env/config and does
not import dummy Pong policies:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_simple_baseline_scorecard \
  --run-id <RUN_ID> \
  --attempt-id <ATTEMPT_ID> \
  --eval-id official-pong-simple-baselines \
  --episodes 1 \
  --max-eval-steps 512 \
  --max-episode-steps 512 \
  --step-detail-limit 2 \
  --policy-set random_legal,noop,fixed_actions
```

Tiny implementation smoke run:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_simple_baseline_scorecard \
  --episodes 1 \
  --max-eval-steps 2 \
  --max-episode-steps 2 \
  --step-detail-limit 0 \
  --policy-set random_legal,noop
```

Result: passed on CPU. It wrote
`training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/eval/official-pong-simple-baselines/lightzero_visual_pong_simple_baselines_episodes1_steps2_seed0_20260510T010406Z.json`
and produced rows for `random_legal` and `noop`. Both returned `0.0` over the
2-step capped smoke and reached the cap; action meanings were
`NOOP`, `FIRE`, `RIGHT`, `LEFT`, `RIGHTFIRE`, `LEFTFIRE`.

Useful 512-step baseline run:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_simple_baseline_scorecard \
  --episodes 8 \
  --max-eval-steps 512 \
  --max-episode-steps 512 \
  --step-detail-limit 4 \
  --policy-set random_legal,noop,fixed_actions \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id official-simple-baselines-episodes8-steps512-s0 \
  --eval-id official-pong-simple-baselines-episodes8-steps512-s0
```

Result artifact:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/official-simple-baselines-episodes8-steps512-s0/eval/official-pong-simple-baselines-episodes8-steps512-s0/lightzero_visual_pong_simple_baselines_episodes8_steps512_seed0_20260510T010511Z.json`.

Plain read:

| Baseline | steps survived mean | return mean | return std | nonzero rewards |
| --- | ---: | ---: | ---: | ---: |
| `random_legal` | 512.0 | -10.125 | 3.05931 | 91 |
| `noop` | 512.0 | -13.0 | 0.0 | 104 |
| `fixed_action_0_noop` | 512.0 | -13.0 | 0.0 | 104 |
| `fixed_action_1_fire` | 512.0 | -13.0 | 0.0 | 104 |
| `fixed_action_2_right` | 512.0 | -13.0 | 0.0 | 104 |
| `fixed_action_3_left` | 512.0 | -13.0 | 0.0 | 104 |
| `fixed_action_4_rightfire` | 512.0 | -13.0 | 0.0 | 104 |
| `fixed_action_5_leftfire` | 512.0 | -13.0 | 0.0 | 104 |

Claim: the official ALE baseline scorecard now gives a concrete short-cap
baseline table for checkpoint evals. Under the 512-step cap, random legal play
is much better than fixed/no-op policies on return, while every simple baseline
survives to the cap.

Non-claim: 512-step survival does not separate these baselines. For survival
time to become useful, run longer-cap or full-episode baseline and checkpoint
evals.

Then use the existing stock evaluator path to compare prior official
checkpoints. This gives official-only checkpoint deltas with LightZero's
evaluator:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
  --parallel \
  --run-stock-evaluator \
  --allow-model-fallback false \
  --eval-id official-pong-stock-evaluator-checkpoint-curve \
  --run-id <RUN_ID> \
  --attempt-id <ATTEMPT_ID> \
  --checkpoint-ref-template training/lightzero-official-visual-pong/<RUN_ID>/checkpoints/lightzero/iteration_{iteration}.pth.tar \
  --selected-iterations 0,1,2,4,8 \
  --max-eval-steps 512 \
  --max-episode-steps 512 \
  --step-detail-limit 2
```

## Minimal Code Changes

### Custom Dummy Pong

Small optional cleanup:

- Add `stay` to the default dummy checkpoint scoreboard baseline set if we want
  the no-op equivalent in every dummy scorecard.
- Either expose `angle_control` as an explicit extended baseline or keep it
  labeled as diagnostic-only. Do not silently include it in "simple baselines"
  without that decision.

### Official/Control LightZero Atari Pong

Implemented smallest useful addition:

- Added `src/curvyzero/infra/modal/lightzero_pong_simple_baseline_scorecard.py`.
  It runs `PongNoFrameskip-v4` under the same patched stock LightZero Atari env
  config and evaluates `random_legal`, `noop`, and fixed-action policies from
  discovered ALE action meanings.
- It reuses the official LightZero config/image/action-meaning helpers and does
  not import dummy Pong policies.
- It writes a Volume JSON summary with rows including `baseline_policy`,
  `return_mean`, `return_std`, `steps_survived_mean`, `episode_length_mean`,
  `episode_lengths`, `action_histogram`, `env_id`, and `action_meanings`.
- Remaining follow-up: extend the official eval manifest/table to include both:
  stock evaluator checkpoint returns and simple baseline returns.

Fastest official implementation path is to add baseline-only official rows
first. Checkpoint-vs-random in the same episode is not meaningful for Atari Pong
because Pong is single-agent control against the built-in Atari opponent; the
right comparison is score/return distribution in the same env and eval config.
