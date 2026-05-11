# 2026-05-09 Modal LightZero Pong Scale128 Control

## Question

What is the next smallest official Atari Pong training scale rung that can show
any real eval signal without becoming a giant run, and does it improve over the
tiny `iteration_1` all-action-0 checkpoint?

## Setup

This is official LightZero Atari Pong cold-start control only:

- env: `PongNoFrameskip-v4`
- source config: `zoo.atari.config.atari_muzero_config`
- entrypoint: `lzero.entry.train_muzero`
- wrapper: `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`
- eval wrapper: `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py`
- model: stock conv MuZero, observation shape `[4, 64, 64]`, action space `6`

The chosen rung stayed on CPU. The trainer wrapper currently declares
`cpu=1.0`, patches `policy.cuda=False`, and the clean GPU path would need a
Modal resource change plus a safe CUDA config switch. For this scouting rung,
that is larger than necessary. The smallest useful scale change was to keep
`num_simulations=2`, one collector/evaluator env, and raise only the hard caps
enough to pass the old 64-step truncation:

```text
max_env_step: 128
max_train_iter: 2
batch_size: 8
max_episode_steps: 128
game_segment_length: 16
num_simulations: 2
```

The wrapper validation caps were widened from smoke-only limits:

```text
max_env_step: 8 -> 128
max_train_iter: 1 -> 2
collect/eval max_episode_steps: 64 -> 128
```

No pytest was run.

## Commands

Compile check:

```sh
python -m py_compile src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py
```

Training rung:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --mode train --max-env-step 128 --max-train-iter 2 --batch-size 8 --max-episode-steps 128 --game-segment-length 16
```

New checkpoint eval:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T174318Z-0536e5b37ea7/checkpoints/lightzero/iteration_1.pth.tar --run-id lz-visual-pong-20260509T174318Z-0536e5b37ea7 --attempt-id attempt-20260509T174318Z-cd5c45363eb8 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T174318Z-0536e5b37ea7/attempts/attempt-20260509T174318Z-cd5c45363eb8/eval/iteration_1_scale128/lightzero_visual_pong_eval_scale128_20260509T174318Z.json --max-env-step 128 --max-train-iter 2 --batch-size 8 --max-episode-steps 128 --game-segment-length 16 --max-eval-steps 128 --step-detail-limit 8 --no-allow-model-fallback
```

Baseline old tiny checkpoint eval under the same 128-step cap:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/iteration_1.pth.tar --run-id lz-visual-pong-20260509T171834Z-1798cd6bef57 --attempt-id attempt-20260509T171834Z-fd4b5559bec6 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/eval/iteration_1_scale128_baseline/lightzero_visual_pong_eval_scale128_baseline_20260509T174600Z.json --max-env-step 128 --max-train-iter 2 --batch-size 8 --max-episode-steps 128 --game-segment-length 16 --max-eval-steps 128 --step-detail-limit 8 --no-allow-model-fallback
```

## Results

Training Modal app:

```text
ap-qoTln2RP7Ly65hjCK3V4On
ok: true
remote_elapsed_sec: 19.894407
train_result.elapsed_sec: 9.980145
return_type: MuZeroPolicy
training_iterations: [0]
final_rewards: [-1.0]
```

The requested `max_train_iter=2` did not produce `iteration_2`; with
`max_env_step=128`, the official trainer stopped after `Training Iteration 0`
and saved `iteration_0` plus `iteration_1`.

New checkpoint refs:

```text
training/lightzero-official-visual-pong/lz-visual-pong-20260509T174318Z-0536e5b37ea7/checkpoints/lightzero/ckpt_best.pth.tar
sha256: 078b718718223cf2cbfd4c2f9905575edca48d682b2830c692f0f5ecd3065027

training/lightzero-official-visual-pong/lz-visual-pong-20260509T174318Z-0536e5b37ea7/checkpoints/lightzero/iteration_0.pth.tar
sha256: 808a89739099f039b1785fa4d5f50f3090fb207a4a2f072c87842fae8f5b1120

training/lightzero-official-visual-pong/lz-visual-pong-20260509T174318Z-0536e5b37ea7/checkpoints/lightzero/iteration_1.pth.tar
sha256: 3ae3564e480e17972eb64c402ec899737305d537b5cce1f441baf2881bfb6420
```

New checkpoint eval Modal app:

```text
ap-xARflZIivWVe3TdtHD4vEL
ok: true
remote_elapsed_sec: 15.777995
steps_run: 128
total_reward: -2.0
policy_eval_step_count: 128
fallback_step_count: 0
action_histogram: {0: 128}
reward_histogram: {-1.0: 2, 0.0: 126}
nonzero rewards: step 60 -> -1.0, step 95 -> -1.0
terminal: step 127, TimeLimit.truncated true, eval_episode_return -2.0
```

New eval artifact:

```text
training/lightzero-official-visual-pong/lz-visual-pong-20260509T174318Z-0536e5b37ea7/attempts/attempt-20260509T174318Z-cd5c45363eb8/eval/iteration_1_scale128/lightzero_visual_pong_eval_scale128_20260509T174318Z.json
sha256: 5cf95cb963b34079bf4058f67766e4c00aae5692039e1c1e0eba7a5ccbe6c6ee
```

Old tiny checkpoint baseline eval Modal app:

```text
ap-D0OopWDZJD8K191krNFBBC
ok: true
remote_elapsed_sec: 23.741761
steps_run: 128
total_reward: -2.0
policy_eval_step_count: 128
fallback_step_count: 0
action_histogram: {0: 128}
reward_histogram: {-1.0: 2, 0.0: 126}
nonzero rewards: step 60 -> -1.0, step 95 -> -1.0
terminal: step 127, TimeLimit.truncated true, eval_episode_return -2.0
```

Old baseline artifact:

```text
training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/eval/iteration_1_scale128_baseline/lightzero_visual_pong_eval_scale128_baseline_20260509T174600Z.json
sha256: 5b1f9856cd53a15a5ffbc6fa3446b00221544b4b65689b3ef1f03fa9fa2016e9
```

Comparison:

| Checkpoint | Eval cap | Actions | Return | Nonzero rewards | Survived |
| --- | ---: | --- | ---: | --- | ---: |
| old tiny `iteration_1` | 128 | `{0:128}` | `-2.0` | `60:-1`, `95:-1` | 128 steps |
| scale128 `iteration_1` | 128 | `{0:128}` | `-2.0` | `60:-1`, `95:-1` | 128 steps |

## Interpretation

The rung produced real official Atari eval signal: the loaded policy acted
through `MuZeroPolicy.eval_mode.forward` for 128 ALE steps with no fallback,
crossed two losing point events, and reported true Atari rewards under the
LightZero wrapper.

It did not improve over the tiny checkpoint. Both old and new `iteration_1`
checkpoints are still all-action-0 under the same eval cap, with identical
return and reward timing. This is a useful cold-start control result, not a
policy-quality win.

CPU is acceptable for this rung because it stayed tiny and completed quickly.
GPU should wait until the wrapper explicitly supports a CUDA resource/config
switch, or until the next rung is large enough that CPU becomes the bottleneck.

## Follow-ups

Do not rerun this exact CPU rung for quality. If the next official Atari rung
is needed, make it explicit and still capped: either raise `max_env_step`
enough to actually produce a later checkpoint, or add a clean GPU option to the
wrapper before increasing MCTS/update cost. Keep this official Atari control
separate from project dummy Pong and CurvyTron reporting.
