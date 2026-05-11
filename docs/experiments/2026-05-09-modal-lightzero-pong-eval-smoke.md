# 2026-05-09 Modal LightZero Pong Eval Smoke

## Question

Can the mirrored stock visual Atari Pong `iteration_1.pth.tar` checkpoint act
through `MuZeroPolicy.eval_mode.forward` against real ALE reset observations
before any longer training run?

## Setup

Added a dedicated eval-only worker:

```text
src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py
```

It uses the same stock visual Pong surface as the tiny trainer/checkpoint probe:

```text
source module: zoo.atari.config.atari_muzero_config
env id: PongNoFrameskip-v4
env type: atari_lightzero
model type: conv
policy class: lzero.policy.muzero.MuZeroPolicy
observation shape: [4, 64, 64]
action space size: 6
num_simulations: 2
max_eval_steps: 8
```

The real LightZero Atari env exposes a single preprocessed frame at reset
(`env observation shape [1, 64, 64]`). The eval smoke keeps a tiny local
4-frame stack for the policy input (`policy observation shape [4, 64, 64]`),
matching the loaded conv MuZero checkpoint contract. Raw env observations and
policy-facing observations are reported separately.

## Commands

```sh
python -m py_compile src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py src/curvyzero/infra/modal/lightzero_pong_checkpoint_probe.py src/curvyzero/infra/modal/lightzero_pong_env_smoke.py src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py
```

```sh
uv run --extra modal python -c "import curvyzero.infra.modal.lightzero_pong_eval_smoke as m; print(m.DEFAULT_CHECKPOINT_REF); print(m.DEFAULT_MAX_EVAL_STEPS)"
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --max-eval-steps 384 --step-detail-limit 8
```

No pytest was run.

## Results

Compile passed. Import check passed.

First Modal run correctly found the frame-stack contract issue:

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-oZ9TMQ03OEEVVMRfNoYPOh
ok: false
env_reset_ok: true
policy_could_act_in_real_env: false
error: expected input[1, 1, 64, 64] to have 4 channels
```

After adding the eval-only 4-frame stack, Modal smoke passed:

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-zRGq8FFHsLhoKC4Ovf37aa
ok: true
remote_elapsed_sec: 17.642767
checkpoint_load_ok: true
strict_policy_model_load_ok: true
env_reset_ok: true
policy_could_act_in_real_env: true
model_fallback_used: false
steps_run: 8
```

Checkpoint ref:

```text
training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/iteration_1.pth.tar
sha256: bf82857b2c74ba96072738cc745e2c141e182802e37e9d37cacb09172cf5e931
bytes: 96204713
```

Output artifact:

```text
training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/eval/iteration_1_tiny/lightzero_visual_pong_eval_smoke_20260509T173129Z.json
sha256: 5c110d7ec8e4b715d2086ecedfe9772a498b9b537060e29ce120980effabaac7
```

Tiny eval summary:

```text
policy API: MuZeroPolicy.eval_mode.forward
policy input shape: [1, 4, 64, 64]
raw env observation shape: [1, 64, 64]
policy observation shape: [4, 64, 64]
action mask shape: [1, 6]
actions: [0, 0, 0, 0, 0, 0, 0, 0]
rewards: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
total_reward: 0.0
done: false
fallback_step_count: 0
```

First policy output sample:

```text
predicted_policy_logits:
  [0.0203092117, -0.0009354891, 0.0021116016,
   -0.0071478975, -0.0121046770, -0.0022327297]
visit_count_distributions: [1, 1, 0, 0, 0, 0]
searched_value: 0.000017193873645737767
predicted_value: -0.00002586841583251953
action: 0
```

Opening/serve capped eval rung:

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-S0HADSUdYxYsy6y1yGj4mP
ok: true
max_eval_steps: 384
max_episode_steps: 64
num_simulations: 2
remote_elapsed_sec: 25.381584
episode_elapsed_sec: 6.651671
steps_run: 64
steps_per_sec: 9.621642
policy_eval_step_count: 64
fallback_step_count: 0
action_histogram: {0: 64}
reward_histogram: {-1.0: 1, 0.0: 63}
nonzero_reward_steps: [{step_index: 60, reward: -1.0}]
done: true
terminal step: 63
terminal info: TimeLimit.truncated true, eval_episode_return -1.0,
  episode_frame_number 269, frame_number 269
raw env observation shape: [1, 64, 64]
policy observation shape: [4, 64, 64]
artifact:
  training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/eval/iteration_1_tiny/lightzero_visual_pong_eval_smoke_20260509T173516Z.json
artifact sha256:
  aa99b492c0d18a5f3288a3dcd7f97feb491d050777d10557710e15a1d06300b1
```

The longer outer eval cap did not bind because the patched tiny env still had
`max_episode_steps=64`. That is acceptable for this rung: it crossed the
opening/serve region, observed a real nonzero ALE reward (`-1.0` at step 60),
and observed `done=true` from the env time-limit wrapper at step 63. The terminal
info explicitly reports `TimeLimit.truncated: true`, so this is not a full Pong
game-over signal.

## Interpretation

The eval-only gate is positive. The mirrored `iteration_1` stock visual Pong
checkpoint loads strictly into `MuZeroPolicy`, resets a real ALE-backed
LightZero Pong env, and chooses actions through policy eval mode for a tiny
capped rollout.

The opening/serve rung is also positive for eval mechanics: the same policy path
ran 64 real env steps, saw a nonzero reward, and terminated through the env's
time-limit truncation path. This is still not policy-quality evidence. The
policy selected action `0` for every step, so the only claim is that the stock
Atari visual eval harness can carry the loaded checkpoint through real ALE
dynamics and telemetry.

Keep this separate from project dummy Pong. Dummy Pong exploration/data
distribution findings do not establish anything about this stock
`PongNoFrameskip-v4` visual path, and this stock visual eval does not make a
dummy Pong target-quality claim.

## Next

Stop the simple opening/serve mechanics question: it is answered. The next
official visual go, only if we want one more mechanics check before training, is
a similarly capped eval with a larger `--max-episode-steps` so the episode does
not truncate at 64 and can show post-point continuation. Do not promote this to
learned Pong or mix it with dummy Pong results.
