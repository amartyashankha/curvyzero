# LightZero Official Visual Pong Pattern - 2026-05-09

Scope: local repo/docs/source scout plus primary-source web cross-check. No
pytest was run.

## Answer

The closest official stock LightZero visual-control MuZero example to Pong is
Atari/ALE Pong itself:

Plain-language note: `official Atari Pong` means LightZero's built-in Pong
example using the Atari emulator. `ALE` is the Atari emulator used by
Gym/LightZero. A `checkpoint` is a saved model file. This whole lane is a
sanity check for LightZero's normal visual training path, not project dummy
Pong or CurvyTron.

- `zoo.atari.config.atari_muzero_config`
- env id: `PongNoFrameskip-v4`
- entry: `lzero.entry.train_muzero`
- env type: `atari_lightzero`
- model: convolutional visual MuZero, stacked grayscale frames
- action space: Atari Pong's 6-action discrete control surface

There is also an official stock segment-collector path:

- `zoo.atari.config.atari_muzero_segment_config`
- default CLI env: `PongNoFrameskip-v4`
- entry: `lzero.entry.train_muzero_segment`
- same `atari_lightzero` env family
- chunked collection via `num_segments=8` and `game_segment_length=20`

For the visual-control question, these Atari files are the central evidence.
TicTacToe/Connect4 are useful controls for sparse terminal rewards only; they
are not visual Pong controls.

## Local Evidence

Repo/vendored scan:

- `third_party/README.md` says the only current third-party clone is
  `third_party/curvytron-reference/`; no vendored LightZero checkout exists
  under this repo.
- The local working docs cite and inspect a separate official LightZero source
  snapshot at `/tmp/lightzero-src`.
- Relevant repo Modal wrappers already exist:
  - `src/curvyzero/infra/modal/lightzero_pong_dry_config_smoke.py`
  - `src/curvyzero/infra/modal/lightzero_pong_env_smoke.py`
  - `src/curvyzero/infra/modal/lightzero_cartpole_tiny_train_smoke.py`
  - `src/curvyzero/infra/modal/lightzero_tictactoe_tiny_train_smoke.py`
  - `src/curvyzero/infra/modal/lightzero_connect4_tiny_train_smoke.py`

Local official LightZero snapshot findings:

- `/tmp/lightzero-src/zoo/atari/config/atari_muzero_config.py`
  defaults `env_id = 'PongNoFrameskip-v4'`, uses `train_muzero`, `collector_env_num=8`,
  `n_episode=8`, `evaluator_env_num=3`, `num_simulations=50`,
  `update_per_collect=None`, `replay_ratio=0.25`, `batch_size=256`,
  `max_env_step=5e5`, `game_segment_length=400`, `cuda=True`, `env_type='not_board_games'`,
  conv model, `observation_shape=(4, 64, 64)`, and `action_space_size` from
  `atari_env_action_space_map`.
- `/tmp/lightzero-src/zoo/atari/config/atari_muzero_segment_config.py`
  exposes `main(env_id, seed)`, defaults CLI `--env PongNoFrameskip-v4`, uses
  `train_muzero_segment`, `num_segments=8`, `game_segment_length=20`,
  `train_start_after_envsteps=2000`, `update_per_collect=None`,
  `replay_ratio=0.25`, `num_simulations=50`, `batch_size=256`, `cuda=True`,
  conv model, and `observation_shape=(4, 64, 64)`.
- `/tmp/lightzero-src/lzero/agent/config/muzero/gym_pongnoframeskip_v4.py`
  is another Pong MuZero config, but it is agent-oriented and heavier:
  `update_per_collect=1000`, `max_env_step=1e6`, `observation_shape=(4, 96, 96)`,
  `game_segment_length=400`, `cuda=True`.
- `/tmp/lightzero-src/lzero/agent/config/efficientzero/gym_pongnoframeskip_v4.py`
  and `/tmp/lightzero-src/lzero/agent/config/sampled_efficientzero/gym_pongnoframeskip_v4.py`
  confirm EfficientZero/SampledEfficientZero Pong paths exist, but they are not
  the smallest MuZero answer.

Existing repo docs/results:

- `docs/experiments/2026-05-09-modal-lightzero-pong-gpu1024-control.md`
  is the current cheap-GPU official Atari control. It used the same L4/T4
  Modal path with `max_env_step=1024`, `max_train_iter=8`, `batch_size=8`,
  256-step train/eval episode caps, and `game_segment_length=16`. The run
  completed on an NVIDIA L4, mirrored checkpoints through `iteration_4`, and
  evaluated `iteration_0`, `iteration_2`, `iteration_4`, plus the old GPU512
  final checkpoint at a 256-step eval cap with no model fallback. The final
  GPU1024 checkpoint used all six Atari actions, scored one positive reward,
  and returned `-3.0`; the same-cap GPU512 baseline returned `-5.0`. This is a
  small real signal, not a solved-policy result.
- `docs/experiments/2026-05-09-modal-lightzero-pong-dry-config-smoke.md`
  confirms the installed `LightZero==0.2.0` package can import/capture the stock
  Atari Pong segment config and patch it to a tiny CPU dry surface:
  `PongNoFrameskip-v4`, `atari_lightzero`, conv model, action space `6`, patched
  `collector_env_num=1`, `evaluator_env_num=1`, `num_simulations=2`,
  `batch_size=4`, `update_per_collect=1`, `cuda=false`, `max_env_step=4`.
- `docs/experiments/2026-05-09-modal-lightzero-pong-env-smoke.md`
  now confirms the ROM-enabled Modal image creates, resets, and steps stock
  ALE `PongNoFrameskip-v4` through the LightZero/DI-engine `atari_lightzero`
  path. Earlier runs reached the ROM gate after OpenCV; the current blocker
  was fixed with explicit AutoROM handling. This is stock ALE Pong, not project
  dummy Pong and not dummy `raster_flat`.
- `docs/experiments/2026-05-09-modal-lightzero-pong-tiny-train-smoke.md`
  confirms a brutally capped Modal whole-job trainer now calls the official
  stock visual Pong path, `zoo.atari.config.atari_muzero_config` plus
  `lzero.entry.train_muzero`, on CPU with one collector/evaluator env,
  `num_simulations=2`, `batch_size=4`, `max_env_step=4`,
  `max_train_iter=1`, and 64-step collect/eval episode caps. It completed,
  returned `MuZeroPolicy`, and mirrored `ckpt_best.pth.tar`,
  `iteration_0.pth.tar`, and `iteration_1.pth.tar` to `curvyzero-runs`.
- `docs/experiments/2026-05-09-modal-lightzero-official-example-sanity.md`
  proves stock CartPole MuZero can train on Modal CPU, but it is state-control,
  not visual control.
- `docs/experiments/2026-05-09-modal-lightzero-official-sparse-example-sanity.md`
  and `docs/experiments/2026-05-09-modal-lightzero-official-connect4-sparse-smoke.md`
  prove sparse board-game examples can train on Modal CPU, but they do not
  exercise Atari images, ALE, ROMs, OpenCV wrappers, or visual control.

Primary-source web cross-check:

- LightZero's official installation/quick-start docs list Pong training as
  `python3 -u zoo/atari/config/atari_muzero_config.py`.
  Source: https://opendilab.github.io/LightZero/tutorials/installation/installation_and_quickstart.html
- Upstream LightZero `main` still has `zoo/atari/config/atari_muzero_config.py`
  with default `PongNoFrameskip-v4`, `atari_lightzero`, conv MuZero, and
  `train_muzero`.
  Source: https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_config.py
- Upstream LightZero `main` still has `zoo/atari/config/atari_muzero_segment_config.py`
  with `train_muzero_segment` and default CLI `PongNoFrameskip-v4`.
  Source: https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_segment_config.py
- Gymnasium Atari docs say ALE Atari environments require ROMs and point to
  license-accepting install paths such as `gymnasium[accept-rom-license]`.
  Source: https://gymnasium.farama.org/v0.28.0/environments/atari/
- AutoROM documents the explicit license path via `AutoROM --accept-license` or
  `pip install "autorom[accept-rom-license]"`.
  Source: https://pypi.org/project/AutoROM/

## Modal Image Dependencies

Current minimal image for stock Atari Pong env smoke:

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "LightZero==0.2.0",
        "opencv-python-headless==4.11.0.86",
        "AutoROM[accept-rom-license]==0.6.1",
    )
    .run_commands("AutoROM --accept-license")
)
```

The explicit helper is:

```text
src/curvyzero/infra/modal/lightzero_atari_rom_image.py
```

The license step is intentionally visible there because Atari ROM handling is
a project decision, not a trainer bug.

Observed package surface from the passing env smoke:

```text
LightZero 0.2.0
DI-engine 0.5.3
torch 2.11.0
gym 0.25.1
gymnasium 0.28.0
ale-py 0.8.1
opencv-python-headless 4.11.0.86
AutoROM 0.6.1
atari-py missing
envpool missing
```

`envpool` and `atari-py` are still missing, but they are not first-smoke
requirements. The LightZero/Gym/ALE path now passes create/reset/step without
them.

## Tiny Smoke Order

Run this first for stock ALE Pong:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_env_smoke
```

Current passing result on 2026-05-09:

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-VVbl7mqbfAyLdzJDREp0OA
ok: true
lightzero_path_ok: true
env_ok: true
reset ok
step ok
env_type: zoo.atari.envs.atari_lightzero_env.AtariEnvLightZero
env_id PongNoFrameskip-v4
train_result null
```

After create/reset/step succeeded, the dry config capture also passed:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_dry_config_smoke
```

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-mKUIV74YIg8xesIUgw5n49
ok: true
mode: dry
module: zoo.atari.config.atari_muzero_segment_config
patched surface: PongNoFrameskip-v4, muzero, atari_lightzero, conv,
  action_space_size 6, collector/evaluator env 1,
  num_simulations 2, batch_size 4, update_per_collect 1,
  cuda false, max_env_step 4
train_result null
```

Then, and only then, add a tiny no-quality trainer wrapper for stock visual
Pong. Recommended first trainer target:

```text
source_module: zoo.atari.config.atari_muzero_config
entry: lzero.entry.train_muzero
env_id: PongNoFrameskip-v4
collector_env_num: 1
evaluator_env_num: 1
n_evaluator_episode: 1
n_episode: 1
num_simulations: 2
batch_size: 4
update_per_collect: 1
cuda: false
max_env_step: 4 or 8
max_train_iter: 1
```

The segment path is a follow-up A/B:

```text
source_module: zoo.atari.config.atari_muzero_segment_config
entry: lzero.entry.train_muzero_segment
num_segments: 1
game_segment_length: 20
collector_env_num: 1
```

Use the non-segment `atari_muzero_config` first because it is the official
quick-start Pong MuZero path and has fewer moving parts. Use segment once the
basic visual env and one-iteration trainer are known-good.

Current trainer result on 2026-05-09:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --mode train
```

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-MbyIGvX6R815WMZcYzcAyu
ok: true
status: completed
mode: train
entry: lzero.entry.train_muzero
source_module: zoo.atari.config.atari_muzero_config
return_type: MuZeroPolicy
training_iterations: [0]
final_rewards: [-1.0]
remote_elapsed_sec: 30.227448
checkpoint refs:
  training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/ckpt_best.pth.tar
  training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/iteration_0.pth.tar
  training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/iteration_1.pth.tar
summary:
  training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/train/summary.json
```

Current checkpoint-load result on 2026-05-09:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_checkpoint_probe
```

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-YqqMmryhwgtKdFTbStnDKJ
ok: true
checkpoint:
  training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/iteration_1.pth.tar
  sha256 bf82857b2c74ba96072738cc745e2c141e182802e37e9d37cacb09172cf5e931
state_dict path: model
tensor_count: 175
strict direct MuZeroModel load: true
strict MuZeroPolicy model load: true
direct zero-observation forward: true
model parameters: 7,997,608
input shape: [1, 4, 64, 64]
latent_state shape: [1, 64, 8, 8]
policy_logits shape: [1, 6]
value shape: [1, 601]
artifact:
  training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/probe/lightzero_visual_pong_checkpoint_load_20260509T172430Z.json
```

The checkpoint-load gate is now positive. The next stock visual Pong step is a
tiny eval-only smoke against a real ALE reset observation, not a longer
training run.

This is still not quality evidence. It proves the official visual trainer
stack runs and writes artifacts under harsh caps.

Current eval-only result on 2026-05-09:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke
```

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-zRGq8FFHsLhoKC4Ovf37aa
ok: true
checkpoint:
  training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/iteration_1.pth.tar
  sha256 bf82857b2c74ba96072738cc745e2c141e182802e37e9d37cacb09172cf5e931
strict policy model load: true
env reset: true
policy could act in real env: true
policy API: MuZeroPolicy.eval_mode.forward
model fallback used: false
steps_run: 8
actions: [0, 0, 0, 0, 0, 0, 0, 0]
rewards: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
done: false
raw env observation shape: [1, 64, 64]
policy observation shape: [4, 64, 64]
artifact:
  training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/eval/iteration_1_tiny/lightzero_visual_pong_eval_smoke_20260509T173129Z.json
```

The first eval attempt exposed an important shape boundary: the direct
LightZero Atari env reset returns one preprocessed frame (`[1, 64, 64]`), while
the stock conv MuZero checkpoint expects a 4-frame stack (`[4, 64, 64]`). The
eval smoke now records raw env observations separately from the tiny local
policy-facing frame stack. This is an eval harness detail, not quality signal.

Current opening/serve capped eval result on 2026-05-09:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --max-eval-steps 384 --step-detail-limit 8
```

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
nonzero reward: step 60, reward -1.0
done: true
terminal: step 63, TimeLimit.truncated true, eval_episode_return -1.0
raw env observation shape: [1, 64, 64]
policy observation shape: [4, 64, 64]
artifact:
  training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/eval/iteration_1_tiny/lightzero_visual_pong_eval_smoke_20260509T173516Z.json
artifact sha256:
  aa99b492c0d18a5f3288a3dcd7f97feb491d050777d10557710e15a1d06300b1
```

This answers the simple official visual eval mechanics question: the loaded
checkpoint can act through `MuZeroPolicy.eval_mode.forward` long enough to cross
opening/serve dynamics and observe a real nonzero ALE reward. The outer
384-step cap did not bind because the tiny env's `max_episode_steps=64`
triggered `TimeLimit.truncated` at step 63. That terminal flag is useful
telemetry, but it is not a full Pong game-over claim.

## Control Separation

Sparse board-game controls:

- TicTacToe and Connect4 are useful to prove LightZero can train sparse
  terminal-outcome examples on Modal CPU.
- They use board-game contracts: compact board planes, masks, player turns or
  bot-mode wrappers, and `env_type='board_games'` / varied action-space
  patterns where appropriate.
- They are not visual-control evidence.

Visual Pong/Atari control:

- The real visual-control path is Atari/ALE `PongNoFrameskip-v4` through
  `atari_lightzero`, conv MuZero, frame stacks, OpenCV wrappers, Gym/ALE, ROMs,
  and the 6-action Atari control surface.
- Current status: ROM-enabled env create/reset/step works on Modal, config
  dry-capture works, and the capped stock visual Pong `train_muzero` whole-job
  smoke completed with mirrored checkpoints. The mirrored `iteration_1` now
  loads strictly through both the direct conv `MuZeroModel` and the compiled
  `MuZeroPolicy` model, with a cheap zero-observation forward. A tiny eval-only
  smoke also proves `MuZeroPolicy.eval_mode.forward` can choose actions from
  the loaded checkpoint in the real ALE-backed LightZero Pong env when given
  the expected 4-frame policy input stack. A slightly longer capped eval reached
  nonzero ALE reward and env time-limit truncation under the same stock visual
  path.

Project dummy Pong:

- `src/curvyzero/training/lightzero_dummy_pong_env.py` is a project custom env,
  not stock Atari Pong.
- Its current LightZero path uses a 3-action ego control surface and tabular or
  raster-flat project observations, not the official Atari visual conv stack.
- Do not use TicTacToe/Connect4 or dummy Pong results as evidence for stock
  visual Pong trainer quality. The stock ALE env gate is now its own passing
  result.

## Recommendation

Central next gate: prove a tiny eval-only path from the loaded stock Atari Pong
checkpoint before spending on longer training. The smallest meaningful
visual-control ladder is not TicTacToe, Connect4, dummy Pong, or dummy
`raster_flat`. It is:

1. Done: `lightzero_pong_env_smoke` passes create/reset/step for
   `PongNoFrameskip-v4`.
2. Done: `lightzero_pong_dry_config_smoke` still captures the tiny visual
   MuZero surface.
3. Done: `lightzero_pong_tiny_train_smoke --mode train` runs one capped Modal
   whole-job and mirrors artifacts.
4. Done: `lightzero_pong_checkpoint_probe` loads the mirrored stock visual
   Pong `iteration_1.pth.tar` checkpoint through the same official Atari config
   surface.
5. Done: `lightzero_pong_eval_smoke` runs a tiny eval-only rollout from the
   loaded `iteration_1` checkpoint against the real ALE-backed LightZero Pong
   env. It records raw `[1, 64, 64]` env frames, policy `[4, 64, 64]` frame
   stacks, actions, rewards, done flags, and eval-mode MCTS outputs.
6. Done: the opening/serve eval rung crossed real Atari dynamics, reached
   reward `-1.0` at step 60, and ended at step 63 with
   `TimeLimit.truncated=true` under the tiny 64-step episode cap.
7. Done: the CPU scale128 official Atari rung widened only wrapper caps
   (`max_env_step=128`, `max_train_iter=2`, 128-step collect/eval caps) and
   ran on Modal app `ap-qoTln2RP7Ly65hjCK3V4On`. It completed, but LightZero
   still stopped after `Training Iteration [0]` and saved only `iteration_0`
   plus `iteration_1`.
8. Done: the new `iteration_1` and the old tiny `iteration_1` were both
   evaluated under the same 128-step cap. New eval app
   `ap-xARflZIivWVe3TdtHD4vEL` and baseline app
   `ap-D0OopWDZJD8K191krNFBBC` matched exactly: actions `{0:128}`, return
   `-2.0`, rewards at steps `60` and `95`, terminal
   `TimeLimit.truncated=true`. This is real post-64-step Atari reward signal,
   but no improvement over all-action-0.
9. Done: the cheap-GPU later-checkpoint control ran on Modal
   `ap-NcECoDQrcIfrbpqmBRODbP`, run
   `lz-visual-pong-20260509T180945Z-29b83d6ee638`, attempt
   `attempt-20260509T180945Z-dc971b1ec0ff`. The wrapper used only the cheap
   GPU list `["L4", "T4"]` and recorded actual runtime GPU `NVIDIA L4`.
   It produced `iteration_0` through `iteration_4`. Eval app
   `ap-AUNehXPKKdkXbPOCW5WM7B` loaded `iteration_4` and ran 128 real ALE steps
   with no fallback. Actions used all six Atari actions
   `{0:21,1:24,2:22,3:25,4:22,5:14}`, but return stayed `-2.0` with rewards
   at steps `60` and `95`, matching the action-0 baselines under this cap.

The official Atari/ALE runtime surface is no longer blocked at reset/step.
Do not rerun scale128 or GPU512 as-is for quality. The next official Atari
rung should either evaluate more seeds/checkpoints to understand the
new nonconstant action distribution, or explicitly raise budget with a higher
eval bar. Keep it separate from project dummy Pong work.

## Completed GPU1024 Official Control

Keep this as an official Atari Pong control, not custom dummy Pong and not a
policy-quality claim.

Purpose: test whether a cheap-GPU checkpoint curve keeps the GPU512 nonconstant
action distribution and whether any checkpoint changes capped Atari return
timing.

Train command used:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke \
  --compute gpu-l4-t4 \
  --mode train \
  --max-env-step 1024 \
  --max-train-iter 8 \
  --batch-size 8 \
  --max-episode-steps 256 \
  --game-segment-length 16
```

Result:

- `gpu-l4-t4` landed on an NVIDIA L4.
- The 1024 env-step cap produced checkpoints through `iteration_4`; it did not
  reach `iteration_8`.
- `iteration_0`, `iteration_2`, `iteration_4`, and the old GPU512 final
  checkpoint were evaluated at a 256-step cap with no model fallback.
- GPU1024 `iteration_4` returned `-3.0`, used all six Atari actions, and got
  one positive reward at step `143`.
- The same-cap GPU512 final baseline returned `-5.0`.

Eval shape used:

```sh
# Actual checkpoints available were 0, 2, and 4.
for ITER in 0 2 4; do
  uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke \
    --checkpoint-ref training/lightzero-official-visual-pong/RUN_ID/checkpoints/lightzero/iteration_${ITER}.pth.tar \
    --run-id RUN_ID \
    --attempt-id ATTEMPT_ID \
    --output-ref training/lightzero-official-visual-pong/RUN_ID/attempts/ATTEMPT_ID/eval/iteration_${ITER}_gpu1024/lightzero_visual_pong_eval_gpu1024_iteration_${ITER}.json \
    --max-env-step 1024 \
    --max-train-iter 8 \
    --batch-size 8 \
    --max-episode-steps 256 \
    --game-segment-length 16 \
    --max-eval-steps 256 \
    --step-detail-limit 8 \
    --no-allow-model-fallback
done
```

Recorded eval fields for each checkpoint:

- action histogram;
- total return;
- nonzero reward steps;
- terminal/truncation info;
- policy eval step count and fallback count.

The allowed conclusion remains narrow: checkpoint exists, actions changed, and
capped return/reward timing changed. It is a small signal, not solved Pong.
