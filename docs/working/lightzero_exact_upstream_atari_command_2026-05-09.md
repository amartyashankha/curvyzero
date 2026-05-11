# LightZero Exact Upstream Atari Pong Command - 2026-05-09

Scope: derive the closest exact-upstream LightZero Atari Pong MuZero command we
can run from this repo on Modal. No pytest. No training was run. One cheap
Modal dry/config import was run.

## Answer

The exact current upstream LightZero Atari Pong command is still the official
quick-start command:

```sh
cd LightZero
python3 -u zoo/atari/config/atari_muzero_config.py
```

That command runs `PongNoFrameskip-v4` through `zoo.atari.config.atari_muzero_config`
and calls `lzero.entry.train_muzero([main_config, create_config], seed=0,
max_env_step=max_env_step)`.

Current GitHub upstream `main` requires the heavyweight Atari settings: `500000`
env steps, `50` MCTS simulations, `8` collector envs, `8` collect episodes,
`3` evaluator envs, `3` evaluator episodes, `batch_size=256`, CUDA, four-frame
`(4, 64, 64)` grayscale visual input, `game_segment_length=400`,
`update_per_collect=None`, `replay_ratio=0.25`, `learning_rate=0.2`,
`target_update_freq=100`, `eval_freq=2000`, and replay buffer size `1000000`.

Plainly: exact current upstream is not a 4096-step or sim10 run. It is a
`500000` env-step, `50`-simulation Atari run. Anything smaller is a staged
control, not exact replication.

## Sources Checked

- Official LightZero quick start says Pong MuZero is:
  `python3 -u zoo/atari/config/atari_muzero_config.py`.
  Source: https://opendilab.github.io/LightZero/tutorials/installation/installation_and_quickstart.html
- Current upstream GitHub `zoo/atari/config/atari_muzero_config.py` defaults to
  `PongNoFrameskip-v4`, `train_muzero`, `collector_env_num=8`,
  `n_episode=8`, `evaluator_env_num=3`, `num_simulations=50`,
  `update_per_collect=None`, `replay_ratio=0.25`, `batch_size=256`, and
  `max_env_step=int(5e5)`.
  Source: https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_config.py
- LightZero's config docs define the knobs in the expected way:
  collector/evaluator env counts, `game_segment_length`, `update_per_collect`,
  `num_simulations`, `eval_freq`, `replay_buffer_size`, and
  `target_update_freq`.
  Source: https://opendilab.github.io/LightZero/tutorials/config/config.html

## Important Package Drift

Our current Modal wrappers install `LightZero==0.2.0`, not a fresh GitHub
checkout. The dry run below imported that package and captured this original
surface:

```text
module: zoo.atari.config.atari_muzero_config
env_id: PongNoFrameskip-v4
collector_env_num: 8
evaluator_env_num: 3
n_episode: 8
n_evaluator_episode: 3
num_simulations: 50
batch_size: 256
update_per_collect: null
game_segment_length: 400
eval_freq: 2000
observation_shape: [4, 64, 64]
max_env_step: 200000
```

So there are two honest definitions:

- exact current GitHub upstream: `500000` env steps;
- exact installed PyPI `LightZero==0.2.0` package surface in our Modal image:
  `200000` env steps.

If the goal is "exact upstream LightZero today", use a pinned GitHub source
checkout or install from a pinned GitHub commit. If the goal is "exact package
we currently run in Modal", keep `LightZero==0.2.0` and use `200000`.

## Cheap Dry Command Run

This import/config-only command was run successfully. It did not instantiate
the trainer and did not train.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --mode dry --compute cpu --update-per-collect -1 --run-id lz-visual-pong-exact-upstream-dry-s0 --attempt-id dry-current-wrapper-stock-auto-upc
```

Result:

```text
Modal app: ap-tHejfPKJK3JhrmdvLyRqO7
ok: true
mode: dry
call_policy: dry_config_patch_only
remote_elapsed_sec: 8.070465
LightZero: 0.2.0
DI-engine: 0.5.3
torch: 2.11.0
gym: 0.25.1
gymnasium: 0.28.0
ale-py: 0.8.1
opencv-python-headless: 4.11.0.86
AutoROM: 0.6.1
```

The dry run preserved stock `update_per_collect=None` via our wrapper sentinel
`--update-per-collect -1`, but the wrapper still patched the rest down to tiny
CPU settings. Its value is import/config evidence, not a training command.

Cheap validation command to keep using before any exact wrapper launch:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke \
  --mode dry \
  --compute cpu \
  --update-per-collect -1 \
  --run-id lz-visual-pong-exact-upstream-dry-s0 \
  --attempt-id dry-current-wrapper-stock-auto-upc
```

For a future exact wrapper, add the equivalent no-train validation command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_upstream_train \
  --mode dry-exact \
  --source github-main \
  --env-id PongNoFrameskip-v4 \
  --seed 0 \
  --run-id lz-visual-pong-exact-github-main-dry-s0 \
  --attempt-id dry-exact-config-surface
```

Expected dry-exact validation: it should print the unmodified upstream config
surface, then print a second surface with only Modal output path metadata
changed. It must not construct the Atari env and must not call `train_muzero`.

## Exact Modal Command Shape

The exact current-upstream Modal command should be a new wrapper mode, not the
existing tiny wrapper with giant CLI values. Shape:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_upstream_train \
  --source github-main \
  --env-id PongNoFrameskip-v4 \
  --seed 0 \
  --max-env-step 500000 \
  --collector-env-num 8 \
  --n-episode 8 \
  --evaluator-env-num 3 \
  --n-evaluator-episode 3 \
  --num-simulations 50 \
  --batch-size 256 \
  --update-per-collect none \
  --replay-ratio 0.25 \
  --game-segment-length 400 \
  --eval-freq 2000 \
  --run-id lz-visual-pong-exact-github-main-s0 \
  --attempt-id train-exact-500k-sim50
```

For an exact installed-package replication instead of current GitHub `main`,
the same command should use `--source pypi-lightzero-0.2.0` and
`--max-env-step 200000`.

## Wrapper Changes Needed

Current file: `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`.
This wrapper is excellent for smokes, but it is intentionally not an exact
upstream runner.

Required for exact current GitHub upstream:

- Add a separate exact-upstream wrapper/module or `--profile exact-upstream`.
  Do not broaden the tiny smoke silently.
- Install or mount a pinned LightZero GitHub checkout when `--source github-main`
  or `--source github-commit:<sha>` is requested. Current Modal image uses
  `LightZero==0.2.0`, which captured `max_env_step=200000`, not GitHub `500000`.
- Keep the existing Atari ROM image behavior:
  `AutoROM[accept-rom-license]==0.6.1`, `AutoROM --accept-license`, and
  `opencv-python-headless==4.11.0.86`.
- Keep only output/artifact patches:
  set `main_config.exp_name` to a Modal-safe path or mirror the stock
  `data_muzero/...` directory into `curvyzero-runs`. Prefer a mounted output
  root or periodic mirror so long runs do not lose checkpoints on interruption.
- Do not patch training semantics in exact mode:
  no episode caps, no collector/evaluator reductions, no `eval_freq=1`, no
  forced `save_ckpt_after_iter=1`, no reduced simulations, no reduced batch,
  no reduced `game_segment_length`, no forced `update_per_collect`.
- Preserve stock `update_per_collect=None`; our current sentinel
  `--update-per-collect -1` is the right CLI idea, but exact mode should report
  it as `None` in the saved config.
- Remove or bypass the current validation caps for exact mode. Current caps are:
  `max_env_step<=8192`, `max_train_iter<=64`, `collector_env_num<=4`,
  `evaluator_env_num<=1`, `n_evaluator_episode<=1`,
  `collect/eval_max_episode_steps<=1024`, `num_simulations<=25`,
  `batch_size<=64`, `game_segment_length<=128`, and
  `update_per_collect<=4` when not `None`.
- Allow `evaluator_env_num=3` and `n_evaluator_episode=3`.
- Allow `collector_env_num=8` and `n_episode=8`.
- Allow `num_simulations=50`, `batch_size=256`, and `game_segment_length=400`.
- Use CUDA exactly as upstream (`policy.cuda=True`). The existing GPU wrapper
  already does this for `--compute gpu-l4-t4`.
- Raise timeout and resource assumptions for a real 500k/sim50 run. The current
  GPU function timeout is `60m`, which is a smoke budget, not an exact Atari
  replication budget.
- Save an immutable run manifest with source identity:
  GitHub URL/commit or PyPI package version, package versions, command args,
  original config, patched config, ROM license note, Modal app id, and artifact
  refs.
- Keep checkpoint mirroring, but make it robust for long runs. The current
  end-of-run mirror is enough for short controls; exact runs need periodic
  manifest/checkpoint sync or direct Volume-backed `exp_name`.

Optional but useful:

- Add `--dry-exact` that imports the selected source, prints the unmodified
  config surface, applies only output-path mirroring, and exits before env
  creation or training.
- Add `--source pypi-lightzero-0.2.0` versus `--source github-main` to make the
  200k-vs-500k distinction impossible to miss.

## Failure-Proof Checklist

Use this before calling anything "exact upstream."

Must remain unpatched:

- `env.env_id = "PongNoFrameskip-v4"`.
- `create_config.env.type = "atari_lightzero"`.
- `create_config.env.import_names = ["zoo.atari.envs.atari_lightzero_env"]`.
- `create_config.env_manager.type = "subprocess"`.
- `create_config.policy.type = "muzero"`.
- `create_config.policy.import_names = ["lzero.policy.muzero"]`.
- Trainer entrypoint is `lzero.entry.train_muzero`.
- `policy.model.model_type = "conv"`.
- `policy.model.observation_shape = (4, 64, 64)`.
- `env.observation_shape = (4, 64, 64)`.
- `frame_stack_num = 4`, `gray_scale = True`, and `image_channel = 1`.
- `policy.model.action_space_size = 6` for `PongNoFrameskip-v4`.
- `env.collector_env_num = 8` and `policy.collector_env_num = 8`.
- `policy.n_episode = 8`.
- `env.evaluator_env_num = 3` and `policy.evaluator_env_num = 3`.
- `env.n_evaluator_episode = 3`.
- `policy.num_simulations = 50`.
- `policy.batch_size = 256`.
- `policy.update_per_collect = None`.
- `policy.replay_ratio = 0.25`.
- `policy.game_segment_length = 400`.
- `policy.cuda = True`.
- `policy.learning_rate = 0.2`.
- `policy.target_update_freq = 100`.
- `policy.eval_freq = 2000`.
- `policy.replay_buffer_size = 1000000`.
- Current GitHub upstream exact: `max_env_step = 500000`.
- Installed PyPI `LightZero==0.2.0` exact: `max_env_step = 200000`, based on
  the Modal dry import captured above.

May be patched for Modal artifact/output only:

- `main_config.exp_name`, but only to move LightZero logs/checkpoints into a
  Modal-safe output root or mounted Volume.
- Run/attempt ids, manifest paths, and Volume refs.
- End-of-run or periodic mirroring of LightZero-produced checkpoints/logs into
  `curvyzero-runs`.
- Additional JSON sidecars that record package versions, source commit/package
  version, command args, config surfaces, Modal app id, ROM license note, and
  artifact checksums.
- Modal image setup required to make the stock env run:
  `opencv-python-headless`, `AutoROM[accept-rom-license]`, and
  `AutoROM --accept-license`.
- Modal resource declarations: GPU class, CPU count, memory, and timeout.
- Wrapper-level dry/import mode, provided it exits before env creation or
  training.
- Wrapper logging around stdout/stderr capture, provided it does not change
  trainer config or trainer call semantics.

Not upstream-faithful anymore if any of these change:

- Reducing `max_env_step` below the exact source value. A lower value can be a
  faithful subset only if all other key settings stay stock and the report says
  it is a subset.
- Reducing `num_simulations` below `50`.
- Reducing `collector_env_num` or `policy.n_episode` below `8`.
- Reducing `evaluator_env_num` or `n_evaluator_episode` below `3`.
- Reducing `batch_size` below `256`.
- Changing `update_per_collect` from `None` to a fixed integer.
- Changing `replay_ratio` from `0.25`.
- Adding `collect_max_episode_steps` or `eval_max_episode_steps` caps.
- Changing `game_segment_length` from `400`.
- Changing `eval_freq` from `2000`.
- Forcing `save_ckpt_after_iter=1` or otherwise changing checkpoint cadence in
  the training config.
- Disabling CUDA for a train run.
- Switching from `train_muzero` to `train_muzero_segment`.
- Switching to `atari_muzero_segment_config.py` without labeling it as the
  segment collector variant.
- Loading a pretrained checkpoint whose model/config shape differs from the
  selected source config.
- Using non-strict model loading, action fallback, checkpoint shape surgery, or
  evaluator fallback.
- Switching to dummy Pong, CurvyZero, PettingZoo Pong, a custom wrapper, or any
  non-ALE environment.

Dry validation failure gates:

- Stop if the selected source does not expose `PongNoFrameskip-v4`.
- Stop if current GitHub source does not report `max_env_step=500000` for
  `atari_muzero_config.py`.
- Stop if PyPI package mode is requested but the imported package surface is
  not explicitly reported as `LightZero==0.2.0`.
- Stop if action space is not `6`.
- Stop if config comparison shows more than output/artifact fields changed.
- Stop if the wrapper cannot record source identity and package versions.
- Stop if ROM license handling is absent from the Modal image that will run env
  creation or training.

## Current Wrapper Ceiling Command

This is the biggest command the current tiny wrapper is designed to accept
without code changes. It is not faithful upstream; it is just the current cap
ceiling.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke \
  --compute gpu-l4-t4 \
  --mode train \
  --max-env-step 8192 \
  --max-train-iter 64 \
  --collector-env-num 4 \
  --evaluator-env-num 1 \
  --num-simulations 25 \
  --batch-size 64 \
  --update-per-collect -1 \
  --max-episode-steps 1024 \
  --game-segment-length 128 \
  --run-id lz-visual-pong-current-wrapper-ceiling-s0 \
  --attempt-id train-8192-sim25-b64-env4-stockupc
```

Do not call that an upstream replication. It changes too many key settings:
env-step budget, collector/evaluator counts, simulations, batch size, episode
caps, eval cadence, checkpoint cadence, and segment length.

## Staged Faithful Subset

The faithful subset should preserve the key upstream settings and reduce only
the outer budget first.

Recommended staged exact-mode ladder after adding the exact wrapper:

| Stage | max env steps | sims | collectors | evaluators | batch | game segment | update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dry exact | `0` | `50` | `8` | `3` | `256` | `400` | `None` |
| env/import | `0` | `50` | `8` | `3` | `256` | `400` | `None` |
| faithful subset A | `10000` | `50` | `8` | `3` | `256` | `400` | `None` |
| faithful subset B | `50000` | `50` | `8` | `3` | `256` | `400` | `None` |
| exact GitHub upstream | `500000` | `50` | `8` | `3` | `256` | `400` | `None` |

The first trainable subset command should look like this once exact mode
exists:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_upstream_train \
  --source github-main \
  --env-id PongNoFrameskip-v4 \
  --seed 0 \
  --max-env-step 10000 \
  --collector-env-num 8 \
  --n-episode 8 \
  --evaluator-env-num 3 \
  --n-evaluator-episode 3 \
  --num-simulations 50 \
  --batch-size 256 \
  --update-per-collect none \
  --replay-ratio 0.25 \
  --game-segment-length 400 \
  --eval-freq 2000 \
  --run-id lz-visual-pong-faithful-10k-s0 \
  --attempt-id train-faithful-10k-sim50
```

This is still not policy-quality Pong. It is a faithful low-budget run because
it preserves the visual model, env family, collector/evaluator shape, search
budget, batch size, replay-ratio semantics, and segment length while reducing
only the training horizon.

## Non-Claims

- This is stock ALE Atari Pong, not project dummy Pong and not CurvyTron.
- The existing 4096/sim10 run was an infrastructure pass and quality fail; it
  should not be upgraded into an upstream claim.
- The Hugging Face pretrained Pong checkpoint remains shape-incompatible with
  the current 64x64 stock config path; do not non-strict-load it as a shortcut.
- No pytest was run for this note.
- No training was run for this note.
