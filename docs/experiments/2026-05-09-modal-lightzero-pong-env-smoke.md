# 2026-05-09 Modal LightZero Pong Env Smoke

## Question

Can the current pinned `LightZero==0.2.0` Modal image create, reset, and step
the stock Atari Pong environment before we attempt any stock Pong trainer?

## Setup

Added a separate no-train Modal module:

```text
src/curvyzero/infra/modal/lightzero_pong_env_smoke.py
```

The module imports the stock Atari Pong MuZero segment config, applies the same
tiny CPU patches as the dry-config smoke, then tries:

- DI-engine `get_vec_env_setting` with the patched `atari_lightzero` env config;
- direct LightZero/DI-engine Atari env class imports;
- plain `gym` and `gymnasium` `PongNoFrameskip-v4` creation as diagnostics.

It does not call `train_muzero_segment`.

## Command

```sh
uv run python -m py_compile src/curvyzero/infra/modal/lightzero_pong_env_smoke.py
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_env_smoke
```

No pytest was run.

## Results

Local compile passed. The first Modal run returned `ok: false` with missing
`cv2`. After adding pinned OpenCV headless to the Modal image, the rerun still
returned `ok: false`, but the failure moved to the expected ROM gate.

Later on 2026-05-09, the smoke was updated to use a separate ROM-enabled Modal
image helper:

```text
src/curvyzero/infra/modal/lightzero_atari_rom_image.py
```

That helper makes the license step explicit at the image boundary:

```python
uv_pip_install("AutoROM[accept-rom-license]==0.6.1")
run_commands("AutoROM --accept-license")
```

With that image, stock ALE Pong reset/step now passes through the
LightZero/DI-engine path. This is still not a trainer run.

Earlier package surface before ROM handling:

```text
LightZero 0.2.0
DI-engine 0.5.3
torch 2.11.0
gym 0.25.1
gymnasium 0.28.0
ale-py 0.8.1
opencv-python-headless 4.11.0.86
AutoROM missing
atari-py missing
envpool missing
```

Stock config capture and tiny patches still worked:

```text
PongNoFrameskip-v4, muzero, atari_lightzero, conv
observation_shape [4, 96, 96], action_space_size 6
patched collector_env_num 1, evaluator_env_num 1
patched num_simulations 2, batch_size 4, update_per_collect 1
patched cuda false, max_env_step 4
```

Before OpenCV prep, environment creation failed before reset/step:

```text
DI-engine/LightZero path:
ModuleNotFoundError: No module named 'cv2'
from zoo.atari.envs.atari_wrappers import wrap_lightzero
import cv2
```

Direct Atari env class imports failed for the same missing `cv2` dependency.

After adding `opencv-python-headless==4.11.0.86`, `cv2` imports cleanly and the
LightZero/DI-engine path reaches `AtariEnvLightZero.reset()`. It then fails
because the Pong ROM is not available:

```text
env_type: zoo.atari.envs.atari_lightzero_env.AtariEnvLightZero
reset(): gym.error.Error: We're Unable to find the game "Pong". Note: Gym no
longer distributes ROMs...
```

Plain Gym/Gymnasium diagnostics fail at the same ROM gate:

```text
gym.error.Error: We're Unable to find the game "Pong". Note: Gym no longer
distributes ROMs...
gymnasium.error.Error: We're Unable to find the game "Pong". Note: Gymnasium
no longer distributes ROMs...
```

The final rerun reported:

```text
ok: false
env_ok: false
lightzero_path_ok: false
imports: cv2 ok
train_result: null
remote_elapsed_sec: 9.057508
```

ROM-enabled rerun:

```text
ok: true
env_ok: true
lightzero_path_ok: true
problems: []
env_type: zoo.atari.envs.atari_lightzero_env.AtariEnvLightZero
reset.ok: true
step.ok: true
packages: AutoROM 0.6.1, LightZero 0.2.0, DI-engine 0.5.3,
  gym 0.25.1, gymnasium 0.28.0, ale-py 0.8.1,
  opencv-python-headless 4.11.0.86
missing but not needed for this pass: atari-py, envpool
train_result: null
remote_elapsed_sec: 8.273704
```

The successful path was `ding.envs.get_vec_env_setting` with the compiled
`atari_lightzero` env config. `reset()` returned a LightZero observation dict
with `observation`, `action_mask`, `to_play`, and `timestep`; `step(0)`
returned a `BaseEnvTimestep`.

## Interpretation

The earlier stock LightZero Modal image could import the OpenCV-dependent Atari
wrappers, but it could not create/reset/step Pong because the Atari ROM was not
installed. That was not a trainer issue; it was runtime dependency and
license/ROM-management.

After explicit AutoROM handling, stock `PongNoFrameskip-v4` create/reset/step
works on Modal through the official LightZero/DI-engine Atari path. This is a
stock ALE Pong result, not project dummy Pong and not dummy `raster_flat`.

AutoROM documents two accepted license paths: running `AutoROM --accept-license`
or installing the accept-license extra, e.g. `pip install "autorom[accept-rom-license]"`
([AutoROM on PyPI](https://pypi.org/project/AutoROM/)). That step downloads
ROMs and makes them discoverable to ALE. This experiment now uses that explicit
license acceptance in `lightzero_atari_rom_image.py`.

The useful result is now positive for the environment gate: stock visual Pong
env reset/step works. The next command is the already-dry trainer config
surface check, not a long train.

The follow-up dry config smoke also passed after the env gate:

```text
ok: true
mode: dry
module: zoo.atari.config.atari_muzero_segment_config
trainer_entrypoint: lzero.entry.train_muzero_segment
patched surface: PongNoFrameskip-v4, muzero, atari_lightzero, conv,
  action_space_size 6, collector/evaluator env 1,
  num_simulations 2, batch_size 4, update_per_collect 1,
  cuda false, max_env_step 4
train_result: null
```

## Artifacts

- Modal run URL:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-V1IOsGMYKXIeK0jlzwRYPl`
- Modal rerun URL after OpenCV prep:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-RN85Uow3R0MhUpj3XNjY2D`
- Modal rerun URL after explicit AutoROM handling:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-VVbl7mqbfAyLdzJDREp0OA`
- Modal dry-config rerun URL after env pass:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-mKUIV74YIg8xesIUgw5n49`
- Code module:
  `src/curvyzero/infra/modal/lightzero_pong_env_smoke.py`
- ROM image helper:
  `src/curvyzero/infra/modal/lightzero_atari_rom_image.py`

## Follow-ups

The next stock Atari Pong command should be a brutally capped, whole-job
LightZero MuZero trainer smoke based on the official `atari_muzero_config`
or segment config. Do not start a long train from this result. The env gate
only proves stock ALE Pong can reset/step on Modal.
