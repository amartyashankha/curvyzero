# LightZero Setup Fidelity Audit - 2026-05-09

Scope: official LightZero Atari Pong setup fidelity only. I compared upstream
LightZero docs/configs and the OpenDILab pretrained Pong card against the local
Modal wrappers and experiment records. No code changes outside this file. No
pytest.

## Verdict

**Partially followed.**

Extra blunt version: Coach built a good doorway into official Atari Pong, then
trained in the doorway and called it the room.

Coach did set up the official LightZero Atari Pong *path*:
`PongNoFrameskip-v4`, `atari_lightzero`, ALE ROM handling, LightZero's Atari
preprocessing, conv MuZero, six Atari actions, `lzero.entry.train_muzero`, and
native `.pth.tar` checkpoints.

Coach did **not** set up official LightZero Atari Pong exactly as upstream
intended for a real training run. The actual commands and wrapper patches turned
the upstream recipe into smoke/control runs: tiny env budgets, tiny search,
tiny batch, one or two collector envs, one evaluator env, explicit short
episode caps, short game segments, forced update counts, frequent checkpointing,
and mostly manual eval harnesses. That is valid plumbing work, but it is not a
faithful upstream Pong setup.

There is now an even sharper version-split verdict: **current GitHub upstream
and the installed `LightZero==0.2.0` package are different exact targets.**
Current GitHub upstream uses `max_env_step=500000`; the installed
`LightZero==0.2.0` package surface captured by the exact-upstream dry worker
uses `max_env_step=200000`. A run that silently mixes those targets is not
exact-upstream before we even discuss our tiny local caps.

If the question is "did we run upstream LightZero Pong as upstream intended?",
the answer is **no**. If the question is "did we prove our Modal image can touch
the official Pong stack?", the answer is **yes**.

## Upstream Baseline

Official quick start says Pong MuZero is:

```text
cd LightZero
python3 -u zoo/atari/config/atari_muzero_config.py
```

Primary source: https://opendilab.github.io/LightZero/tutorials/installation/installation_and_quickstart.html

Current upstream `zoo/atari/config/atari_muzero_config.py` defaults:

| Setting | Upstream value |
| --- | --- |
| env id | `PongNoFrameskip-v4` |
| env type | `atari_lightzero` |
| trainer | `lzero.entry.train_muzero` |
| env manager | `subprocess` |
| model | conv MuZero |
| observation | `(4, 64, 64)`, grayscale stack |
| action space | `6` |
| model downsample | `True` |
| collector envs / episodes | `8` / `8` |
| evaluator envs / episodes | `3` / `3` |
| num simulations | `50` |
| update per collect | `None` |
| replay ratio | `0.25` |
| batch size | `256` |
| max env steps | `500,000` |
| CUDA | `True` |
| game segment length | `400` |
| eval freq | `2000` |
| replay buffer size | `1,000,000` |

Primary source: https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_config.py

Installed-package split: local exact-upstream dry work found that our Modal
image's installed `LightZero==0.2.0` package has the same broad Pong config
surface but `max_env_step=200,000`, not GitHub `500,000`. That means there are
two valid exact targets:

| Target | Exact `max_env_step` |
| --- | ---: |
| Current GitHub upstream `main` | `500,000` |
| Installed PyPI `LightZero==0.2.0` in our Modal image | `200,000` |

Source: `docs/working/lightzero_exact_upstream_atari_command_2026-05-09.md`

There is also an upstream segment collector config,
`zoo/atari/config/atari_muzero_segment_config.py`, with the same `PongNoFrameskip-v4`
Atari family, `50` simulations, `256` batch size, `500,000` env steps, and
`train_muzero_segment`; it uses `num_segments=8`, `game_segment_length=20`, and
`train_start_after_envsteps=2000`.

Primary source: https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_segment_config.py

LightZero's config docs define the audited knobs directly: collector and
evaluator env counts, frame stack, downsample, game segment length,
`update_per_collect`, `batch_size`, `num_simulations`, `eval_freq`,
`replay_buffer_size`, and `target_update_freq`.

Primary source: https://opendilab.github.io/LightZero/tutorials/config/config.html

## Local Setup Evidence

Local trainer wrapper:

- `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`
- It explicitly describes itself as a "brutally capped" smoke, not a quality
  run.
- It imports `zoo.atari.config.atari_muzero_config`.
- It calls `lzero.entry.train_muzero`.
- It patches env counts, action size, CUDA, simulations, batch size,
  `update_per_collect`, `game_segment_length`, `eval_freq`, episode caps, and
  checkpoint cadence.

Local ROM/image helper:

- `src/curvyzero/infra/modal/lightzero_atari_rom_image.py`
- Installs `LightZero==0.2.0`, `opencv-python-headless==4.11.0.86`, and
  `AutoROM[accept-rom-license]==0.6.1`.
- Runs `AutoROM --accept-license`.

Local eval wrapper:

- `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py`
- Default eval is manual `MuZeroPolicy.eval_mode.forward`, with optional
  `--run-stock-evaluator`.
- Later parity used `lzero.worker.MuZeroEvaluator` and matched manual actions,
  but the default eval smoke is still a custom harness.

Local experiment records read:

- `docs/experiments/2026-05-09-modal-lightzero-pong-env-smoke.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-tiny-train-smoke.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-scale128-control.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-gpu512-control.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-gpu1024-control.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-gpu2048-control.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-4096-sim10.md`
- `docs/working/lightzero_stock_replication_control_plan_2026-05-09.md`

## Deviation Table

| Item | Upstream intended | Coach/local setup | Fidelity |
| --- | --- | --- | --- |
| Env id | `PongNoFrameskip-v4` | `PongNoFrameskip-v4` | Followed |
| ROM setup | Atari ROMs must be available for ALE/Gym | Explicit AutoROM license acceptance and ROM install | Followed, with visible local license handling |
| Env type | `atari_lightzero` | `atari_lightzero` | Followed |
| Atari preprocessing | LightZero Atari wrappers, grayscale frame stack | Real env reset returns `[1,64,64]`; policy-facing stack is `[4,64,64]` | Mostly followed; eval stack is custom but matches model input |
| Image size | Current upstream config uses `(4,64,64)` | Current local stock train/eval uses `(4,64,64)` | Followed for current config |
| Downsample | `policy.model.downsample=True` | Not patched away; retained | Followed |
| Trainer entry | `train_muzero` for quick-start config | `train_muzero` | Followed |
| Segment config | Separate official `train_muzero_segment` path exists | Dry-captured earlier; not used for real train controls | Not followed for segment recipe |
| `num_simulations` | `50` | `2` for most runs; `10` for 4096/sim10 | Not followed |
| `max_env_step` | GitHub `500,000`; installed `LightZero==0.2.0` `200,000` | `4`, `128`, `512`, `1024`, `2048`, `4096`; wrapper cap allows `8192` | Not followed; also mixed-target risk |
| `max_train_iter` | Not part of upstream script's normal Atari cap; env steps dominate | Added explicit caps: `1`, `2`, `4`, `8`, `16`, `32` | Not followed |
| Collector envs | `8` | Usually `1`; 4096/sim10 used `2` | Not followed |
| Collector episodes | `n_episode=8` | Usually `1`; 4096/sim10 effectively `2` via collector count | Not followed |
| Evaluator envs | `3` | `1` | Not followed |
| Evaluator episodes | `3` | `1` | Not followed |
| `update_per_collect` | `None` with `replay_ratio=0.25` | Usually `1`; 4096/sim10 used `2`; wrapper has a sentinel to restore `None`, but commands did not use it | Not followed |
| Replay ratio | `0.25` | Preserved only if `update_per_collect=None`; practical runs overrode update count | Partially followed |
| Replay buffer size | `1,000,000` | Not patched, but only a few thousand transitions can enter it | Config followed; effective data regime not followed |
| Batch size | `256` | `4`, `8`, or `32` | Not followed |
| Game segment length | `400` in non-segment config; `20` in segment config | `16` in most runs; `64` in 4096/sim10 | Not followed |
| Episode length caps | Upstream config leaves Atari episode caps commented out | Added `collect_max_episode_steps` and `eval_max_episode_steps`, often `64`, `128`, `256`, or `512` | Not followed |
| Eval frequency | `2000` | Patched to `1` | Not followed |
| Checkpoint cadence | Normal LightZero cadence | Patched `save_ckpt_after_iter=1` | Not followed |
| CUDA | `True` | CPU tiny runs used `False`; GPU controls used `True` on L4 | Partially followed |
| Evaluator class | `train_muzero` wires LightZero MuZero evaluator | Training path uses stock machinery; default eval wrapper is manual; later parity used `lzero.worker.MuZeroEvaluator` | Partially followed |
| Checkpoint format | LightZero `.pth.tar` | Native LightZero `.pth.tar`, mirrored without conversion | Followed |
| Pretrained config | Use matching checkpoint/config pair | OpenDILab HF pretrained checkpoint is older `96x96`/downsample surface and strict-loads fail against current `64x64` config | Not followed / unresolved |

## Specific Findings

## Extra-Blunt Setup Holes

These are the holes where we may simply have failed to follow upstream, not
discovered a mysterious MuZero learning pathology.

### 1. Wrong authority soup

We mixed at least three "official" surfaces:

- current upstream `main` Atari config: `4x64x64`, `atari_muzero_config.py`;
- pinned package: `LightZero==0.2.0`, same broad surface but
  `max_env_step=200000` instead of GitHub `500000`;
- OpenDILab Hugging Face pretrained card: older `4x96x96` checkpoint/config
  surface.

That is not one reproduction target. It is a triangle with a training run in
the middle. A faithful run must choose exactly one source surface and keep the
checkpoint/config/model/env tuple together.

Verdict: **setup hole**. Not fatal for from-scratch smoke tests; fatal for
pretrained-control claims.

### 1a. Wrong exact-upstream target

This deserves its own slap on the desk. We cannot say "exact upstream" while
using GitHub's `500000` number in prose, installing PyPI `LightZero==0.2.0`
whose captured surface says `200000`, and then actually running `4096`.

Those are three different targets:

```text
GitHub upstream today:        500000
installed LightZero==0.2.0:   200000
our staged control:             4096 or less in earlier runs
```

Pick one. Label it. Do not blend them into a single "official" claim.

Verdict: **not exact-upstream if mixed**.

### 2. Wrong config scale

The upstream Pong recipe is not "whatever config imports successfully." It is
the values in the config. We kept the name and gutted the budget:

```text
official: 500000 env steps, 50 sims, batch 256, collectors 8, evaluators 3
package:  200000 env steps, 50 sims, batch 256, collectors 8, evaluators 3
ours:        4 to 4096 env steps, 2 or 10 sims, batch 4/8/32, collectors 1/2, evaluator 1
```

That is not a near miss. That is a scale-model airplane made of the same paint.

Verdict: **not followed**.

### 3. Wrong `game_segment_length`

Current non-segment upstream uses `game_segment_length=400`. Segment upstream
uses `game_segment_length=20` with `train_muzero_segment` and `num_segments=8`.
We used `16` for most runs and `64` for the 4096/sim10 rung, while still using
the non-segment `train_muzero` path.

This is a particularly nasty deviation because it changes replay/target
geometry while looking like a harmless "make it cheaper" knob.

Verdict: **not followed**.

### 4. Wrong collector/evaluator shape

Upstream collects from `8` envs / `8` episodes and evaluates with `3` envs /
`3` episodes. We usually used `1` and `1`. The 4096/sim10 run moved to
`2` collectors, still nowhere near upstream.

One evaluator episode under a short cap is not an evaluation. It is a single
weather report.

Verdict: **not followed**.

### 5. Wrong update semantics

Upstream current Atari config uses:

```text
update_per_collect=None
replay_ratio=0.25
```

We usually forced:

```text
update_per_collect=1
```

and later:

```text
update_per_collect=2
```

That is not just "less training." It changes how LightZero decides learner work
from collected replay.

Verdict: **not followed**.

### 6. Wrong `max_train_iter` mental model

Upstream quick-start Pong calls `train_muzero(..., max_env_step=max_env_step)`.
Our wrapper adds `max_train_iter` and then we talk about "iteration_4" or
"iteration_8" as if that means comparable progress.

It does not. Several local records say the requested `max_train_iter` was not
reached because `max_env_step` stopped the run. So the command surface itself
encouraged false confidence: "I asked for 32 iterations" did not mean "I got 32
meaningful learner iterations."

Verdict: **not upstream semantics**.

### 7. Wrong `max_env_step` use

Using `max_env_step=4`, `128`, `512`, `1024`, `2048`, or `4096` is fine for an
import/plumbing smoke. It is not a LightZero Pong setup. Current GitHub
upstream says `500000`; the installed `LightZero==0.2.0` package says
`200000`. Our numbers match neither.

Also, pairing tiny `max_env_step` with explicit short episode caps means the
run is dominated by truncation and early-game states. It answers "can it run?",
not "did it train Pong?"

Verdict: **not followed for training**.

### 8. Wrong eval cap

The official config evaluates episodes. We evaluated fixed windows like
`64`, `128`, or `256` steps, often ending in `TimeLimit.truncated`. That is a
debug lens, not Pong evaluation.

The 256-step eval curve is useful for comparing local checkpoints under one
microscope. It is not the upstream evaluation contract and should not be read
as "Pong score."

Verdict: **not followed**.

### 9. Wrong eval harness by default

The default local eval wrapper manually:

- loads the checkpoint;
- creates one env;
- maintains a frame stack;
- calls `MuZeroPolicy.eval_mode.forward`.

Later, `--run-stock-evaluator` used `lzero.worker.MuZeroEvaluator` and matched
the manual action prefix for a matching 64x64 checkpoint. Good. But the default
eval path is still a bespoke probe, not the upstream evaluator loop.

Verdict: **partially followed after parity check; default path not upstream**.

### 10. Wrong checkpoint location, not format

Checkpoint *format* is fine: native LightZero `.pth.tar`.

Checkpoint *location* is not upstream: upstream writes under its `exp_name`
tree such as `data_muzero/...`. We rewrite `exp_name` to `/tmp/...`, scan it,
then mirror checkpoints into a CurvyZero Modal Volume manifest.

That is acceptable artifact plumbing. It is not exact upstream filesystem
layout. If a future loader assumes upstream relative paths, this can bite.

Verdict: **format followed; directory layout not followed**.

### 11. Seed/dynamic-seed ambiguity

`seed=0` matches upstream examples. But the local direct env/eval probes
explicitly use `dynamic_seed=False` in several places, while the training path
is delegated through LightZero's env manager.

I do not see this as the top failure. But it is a setup ambiguity: manual eval
may be more deterministic than the real upstream evaluator manager, and one
seed plus one evaluator episode is too thin to claim anything.

Verdict: **probably okay for smoke; insufficient for reproduction**.

### 12. ROM/env id

This was mostly done right. The env id is `PongNoFrameskip-v4`; AutoROM made
ALE reset/step pass. Earlier failures were plain missing `cv2` and missing ROM,
not algorithm bugs.

Verdict: **followed after the ROM fix**.

### 13. Frame size/model architecture

For current-from-scratch runs, `4x64x64` conv MuZero is correct for current
upstream `main`.

For the OpenDILab pretrained checkpoint, it is wrong. That checkpoint belongs
to an older `4x96x96`/different-downsample model surface. Strict load failed
for exactly that reason. Non-strict load or shape surgery would be fake
reproduction.

Verdict: **current scratch path followed; pretrained path not followed**.

## Minimal Exact Upstream Command

This is the smallest honest "do upstream" command. It deliberately avoids the
CurvyZero wrappers.

```sh
git clone https://github.com/opendilab/LightZero.git
cd LightZero
pip install -e .
pip install "AutoROM[accept-rom-license]" opencv-python-headless
AutoROM --accept-license
python3 -u zoo/atari/config/atari_muzero_config.py
```

This is exact for current GitHub upstream, whose Atari Pong budget is
`max_env_step=500000`.

If using the pinned package instead of a clone, then the target is **the package
version**, not current upstream `main`. For our current Modal image,
`LightZero==0.2.0` means the exact package surface is `max_env_step=200000`.
Record the exact installed `LightZero` and `DI-engine` versions and do not mix
in a checkpoint or budget from a different config surface.

Minimal exact Modal equivalent:

```text
image:
  install LightZero from the chosen source/version
  install Atari/OpenCV deps
  install/accept Atari ROMs

command:
  cd LightZero
  python3 -u zoo/atari/config/atari_muzero_config.py
```

No wrapper arguments. No patched caps. No `--max-env-step 4096`. No
`--batch-size 8`. No `--game-segment-length 16`. No local eval shortcut.

Minimal exact package command, if intentionally targeting installed
`LightZero==0.2.0`, is conceptually the same but must report the package
surface:

```text
install LightZero==0.2.0
import/run its zoo.atari.config.atari_muzero_config unchanged
use captured max_env_step=200000
```

Do not call that "current GitHub upstream." Call it "exact PyPI
LightZero==0.2.0 package surface."

## Faithful Setup Checklist

A run only deserves "followed upstream" if every box below is true:

- Source/version: one chosen LightZero source is used end to end; no current
  config with older pretrained checkpoint.
- Command: `python3 -u zoo/atari/config/atari_muzero_config.py`, or an exact
  programmatic equivalent with unchanged `main_config` and `create_config`.
- Env id: `PongNoFrameskip-v4`.
- ROM: ALE can reset/step Pong before training; ROM license handling is
  documented.
- Env type: `atari_lightzero`; env manager `subprocess`.
- Observation/model: `(4,64,64)` grayscale for current upstream config;
  `downsample=True`; conv MuZero; action space `6`.
- Collect: `collector_env_num=8`, `n_episode=8`.
- Eval: `evaluator_env_num=3`, `n_evaluator_episode=3`, stock
  `MuZeroEvaluator`, no artificial 64/128/256-step score cap.
- Search: `num_simulations=50`.
- Training budget: `max_env_step=500000` for current GitHub upstream, or
  `max_env_step=200000` if and only if the explicit target is installed
  `LightZero==0.2.0`.
- Updates: `update_per_collect=None`, `replay_ratio=0.25`.
- Batch/replay: `batch_size=256`, `replay_buffer_size=1000000`.
- Segment length: `game_segment_length=400` for non-segment config. If using
  the segment config instead, switch to `train_muzero_segment` and use its
  `num_segments=8`, `game_segment_length=20`, and warmup semantics.
- Checkpoints: native LightZero checkpoints under the LightZero `exp_name`
  tree; mirrors are allowed only as copies, not as the source of truth.
- Seed: seed is recorded; reproduction uses more than one eval episode and
  does not infer policy quality from one seed/window.
- Pretrained: if using OpenDILab HF `pytorch_model.bin`, use its matching
  `policy_config.py`/96x96 model surface. Do not load it into current 64x64
  config.

Anything else is a smoke, forked recipe, or controlled ablation. Useful, but
not upstream fidelity.

### Env, ROM, and preprocessing

This part is the strongest. Coach did move from a missing-ROM failure to a real
ALE-backed LightZero `PongNoFrameskip-v4` reset/step pass by adding explicit
AutoROM handling. The successful env smoke used `AtariEnvLightZero`, returned a
LightZero observation dict, and could step the env.

The preprocessing is close to current upstream: current upstream `main` and the
local train/eval path use `4x64x64` grayscale visual input with downsample
enabled. The one caveat is eval harnessing: raw env reset exposes one
preprocessed frame and the local eval wrapper manually maintains the four-frame
policy stack. The stock-evaluator parity rerun later confirmed the manual path
matches `MuZeroEvaluator` actions for the matching 64x64 checkpoint, so this is
not the main fidelity problem.

### Training scale

This is where fidelity breaks. Upstream Atari Pong is a `500,000` env-step,
`50` simulation, `256` batch, `8` collector env, `3` evaluator env setup. The
local runs are tiny controls. Even the largest staged run, 4096/sim10, used:

```text
max_env_step=4096
max_train_iter=32
collector_env_num=2
evaluator_env_num=1
num_simulations=10
batch_size=32
update_per_collect=2
max_episode_steps=512
game_segment_length=64
```

That is still under 1% of upstream env steps and only 20% of upstream MCTS
simulations. The smaller GPU512/GPU1024/GPU2048 runs used `num_simulations=2`,
`batch_size=8`, one collector, one evaluator, and short eval caps. Those are
smoke settings, not upstream Pong settings.

### Replay and updates

The replay buffer capacity mostly stayed upstream-shaped at `1,000,000`, but
the populated replay distribution did not. A 512, 1024, 2048, or 4096 env-step
run cannot behave like the upstream replay regime, regardless of the configured
capacity.

The `update_per_collect` change is a real semantic deviation. Upstream current
Atari config uses `update_per_collect=None` and `replay_ratio=0.25`. Local runs
usually forced `update_per_collect=1`, and 4096/sim10 forced `2`. That changes
the learner/collector ratio from the official recipe.

### Evaluator

Training through `train_muzero` is the official entry, so the training loop is
using stock LightZero wiring. The eval-only wrapper, however, is custom by
default: it loads the checkpoint, creates one env, maintains a frame stack, and
calls `MuZeroPolicy.eval_mode.forward` directly. That is useful and was later
checked against `lzero.worker.MuZeroEvaluator`, but it is not the plain
upstream eval pattern unless `--run-stock-evaluator` is used.

### Checkpoint format

This was followed. Local runs preserved native LightZero checkpoint files:
`ckpt_best.pth.tar`, `iteration_0.pth.tar`, `iteration_1.pth.tar`, and later
iteration checkpoints. The wrapper mirrors them but does not convert them.

### Pretrained mismatch

The OpenDILab Hugging Face model card is an upstream pretrained reference for
`PongNoFrameskip-v4-MuZero`. It reports `500,000` training steps and mean reward
`20.4 +/- 0.49`, with the older config surface showing `obs_shape` /
`observation_shape` as `4x96x96`, `batch_size=256`, `num_simulations=50`,
`game_segment_length=400`, and `replay_buffer_size=1,000,000`.

Primary source: https://huggingface.co/OpenDILabCommunity/PongNoFrameskip-v4-MuZero

Local strict-load probing found that this pretrained checkpoint does not fit
the current local/current-upstream 64x64 stock config. The blocker is model
surface drift, including unexpected downsample keys and FC/projection shape
mismatches. So Coach did not establish a faithful pretrained official Pong
control; the honest choices are either train closer to the current official
recipe from scratch, or use the matching older 96x96 config with that
pretrained checkpoint.

## Bottom Line

Coach followed the official *identity* of the problem: official ALE Pong,
official LightZero Atari env, official MuZero policy family, official current
64x64 visual stack, and official checkpoint format.

Coach did not follow the official *training setup* closely enough to call the
runs upstream-intended Pong training. The setup is best labeled:

```text
stock LightZero Atari Pong infrastructure smoke/control, heavily downscaled
```

It should not be labeled:

```text
official LightZero Atari Pong reproduction
official LightZero Atari Pong training setup
pretrained OpenDILab Pong evaluation
```

Final verdict: **partially followed**, and **not followed exactly**.
