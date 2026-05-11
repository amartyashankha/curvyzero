# MuZero Reference Examples

Research snapshot: 2026-05-09.

Role in current hierarchy: LightZero is no longer only a stock-reference lane.
It is the current implementation direction for the next real dummy Pong MuZero
attempt. Use stock CartPole as the already-proven sanity reference, then build
the custom dummy Pong LightZero adapter smokes. Mctx is fallback/comparison.

## Current Truth / No More Pretending

- We validated stock LightZero CartPole MuZero progression.
- We validated an Mctx search benchmark.
- We validated a CEM-v2 Pong baseline.
- We validated a raster-only MLP Pong baseline.
- We have not run an actual LightZero custom dummy Pong MuZero trainer yet.
- We have not run an actual project-owned MuZero/Mctx train loop for Pong or
  Curvy; that is now fallback/comparison, not the immediate lane.
- CEM-v2 and the MLP are baselines and scaffolding only. They are not MuZero
  progress.
- The next main lane is LightZero custom dummy Pong MuZero: config/import smoke
  first, tiny trainer smoke second.

Prevention rules:

- Prove the target is scoreable before scaling it.
- Keep baselines separate from MuZero.
- Name the algorithm in every experiment title, command, and summary.
- Distinguish stock LightZero MuZero, LightZero custom-env MuZero, and fallback
  project-owned MuZero/Mctx.
- Do not describe CEM, imitation, or MLP results as MuZero progress.

## Short Answer

LightZero is the next implementation lane because it is the only external
repository that we have already proven can run an actual MuZero trainer in
Modal. That proof is stock LightZero CartPole MuZero, not CurvyZero Pong yet.
The next step is to adapt dummy Pong to LightZero's custom environment
interface and call LightZero's trainer under brutal caps.

Mctx stays useful, but only as fallback/comparison right now. It gives search,
not replay, trainer, checkpointing, or evaluator machinery. Do not start by
writing those pieces ourselves while the LightZero custom-env path is untested.

The closest stock reference for Atari-like visual MuZero remains LightZero's
Atari MuZero segment config:

```bash
cd LightZero
python3 -u zoo/atari/config/atari_muzero_segment_config.py --env PongNoFrameskip-v4 --seed 0
```

That is a stock reference, not the next implementation command. CurvyZero's
first main command should be the custom dummy Pong config/import smoke:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0
```

Then run the tiny trainer smoke:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0 \
  --opponent-policy random_uniform \
  --max-env-step 64 \
  --max-train-iter 2 \
  --num-simulations 2 \
  --batch-size 8 \
  --update-per-collect 1 \
  --n-evaluator-episode 1
```

For the cheapest stock LightZero sanity path after installing LightZero, use
CartPole MuZero before Pong:

```bash
cd LightZero
python3 -u zoo/classic_control/cartpole/config/cartpole_muzero_config.py
```

Even that is a training run unless patched or stopped early. The first
CurvyZero replication step is therefore an import/config smoke, not a trainer:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dependency_smoke
```

That smoke installs pinned `LightZero==0.2.0` in a CPU Modal image, imports
LightZero/DI-engine/PyTorch, imports the stock CartPole and Atari Pong config
modules, checks CartPole's top-level MuZero config, and captures the Pong config
by monkeypatching `lzero.entry.train_muzero_segment` before calling the stock
config builder. It does not start training.

Current answer after the 2026-05-09 stock lane: yes, we validated stock
LightZero CartPole MuZero progression as an external reference. That is not a
project-owned MuZero/Mctx Pong or Curvy train loop. Stock Pong has now gone one
step past dry config, and that step is negative: the current Modal image cannot
create/reset/step stock Atari Pong yet.

The meaningful cheap trainer command is now:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke \
  --mode progression
```

This CPU Modal run patches the installed stock CartPole config to one collector
env, one evaluator env, one evaluator episode, `num_simulations: 5`,
`batch_size: 16`, `update_per_collect: 4`, `eval_freq: 1`, `max_train_iter: 4`,
and `max_env_step: 128`. The 2026-05-09 run returned `ok: true`, wrote
LightZero evaluator/learner logs, wrote a TensorBoard event file, saved
`ckpt_best.pth.tar`, `iteration_0.pth.tar`, and `iteration_4.pth.tar`, and
reported `reward_mean: 33.0`, `eval_episode_return_mean: 33.0`,
`total_loss_avg: 45.577473`, and `policy_loss_avg: 3.855631`.

Interpretation: stock CartPole MuZero is validated as an existing-example lane
with a real trainer progression signal. It is still not a quality or
convergence claim, and it is not project-owned Pong/Curvy MuZero progress.
Stock Pong remains unvalidated for training; the latest Pong env smoke reaches
the Atari ROM gate after OpenCV prep but still cannot reset/step.

The latest stock Pong dependency command is:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_env_smoke
```

Run result on 2026-05-09 after adding `opencv-python-headless==4.11.0.86`:
`ok: false`. The LightZero/DI-engine Atari wrapper imports `cv2` successfully
and reaches `AtariEnvLightZero.reset()`, then fails because the Pong ROM is
missing. Plain Gym/Gymnasium diagnostics fail at the same ALE ROM gate. The
smoke does not install AutoROM or accept the ROM license automatically.
Therefore the recommendation is explicit ROM approval/prep first, not stock
Pong training.

Decision for the next stock-reference MuZero smoke: rerun or slightly extend
the stock LightZero CartPole progression lane. Do not use CEM or supervised
dummy Pong MLP as a MuZero substitute.

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke \
  --mode progression
```

Reason: this is the smallest stock command that already calls a real MuZero
trainer and produces evaluator/learner/checkpoint signals. Unblocking stock
Pong first would require explicit Atari ROM license approval before any trainer
signal can exist. The main custom-env lane is different: adapt dummy Pong into
LightZero and call LightZero's trainer. Keep project-owned JAX/Mctx as fallback
if LightZero cannot preserve telemetry, traces, or artifacts.

Hard line:

- Actual MuZero already run here: stock LightZero CartPole on Modal.
- Actual stock Pong MuZero: not run; blocked before reset/step by Atari ROM
  setup.
- Project-owned MuZero/Mctx trainer: not built yet; current Mctx work is search
  and benchmark only.
- CEM, supervised raster MLP, imitation, value-only, and dummy self-play:
  useful baselines or scaffolds only, not MuZero.

Next decision:

- If the goal is to prove an external MuZero trainer still works, rerun the
  LightZero CartPole progression command above.
- If the goal is useful CurvyZero progress, start the LightZero dummy Pong
  custom-env config smoke, then the tiny LightZero dummy Pong trainer smoke.
- If the goal is stock visual Pong, first approve and prepare Atari ROM
  handling, then rerun `lightzero_pong_env_smoke`; do not start
  `train_muzero_segment` until reset and step pass.

Exact Pong unblocker after license approval: create a ROM-enabled Modal image
variant with `uv_pip_install("AutoROM[accept-rom-license]")` and
`run_commands("AutoROM --accept-license")`, then rerun
`uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_env_smoke`.
Do not call `lzero.entry.train_muzero_segment` until that env smoke can
create/reset/step `PongNoFrameskip-v4`.

## Why LightZero Was First

LightZero won the first-repository slot for practical reasons, not because it
is the perfect CurvyZero backbone:

- It is a real MuZero-family training framework, not just pseudocode or search.
- Its README names stock MuZero commands for CartPole and Pong.
- Its Atari config is visual and discrete-action, which is the closest stock
  shape to a Pong-like raster task.
- Its PyPI package has a CPython 3.11 Linux wheel, so a Modal smoke can install
  it without cloning or compiling a checkout.
- We already have repo-local Modal smokes around it:
  `lightzero_dependency_smoke`, `lightzero_cartpole_tiny_train_smoke`,
  `lightzero_pong_dry_config_smoke`, and `lightzero_pong_env_smoke`.
- The CartPole progression smoke returned a real `MuZeroPolicy`, evaluator and
  learner metrics, TensorBoard events, and checkpoints.

The tradeoff is just as important: LightZero is DI-engine/PyTorch shaped and
single-agent or alternating-game shaped. CurvyZero is moving toward
simultaneous ego-perspective self-play. The next step is to test that mismatch
directly with a dummy Pong ego-view wrapper instead of writing our own trainer
first.

## Is It Still The Best Near-Term Choice?

Yes. It is now the next practical custom-env path.

Use LightZero when the question is: "Can we run a real MuZero trainer on a
CurvyZero-shaped custom env?" The next steps are a dummy Pong custom-env config
smoke and then a tiny LightZero dummy Pong trainer smoke. Keep CartPole
progression as the known-good external rerun.

Use project-owned Mctx only as fallback/comparison if LightZero cannot preserve
the custom env contract, scorecard telemetry, or artifact refs.

Do not use dummy Pong CEM or the raster MLP to answer either question. They are
baselines. They may tell us whether the target is scoreable, but they do not
exercise learned dynamics plus MCTS.

## Repository Fit

| Candidate | Actual MuZero training | Atari/Pong support | Custom visual envs | Modal install friction | Checkpointing | Simplicity |
| --- | --- | --- | --- | --- | --- | --- |
| LightZero | Strongest external fit. We ran stock CartPole MuZero progression in Modal. | Strong and now proven locally/Modal for the installed package: stock Atari MuZero uses `PongNoFrameskip-v4`, and later work got the ALE-backed Pong control path running. | Medium. Gym/DI-engine style is workable for an ego wrapper, but simultaneous multiplayer is not turnkey. | Medium. `LightZero==0.2.0` installs from wheel, but pulls a large PyTorch/DI-engine stack; Atari also needs OpenCV and ROM prep. | Good for stock runs: logs, TensorBoard events, and `.pth.tar` checkpoints already appeared. | Medium. Easy to smoke, harder to bend cleanly into CurvyZero semantics. |
| Project-owned Mctx loop | Not built yet. Mctx supplies MuZero/Gumbel MuZero search, not replay/training/checkpoint code. | None built in. We would connect it to dummy Pong or CurvyZero ourselves. | Best control. We own observations, action masks, rewards, simultaneous-action handling, replay rows, and artifacts. | Medium. JAX/Mctx GPU smokes passed, but a trainer still needs our code and JAX checkpoint choices. | Not automatic. Must be designed into our run-management layout. | Medium after built; simplest long-term shape for our needs, but more initial code. |
| muzero-general | Real educational MuZero implementation. | Has Gym and Atari examples, but the listed Atari example is Breakout, not Pong. | Medium for simple envs; less attractive for our visual/simultaneous case. | Higher. Ray, older dependency style, and nonbatched MCTS make it awkward for a small Modal lane. | Has TensorBoard and automatic model checkpoints. | Good for reading; risky as a backbone. |
| Muax | Provides Mctx-based MuZero helpers for gym-style environments. | CartPole/LunarLander examples, not an Atari/Pong-first repo. | Medium if the env is gym-like; weak fit for our simultaneous multi-agent path. | Medium to high. It brings JAX/Haiku/Optax assumptions and has older optional dependency notes. | Has model save/load examples, but not our artifact contract. | Simple reference for Mctx trainer pieces, not a proven CurvyZero backbone. |
| EfficientZero repo | Real EfficientZero training code, which is MuZero-family but not the plainest first MuZero baseline. | Strong Atari focus, usually Breakout examples. Pong may be possible by env name, but it is not our cheapest proof. | Lower near-term fit. New env registration means config, model, and wrapper work. | High. Requires C++/Cython tree build, GCC, Ray, and a heavier multi-worker setup. | Built for serious Atari experiments, but with its own layout. | Low for our first smoke; useful later if sample efficiency becomes the main question. |

Short read:

- LightZero is the first external repo because it is the only one with a
  proven Modal trainer smoke plus a stock Pong-shaped config.
- Mctx is the best near-term implementation dependency for CurvyZero because it
  is small search code, not a training framework that wants to own the system.
- muzero-general and Muax are references to read for trainer/replay shapes.
- EfficientZero is a later Atari/sample-efficiency comparison, not the first
  CurvyZero integration.

## Local Inspection

Commands run locally:

```bash
rg --files -g '*{muzero,Muzero,MuZero,mewzero,MewZero,lightzero,LightZero,pong,Pong,atari,Atari}*' -g '!node_modules' -g '!**/.git/**'
rg -n "MuZero|Muzero|muzero|MewZero|mewzero|LightZero|lightzero|Atari|Pong|pong|self-play|self play" docs src scripts modal-projects examples . -g '!node_modules' -g '!**/.git/**'
rg -n "mctx|LightZero|lzero|muzero-general|muax|MewZero|Mew|Atari|Pong" pyproject.toml uv.lock README.md docs/research docs/design docs/decisions docs/runbooks docs/experiments docs/working third_party -g '!**/.git/**'
find . -maxdepth 4 -type d -iname '*lightzero*' -o -type d -iname '*muzero*' -o -type d -iname '*mewzero*' -o -type d -iname '*mctx*'
uv run python -c 'import importlib.util, json; names=["mctx","lzero","ding","muzero","jax","torch","gymnasium","ale_py"]; print(json.dumps({name: bool(importlib.util.find_spec(name)) for name in names}, sort_keys=True))'
```

Important outputs:

```text
rg: modal-projects: No such file or directory (os error 2)
rg: examples: No such file or directory (os error 2)
```

```json
{"ale_py": false, "ding": false, "gymnasium": false, "jax": false, "lzero": false, "mctx": false, "muzero": false, "torch": false}
```

Findings:

- No local `modal-projects/` or `examples/` directory exists in this checkout.
- `third_party/` only contains the CurvyTron reference clone.
- `pyproject.toml` depends on `numpy` only, with optional `pytest`, `ruff`, and
  `modal`; `uv.lock` has no LightZero/Mctx/JAX/PyTorch stack.
- Local Pong files under `src/curvyzero/training/` are dummy Pong replay,
  imitation, value, eval, and self-play scaffolds. They are not a stock MuZero
  implementation.
- Local docs already critique LightZero and Mctx, and now record stock
  LightZero import/config, CartPole trainer, CartPole progression, and Pong
  dry-config smoke results.

No pytest was run. No GPU work was run for the LightZero lane. LightZero
training was run only inside capped CPU Modal CartPole smokes.

## What The Reference Proves

LightZero proves that a maintained PyTorch MuZero-family toolkit has a stock
Atari/Pong path, including image-shaped observations, MCTS, replay, and a
training entrypoint. The current LightZero README's quick start names
`zoo/atari/config/atari_muzero_segment_config.py` as the MuZero Pong command.
The same README says LightZero supports MuZero, EfficientZero, Gumbel MuZero,
Stochastic MuZero, UniZero, and related MCTS/RL algorithms, with Atari listed
as a supported environment class.

That does not prove CurvyZero's architecture. It only gives us a sane external
baseline shape to compare against.

The stock Pong config is heavyweight for this lane. Its upstream defaults use
multiple collector/evaluator envs, a convolutional model on stacked frames, 50
MCTS simulations, batch size 256, CUDA enabled, replay, and up to 500k env
steps. That is not a cheap local smoke unless we make a separate throwaway
LightZero checkout and intentionally patch the config down.

## Visual Atari/Pong Fit

This is the right visual reference for Pong-like work:

- ALE Pong exposes image observations and a 6-action discrete control space.
- LightZero's current GitHub `main` Atari config uses observation shape
  `(4, 64, 64)`, grayscale frame stacking, a convolutional model, and
  `PongNoFrameskip-v4` as its default `--env`.
- The pinned PyPI `LightZero==0.2.0` config captured by the Modal smoke uses
  observation shape `(4, 96, 96)` for `PongNoFrameskip-v4`. Treat this as
  upstream version drift, not as a CurvyZero env fact.
- The OpenDILab Hugging Face model card for `PongNoFrameskip-v4-MuZero`
  describes a trained LightZero/DI-engine MuZero policy for Atari Pong and
  records a 6-action, frame-stacked visual config.

CurvyZero's dummy Pong raster path is directionally similar because it is visual
and discrete-action, but it is much smaller and project-owned. It should be
used as a learnability toy, not represented as an Atari-equivalent benchmark.

## Multiplayer And Self-Play

For our purposes, LightZero should be treated as single-agent or alternating
board-game shaped unless a wrapper smoke proves otherwise.

- ALE single-agent Pong controls the right paddle against the built-in computer
  opponent.
- ALE/PettingZoo multi-agent Pong exists and uses two agents with parallel API
  support, image observations, and 6 minimal actions per agent. It is a useful
  source of Pong semantics.
- LightZero's documented custom-env path and Atari config are not a turnkey
  simultaneous two-player CurvyTron self-play system.

Blunt read: if CurvyZero needs simultaneous multi-agent self-play, stock
LightZero Pong is a reference, not the answer. It can validate "what does a
real visual MuZero Atari stack look like?" It cannot validate our current Pong
learner design by itself.

## What We Can Run Now

This local dependency smoke can run in the current repo:

```bash
uv run python -c 'import importlib.util, json; names=["mctx","lzero","ding","muzero","jax","torch","gymnasium","ale_py"]; print(json.dumps({name: bool(importlib.util.find_spec(name)) for name in names}, sort_keys=True))'
```

Output:

```json
{"ale_py": false, "ding": false, "gymnasium": false, "jax": false, "lzero": false, "mctx": false, "muzero": false, "torch": false}
```

Conclusion: no local stock MuZero/MewZero/LightZero example is runnable in the
repo environment today. The honest next command is not a trainer. It is a
contained Modal dependency/config smoke.

## First Contained Modal Smokes

Already completed path: Mctx/JAX. ADR-0004 picks Mctx first, and Mctx proved
the missing search runtime directly with a tiny synthetic `gumbel_muzero_policy`
call. The smoke does not import the simulator, replay, or trainer, and it does
not run Pong or CartPole training.

Implementation:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_dependency_smoke --kind cpu
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_gpu_dependency_smoke
```

The CPU image pins `mctx==0.0.6`, `jax==0.7.0`, and `jaxlib==0.7.0`. The GPU
image pins `mctx==0.0.6` with `jax[cuda12]==0.7.0`, requests `gpu=["L4", "T4"]`,
and runs a tiny fixed profile: `B=4`, `A=3`, hidden dim 8, 4 simulations, max
depth 4. Treat `ok: true`, a GPU JAX backend for the GPU module, finite
`action_weights`, and row sums near 1.0 as pass criteria.

Expected risk: the first run may spend a few minutes building the Modal image
and downloading JAX wheels. The actual remote search is tiny and should finish
in seconds once the image is built. L4/T4 keeps this as a cheap dependency
smoke, not a benchmark. Do not scale batch size, simulation count, or GPU class
from this module.

Result on 2026-05-09: both smokes passed. CPU ran with JAX backend `cpu`; GPU
ran on Modal `NVIDIA L4` with JAX backend `gpu` and device `cuda:0`. See
`docs/experiments/2026-05-09-modal-mctx-dependency-smoke.md`.

LightZero status: LightZero is pip-installable as `pip install LightZero`
according to its official installation docs, and the same docs also support a
GitHub editable install for the latest development checkout. PyPI currently
has `LightZero==0.2.0` wheels for CPython 3.11 on Linux, which is enough for a
CPU Modal import/config smoke.

Chosen stock replication path:

1. `lightzero_dependency_smoke`: CPU Modal import/config check only.
2. `lightzero_cartpole_tiny_train_smoke`: separate stock CartPole MuZero path
   with a default dry/config-patch mode. It copies the installed stock config,
   patches it to CPU, one collector/evaluator env, one evaluation episode, one
   collected episode, 2 MCTS simulations, batch size 4, `max_train_iter=1`, and
   `max_env_step=4`, then reports the exact trainer call surface. The opt-in
   `--mode train` path calls LightZero's own `lzero.entry.train_muzero` under
   those caps.
3. `lightzero_cartpole_tiny_train_smoke --mode progression`: same stock
   CartPole config, still CPU and one env, but with `num_simulations=5`,
   `batch_size=16`, `update_per_collect=4`, `eval_freq=1`,
   `max_train_iter=4`, and `max_env_step=128`. This is the first real progress
   signal lane because it returns parsed evaluator metrics, learner metrics,
   checkpoint-save signals, and remote artifact inventory.
4. Only after CartPole is controlled, repeat the same capped approach for the
   Atari Pong config.

Upstream knobs to keep small for fast examples: LightZero's config guide calls
out `collector_env_num`, `num_simulations`, `update_per_collect`, `batch_size`,
and `max_env_step` as frequently changed parameters that affect performance and
training speed. It also warns to choose parallel environment counts according
to available compute, and recommends tensorboard/log monitoring during
training. That matches the Modal patches here: CPU, one collector env, one
evaluator env, low simulation count, small batch, low update count, explicit
`max_train_iter`, explicit `max_env_step`, and returned logs/artifacts.

The current contained LightZero smoke:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dependency_smoke
```

Pass criteria:

- returned JSON has `ok: true`;
- `packages.LightZero`, `packages.DI-engine`, `packages.torch`, and
  `packages.easydict` are not `missing`;
- `imports.lzero`, `imports.ding`, `imports.torch`, and `imports.easydict` are
  `ok`;
- CartPole reports `env_id: CartPole-v0`, `policy_type: muzero`,
  `env_type: cartpole_lightzero`, `model_type: mlp`, `action_space_size: 2`;
- Pong reports `call_policy: trainer_entrypoint_monkeypatched_to_capture_config`,
  `env_id: PongNoFrameskip-v4`, `policy_type: muzero`,
  `env_type: atari_lightzero`, `model_type: conv`, `action_space_size: 6`,
  `captured_max_env_step: 500000`, `num_simulations: 50`, and
  `batch_size: 256`.

Fail interpretation:

- dependency/import failure means the PyTorch/DI-engine stack needs pinning or
  a GitHub editable checkout before any stock trainer run;
- CartPole config mismatch means do not run a capped trainer until the installed
  stock config is understood;
- Pong token/action-space mismatch means the visual reference moved upstream and
  the docs/patch plan must be refreshed.

This is deliberately weaker than training, but it is the smallest honest stock
MuZero example replication step that avoids starting an uncontrolled trainer.

Run result, 2026-05-09:

```text
ok: true
packages: LightZero 0.2.0, DI-engine 0.5.3, torch 2.11.0, easydict 1.13
CartPole: CartPole-v0, muzero, cartpole_lightzero, mlp, action_space_size 2,
  collector_env_num 8, evaluator_env_num 3, num_simulations 25,
  batch_size 256, cuda true, max_env_step 100000
Pong: PongNoFrameskip-v4, muzero, atari_lightzero, conv, action_space_size 6,
  observation_shape [4, 96, 96], collector_env_num 8, evaluator_env_num 3,
  num_simulations 50, batch_size 256, cuda true, captured_max_env_step 500000
```

The first image build downloaded the full PyTorch/CUDA dependency stack even for
this CPU smoke, so expect a large first build. Repeated runs reused the image
and took about a minute remotely, mostly import/config overhead and LightZero
warnings.

Next contained stock CartPole smoke:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke
```

This default command is still dry. It must return `ok: true`, `mode: dry`,
`call_policy: dry_config_patch_only`, stock task `CartPole-v0`, algorithm
`MuZero`, trainer entrypoint `lzero.entry.train_muzero`, and patched caps no
larger than one env, one evaluation episode, one collected episode, two
simulations, batch size 4, `max_train_iter=1`, and `max_env_step=4`.

The real tiny trainer command is explicit:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke \
  --mode train
```

Treat a successful `--mode train` result as proof that the stock LightZero
CartPole MuZero trainer entrypoint can start and stop under brutal caps. Do not
read it as policy quality. If it fails or times out, do not increase caps until
the failure is diagnosed.

Run result, 2026-05-09:

```text
dry smoke: ok true, remote_elapsed_sec 12.847235
train smoke: ok true, remote_elapsed_sec 13.269896
train_result: ok true, return_type MuZeroPolicy, elapsed_sec 4.356072
packages: LightZero 0.2.0, DI-engine 0.5.3, torch 2.11.0, easydict 1.13
patched CartPole: collector_env_num 1, evaluator_env_num 1,
  n_evaluator_episode 1, n_episode 1, num_simulations 2, batch_size 4,
  update_per_collect 1, cuda false, max_train_iter 1, max_env_step 4
```

The real tiny trainer completed one initial evaluation episode with logged
`envstep_count 9.0`, then one learner update, and returned a `MuZeroPolicy`.
That is acceptable for the smoke because CartPole episodes are tiny and the
trainer still stopped under the explicit caps and Modal timeout. It is not a
learning claim.

Progression command:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke \
  --mode progression
```

Run result, 2026-05-09:

```text
ok: true, remote_elapsed_sec 15.603642
train_result: ok true, return_type MuZeroPolicy, elapsed_sec 5.651327
patched CartPole: collector_env_num 1, evaluator_env_num 1,
  n_evaluator_episode 1, n_episode 1, num_simulations 5, batch_size 16,
  update_per_collect 4, eval_freq 1, cuda false, max_train_iter 4,
  max_env_step 128
signals: final_rewards [33.0], max_checkpoint_iteration 4,
  reward_mean 33.0, eval_episode_return_mean 33.0,
  total_loss_avg 45.577473, policy_loss_avg 3.855631,
  value_loss_avg 38.391567
artifacts: ckpt_best.pth.tar, iteration_0.pth.tar, iteration_4.pth.tar,
  evaluator_logger.txt, learner_logger.txt, events.out.tfevents.*
```

This is now the concrete existing-example validation result. It still does not
prove policy quality or CartPole convergence, but it does prove that a stock
MuZero trainer can run for a tiny capped progression and emit sane metrics
inside the current Modal setup.

Next stock visual reference step: dry LightZero Pong config only.

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_dry_config_smoke
```

This separate module imports the stock
`zoo.atari.config.atari_muzero_segment_config`, monkeypatches
`lzero.entry.train_muzero_segment`, calls the stock config builder for
`PongNoFrameskip-v4`, and patches the captured config surface to CPU/tiny caps.
It does not instantiate ALE/Gym/EnvPool, does not require Atari ROMs, and does
not call the trainer.

Run result, 2026-05-09:

```text
ok: true, remote_elapsed_sec 12.235089
packages: LightZero 0.2.0, DI-engine 0.5.3, torch 2.11.0, easydict 1.13
original Pong: PongNoFrameskip-v4, muzero, atari_lightzero, conv,
  observation_shape [4, 96, 96], action_space_size 6,
  collector_env_num 8, evaluator_env_num 3, n_evaluator_episode 3,
  num_simulations 50, batch_size 256, cuda true, max_env_step 500000
patched Pong: collector_env_num 1, evaluator_env_num 1,
  n_evaluator_episode 1, num_simulations 2, batch_size 4,
  update_per_collect 1, cuda false, max_env_step 4
trainer_entrypoint: lzero.entry.train_muzero_segment
train_result: null
```

The import emitted Gym and optional dependency warnings, but no failures. Treat
that as a useful config capture, not as proof that real Atari training is cheap.
The real dependency trap remains the next layer: ALE/Gym/EnvPool environment
creation, Atari ROM availability, replay, and the stock visual trainer loop.

Next env-creation smoke:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_env_smoke
```

Run result, 2026-05-09:

```text
ok: false, env_ok: false, lightzero_path_ok: false
packages: LightZero 0.2.0, DI-engine 0.5.3, gym 0.25.1,
  gymnasium 0.28.0, ale-py 0.8.1,
  opencv-python-headless 4.11.0.86, AutoROM missing, envpool missing
imports: cv2 ok
LightZero/DI-engine failure: reset reaches ALE, then Pong ROM is missing
Gym/Gymnasium fallback failure: unable to find the game "Pong"; ROM missing
```

Interpretation: the stock Pong env cannot reset/step in the current Modal
image. This is useful negative evidence. OpenCV is prepared; explicit ROM
management remains before any stock Pong trainer.

## Blocked Or Remaining

- Need explicit Atari ROM prep before any real `train_muzero_segment` call for
  Pong. The env smoke now imports `cv2`, then fails at the Pong ROM gate. Plain
  Gym/Gymnasium also report missing Pong ROMs.
- Need to avoid copying LightZero architecture into CurvyZero until a tiny
  external run has been measured beyond import/config capture.

## Recommendation

Use LightZero's Atari MuZero Pong config as the visual reference and CartPole
MuZero as the validated stock progression reference. Do not treat the current
dummy Pong self-play trainer, CEM-v2, or raster-only MLP as a stock or
project-owned MuZero path. They are useful baselines and scaffolds, but the
present project-owned loop has not run MuZero/Mctx training.

The stock-reference lane now has enough signal for a main-thread decision:

1. Keep the contained LightZero import/config, CartPole tiny-train, and Pong
   dry/env smokes as reference checks outside core code.
2. Pause before any real stock LightZero Pong trainer.
3. Prepare ALE ROM handling only after explicit license approval, then rerun
   the Pong env smoke until LightZero/DI-engine create/reset/step succeeds.
4. Compare the captured stock control surfaces against CurvyZero's dummy Pong
   docs before adding more local training architecture.

After the 2026-05-09 Pong env smoke, the recommendation is ROM approval/prep
first. The current image cannot reset the stock Pong env, so a trainer would
fail before producing a meaningful MuZero signal.

## Sources

- LightZero GitHub README and quick start:
  https://github.com/opendilab/LightZero
- LightZero Atari MuZero config:
  https://github.com/opendilab/LightZero/blob/main/zoo/atari/config/atari_muzero_segment_config.py
- LightZero CartPole MuZero config:
  https://github.com/opendilab/LightZero/blob/main/zoo/classic_control/cartpole/config/cartpole_muzero_config.py
- LightZero installation and quick start:
  https://opendilab.github.io/LightZero/tutorials/installation/installation_and_quickstart.html
- LightZero configuration guide:
  https://opendilab.github.io/LightZero/tutorials/config/config.html
- LightZero PyPI release:
  https://pypi.org/project/LightZero/
- Mctx GitHub README and quick start:
  https://github.com/google-deepmind/mctx
- muzero-general GitHub README:
  https://github.com/werner-duvaud/muzero-general
- Muax GitHub README:
  https://github.com/bwfbowen/muax
- EfficientZero GitHub README:
  https://github.com/YeWR/EfficientZero
- OpenDILab PongNoFrameskip-v4 MuZero model card:
  https://huggingface.co/OpenDILabCommunity/PongNoFrameskip-v4-MuZero
- ALE single-agent Pong docs:
  https://ale.farama.org/environments/pong/
- ALE/PettingZoo multi-agent Pong docs:
  https://ale.farama.org/multi-agent-environments/pong/
- MuZero Nature paper:
  https://www.nature.com/articles/s41586-020-03051-4
