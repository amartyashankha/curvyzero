# Non-LightZero Control Scout - 2026-05-11

Scope: quick scout while LightZero Pong runs. Do not edit or revert other
agents' work. Goal is one credible non-LightZero MuZero/AlphaZero control that
can launch quickly, or a clear reason not to launch it yet.

## Short Answer

Update from replication-control worker at 2026-05-11 11:55 EDT:

One distinct outside-LightZero AlphaZero training control is now completed:
OpenSpiel AlphaZero TicTacToe, capped to one learner step on Modal CPU. This is
not MuZero and not visual Pong, but it is a published DeepMind/OpenSpiel
AlphaZero-style actor/replay/learner/checkpoint control. It wrote
`checkpoint-0`, `checkpoint-1`, `config.json`, `learner.jsonl`, and actor/learner
logs to the Modal Volume.

Command:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.openspiel_alphazero_quick_control \
  --run-id openspiel-alphazero-tictactoe-20260511-s300c \
  --attempt-id train-tictactoe-1step-mlp-cpu
```

Artifact refs:

```text
training/openspiel-alphazero-tictactoe/openspiel-alphazero-tictactoe-20260511-s300c/attempts/train-tictactoe-1step-mlp-cpu/train/summary.json
training/openspiel-alphazero-tictactoe/openspiel-alphazero-tictactoe-20260511-s300c/attempts/train-tictactoe-1step-mlp-cpu/train/openspiel_alpha_zero
```

Result facts:

```text
ok: true
elapsed_sec: 9.099800607
packages: open_spiel==1.6.13, jax==0.10.0, jaxlib==0.10.0, flax==0.12.7, optax==0.2.8, chex==0.1.91, numpy==2.4.4
surface: tic_tac_toe, terminal rewards, sequential dynamics, 2 players, observation [3,3,3], 9 actions
control cap: max_steps=1, actors=1, evaluators=0, max_simulations=2, replay_buffer_size=8, train_batch_size=2, mlp width=16 depth=1
learner row: step=1, total_states=15, total_trajectories=2, game_length_avg=7.5, loss_sum=1.8990381956100464
checkpoint evidence: checkpoint-0 and checkpoint-1 directories present
```

Two non-result attempts preceded it:

- `openspiel-alphazero-tictactoe-20260511-s300`: Modal image snapshot failed
  because another repo file changed during build. Not an OpenSpiel result.
- `openspiel-alphazero-tictactoe-20260511-s300b`: Python 3.10 image imported
  repo helper code that requires `datetime.UTC` from Python 3.11. Not an
  OpenSpiel result; fixed by switching the wrapper to Python 3.11.

MiniZero is the strongest non-LightZero external replication candidate, but not
a same-turn quick launch. Its official path expects Linux, Docker or podman, an
NVIDIA GPU, and its server/self-play/optimizer/storage runtime. That is credible
for a later Modal container build, not a tiny safe smoke.

`muzero-general` is lighter and was cloned to:

```text
/tmp/curvyzero-nonlightzero-muzero-general
```

Clone revision:

```text
0825bd544fc172a2e2dcc96d43711123222c4a2f
```

It has built-in TicTacToe/simple-grid and CPU configs, but local deps are not
present in the repo `uv` environment (`ray` and `nevergrad` missing; `torch`
missing under `uv run`). Installing them would be a fresh dependency job, not a
tiny safe smoke.

The only tiny safe non-LightZero launch already present in this repo is the
Mctx/JAX Gumbel MuZero dependency smoke. It is not a trainer and not a
replication result, but it proves the non-LightZero MuZero search runtime can
execute a real `mctx.gumbel_muzero_policy` call on Modal.

## Command Launched

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_dependency_smoke --kind cpu
```

Expected artifact: Modal stdout JSON from
`curvyzero.infra.modal.mctx_dependency_smoke.cpu_smoke`, including:

- `ok`
- package versions for `mctx`, `jax`, `jaxlib`, and `numpy`
- JAX backend/devices
- finite normalized action weights
- action histogram
- compile-plus-first-run and second-run timings

No Modal Volume checkpoint is expected from this command.

Modal app:

```text
ap-E3GwfOaPOHu8Ce3IeFj5KA
```

Result:

```text
ok: true
backend: cpu
device: TFRT_CPU_0
packages: mctx==0.0.6, jax==0.7.0, jaxlib==0.7.0, numpy==2.4.4
action_histogram: [0, 0, 4]
action_weight_row_sums: [1.0, 1.0, 1.0, 1.0]
compile_plus_first_run_sec: 1.7602571709999997
second_run_sec: 0.001390027000000238
problems: []
```

## What It Proves

- Modal can build a non-LightZero JAX/Mctx image.
- Mctx can import and execute one tiny batched Gumbel MuZero search with
  `B=4`, `A=3`, hidden dim 8, 4 simulations, and max depth 4.
- Returned action weights are finite and row-normalized if `ok: true`.

## What It Does Not Prove

- No MuZero or AlphaZero trainer ran.
- No replay, target builder, optimizer update, checkpoint, or eval loop ran.
- No Pong, Atari, MiniZero, or `muzero-general` learning claim is established.
- It is search-runtime evidence only, not a replication-control result.

## Candidate Notes

### MiniZero

Official docs describe support for AlphaZero, MuZero, Gumbel AlphaZero, and
Gumbel MuZero across board games and Atari 57, including TicTacToe and Atari.
The quick start runs through a container and examples such as Go AlphaZero and
Atari MuZero/Gumbel MuZero. Because the prerequisite stack is container/GPU
oriented, the safe next step is a dedicated Modal build/run script, not an ad hoc
launch.

Durable local source fact: `/private/tmp/minizero-main/tools/quick-run.sh`
requires `nvidia-smi` and then builds game executables before launching
zero-server, self-play workers, and the optimization worker. The script's Atari
allow-list includes `pong`, so a Pong `env_atari_name=pong` run is source-backed
by the local official checkout, while the official README examples still use
Ms. Pac-Man.

Official documented tiny board-game command shape:

```bash
tools/quick-run.sh train tictactoe az 50
```

Minimum next command inside a MiniZero runtime container/GPU image:

```bash
tools/quick-run.sh train tictactoe az 1 \
  -n tictactoe_az_modal_smoke_20260511 \
  -conf_str actor_num_simulation=4:zero_num_parallel_games=1:zero_num_threads=1
```

Atari MuZero source-backed next command after the board-game smoke:

```bash
tools/quick-run.sh train atari mz 1 \
  -n pong_mz_modal_smoke_20260511 \
  -conf_str env_atari_name=pong:actor_num_simulation=4:zero_num_parallel_games=1:zero_num_threads=1
```

Possible later bounded command shape after a Modal wrapper exists:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.minizero_quick_train_smoke --game tictactoe --algorithm az --iterations 1 --num-simulations 4
```

2026-05-11 MiniZero control-lane update:

- Added wrapper:
  `src/curvyzero/infra/modal/minizero_quick_run_control.py`.
- Launched the official MiniZero quick-run path on Modal L4 with MiniZero source
  from `/private/tmp/minizero-main`, CUDA 11.8, PyTorch `2.1.2+cu118`, ALE built
  from source at `d59d00688b58c5c14dff5fc79db5c22e86987f5d`, Boost, OpenCV, and
  pybind11.
- Reached MiniZero's own `tools/quick-run.sh train tictactoe az 1` and its
  internal `scripts/build.sh tictactoe`.
- Blocker: MiniZero C++ build failed before training artifacts because
  `minizero/utils/vector_map.h` uses `std::out_of_range` without including
  `<stdexcept>` under the Modal Ubuntu 22.04/GCC 11/PyTorch image. No model,
  SGF, or training directory was produced.

Command that reached the blocker:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.minizero_quick_run_control \
  --run-id minizero-tictactoe-az-quickrun-20260511-s0d \
  --attempt-id train-tictactoe-az-iter1-l4t4
```

Artifact refs:

```text
training/minizero-official-quick-run/minizero-tictactoe-az-quickrun-20260511-s0d/attempts/train-tictactoe-az-iter1-l4t4/train/summary.json
training/minizero-official-quick-run/minizero-tictactoe-az-quickrun-20260511-s0d/attempts/train-tictactoe-az-iter1-l4t4/train/minizero
```

Exact next command if MiniZero is worth one more narrow push is a source-compat
image patch, then the same official quick-run:

```bash
cd /opt/minizero && \
python - <<'PY'
from pathlib import Path
p = Path("minizero/utils/vector_map.h")
text = p.read_text()
if "#include <stdexcept>" not in text:
    p.write_text("#include <stdexcept>\n" + text)
PY
tools/quick-run.sh train tictactoe az 1 \
  -n tictactoe_az_modal_smoke_20260511 \
  -conf_str actor_num_simulation=1:zero_num_parallel_games=1:zero_num_threads=1:zero_num_games_per_iteration=1:learner_training_step=1:learner_batch_size=2:nn_num_hidden_channels=16:nn_num_value_hidden_channels=16:program_use_color_message=false
```

Exact Modal wrapper command after adding that image-layer patch:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.minizero_quick_run_control \
  --run-id minizero-tictactoe-az-quickrun-20260511-s0e \
  --attempt-id train-tictactoe-az-iter1-l4t4-stdexcept-patch
```

Do not spend more time on MiniZero ahead of live MuZero replication unless that
one-line C++ include compatibility patch is accepted.

### OpenSpiel AlphaZero

Official OpenSpiel docs describe the AlphaZero implementation as an illustrative
reimplementation with actors, learner, replay buffer, evaluators, checkpoints,
and `learner.jsonl`; the docs give command shapes for
`tic_tac_toe_alpha_zero.py` and `alpha_zero.py --game connect_four --nn_model
mlp --actors 10`. The completed Modal control above uses the same official
Python AlphaZero implementation directly with a smaller TicTacToe config so it
can finish in one worker turn.

What it proves:

- OpenSpiel's non-LightZero AlphaZero actor/learner path can run in this Modal
  context.
- A real self-play actor generated trajectories, the learner updated once, and
  checkpoint/log artifacts were written.

What it does not prove:

- No MuZero dynamics model was trained.
- No visual Atari/Pong control ran.
- The one-step cap is a plumbing and artifact control, not a playing-strength
  result.

### muzero-general

Official README describes it as a documented educational MuZero
implementation with Ray, PyTorch, TensorBoard, single/two-player examples,
pretrained weights, TicTacToe, Connect4, Gridworld, and Atari Breakout. It also
lists Batch MCTS and more-than-two-player support as not implemented.

Local dependency check:

```text
system python: torch present, ray missing, nevergrad missing, numpy present
uv run: torch missing, ray missing, nevergrad missing, numpy present
```

Do not launch until a dependency install/build budget is approved. A credible
tiny command after dependencies would be a heavily capped TicTacToe or simple
grid smoke that records `results_path`, checkpoint presence, and one eval.

## Recommendation

Do not start MiniZero or `muzero-general` as an ad hoc background job today.
Count OpenSpiel AlphaZero TicTacToe as the completed outside-LightZero
AlphaZero-style trainer plumbing control, Mctx CPU smoke as the completed
outside-LightZero MuZero-search runtime control, and schedule a real MiniZero
board-game Modal wrapper if we want a full-system non-LightZero
AlphaZero/MuZero replication control.
