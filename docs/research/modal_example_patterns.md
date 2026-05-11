# Modal Example Patterns for CurvyZero

Last updated: 2026-05-08

Scope: concrete implementation patterns extracted from local Modal examples under
`/Users/shankha/modal-examples`, `/Users/shankha/modal-projects`, and the existing
CurvyZero note `docs/research/modal_patterns.md`. Web was not needed.

## Copy These Patterns First

CurvyZero should use Modal as a coarse compute, smoke-test, profiling, and artifact
layer. Keep environment stepping, model inference batches, MCTS expansion, replay
sampling, and gradient updates inside one container/process group.

1. Keep remote pytest and smoke checks in `src/curvyzero/infra/modal/smoke.py`.
2. Factor reusable Images into a later `src/curvyzero/infra/modal/images.py`.
3. Add a `curvyzero-runs` Volume for checkpoints, replay chunks, profiles, and
   benchmark output.
4. Add GPU package smokes before real MCTS/training jobs.
5. Add a profiler wrapper that writes traces to a Volume, copied from the PyTorch
   profiling example.
6. Use deployed apps and `modal.Function.from_name` only after smoke jobs are stable.
7. Use Queues and Dicts only for coarse coordination, never hot-loop state.

## Remote Pytest

CurvyZero already has the right seed pattern:

- `src/curvyzero/infra/modal/smoke.py` builds an Image from `src`, `tests`,
  `scripts`, `pyproject.toml`, and `README.md`, sets `PYTHONPATH=/repo/src`, and
  runs `python -m pytest -q` inside a Modal Function.

Keep that path and tighten it over time:

- Add `include_source=False` once all needed source is explicitly copied. The
  Lovable deployed compute app uses this to avoid accidental implicit source
  mounts.
- Keep stdout capture and return it for quick CI-style feedback.
- For test artifacts larger than stdout, mount `curvyzero-runs` and write a
  compact `pytest/<run_id>/summary.json`.
- Do not run pytest in a Sandbox unless the tests or generated code are untrusted.
  `/Users/shankha/modal-examples/misc/test_case_generator.py` is the Sandbox
  pattern for untrusted generated tests: it runs `poetry run pytest` inside a
  Sandbox with a mounted Volume and serves reports through encrypted ports.

Recommended commands now:

```sh
modal run -m curvyzero.infra.modal.smoke::run_tests
modal run -m curvyzero.infra.modal.smoke --kind tests
```

Local examples inspected:

- `/Users/shankha/modal-examples/internal/examples_test.py`
- `/Users/shankha/modal-examples/internal/run_example.py`
- `/Users/shankha/modal-examples/internal/CLAUDE.md`
- `/Users/shankha/modal-examples/misc/test_case_generator.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/deployments/lovable_loop_compute.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/scripts/modal_deployed_compute_client.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/tests/test_modal_deployed_compute.py`
- `src/curvyzero/infra/modal/smoke.py`

## Images

Use small, explicit Images instead of one giant training Image.

Recommended Images:

- `cpu_test_image`: Python 3.11, NumPy, pytest, ruff, CurvyZero source.
- `cpu_benchmark_image`: same base, plus any profiler/serialization tools.
- `torch_gpu_smoke_image`: pinned CUDA-compatible torch, minimal extras.
- `jax_mctx_image`: pinned JAX CUDA wheel, Mctx, Flax/Optax or Equinox.
- `train_image`: only after the smoke Images are proven.

Patterns to copy:

- `modal.Image.debian_slim(python_version="3.11").uv_pip_install(...)` for
  normal Python dependencies.
- `modal.Image.from_registry(..., add_python="3.11").entrypoint([])` when a
  CUDA or framework base image is the safer starting point.
- `with image.imports():` for imports that exist only inside the Modal Image.
- Exact dependency pins for GPU/compiler-sensitive packages.
- Build-time `run_commands(..., gpu=GPU, volumes={...})` only for expensive,
  reproducible GPU compile/warmup layers.

Do not copy replay, checkpoints, `docs`, `.venv`, `.git`, or benchmark outputs
into Images. Mount Volumes for runtime state.

Local examples inspected:

- `/Users/shankha/modal-examples/02_building_containers/import_sklearn.py`
- `/Users/shankha/modal-examples/02_building_containers/install_cuda.py`
- `/Users/shankha/modal-examples/02_building_containers/install_flash_attn.py`
- `/Users/shankha/modal-examples/06_gpu_and_ml/import_torch.py`
- `/Users/shankha/modal-examples/06_gpu_and_ml/hyperparameter-sweep/hp_sweep_gpt.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/serving/modal_sglang.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/deployments/qwen3_8b.py`
- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/runtime_common.py`

## Adding Local Source

There are two good source-copy patterns.

For CurvyZero today, keep the explicit src-layout pattern:

```python
REMOTE_ROOT = Path("/repo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy>=1.26", "pytest>=8")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(ROOT / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(ROOT / "tests", remote_path=str(REMOTE_ROOT / "tests"), copy=True)
    .add_local_file(ROOT / "pyproject.toml", remote_path=str(REMOTE_ROOT / "pyproject.toml"))
)
```

Later, if the package is installed/importable cleanly from the local environment,
use the package-source pattern from SGLang serving:

```python
image = image.add_local_python_source("curvyzero", copy=True)
```

The deployed Lovable compute app shows the best production shape: copy only the
repo subtrees needed by the remote Function, set `include_source=False`, mount
runtime/source Volumes, and keep client scripts calling deployed named Functions.

Local examples inspected:

- `/Users/shankha/modal-projects/flash-projects/lovable/deployments/lovable_loop_compute.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/scripts/modal_function_build_current_head_runtime.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/serving/modal_sglang.py`
- `/Users/shankha/modal-examples/09_job_queues/doc_ocr_webapp.py`
- `/Users/shankha/modal-examples/06_gpu_and_ml/hyperparameter-sweep/hp_sweep_gpt.py`

## Volumes for Artifacts

Use a Volume named something like `curvyzero-runs` for active experiment state.
Use separate cache Volumes only when they have a distinct lifecycle, for example
Hugging Face or JAX compilation caches.

Recommended layout:

```text
/experiments/<run_id>/
  config.json
  attempts/
    <attempt_id>/
      pytest/
      benchmarks/
      profiles/
      logs/
  checkpoints/
    ckpt_<step>/
    latest.json
  replay/
    shard=<actor_id>/
      part-000000.npz
  eval/
    ckpt_<step>/
      metrics.json
  complete.json
```

Rules to copy:

- Write immutable attempt-specific paths.
- Commit after material artifacts that other Functions or local clients must see.
- Publish a small manifest only after payload files exist.
- Use `batch_upload(force=False)` or exclusive local file creation for summaries
  that must never overwrite a previous run.
- Fetch exact refs for debugging instead of downloading whole run trees.
- Use `volume.reload()` in web/profile readers before serving updated files.

Good concrete examples:

- `/Users/shankha/modal-projects/flash-projects/lovable/scripts/modal_function_full_loop_rows.py`
  writes each row under `rows/<row>/attempts/<attempt>/...`, commits the run
  Volume, and writes `complete.json` only after success.
- `/Users/shankha/modal-examples/06_gpu_and_ml/long-training.py` stores
  checkpoints on a Volume and resumes from the latest checkpoint.
- `/Users/shankha/modal-examples/06_gpu_and_ml/text-to-video/ltx.py` writes
  outputs to a Volume, commits, returns the remote file name, and supports exact
  local fetch.
- `/Users/shankha/modal-examples/06_gpu_and_ml/torch_profiling.py` and
  `/Users/shankha/modal-examples/06_gpu_and_ml/tensorflow/tensorflow_tutorial.py`
  use Volume-backed TensorBoard/profiling output plus `reload()`.

Useful commands:

```sh
modal volume ls curvyzero-runs
modal volume get curvyzero-runs <exact-ref-from-summary> <local-file> --force
```

## GPU Smoke

Keep `src/curvyzero/infra/modal/smoke.py::gpu_smoke` as the first device check.
It currently asks Modal for `gpu=["L4", "T4"]` and runs `nvidia-smi`.

Next GPU smoke to add:

```python
@app.function(image=torch_gpu_smoke_image, gpu=["L4", "T4"], timeout=10 * 60)
def torch_gpu_smoke() -> dict:
    import torch

    device = torch.cuda.get_device_properties("cuda:0")
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    return {"name": device.name, "shape": list(y.shape)}
```

For JAX/Mctx, mirror the same shape: import packages, list devices, run one
small fixed-shape matmul/search call, and return structured metrics.

Recommended commands now:

```sh
modal run -m curvyzero.infra.modal.smoke::gpu_smoke
modal run -m curvyzero.infra.modal.smoke --kind gpu
```

Local examples inspected:

- `/Users/shankha/modal-examples/02_building_containers/install_cuda.py`
- `/Users/shankha/modal-examples/06_gpu_and_ml/import_torch.py`
- `/Users/shankha/modal-examples/02_building_containers/install_flash_attn.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/deployments/qwen3_8b.py`
- `src/curvyzero/infra/modal/smoke.py`

## Profiling

Copy the wrapper pattern from `torch_profiling.py`:

- The target Function and profiler Function use the same `config` dict for Image
  and GPU, so profiling does not change the runtime environment.
- The profiler calls the target with `.local(...)` inside the same container,
  avoiding a remote Function call in the measured hot path.
- PyTorch profiler writes traces into a Volume and returns both trace text and a
  relative Volume path.
- A TensorBoard WSGI Function mounts the same Volume and reloads it on page
  refresh.

CurvyZero use:

- Add `profile_env_benchmark` for CPU step/reset/observation timings.
- Add `profile_torch_mcts_smoke` if the PyTorch/LightZero path stays alive.
- Add `profile_jax_mctx_smoke` with JAX-native profiling tools for the Mctx path.
- Store traces under `/experiments/<run_id>/profiles/<label>/`.

Important profiling rule: profile the hot loop inside the container that owns
the hot loop. Do not profile a chain of Modal Function calls and mistake network
latency for simulator or search cost.

Local examples inspected:

- `/Users/shankha/modal-examples/06_gpu_and_ml/torch_profiling.py`
- `/Users/shankha/modal-examples/06_gpu_and_ml/tensorflow/tensorflow_tutorial.py`
- `/Users/shankha/modal-examples/06_gpu_and_ml/hyperparameter-sweep/hp_sweep_gpt.py`

## Queues and Dicts

Use Queues and Dicts as coordination planes only.

Good uses to copy:

- Queue of coarse work refs: checkpoint eval jobs, replay compaction chunks, or
  sandbox/warm-session refs.
- Dict of tiny state: latest checkpoint pointer, run status, replica URL, lease,
  heartbeat, or kill switch.

Uses to reject for CurvyZero:

- Per environment tick.
- Per action.
- Per MCTS node.
- Per model inference request inside search.
- Replay buffer, model weights, full metrics streams, or mutable nested state.

Local examples make this boundary clear:

- `/Users/shankha/modal-examples/09_job_queues/dicts_and_queues.py` uses Queue
  batches and a Dict stop flag for a crawler. This is coarse URL work, not a
  millisecond simulator loop.
- `/Users/shankha/modal-examples/13_sandboxes/sandbox_pool.py` uses a Queue for
  warm Sandbox references.
- `/Users/shankha/modal-projects/flash-projects/lovable/deployments/modal_codex_chat/app.py`
  uses a Queue for a warm Sandbox pool and a Volume for durable session state.
- `/Users/shankha/modal-projects/flash-projects/lovable/serving/modal_sglang.py`
  uses a Dict as a replica registry.

CurvyZero should add tests later that fail if `src/curvyzero/env` imports
`modal` or if train/search hot-loop modules use `modal.Queue`, `modal.Dict`, or
`.remote()` in step/search/update code.

## Function Fanout and Deployed Calls

Use `.map` and `.starmap` for bounded sweeps and independent shards. Use
`.spawn` for background jobs when results are stored in a Volume and polled
later. Use deployed `Function.from_name` once the app is stable and clients
should not create ephemeral apps.

Patterns to copy:

- `/Users/shankha/modal-examples/03_scaling_out/basic_grid_search.py` uses
  `.map(range(...))` for a small hyperparameter sweep.
- `/Users/shankha/modal-examples/06_gpu_and_ml/hyperparameter-sweep/hp_sweep_gpt.py`
  uses `.starmap(..., order_outputs=False)` for parallel training cells and then
  continues the best run.
- `/Users/shankha/modal-examples/09_job_queues/doc_ocr_jobs.py` deploys a named
  Function and documents `modal.Function.from_name(...).spawn(...)`.
- `/Users/shankha/modal-examples/09_job_queues/doc_ocr_webapp.py` uses
  `Function.from_name`, `.spawn`, and `FunctionCall.from_id` polling.
- `/Users/shankha/modal-projects/flash-projects/lovable/scripts/modal_function_full_loop_rows.py`
  uses `.map(..., kwargs=..., order_outputs=True)` for parallel row execution.
- `/Users/shankha/modal-projects/flash-projects/lovable/scripts/modal_deployed_compute_client.py`
  centralizes `Function.from_name` and fails loudly if the deployed app is absent.

CurvyZero sequence:

1. Keep ephemeral `modal run` for smoke functions.
2. Add deployed app only when there are multiple client commands or scheduled
   jobs.
3. Make clients fail loudly if deployed Functions are missing.
4. Never silently fall back from deployed wide runs to old ephemeral scripts.

## CLI Commands Worth Copying

Current CurvyZero:

```sh
uv run pytest -q
uv run python scripts/benchmark_env.py --episodes 100 --max-steps 500
modal run -m curvyzero.infra.modal.smoke::run_tests
modal run -m curvyzero.infra.modal.smoke::benchmark_env --episodes 1000 --max-steps 2000
modal run -m curvyzero.infra.modal.smoke::gpu_smoke
```

Future CurvyZero deployed shape:

```sh
modal deploy -m curvyzero.infra.modal.train
python -m scripts.modal_build_curvyzero_runtime --output-json tmp/<run_id>_runtime.json
python -m scripts.modal_run_curvyzero_train --runtime-json tmp/<run_id>_runtime.json --run-id <run_id>
python -m scripts.modal_fetch_curvyzero_artifact --summary-json tmp/<run_id>.json --ref <exact-ref>
```

Source examples for command style:

- `/Users/shankha/modal-examples/06_gpu_and_ml/long-training.py`:
  `modal run --detach 06_gpu_and_ml/long-training.py`
- `/Users/shankha/modal-examples/06_gpu_and_ml/torch_profiling.py`:
  `modal run torch_profiling.py --function underutilize --print-rows 10` and
  `modal deploy torch_profiling`
- `/Users/shankha/modal-examples/09_job_queues/doc_ocr_jobs.py`:
  `modal run doc_ocr_jobs.py` and `modal deploy doc_ocr_jobs.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/00_START_HERE.md`:
  `modal deploy -m deployments.lovable_loop_compute`, then Python clients that
  call named deployed Functions.

## CurvyZero Implementation Recommendations

Near-term:

- Keep `src/curvyzero/infra/modal/smoke.py` as the smoke app.
- Add a run Volume to the smoke app for pytest and benchmark summaries.
- Add torch and JAX GPU smoke Functions before adding MCTS code.
- Add one profiling module copied from `torch_profiling.py`.
- Keep local benchmark and Modal benchmark output shapes identical.

Before first training run:

- Create `src/curvyzero/infra/modal/images.py`.
- Create `src/curvyzero/infra/modal/train.py` with app name `curvyzero-train`.
- Mount `curvyzero-runs` at `/runs`.
- Write checkpoints and replay chunks under immutable run/attempt paths.
- Return only compact summaries with Volume refs.
- Add an exact-ref artifact fetch script rather than downloading whole run trees.

Before scale-out:

- Deploy `curvyzero-train`.
- Add client scripts that use `modal.Function.from_name`.
- Use `.map` or `.starmap` for self-play/eval shards.
- Put `max_containers` on sweeps.
- Keep Queue/Dict out of hot loops and use them only for coarse work/ref state.

## Exact Local Files Inspected

- `/Users/shankha/modal-examples/internal/examples_test.py`
- `/Users/shankha/modal-examples/internal/run_example.py`
- `/Users/shankha/modal-examples/internal/CLAUDE.md`
- `/Users/shankha/modal-examples/01_getting_started/inference_full.py`
- `/Users/shankha/modal-examples/02_building_containers/import_sklearn.py`
- `/Users/shankha/modal-examples/02_building_containers/install_cuda.py`
- `/Users/shankha/modal-examples/02_building_containers/install_flash_attn.py`
- `/Users/shankha/modal-examples/03_scaling_out/basic_grid_search.py`
- `/Users/shankha/modal-examples/03_scaling_out/cls_with_options.py`
- `/Users/shankha/modal-examples/06_gpu_and_ml/import_torch.py`
- `/Users/shankha/modal-examples/06_gpu_and_ml/long-training.py`
- `/Users/shankha/modal-examples/06_gpu_and_ml/torch_profiling.py`
- `/Users/shankha/modal-examples/06_gpu_and_ml/tensorflow/tensorflow_tutorial.py`
- `/Users/shankha/modal-examples/06_gpu_and_ml/hyperparameter-sweep/hp_sweep_gpt.py`
- `/Users/shankha/modal-examples/06_gpu_and_ml/text-to-video/ltx.py`
- `/Users/shankha/modal-examples/07_web_endpoints/fasthtml-checkboxes/cbx_load_test.py`
- `/Users/shankha/modal-examples/09_job_queues/dicts_and_queues.py`
- `/Users/shankha/modal-examples/09_job_queues/doc_ocr_jobs.py`
- `/Users/shankha/modal-examples/09_job_queues/doc_ocr_webapp.py`
- `/Users/shankha/modal-examples/13_sandboxes/sandbox_pool.py`
- `/Users/shankha/modal-examples/misc/test_case_generator.py`
- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/runtime_common.py`
- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/main.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/00_START_HERE.md`
- `/Users/shankha/modal-projects/flash-projects/lovable/deployments/lovable_loop_compute.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/deployments/modal_codex_chat/app.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/deployments/qwen3_8b.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/scripts/modal_deployed_build_current_head_runtime.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/scripts/modal_deployed_compute_client.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/scripts/modal_deployed_full_loop_rows.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/scripts/modal_function_build_current_head_runtime.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/scripts/modal_function_full_loop_rows.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/serving/modal_sglang.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/tests/test_modal_deployed_compute.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/tests/test_modal_function_full_loop_rows.py`
- `/Users/shankha/curvy/docs/research/modal_patterns.md`
- `/Users/shankha/curvy/docs/design/modal_architecture.md`
- `/Users/shankha/curvy/docs/decisions/0002-modal-hot-loop-locality.md`
- `/Users/shankha/curvy/docs/experiments/2026-05-08-env-smoke-benchmark.md`
- `/Users/shankha/curvy/src/curvyzero/infra/modal/smoke.py`
- `/Users/shankha/curvy/scripts/benchmark_env.py`
