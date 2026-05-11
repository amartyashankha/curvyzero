# Modal Training Loop Patterns

Last updated: 2026-05-08

Purpose: keep just enough Modal guidance to build CurvyZero's near-term
training loop. This is not a broad Modal research lane.

## Recommendation

Start with one coarse Modal training Function. It should own a whole training
attempt, write durable artifacts to a Volume, and resume from the latest
checkpoint. Use Modal for job orchestration and storage boundaries, not per-step
environment, replay, or MCTS communication.

Do the first implementation on CPU or a cheap single GPU only after the local
training loop exists. Multi-node is later.

## Pattern To Copy Soon

Local examples:

- `/Users/shankha/modal-examples/06_gpu_and_ml/long-training.py`
  - Best small example for resumable training.
  - Patterns: `modal.Volume.from_name`, mounted checkpoint directory,
    `modal.Retries`, explicit `timeout`, `single_use_containers=True`,
    `.spawn(...).get()`, and `modal run --detach`.
  - Command:
    ```sh
    cd /Users/shankha/modal-examples
    modal run --detach 06_gpu_and_ml/long-training.py
    ```

- `/Users/shankha/modal-examples/06_gpu_and_ml/hyperparameter-sweep/hp_sweep_gpt.py`
  - Useful only for artifact layout and checkpoint commits, not as a full
    CurvyZero template.
  - Patterns: one Volume for data/checkpoints/logs, periodic `volume.commit()`,
    resume from existing checkpoint directory, and coarse `starmap` sweeps.
  - Command:
    ```sh
    cd /Users/shankha/modal-examples
    modal run 06_gpu_and_ml/hyperparameter-sweep/hp_sweep_gpt.py \
      --n-steps 200 \
      --n-steps-before-checkpoint 50 \
      --n-steps-before-eval 50
    ```

- `/Users/shankha/modal-examples/06_gpu_and_ml/torch_profiling.py`
  - Later single-node GPU profiling wrapper.
  - Pattern: keep the training Function clean; add a separate profiler Function
    with the same image/GPU config and write traces to a Volume.
  - Command:
    ```sh
    cd /Users/shankha/modal-examples
    modal run 06_gpu_and_ml/torch_profiling.py --function underutilize --print-rows 10
    ```

- `/Users/shankha/modal-examples/14_clusters/simple_torch_cluster.py`
  - Later multi-node PyTorch note only. Do not copy for the first loop.
  - Pattern: `@modal.experimental.clustered`, `get_cluster_info()`, launch
    `torch.distributed.run`.
  - Command:
    ```sh
    cd /Users/shankha/modal-examples
    modal run 14_clusters/simple_torch_cluster.py
    ```

## First CurvyZero Modal Job

Build a small `training_smoke` job around the chosen learner path. Local should
only prove tiny debug behavior; Modal should own serious train/eval attempts.

It should include:

- One Modal Function for the whole attempt.
- A `curvyzero-runs` Volume mounted at a stable path like `/runs`.
- A small config artifact with `run_id`, seed, ruleset id, observation schema id,
  max steps, checkpoint interval, and code/dependency marker.
- Checkpoints written to immutable paths such as
  `training/<run_id>/checkpoints/step_000100/`.
- A tiny `checkpoints/latest.json` pointer written after each full checkpoint.
- `metrics.jsonl` and `manifest.json`.
- Resume on start by reading `latest.json` if it exists.
- Explicit `volume.commit()` after checkpoint and final manifest writes.
- `modal.Retries` only after resume is idempotent.

Sketch:

```python
runs = modal.Volume.from_name("curvyzero-runs", create_if_missing=True)
RUN_ROOT = Path("/runs")

@app.function(
    image=train_image,
    volumes={RUN_ROOT: runs},
    timeout=60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    single_use_containers=True,
)
def training_smoke(run_id: str, max_steps: int = 200):
    run_dir = RUN_ROOT / "training" / run_id
    latest = run_dir / "checkpoints" / "latest.json"
    state = load_state(latest) if latest.exists() else init_state(run_id)

    for step in range(state.step, max_steps):
        state = train_step(state)
        if step % 25 == 0:
            write_checkpoint(run_dir, state)
            runs.commit()

    write_manifest(run_dir, state)
    runs.commit()
    return {"run_id": run_id, "step": state.step, "manifest": str(run_dir / "manifest.json")}
```

Suggested command shape:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.training_smoke \
  --run-id train-smoke-YYYYMMDD-001 \
  --max-steps 200
```

## Artifact Rules

- Prefer fewer, larger files over many tiny files.
- Never have multiple writers update the same checkpoint file.
- Write checkpoints immutably, then update a small latest pointer.
- Readers or monitors should call `volume.reload()` before reading newly
  committed data, with no open files under the Volume.
- Keep replay chunks, checkpoint manifests, and metrics schema-versioned from
  the start.

## Later Single-node GPU

Once the CPU Modal smoke and chosen learner path are boring:

- Add `gpu="L40S"` or a conservative fallback list.
- Record `nvidia-smi`, framework device visibility, peak memory, and throughput.
- Add a profile mode for a few steps only.
- Separate compile/setup time from steady-state step time, especially for JAX.
- Use coarse `map`/`starmap` only for independent jobs like seed sweeps,
  checkpoint evals, and profiler variants.

I found no local Modal example for JAX distributed training in the searched
paths. Treat JAX/Mctx as single-container first.

## Later Multi-node Note

Only revisit Modal clusters after single-node GPU training is clearly
bottlenecked and checkpoint/replay schemas are stable.

Relevant example:

- `/Users/shankha/modal-examples/14_clusters/simple_torch_cluster.py`
- `/Users/shankha/modal-examples/14_clusters/simple_torch_cluster_script.py`

Constraints from Modal docs:

- Multi-node clusters are beta.
- Clustered jobs are gang scheduled.
- Rank 0's output is the function result.
- If one clustered container is preempted, Modal terminates the remaining
  containers and retries the input.
- Official docs say clustered functions require GPUs, and starting 2026-05-31
  clustered functions must use 8 GPU devices per node.

## Sources

- Modal long training: https://modal.com/docs/examples/long-training
- Modal Volumes: https://modal.com/docs/guide/volumes
- Modal torch profiling: https://modal.com/docs/examples/torch_profiling
- Modal multi-node clusters: https://modal.com/docs/guide/multi-node-training
