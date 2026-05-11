# Modal Patterns for CurvyTron RL

Last researched: 2026-05-08

Scope: Modal primitives and operating patterns for CurvyTron RL, especially self-play, MuZero/MCTS, replay, checkpoints, experiments, and evaluation. This note uses current official Modal docs plus local examples under `/Users/shankha/modal-projects`.

## Recommendation Summary

Use Modal as the coarse-grained compute and artifact layer, not as the per-step runtime for the game or search loop.

- Keep the CurvyTron simulator, vectorized environment stepping, MCTS expansion, neural inference, replay sampling, and gradient update loop inside one Modal container or one tightly-coupled local process group.
- Use Modal Functions for coarse jobs: train run, self-play shard, evaluation shard, replay conversion, checkpoint validation, benchmark sweep, and artifact packaging.
- Use Volumes first for hot experiment artifacts and checkpoints; graduate long-lived replay archives to object storage through `CloudBucketMount` or direct SDK access when replay volume or compliance demands it.
- Use Dicts and Queues only for small coordination state and job dispatch. They have network latency and are not hot-loop primitives.
- Use `Function.map`, `starmap`, `spawn`, and deployed `Function.from_name` for parallel job control. Use `modal.Queue` only when a live producer/consumer queue is truly easier than function input lists.
- Treat long runs as resumable. Modal Function calls have a 24-hour maximum timeout and are preemptible, so checkpoint frequently and make retries idempotent.
- Use Sandboxes and Sandbox snapshots for untrusted code, prepared runtime fanout, and branch replay. They are not the default for CurvyTron self-play or model training.
- Use Function Memory Snapshots for cold-start reduction after profiling import/JIT cost. Do not confuse snapshots with durable replay/checkpoint storage.

## Hot-Loop Dangers

These are the failure modes most likely to make CurvyTron RL slow, flaky, or expensive:

| Danger | Why it is dangerous | Safer pattern |
| --- | --- | --- |
| Modal Queue per action, per game tick, or per MCTS node | Queue operations require network communication and add tens of milliseconds. | Batch many environments and MCTS calls inside one container; queue only coarse jobs. |
| Modal Dict for replay, weights, per-step counters, or per-node search state | Dict reads/writes go over the network and add a few dozen ms; mutation semantics differ from local dicts. | Store replay/checkpoints as chunked files; use Dict only for small metadata and pointers. |
| One Modal Function call per environment step | Function invocation overhead dominates a microsecond/millisecond simulator. | One long-running function owns a vectorized env batch and emits chunked artifacts. |
| Millions of tiny files in a Volume | Volume attach/modify latency scales with file count; docs recommend staying under 50,000 files and note a hard 500,000 inode limit. | Write replay in larger chunks such as `.npz`, `.parquet`, `.safetensors`, or sharded binary records. |
| Multiple writers committing the same Volume file | Last write wins; data not seen by the final committer can be lost. | Write immutable per-run/per-shard paths; publish a manifest pointer after successful commit. |
| CloudBucketMount append/random writes | Mountpoint-backed bucket mounts do not support append mode or arbitrary offset writes. | Write temp files then move/copy; prefer Volumes for append-heavy training logs. |
| Uncheckpointed 24-hour training call | Function timeout/preemption can restart the same input and lose progress. | Frequent checkpoints plus idempotent resume from latest manifest. |
| GPU Memory Snapshot as a magic model-loader | Modal says GPU snapshots do not speed up weight loading from storage and can add overhead. | Use snapshots for import/JIT/warmup work after measuring startup profile. |
| Memory snapshots with hidden randomness | Snapshot-captured random state can repeat across restored containers. | Reseed after restore; keep run ids and worker ids explicit. |
| CPU-only Memory Snapshot touching CUDA | Accidental `torch.cuda` calls during CPU-only snapshot can initialize CUDA incorrectly. | Split `@modal.enter(snap=True)` CPU prep from `@modal.enter(snap=False)` GPU initialization, or use GPU snapshots carefully. |

## Modal Apps

Modal Apps group Functions and Clses for atomic deployment and shared identity/log collection. Official docs distinguish ephemeral Apps created by `modal run` or `app.run` from deployed Apps created by `modal deploy`. Deployed Apps persist until stopped and are the right target for schedules, stable web endpoints, and `Function.from_name`.

CurvyTron pattern:

- `curvytron-rl-dev`: development smoke app.
- `curvytron-rl-train`: deployed training/evaluation app.
- `curvytron-rl-serve`: optional policy demo or evaluator service.
- Keep Apps named and environment-specific. Avoid depending on ephemeral app identity for durable experiments.
- Use `app.include(...)` when splitting proxy/web/training modules, as seen in `/Users/shankha/modal-projects/claude-slack-gif-creator/src/main.py:18`.

Local example:

- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/main.py:18` defines a named `modal.App` and includes a proxy app.
- `/Users/shankha/modal-projects/flash-projects/lovable/deployments/qwen3_8b.py:16` defines a dedicated deployed app for an OpenAI-compatible GPU model endpoint.
- `/Users/shankha/modal-projects/flash-projects/lovable/00_START_HERE.md:24` documents a deployed app path where local clients call named deployed functions instead of relying on ephemeral `app.run`.

Sources:

- Modal Apps guide: https://modal.com/docs/guide/apps
- Modal App reference: https://modal.com/docs/reference/modal.App

## Modal Functions

Functions are the core serverless execution unit. Register them with `@app.function`; call them with `.remote`, `.map`, `.starmap`, `.spawn`, or deployed lookup via `modal.Function.from_name`.

Good CurvyTron uses:

- `train_run.remote(config_ref)`: one resumable trainer/self-play loop.
- `selfplay_shard.map(shard_specs)`: independent actor shards over seed ranges.
- `evaluate_checkpoint.starmap((checkpoint_ref, opponent_ref, seed_range)...)`.
- `convert_replay.spawn_map(replay_shards)` for background conversion jobs that write results to a Volume or bucket.
- `Function.from_name` from local orchestration scripts after `modal deploy`, so wide runs do not require code to run inside an ephemeral app.

Function sizing:

- Set `timeout` explicitly. Default is 300 seconds; max is 24 hours.
- Use `startup_timeout` for slow image/model startup.
- Use `retries` for idempotent jobs only.
- Use `max_containers` to cap sweeps and cost.
- Use `min_containers`, `buffer_containers`, and `scaledown_window` only when latency matters enough to pay for warm capacity.

Parallelism:

- `Function.map` returns ordered results and can process many inputs in parallel.
- `Function.spawn` returns a `FunctionCall` handle for later polling.
- `spawn_map` is useful for background batch submission when outputs are stored externally.
- Official scaling docs mention pending/total input limits and note that `.spawn()` async jobs allow more pending inputs than ordinary calls. Design launchers to chunk large sweeps.

Local examples:

- `/Users/shankha/modal-projects/flash-projects/lovable/00_START_HERE.md:260` records a deployed-function path using `modal.Function.from_name`.
- `/Users/shankha/modal-projects/flash-projects/lovable/00_START_HERE.md:354` records row execution through `Function.map`.
- `/Users/shankha/modal-projects/flash-projects/modal-decagon/training/benchmarks/volume_throughput/archive/exp_large_pipeline.py:463` uses `.spawn()` for a consumer and `.starmap()` for producers.

Sources:

- Invoking deployed functions: https://modal.com/docs/guide/trigger-deployed-functions
- Scaling out: https://modal.com/docs/guide/scale
- Job processing: https://modal.com/docs/guide/job-queue
- Batch processing: https://modal.com/docs/guide/batch-processing
- Function reference: https://modal.com/docs/reference/modal.Function

## Images

Images define the container filesystem and dependencies. Use method chaining from a base image, pin dependencies tightly, and place frequently changing layers late to preserve image cache.

CurvyTron pattern:

- CPU env image: simulator, numpy, numba optional, gymnasium/pettingzoo, test/debug tooling.
- GPU JAX image: pinned `jax[cuda13]` or `jax[cuda12]`, `mctx`, `flax` or `equinox`, `optax`, profiler tools.
- GPU PyTorch image: pinned torch, CUDA-compatible libs, TensorBoard/W&B, optional LightZero.
- Serving image: if needed, separate from training image.
- Use `add_local_python_source` or `add_local_dir(copy=True)` for project code that must be available at build time.
- Use `run_function` or `run_commands(..., gpu=...)` for expensive compile/setup layers only when reproducibility is clear.
- Do not copy large replay/checkpoint/runtime folders into the image. Mount Volumes or buckets.

Local examples:

- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/runtime_common.py:12` builds a sandbox image with apt, pip, run commands, and local agent code.
- `/Users/shankha/modal-projects/flash-projects/lovable/serving/modal_sglang.py:16` starts from an existing SGLang registry image and adds local Python source.
- `/Users/shankha/modal-projects/flash-projects/lovable/serving/modal_sglang.py:68` runs GPU-backed build-time compilation while mounting the Hugging Face cache Volume.

Sources:

- Modal Images guide: https://modal.com/docs/guide/images
- Existing images: https://modal.com/docs/guide/existing-images

## Volumes

Volumes are the default CurvyTron experiment storage primitive. Modal describes them as a distributed filesystem optimized for write-once, read-many workloads such as ML weights and inference distribution. You must understand commit/reload semantics:

- A container sees the Volume state mounted at container creation.
- If another container commits changes, an existing container must call `.reload()` to see them.
- A container's changes become visible outside that container after background commit, final shutdown commit, or explicit `.commit()`.
- Concurrent modifications to the same file should be avoided; last writer wins.

CurvyTron layout:

```text
/experiments/<run_id>/
  config.json
  status.json
  checkpoints/
    ckpt_000010000/
      model.safetensors
      optimizer.npz
      trainer_state.json
    latest.json.tmp
    latest.json
  replay/
    shard=<actor_id>/
      part-000000.npz
      part-000001.npz
  eval/
    ckpt_000010000/
      metrics.json
      videos/
  logs/
    events.out.tfevents...
```

Write rules:

- Write immutable files under attempt/shard-specific paths.
- Commit after material checkpoints and replay chunks.
- Publish small manifests after the payload is committed.
- Use temp manifest names then rename/write final manifest to avoid readers seeing incomplete JSON.
- Use per-attempt folders for retry safety.
- Keep local outputs compact: run id, remote refs, metrics summary, and failure samples.

Local examples:

- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/runtime_common.py:10` creates a shared Volume.
- `/Users/shankha/modal-projects/claude-slack-gif-creator/docs/experiment-footguns.md:51` records the lesson that runner writes may need explicit commit before sandbox creation.
- `/Users/shankha/modal-projects/flash-projects/lovable/00_START_HERE.md:286` says wide artifacts stay on mounted v2 Volumes and local output stays compact JSON.
- `/Users/shankha/modal-projects/flash-projects/lovable/00_START_HERE.md:201` documents retry-safe per-attempt folders after preemption/retry collision.
- `/Users/shankha/modal-projects/flash-projects/modal-decagon/training/benchmarks/volume_throughput/archive/exp_large_pipeline.py:121` explicitly commits after producer writes and signals readiness through a Dict.

Sources:

- Volumes guide: https://modal.com/docs/guide/volumes
- Volume reference: https://modal.com/docs/reference/modal.Volume

## CloudBucketMounts

`modal.CloudBucketMount` mounts cloud object storage as a mutable filesystem. Current Modal docs say it supports AWS S3, Cloudflare R2, and Google Cloud Storage. It is built on Mountpoint and inherits Mountpoint limitations.

Use for:

- Long-lived replay archive.
- Evaluation videos.
- Exported datasets.
- Large immutable corpora that outgrow a Modal Volume.
- Compliance/security locations where artifacts must land in a specific bucket.

Avoid for:

- Hot per-step writes.
- Append-heavy logs.
- Random-offset checkpoint writers.
- Libraries that require append mode or seek+write unless you wrap them with temp-file staging.

CurvyTron pattern:

- Write replay/checkpoints first to a Modal Volume during active training.
- Promote completed immutable chunks to bucket paths.
- Keep manifests in both places, or keep a Volume pointer to the canonical bucket manifest.
- For bucket writes, write to a local temp path or Volume temp path, then copy/move into the mounted bucket in truncate/write mode.

Sources:

- Cloud bucket mounts guide: https://modal.com/docs/guide/cloud-bucket-mounts
- CloudBucketMount reference: https://modal.com/docs/reference/modal.CloudBucketMount

## Queues

Modal Queues are distributed FIFO queues. Official docs say they are best for communication between active functions and should not be relied on for persistent storage. Queue interactions require network communication and add latency on the order of tens of milliseconds.

Good CurvyTron uses:

- Coarse work queue: "evaluate checkpoint X against opponent set Y".
- Backfill queue: "convert replay chunk Z".
- Lightweight live actor coordinator if `.map`/`.spawn_map` is not enough.
- Producer/consumer experiments where the queue item is a chunk ref, not the chunk payload.

Bad CurvyTron uses:

- Per environment step.
- Per action.
- Per MCTS node.
- Per inference request inside search.
- Durable replay storage.

Queue limits from current docs/reference:

- Up to 100,000 partitions.
- Up to 5,000 items per partition.
- Up to 1 MiB per item.
- Default partition TTL is 24 hours after the last put.

Sources:

- Queues guide: https://modal.com/docs/guide/queues
- Queue reference: https://modal.com/docs/reference/modal.Queue

## Dicts

Modal Dicts provide distributed key-value storage. They are persistent across app redeploys, but reads/writes go over the network. Current docs recommend smaller objects and note a per-object size limit of 100 MiB, max 10,000 entries per update, and 7-day inactivity expiry for entries.

Good CurvyTron uses:

- Current run status.
- Latest checkpoint pointer.
- Lease/heartbeat records for coarse actor shards.
- Small counters, progress summaries, and URLs.
- Snapshot id registry for sandbox fanout experiments.

Bad CurvyTron uses:

- Replay buffer.
- Model weights.
- Per-step state.
- Mutable nested updates without putting the object back.
- High-frequency training metrics.

Local examples:

- `/Users/shankha/modal-projects/flash-projects/lovable/serving/modal_sglang.py:12` uses a Dict as a replica registry.
- `/Users/shankha/modal-projects/flash-projects/modal-decagon/training/benchmarks/volume_throughput/archive/exp_large_pipeline.py:69` uses a Dict as readiness coordination after Volume commits.
- `/Users/shankha/modal-projects/flash-projects/modal-decagon/training/benchmarks/volume_throughput/archive/exp_i6pn_streaming.py:16` explicitly uses a Dict for coordination in a networking benchmark, not as the data plane.

Sources:

- Dicts guide: https://modal.com/docs/guide/dicts
- Dict reference: https://modal.com/docs/reference/modal.Dict

## Sandboxes

Sandboxes are secure containers for running arbitrary or untrusted code. They can be created dynamically, have independent lifecycle controls, can use Images and Volumes, and support named lookup in deployed apps.

Good CurvyTron uses:

- Running untrusted submissions or generated code against a CurvyTron environment API.
- Replaying external agent code in isolation.
- Branching a prepared source/runtime state for many evaluation variants.
- Debugging environment/package setup separate from training Functions.

Bad CurvyTron uses:

- Main self-play hot loop.
- Per-game or per-step simulation.
- GPU MuZero training unless the isolation/fanout requirement dominates.

Sandbox lifecycle:

- Default max lifetime is 5 minutes.
- `timeout` can be raised up to 24 hours.
- `idle_timeout` can terminate inactive sandboxes.
- For more than 24 hours, docs recommend preserving state with Filesystem Snapshots and restoring into a later Sandbox.
- Named Sandboxes are unique within a deployed app, and `from_name` only finds currently running Sandboxes.
- Clean up client-side connections with `detach()` when done.

Local examples:

- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/main.py:95` looks up a named Sandbox and creates it if absent.
- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/main.py:97` creates a Sandbox with image, secrets, Volume, env, idle timeout, timeout, and name.
- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/proxy.py:33` validates live Sandbox identity before proxying a secret-backed API request.

Sources:

- Sandboxes guide: https://modal.com/docs/guide/sandboxes
- Sandbox snapshots guide: https://modal.com/docs/guide/sandbox-snapshots

## Snapshots and Memory Snapshots

There are two related but different ideas:

1. Sandbox snapshots preserve Sandbox filesystem or runtime state for later Sandbox restoration.
2. Function Memory Snapshots reduce cold starts by restoring Function container memory after initialization.

Sandbox snapshots:

- Filesystem Snapshots are Images and persist indefinitely until explicitly deleted.
- Directory Snapshots are beta and expire 30 days after last use.
- Sandbox Memory Snapshots are alpha and expire 7 days after creation.
- Sandbox Memory Snapshots clone memory plus filesystem state, but current limitations include no GPUs for snapshot-enabled/restored sandboxes, closed TCP connections on snapshot, snapshotting terminates the sandbox, and exact instance-type restoration requirements.

Function Memory Snapshots:

- Enable with `enable_memory_snapshot=True` and deploy the app.
- Put pre-snapshot init in `@modal.enter(snap=True)`.
- CPU snapshots capture CPU memory.
- GPU Memory Snapshots are alpha and require `experimental_options={"enable_gpu_snapshot": True}`.
- GPU snapshots can help skip import/JIT/warmup work but do not speed up model loading from storage.
- GPU snapshots are generally incompatible with practical multi-GPU code and may require code changes.
- CPU-only snapshots block GPU access during the snapshot phase; split CPU prep and GPU init if needed.

CurvyTron pattern:

- Use filesystem/directory snapshots for prepared sandboxes when branching many replay/eval variants from the same runtime state.
- Use Function Memory Snapshots only after measuring cold start costs for JAX/PyTorch imports, Triton/DeepGEMM compilation, or model warmup.
- Store snapshot ids in run metadata or a small Dict, but keep durable source/replay/checkpoint proof in Volumes/buckets/manifests.
- Do not make memory snapshots the only way to recover an experiment.

Local examples:

- `/Users/shankha/modal-projects/flash-projects/lovable/docs/SOURCE_RECONSTRUCTION_BOUNDARY.md:245` treats dependency install/build warm state as a Modal sandbox snapshot or hot cache, but only as an execution accelerator.
- `/Users/shankha/modal-projects/flash-projects/lovable/docs/SOURCE_RECONSTRUCTION_BOUNDARY.md:287` says filesystem/directory snapshots should accelerate execution and fanout, not replace manifests or provenance.
- `/Users/shankha/modal-projects/flash-projects/lovable/docs/SOURCE_RECONSTRUCTION_BOUNDARY.md:293` cautions that memory snapshots are not durable source proof.

Sources:

- Sandbox snapshots guide: https://modal.com/docs/guide/sandbox-snapshots
- Function Memory Snapshots: https://modal.com/docs/guide/memory-snapshots

## GPU Functions

Use GPUs for batched neural inference/training, not for unbatched micro-calls. Current Modal GPU docs list GPU strings including `T4`, `L4`, `A10`, `L40S`, `A100`, `A100-40GB`, `A100-80GB`, `RTX-PRO-6000`, `H100`/`H100!`, `H200`, and `B200`/`B200+`. GPU counts are requested with `:n`; current docs say many types support up to 8 GPUs on one physical machine, with multi-node training in private beta.

CurvyTron GPU progression:

1. CPU simulator benchmark on Modal.
2. GPU package smoke: JAX/PyTorch sees the device.
3. Synthetic MCTS/model benchmark with fixed shapes and batch sizes 64/256/1024.
4. Integrated self-play batch with environment stepping local to the GPU container.
5. Scale out only after GPU utilization, power, and profiler traces show the bottleneck.

GPU selection:

- Start with L40S/A100 for early debugging if memory is enough.
- Use H100/H200/B200 only when profiling shows a real throughput need.
- Pin `H100!` or exact A100 size only when benchmarking requires avoiding automatic upgrades.
- Avoid multi-GPU first unless a single-container benchmark shows batch size/model size demands it.

Concurrency and batching:

- `@modal.concurrent` is good for I/O-heavy endpoints and continuous batching engines such as vLLM/SGLang. It is not a free win for CPU-bound simulator code.
- `@modal.batched` can batch independent Function inputs for GPU throughput, but CurvyTron MCTS should usually batch inside the trainer/search process to avoid Function-call granularity.
- Monitor GPU utilization and power, but do not treat utilization alone as proof of efficient FLOP/memory throughput. Profile.

Local examples:

- `/Users/shankha/modal-projects/flash-projects/lovable/deployments/qwen3_8b.py:69` serves SGLang from a Modal `Cls` with `gpu="a100-80gb:1"`, Volume cache, Secret, `@modal.enter` warmup, and concurrency.
- `/Users/shankha/modal-projects/flash-projects/lovable/serving/modal_sglang.py:78` runs a long-lived GPU Function with 24-hour timeout, Volume cache, scaledown window, and a readiness/warmup loop.
- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/main.py:248` uses `@modal.concurrent(max_inputs=100)` for an ASGI Slack app; this is an I/O-heavy web pattern, not a CPU training-loop pattern.

Sources:

- GPU acceleration: https://modal.com/docs/guide/gpu
- GPU metrics: https://modal.com/docs/guide/gpu-metrics
- Input concurrency: https://modal.com/docs/guide/concurrent-inputs
- Dynamic batching: https://modal.com/docs/guide/dynamic-batching

## Secrets

Use Modal Secrets for credentials. Inject them with `secrets=[...]` on `@app.function` or `@app.cls`.

CurvyTron uses:

- W&B/TensorBoard cloud credentials if used.
- Hugging Face tokens for model pulls.
- S3/R2/GCS credentials when not using OIDC.
- Modal token secret only when a container must call Modal APIs itself.

Rules:

- Keep secrets out of Sandboxes unless the sandbox truly needs them.
- Prefer proxy patterns for untrusted code: trusted Function holds the secret and authorizes sandbox requests by sandbox id or run id.
- Modal docs limit key/value pair sizes; put larger config blobs in a Volume and pass a ref.
- Separate reusable secrets so later secrets in a list do not accidentally override earlier keys.

Local examples:

- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/main.py:21` loads Slack credentials from a named Secret.
- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/proxy.py:10` holds the Anthropic API key in a proxy Function rather than handing it to the Sandbox.
- `/Users/shankha/modal-projects/flash-projects/lovable/deployments/qwen3_8b.py:89` injects a Hugging Face Secret for model serving.

Sources:

- Secrets guide: https://modal.com/docs/guide/secrets
- Secret reference: https://modal.com/docs/reference/modal.Secret

## Long-Running Jobs

Modal Function executions default to 5 minutes and can run up to 24 hours. Long-running training must be resumable because Functions are preemptible by default and preempted work is restarted on the same input.

CurvyTron long-run contract:

- Every training call starts by loading `latest.json` or a specified checkpoint ref.
- Every replay actor writes immutable chunks and returns the list of chunk refs.
- Checkpoints are written at fixed step/time intervals and on graceful exit.
- The checkpoint includes model weights, optimizer state, replay cursor or replay manifest, RNG state, environment config hash, code/version metadata, and trainer progress.
- Retries must be idempotent. A retried job should detect existing complete attempt output and either resume or write to a fresh attempt folder.
- Use `@modal.exit()` for cleanup and final checkpoint attempts, but do not rely on exit alone for progress.
- Use small time-sliced jobs for harvest/replay conversion where possible. A schedule can drain a queue of chunks and update compact checkpoints.

Local examples:

- `/Users/shankha/modal-projects/flash-projects/lovable/00_START_HERE.md:199` documents retry collision after preemption and the fix: per-attempt folders plus exact artifact refs.
- `/Users/shankha/modal-projects/flash-projects/lovable/docs/SOURCE_RECONSTRUCTION_BOUNDARY.md:582` recommends scheduled Modal for eventual long-running harvest, with remote-first Volume storage and time-slice/root-chunk partials.
- `/Users/shankha/modal-projects/flash-projects/lovable/serving/modal_sglang.py:78` sets a 24-hour timeout for a long-lived model endpoint function.

Sources:

- Timeouts: https://modal.com/docs/guide/timeouts
- Preemption: https://modal.com/docs/guide/preemption
- Long training example: https://modal.com/docs/examples/long-training
- Failures and retries: https://modal.com/docs/guide/retries

## Scheduling

Modal supports schedules on deployed functions with `modal.Period` and `modal.Cron`.

Use schedules for:

- Nightly evaluation of latest checkpoint against fixed opponents.
- Replay compaction and promotion from Volume to bucket.
- Stale-run watchdogs.
- Warm-pool autoscaler adjustments for demo/serving apps.
- Periodic smoke tests that validate the deployed training app and storage.

Cautions:

- Schedules require deployed Apps.
- `modal.Period` resets on redeploy; use `modal.Cron` for wall-clock schedules not disturbed by deploys.
- Current docs say schedules cannot be paused; remove the schedule and redeploy instead.
- Scheduled jobs should be idempotent and should check a lock/lease before launching expensive work.

Sources:

- Scheduling and cron jobs: https://modal.com/docs/guide/cron
- Cron reference: https://modal.com/docs/reference/modal.Cron

## Experiment, Replay, and Checkpoint Pattern for CurvyTron RL

Recommended first Modal shape:

```text
curvytron-rl-train deployed app

  build_env_benchmark(config_ref) -> metrics
  gpu_smoke(config_ref) -> device_info
  mctx_synthetic_benchmark(config_ref) -> throughput_report
  train_run(run_ref) -> final_summary_ref
  selfplay_shard(shard_spec) -> replay_chunk_refs
  evaluate_checkpoint(eval_spec) -> eval_summary_ref
  compact_replay(compaction_spec) -> compacted_refs
  promote_artifacts(promote_spec) -> bucket_manifest_ref
```

Run identity:

```json
{
  "run_id": "curvytron-muzero-v0-20260508-001",
  "code_ref": "<git sha or package version>",
  "env_config_hash": "<sha256>",
  "algo_config_hash": "<sha256>",
  "modal_app": "curvytron-rl-train",
  "storage_root": "/experiments/<run_id>"
}
```

Replay chunk contract:

- Chunk by actor, generation, and monotonic part id.
- Include env config hash and policy/checkpoint ref in every chunk header.
- Store observations/actions/rewards/dones/search_policy/value_targets/player_perspective/seed.
- Prefer medium chunks: large enough to avoid file-count problems, small enough for partial recovery.
- Maintain `replay/index.jsonl` or per-shard manifests.
- Readers consume only chunks listed in committed manifests.

Checkpoint contract:

- Write checkpoint payload under a unique directory.
- Validate loadability in a separate Function before marking it latest.
- Publish `latest.json` with checkpoint path, training step, replay index version, metrics summary, and config hashes.
- Keep `best.json` separate from `latest.json`.
- Do not mutate old checkpoint directories.

Evaluation contract:

- Evaluation is its own Modal Function over fixed seed ranges.
- Store raw match summaries and compact aggregate metrics.
- Store small videos/gifs only for sampled games and failure cases.
- Compare raw policy vs MCTS-improved policy.
- Run evaluation on a schedule and after each accepted checkpoint.

Sweep contract:

- Use `Function.map`/`starmap` for bounded sweeps.
- Use `max_containers` to cap cost.
- Each sweep cell writes to an independent path.
- Local output is a compact summary with remote refs.

Failure/retry contract:

- Every job has `attempt_id`.
- Outputs land under `attempts/<attempt_id>/`.
- A job writes `complete.json` only after all expected artifacts exist.
- On retry, discover the latest complete attempt or resume from the latest checkpoint.
- Treat "return code 0" as insufficient; verify artifact postconditions.

Local examples:

- `/Users/shankha/modal-projects/claude-slack-gif-creator/docs/experiment-footguns.md:88` notes that return code 0 is not enough and artifact postconditions matter.
- `/Users/shankha/modal-projects/flash-projects/lovable/00_START_HERE.md:61` says command completion alone is not a pass.
- `/Users/shankha/modal-projects/flash-projects/lovable/00_START_HERE.md:311` describes bounded artifact fetches from Modal run Volume instead of whole-tree downloads.
- `/Users/shankha/modal-projects/flash-projects/lovable/docs/SOURCE_RECONSTRUCTION_BOUNDARY.md:497` records a remote-first Volume worker that commits per project and returns compact checkpoints.

## Source Index

Official Modal docs:

- Apps, Functions, and entrypoints: https://modal.com/docs/guide/apps
- App reference: https://modal.com/docs/reference/modal.App
- Function reference: https://modal.com/docs/reference/modal.Function
- Invoking deployed functions: https://modal.com/docs/guide/trigger-deployed-functions
- Scaling out: https://modal.com/docs/guide/scale
- Batch processing: https://modal.com/docs/guide/batch-processing
- Job processing: https://modal.com/docs/guide/job-queue
- Images: https://modal.com/docs/guide/images
- Existing images: https://modal.com/docs/guide/existing-images
- Volumes: https://modal.com/docs/guide/volumes
- Volume reference: https://modal.com/docs/reference/modal.Volume
- Cloud bucket mounts: https://modal.com/docs/guide/cloud-bucket-mounts
- CloudBucketMount reference: https://modal.com/docs/reference/modal.CloudBucketMount
- Queues: https://modal.com/docs/guide/queues
- Queue reference: https://modal.com/docs/reference/modal.Queue
- Dicts: https://modal.com/docs/guide/dicts
- Dict reference: https://modal.com/docs/reference/modal.Dict
- Sandboxes: https://modal.com/docs/guide/sandboxes
- Sandbox snapshots: https://modal.com/docs/guide/sandbox-snapshots
- Memory Snapshots: https://modal.com/docs/guide/memory-snapshots
- GPU acceleration: https://modal.com/docs/guide/gpu
- GPU metrics: https://modal.com/docs/guide/gpu-metrics
- Input concurrency: https://modal.com/docs/guide/concurrent-inputs
- Dynamic batching: https://modal.com/docs/guide/dynamic-batching
- Secrets: https://modal.com/docs/guide/secrets
- Scheduling and cron jobs: https://modal.com/docs/guide/cron
- Timeouts: https://modal.com/docs/guide/timeouts
- Preemption: https://modal.com/docs/guide/preemption
- Failures and retries: https://modal.com/docs/guide/retries
- Long training example: https://modal.com/docs/examples/long-training

Local examples inspected:

- `/Users/shankha/curvy/curvytron_muzero_modal_handoff.md`
- `/Users/shankha/curvy/docs/research/training_architecture_notes.md`
- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/runtime_common.py`
- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/main.py`
- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/proxy.py`
- `/Users/shankha/modal-projects/claude-slack-gif-creator/docs/experiment-footguns.md`
- `/Users/shankha/modal-projects/flash-projects/lovable/00_START_HERE.md`
- `/Users/shankha/modal-projects/flash-projects/lovable/docs/SOURCE_RECONSTRUCTION_BOUNDARY.md`
- `/Users/shankha/modal-projects/flash-projects/lovable/deployments/qwen3_8b.py`
- `/Users/shankha/modal-projects/flash-projects/lovable/serving/modal_sglang.py`
- `/Users/shankha/modal-projects/flash-projects/modal-decagon/training/benchmarks/volume_throughput/archive/exp_large_pipeline.py`
- `/Users/shankha/modal-projects/flash-projects/modal-decagon/training/benchmarks/volume_throughput/archive/exp_i6pn_streaming.py`

## Implementation Checklist

- Define the deployed Modal app names and environments.
- Create one CPU env benchmark Function before any MuZero work.
- Create one GPU package/device smoke Function before any training run.
- Implement storage wrappers over Volume/bucket refs before training.
- Define replay chunk, checkpoint, run manifest, and evaluation schemas.
- Add hot-loop tests that assert no Queue/Dict/Function calls occur inside env step, MCTS expansion, or replay sampling.
- Add retry/resume tests with duplicate attempt folders.
- Add a scheduled evaluation Function only after checkpoint manifests are stable.
- Revisit Function Memory Snapshots after measuring cold start profiles.
