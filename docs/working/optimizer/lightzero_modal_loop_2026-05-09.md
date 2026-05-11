# LightZero Modal Loop

Date: 2026-05-09

Status: optimizer working memory. This is a speed/setup note, not a
training-quality claim.

## 2026-05-10 Update

The coach lane now reports a weak but real-looking movement signal on the
faithful-short installed-package Atari Pong control: `iteration_0` was around
`-13`, final `iteration_3697` was `-8` under stock-ish eval and `-5` in manual
telemetry. This is not solved Pong and not a reproduction claim, but it changes
the optimizer stance: LightZero is worth profiling as the current concrete slow
loop, not just auditing as a possible dead end.

The active scale probe is:

```text
attempt_id: train-faithful-short-installed-0.2.0-s0-32768-relpath
app: ap-xiGLACKHPZLvL1eYgygqvm
compute: gpu-l4-t4
purpose: bounded scale/accounting plus live checkpoint eval
claim: no learning-quality claim from optimizer
```

Sub-agent wave requested on 2026-05-10:

- deep disaggregation review: whether the one-function Modal architecture
  should be split and what measurements would justify it;
- LightZero internals map: collector/env/replay/search/learner/evaluator
  placement and instrumentation hooks;
- CPU/GPU Amdahl map: what likely runs on host versus GPU and what evidence is
  missing;
- CurvyTron transfer critique: what lessons carry to repo-native simultaneous
  `[B, P]` and what does not.

Local quick probes run the same day reinforced the generic Amdahl warning:
repo-native source/trainer profiles are currently observation-packing-heavy
with tiny CPU policies, while synthetic search becomes dominant as soon as the
decision box gets heavier. These probes do not rank the LightZero bottleneck;
they say we need real LightZero collect/search/replay/learner timings.

Implementation note: the exact reproduction wrapper now has opt-in phase
profiling wired through the CLI/remote functions. Defaults remain off, so
existing train behavior is unchanged. The first profiler patch was hardened
after review: it now patches selected LightZero methods in place instead of
replacing whole classes, records installed hooks, restores partial installs on
failure, checks both `lzero.mcts` plus `train_muzero.__globals__` for
GameBuffer classes, resolves inherited method owners, and guards profiler-only
count extraction so it should not break training after the underlying LightZero
method succeeds. Local `py_compile`, `ruff`, and a fake-LightZero profiler
smoke pass, including inherited GameBuffer methods. The profile is intended for
a fresh hook-stopped profile/control rung, not for mutating the active coach
run.

Correction after the first profile attempt: `max_env_step=2048` was still the
wrong profiling control. The run reached learner iteration `1300` and was
stopped manually by killing the local Modal process and stopping app
`ap-CLIw2m3bXwNbKVHItDQP33`. This was too training-shaped for optimizer timing.
Later correction: `max_train_iter_override` also failed to stop the hot loop
cheaply. Future stock-loop phase profiles should use
`profile_stop_after_learner_train_calls`, or preferably a direct in-container
one-collect/sample/train harness with no evaluator when steady-state timing is
the goal.

## Short Read

The current serious LightZero control lane runs as one Modal training function
with one mounted Volume and one cheap GPU allocation. That is probably the
right shape until profiling proves otherwise.

Do not split env steps, MCTS calls, replay rows, or learner updates across
Modal network primitives yet. Modal `Queue` and `Dict` calls have network
latency on the order of tens of milliseconds, which is too large for per-action
or per-search-node work. Use Modal for coarse jobs: train attempt, eval pass,
artifact scan, checkpoint diff, summary.

The immediate optimizer job is to instrument the single-container LightZero
loop, not redesign it.

Plain answer for now: yes, it should stay one Modal training function until a
profile proves a specific split is worth the latency, staleness, and
operational complexity.

## Current Code Path

Main stock-ish reproduction wrapper:

- `src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py`
- app: `curvyzero-lightzero-pong-exact-reproduction`
- image installs `LightZero==0.2.0`, `opencv-python-headless`, and
  `AutoROM[accept-rom-license]`, then runs `AutoROM --accept-license`
- local `src/` is copied to `/repo/src`; `PYTHONPATH=/repo/src`
- Volume: `curvyzero-runs` mounted at `/runs`
- train mode changes cwd to `/runs` and sets a relative `exp_name` so
  LightZero/DI-engine checkpoint paths stay in the Volume

GPU train function:

```text
cpu=8
memory=32768
gpu=["L4", "T4"]
timeout=18h
```

CPU function is for dry/config validation and blocks CPU training by policy.

Stock-ish settings captured by the wrapper:

```text
env_id=PongNoFrameskip-v4
env_manager=subprocess
collector_env_num=8
evaluator_env_num=3
num_simulations=50
batch_size=256
update_per_collect=None
replay_ratio=0.25
game_segment_length=400
cuda=True
installed LightZero==0.2.0 max_env_step=200000
faithful-short override currently uses smaller train_muzero(max_env_step=...)
```

Distinct profiler/control path: `profile_stop_after_learner_train_calls` stops
after bounded learner work. `max_train_iter_override` is retained as older
scaffolding, but it did not reliably cap the actual hot learner loop. Any run
using these profiler controls is non-exact and not coach learning evidence.

The completed `8192` faithful-short run was on L4 with CUDA available. It
reported about `1326s` remote elapsed, overshot collection to `14791` env
steps, and wrote three intended checkpoints totaling about `256MB`.

That is roughly `11` collected env steps/sec using the overshot collector count
and remote elapsed. This rough rate is only an accounting clue because it
includes setup, initial eval, collection, replay/targets, learner updates,
checkpointing, artifact scans, and any idle time. Do not call it env-step speed.

## Where The Pieces Live

Inside the same Modal container/process, LightZero `train_muzero` creates:

- subprocess Atari collector env manager;
- subprocess Atari evaluator env manager;
- MuZero policy with learn/collect/eval modes;
- DI-engine `BaseLearner`;
- LightZero `MuZeroGameBuffer`;
- LightZero `MuZeroCollector`;
- LightZero `MuZeroEvaluator`;
- TensorBoard logger and checkpoint hooks.

The main LightZero loop is:

```text
maybe eval
collector.collect(...)
replay_buffer.push_game_segments(...)
replay_buffer.remove_oldest_data_to_fit()
update_per_collect = calculate_update_per_collect(...)
repeat update_per_collect:
  replay_buffer.sample(batch_size, policy)
  learner.train(train_data, collector.envstep)
  maybe replay_buffer.update_priority(...)
stop when collector.envstep >= max_env_step or train_iter >= max_train_iter
```

The Atari env workers are subprocesses inside the same Modal function, not
separate Modal functions. Replay is in Python process memory, not a Modal
Queue. Checkpoints/logs are regular files under the mounted Volume path.

## CPU Versus GPU

Known:

- Atari env stepping, ALE preprocessing, subprocess env manager orchestration,
  replay bookkeeping, Python object movement, artifact scans, and Volume
  commit/reload behavior are CPU/host-side work.
- MuZero network inference and learner updates should use GPU when
  `cfg.policy.cuda=True` and `torch.cuda.is_available()`.
- The current exact wrapper verifies CUDA and records device count/name.
- Eval can accidentally be CPU or GPU depending on the eval wrapper. The
  corrected `8192` stock-ish eval ran CUDA false because that wrapper path used
  CPU.

Likely but not yet measured:

- MCTS tree bookkeeping is at least partly CPU-side, even if model inference is
  on GPU.
- `MuZeroGameBuffer.sample` and target construction mix NumPy/Python work with
  target-model inference.
- LightZero `collector.collect(...)` probably mixes Atari subprocess stepping,
  policy/search calls, model inference, Python tree bookkeeping, and game
  segment construction; treating it as one opaque bucket is not enough.
- Host/device transfer may matter because observations and hidden states cross
  CPU/GPU boundaries around search, replay sampling, and learner update.
- GPU utilization may be low if the loop waits on CPU env/search/replay or if
  model calls are too small.

Do not decide CPU-vs-GPU optimization by intuition. Add timers and GPU sampling.

## Amdahl Read

The observed `8192` faithful-short wall clock is slow enough to care, but the
current summary does not split the wall clock into useful buckets. A 10x
environment optimization is irrelevant if the dominant bucket is MCTS/model
search, replay/target construction, learner update, checkpoint/eval, or setup.

The first LightZero profile must split:

- setup/import/config/env creation;
- initial eval and periodic eval;
- collector time;
- env steps and episodes collected;
- MCTS/search/policy-action time if exposed;
- replay push/remove time;
- replay sample/target construction time;
- learner train/update time;
- checkpoint save time and bytes;
- artifact scan/Volume commit time;
- GPU utilization and memory during collect/search/learn;
- envstep/sec, train_iter/sec, learner updates/sec, and wall-clock overshoot.

Until that exists, bottleneck ranking is unknown.

Minimum useful profile row:

```text
wall_sec
setup_sec
eval_sec
collect_sec
replay_push_sec
replay_sample_target_sec
learner_train_sec
checkpoint_sec
artifact_scan_commit_sec
collector_envstep_delta
learner_train_iter_delta
updates_per_collect
gpu_util_p50/p95 or sampled_busy_fraction
checkpoint_count/bytes
```

If only one hook can be added first, wrap the top-level LightZero loop phases:
`evaluator.eval`, `collector.collect`, replay push/remove, replay sample,
`learner.train`, checkpoint save, final artifact scan. That gives Amdahl
shares without requiring a full rewrite of LightZero internals.

## Disaggregation Verdict

Keep as one training Modal function for now.

Clean separations that are already good or probably good:

- train attempt as one Modal job;
- eval over checkpoints as separate Modal jobs;
- checkpoint diff/probe as separate Modal jobs;
- artifact summarization and manifest repair as separate Modal jobs;
- local/client orchestration only at run boundaries.

Do not split yet:

- one env step per Modal call;
- MCTS node/model calls over Modal Queue/Dict/function boundaries;
- replay row streaming through Modal Queue;
- learner sampling directly from a networked Modal primitive;
- per-step opponent inference in another Modal function.

Possible future splits, gated by measurement:

- CPU actor processes plus central GPU inference/search worker, only if
  policy/search dominates and batching beats added latency;
- separate learner process/container, only if actors are CPU-bound, replay
  chunks are large, policy staleness is bounded, and Queue/Volume transfer cost
  is measured;
- multi-GPU single-container learner/search, only if GPU-bound and LightZero or
  a project-owned runner can actually use it;
- Volume v2 or object-store replay/checkpoint layout, only if file count,
  checkpoint burst, or Volume commit/reload cost becomes a measured bottleneck.

## Instrumentation Hooks

Best next target is an iteration-capped profiled LightZero control rung, not a
bigger unprofiled run and not a `max_env_step` faithful-short profile.

Currently wired in `lightzero_pong_exact_reproduction.py` when
`--profile-phases` is enabled:

- wrap or monkeypatch `MuZeroCollector.collect`;
- wrap `MuZeroGameBuffer.push_game_segments`;
- wrap `MuZeroGameBuffer.remove_oldest_data_to_fit`;
- wrap `MuZeroGameBuffer.sample`;
- wrap `MuZeroGameBuffer.update_priority`;
- wrap `BaseLearner.train`;
- wrap `BaseLearner.save_checkpoint`;
- wrap learner hooks;
- wrap `MuZeroEvaluator.eval`;
- record collect calls, `collector.envstep` deltas, replay calls, learner
  train calls, checkpoint-save calls, final artifact scan time, and sampled
  `nvidia-smi` GPU utilization/memory.

Still missing or opaque:

- env manager `step` timing inside `collector.collect`;
- MuZero policy collect/eval/learn forward timing;
- model `initial_inference` and `recurrent_inference` timing;
- CUDA event timing for model/search/learner work;
- exact host/device transfer timing;
- MCTS tree bookkeeping split from model inference;
- replay target construction sub-splits inside `GameBuffer.sample`;
- Volume commit timing around progress snapshots and final summary writes.

If the first profile shows collect/search dominates, the next best deep hook is
inside `MuZeroMCTSCtree.search`: split C++ tree traverse/backprop, latent-state
gathering, H2D tensor creation, `model.recurrent_inference` CUDA time,
D2H/detach-to-NumPy, and root action selection. Pair that with a MuZero policy
hook around root `initial_inference` in collect/eval. Do this only after the
coarse profile says search deserves deeper surgery.

For a cheap optimizer profile, cap learner iterations directly:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-profiler-iter20-installed-0.2.0-s0-v0 \
  --progress-interval-sec 30 \
  --max-train-iter-override 20 \
  --profile-phases \
  --gpu-sample-interval-sec 5
```

Use an even smaller cap first if all we need is hook sanity. The key correction
is to cap learner iterations directly; `max_env_step` alone can still allow a
surprisingly long profile. Runs with `max_train_iter_override` are profile/
control runs, not exact reproduction or coach learning evidence.

Second correction: `max_train_iter_override=5` still did not cap the actual
hot learner loop. It reached `Training Iteration 600` before app
`ap-xkTDXj5wNV8DiwVvLFpYJ9` was stopped. The next profile should use the
profiler hook cap:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 0 \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-profiler-traincalls5-installed-0.2.0-s0-v0 \
  --progress-interval-sec 15 \
  --profile-phases \
  --gpu-sample-interval-sec 2 \
  --profile-stop-after-learner-train-calls 5
```

This is still heavier than a pure direct harness because it includes stock
startup/eval/first collect, but it should stop with a final summary after
bounded learner work.

Preferred next path after the second abort: build a direct one collect + replay
sample + learner train harness that copies `train_muzero` setup, omits
`Evaluator`, and labels itself profile-control. Stock `train_muzero` has an
unconditional initial eval, so hook-stopped stock-loop profiling is useful only
for stock-entry accounting, not for cheap steady-state timing.

Deep-hook rule from the profiler-hook review: next add discovery metadata
inside the Modal image before new monkey patches. Candidate targets are
`MuZeroPolicy._forward_collect`, `_forward_eval`, runtime model
`initial_inference`/`recurrent_inference`, `MuZeroMCTSCtree.search`,
env-manager `step`, and `MuZeroGameBuffer.sample` helper candidates. Add actual
deep timers only after the coarse profile says collect/search or replay sample
is the bucket worth splitting.

Discovery-only metadata has been implemented behind `--profile-phases` as
`phase_profile.candidate_hooks` and `phase_profile.deep_hook_discovery_notes`.
It is import/inspect only; it does not alter LightZero behavior. The next fresh
profile after that patch should verify which owners exist in the Modal image
before deep timers are added.

Do not mutate the active 32768 run unless the coach asks.

## External Modal Constraints

Current Modal docs support the caution:

- Volumes are good for write-once/read-many model artifacts and checkpoints,
  but commits/reloads are explicit and concurrent writes to the same files are
  unsafe. v1 Volumes work best below about `50,000` files.
- Queues are for active-function communication, not durable replay storage;
  each item is capped at `1 MiB`, and queue operations cross the network.
- Dicts are persistent metadata storage, but reads/writes cross the network and
  are recommended for small objects.
- Modal can allocate multiple GPUs per container, but the bottleneck should be
  understood before buying larger GPUs or more GPUs.

Useful current docs:

- Modal Queues: <https://modal.com/docs/guide/queues>
- Modal Dicts: <https://modal.com/docs/guide/dicts>
- Modal Volumes: <https://modal.com/docs/guide/volumes>
- Modal GPU metrics: <https://modal.com/docs/guide/gpu-metrics>
- Modal GPU guide: <https://modal.com/docs/guide/gpu>

## Next Questions

- For the active 32768 run, can we read progress/log artifacts enough to infer
  envstep/sec and train_iter/sec over time?
- Can an iteration-capped profile/control rung wrap LightZero internals cheaply
  enough to rank phase shares?
- Is collect/search or learner/update the largest bucket at stock-ish
  `num_simulations=50`, `batch_size=256`, `collector_env_num=8`?
- Is the L4 GPU busy during collect/search/learn, or mostly idle behind CPU
  env/search/replay work?
- How much wall time is checkpoint/eval/artifact scanning?
- What policy staleness would be acceptable before considering actor/learner
  separation for CurvyTron?
