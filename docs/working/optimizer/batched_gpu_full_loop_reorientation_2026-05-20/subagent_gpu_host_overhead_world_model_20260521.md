# GPU Host-Overhead World Model

Date: 2026-05-21

Status: docs-level research note. No production code, training launch, trainer
default, tournament default, or checkpoint path was changed.

## Plain Model

A GPU win becomes a training win only when enough of the loop stays batched and
resident long enough to amortize the host boundary.

GPUs are throughput machines. They like doing the same shaped work over many
elements at once. A small render or model call for one row can be slower than a
CPU path because the host still has to enqueue GPU work, move data if the tensor
lives on CPU, and sometimes wait for the result. A large `[B, players, ...]`
batch gives the GPU enough parallel work to hide latency and use memory
bandwidth.

Most CUDA/PyTorch/JAX GPU work is queued asynchronously from the host. The CPU
asks the GPU to run kernels or copies, then can continue until it needs a value
back. The moment host code reads a device result, converts it to NumPy, calls
`.item()`, copies a tensor to CPU, times with a forced synchronize, or otherwise
needs a completed device value, the queue must drain far enough to make that
value real on the host.

Host-to-device (`H2D`) and device-to-host (`D2H`) copies are not just bandwidth
math. They also have fixed per-transfer overhead and ordering constraints. One
contiguous 64 MiB copy is very different from 1024 Python-mediated 64 KiB row
copies, even though the payload bytes are the same. The batched copy has one
enqueue/copy boundary and can sometimes overlap with other work; the scalar row
path has many object constructions, possible tiny copies, cache misses, and
sync/readback chances.

Stable external facts behind this model:

- NVIDIA describes CUDA as host CPU plus device GPU with distinct memories, and
  warns that transfers should be minimized and data kept on device when possible:
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#what-runs-on-a-cuda-enabled-device>
- NVIDIA's best-practices guide says batching many small transfers into one
  larger transfer can perform much better because each transfer has overhead:
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device>
- CUDA kernel launches and many copy operations are asynchronous with respect to
  the host; async copies involving non-page-locked host memory may become
  synchronous:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution>
- PyTorch notes that CUDA execution is usually asynchronous to the caller, but
  CPU/GPU copies and explicit `torch.cuda.synchronize()` are synchronization
  points; pinned memory plus `non_blocking=True` can allow async copies:
  <https://docs.pytorch.org/docs/2.12/notes/cuda.html#cuda-semantics>

## CurvyTron Tensor Sizes

Current resident/probe shape:

```text
[B, 2, 4, 64, 64]
 = B * 2 players * 4 stack frames * 64 * 64 pixels
 = B * 32,768 elements
```

Approximate payload sizes, using binary MiB:

| B | rows (`B*2`) | uint8 `[B,2,4,64,64]` | float32 `[B,2,4,64,64]` |
| ---: | ---: | ---: | ---: |
| 64 | 128 | 2 MiB | 8 MiB |
| 512 | 1024 | 16 MiB | 64 MiB |
| 1024 | 2048 | 32 MiB | 128 MiB |

One scalar LightZero policy row is `[4,64,64]`:

| dtype | one scalar row | B512 total scalar rows |
| --- | ---: | ---: |
| uint8 | 16 KiB | 16 MiB across 1024 rows |
| float32 | 64 KiB | 64 MiB across 1024 rows |

The B512 float32 scalar path and the B512 float32 batched path can both move
about 64 MiB of observation data. They do not cost the same. The scalar path
also creates or copies 1024 row objects, action masks, rewards, dones, infos,
and possibly pickle payloads. It creates many opportunities for Python overhead
and accidental synchronization. The batched path can present the same bytes as
one contiguous tensor/array boundary, or keep it on device and avoid the copy
entirely until an outer consumer truly needs host data.

This is why the current profile-only resident chunk canary matters. At
B512/A16/sim8 it reported roughly `10.98k` scalar roots/sec on H100 and `5.84k`
on L4/T4, while explicitly not calling stock `train_muzero`. That number is not
training throughput. It is evidence that a resident batched chunk can create
headroom that a scalar LightZero-shaped edge can destroy.

## Current Or Likely Sync Points

From the local boundary docs and profile code shape, the current GPU lane is
still a GPU island inside a host-owned loop:

- CPU env/source state is packed into compact render state before H2D.
- The JAX render path copies compact state to device, runs the batched render,
  then uses a readiness/readback boundary before returning NumPy frames.
- Rendered `uint8` frames are read back to CPU, then pushed into a host stack.
- Current trainer-visible stacks and LightZero timesteps are host arrays/objects.
- Scalar materialization splits `[B,2,...]` back into one row per acting
  LightZero env/player row.
- Any `np.asarray(device_array)`, `.block_until_ready()`, `.cpu()`, `.numpy()`,
  `.item()`, or explicit CUDA/JAX synchronization in profiling can turn queued
  device work into host blocking time.
- Stock LightZero collection/search/replay/learner boundaries are still likely
  shaped around Python objects, NumPy arrays, subprocess IPC, and scalar env ids.
- RND is separate but can add its own GPU/CPU train, estimate, metric snapshot,
  and state-hash boundaries depending on cadence.

Practical interpretation: a fast batched renderer can improve observation cost
without making the whole training loop fast, because the full loop still pays
env manager, scalar timestep, search, replay, learner, RND, checkpoint/eval, and
IPC costs.

## Project Heuristics

Use these as working thresholds, not laws:

- If deleting observation entirely improves full-loop throughput by only
  `~1.2x`, renderer-only work cannot produce a `5x-10x` training win. It can
  still be worth doing for headroom, but the next large win must come from
  residency, batching, actor topology, search/replay pressure, or fewer scalar
  boundaries.
- For B512, treat `16 MiB` uint8 and `64 MiB` float32 as the first-order payload
  sizes. If profile bytes are much larger, scalar object/pickle duplication or
  dtype expansion is probably happening.
- Prefer one batched copy over many scalar row copies when host transfer is
  unavoidable. Prefer no copy when the next consumer can run on the same device.
- A sync per batch can be acceptable if it encloses substantial GPU work. A sync
  per row is usually suspect at `B=512` because it creates up to 1024 waits per
  outer step.
- H2D/D2H copy time alone can look small while host overhead is still large.
  Track copy bytes, sync count, scalar materialization, object construction, and
  queue wait/block time together.
- Pinned/nonblocking copies are useful only when the surrounding code lets copy
  and compute overlap. They do not fix a design that immediately reads the data
  back on the host.
- Synthetic resident probes should remain labeled synthetic until they add real
  or realistic policy/search/replay pressure and report in the same denominator
  as stock trainer rows.
- H100 beating L4/T4 in a resident probe does not prove the real trainer is
  GPU-bound. Low full-loop GPU utilization or small zero-observation deltas still
  point back to host orchestration and scalar boundaries.

## Telemetry To Add Or Keep

Keep this minimal first. Five fields that would make future rows much easier to
read:

1. `gpu_h2d_bytes_total`: total host-to-device bytes per profiled batch/window.
2. `gpu_d2h_bytes_total`: total device-to-host bytes per profiled batch/window.
3. `gpu_sync_count`: count of explicit or known blocking readiness/readback
   points in the measured path.
4. `gpu_sync_block_sec`: wall time spent waiting at explicit readiness/readback
   points.
5. `scalar_timestep_materialize_sec`: time to split batched stack/action/reward
   data into LightZero-shaped scalar timesteps.

Nice-to-have later: `scalar_timestep_count`, `scalar_payload_bytes`,
`host_stack_update_sec`, `probe_device_work_sec`, `probe_normalize_sec`,
`replay_device_resident_bytes`, and separate RND collect/estimate/train timers.

## Bottom Line

The useful mental model is not "GPU rendering is fast, so training is fast."
It is:

```text
training speed =
  batched useful GPU work
  - host/device transfer overhead
  - synchronization/readback overhead
  - scalar Python/NumPy object overhead
  - manager/search/replay/learner/RND/I/O overhead
```

The next architecture research should ask whether CurvyTron can keep
`[B,2,4,64,64]` stacks, policy/search inputs, and replay-shaped samples batched
and device-resident across more than one tiny island. If the design falls back
to scalar host rows immediately after render, the renderer win is real but the
full-loop training win will remain bounded.
