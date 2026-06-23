# GPU / Host Overhead World Model

Date: 2026-05-21

Scope: optimizer working memory. This document is about how to reason about
GPU batching and host overhead for CurvyTron. It is not Coach launch advice.

## Plain Model

A GPU is fast when it gets a large block of similar work and can run that work
without stopping to ask Python what to do next.

The slow pattern is:

```text
one env row -> Python object -> copy one tiny array -> GPU call -> read result
-> Python object -> repeat thousands of times
```

The fast pattern is:

```text
many env rows -> one tensor batch -> GPU work queue -> one coarse result
-> only read back what the host truly needs
```

The important costs are not only arithmetic. They are also:

- kernel launch overhead: starting GPU work from the host;
- host-to-device copies: CPU memory to GPU memory;
- device-to-host copies: GPU memory back to CPU memory;
- synchronization: waiting for the GPU so Python can inspect a value;
- scalar Python object churn: building thousands of little dicts/timesteps.

The GPU can overlap and queue work. That stops being useful when we repeatedly
force a readback or build scalar Python objects between small kernels.

## CurvyTron Tensor Sizes

Policy observation shape today:

```text
[B, 2, 4, 64, 64]
```

That means:

```text
roots = B * 2
pixels per batch = B * 2 * 4 * 64 * 64 = B * 32768
```

Approximate tensor sizes:

| B | roots | uint8 stack | float32 stack |
| ---: | ---: | ---: | ---: |
| 64 | 128 | `2.0 MiB` | `8.0 MiB` |
| 512 | 1024 | `16.0 MiB` | `64.0 MiB` |
| 1024 | 2048 | `32.0 MiB` | `128.0 MiB` |

One `16 MiB` copy can be fine. Thousands of tiny copies plus Python objects are
not the same thing. The problem is usually not raw PCIe bandwidth alone; it is
bandwidth plus synchronization plus Python scheduling.

## Current CurvyTron Boundary

The current profile-only resident chunk shape is:

```text
compact CurvyTron rows
-> renderer-backed [B,2,4,64,64] uint8 stack
-> GPU resident replay-like write/sample
-> GPU policy/search-shaped synthetic work
-> optional scalar LightZero materialization
```

The trusted Coach training lane is still stock-LightZero-shaped. That means the
real boundary still wants env-id keyed observations and `BaseEnvTimestep` rows.
This is where the batch can die.

The resident profile rows prove that the GPU side can be fast enough. They do
not prove full training speed, because they do not call stock `train_muzero`,
real MCTS, replay target building, learner updates, RND, checkpointing, eval,
GIF, or tournament code.

## Current Measurements

Medium profile-only resident chunk rows, B512/A16/sim8, 60 measured steps, 20
warmup steps, no scalar LightZero materialization:

| compute | scalar roots/sec | physical rows/sec | measured sec | observation sec | resident probe sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| H100 | `10980.47` | `5490.24` | `5.60` | `2.33` | `0.58` |
| L4/T4 | `5839.67` | `2919.83` | `10.52` | `3.44` | `2.87` |

Plain read:

- H100 is much better once the synthetic policy/search-shaped GPU work is large
  enough.
- L4/T4 is still useful, but the resident synthetic work is visibly slower.
- The host/actor wall is still present: actor step plus parent/send/receive is
  seconds across the row, even in a profile-only canary.
- This is still not training speed.

B1024 H100 did not materially improve the older resident synthetic profile ceiling:

```text
B512 scalar-off:  ~10.98k roots/sec
B1024 scalar-off: ~11.07k roots/sec
B512 scalar-on:   ~7.62k roots/sec
B1024 scalar-on:  ~6.75k roots/sec
```

That means the older synthetic resident probe was not limited only by GPU
arithmetic. Do not use this row to reject B1024 generally. The latest
real-checkpoint MCTX B1024 scalar-off row is the clean current optimizer row:
`19,334` steps/sec vs direct CTree `8,792`, speedup `2.20x`.

## Why Scalar Materialization Matters

When scalar materialization is on, the same batch is reshaped into individual
LightZero-style timesteps. That work is useful as an edge measurement, but it
is exactly the shape we are trying not to put in the hot path too early.

Correct target:

```text
Keep scalar materialization outside the hot observation -> policy/search ->
replay loop, or pay it once per chunk.
```

Wrong target:

```text
Draw fast on GPU, then immediately create thousands of Python timestep objects
before policy/search can use the batch.
```

## Practical Heuristics

These are working rules, not laws:

- A single `16-64 MiB` tensor copy per chunk is acceptable if the chunk feeds a
  lot of GPU work.
- A readback every env row is bad. A readback once per chunk for a checksum,
  metric, or checkpoint edge can be fine.
- GPU speedups need high batch size and static-ish shapes. Tiny batches make
  kernel launch and Python overhead dominate.
- H100 should help most when policy/search/replay-shaped GPU work is heavy.
  It cannot fix scalar Python env rows by itself.
- If deleting observation gives only a small full-loop win, the next bottleneck
  is search/collector/replay/host structure, not rendering alone.

## Telemetry To Keep

Every serious optimizer row should report:

- `host_to_device_bytes` and `host_to_device_sec`;
- `readback_bytes` and `readback_sec`;
- scalar materialization time and materialized row count;
- root count, model/search call count, and roots per GPU call;
- actor/parent send-receive wall time separate from GPU device time;
- stack dtype and shape;
- GPU name, memory, and utilization snapshot.

## External Patterns

The literature points the same way:

- CuLE moved Atari emulation and rendering onto GPU and avoided CPU/GPU frame
  traffic by running thousands of games in parallel:
  <https://arxiv.org/abs/1907.08467>
- Brax keeps environment processing and learning on accelerators:
  <https://arxiv.org/abs/2106.13281>
- Pgx uses JAX-native vectorized game state/step functions:
  <https://github.com/sotetsuk/pgx>
- Mctx implements batched JAX-native MCTS:
  <https://github.com/google-deepmind/mctx>
- LightZero provides useful PyTorch/C++ MCTS pieces, but its normal
  environment boundary is still Python/env-manager shaped:
  <https://github.com/opendilab/LightZero>

## Current Recommendation

Do not spend the next pass on isolated renderer kernels unless a profile says
rendering has become dominant again.

Spend the next pass on the batch ownership boundary:

```text
Can we keep [B,2,4,64,64] alive through real policy/search/replay-shaped work
before scalar LightZero row materialization?
```

That is the next possible large speedup. If it fails, the fallback is to use
compiled/parallel CPU actors plus central batched inference/search, not to keep
tuning one-frame rendering.
