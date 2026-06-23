# Resident Torch Input Probe Plan

Date: 2026-05-21

Status: implemented as a profile-only array-ceiling input split. Do not touch
live training runs.

## Plain Problem

The array-ceiling rows are fast, but they still pay a real host-to-device copy.

```text
H100 policy_arrays H2D:   ~2.41s over 120 measured calls
H100 recurrent_toy H2D:   ~2.40s over 120 measured calls
```

This happens because the current LightZero toy takes a host `uint8` stack,
turns it into a Torch CUDA `float32` tensor, then runs the model. That is a
real cost in the current probe. It is probably not fundamental.

## What To Test

| Row | Purpose |
| --- | --- |
| host `uint8` current row | Baseline. Reproduce about `2.4s` H2D. |
| pinned host `uint8` + non-blocking copy | Test whether pageable memory/sync is the transfer tax. |
| host `float32` pre-normalized | Split dtype conversion from transfer bandwidth. |
| uint8 H2D then normalize on device | Separate copy from dtype/normalization allocation. |
| resident latest-frame control | Prove device-resident input can avoid stack H2D in a synthetic path. |
| resident Torch initial-only | Real Torch model with near-zero stack H2D. |
| resident Torch `recurrent_toy` | Same toy pressure with stack H2D removed. |

## Smallest Code Hook

Add one profile-only axis to the array-ceiling probe:

```text
hybrid_lightzero_array_ceiling_input_mode
```

First values:

```text
host_uint8           # current baseline
host_uint8_pinned    # pin CPU tensor, non-blocking copy
host_float32         # pre-normalize on host, copy float32
resident_torch_reuse # cache/reuse a CUDA tensor after first fill
```

Keep this on the array-ceiling probe first, not public collect-forward. The
array-ceiling probe isolated the H2D bucket; public MCTS collect has too much
other noise for the first transfer split.

Focused tests:

```text
host_float32 skips uint8 normalize
pinned mode reports pin/copy timing
resident_torch_reuse reports first-fill separately and near-zero measured H2D
grid builder emits the input mode flag
validation accepts only known modes
```

## How To Read Results

- If pinned memory helps, fix transfer mechanics first.
- If host `float32` gets worse, bytes are the issue.
- If uint8 device-normalize helps, conversion/allocation is the issue.
- If resident Torch input works, the real architecture should keep rendered
  observation stacks on the model device and only materialize host payloads at
  compatibility edges.

## Guardrail

Do not confuse this with renderer optimization. The renderer is not the current
main wall. This is about removing an avoidable boundary copy once the compact
MCTS path is being designed.

## First H100 Rows

Common shape:

```text
H100
B512 physical rows
2 roots per row = 1024 roots/call
steps=30, warmup=6
direct_gray64 persistent renderer
uint8 stack storage
array-ceiling recurrent_toy
sim8
scalar timestep materialization off
```

| input mode | run id | roots/sec | probe total | H2D | host prep | device/model bucket | read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `host_uint8` | `ap-8cD9SIXhWFkQyNYxTf4BbK` | `10086.23` | `1.184s` | `0.597s` | `0.000s` | `0.455s` | baseline pageable copy |
| `host_uint8_pinned` | `ap-ILMAfyK6FArmAMUa4BSrff` | `12295.15` | `0.625s` | `0.071s` | `0.051s` pin | `0.429s` | low-effort transfer win |
| `host_float32` | `ap-Ol4ORUHyRgLFwLiKeoGNSj` | `9641.80` | `1.341s` | `0.171s` | `0.676s` | `0.378s` | worse once host normalization is counted |
| `resident_torch_reuse` | `ap-PPENgpZQpWCTP4KLKilPSJ` | `14414.56` | `0.497s` | `0.000s` | first fill outside measured loop | `0.384s` | synthetic resident-input ceiling |

Important correction:

```text
An earlier host_float32 row reported host_prenormalize_sec but did not include
it in lightzero_array_ceiling_total_sec. The corrected row above includes that
time. Host float32 preprocessing is not a win.
```

2026-05-21 accounting cleanup:

```text
Pinned rows now report pin/input-prep time separately from H2D transfer time in
new code. Older rows may have bundled pinning into H2D. Do not sum pin time and
old H2D values from historical rows unless the row explicitly says the fields
are split.
```

Plain read:

```text
Pinned host uint8 is a real small transfer fix. A truly resident tensor is a
larger ceiling. Host float32 preprocessing just moves work to the CPU and makes
the measured path slower.
```

Next use:

```text
Carry pinned/resident-input ideas into the compact MCTS-boundary design. Do not
recommend this as trainer advice by itself; it is a profile-only transfer split.
```
