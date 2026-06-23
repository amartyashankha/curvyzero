# External Patterns Memo, Late 2026-05-22

Scope: sidecar external-pattern research for the CurvyTron/MuZero optimizer
lane. This is research-only. No code, trainer defaults, live Coach runs,
checkpoints, evals, GIFs, tournaments, or Modal state were changed.

## Short Read

The external systems agree with the current local bottleneck read:

```text
The main wall is not pure rendering and not "CTree should be C++."
It is the compact env/observation/search/replay ownership boundary.
```

The strongest pattern is:

```text
native/vector env buffers
-> stable compact row/player roots
-> device-resident or array-native batched search
-> compact action/visit/value output
-> compact replay/target rows
-> stock LightZero objects only as validation/adapters
```

Local late-day evidence sharpened the picture. MCTX/JAX search on real compact
visual roots showed 10x-class search-boundary headroom, but the repeated closed
compact loop fell back to a few thousand roots/sec until native actor buffers,
live-prefix trimming, and render-state filtering improved it. The latest split
shows actual env physics is tiny; the hot bucket is still state handoff for
rendering plus observation/stack update. Search is now small in the matched
closed-loop denominator.

## Sources Checked

Local docs reviewed:

- `README.md` in this folder.
- `current_hot_path_bottleneck_map_20260522.md`
- `reorientation_20260522_fast_falsifiers.md`
- `compact_search_replay_contract_plan_20260522.md`
- `mock_search_service_ceiling_plan_20260522.md`
- `subagent_mctx_gpu_search_research_20260522.md`
- `subagent_external_search_systems_20260522.md`
- `subagent_fast_rl_architecture_patterns_20260522.md`
- `subagent_radical_external_architecture_critique_20260522.md`
- `puffer_style_contiguous_buffer_attach_audit_20260522.md`
- `subagent_device_resident_observation_boundary_20260522.md`
- late `experiment_log.md` entries through the closed compact MCTX loop and
  env-step timing split.

External sources opened/verified:

- PufferLib docs: https://puffer.ai/docs.html
- PufferLib repo: https://github.com/pufferai/pufferlib
- MCTX repo: https://github.com/google-deepmind/mctx
- MiniZero repo: https://github.com/rlglab/minizero
- KataGo analysis engine: https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md
- OpenSpiel AlphaZero docs: https://openspiel.readthedocs.io/en/latest/alpha_zero.html
- LightZero CTree docs: https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html

## Patterns That Matter Now

### 1. PufferLib: Own Flat Buffers, Not Wrappers

PufferLib matters most as an env/collector boundary reference, not as an
algorithm drop-in. Its current docs emphasize static contiguous allocations,
CUDA graph-friendly memory, chunked vector env buffers, pinned async transfers,
and C-native envs writing observations/actions/rewards/terminals into large
contiguous blocks.

CurvyTron translation:

```text
obs_uint8[B,P,4,64,64]
legal_mask[B,P,3]
reward[B,P]
done[B]
row_id/player_id
joint_action[B,P]
terminal/final_observation/autoreset sidecars
```

The useful lesson is to stop rebuilding scalar `BaseEnvTimestep` objects,
per-env action dicts, and per-step info rows in the optimized lane. The latest
native actor-buffer rows validate this locally: removing a host-copy layer gave
real closed-loop speedups, especially at B1024, but did not solve the whole
boundary.

### 2. MCTX: The Clean Device-Resident Search Shape

MCTX is the best reference for what the search boundary should look like:
batched root arrays, recurrent function inside a compiled loop, dense tree
state, invalid-action masks, and compact action/action-weight output.

The local MCTX rows now matter more than the generic claim. On real compact
visual CurvyTron roots, MCTX/JAX fresh-boundary search reached about 124k
roots/sec at B512/sim16 and about 51k roots/sec at B512/sim32, far above the
current LightZero direct CTree boundary. But in the repeated closed loop,
throughput dropped because env/observation/replay edges re-entered the
denominator.

CurvyTron translation: use MCTX or an equivalent device-resident search service
as the search-core reference, but do not bridge the current PyTorch LightZero
model into JAX and call that "resident." That would recreate the host boundary.

### 3. MiniZero And KataGo: Batch Many Roots/Positions

MiniZero is the closest MuZero-style system reference: multiple MCTS instances
per self-play worker, leaf collection across them, batched GPU inference, then
records to storage/replay. KataGo's analysis engine makes the same systems
point from serving: many positions in flight can be faster than one position
because the neural net sees large batches.

CurvyTron translation:

```text
B physical rows * P player views = B*P roots
fixed A=3
fixed sim count
batched recurrent/search
compact action/visit/value arrays back
```

This supports a search/evaluator service only if the surrounding env and replay
path stay compact. The late closed-loop rows show exactly why: fast search
alone is swallowed when the next step must rebuild host render/stack state.

### 4. AlphaZero/MuZero/OpenSpiel: Separate Actors, Search, Replay, Learner

The scalable Zero-family pattern is not a single optimized collect call. It is
actor/search/learner separation with batched inference and replay ownership.
OpenSpiel's docs are especially useful because they contrast the slow Python
path with the C++ path that uses threads, shared cache, batched inference, and
GPU support.

CurvyTron translation: the first production-shaped service should be
fixed-opponent and bounded, not a full league. But every compact chunk should
carry checkpoint/search metadata now, so later actor staleness and replay age
controls are not retrofitted into anonymous arrays.

### 5. LightZero CTree: Already C++ Inside, Wrong Boundary Outside

LightZero's docs say MuZero CTree core `batch_traverse` and
`batch_backpropagate` are C++. That means "rewrite CTree in C++" is not the
diagnosis. The local wall is the Python/list/CPU shell:

```text
root logits/value -> CPU/list
batch_traverse -> Python lists
recurrent on GPU
reward/value/policy -> CPU/list
batch_backpropagate
dict output and replay objects
```

Array-native fixed-A3 CTree remains useful as a conservative bridge, but recent
flat-A3 and full-loop rows say it is not the next big move by itself.

## Patterns Not Applicable As Stated

- PufferLib trainer replacement: PufferLib's training path is PPO/V-trace-ish,
  not MuZero search with replay targets. Steal the buffer contract, not the
  learner.
- Gym/PettingZoo emulation: useful for compatibility smoke, but it preserves
  scalar wrappers and misses the speed path.
- "Just use MCTX": not a patch unless model/search/replay become JAX-shaped.
  A PyTorch-to-JAX bridge is likely the same bad boundary in a new costume.
- "Just more C++" or "just more CPU": LightZero CTree is already C++ in the
  middle, and CPU64 already failed as a capacity fix.
- KataGo-style NN/transposition cache as the main bet: cross-position batching
  matters; Go-like repeated-state caching probably matters less for CurvyTron.
- Full self-play league/reanalysis first: useful later, but too broad before
  the fixed-opponent compact boundary is stable.
- Pure renderer rewrite as the main lane: renderer/observation now matters
  again inside the closed compact loop, but the problem is ownership and
  readback/stack handoff, not isolated drawing speed.

## Current Local Read

Late evidence to preserve:

- `direct_ctree_gpu_latent + CompactReplayIndexRowsV1` proof at B512/sim16:
  about 6222 roots/sec; compact replay proof cost about 0.103s over 61440 rows;
  public LightZero output bytes 0.
- Precomputed recurrent output improved direct only modestly; recurrent is not
  the whole wall.
- Mock search service stayed around 1.8x-2x over direct in same-denominator
  compact rows; useful headroom, not a 10x full-loop proof.
- MCTX/JAX on real compact visual roots proved large search-boundary headroom.
- Repeated closed compact MCTX loop moved the wall to env/observation/replay
  edges. Native actor buffers and live-prefix trimming helped, but latest
  matched timing still shows `env_step_sec` dominating, with actual env runtime
  tiny and observation/render/stack handoff hot.

Plain conclusion:

```text
Search-core headroom exists.
Compact replay/index rows are cheap enough.
The next wall is the closed compact observation/state handoff around the
search service: GPU render -> host stack -> device_put -> search -> host action
edge, plus render-state reconstruction and replay/RND sidecars.
```

## Concrete Experiment To Run Next

Run the device-resident observation boundary experiment from
`subagent_device_resident_observation_boundary_20260522.md`.

Experiment:

```text
HybridBatchedObservationProfileManager
-> persistent JAX policy framebuffer render
-> use renderer.last_output_device [B,2,1,64,64]
-> maintain resident JAX FIFO stack [B,2,4,64,64]
-> feed MCTX from resident device stack
-> keep host HybridCompactBatch / CompactRootBatchV1 for masks, row/player ids,
   final-observation validation, and CompactReplayIndexRowsV1
```

Compare against the latest matched closed compact MCTX row:

```text
B1024/P2/sim16/h64-or-h64v16/loop16
native_actor_buffer=true
persistent renderer
same action masks, replay-index proof, and legality checks
```

Required telemetry:

- closed-loop roots/sec and one-step roots/sec;
- `env_step_sec` fraction and leaves;
- renderer render time;
- render-state write / production-to-compact time;
- host observation/stack update time;
- `obs_h2d_bytes`, which should go to zero for MCTX input;
- mask H2D bytes separately;
- sampled parity between resident device stack and host stack;
- reset/autoreset/final-observation counters;
- selected-action legality and visit-policy normalization;
- CompactReplayIndexRowsV1 count and timing.

Keep condition:

```text
Closed-loop roots/sec improves by at least 1.2x, preferably 1.4x+, without
stack parity drift, illegal actions, reset/final-observation drift, or replay
identity drift.
```

Kill or redirect condition:

```text
If closed-loop throughput barely moves, the next wall is not obs H2D/readback.
Split actor stepping, render-state reconstruction, stack update, replay-index,
and RND sidecars further before building a larger MCTX/search service.
```

Why this is the right next experiment:

```text
PufferLib says the env/search boundary should be static contiguous buffers.
MCTX says search can be fast if the input is resident and batch-first.
MiniZero/KataGo say many row/player roots should feed one batched search owner.
The local closed-loop rows say search is no longer the dominant bucket.
So the next falsifier should remove the GPU-render -> host-stack -> device-put
double bounce while keeping host compact validation intact.
```

