# Full Iteration Dataflow Reorientation, 2026-05-22

Status: docs-only follow-up. No source, trainer defaults, Modal jobs, live
training runs, checkpoints, evals, GIFs, or tournament artifacts were touched.

## Plain Read

The current bottleneck is no longer "draw the frame faster." In the compact
closed-loop rows, the large bucket is now:

```text
selected actions
-> advance CurvyTron state
-> prepare the next observation/stack/root batch
-> feed the next search call
```

Actual CurvyTron mechanics are small inside that bucket. The expensive part is
state ownership and handoff: render-state writes, renderer state packing, stack
ownership, root/input materialization, masks, and synchronization.

In the trusted stock `train_muzero` denominator, the wall is still the
LightZero collect/search/object contract. In the profile-only compact
denominator, search can already be made small enough that the next-input
handoff becomes the wall.

## Two Lanes

### A. Current Trusted Stock Training Path

This is the Coach-facing shape unless promotion gates say otherwise:

```text
scalar LightZero envs
-> observation dict per env
-> host float/uint8 stack [4,64,64]
-> MuZeroPolicy._forward_collect
-> GPU initial_inference
-> CPU root prep / CTree roots
-> Python MCTS loop
-> GPU recurrent_inference per simulation
-> CPU reward/value/policy readback for CTree
-> per-env action dict
-> env_manager.step(...)
-> BaseEnvTimestep / GameSegment rows
-> stock replay sample / target builder
-> learner batch
-> optional RND CPU/object lane
```

Large objects:

- observation stack: `[B,4,64,64]` or scalar `[4,64,64]`;
- latent roots and recurrent hidden states on GPU;
- replay/game segments over many timesteps.

Small objects:

- action space `A=3`;
- legal masks `[B,3]`;
- rewards/dones/values `[B]`;
- selected actions `[B]`.

The trusted path is semantically valuable because stock LightZero owns replay,
target construction, learner scheduling, RND hooks, checkpoints, and eval
compatibility. It is slow because the same batch repeatedly becomes Python
dicts, lists, root objects, NumPy arrays, and CPU-visible CTree payloads.

### B. Profile-Only Compact Path

This is the current optimizer evidence path, not Coach launch advice:

```text
VectorMultiplayerEnv[B,2]
-> HybridCompactBatch
-> persistent GPU policy framebuffer / host or resident stack
-> CompactRootBatchV1
-> compact search service control
-> CompactSearchResultV1
-> selected joint actions
-> next HybridCompactBatch
-> CompactReplayIndexRowsV1
-> materialized target rows only at validation/sampler edge
```

Large objects:

- compact visual stack `[B,2,4,64,64]`;
- render state / trail state;
- optional resident GPU stack;
- compact replay ring or validation chunk.

Small objects:

- action mask `[B,2,3]`;
- selected action `[active_roots]`;
- visit policy `[active_roots,3]`;
- root value `[active_roots]`;
- replay index rows and sidecars.

The profile compact path proves the batch can survive through search and replay
with much less public LightZero object fanout. The latest rows say replay-index
rows are cheap; the repeated-loop wall is now preparing the next search input.

## One Collect Tick: What Moves Where

| Step | Stock/trusted lane | Compact/profile lane | Large/small read |
| --- | --- | --- | --- |
| 1. Actor/env state | Scalar env instances hidden behind LightZero env manager. | `VectorMultiplayerEnv[B,2]` plus compact sidecars. | State/trails are large. Actions/rewards/masks are small. |
| 2. Observation | Env returns dict: `observation`, `action_mask`, `to_play`, info. | Manager builds or borrows compact render state, renders latest frames, updates host or resident stack. | Full stack is large; latest frame is medium; masks are tiny. |
| 3. Stack/root batch | Stock stacks are host arrays copied into policy input. | `CompactRootBatchV1` carries observation, legal mask, active root mask, row/player ids, reward/done/final sidecars. | Root observation copy is large; root sidecars are small. No-copy root batches made this visible. |
| 4. Model root pass | GPU `initial_inference`; stock immediately CPU-reads values/logits for roots. | GPU or JAX model/search input from compact stack; mock/MCTX/direct controls vary. | Values/logits are small, but stock converts them into Python lists/root objects. |
| 5. MCTS/search | LightZero CTree owns CPU tree; each sim calls GPU recurrent then CPU-reads reward/value/policy. | MCTX/JAX or direct compact service returns compact selected actions and visit policies. | Per-sim payload is small by bytes but expensive by sync/list/control frequency. |
| 6. Selected actions | Stock builds per-env action dict. | Compact selected root actions become `[B,P]` joint actions for next env step. | Tiny data. Critical sync if env step is CPU. |
| 7. Replay rows | Stock GameSegment/replay objects. | `CompactReplayIndexRowsV1` in hot path; full `CompactReplayChunkV1` only for validation/materialization. | Index rows are small and measured cheap; full observation target rows are large and should stay out of collection hot path. |
| 8. Learner-facing data | Stock replay sample/target builder materializes learner batches. | Compact index rows plus replay chunk can materialize learner-shaped rows later. | Learner tensors are large; mapping sidecars should stay compact. |

## Measured Timing Buckets

### Stock `train_muzero` Profile

Matched H100 C64/sim16/3-learner/no-RND/no-death:

| Row | Throughput | Wall | Collect | Policy collect | MCTS | Learner | Notable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stock | `433.17 steps/sec` | `37.82s` | `26.02s` | `17.10s` | `12.09s` | `4.17s` | stock object/search path |
| direct output-fast | `566.19 steps/sec` | `28.94s` | `19.41s` | `10.31s` | `8.06s` | `1.03s` | direct D2H `2.47s`, output assembly `0.077s`, fallback `0` |

Matched RND hash-fixed profile:

| Row | Throughput | Wall | Policy collect | MCTS | RND train/hash/estimate |
| --- | ---: | ---: | ---: | ---: | --- |
| stock | `351.02 steps/sec` | `46.68s` | `23.62s` | `17.30s` | `0.590s / 0.131s / 0.086s` |
| direct | `448.52 steps/sec` | `36.53s` | `13.32s` | `10.66s` | `0.603s / 0.140s / 0.093s` |

Read:

- Direct/output-fast is a real `~1.28x-1.31x` profile-loop win.
- Output assembly is now tiny (`~0.077s`), so do not polish it next.
- The remaining stock wall is collect/search topology: root prep, CTree
  CPU/list API, per-simulation recurrent output D2H/listifying, and the stock
  collector/replay shell.

### Compact Closed-Loop Profile

Recent H100 B1024/P2/body4096/h64/depth16/native/no-copy/replay rows:

| Row | Roots/sec | Env frac | Search frac | Mechanics inside env | Observation/search-input handoff | GPU draw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| host stack sim16 | `23,109` | `68.1%` | `7.5%` | `9.6%` | `80.0%` | `0.0066s` |
| resident stack sim16 | `30,297` | `68.3%` | `9.6%` | `8.4%` | `79.2%` | `0.0054s` |
| host stack sim32 | `19,485` | `63.0%` | `15.2%` | `8.8%` | `79.8%` | `0.0071s` |
| resident stack sim32 | `26,805` | `59.8%` | `19.7%` | `10.8%` | `75.8%` | `0.0057s` |
| refresh-off sim16 ceiling | `57,895` | `26.1%` | `18.5%` | `43.1%` | `0.0%` | skipped |

After fast visual/no-copy:

| Row | Roots/sec | Production->compact | Observation | Root-build | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| old host+copy | `20,686` | `0.374s` | `0.734s` | `0.062s` | old best current row |
| fast visual, copy | `19,800` | `0.054s` | `0.427s` | `0.251s` | root copy exposed |
| fast visual, no-copy | `26,610` | `0.057s` | `0.412s` | `0.009s` | `1.29x` refresh-on win |
| no refresh, no-copy | `63,560` | skipped | near zero | `0.007s` | ceiling row |

After borrowed single-actor render state:

| Row | Roots/sec | Env frac | Search frac | Actor render-state write | Observation | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| host sim16 copied | `26,603` | `64.1%` | `8.5%` | `0.299s` | `0.599s` | copied parent buffers |
| host sim16 borrowed | `32,830` | `57.6%` | `10.7%` | `0.000s` | `0.564s` | `1.23x` |
| resident sim16 copied | `32,873` | `62.5%` | `10.5%` | `0.305s` | `0.347s` | copied resident baseline |
| resident sim16 borrowed | `48,579` | `53.8%` | `15.2%` | `0.000s` | `0.303s` | `1.48x` |
| resident sim32 copied | `24,020` | `51.5%` | `20.9%` | `0.238s` | `0.458s` | copied resident sim32 |
| resident sim32 borrowed | `36,041` | `43.3%` | `28.3%` | `0.000s` | `0.331s` | `1.50x` |

Read:

- Raw GPU drawing is already tiny.
- Actual mechanics are not the wall.
- Borrowing render state proved the state-ownership hypothesis.
- The current next wall is the remaining observation/renderer/stack/root-input
  handoff plus search as sim count rises.

### Rough Per-Decision Scale

These are approximate conversions from the measured profile denominators, useful
only for intuition:

| Denominator | Measured throughput | Rough one decision wave |
| --- | ---: | ---: |
| stock H100 C64/sim16/no-RND | `433.17 scalar env steps/sec` | `~148 ms` per 64-env wave |
| direct output-fast H100 C64/sim16/no-RND | `566.19 scalar env steps/sec` | `~113 ms` per 64-env wave |
| stock H100 C64/sim16/RND | `351.02 scalar env steps/sec` | `~182 ms` per 64-env wave |
| direct output-fast H100 C64/sim16/RND | `448.52 scalar env steps/sec` | `~143 ms` per 64-env wave |
| compact resident borrowed B1024/P2/sim16 | `48,579 active roots/sec` | `~42 ms` per 2048-root tick |
| compact resident borrowed B1024/P2/sim32 | `36,041 active roots/sec` | `~57 ms` per 2048-root tick |

Do not compare the stock and compact rows as the same unit. The stock rows call
`train_muzero` and include replay/learner/RND denominators as configured. The
compact rows are profile-only repeated search-input loops that do not prove
Coach training speed.

## CPU/GPU Residency Table

| Data | Stock/trusted residency | Compact/profile residency | Desired hot residency |
| --- | --- | --- | --- |
| CurvyTron mechanics state | CPU/NumPy/env objects | CPU/NumPy vector env slabs | CPU or native slab until a GPU env exists |
| Render state/trails | CPU env state, copied to renderer inputs | CPU compact state; borrowed-state canary avoids parent copy | Renderer-native compact owner, no rebuild per tick |
| Latest visual frame | Host array after render in stock; optional GPU renderer D2H | Host stack or resident JAX device frame | Device latest frame plus sampled host validation |
| Observation stack `[B,P,4,64,64]` | Host policy obs, moved to Torch CUDA for model | Host or resident GPU stack | Resident stack for search/RND, host only for validation |
| Legal/action masks | Host NumPy/list | Host compact mask, H2D for search | Persistent device mask plus host sidecar |
| Root values/logits | GPU model output then CPU NumPy/list for CTree | Depends on service; direct CTree still CPU/list | Device arrays, or one compact CPU array boundary only |
| MCTS tree state | CPU LightZero CTree objects | Direct CTree CPU, MCTX device in profile lanes | Service-owned array tree, preferably device-resident |
| Recurrent hidden states | GPU tensors, but stock copies latents in some paths | GPU-latent direct keeps hidden pool on CUDA | Device resident, no per-sim CPU payload except final action if needed |
| Selected actions | CPU dict per env | Compact CPU/JAX action array then joint-action array | Small CPU sync is acceptable if env step is CPU |
| Replay rows | Python GameSegment/target objects | Compact index rows; full rows at validation edge | Compact ring/index owner, materialize only for learner/sample |
| RND latest frame | CPU extraction/hashing/list storage | Compact extraction possible, still not fully resident | Device latest-frame ring and sparse metrics sync |
| Learner batch | Stock replay materializes then model device copy | Adapter can materialize from compact rows | Array-native learner input from compact replay |

## Copying And Object Materialization

Stock/trusted copies and objects:

- env observation dicts and `BaseEnvTimestep`/`MockBaseEnvTimestep` rows;
- host stack copies, stack roll/update, action mask copies;
- `pred_values.detach().cpu().numpy()` and policy logits `.cpu().numpy().tolist()`
  during root prep;
- CTree `Roots.prepare(...)`, `batch_traverse(...)`, and
  `batch_backpropagate(...)` through Python/list-shaped APIs;
- per-simulation recurrent output readback: reward/value/policy to CPU before
  backprop;
- `roots.get_distributions()`, `roots.get_values()`, selected action dicts;
- GameSegment/replay sample/target rows;
- RND latest-frame CPU extraction, CPU hashing, tensor clones in Python lists,
  and metric scalar readbacks.

Compact/profile copies and objects:

- actor compact sidecar writes are small; render-state copies were large enough
  that borrowed state moved total wall by `1.48x-1.50x` in resident rows;
- production-to-compact conversion dropped from roughly `0.374s-0.517s` to
  `~0.054s-0.057s` with the fast visual adapter;
- root observation copy became visible, then no-copy root batches cut root-build
  to `~0.009s`;
- replay-index rows were cheap: `0.103s` over `61,440` rows, about
  `1.68 us/row`;
- full materialized target rows remain large and should stay out of the
  collection hot path.

## Synchronization Rules

Useful syncs:

- before CPU CTree root prep in the stock/direct CTree path, because CTree needs
  CPU values/logits;
- once per recurrent simulation in the current direct CTree path, because CTree
  backprop consumes CPU reward/value/policy;
- before CPU env step when selected actions come from GPU/JAX search;
- before CPU replay-index validation if it consumes search result arrays;
- at learner batch handoff if sampled host data must move to the model device;
- in explicit profiling rows, where synchronized bucket attribution is the
  point.

Syncs to avoid or defer in the optimized lane:

- full-frame D2H just so the next search can H2D it again;
- blocking immediately after resident stack FIFO update if search will be the
  first real consumer;
- materializing `CompactRootBatchV1.observation` in the hot loop when search
  uses the resident stack;
- per-step CPU hashing/metrics for RND;
- exact public LightZero output construction except at validation/adaptor edges.

The rule of thumb: sync where ownership crosses from device search to CPU env,
or from compact data to a CPU validation/adaptor. Do not sync merely to observe
a tensor that the next stage could consume in place.

## Amdahl Intervention Points

Do now:

1. Attack compact state/search-input ownership. Latest compact rows put
   `env_step_sec` at roughly `43-54%` even after borrowed render state, and most
   of that is observation/search-input handoff rather than mechanics.
2. Keep resident stack and no-copy root batches as the default profile
   denominator. They have real wins after root-copy removal.
3. Split and reduce observation/update/delta-pack/H2D/root residuals. GPU draw
   itself is already tiny, so focus on data ownership and synchronization.

Do next after that:

4. Revisit search service pressure at sim32+. Search is `28.3%` in the borrowed
   resident sim32 row, so it is becoming hot again as the handoff shrinks.
5. Keep compact replay/RND/learner ownership honest. Replay-index rows are cheap,
   but full target materialization and RND CPU/list behavior will reappear once
   search/input handoff improves.

Do not prioritize now:

- output assembly in the stock direct path: measured `~0.077s`;
- raw GPU drawing: measured `~5-7ms` in the compact rows;
- more CPUs for CTree: CPU64 made the current search-boundary row slower;
- flat-A3 as a launch setting: it passed no-model parity/speed gates but matched
  full-loop direct `516.55 steps/sec` vs flat-A3 `509.69 steps/sec`;
- replay-index micro-optimization: current proof is about `1.68 us/row`.

## Ten Design Alternatives And Critiques

1. **Keep stock LightZero and polish output assembly.** Already done enough.
   Good as a low-risk profile win, but the bucket is now too small to move wall.

2. **Direct CTree GPU-latent hook.** Valuable practical baseline. It preserves
   stock output/replay semantics and gave `~1.3x`, but it still pays CPU/list
   roots, per-sim D2H, Python control, and stock collector/replay objects.

3. **Flat-A3 CTree payload API.** Valid proof that list ABI costs real time in
   no-model rows. Critique: matched full-loop did not move, so it is a bounded
   CTree probe, not the next Coach speed setting.

4. **Dense Torch MCTS.** Good fixed-shape search experiment. Critique: eager and
   compile lanes have not beaten direct CTree at sim16/sim32 reliably, and
   semantic parity/replay integration are still separate problems.

5. **JAX/MCTX search service.** Best reference for accelerator-native search
   arrays. Critique: search-only roots/sec is not full training speed; the
   closed loop must include env step, observation stack, action readback, replay
   index, and learner-facing data.

6. **Mock search service ceiling.** Useful falsifier. It showed compact replay
   and public-output-free service shape can be much faster. Critique: fake
   search has no MuZero semantics, so use it only to price non-search overhead.

7. **Persistent compact render-state owner.** Highest current Amdahl target.
   Borrowed state already removed parent render-state copy and improved resident
   rows. Critique: terminal/autoreset final-observation correctness needs a
   snapshot protocol before this becomes more than a no-terminal canary.

8. **Device-resident stack/root owner.** Strong next step: latest frame, FIFO
   stack, root observations, masks, and RND latest-frame slices should be
   resident with sampled host validation. Critique: careless sync removal can
   simply move waits into search; judge total wall only.

9. **Puffer-style native/vector slab.** Long-term env/replay ownership shape:
   static buffers, worker-owned slices, compact sidecars, scalar objects only at
   edges. Critique: bigger refactor; first falsifier must prove it beats current
   native actor buffer by more than noise.

10. **Compact replay/RND/learner owner.** Required before a real 5-10x training
    claim. Critique: replay-index rows are already cheap, so the value is not in
    the index builder alone; it is in avoiding later stock target/RND/list
    materialization when the learner consumes data.

## Next Three Profile Experiments

1. **Resident stack sync and root-input handoff split.**
   Same H100 B1024/P2/sim16 and sim32 borrowed-state rows. Compare explicit
   resident stack block vs deferred sync, persistent device masks vs per-step
   mask H2D, and hot-loop no-copy root observation with sampled validation.
   Pass only on total wall/roots/sec, not bucket movement.

2. **Terminal-safe compact render-state ownership.**
   Extend the borrowed-state canary with a pre-reset/final-observation snapshot
   protocol, then run no-terminal and terminal/autoreset hostile profile rows.
   Gate on exact copied-vs-borrowed observation/stack/replay-index parity and
   on keeping `actor_render_state_write_sec` near zero.

3. **Compact learner-facing denominator.**
   Run a profile-only loop where search-selected actions drive env step,
   `CompactReplayIndexRowsV1` writes stay hot, and learner-facing rows are
   materialized from compact indices at a sampled/controlled cadence. Compare
   no-op learner, prebuilt tensor learner, and stock materialization adapter so
   the next wall after search-input handoff is visible before building around
   the wrong edge.
