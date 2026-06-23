# Timing Budget And Amdahl Headroom, 2026-05-22

Scope: docs-only sidecar note for the compact-buffer/search-service discussion.
No source code, live Coach training run, Modal volume, checkpoint, eval, GIF, or
tournament artifact was touched.

Inputs:

- latest known H100 compact-loop rows supplied in the prompt;
- current working docs:
  - `whole_loop_denominator_ledger_20260521.md`;
  - `current_hot_path_bottleneck_map_20260522.md`;
  - `gpu_host_overhead_world_model_20260521.md`;
  - `subagent_host_overhead_sync_audit_20260522.md`;
  - `subagent_gpu_sync_model_20260522.md`;
  - `compact_search_replay_service_contract_20260522.md`.

## Plain Read

The current wall is not raw GPU drawing.

The current wall is the repeated boundary around:

```text
CPU env / public sidecars
-> compact visual/search-input ownership
-> production-to-compact
-> delta pack
-> H2D / renderer update waits
-> resident stack readiness
-> MCTX/search
-> selected-action readback for the next CPU step
```

At `sim16`, observation handoff dominates and search is still a secondary
bucket. At `sim32`, search grows into a co-equal wall with observation handoff.
The raw draw leaf is only `0.0094s` in both supplied refresh-on rows, so deleting
the draw kernel itself buys about `0.5-0.6%` total speed.

These are profile-only compact-loop numbers, not Coach training speed. Also,
several leaf timers are nested under `env_step_sec`; do not sum them as an
exclusive profile. Use each "goes to zero" row as an Amdahl smell test for that
one bucket against the same total wall.

## Current Budget: Refresh-On Sim16

Row:

```text
H100 compact refresh-on sim16
active roots/sec: 62,651.9
total: 1.569s
env: 71.2%
search: 19.8%
```

| Bucket | Time | Share of total | Wall if zero | Max speedup if zero | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| Top-level env step | `1.117s` | `71.2%` | `0.452s` | `3.47x` | Inclusive: handoff, observation, renderer, public/env work. Not pure mechanics. |
| Top-level search | `0.311s` | `19.8%` | `1.258s` | `1.25x` | Search is real, but not the sim16 primary wall. |
| Observation handoff leaf | `0.664s` | `42.3%` | `0.905s` | `1.73x` | Biggest named leaf. This is the compact-buffer/residency target. |
| Public packaging | `0.238s` | `15.2%` | `1.331s` | `1.18x` | Large enough that scalar/public objects still matter. |
| Game mechanics | `0.189s` | `12.0%` | `1.380s` | `1.14x` | Meaningful, but not enough alone. |
| Production-to-compact | `0.192s` | `12.2%` | `1.377s` | `1.14x` | Host ownership conversion is still visible. |
| Delta pack | `0.224s` | `14.3%` | `1.345s` | `1.17x` | Larger than raw draw by about `24x`. |
| H2D | `0.174s` | `11.1%` | `1.395s` | `1.12x` | Bytes may be small-ish; waits/array count can still hurt. |
| Raw GPU draw | `0.0094s` | `0.6%` | `1.560s` | `1.006x` | Not the wall. |
| Renderer plumbing subtotal | `0.599s` | `38.2%` | `0.970s` | `1.62x` | `production-to-compact + delta pack + H2D + draw`; likely overlaps observation handoff. |

Observed refresh-off ceiling:

| Row | Total | Active roots/sec | Speedup vs refresh-on | Read |
| --- | ---: | ---: | ---: | --- |
| refresh-on sim16 | `1.569s` | `62,651.9` | `1.00x` | Real compact refresh path. |
| refresh-off sim16 | `0.998s` | `98,546.5` | `1.57x` | Upper bound for deleting observation refresh; not a trainer-valid mode. |

Sim16 implication: deleting observation refresh gets a real `~1.5-1.7x` class
win, not `5x`. Search-only work has even less current sim16 headroom unless it
also removes surrounding CPU/list/search-service boundaries.

## Current Budget: Refresh-On Sim32

Row:

```text
H100 compact refresh-on sim32
active roots/sec: 49,096.9
total: 2.002s
env: 54.1%
search: 38.9%
```

| Bucket | Time | Share of total | Wall if zero | Max speedup if zero | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| Top-level env step | `1.083s` | `54.1%` | `0.919s` | `2.18x` | Still the largest top-level bucket, but less dominant than sim16. |
| Top-level search | `0.779s` | `38.9%` | `1.223s` | `1.64x` | Now large enough to justify search-service work. |
| Observation handoff leaf | `0.645s` | `32.2%` | `1.357s` | `1.48x` | Still large; slightly less share because search grew. |
| Public packaging | `0.233s` | `11.6%` | `1.769s` | `1.13x` | Scalar/public boundary remains visible. |
| Game mechanics | `0.182s` | `9.1%` | `1.820s` | `1.10x` | Smaller share at sim32. |
| Production-to-compact | `0.169s` | `8.4%` | `1.833s` | `1.09x` | Visible but not dominant alone. |
| Delta pack | `0.228s` | `11.4%` | `1.774s` | `1.13x` | Medium bucket. |
| H2D | `0.175s` | `8.7%` | `1.827s` | `1.10x` | Medium bucket. |
| Raw GPU draw | `0.0094s` | `0.5%` | `1.993s` | `1.005x` | Not the wall. |
| Renderer plumbing subtotal | `0.581s` | `29.0%` | `1.421s` | `1.41x` | Likely overlaps observation handoff. |

Observed refresh-off ceiling:

| Row | Total | Active roots/sec | Speedup vs refresh-on | Read |
| --- | ---: | ---: | ---: | --- |
| refresh-on sim32 | `2.002s` | `49,096.9` | `1.00x` | Real compact refresh path. |
| refresh-off sim32 | `1.313s` | `74,870.6` | `1.52x` | Upper bound for deleting observation refresh; search stays large. |

Sim32 implication: observation refresh removal and search-service work both
matter. Neither is a `5x` story by itself. A real architecture win has to reduce
both the handoff bucket and the search bucket while avoiding a new service tax.

## Why Raw GPU Draw Is Not The Current Wall

Raw draw is `9.4ms` in both refresh-on rows:

| Row | Total | Draw | Draw share | Max speedup from deleting draw |
| --- | ---: | ---: | ---: | ---: |
| sim16 refresh-on | `1.569s` | `0.0094s` | `0.6%` | `1.006x` |
| sim32 refresh-on | `2.002s` | `0.0094s` | `0.5%` | `1.005x` |

The renderer envelope is large because it includes host-state ownership and
synchronization work, not because the draw kernel is slow. The expensive pieces
are production-to-compact, delta packing, H2D/update waits, public packaging,
resident stack readiness, and the fact that the CPU must still inspect selected
actions before the next env step.

So "make the draw kernel faster" is currently a rounding-error optimization.
"Stop rebuilding/copying/search-input-owning the same compact visual facts every
step" is the live Amdahl target.

## What Has To Disappear For 5x

A `5x` result means:

| Row | Current total | 5x target total | Time that must disappear |
| --- | ---: | ---: | ---: |
| sim16 refresh-on | `1.569s` | `0.314s` | `1.255s` |
| sim32 refresh-on | `2.002s` | `0.400s` | `1.602s` |

Against the top-level `env + search` buckets, that means removing roughly:

| Row | Env + search time | Required removal for 5x | Share of env + search that must disappear |
| --- | ---: | ---: | ---: |
| sim16 | `1.428s` | `1.255s` | `88%` |
| sim32 | `1.862s` | `1.602s` | `86%` |

That is why a compact-buffer/search-service architecture has to remove whole
contracts, not just speed leaves:

1. Actor/env must own compact state directly enough that
   production-to-compact is a pointer/view/update, not a rebuild.
2. Renderer deltas and H2D payloads need batched, preallocated, low-wait
   ownership; full visual stack readback should stay out of the hot path.
3. Search must consume resident stacks and compact sidecars without per-root
   Python objects or per-simulation CPU listification.
4. Replay/target rows need compact array ownership so visit policy/root value
   payloads do not re-expand into scalar LightZero records before the learner
   needs them.
5. Scalar public/timestep objects should become debug or compatibility edges,
   not hot-loop owners.

Concrete ceiling check: the observed refresh-off rows are only `1.52-1.57x`
faster than refresh-on. If search were also magically deleted from those
refresh-off rows, the remaining walls would still be about:

| Row | Refresh-off total | Refresh-off search | Remaining if search also zero | Speedup vs refresh-on |
| --- | ---: | ---: | ---: | ---: |
| sim16 | `0.998s` | `0.297s` | `0.701s` | `2.24x` |
| sim32 | `1.313s` | `0.769s` | `0.544s` | `3.68x` |

Even the fantasy "no observation refresh and no search" row is not a clean `5x`
at sim16, and only approaches it at sim32 if the remaining public/game/replay
work also shrinks. The 5x path needs broad ownership change across env handoff,
search, and replay/target materialization.

## Host/GPU Sync: When It Is Okay

Acceptable sync points:

- Selected-action readback before CPU env step. The CPU cannot step without the
  actions, and the payload is small.
- Device-side dependency from renderer to search. Search must see the latest
  frame, but the host should not have to read it.
- Replay commit boundary. Visit policy, root value, actions, rewards, dones,
  masks, ids, and final observations must be consistent before rows are
  sample-visible; chunking is fine.
- Terminal final-observation capture. It is okay to take a slower exact path
  for terminal/autoreset rows.
- Warmup/end-of-run profiling barriers, if clearly labeled.
- Sampled parity checks, first-N checks, every-K checks, and terminal checks.
  Do not pay them every hot step.

Suspect sync points:

- `block_until_ready()` immediately after render/update when the next consumer
  is search.
- `block_until_ready()` on the resident stack before search if it only moves
  wait time between buckets.
- Many small `device_put`s for delta/compose sidecars.
- Host inspection of JAX arrays or Torch tensors inside the per-step loop.
- Full frame or full stack D2H every step.
- Per-root Python dict/timestep/list materialization before policy/search or
  replay truly needs it.

## Large, Medium, Small

For these rows, use this practical budget:

- Large: `>0.30s` or `>20%` of wall. Current examples are observation handoff
  at both sims, top-level env, and sim32 search.
- Medium: `0.10-0.30s` or `5-20%`. Current examples are public packaging, game
  mechanics, production-to-compact, delta pack, H2D, and sim16 search.
- Small: `<0.02s` or `<1%`. Current example is raw GPU draw. Replay-index and
  root-value extraction were also small in the latest replay-valid docs after
  the direct root-node extraction fix.

Byte size rule of thumb:

- Small by bytes: selected actions, legal masks, visit policy, root values.
  These can still be bad if each read forces a queue-wide sync.
- Large by bytes: full `[B,P,4,64,64]` stacks, float32 stacks, full latest-frame
  mirrors, visual trail arrays. These should stay resident or be sampled.
- Large by object count: per-env timesteps, dicts, listified CTree outputs, and
  scalar public records. These are exactly what compact-buffer/search-service
  work should keep off the hot path.

## Mock Search-Service Row

Latest supplied service row:

```text
mock search service H100 B512/A16: 17,711.9 steps/sec
service-tax row: 12,461.6 steps/sec
direct CTree row: pending in main thread
```

Read:

- The service-tax row is `0.70x` the mock throughput, or about `1.42x` more
  time per step.
- Per-step time moves from about `56.5us` to `80.2us`, a tax of about `23.8us`.
- This is not directly comparable to the active-roots/sec compact MCTX rows.
  Treat it as evidence that a service boundary can easily erase wins if it adds
  serialization, copies, or waits.
- Do not use this row to project real CTree/MCTX speed until the pending direct
  CTree row lands on the same denominator.

## Missing Or Suspect Measurements

1. Direct CTree row for the mock search-service denominator is still pending.
   Without it, the service-tax row cannot rank real search options.
2. The supplied leaf buckets are not exclusive. `public`, `game`,
   `production-to-compact`, `delta`, `H2D`, and `draw` live under broader
   env/observation spans. Every future row should report bucket sum and
   residual.
3. Active roots/sec and steps/sec are different currencies. Keep them separate
   unless the row states the exact root/step mapping.
4. H2D needs bytes, array count, pinned/non-pinned mode, queue time, and wait
   time. A small payload can still be a large sync.
5. Resident-stack waits can move between renderer, H2D, and search buckets.
   Judge total wall, not only relocated timer labels.
6. Refresh-off is a ceiling, not a valid trainer path. It may feed stale or
   placeholder observations and should not be used as a learning claim.
7. No-death rows can overprice long-trail mechanics and underprice
   terminal/autoreset handling. Normal-death companion rows are still needed.
8. Replay/target/RND are not fully priced in this compact-loop budget. A
   trainer-facing claim needs compact replay/target/RND parity and timing.
9. Profiling sync overhead should be paired with an unsynchronized throughput
   companion before any runtime recommendation.
10. Public packaging and game mechanics need a stricter split: public
    sidecar prep/info/batch pack versus actual `VectorMultiplayerEnv` runtime.

## Recommendation

Use sim16 as the handoff-residency budget and sim32 as the combined
handoff-plus-search budget.

The next useful compact-buffer/search-service milestone is not another raw draw
kernel win. It is a same-denominator row that removes scalar/object ownership
from the observation/search/replay path while preserving compact root, search
result, target-row, final-observation, legal-mask, player-perspective, and RND
latest-frame contracts.

Promotion bar for the architecture lane:

```text
same semantics, same denominator, no hidden fallback,
observation handoff materially lower,
search materially lower at sim32,
service tax explicitly priced,
replay/target rows still valid
```

Anything below that is useful profile evidence, not yet a 5x training plan.
