# Amdahl Critique: Next Big Moves, 2026-05-22

Status: critique only. No code changes, no Coach launch advice.

## Plain Verdict

The current compact closed-loop lane is not a failure, but it is very close to
a rabbit hole if the next work goes back to search micro-optimizations.

The latest repeated-loop rows say the hot wall has moved:

| shape | row | roots/sec | env step sec | env step wall |
| --- | --- | ---: | ---: | ---: |
| B512/sim16 | no native actor buffer | `5792` | `1.212s` | `85.7%` |
| B512/sim16 | native actor buffer | `6823` | `0.893s` | `74.4%` |
| B1024/sim16 | no native actor buffer | `6253` | `2.406s` | `91.8%` |
| B1024/sim16 | native actor buffer | `8918` | `1.493s` | `81.3%` |

Native actor-buffer work helped: about `1.18x` at B512 and `1.43x` at B1024.
But after the win, env/actor/observation stepping is still the dominant wall.
Search is only a small bucket in these rows, roughly the wrong place to expect
the next big multiplier.

## Are We In A Rabbit Hole?

Yes, if "optimizer" means polishing flat-A3, direct output assembly, CPU count,
or standalone search throughput on this denominator.

No, if the lane is now reoriented to the repeated closed compact edge:

```text
compact env state
-> action application / collision / autoreset
-> observation + stack update
-> compact root batch
-> search
-> compact replay/RND/target inputs
```

The sober read is that MCTX/search proved enough headroom to stop being the
first wall. The repeated loop then exposed the cost of feeding the next search
call. That is healthy progress, but only if we follow the wall.

## Amdahl Map

The current native rows have this ceiling if only the remaining `env_step_sec`
bucket is attacked:

```text
B512 native, env step 74.4%:
  remove env step entirely: theoretical max 3.9x
  make env step 2x faster:  about 1.6x total, ~10.9k roots/sec
  make env step 4x faster:  about 2.3x total, ~15.4k roots/sec

B1024 native, env step 81.3%:
  remove env step entirely: theoretical max 5.3x
  make env step 2x faster:  about 1.7x total, ~15.0k roots/sec
  make env step 4x faster:  about 2.6x total, ~22.8k roots/sec
```

By contrast, if search is only `3-5%` of closed-loop wall, even deleting it
would only buy about `1.03x-1.05x`. That is the whole Amdahl story in one
sentence: search is strategically important, but in the current closed-loop
measurement it is no longer the next throughput wall.

## Bigger Change That Would Matter

The next big move is not "faster MCTS" by itself. It is a compact resident
step/update owner.

Concretely, the useful architecture should stop rebuilding and moving the same
root-shaped stack through the parent loop every iteration. The high-value shape
is:

```text
preallocated compact row/player buffers
-> native/vector actor step writes state and sidecars in place
-> observation/latest-frame/stack update stays compact or device-resident
-> MCTX/search consumes that root batch directly
-> replay index/RND/target adapters read compact sidecars at the edge
```

This can still use the current contracts:

- `HybridCompactBatch` as the boundary of truth.
- `CompactRootBatchV1` and `CompactSearchResultV1` for search identity.
- `CompactReplayIndexRowsV1` for replay without hot-path tensor copies.

But the owner has to move down a level. A Python parent loop coordinating
actor/env step, observation stack rebuild, and compact consumer calls is now
the suspicious part.

## Missing Validation

Before claiming a big architecture win, the next rows need these gates:

- Same-denominator no-native/native repeats with full timer breakdown inside
  `env_step_sec`: actor step, collision/body update, renderer/latest frame,
  stack update, replay-index construction, RND latest-frame input, and sync.
- Normal death/autoreset rows, not only no-death or short clean loops.
- RND rows with `rnd_meter_v0`, including target/predictor state and latest
  observation extraction costs.
- Active-root accounting: distinguish physical rows, player roots, active
  roots, terminal roots, and padded roots in every roots/sec claim.
- Semantic parity at the compact edge: selected actions drive the next env
  step, legal masks stay binary, terminal/final-observation sidecars are valid,
  and target rows match the existing source-state builder at validation edges.
- A train-facing bridge check after the compact loop is fast: stock/profile
  denominator, no-RND and RND, with scalar LightZero objects used as adapters
  rather than silently reentering the hot loop.

## How To Explain The Current Speedup

Plainly:

```text
We made the compact closed loop faster by removing one object/payload layer.
That helped, especially at B1024. But the profiler says most time is still
spent stepping/updating the compact environment and observation batch around
search. So the next big win is not another search-only patch; it is making the
closed compact step/update path resident, lower-copy, and less parent-loop
synchronized.
```

The short version:

```text
Native actor buffer: real 1.2x-1.4x local win.
Search optimization from here: probably small on this denominator.
Next possible 2x+ move: cut env/observation/stack update itself.
5-10x claim: still unproven until repeated compact loop plus replay/RND edge
stays fast under normal death/autoreset and train-facing validation.
```
