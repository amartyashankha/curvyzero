# 2026-05-09 Vector Batch-Row Timing

## Question

Can the current fixture-seeded vector rows be stacked and stepped as many rows in
one NumPy call path, avoiding one Python step call per row?

This is not a full CurvyTron environment benchmark. It is a narrow batch-row
timing check for the currently supported fixture-seeded transition slice.

## Script

New script:

```sh
python3 scripts/benchmark_vector_batch_rows.py \
  --batch-sizes 128 \
  --event-modes debug-event no-event \
  --repeat 1000 \
  --warmup 100 \
  --body-capacity 4 \
  --format plain
```

What it does:

- Seeds the same eight currently supported fixtures.
- Runs source/common-trace state and event preflight once per fixture.
- Groups rows by fixed array profile: `P=2,K=4` and `P=3,K=4`.
- Builds one stacked state per group with leading row axis `B`.
- Calls one batched source-ordered NumPy step with row-specific
  `source_moves[B,P]` and `step_ms[B]`.
- Resets and emits the same fixed event rows as the single-row comparator for
  the supported event types: `position`, `point`, `die`, `score:round`,
  `score`, and `round:end`.
- Uses batched NumPy event-row writers for the common mask-based event emits in
  this benchmark path, with scalar single-row output still used as the preflight
  oracle.
- Checks the stacked state and event arrays against individual single-row
  comparator output before timing.
- Times reset-copy separately from the batched step bucket.
- Reports the measured fixed-event reset/emit/owner-lookup bucket as
  `event_sec` and as a percent of the timed step bucket.
- Can run `--event-modes debug-event no-event` to time the same state path with
  debug event rows emitted or skipped. Normal debug-event source and B>1 event
  preflight still runs; the no-event path adds a state-only preflight against the
  scalar debug-event output.

What is still fake or incomplete:

- Batch rows are made by cycling the current supported fixture seeds.
- `P=2` and `P=3` fixtures are not padded into one mixed production batch.
- Event rows are only covered for the eight default benchmark fixtures and the
  existing fixed event types.
- PrintManager/trail gaps are not in the supported batch-row slice.
- Observations, rewards, done/truncated outputs, reset/autoreset, replay, policy,
  and MCTS/search are not measured.
- This is CPU/NumPy only. It is not a GPU speed claim.

## Result

Local runtime labels from the script: macOS arm64, Python 3.11.14, NumPy 2.4.0.

Preflight passed: 8 fixtures passed, 0 failed, 0 unsupported. Batch state and
event preflight passed for both fixed-shape groups. The explicit no-event
state-only preflight also passed for both groups.

Event-mode comparison command:

```sh
python3 scripts/benchmark_vector_batch_rows.py \
  --batch-sizes 128 \
  --event-modes debug-event no-event \
  --repeat 1000 \
  --warmup 100 \
  --body-capacity 4 \
  --format plain
```

| Group | Mode | B | Rows | Step bucket | Event bucket | Event % | Rows/sec, step only | Top phase |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `P=2,K=4` | debug-event | 128 | 128,000 | 0.544871s | 0.276977s | 50.8% | 234,918.2 | `event_emit_sec:0.275126s` |
| `P=2,K=4` | no-event | 128 | 128,000 | 0.246904s | 0.000000s | 0.0% | 518,420.5 | `terminal_score_state_sec:0.100817s` |
| `P=3,K=4` | debug-event | 128 | 128,000 | 0.322040s | 0.146605s | 45.5% | 397,466.0 | `event_emit_sec:0.081500s` |
| `P=3,K=4` | no-event | 128 | 128,000 | 0.165829s | 0.000000s | 0.0% | 771,879.7 | `movement_sec:0.041800s` |

| Group | Debug step | No-event step | Debug minus no-event | Cost share | No-event step speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `P=2,K=4` | 0.544871s | 0.246904s | 0.297967s | 54.7% | 2.207x |
| `P=3,K=4` | 0.322040s | 0.165829s | 0.156211s | 48.5% | 1.942x |

Simple conclusion: debug event logging is currently about half of the measured
B=128 batch-row step bucket for this fixture slice. Turning it off raises the
micro-step rate from about 0.235M to 0.518M rows/sec for `P=2,K=4` and from
about 0.397M to 0.772M rows/sec for `P=3,K=4`.

The top visible phase is from lightweight timers inside the timed step loop, so
the reported step bucket includes that instrumentation overhead.

Change in this pass: the benchmark-local mask-based event emitters now append
rows with array assignments instead of one Python `_emit_event_row` call per
row. Event preflight still compares the stacked event arrays against the
single-row comparator before timing.

| Group | B | Timed rows | Step bucket | Event bucket | Event % | Rows/sec, step only | Rows/sec, reset+step | Top visible phase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `P=2,K=4` | 1 | 3,000 | 0.303151s | 0.074944s | 24.7% | 9,896.1 | 8,970.9 | event emit |
| `P=2,K=4` | 8 | 24,000 | 0.523386s | 0.181389s | 34.7% | 45,855.2 | 43,176.0 | event emit |
| `P=2,K=4` | 32 | 96,000 | 0.749409s | 0.313000s | 41.8% | 128,101.0 | 122,567.1 | event emit |
| `P=2,K=4` | 128 | 384,000 | 1.642202s | 0.834801s | 50.8% | 233,832.4 | 228,103.6 | event emit |
| `P=3,K=4` | 1 | 3,000 | 0.372288s | 0.079828s | 21.4% | 8,058.3 | 7,442.2 | body collision |
| `P=3,K=4` | 8 | 24,000 | 0.597601s | 0.187316s | 31.3% | 40,160.5 | 38,104.1 | event emit |
| `P=3,K=4` | 32 | 96,000 | 0.648204s | 0.230176s | 35.5% | 148,101.5 | 140,738.0 | event emit |
| `P=3,K=4` | 128 | 384,000 | 0.974900s | 0.444706s | 45.6% | 393,886.4 | 377,232.1 | event emit |

Small before/after check at `B=128`, `--repeat 1000 --warmup 100`:

| Group | Before rows/sec | After rows/sec | Before event bucket | After event bucket | Event % before -> after |
| --- | ---: | ---: | ---: | ---: | ---: |
| `P=2,K=4` | 144,885.0 | 232,555.8 | 0.610669s | 0.281105s | 69.1% -> 51.1% |
| `P=3,K=4` | 176,317.9 | 411,054.0 | 0.546257s | 0.143369s | 75.2% -> 46.0% |

Post-change phase split at `B=128`, `--repeat 1000 --warmup 100` shows event
reset is not the culprit: `event_reset_sec` is about 0.002s for both groups.
The remaining event bucket is mostly batched emit writes for `P=2,K=4`
(`event_emit_sec` about 0.280s) and split between emit writes and body-hit owner
lookup for `P=3,K=4` (`event_emit_sec` about 0.079s,
`event_body_hit_owner_sec` about 0.063s).

The main useful fact: the current fixture-seeded row transition can run as real
B>1 stacked arrays while carrying fixed event rows. At `B=128` it is roughly
0.23M to 0.39M supported rows/sec on this local CPU with the benchmark-local
batched event-row writers. The event bucket is still visible at larger `B`, but
the cost is now specific: remaining emit writes for score-heavy rows and the
body-hit owner lookup path, not event reset.

With debug events skipped, the remaining local micro-step work is state update
work: terminal score state updates in the `P=2,K=4` wall-death-heavy group and
movement/body/collision work in the `P=3,K=4` group. Those are the next honest
targets after deciding which event rows need to exist on the production hot
path.

## Next Bottlenecks

1. Decide which debug event rows must stay in the production hot path; skipping
   them roughly doubles the B=128 micro-step rate on this fixture slice.
2. Add PrintManager and trail-gap semantics before broadening the fixture mix.
3. Replace the first debug observation/reward packer with a trainer-facing
   observation/reward schema when source state/events are ready.
4. Add row reset/autoreset without changing batch shape.
5. Connect the env rows to policy/search rows and replay staging, then report
   both env rows/sec and ego decision rows/sec.
