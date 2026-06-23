# Long-Horizon Curriculum Plan

Status: active reorientation. Leaderboard feedback comes later.

## Core Change

Early training signal often takes a long time. Do not read a healthy row as
failed because it is flat at 30k or 50k. The first broad H100 wave should use
simple, static curricula that can run long enough to show whether the learner
ever finds a better checkpoint.

The early curriculum should be:

- exact trained checkpoint refs, not live leaderboard slots
- one shared trained initial seed where the axis is not seed quality
- simple hard-coded blank/wall immortal pressure
- mostly mortal checkpoint opponents, with small explicit immortal slices only
  when the recipe asks for them
- no trainer-facing leaderboard refresh
- no assignment refresh pointers for static controls

Leaderboard and tournament feedback are downstream selection tools. They should
not be the first curriculum source for this campaign.

## Keep Versus Simplify

Keep these old workarounds because they produced useful signal:

- Train from an already trained checkpoint instead of always starting cold.
- Use exact immutable `iteration_N.pth.tar` refs for opponent checkpoints.
- Include blank-canvas and wall-avoidant hard-coded opponents.
- Keep blank/wall hard-coded opponents immortal.
- Preserve best-so-far checkpoints, not only latest checkpoints.
- Run long enough to see mid-run peaks and late retention.

Simplify these early:

- Do not require a stable source leaderboard before launch.
- Do not run trainer-facing leaderboard refresh in Wave A.
- Do not mix multiple initial seeds inside the same reward/recipe comparison.
- Do not make every checkpoint opponent immortal.
- Do not use tournament rank as the first readout.
- Do not expand a live feedback loop until static exact-ref lanes have signal.

## Horizon Rules

Treat these as interpretation rules, not hard stop times:

| Horizon | Meaning |
| --- | --- |
| 0-30 minutes | Health only: heartbeat, progress, checkpoint/eval/RND metrics. |
| 30k-50k | Wiring and collapse check. Absence of learning is not meaningful yet. |
| 100k-170k | First useful directional read. AUC and best-so-far start to matter. |
| 240k-300k | Minimum retention read for broad rows. Do not promote before this. |
| 300k-600k | Continue if best/AUC is still improving or if collapse timing is the question. |
| 600k+ | Only for narrowed winners or explicit long-horizon falsification. |

Default posture: a healthy broad row gets to 300k unless it trips a real stop
condition. Flat early survival is not a stop condition.

Stop early only for:

- missing health artifacts after startup
- repeated crashes
- persistent action collapse across consecutive checkpoints
- wrong manifest/config/reward/opponent contract
- RND metrics missing or nonfinite on enabled rows
- no-refresh control writing refresh artifacts

## Static Exact-Ref Curriculum

The most useful early opponent source is a frozen file of exact checkpoint refs.
Current available source:

```text
artifacts/local/curvytron_restart_source_refs/restart18-source-loop18-top96-nonzero-20260515a/refs.txt
```

Use it for:

- initial policy seed, preferably one shared trained checkpoint
- static rank slots in opponent mixtures
- fixed-opponent RND bridge rows after the blank-canvas RND sweep
- long-horizon reward rows without live feedback

Use scratch only when the explicit question is cold-start learnability or RND
blank-canvas novelty. Scratch rows are slower to read and should not dominate
the first quality wave.

## Recommended Early Recipe Family

Keep recipes simple and exact-ref based:

| Recipe | Shape | Why |
| --- | --- | --- |
| `b20w05r1` | 20% blank, 5% wall, 75% trained checkpoint rank1 refs | Historical survival anchor, simplified. |
| `b10w05r1` | 10% blank, 5% wall, 85% trained checkpoint rank1 refs | Tests whether blank exposure mattered. |
| `b20w10r1` | 20% blank, 10% wall, 70% trained checkpoint rank1 refs | Tests wall pressure. |
| `b20w05top2` | 20% blank, 5% wall, split trained rank1/rank2 refs | Tests checkpoint diversity. |
| `b100` | 100% blank | Pure exploration/survival control. |
| `w100` | 100% wall | Pure scripted pressure control. |

Do not use a large ladder until a smaller recipe set has signal. Ladder recipes
are useful, but they make attribution harder.

## Wave A Shape After Leaderboard Deferral

Use broad parallelism for short health/breadth reads, but narrow before spending
long-horizon wall-clock. Do not spend the long-run budget on live leaderboard
feedback.

| Lane | Rows | Shape |
| --- | ---: | --- |
| RND blank wide | 45 | 9 points x 5 replicas, scratch/blank, no tournament. |
| Static exact-ref reward isolate | 18 | 3 rewards x 3 recipes x 2 noise, no refresh, seeded exact refs. |
| Long-horizon pretrained replicas | 18 | six replica manifests selecting `r005/r011/r017`: sparse, no-outcome, plus-outcome on the same clean rank1-heavy recipe. |
| Cadence/support panel | 9 | three knob manifests selecting `r005/r011/r017` under support/search/batch/TD changes. |
| Buffer | 10 | relaunches, failed capacity, or immediate fixed-opponent RND bridge. |

This is embarrassingly parallel and still interpretable. The main difference
from CZ26-style live feedback is that opponent source is fixed and auditable.

Runtime rule: the full 90-row shape is a short-sweep menu. For `2h-8h`, keep at
most 40 active H100 rows. For `8h+`, keep 10-20 rows and prioritize replicated
low-weight RND plus the smallest non-RND reward/cadence controls.

## Leaderboard Later

Attach tournament/leaderboard only after there are trained nonzero checkpoints:

1. Run diagnostic tournament over best-so-far and latest checkpoints.
2. Exclude or label `iteration_0` seeds.
3. Require exposure counts before reading rank.
4. Use tournament to pick opponent refs for the next static wave.
5. Only then consider trainer-facing assignment refresh.

Trainer-facing leaderboard refresh is a Wave C feature, not a Wave A
requirement.

## RND Long-Horizon Note

RND should still start with blank-canvas controls because that isolates novelty
plumbing. But if low weights show a real signal, the next RND bridge should use
the same exact-ref curriculum above, not a live leaderboard.

Also pass RND cadence explicitly. The builder default for
`--rnd-update-per-collect` is not the serious setting from the old RND notes.
Use `--rnd-update-per-collect 100` unless intentionally ablating cadence.
