# Strategy Reorientation, 2026-05-23b

Status: active optimizer working memory. No live Coach training runs should be
touched from this lane.

## Plain Answer

We are not ready to recommend a new faster training backend to Coach.

We are ready to take a bigger optimizer move in the profile lane:

```text
build and test a compact search/dataflow slab that keeps batches alive through
observation -> search -> action -> env -> replay/sample
```

The important correction is that the recent fast rows are mostly profile-only
probes. They are useful evidence, but they are not real Coach training speed.

2026-05-23c update:

```text
This still holds after the validation-ladder and MCTX sidecar reports.
The big move should be search/dataflow ownership behind CompactSearchServiceV1,
not polishing eager compact Torch and not changing Coach training defaults.
```

## What Is Actually Going On

The old render work helped, but it did not unlock a large real-loop speedup by
itself. The current wall is the shape of the loop:

```text
CPU env rows
-> observation/root batch
-> GPU model
-> LightZero CTree search
-> CPU/list/object materialization
-> replay/RND/learner edges
```

Some payloads are small by bytes, like actions, values, masks, and visit
policies. They can still be expensive if they force a wait or Python object
work many times per search.

The current eager `compact_torch_search_service` is not good enough. Fresh
same-denominator rows say it loses to direct CTree:

```text
H100 B512/A16, 80 measured + 20 warmup, replay proof on, root noise 0.0

sim16:
  direct CTree:   5.47k steps/sec
  compact Torch:  4.05k steps/sec
  service-tax:    7.81k steps/sec
  mock ceiling:   7.46k steps/sec

sim32:
  direct CTree:   3.14k steps/sec
  compact Torch:  2.67k steps/sec
  service-tax:    5.19k steps/sec
  mock ceiling:   9.17k steps/sec
```

Plain read:

```text
Do not polish eager compact Torch search as the next main lane.
Do keep the compact service/replay contract.
Do build a better backend behind that contract.
```

## Current Amdahl Read

The current 5x-10x hypothesis is not "render faster" or "buy a bigger GPU" by
itself.

The current 5x-10x hypothesis is ownership:

```text
compact/vector state owner
resident observation stack or compact root batch
batched search service with minimal CPU sync
selected actions returned once per env tick
replay/RND payloads flushed in chunks
stock LightZero rows only at validation/sample/debug edges
```

The unavoidable sync while the env is CPU:

```text
selected joint actions -> CPU env step
```

The syncs and materialization we should avoid:

```text
full observation stack bouncing host/device
per-simulation recurrent output readback
Python list/tree/object fanout
full replay rows during collection
RND per-frame list/tensor work in the hot loop
```

## External Pattern Check

Fresh web research matches our local conclusion:

- MCTX is JAX-native, batched, and JIT-compilable. It is the cleanest reference
  for accelerator-native MuZero/Gumbel MuZero search, but using it inside a
  PyTorch/LightZero loop can create a bridge tax if we cross frameworks too
  often.
- PufferLib/EnvPool-style systems use shared/static buffers and zero-copy
  batches. The lesson for CurvyTron is not "switch libraries tomorrow"; it is
  to stop passing scalar Python env objects through the hot path.
- OpenSpiel's AlphaZero docs separate actors, learner, evaluators, and analysis.
  Their Python path is deliberately simple and CPU-heavy; it is a warning, not
  a speed model.
- Large AlphaZero-style systems scale by many self-play actors plus batched
  accelerator inference/search/training. That does not remove the need for a
  clean local compact data path.

Sources to keep handy:

- https://github.com/google-deepmind/mctx
- https://puffer.ai/docs.html
- https://pufferai.github.io/build/html/rst/landing.html
- https://openspiel.readthedocs.io/en/latest/alpha_zero.html
- https://github.com/opendilab/LightZero

## Wrong Next Moves

- Telling Coach to use `compact_torch_search_service`.
- Claiming profile-only roots/sec as real `train_muzero` training speed.
- More tiny render tuning without showing it moves the full denominator.
- Buying H100s as the primary fix while Python/object/search boundaries remain
  hot.
- Rewriting replay or learner first without proving search/dataflow is no
  longer the wall.
- Adding another parent-side full trail/state copy and calling it a GPU path.

## Best Next Big Move

The next aggressive move should be:

```text
CompactSearchServiceV1-backed profile slab
with a better search backend candidate
and full observation->sample validation
```

Backend candidates, in current order:

1. Compiled/fused fixed-shape search. Good if it can beat direct CTree in the
   same denominator; current eager Torch failed that test.
2. JAX/MCTX comparator. Good reference for accelerator-native search; risky as
   a production bridge unless the whole search/model path stays in JAX.
3. Puffer-style native/vector slab. Bigger rewrite, but likely the cleanest
   route if env/replay/object ownership becomes the true wall.
4. Fixed-action flat-A3 CTree. Keep as a valid control and compatibility
   bridge, but demote it as a main speed lane because the matched H100
   train-profile row was direct CTree `516.55` steps/sec versus flat-A3
   `509.69` steps/sec.

## Immediate Plan

- Keep live Coach runs read-only.
- Keep stock LightZero training as the semantic oracle.
- Use profile-only rows for speed experiments.
- Update summaries so every row says whether it is Coach training, stock
  full-loop profile, or optimizer probe.
- Run the next proof in the compact slab path, not by changing trainer defaults.
- Kill any backend that cannot beat direct CTree on a no-noise same-denominator
  profile and cannot pass replay/RND/terminal/sample gates.

## Parallel Critique Wave

Dispatched on 2026-05-23:

- `subagent_current_bottleneck_recritique_20260523b.md`
- `subagent_external_fast_rl_patterns_20260523b.md`
- `subagent_validation_ladder_recritique_20260523b.md`
- `subagent_big_move_options_20260523b.md`
- `subagent_mctx_comparator_status_20260523b.md`
- `subagent_fixed_a3_ctree_status_20260523b.md`

Fold those results back into `world_model.md`, `task_board.md`, and this file
before promoting the next implementation lane.
