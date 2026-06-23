# Hybrid Policy/Search Probe Risk Note

Date: 2026-05-21

Scope: design critique for a profile-only injected policy/search pressure probe.
No source code, trainer defaults, tournament defaults, checkpoint/eval paths, or
live runs were changed for this note.

## Context

The current hybrid profile path is intentionally not stock LightZero training:

```text
in-process CurvyTron actors
-> parent merge
-> zero or injected renderer-backed [B,P,4,64,64] stack
-> scalar LightZero-shaped rows at the outer edge
-> optional injected policy/search pressure probe over flat_obs
```

That is the right dependency direction. The training module owns only the
profile seam and probe protocol; Modal can own the JAX implementation. The risk
is interpretive: this can be a useful pressure meter, but it is not evidence
that real LightZero MCTS is fast unless the output makes that boundary painfully
explicit.

## Ways The Probe Can Mislead

- It can time a convenient batch kernel instead of the real collection loop. The
  hybrid harness currently samples random actions before the probe and does not
  feed probe-selected actions back through LightZero collector semantics,
  replay, learner targets, or action selection.

- It can hide scalarization cost. The probe receives `flat_obs`, which already
  exists after scalar materialization. If a future design bypasses this path, or
  if stock LightZero uploads observations differently through Torch, the probe
  timing will not equal real policy/search timing.

- It can confuse model-pressure with MCTS. A JAX function that does batched
  matrix multiplies, top-k, or toy tree loops may create useful GPU load, but it
  is not LightZero MCTS unless it preserves the same root count, legal action
  masking, priors/value handling, dynamics/recurrent model calls, backup logic,
  exploration noise/temperature behavior, terminal handling, and simulation
  count.

- It can double-count or under-count host/device movement. The renderer path may
  already copy env state to GPU and frames back to host; the probe may then copy
  flat observations back to GPU. That is honest for the current host-bound
  scaffold, but not for a hypothetical device-resident search path.

- It can be distorted by JAX compile and async dispatch. Warmup calls exist, but
  probe telemetry must distinguish compile/prewarm from measured device work and
  force synchronization before reporting `device_sec`.

- It can contend with the renderer on the same GPU in a way that is either real
  or artificial. If renderer and probe serialize on one stream/device, this is a
  profile of the current Modal harness, not proof that a production pipeline
  cannot overlap work.

- It can look better than normal-death training. No-death rows keep root batch
  size stable. Normal death, partial row renders, autoreset, final observation,
  and live-root collapse change both search shape and observation shape.

- Compact Modal output can erase the evidence. The training-side profile returns
  `policy_search_probe_backend_name` and `policy_search_probe_calls`, but the
  compact Modal result must also preserve these fields plus probe shape/semantic
  metadata. Otherwise downstream notes may compare a probe row without knowing
  what was actually timed.

## Required Metrics

Every serious probe row should report:

- `profile_only=true`, `calls_train_muzero=false`,
  `stock_lightzero_integrated=false`, `touches_live_runs=false`, and exact
  `policy_search_probe_backend_name`.
- Probe call count, measured root count per call, total roots, simulations or
  synthetic simulation-equivalent count, model-eval count, action count, and
  whether action masks were consumed.
- Input/output shapes and dtypes: flat observation shape, batch/root shape,
  policy logits/action output shape, value output shape, and any readback shape.
- Timing split: probe total, host-to-device, device compute, readback,
  synchronization, and any compile/prewarm time excluded from measured rows.
- Bytes moved: observation H2D bytes, output/readback bytes, and whether arrays
  were newly allocated or reused.
- Renderer/probe separation: actor step, parent send/receive, gather/merge,
  renderer render/device render, scalar materialization, probe timing, and
  measured wall clock in the same row.
- Workload anchors: batch size, actor count, player count, rows per step,
  timesteps, terminal/autoreset/done counts, warmup steps, measured steps,
  `jax.default_backend`, devices, GPU memory/utilization.
- Semantic declaration: `probe_semantics` such as `synthetic_policy_pressure`,
  `synthetic_tree_pressure`, or `lightzero_mcts_compatible_canary`; default
  should be synthetic until proven otherwise.

## Guardrails

- Name it as a probe everywhere. Avoid row labels, field names, or chart titles
  that say "MCTS" unless the implementation is actually exercising LightZero
  MCTS-compatible semantics.

- Fail closed on identity. A requested GPU/JAX probe should fail before the
  measured loop if the probe is missing, reports an unexpected backend name, or
  omits required telemetry.

- Keep the injection one-way. The training module should accept the protocol;
  Modal should construct the JAX implementation. Do not import Modal/JAX probe
  internals into the training module.

- Preserve profile-only flags in compact outputs. Compact summaries should carry
  probe backend, probe calls, probe semantic label, and known gaps, not only
  aggregate timing fields.

- Do not use probe throughput as launch advice. Compare it only against profile
  anchors and label it as topology/search-pressure evidence, not real training
  throughput.

- Keep action application separate until explicitly tested. If the probe starts
  choosing actions, add row/player/action-mask tests before interpreting speed,
  because a fast wrong-action loop is worse than a slow honest meter.

- Keep RND, normal death, subprocess IPC, and real stock `train_muzero` rows as
  separate gates. A no-death in-process probe can show headroom, but it cannot
  validate full-loop LightZero behavior.

## Bottom Line

This design is sound as a pressure meter if the result schema makes the
synthetic boundary explicit. The most important additions are not more FLOPs;
they are identity, semantic labels, shape/root counts, byte movement, sync-safe
timing, and compact-output guardrails. Without those, the probe could become a
fast benchmark for something adjacent to policy/search and then be mistaken for
real LightZero MCTS progress.
