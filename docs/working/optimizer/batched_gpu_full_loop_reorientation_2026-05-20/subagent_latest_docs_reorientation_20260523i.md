# Latest Docs Reorientation, 2026-05-23i

Scope: docs-only sidecar pass. I read the requested optimizer and training
source-of-truth docs. I did not edit code, touch live Coach runs, start Modal
jobs, mutate checkpoints, or change trainer defaults.

## Verdict

We are properly oriented, not in a random rabbit hole yet.

The lane stays sane only if it remains profile-only until the real LightZero
checkpoint bridge and compact-service validation gates pass. MCTX/JAX is worth
continuing. The older fast rows used a toy JAX model, but that is now
superseded by real immutable-checkpoint MCTX shadow rows. Current real-shadow
profile rows show `1.36x-2.37x` on the measured rows, with B1024 scalar-off at
`2.20x`. They are still not Coach training speed.

## Current High-Level Goal

Make CurvyTron training faster without changing what Coach is trying to train.
The trusted Coach lane remains stock LightZero `train_muzero`; the optimizer
lane is trying to prove a faster compact search/dataflow backend behind
`CompactSearchServiceV1` while preserving observation, action, replay, RND,
player-perspective, sampler, and learner-facing sample meaning.

Plain shape:

```text
Keep Coach on stock LightZero.
Use profile-only compact service gates to price and validate faster search.
Only discuss Coach-facing experiments after real-model parity and matched
stock-vs-candidate full-loop-shaped smoke pass.
```

## Trusted

- Coach truth: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train`
  with stock LightZero `train_muzero`, not old `two-seat-selfplay`.
- Current optimizer semantic boundary: `CompactRootBatchV1` ->
  `CompactSearchServiceV1` -> `CompactSearchResultV1` ->
  `CompactReplayIndexRowsV1`.
- Compact replay/index validation is meaningful: identity sidecars, selected
  action, visit policy, root value, terminal `final_observation`, RND
  latest-frame attachment, public LightZero sampler parity, and a real
  direct-CTree compact-service closed-loop smoke are covered locally.
- Fresh-model PyTorch -> JAX shadow parity passed on Modal L4 GPU at practical
  tolerance. Current immutable checkpoint `cz26a` `iteration_260000.pth.tar`
  loaded strictly and consumed required keys.
- Recurrent-from-PyTorch-latent diagnostics say dynamics/prediction are mostly
  aligned. The remaining checkpoint mismatch is root representation drift, so
  this is a partial diagnostic, not a pass.

## Profile-Only

These are optimizer evidence, not Coach launch advice:

- MCTX/JAX compact service rows. A real checkpoint-backed JAX shadow model is
  now wired and profiled, but MCTX search semantics differ from LightZero CTree,
  checkpoint parity is close-not-exact, and no `train_muzero` backend exists.
- `direct_ctree_gpu_latent`, `dense_torch_mcts`,
  `compact_torch_search_service`, `service_tax_probe`, and
  `mock_search_service`.
- Compact slab, borrowed render state, parent compact render-state buffer,
  resident-stack, refresh-off, closed-loop deferred payload, and other
  profile-only observation/search ownership rows.
- Current speed claims unless they name the denominator exactly: Coach
  training, stock full-loop profile, or profile-only compact boundary.

## Stale Or Should Not Be Used

- Mutable checkpoint aliases: `latest.pth.tar` and `ckpt_best.pth.tar`.
- Old checkpoint refs from docs that are no longer present, including the
  documented `iteration_32.pth.tar` ref.
- Old custom `--mode two-seat-selfplay` and old historical launch generators
  for learning claims.
- Historical render-mode notes such as `fast_gray64_direct` or old
  browser/body-circles conclusions as current optimizer advice. Rendering work
  helped, but the current wall is search/control/dataflow plus env/observation
  handoff.
- Dense/eager Torch search, flat-A3 CTree, parent compact render-state buffer,
  service-tax, and mock rows as promoted paths. They are controls, diagnostics,
  or staging surfaces unless new same-denominator validation overturns the
  current read.

## Next 3 Gates

1. Finish current checkpoint parity diagnostics on immutable checkpoints.
   Include scalar support-transformed value/reward checks, recurrent from JAX
   latent, recurrent from PyTorch latent, and practical tolerance gates on
   `cz26a` `iteration_260000` plus the r18fresh champion `iteration_250000`.

2. If semantic checkpoint diagnostics pass, wire the real JAX shadow model into
   `MctxCompactSearchServiceV1` behind `CompactSearchServiceV1`, still labeled
   `profile_only`, `calls_train_muzero=false`, and `touches_live_runs=false`.
   Preserve legal-mask, active-root, selected-action, visit-policy, root-value,
   replay-index, terminal, RND, and player identity gates.

3. Run matched same-denominator profile rows: direct CTree, real-model MCTX,
   toy-MCTX ceiling, service-tax, and mock ceilings. Only after that should the
   thread decide whether a capped Coach-facing experimental backend is worth
   designing.

## Contradictions And Misleading Stale Reads

- There is no real contradiction between "MCTX is promising" and "MCTX is not
  Coach-ready." The docs agree: it is a real profile-only speed signal, but not
  LightZero-equivalent until the real checkpoint bridge passes.
- The checkpoint docs can mislead if read too quickly: fresh-model parity
  passed, but current checkpoint parity has not passed. Strict load/key coverage
  passed; raw latent tolerance failed; recurrent-from-PyTorch-latent narrowed
  the problem to representation drift.
- Older training docs are useful for Coach truth but stale for optimizer speed
  direction. They preserve render/cadence/run-history context, while the current
  optimizer docs say not to chase render-only or old custom trainer paths.
- "Fast roots/sec" rows are easy to overread. Service-tax/mock/dense/MCTX rows
  are denominator probes, not learning-quality or run-quality evidence.

## Bottom Line

Stay on this lane, but keep the guardrails tight: current immutable checkpoint
parity first, real-model MCTX compact-service smoke second, matched
same-denominator profile third. If any gate fails structurally, stop the JAX/MCTX
bridge and return to the broader compact search/dataflow architecture question.
