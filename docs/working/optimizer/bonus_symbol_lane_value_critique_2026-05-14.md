# Bonus Symbol Lane Value Critique

Date: 2026-05-14

Scope: critique whether a simple-symbol bonus renderer is worth pursuing now.
The criterion is not human icon clarity or pixel-perfect sprite art. The policy
only needs bonus classes to be distinguishable enough in the final `[4,64,64]`
grayscale observation stack to learn their effects.

## Bottom Line

Do not implement a new simple-symbol renderer now.

The current alternatives already cover the useful ends of the tradeoff:
`browser_sprites` is the trusted reference/default, and the current fast
luma-coded bonus circles already give the policy a deliberately class-coded
signal. A simple-symbol lane would mostly add another approximation surface
unless it proves better separability or cheaper GPU implementation than luma
circles.

## Amdahl Upside

The measured large wins are from broad render-path changes, not from bonus
icons specifically. Local env-only direct gray64 can be about `3x` to `8x`
faster than `browser_lines`, but stock full-loop wins seen so far are only about
`1.3x` to `1.5x` because collection, search, policy forward, subprocess
overhead, and learner work remain.

A bonus-symbol-only change is therefore small unless bonus drawing is a
surprisingly large sub-bucket. It cannot plausibly exceed the whole fast-render
full-loop gain, and in the likely case where bonus drawing is a minority of
render time, the end-to-end gain is probably low single digits. The only strong
Amdahl case is if symbols remove a blocker for a broader GPU/compiled renderer.

## Implementation Cost

This is not just drawing twelve masks. A credible lane needs:

- an explicit opt-in bonus render mode;
- train/eval metadata so render mismatch cannot happen silently;
- CPU and possible GPU implementations or parity rules;
- class separability tests over offsets, edges, downsample phase, draw order,
  trails/heads, and player-perspective remap;
- matched speed and learning rows before any recommendation.

That cost is hard to justify while luma circles already encode bonus class and
`browser_sprites` remains the reference.

## Are Separability Tests Enough?

No. Separability tests are a gate, not a decision.

They can prove the renderer is not obviously broken: no exact class collisions,
enough pairwise margin after grayscale/downsample, no remap collisions, and no
fragility at common offsets or overlaps. They cannot prove the policy learns
the same effects, because class coding changes the input distribution and may
interact with heads, trails, temporal stacking, and reward timing.

The current signature read actually argues against urgency: `browser_sprites`
are unique but close after downsample, while existing luma-coded circles are
more separated by construction. The direct fast path has a specific luma-remap
risk around `BonusGameClear`; fixing/reserving luma codes is cheaper than
inventing a full symbol renderer.

## Deciding Experiment

Before implementing symbols, run a local/offline bound:

1. Measure the maximum bonus-only speedup with a `no_bonus_draw` or stub branch
   inside the final `[4,64,64]` observation path.
2. Run separability on `browser_sprites`, current luma circles, and the direct
   fast luma path across offsets, edges, overlaps, and perspective remap.
3. Only prototype symbols if current luma circles fail separability or if
   bonus sprite handling is a measured blocker for the GPU/compiled renderer.

Promotion threshold should be strict: at least about `5%` stock full-loop wall
improvement or a clear GPU-render unblock, plus separability margins better
than current luma circles. Otherwise keep the existing paths.

## Recommendation

Wait. Keep `browser_sprites` as reference and keep current luma circles as the
fast approximation. Spend optimizer time on full-loop attribution,
collection/search/process overhead, and full-fidelity GPU/compiled rendering.

Simple symbols are worth a quick probe only if the existing luma-coded classes
fail final-stack separability or the GPU renderer needs mask symbols to avoid a
larger bonus-sprite implementation cost.
