# Failure Analysis

Date: 2026-05-23

Question: why have we been saying "compact ownership" for days without making
it the real training path?

## Plain Answer

We implemented pieces of compact ownership, but we did not force one candidate
through the real Coach denominator.

We built useful sidecars and proofs. We did not make the trusted training loop
use those proofs end to end.

## What We Actually Built

- compact root/search/replay contracts;
- a compact search service boundary;
- compact rollout slab/profile harnesses;
- direct CTree and MCTX/JAX profile experiments;
- observation/render optimizations;
- sample-gate and replay-index proofs;
- profile manifests that protect live runs.

That work is not garbage. It is scaffolding and evidence.

## What We Did Not Build

- a Coach-ready compact-owned collector/search/replay path;
- a trainer-facing path that keeps data compact through collection, search,
  replay, RND, and learner input;
- a matched actual Coach training run proving the candidate path is faster;
- a semantic parity gate that says the fast search path is close enough to
  LightZero for training;
- a single current recommendation file that overrides stale old modes.

## Real Blockers

- The trusted lane changed several times: custom two-seat, stock
  fixed-opponent, stock current-policy, RND additions, rendering changes,
  action cadence changes.
- The stock LightZero path expects scalar env timesteps and replay/game objects.
  Compact ownership fights that boundary.
- MCTX/JAX is fast, but it does not currently match LightZero CTree behavior
  closely enough to be a drop-in replacement.
- RND and replay semantics matter. A fast collector that feeds the wrong replay
  rows is not useful.
- Live Coach runs must not be disturbed, so promotion needs small proof gates.

## Self-Inflicted Blockers

- We mixed speed currencies: actual iterations/hour, profile steps/sec,
  roots/sec, render medians, and synthetic ceilings.
- We kept too many old modes and stale docs in active language.
- We celebrated sub-loop wins before asking whether the real denominator moved.
- We used "probably" when the right move was to list competing hypotheses and
  kill them with matched ablations.
- We let profile-only code grow large without a strict promote/kill decision.
- We treated "do not touch live Coach runs" as if it meant "do not design a
  Coach-facing experimental interface." Those are different. We can build a
  fail-closed, opt-in smoke path without interfering with live runs.

## The Smallest Rule That Would Have Helped

Every optimizer result should end with exactly one label:

- promote to matched full-loop profile;
- keep as architecture evidence only;
- kill;
- rerun because denominator was wrong.

Every speed number should name its currency in the first sentence.

## Coach Promotion Contract

Every optimizer lane now needs this checklist before it can be discussed as a
Coach speedup:

```text
Does it call train_muzero? yes/no
Where is the trainer insertion point?
What stock path does it replace?
What learner-visible sample equivalence test must pass?
What capped stock-vs-candidate smoke promotes it?
If no trainer insertion exists, label result profile-only.
```

This would have prevented the main failure: building a good proof harness while
postponing the exact interface that would make it matter to training.
