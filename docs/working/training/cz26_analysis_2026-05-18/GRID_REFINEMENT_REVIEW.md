# Grid Refinement Review

Captured: 2026-05-18.

This is not a launch decision. It is the current interpretation of what the
CZ26 data suggests for the next experiment design.

## Stronger Signals

- Intermediate checkpoints matter. Many runs peak before the latest checkpoint.
- Pure rank1 opponent training is weak in Grid B.
- Some blank/wall support is useful.
- `n10` is the cleanest eval-survival noise setting.
- `out67` and `out100` are still worth studying because they produce useful
  mid-run or tournament signals.
- `b20w05r1`, `b10w05r1`, `b20w05top2`, `b25w25r1`, and `b20w05lad4` are the
  most interesting recipe families.

## Weaker Signals

- `out33` looks weak.
- Grid A does not support `imm10`.
- Pure `r1` looks bad.
- Pure hard-coded recipes can spike survival but tend to regress.
- Fine tournament ordering is weak because many top learned checkpoints have
  only 1-6 battles.

## Confounded Signals

- Recipes change blank mass, wall mass, rank concentration, and rank diversity
  at the same time.
- Reward curves can drop because opponents get stronger, not only because the
  policy gets worse.
- Latest survival can be low even when an intermediate checkpoint ranked well.
- Tournament rank can be a sparse exposure spike.

## What To Preserve In The Next Analysis Loop

- Always keep best checkpoint, latest checkpoint, and tournament champion
  separate.
- Always report games and battles beside tournament rank.
- Always ignore `iteration 0` for learned-policy analysis.
- Always run exact-horizon and matched comparisons before interpreting a
  projection table.
- Always keep reward, survival, and tournament rank separate.

## Candidate Follow-Up Axes

These are questions to refine, not final decisions.

1. Reward outcome strength:
   - keep `out0`;
   - keep a middle outcome setting;
   - keep a stronger outcome setting only if tournament rank continues to
     justify it.
2. Noise:
   - keep clean and `n10`;
   - use `n20` only where tournament diversity is the question.
3. Recipes:
   - preserve `b20w05r1` as an anchor;
   - preserve `b25w25r1` because it retained survival well in Grid B;
   - keep a ladder recipe because it may help matchups;
   - avoid pure `r1` except as a diagnostic control.
4. Immortality:
   - do not treat `imm10` as default from current evidence;
   - if testing immortality again, test it in recipes where leaderboard slots
     are actually the main pressure.

## Needed Before Next Launch

- Decide whether the next batch selects initial policies from best learned
  tournament checkpoint, best survival checkpoint, or a fixed seed.
- Decide whether opponent refresh should happen only at checkpoint boundaries.
- Decide how to preserve intermediate checkpoints that outperform the latest
  checkpoint.
- Decide whether to increase tournament exposure for promising learned
  checkpoints before using their rank as selection evidence.
