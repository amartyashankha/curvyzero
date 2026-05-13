# CurvyTron Learning Gates

Purpose: separate plumbing success from learning success.

## Not Enough

These are useful, but not learning proof:

- code compiles;
- Modal job runs;
- weights change;
- checkpoints save;
- GIFs change;
- greedy action changes;
- one metric improves at one checkpoint without a curve.

## Required For A Learning Claim

A run must report:

- trainer path and whether `train_muzero` was called;
- replay/target source: native LightZero or repo-owned;
- opponent source and exact self-play meaning;
- checkpoint ladder;
- survival curve;
- sparse outcome curve;
- reward component curve;
- action distribution curve;
- reset/randomness policy;
- eval protocol and number of episodes.

## Near-Term Gates

1. Stock fixed/frozen opponent canary calls `train_muzero` and strictly loads a
   checkpoint opponent. Status: passed on CPU in
   `stock-frozen-canary-source-state-s304-20260512` and on L4 GPU in
   `stock-frozen-gpu-base-canary-source-state-s304-20260512b`.
2. Stock fixed/frozen opponent control learns something.
3. Recent frozen-opponent stock route is either validated as a practical
   training route or rejected with evidence.
4. Stock centralized joint-action control learns something.
5. Native replay bridge passes target parity. Status: passed for a tiny
   hand-authored three-tick trace in Modal/LightZero. This is a bridge proof,
   not an integrated two-seat trainer proof.
6. True two-seat current-policy run improves survival and outcome under the
   native or parity-tested replay path.
