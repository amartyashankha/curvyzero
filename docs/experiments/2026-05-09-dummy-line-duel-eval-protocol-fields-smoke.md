# 2026-05-09 Dummy Line Duel Eval Protocol Fields Smoke

## Question

Can Tiny Line Duel eval summaries expose the v0 eval protocol fields needed for
paired-seat multiplayer claims?

## Setup

- Code change: `run_dummy_line_duel_eval.py` accepts `--split-id` and
  `--split-role`.
- Summary now includes `eval_split`, `paired_seat_group_count`, and
  `pair_groups`.
- Pair-group summaries include wins by policy, wins by seat, seat delta,
  terminal causes, truncations, draws, mean steps, and rewards by policy.

## Command

```sh
python3 -m py_compile \
  src/curvyzero/training/dummy_line_duel_eval.py \
  scripts/run_dummy_line_duel_eval.py

PYTHONPATH=src python3 scripts/run_dummy_line_duel_eval.py \
  --episodes 2 \
  --seed 0 \
  --split-id dummy_line_duel_debug_v0 \
  --split-role debug \
  --output-dir artifacts/local/dummy_line_duel_eval_split_metadata_smoke
```

## Results

- Smoke completed with 18 total seated episodes.
- `eval_split` recorded split id, role, seed generation, seed count, seed-list
  hash, and `paired_seat: true`.
- `pair_groups` exposed policy-level aggregation and seat deltas.
- `one_step_safe` beat both `random_uniform` and `random_sticky` in paired
  groups with zero seat delta on this tiny smoke.

## Interpretation

The multiplayer eval artifact is now closer to the protocol. Future learned
Line Duel checkpoint evals can be judged by paired-seat groups rather than raw
seated rows, which helps catch seat-order wins, timeout farming, and action
collapse.

This is a tiny debug smoke, not a quality claim.

## Artifacts

- `artifacts/local/dummy_line_duel_eval_split_metadata_smoke/summary.json`
- `artifacts/local/dummy_line_duel_eval_split_metadata_smoke/episodes.jsonl`

## Follow-ups

- Add planner-only/untrained-model baseline for learned Line Duel checkpoint
  evals before making learning claims.
- Use at least selection/heldout split roles before treating a learned
  checkpoint as improved.
