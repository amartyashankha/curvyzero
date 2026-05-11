# Toy Baseline Smoke

Date: 2026-05-08

## Question

Can the repo run a tiny training-adjacent harness that produces structured
episode metrics and optional local artifacts?

## Setup

Local harness:

```sh
uv run python scripts/run_toy_baseline.py --episodes 100 --seed 0
```

Artifact smoke:

```sh
uv run python scripts/run_toy_baseline.py \
  --episodes 2 \
  --seed 0 \
  --output-dir artifacts/local/toy_baseline_smoke
```

## Result

The harness ran two matchups:

| Matchup | Episodes | Player 0 wins | Player 1 wins | Draws | Mean steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| random_vs_random | 100 | 47 | 43 | 10 | 4.63 |
| privileged_survival_heuristic_vs_random | 100 | 29 | 58 | 13 | 4.04 |

The artifact smoke wrote:

- `artifacts/local/toy_baseline_smoke/summary.json`
- `artifacts/local/toy_baseline_smoke/episodes.jsonl`

## Interpretation

This is an infrastructure smoke, not a learnability gate. The privileged
heuristic losing to random means the current toy env/control/heuristic surface
is not yet a meaningful baseline. Do not promote this result as evidence that
the game is learnable or not learnable.

Useful takeaway: the repo now has a simple command shape that emits structured
run summaries. The next training-loop scaffold should keep this artifact style
but move to a deliberately single-player survival task before adding real
MuZero search or multiplayer self-play.

## Follow-ups

- Implement the solo turning-survival dummy training loop.
- Use one compact JSON/JSONL/NPZ artifact layout for dummy runs.
- Add a coarse Modal wrapper only after the local dummy loop exists.
- Revisit 1v1 heuristic baselines after the env interface and toy task are less
pathological.
