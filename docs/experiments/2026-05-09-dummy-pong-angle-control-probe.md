# 2026-05-09 dummy-pong-angle-control-probe

## Question

Can a simple scripted Pong policy intentionally use off-center paddle contacts,
and is that enough to beat the existing `track_ball` baseline?

## Setup

- Environment: `dummy_pong_v0`
- Probe policy: `angle_control`
- Baselines: `track_ball`, `random_uniform`
- Episodes: 16 per seating
- Step cap: 120
- Seed: 0
- Probe artifact schema: `dummy_pong_angle_control_probe_v0`

`angle_control` tracks normally while the ball is moving away. When the ball is
incoming, it predicts the contact row and aims for a top or bottom paddle hit.

## Command

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_eval.py \
  scripts/run_dummy_pong_angle_control_probe.py
```

```sh
uv run python scripts/run_dummy_pong_angle_control_probe.py \
  --episodes 16 \
  --seed 0 \
  --output-dir artifacts/local/dummy-pong-angle-control-probe-smoke
```

## Results

- `angle_control` versus `random_uniform`, paired seats:
  - `angle_control` won 32/32 games.
  - No games truncated.
  - `angle_control` made 23/23 off-center contacts.
  - `random_uniform` made 9/12 off-center contacts.
- `angle_control` versus `track_ball`, paired seats:
  - 32/32 games truncated at 120 steps.
  - No wins for either policy.
  - `angle_control` made 172/172 off-center contacts.
  - `track_ball` made 163/180 off-center contacts.
- Output artifacts:
  - `artifacts/local/dummy-pong-angle-control-probe-smoke/summary.json`
  - `artifacts/local/dummy-pong-angle-control-probe-smoke/episodes.jsonl`

## Interpretation

The probe can deliberately create off-center contacts. That part of the mini
North Star is measurable now.

Off-center contact alone is not enough to beat `track_ball` in the current tiny
Pong setup. Against `track_ball`, the policy creates only top/bottom returns but
the games still time out. This means the next gap is not paddle-angle plumbing;
it is learning or searching for a sequence of returns that makes `track_ball`
arrive late.

The random result is a useful sanity check: the same scripted policy wins when
the opponent fails to track well, so the probe is not broken or score-blind.

## Artifacts

- `src/curvyzero/training/dummy_pong_eval.py`
- `scripts/run_dummy_pong_angle_control_probe.py`
- `artifacts/local/dummy-pong-angle-control-probe-smoke/summary.json`
- `artifacts/local/dummy-pong-angle-control-probe-smoke/episodes.jsonl`

## Follow-ups

- Add learned/search policy support to this probe only after a policy can choose
  between return patterns, not just single off-center hits.
- Consider a score-pressure eval with a longer horizon or smaller paddle if
  `track_ball` remains too stable for this geometry.

## Implementation Follow-up

The next LightZero dummy Pong move is now implemented as an explicit custom
curriculum reset profile, not as stock Atari Pong benchmark replication:
`pong_reset_profile=contact_pressure`. It biases reset states toward imminent
paddle-contact/scoring-pressure positions while keeping the env reward sparse
`+1/-1/0`. See
`docs/experiments/2026-05-09-lightzero-dummy-pong-contact-pressure-curriculum-smoke.md`.
