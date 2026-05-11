# 2026-05-09 dummy Pong contact-pressure scoreability probe

## Question

Before more curriculum training, are real `pong_reset_profile=contact_pressure`
starts scoreable and action-sensitive under the true sparse dummy Pong reward?

## Setup

- Environment: project-owned dummy Pong.
- Reset profile: `contact_pressure`.
- Sample: 32 seeds with `pressure_agent=player_0` and 32 seeds with
  `pressure_agent=player_1`, for 64 reset states total.
- Action sweep: clone each reset state and try legal ego actions
  `up/stay/down`.
- Ego action mode: hold the candidate ego action until first contact, then use
  `track_ball` as ego continuation.
- Opponent policies: `track_ball`, `lagged_track_ball_1`, and `stay`.
- Reward: unchanged env reward only, `+1/-1/0`. No training and no pytest.

## Commands

```sh
uv run python -m py_compile scripts/probe_dummy_pong_contact_pressure_scoreability.py
```

```sh
uv run python scripts/probe_dummy_pong_contact_pressure_scoreability.py \
  --states-per-pressure-agent 32 \
  --seed 20260509 \
  --max-steps 64 \
  --output-dir artifacts/local/dummy-pong-contact-pressure-scoreability-probe-2026-05-09
```

## Results

- Rows: 576 rollouts = 64 reset states x 3 opponents x 3 ego actions.
- Groups: 192 reset/opponent groups.
- Action-sensitive groups: 192/192 (`1.0`).
- Contact-angle-sensitive groups: 192/192.
- Score-return-spread groups: 188/192 (`0.9792`).
- Scoreable groups, meaning at least one ego action scores: 105/192
  (`0.5469`).
- Row-level ego score probability across all action rollouts: `0.2899`.
- Mean score return by candidate action: up `-0.2448`, stay `0.4740`,
  down `-0.5156`.
- Mean survival steps by candidate action: up `23.74`, stay `43.125`,
  down `16.375`.

By opponent:

| Opponent | Action-sensitive | Scoreable | Score-return spread | Read |
| --- | ---: | ---: | ---: | --- |
| `lagged_track_ball_1` | 64/64 | 46/64 (`0.7188`) | 64/64 | scoreable curriculum target |
| `stay` | 64/64 | 59/64 (`0.9219`) | 64/64 | very weak sanity target |
| `track_ball` | 64/64 | 0/64 (`0.0`) | 60/64 | angle/survival-sensitive but not scoreable |

Example `player_0` seed `20260509` versus `lagged_track_ball_1`:

- `up`: center contact, outgoing `ball_vy=0`, truncates at 64, score `0`.
- `stay`: top contact, outgoing `ball_vy=-1`, wins in 38 steps, score `+1`.
- `down`: misses first contact, loses in 5 steps, score `-1`.

Example `player_0` seed `20260510` versus `track_ball`:

- `up`: misses, loses in 5 steps, score `-1`.
- `stay`: bottom contact, outgoing `ball_vy=1`, truncates at 64, score `0`.
- `down`: top contact, outgoing `ball_vy=-1`, truncates at 64, score `0`.

## Interpretation

The `contact_pressure` reset profile creates real action-sensitive states under
the true sparse reward. Legal ego action choice changes contact angle in every
sampled reset/opponent group and changes score return in nearly all groups.

The important caveat is the opponent. These sampled states are scoreable
against `lagged_track_ball_1` and `stay`, but not against default
`track_ball`. Default `track_ball` remains a bad score target even from this
contact-pressure slice; it can still diagnose angle and survival sensitivity.

## Decision

Go only for one modest same-curriculum diagnostic rung if it uses the
scoreable opponent slice, especially `lagged_track_ball_1`, preserves sparse
env reward, and scores initialization/final/best checkpoints under the same
eval. Stop for any training framed as "beat `track_ball` from
`contact_pressure`" or as stock Atari Pong progress.

## Artifacts

- `scripts/probe_dummy_pong_contact_pressure_scoreability.py`
- `artifacts/local/dummy-pong-contact-pressure-scoreability-probe-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-contact-pressure-scoreability-probe-2026-05-09/rows.jsonl`
