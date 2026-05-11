# 2026-05-09 dummy pong imitation v0 heldout scoreboard

## Question

Does the selected epoch-1000 imitation checkpoint hold up on a separate heldout
scoreboard split?

## Setup

- Selection record:
  `artifacts/local/dummy-pong-imitation-v0-e1000-selection-record-2026-05-09/selection_record.json`
- Heldout scoreboard output:
  `artifacts/local/dummy-pong-imitation-v0-e1000-scoreboard-heldout-2026-05-09`
- Selected checkpoint:
  `artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz`
- Previous checkpoint:
  `artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-000750/checkpoint.npz`
- Split: `dummy_pong_imitation_v0_heldout`
- Split role: `heldout`
- Episodes per seated matchup: `32`
- No pytest.

## Command

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 32 \
  --seed 211 \
  --split-id dummy_pong_imitation_v0_heldout \
  --split-role heldout \
  --checkpoint selected_best=artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz \
  --checkpoint previous=artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-000750/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-imitation-v0-e1000-scoreboard-heldout-2026-05-09
```

## Results

- Baseline sanity:
  - `track_ball` beat `random_uniform` 64/64.
  - `track_ball` versus `track_ball` truncated 32/32.
- Selected epoch 1000:
  - beat `random_uniform` 37/64.
  - won 0/64 against `track_ball`.
  - `track_ball` won 9/64, with 55 truncations.
- Previous epoch 750:
  - beat `random_uniform` 41/64.
  - won 0/64 against `track_ball`.
  - `track_ball` won 48/64, with 16 truncations.
- Selected versus previous:
  - selected won 50/64.
  - previous won 6/64.
  - 8 truncations.

## Interpretation

The selected checkpoint is useful as the current best imitation checkpoint, but
not as a good Pong policy. It strongly beats the previous checkpoint and survives
much longer against `track_ball`, but it still scores zero wins against
`track_ball`.

The random baseline result is mixed: previous beat random 41/64 while selected
beat random 37/64 on this heldout split. Do not claim broad policy improvement.
The honest claim is narrower: selected is better than previous in direct
checkpoint play and less often loses to `track_ball`, but still cannot beat the
scripted gate.

## Artifacts

- `artifacts/local/dummy-pong-imitation-v0-e1000-scoreboard-heldout-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-imitation-v0-e1000-scoreboard-heldout-2026-05-09/episodes.jsonl`

## Follow-ups

- Improve the learner objective instead of extending plain imitation.
- Keep `track_ball` wins as the main gate. Longer survival against `track_ball`
  is useful pressure, not success.
