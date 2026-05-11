# 2026-05-09 dummy pong selection record v2 smoke

## Question

Does the Pong checkpoint selector avoid treating faster losses to `track_ball`
as better when every learned checkpoint has 0 wins?

## Command

```sh
uv run python scripts/select_dummy_pong_checkpoint.py \
  --summary artifacts/local/dummy-pong-lookahead-angle-tie-g32-h32-scoreboard-monitor-2026-05-09/summary.json \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-g32-h32-selection-record-v2-smoke-2026-05-09
```

## Result

The selector still chose `imitation1000`.

The new ranking rule is:

```text
track_ball wins,
then fewer losses to track_ball,
then more truncations against track_ball,
then random_uniform win rate
```

On this summary, every learned checkpoint had 0 wins against `track_ball`, so
the useful pressure order was:

| checkpoint | track_ball losses | track_ball truncations | random wins |
| --- | ---: | ---: | ---: |
| `imitation1000` | 10 | 54 | 44 |
| `lookahead1000` | 32 | 32 | 40 |
| `lookahead250` | 37 | 27 | 41 |
| `lookahead750` | 48 | 16 | 39 |
| `lookahead500` | 52 | 12 | 38 |

## Interpretation

This fixes a misleading tie-break. When all candidates have 0 wins against
`track_ball`, fewer losses and more truncations are a better pressure signal
than lower total truncation rate. This still does not prove quality; it only
makes the selection record harder to fool.

## Artifacts

- `artifacts/local/dummy-pong-lookahead-angle-tie-g32-h32-selection-record-v2-smoke-2026-05-09/selection_record.json`
