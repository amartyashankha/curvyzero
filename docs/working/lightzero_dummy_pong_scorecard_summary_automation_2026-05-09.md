# LightZero Dummy Pong Scorecard Summary Automation - 2026-05-09

Goal: make recent dummy Pong scorecard comparisons one command instead of
manual table copying.

No training was run. No pytest was run.

## Script Update

`scripts/summarize_lightzero_pong_scorecards.py` supports:

- `--compact` for decision-table columns:
  source, opponent, checkpoint, raw score, shaped return, survival mean/p90,
  action histogram, normalized action entropy, feature/reset profile, warnings.
- `--baseline-opponents-only` to hide learned-vs-learned rows.
- `--preset recent-dummy-pong` for the recent sparse rung1, UPC25,
  epscollect, contact-pressure modest, and raster smoke scorecard refs.
- `--allow-missing` to skip unavailable refs and print the refs needed for a
  full comparison.

## Modal Access

The expected scorecard summaries were fetched from the `curvyzero-runs` Modal
Volume with `modal volume get`. Only the five preset `summary.json` files were
downloaded.

Local artifact root:

```text
artifacts/local/lightzero-dummy-pong-scorecard-summaries-2026-05-09
```

Fetched files:

```text
training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/attempts/train-2048x32-sparse-h120/eval/mcts-scoreboard-rung1-s10-2x-iter0-iter32-best-s1701/summary.json
training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/attempts/train-2048x32-sim8-upc25-sparse-h120/eval/mcts-scoreboard-upc25-sim8-iter0-iter50-best-e8/summary.json
training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-epscollect/attempts/train-2048x32-sim8-upc25-epscollect-sparse-h120/eval/mcts-scoreboard-upc25-epscollect-sim8-iter0-iter50-best-e8/summary.json
training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/attempts/train-512x8-raster-h120/eval/mcts-scoreboard-iter8-raster-h120-s1701-small/summary.json
training/lightzero-dummy-pong/lz-dpong-20260509T175407Z-77159cc3a6b4/attempts/attempt-20260509T175407Z-8105d62c1e00/eval/mcts-scoreboard-contact-pressure-modest-rung/summary.json
```

The summarizer can read these refs by setting the local artifact root as
`--volume-root`.

## Local Run

Command:

```sh
python scripts/summarize_lightzero_pong_scorecards.py \
  --preset recent-dummy-pong \
  --allow-missing \
  --compact \
  --baseline-opponents-only \
  --volume-root artifacts/local/lightzero-dummy-pong-scorecard-summaries-2026-05-09 \
  --output artifacts/local/lightzero-dummy-pong-scorecard-summaries-2026-05-09/recent-dummy-pong-compact-baseline-table.md
```

Outcome: all five preset summaries loaded. No refs were missing.

Output:

| source | opponent | checkpoint | score | shaped | survival mean/p90 | actions | entropy | feature/reset | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contact-pressure-modest | lagged_track_ball_1 | best | 0.0625 | 0.09814 | 24.62/64 | up=271 stay=123 down=0 | 0.5651 | tabular_ego/contact_pressure:player_0 | zero_down |
| contact-pressure-modest | random_uniform | best | -0.25 | -0.1953 | 15.62/38 | up=176 stay=74 down=0 | 0.5529 | tabular_ego/contact_pressure:player_0 | zero_down,worse_than_random |
| contact-pressure-modest | track_ball | best | -0.875 | -0.7695 | 21.5/45.5 | up=221 stay=123 down=0 | 0.5935 | tabular_ego/contact_pressure:player_0 | zero_down,no_wins,no_survival_gain_if_comparable |
| contact-pressure-modest | lagged_track_ball_1 | iter0 | -0.375 | -0.3433 | 15.88/45 | up=170 stay=84 down=0 | 0.5777 | tabular_ego/contact_pressure:player_0 | zero_down |
| contact-pressure-modest | random_uniform | iter0 | 0.125 | 0.1616 | 14.56/26.5 | up=147 stay=86 down=0 | 0.5994 | tabular_ego/contact_pressure:player_0 | zero_down |
| contact-pressure-modest | track_ball | iter0 | -0.75 | -0.6489 | 28.94/64 | up=292 stay=171 down=0 | 0.5995 | tabular_ego/contact_pressure:player_0 | zero_down,no_wins |
| contact-pressure-modest | lagged_track_ball_1 | iter3 | -0.625 | -0.5879 | 21.75/64 | up=235 stay=113 down=0 | 0.5738 | tabular_ego/contact_pressure:player_0 | zero_down |
| contact-pressure-modest | random_uniform | iter3 | -0.125 | -0.06201 | 16.25/27 | up=163 stay=97 down=0 | 0.6013 | tabular_ego/contact_pressure:player_0 | zero_down,worse_than_random |
| contact-pressure-modest | track_ball | iter3 | -0.625 | -0.5493 | 33.69/64 | up=361 stay=178 down=0 | 0.5774 | tabular_ego/contact_pressure:player_0 | zero_down,no_wins |
| epscollect | lagged_track_ball_1 | best-epscollect | -0.25 | -0.2177 | 12.12/19 | up=194 stay=0 down=0 | 0 | tabular_ego/default | zero_down,single_action_gt_95pct |
| epscollect | random_uniform | best-epscollect | 0 | 0.01953 | 9.375/13.5 | up=150 stay=0 down=0 | 0 | tabular_ego/default | zero_down,single_action_gt_95pct |
| epscollect | track_ball | best-epscollect | -0.875 | -0.8172 | 28.88/80.5 | up=462 stay=0 down=0 | 0 | tabular_ego/default | zero_down,single_action_gt_95pct,no_wins |
| epscollect | lagged_track_ball_1 | iter0-epscollect | 0.25 | 0.2776 | 31.62/80.5 | up=170 stay=336 down=0 | 0.581 | tabular_ego/default | zero_down |
| epscollect | random_uniform | iter0-epscollect | -0.375 | -0.3406 | 11.44/19 | up=68 stay=115 down=0 | 0.6006 | tabular_ego/default | zero_down,worse_than_random |
| epscollect | track_ball | iter0-epscollect | -0.875 | -0.8086 | 30.94/80.5 | up=175 stay=320 down=0 | 0.5913 | tabular_ego/default | zero_down,no_wins |
| epscollect | lagged_track_ball_1 | iter50-epscollect | -0.25 | -0.2198 | 26.12/69.5 | up=400 stay=0 down=18 | 0.1616 | tabular_ego/default | single_action_gt_95pct |
| epscollect | random_uniform | iter50-epscollect | -0.25 | -0.2148 | 12.12/19 | up=185 stay=0 down=9 | 0.1709 | tabular_ego/default | single_action_gt_95pct,worse_than_random |
| epscollect | track_ball | iter50-epscollect | -0.8125 | -0.751 | 37.25/120 | up=573 stay=0 down=23 | 0.1488 | tabular_ego/default | single_action_gt_95pct,no_wins |
| raster-smoke | lagged_track_ball_1 | iter8-raster-h120 | 0.25 | 0.2583 | 37.38/120 | up=181 stay=118 down=0 | 0.6106 | raster_flat/default | zero_down |
| raster-smoke | random_uniform | iter8-raster-h120 | -0.75 | -0.7036 | 12.12/19 | up=87 stay=10 down=0 | 0.302 | raster_flat/default | zero_down,worse_than_random |
| raster-smoke | track_ball | iter8-raster-h120 | -1 | -0.9094 | 21.75/30 | up=156 stay=18 down=0 | 0.3027 | raster_flat/default | zero_down,no_wins,no_survival_gain_if_comparable |
| sparse-rung1 | lagged_track_ball_1 | best-s10-2x | -0.25 | -0.2177 | 12.12/19 | up=194 stay=0 down=0 | 0 | tabular_ego/default | zero_down,single_action_gt_95pct |
| sparse-rung1 | random_uniform | best-s10-2x | 0 | 0.01953 | 9.375/13.5 | up=150 stay=0 down=0 | 0 | tabular_ego/default | zero_down,single_action_gt_95pct |
| sparse-rung1 | track_ball | best-s10-2x | -0.875 | -0.8172 | 28.88/80.5 | up=462 stay=0 down=0 | 0 | tabular_ego/default | zero_down,single_action_gt_95pct,no_wins |
| sparse-rung1 | lagged_track_ball_1 | iter0-s10-2x | 0.125 | 0.1432 | 25.44/69.5 | up=365 stay=42 down=0 | 0.3022 | tabular_ego/default | zero_down |
| sparse-rung1 | random_uniform | iter0-s10-2x | -0.375 | -0.3406 | 11.44/19 | up=159 stay=24 down=0 | 0.3537 | tabular_ego/default | zero_down,worse_than_random |
| sparse-rung1 | track_ball | iter0-s10-2x | -0.875 | -0.8029 | 32.31/80.5 | up=459 stay=58 down=0 | 0.3195 | tabular_ego/default | zero_down,no_wins |
| sparse-rung1 | lagged_track_ball_1 | iter32-s10-2x | -0.25 | -0.2198 | 26.12/69.5 | up=418 stay=0 down=0 | 0 | tabular_ego/default | zero_down,single_action_gt_95pct |
| sparse-rung1 | random_uniform | iter32-s10-2x | -0.25 | -0.2148 | 12.12/19 | up=194 stay=0 down=0 | 0 | tabular_ego/default | zero_down,single_action_gt_95pct,worse_than_random |
| sparse-rung1 | track_ball | iter32-s10-2x | -0.8125 | -0.751 | 37.25/120 | up=596 stay=0 down=0 | 0 | tabular_ego/default | zero_down,single_action_gt_95pct,no_wins |
| upc25 | lagged_track_ball_1 | best-upc25 | -0.25 | -0.2177 | 12.12/19 | up=194 stay=0 down=0 | 0 | tabular_ego/default | zero_down,single_action_gt_95pct |
| upc25 | random_uniform | best-upc25 | 0 | 0.01953 | 9.375/13.5 | up=150 stay=0 down=0 | 0 | tabular_ego/default | zero_down,single_action_gt_95pct |
| upc25 | track_ball | best-upc25 | -0.875 | -0.8172 | 28.88/80.5 | up=462 stay=0 down=0 | 0 | tabular_ego/default | zero_down,single_action_gt_95pct,no_wins |
| upc25 | lagged_track_ball_1 | iter0-upc25 | 0 | 0.02031 | 24.75/69.5 | up=200 stay=196 down=0 | 0.6309 | tabular_ego/default | zero_down |
| upc25 | random_uniform | iter0-upc25 | -0.25 | -0.212 | 13.5/19 | up=104 stay=112 down=0 | 0.6303 | tabular_ego/default | zero_down,worse_than_random |
| upc25 | track_ball | iter0-upc25 | -0.875 | -0.8029 | 32.31/80.5 | up=250 stay=267 down=0 | 0.6304 | tabular_ego/default | zero_down,no_wins |
| upc25 | lagged_track_ball_1 | iter50-upc25 | -0.25 | -0.2198 | 26.12/69.5 | up=389 stay=0 down=29 | 0.2294 | tabular_ego/default |  |
| upc25 | random_uniform | iter50-upc25 | -0.25 | -0.2148 | 12.12/19 | up=175 stay=0 down=19 | 0.2918 | tabular_ego/default | worse_than_random |
| upc25 | track_ball | iter50-upc25 | -0.8125 | -0.7453 | 38.62/120 | up=592 stay=0 down=26 | 0.1588 | tabular_ego/default | single_action_gt_95pct,no_wins |

## Read

The access blocker is cleared for the preset. The summary files are now local
in a clean mirror of the Modal Volume paths, and the saved compact table is:

```text
artifacts/local/lightzero-dummy-pong-scorecard-summaries-2026-05-09/recent-dummy-pong-compact-baseline-table.md
```

The comparison still shows action collapse. Most default-reset learned rows have
no `down` actions. The later sparse, UPC25, and epscollect checkpoints mostly
collapse to `up`, with `single_action_gt_95pct` on the best/final rows. Raster
smoke and contact-pressure modest keep some `stay`, but still have `down=0`.
