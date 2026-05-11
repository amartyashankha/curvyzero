# LightZero Later Eval Launches - 2026-05-10

Purpose: record the compact later eval wave launched after third-wave results
showed a small stock-evaluator signal on normal seed `13`.

## Claim

Four normal LightZero Atari Pong eval roots were launched with strict model
loading, no fallback, stock evaluator, 2048-step cap, and quiet stdout. These
target later checkpoints because the old positive runs improved around
`10000` to `18000`, while early `1000` to `3000` rows were mostly flat.

## Non-claim

These launches do not prove learning until their manifests are fetched and
summarized. They are not survival-shaped runs.

## Launched Roots

| Local session | Run | Eval id | Selected checkpoints visible at launch |
| ---: | --- | --- | --- |
| `15752` | `lz-visual-pong-exact-installed-0.2.0-s13-sweep65k-l4` | `sweep65k-s13-compact-0-5k-8k-stock2048-seed13` | `0,5000,8000` |
| `66292` | `lz-visual-pong-exact-installed-0.2.0-s18-sweep65k-h100cpu16` | `sweep65k-s18-compact-0-7k-10k-13k-stock2048-seed18` | `0,7000,10000,13000` |
| `26819` | `lz-visual-pong-exact-installed-0.2.0-s19-sweep65k-h100cpu16` | `sweep65k-s19-compact-0-1k-5k-10k-stock2048-seed19` | `0,1000,5000,10000` |
| `88820` | `lz-visual-pong-exact-installed-0.2.0-s1-repeatB-65536-l4` | `repeatB-s1-compact-0-5k-8k-stock2048-seed1` | `0,5000,8000` |

## Fetch After Completion

Fetch each eval root from the `curvyzero-runs` Modal Volume into:

```text
artifacts/local/lightzero-eval-manifests/<eval-id>
```

Then summarize with:

```text
uv run python scripts/summarize_lightzero_pong_eval_manifest.py --baseline-deltas --format tsv artifacts/local/lightzero-eval-manifests/<eval-id>
```

Report stock steps survived first, then stock return and stock reward counts.
