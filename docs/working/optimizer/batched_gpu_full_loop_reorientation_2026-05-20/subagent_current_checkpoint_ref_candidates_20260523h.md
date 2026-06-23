# Current LightZero Checkpoint Ref Candidates, 2026-05-23h

Scope: read-only sidecar pass. I did not edit runtime code, touch live
training, delete anything, or run checkpoint parity. I only reviewed local docs
and used read-only `modal volume ls` listings against `curvyzero-runs-v2`.

## Context Read

- `lightzero_jax_shadow_parity_status_20260523g.md` says the fresh-model JAX
  shadow parity gate passed and the checkpoint gate needs a current immutable
  `iteration_N.pth.tar` ref. Mutable refs are rejected.
- `v2_volume_migration_2026-05-14.md`, `NOW.md`, `TODO.md`, and
  `FULL_LOOP_PROOF.md` identify `curvyzero-runs-v2` as the current runs
  Volume. Older non-v2 and `v2refresh18p`/`r18v2` notes are historical.
- The current volume root contains newer optimizer/profile runs from
  2026-05-20 through 2026-05-22 and larger training families from
  2026-05-17. I preferred existing current-volume refs over old archived docs.

## Exact Commands Used

Local doc/context commands:

```bash
pwd
rg -n "LightZero|checkpoint|iteration_[0-9]+\.pth\.tar|curvyzero-runs-v2|parity|shadow-model" docs -S
ls docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20
sed -n '1,220p' docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/lightzero_jax_shadow_parity_status_20260523g.md
sed -n '1,320p' docs/working/training/leaderboard_to_training_2026-05-13/v2_volume_migration_2026-05-14.md
rg -n "curvyzero-runs-v2|iteration_[0-9]+\.pth\.tar|current|latest|checkpoint ref|checkpoint_ref" docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20 docs/working/training -S
rg -n "modal volume ls|modal volume get|modal volume|Volume\.from_name|curvyzero-runs-v2" scripts src docs/working -S
sed -n '1060,1175p' docs/working/training/leaderboard_to_training_2026-05-13/NOW.md
sed -n '1720,1845p' docs/working/training/leaderboard_to_training_2026-05-13/NOW.md
sed -n '270,315p' docs/working/training/leaderboard_to_training_2026-05-13/TODO.md
rg -n "r18bootfix|iteration_240000|iteration_3021|iteration_5300|curvy-r18|bootfix|canary" docs/working/training/leaderboard_to_training_2026-05-13 -S
```

One local `rg` over `artifacts` was accidentally too broad and produced noisy
output; it was read-only and not used for the final candidate selection.

Read-only Modal Volume commands:

```bash
uv run --extra modal modal volume ls curvyzero-runs-v2 training/lightzero-curvytron-visual-survival --json
uv run --extra modal modal volume ls curvyzero-runs-v2 training/lightzero-curvytron-visual-survival/opt-search-hook-stock-c64-sim16-10learn-20260522a/attempts --json
uv run --extra modal modal volume ls curvyzero-runs-v2 training/lightzero-curvytron-visual-survival/opt-search-hook-stock-rndhashfix-c64-sim16-3learn-20260522a/attempts --json
uv run --extra modal modal volume ls curvyzero-runs-v2 training/lightzero-curvytron-visual-survival/rnd-blank-sweep-fastckpt-20260519a-stock-w0-rep00-s20260519/attempts --json
uv run --extra modal modal volume ls curvyzero-runs-v2 training/lightzero-curvytron-visual-survival/cz26a-r001-out0-n0-imm0-b20w05r1/attempts --json
uv run --extra modal modal volume ls curvyzero-runs-v2 training/lightzero-curvytron-visual-survival/opt-search-hook-stock-c64-sim16-10learn-20260522a/attempts/profile/train/lightzero_exp/ckpt --json
uv run --extra modal modal volume ls curvyzero-runs-v2 training/lightzero-curvytron-visual-survival/opt-search-hook-stock-rndhashfix-c64-sim16-3learn-20260522a/attempts/profile/train/lightzero_exp/ckpt --json
uv run --extra modal modal volume ls curvyzero-runs-v2 training/lightzero-curvytron-visual-survival/rnd-blank-sweep-fastckpt-20260519a-stock-w0-rep00-s20260519/attempts/try-rnd-blank-sweep-fastckpt-20260519a-stock-w0-rep00-s20260519/train/lightzero_exp/ckpt --json
uv run --extra modal modal volume ls curvyzero-runs-v2 training/lightzero-curvytron-visual-survival/cz26a-r001-out0-n0-imm0-b20w05r1/attempts/try-cz26a-r001-out0-n0-imm0-b20w05r1/train/lightzero_exp/ckpt --json
```

Compact sorted verification commands:

```bash
uv run --extra modal modal volume ls curvyzero-runs-v2 training/lightzero-curvytron-visual-survival/cz26a-r001-out0-n0-imm0-b20w05r1/attempts/try-cz26a-r001-out0-n0-imm0-b20w05r1/train/lightzero_exp/ckpt --json | jq -r '.[] | select(.Filename | test("iteration_[0-9]+\\.pth\\.tar$")) | .Filename' | sort -V | tail -5
uv run --extra modal modal volume ls curvyzero-runs-v2 training/lightzero-curvytron-visual-survival/rnd-blank-sweep-fastckpt-20260519a-stock-w0-rep00-s20260519/attempts/try-rnd-blank-sweep-fastckpt-20260519a-stock-w0-rep00-s20260519/train/lightzero_exp/ckpt --json | jq -r '.[] | select(.Filename | test("iteration_[0-9]+\\.pth\\.tar$")) | .Filename' | sort -V | tail -5
uv run --extra modal modal volume ls curvyzero-runs-v2 training/lightzero-curvytron-visual-survival/opt-search-hook-stock-rndhashfix-c64-sim16-3learn-20260522a/attempts/profile/train/lightzero_exp/ckpt --json | jq -r '.[] | select(.Filename | test("iteration_[0-9]+\\.pth\\.tar$")) | .Filename' | sort -V | tail -5
```

The first sandboxed attempts at the compact sorted commands hit a local `uv`
cache permission error; the same read-only commands were rerun with approved
escalation.

## Candidate Refs

Recommended first candidate, current-volume larger training row:

```text
training/lightzero-curvytron-visual-survival/cz26a-r001-out0-n0-imm0-b20w05r1/attempts/try-cz26a-r001-out0-n0-imm0-b20w05r1/train/lightzero_exp/ckpt/iteration_260000.pth.tar
```

Why: exists in `curvyzero-runs-v2`, immutable, high numbered checkpoint, created
under the May 17 `cz26a` training family. The listing also showed metadata
sidecars through `iteration_260000`.

Second candidate, newer May 19 stock/RND sweep:

```text
training/lightzero-curvytron-visual-survival/rnd-blank-sweep-fastckpt-20260519a-stock-w0-rep00-s20260519/attempts/try-rnd-blank-sweep-fastckpt-20260519a-stock-w0-rep00-s20260519/train/lightzero_exp/ckpt/iteration_77500.pth.tar
```

Why: exists in `curvyzero-runs-v2`, immutable, current-ish run family, and has a
nonzero training iteration with frequent checkpoint cadence.

Third candidate, most recent but profile-only/iteration-zero:

```text
training/lightzero-curvytron-visual-survival/opt-search-hook-stock-rndhashfix-c64-sim16-3learn-20260522a/attempts/profile/train/lightzero_exp/ckpt/iteration_0.pth.tar
```

Why: exists in `curvyzero-runs-v2` and is the most recent checked ref
(`2026-05-22`). Use it only if the parity gate wants the freshest current model
surface over a learned/nonzero checkpoint, because this checked run exposed only
`iteration_0.pth.tar`.

## Suggested Use

For checkpoint parity, start with the `cz26a` `iteration_260000.pth.tar` ref.
It is the best balance of current v2 storage, real training, immutable naming,
and nonzero/high iteration.
