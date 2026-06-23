# MCTX Scaling Grid Summary, 2026-05-23e

Superseded-context note, 2026-05-23j: this file describes the older toy-model
MCTX scaling grid. For current headline claims, quote the real checkpoint-backed
MCTX/JAX shadow row instead: B1024/A16/sim8 scalar-off `19,334` active
steps/sec versus direct CTree `8,792`, speedup `2.20x`, profile-only.

Scope: profile-only optimizer rows. These rows do not call `train_muzero`, do
not touch live Coach runs, and are not training recommendations by themselves.

Artifact:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/opt-mctx-scaling-h100-l4-20260523e/
```

## Shape

```text
backend: mctx_compact_search_service_profile_only_v0
model: toy JAX model, not LightZero PyTorch
search: MCTX/Gumbel MuZero, not LightZero CTree
actor_count: 16
steps: 40 measured
warmup: 10
scalar materialization: off
compact slab: on
```

## Results

| compute | batch | sims | steps/sec | slab roots/sec | measured sec | slab sec | search sec | H2D sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | 512 | 8 | `10,170` | `79,439` | `4.028` | `0.516` | `0.134` | `0.263` |
| H100 | 512 | 16 | `12,886` | `81,578` | `3.179` | `0.502` | `0.243` | `0.156` |
| H100 | 512 | 32 | `9,523` | `42,565` | `4.301` | `0.962` | `0.621` | `0.222` |
| H100 | 512 | 64 | `7,111` | `18,586` | `5.760` | `2.204` | `1.668` | `0.351` |
| H100 | 1024 | 8 | `12,946` | `115,820` | `6.328` | `0.707` | `0.140` | `0.404` |
| H100 | 1024 | 16 | `14,803` | `99,251` | `5.534` | `0.825` | `0.264` | `0.408` |
| H100 | 1024 | 32 | `15,639` | `75,578` | `5.238` | `1.084` | `0.628` | `0.296` |
| H100 | 1024 | 64 | `11,724` | `37,275` | `6.987` | `2.198` | `1.613` | `0.423` |
| H100 | 2048 | 8 | `15,612` | `101,673` | `10.494` | `1.611` | `0.160` | `1.158` |
| H100 | 2048 | 16 | `14,694` | `92,798` | `11.150` | `1.766` | `0.305` | `1.146` |
| H100 | 2048 | 32 | `15,174` | `84,159` | `10.797` | `1.947` | `0.677` | `1.002` |
| H100 | 2048 | 64 | `10,171` | `48,067` | `16.109` | `3.409` | `1.724` | `1.359` |
| L4 | 512 | 8 | `12,315` | `73,375` | `3.326` | `0.558` | `0.166` | `0.262` |
| L4 | 512 | 16 | `13,802` | `72,445` | `2.968` | `0.565` | `0.264` | `0.186` |
| L4 | 512 | 32 | `9,818` | `38,244` | `4.172` | `1.071` | `0.636` | `0.285` |
| L4 | 512 | 64 | `8,309` | `21,061` | `4.930` | `1.945` | `1.488` | `0.300` |
| L4 | 1024 | 8 | `8,145` | `66,035` | `10.058` | `1.241` | `0.229` | `0.788` |
| L4 | 1024 | 16 | `16,581` | `93,519` | `4.941` | `0.876` | `0.354` | `0.341` |
| L4 | 1024 | 32 | `13,315` | `57,893` | `6.152` | `1.415` | `0.747` | `0.474` |
| L4 | 1024 | 64 | `13,124` | `36,348` | `6.242` | `2.254` | `1.606` | `0.432` |
| L4 | 2048 | 8 | `9,845` | `61,475` | `16.643` | `2.665` | `0.339` | `1.912` |
| L4 | 2048 | 16 | `18,596` | `96,826` | `8.810` | `1.692` | `0.462` | `0.938` |
| L4 | 2048 | 32 | `9,608` | `49,883` | `17.053` | `3.284` | `1.342` | `1.609` |
| L4 | 2048 | 64 | `15,246` | `49,427` | `10.747` | `3.315` | `1.946` | `1.026` |

## Plain Read

The MCTX search backend is still fast, but the full profile loop is now limited
by more than search.

For example, H100 B1024/sim32:

```text
measured total: 5.238s
compact slab:  1.084s
search:        0.628s
H2D:           0.296s
observation:   2.283s
actor wall:    1.828s
```

That means:

```text
If search became free, this specific row would not become 10x faster.
The next Amdahl wall would be observation/env/handoff.
```

## Current Decision

Keep MCTX/JAX as the leading optimizer lane for the search backend. Do not
promote it directly to Coach.

The next necessary proof is a matched direct-CTree baseline with the same
40-step/10-warmup shape, because the earlier direct rows used nearby but not
identical profile conditions. That matched baseline is running as:

```text
opt-mctx-matched-direct-baseline-h100-l4-20260523e
```

## Matched Direct-CTree Baseline, 40/10 Rows

Artifact:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/opt-mctx-matched-direct-baseline-h100-l4-20260523e/
```

Matched comparison:

| compute | batch | sims | direct CTree steps/sec | MCTX steps/sec | MCTX speedup | direct slab roots/sec | MCTX slab roots/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | 512 | 16 | `5,109` | `12,886` | `2.52x` | `8,540` | `81,578` |
| H100 | 512 | 32 | `3,827` | `9,523` | `2.49x` | `5,567` | `42,565` |
| H100 | 1024 | 16 | `6,261` | `14,803` | `2.36x` | `10,354` | `99,251` |
| H100 | 1024 | 32 | `5,145` | `15,639` | `3.04x` | `7,114` | `75,578` |
| L4 | 512 | 16 | `4,370` | `13,802` | `3.16x` | `5,849` | `72,445` |
| L4 | 512 | 32 | `3,124` | `9,818` | `3.14x` | `4,162` | `38,244` |
| L4 | 1024 | 16 | `4,802` | `16,581` | `3.45x` | `6,303` | `93,519` |
| L4 | 1024 | 32 | `2,399` | `13,315` | `5.55x` | `2,947` | `57,893` |

Plain read:

```text
The matched 40/10 check still supports the MCTX lane. The exact multiplier is
not a training speedup, but the search-backend gap did not disappear when the
direct baseline was rerun under the same row length/warmup.
```

Next strict check:

```text
Run H100-only 80 measured / 20 warmup rows for B512/B1024 and sim16/sim32:
  opt-mctx-strict-h100-8020-20260523f
  opt-direct-strict-h100-8020-20260523f
```

## Strict H100 80/20 Rows

Artifacts:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/opt-mctx-strict-h100-8020-20260523f/
artifacts/local/curvytron_hybrid_observation_profile_results/opt-direct-strict-h100-8020-20260523f/
```

Matched comparison:

| batch | sims | direct CTree steps/sec | MCTX steps/sec | MCTX speedup | direct slab roots/sec | MCTX slab roots/sec | direct search sec | MCTX search sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 16 | `5,864` | `11,826` | `2.02x` | `10,084` | `78,954` | `4.900` | `0.504` |
| 512 | 32 | `4,781` | `8,667` | `1.81x` | `6,967` | `41,416` | `8.259` | `1.218` |
| 1024 | 16 | `4,947` | `11,700` | `2.36x` | `9,067` | `77,350` | `9.665` | `0.632` |
| 1024 | 32 | `4,400` | `13,964` | `3.17x` | `6,050` | `72,311` | `18.604` | `1.280` |

Plain read:

```text
The stricter H100 rows still support the MCTX lane.
The full-profile speedup is not a stable 3x everywhere; call it roughly
`1.8x-3.2x` on this strict denominator.

The search sub-bucket speedup is much larger, but Amdahl now matters:
observation/env/handoff still remain after search gets cheap.
```

After that, choose between:

```text
1. continue toward a real training-compatible compiled search path, or
2. attack the observation/env/handoff wall that appears after MCTX makes search cheaper.
```

The likely answer is both, but only one should be promoted to Coach-facing work
after the matched baseline and validation sidecars report back.
