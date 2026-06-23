# Compact Slab H100 Profile Summary, 2026-05-23d

Scope: profile-only optimizer rows. These rows do not call `train_muzero`, do
not touch live Coach runs, and are not Coach training recommendations by
themselves.

## What Changed

The slab path now runs this loop:

```text
compact batch
-> CompactRootBatchV1
-> CompactSearchServiceV1
-> selected joint action
-> next env step
-> compact replay-index rows
```

The first summary denominator was wrong. It divided all roots by the last
search call. The fixed denominator is aggregate `compact_rollout_slab_sec`.
New rows also preserve `compact_rollout_slab_telemetry_totals`.

## Real Search Grid

H100, actor count 16, root noise off, scalar rows off, slab on.

| impl | batch | sims | steps/sec | slab roots/sec |
| --- | ---: | ---: | ---: | ---: |
| `direct_ctree_gpu_latent` | 256 | 8 | `4657` | `8909` |
| `direct_ctree_gpu_latent` | 256 | 16 | `3894` | `6201` |
| `direct_ctree_gpu_latent` | 256 | 32 | `2910` | `3815` |
| `direct_ctree_gpu_latent` | 512 | 8 | `7130` | `11547` |
| `direct_ctree_gpu_latent` | 512 | 16 | `6522` | `9538` |
| `direct_ctree_gpu_latent` | 512 | 32 | `4177` | `5358` |
| `direct_ctree_gpu_latent` | 1024 | 8 | `8291` | `12937` |
| `direct_ctree_gpu_latent` | 1024 | 16 | `6992` | `9668` |
| `direct_ctree_gpu_latent` | 1024 | 32 | `5314` | `6542` |

Direct arrays controls:

| impl | batch | sims | steps/sec | slab roots/sec |
| --- | ---: | ---: | ---: | ---: |
| `direct_ctree_arrays` | 512 | 8 | `4573` | `6028` |
| `direct_ctree_arrays` | 512 | 16 | `3437` | `4134` |
| `direct_ctree_arrays` | 512 | 32 | `1689` | `1852` |
| `direct_ctree_arrays` | 1024 | 8 | `4192` | `4904` |
| `direct_ctree_arrays` | 1024 | 16 | `2613` | `2902` |
| `direct_ctree_arrays` | 1024 | 32 | `947` | `981` |

Plain read:

```text
direct_ctree_gpu_latent is the correct real-search baseline.
It beats direct_ctree_arrays everywhere, especially at larger batch/simulation.
```

## Ceilings

| mode | batch | sims | steps/sec | slab roots/sec |
| --- | ---: | ---: | ---: | ---: |
| `service_tax_probe` | 512 | 16 | `10312` | `22873` |
| `service_tax_probe` | 512 | 32 | `9260` | `16662` |
| `service_tax_probe` | 1024 | 16 | `12072` | `23741` |
| `service_tax_probe` | 1024 | 32 | `10614` | `18400` |
| `mock_search_service` | 512 | 16 | `11071` | `44771` |
| `mock_search_service` | 512 | 32 | `14460` | `42899` |
| `mock_search_service` | 1024 | 16 | `15590` | `49820` |
| `mock_search_service` | 1024 | 32 | `15104` | `39098` |

Ratios versus `direct_ctree_gpu_latent`:

| shape | service-tax | mock |
| --- | ---: | ---: |
| B512/sim16 | `1.58x` | `1.70x` |
| B512/sim32 | `2.22x` | `3.46x` |
| B1024/sim16 | `1.73x` | `2.23x` |
| B1024/sim32 | `2.00x` | `2.84x` |

Plain read:

```text
There is real headroom, but it is not 10x in this current slab shape.
The next speed win must reduce/replace the search body and its CPU/GPU control
loop, not just move fields around the current LightZero CTree call.
```

## Actor Count

B512/sim16/direct GPU-latent:

| actors | steps/sec | slab roots/sec |
| ---: | ---: | ---: |
| 8 | `5986` | `8187` |
| 16 | `6522` | `9538` |
| 32 | `5644` | `8896` |

Plain read: actor count 16 remains the best default for this denominator.

## Split Telemetry Sanity

Patched-code B512/A16/sim16 direct GPU-latent, 20 measured steps:

| bucket | sec |
| --- | ---: |
| measured total | `2.948` |
| slab total | `2.076` |
| slab search | `1.288` |
| slab model | `0.427` |
| slab H2D | `0.362` |

Patched-code B512/A16/sim16 service-tax:

| bucket | sec |
| --- | ---: |
| measured total | `1.591` |
| slab total | `0.838` |
| slab search | `0.047` |
| slab model | `0.406` |
| slab H2D | `0.332` |

Plain read:

```text
For real direct GPU-latent search, search is the biggest measured slab bucket.
Model and H2D are also meaningful. Even if search became almost free, total
speed would not become infinite because env/observation/handoff still remain.
```

## Precomputed Recurrent Falsifier

This is not a valid training backend. It replaces recurrent model calls with
synthetic resident outputs, so it only prices one possible bucket.

| mode | batch | sims | steps/sec | slab roots/sec |
| --- | ---: | ---: | ---: | ---: |
| `direct_ctree_gpu_latent_precomputed_recurrent` | 512 | 16 | `7086` | `11526` |
| `direct_ctree_gpu_latent_precomputed_recurrent` | 512 | 32 | `5967` | `8200` |
| `direct_ctree_gpu_latent_precomputed_recurrent` | 1024 | 16 | `8875` | `12571` |
| `direct_ctree_gpu_latent_precomputed_recurrent` | 1024 | 32 | `8055` | `10279` |

Ratios versus real `direct_ctree_gpu_latent`:

| shape | ratio |
| --- | ---: |
| B512/sim16 | `1.09x` |
| B512/sim32 | `1.43x` |
| B1024/sim16 | `1.27x` |
| B1024/sim32 | `1.52x` |

Plain read:

```text
Deleting recurrent model calls helps, especially at sim32, but it does not
erase the wall. The remaining CTree/search/control/list path is still large.
This supports a stronger search-service backend, not a narrow recurrent-only
patch.
```

## Current Recommendation

- For profile-only real search, use H100, B1024/A16/sim16 if we want a
  balanced stronger-search row, or B1024/A16/sim8 if we want max throughput.
- Do not use `direct_ctree_arrays` except as a regression/control row.
- Do not claim a Coach training speedup from this yet.
- Next optimizer lane: build or test a stronger backend behind
  `CompactSearchServiceV1` that attacks the CTree/search control loop. The
  clean falsifier is B512/B1024 with sim16/sim32 against the rows above.
