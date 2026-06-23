# Evidence Ledger

Status: active ledger. Facts here should be backed by concrete files, command
outputs, or Modal state. Avoid claims without source anchors.

## Recovered Modal State

| Field | Value | Evidence |
| --- | --- | --- |
| Modal environment | `david.wang-dev` | `modal environment list --json` showed environment present. |
| Live app name | `curvytron-flash` | `modal app list --env david.wang-dev --json`. |
| Live app id | `ap-2W6gvP1phQjSJ03aoO9OIV` | `modal app list --env david.wang-dev --json`. |
| Function id | `fu-jwHEbczt73Q800EUoRBVYB` | `modal app logs ap-2W6gvP1phQjSJ03aoO9OIV --env david.wang-dev ...`. |
| Container id | `ta-01KVTQVRHA9047VJ8PSMZ230PR` | `modal container list --env david.wang-dev`. |
| Container source root | `/app` | `MODAL_ENVIRONMENT=david.wang-dev modal container exec ... pwd`. |
| Mounted checkpoint path | `/checkpoints` | container `ls -la /` showed symlink to Modal volume. |

## Local Recovery Artifacts

| Artifact | Notes |
| --- | --- |
| `artifacts/modal_deploy_downloads/curvytron-flash-app-20260623-101758/` | Extracted source tree from `/app`. |
| `artifacts/modal_deploy_downloads/curvytron-flash-app-20260623-101758.tar.gz` | Compressed source tree recovered via base64 transport. |
| `artifacts/modal_deploy_downloads/curvytron-flash-app-20260623-101758.tar.gz.b64` | ASCII transport artifact used because binary stdout hit a Modal CLI decode bug. |
| `artifacts/modal_deploy_downloads/curvytron-flash-app-20260623-101758.files.txt` | `find /app -type f` manifest from container. |
| `artifacts/modal_deploy_downloads/curvytron-flash-checkpoint-exports-20260623/` | Extracted `/checkpoints/exports` bundle. |
| `artifacts/modal_deploy_downloads/curvytron-flash-checkpoint-exports-20260623.tar.gz` | Compressed checkpoint/export bundle recovered via base64 transport. |
| `artifacts/modal_deploy_downloads/curvytron-flash-checkpoint-exports-20260623.tar.gz.b64` | ASCII transport artifact for checkpoint/export bundle. |

## Checksums

| Artifact | SHA-256 |
| --- | --- |
| App tarball | `90e731425394036eb1ff38d07b83f3bbe758a4905f43f1edfaa9b4f6b802111d` |
| App base64 transport | `c582d17c54625e528246ccbe214fc9b5b410205f94bab380825efa7e89832420` |
| App file manifest | `7753f887eb1fd3465753fe6ef4e795aa6fe56b85ebb73fe4ec386aeae4f1727d` |
| Checkpoint exports tarball | `ac40eef104eabfca5fd30549faf2a238a480caccbb7833197cb2f6c8e0f03407` |
| Checkpoint exports base64 transport | `49635e0f6fd65059cb8e226af8f9986da495044368c86df747ba97e5fea1270f` |

## Immediate Recovery Facts

- Recovered app tree size after extraction: `23M`.
- Recovered app file manifest count: `2742`.
- Recovered `/checkpoints/exports` files include `latest.json`,
  `ppo-raycast-v1-gridreach-10h-latest.json`, `ppo-raycast-v1-shaped-latest.json`,
  and smoke eval/export JSONs.
- GitHub CLI searches for distinctive names such as `modal_curvytron_flash.py`,
  `CurvytronFlash`, and `ppo-raycast-v1-gridreach` did not find a matching repo
  during initial recovery.
- Recovered `flash_app/README.md` states Modal-only execution and lists `curvytron/`, `bots/`,
  `reference/`, `accelerated/`, and `training/` as top-level concerns
  (`artifacts/modal_deploy_downloads/curvytron-flash-app-20260623-101758/README.md:5-22`).
- Flash playable app starts a Node server, launches configured bots, and forwards
  port 8080 through Modal Flash in `us-west`
  (`artifacts/modal_deploy_downloads/curvytron-flash-app-20260623-101758/modal_curvytron_flash.py:60-80`).
- Flash training uses CUDA-only PPO over `AcceleratedCurvytronEnv` with
  `observation_mode="raycast_v1"`
  (`artifacts/modal_deploy_downloads/curvytron-flash-app-20260623-101758/training/ppo.py:94-129`).
- Flash DDP reserves `H100:8` and README records a current four-update no-sync
  baseline of about `11.85M` agent steps/sec
  (`artifacts/modal_deploy_downloads/curvytron-flash-app-20260623-101758/training/README.md:97-127`).
  This is a recovered README claim, not a packet-run H100 artifact. Require a
  DDP summary plus rank logs before using it as evidence; recovered defaults can
  produce one update unless `total_agent_steps` is high enough.
- CurvyZero `goal.md` says only same-work H100 full-loop rows can prove speed;
  raw rows and local proof are decision inputs (`goal.md:10-40`).

## Checks Run From This Packet

| Check | Result |
| --- | --- |
| Flash recovered Python syntax check | Passed locally with `python -m py_compile` over accelerated/training/modal files; terminal output not persisted. |
| Flash local reference smoke | Passed; `artifacts/local/flash_controls/reference/reference_smoke_20260623.txt`. |
| Flash local reference eval | Passed 34 scenarios; `artifacts/local/flash_controls/reference/reference_eval_20260623.txt`. |
| Flash local reference benchmark | Passed; positive throughput and hashes in `artifacts/local/flash_controls/reference/reference_benchmark_20260623.json`. |
| Flash Modal reference smoke/eval | Passed; smoke output matched local hash, eval passed 34 scenarios; `artifacts/local/flash_controls/reference/modal_reference_eval_20260623.txt`. |
| Flash Modal reference benchmark | Passed; `artifacts/local/flash_controls/reference/modal_reference_benchmark_20260623.txt`. |
| Flash accelerated validate | Passed; `artifacts/local/flash_controls/raw_env/h100_accelerated_validate_20260623.txt`. |
| Flash accelerated semantic parity | Passed: 54 scenarios, 106 segments, 874 checks; exact final-hash checks unsupported by the parity harness; `artifacts/local/flash_controls/raw_env/h100_accelerated_parity_20260623.txt`. |
| Flash raw H100 grid benchmark | Passed with `grid_overflow=0`; `161.55M env/s`, `323.10M agent_steps/s`; `artifacts/local/flash_controls/raw_env/h100_grid_baseline_20260623.txt`. |
| Flash raw H100 grid CUDA graph benchmark | Passed with `grid_overflow=0`; `152.01M env/s`, `304.02M agent_steps/s`; `artifacts/local/flash_controls/raw_env/h100_grid_cuda_graph_20260623.txt`. |
| Flash raw H100 compact-observation benchmark | Passed with `grid_overflow=0`; `145.30M env/s`, `290.60M agent_steps/s`; `artifacts/local/flash_controls/raw_env/h100_grid_compact_obs_20260623.txt`. |
| Flash raw H100 raycast benchmark | Passed with `grid_overflow=0`; `15.65M env/s`, `31.30M agent_steps/s`; `artifacts/local/flash_controls/raw_env/h100_grid_raycast_v1_20260623.txt`. |
| Flash PPO diagnostic profile | Passed: `438386.74 agent_steps/s`, `524288` agent steps, one update; top buckets reset `40.96%`, reward `14.70%`, PPO update `14.60%`; `artifacts/local/flash_controls/ppo_profile/h100_ppo_profile_diagnostic_20260623.txt`. |
| CurvyZero focused reward/config/compact tests | Passed: 59 tests, 2 warnings; terminal output not persisted. |
| CurvyZero compact search/profiler/Wave-A slice | Passed: 71 tests, 2 warnings; terminal output not persisted. |
| CurvyZero direct-CTree stock comparison | Passed exact: action agreement `1.0`, max diffs `0.0`, illegal actions `0.0`; `artifacts/local/curvyzero_comparison_controls/direct_ctree_stock_compare_20260623.json`. |
| Flash control manifest | Created denominator manifest for fresh Flash rows; `artifacts/local/flash_controls/manifest_20260623.json`. |

## Remaining Evidence

- A fresh Flash DDP row only if we need to test the recovered `H100:8` headline.
- CurvyZero H100 row only through an existing accepted optimizer or reward gate.
- Policy-quality evidence requires eval/tournament artifacts; checkpoint/export
  existence is not enough.
