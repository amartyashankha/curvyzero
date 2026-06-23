# Curvytron Flash Comparison - 2026-06-23

Status: active working packet. This compares the recovered `curvytron-flash`
Modal deploy against the current CurvyZero implementation and turns the useful
differences into run/profile/documentation decisions.

Fresh Flash reference, H100 raw-env, parity/validate, and PPO profile controls
have been run from this packet. No CurvyZero H100 speed row has been launched
from this packet; CurvyZero speed proof still belongs to the accepted same-work
H100 gate in `goal.md`.

## Plain Read

Verdict: do not port Flash wholesale. Use it as a raw-env/control benchmark
source, a playable export-loop model, and an organization critique of CurvyZero.

The recovered deploy is a compact Modal-first Curvytron bot lab with a playable
server, WebSocket bots, a Node reference environment, a GPU-resident accelerated
environment, and PPO training/export code. It is not a LightZero/MuZero system.

CurvyZero is a larger research workspace with source-state visual LightZero
training, reward/tournament/control-plane plumbing, and a compact-owned
MuZero-speed lane. Its current problem is not merely environment throughput; it
is whole-loop ownership, learning retention, and reliable experiment evidence.

The useful comparison is therefore not "which repo is better." The useful
comparison is:

```text
what Curvytron Flash simplified enough to move fast
what CurvyZero already hardened that Flash lacks
which Flash ideas can become profiling controls or implementation components
which differences are algorithmic and therefore not directly comparable
```

## Read Order

1. `README.md`: verdict and current status.
2. `COMPARISON_MATRIX.md`: decisions, denominators, and organization verdicts.
3. `PROFILE_RUN_PLAN.md`: exact local/Modal/H100 ladder.
4. `SYSTEM_MAPS.md`: compact source map for both systems.
5. `EVIDENCE_LEDGER.md`: recovery artifacts, checksums, and run outputs.

## Source Anchors

- Recovered app:
  `artifacts/modal_deploy_downloads/curvytron-flash-app-20260623-101758/`
- Recovered checkpoint exports:
  `artifacts/modal_deploy_downloads/curvytron-flash-checkpoint-exports-20260623/`
- CurvyZero optimizer north star: `goal.md`
- CurvyZero training refactor:
  `docs/working/training/training_loop_extension_refactor_2026-05-19/`
- CurvyZero reward/H100 campaign:
  `docs/working/training/reward_axis_h100_plan_2026-06-23/`
- CurvyZero optimizer state:
  `docs/working/optimizer/reorientation_2026-05-23/`

## Speed Readout

The speed comparison is intentionally split by denominator:

| Denominator | Fresh row | Speed | Use |
| --- | --- | ---: | --- |
| Flash raw H100 mechanics | Grid, compact obs off | `161.6M env/s` | Ceiling/control only. |
| Flash raw H100 raycast | Grid, `raycast_v1` obs on | `15.65M env/s` | Observation-cost control only. |
| Flash PPO profile | One diagnostic update | `438k agent_steps/s` | Coarse PPO full-profile control, not MuZero proof. |
| CurvyZero accepted baseline | OPT-104 same-work H100 full loop | `12,689 env/s` | Only valid speed-claim baseline. |
| CurvyZero fastest support row found | Compact support row | `15,853 env/s` | Useful support evidence, not repeatable 2x. |

Do not read this as "Flash is N times better." Flash PPO and CurvyZero MuZero do
different work, and Flash raw-env rows omit search/replay/learner ownership.
The real systems lesson is that CurvyZero's speed headroom is probably in
whole-loop ownership and CPU/GPU boundaries, not raw game stepping alone.

## Current Readout

- Flash is a recovered, runnable-looking Modal/PPO/Triton/playable-bot lab.
- CurvyZero is a LightZero/MuZero training system plus compact-owner speed lab.
- Local checks now passed for Flash Python syntax, Flash reference smoke/eval,
  Flash local and Modal reference benchmarks, Flash accelerated validate/parity,
  Flash H100 raw-env controls, Flash PPO diagnostic profile, CurvyZero focused
  reward/config/compact tests, CurvyZero compact-search/phase/Wave-A tests, and
  CurvyZero direct-CTree exact comparison.
- Keep all Flash output under a Flash-control artifact prefix. Do not feed it
  into the CurvyZero optimizer speed ledger.
