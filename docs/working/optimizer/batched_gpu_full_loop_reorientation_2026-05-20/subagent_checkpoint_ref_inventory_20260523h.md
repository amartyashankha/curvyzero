# Checkpoint Ref Inventory - 2026-05-23h

Scope: read-only inventory for the optimizer's profile-only LightZero PyTorch
to JAX shadow-model parity probe. I did not touch live training runs, did not
mutate Modal volumes, and did not edit training code.

## Plain Read

Use a real immutable `iteration_N.pth.tar` checkpoint from `curvyzero-runs-v2`.
Do not use `latest.pth.tar` or `ckpt_best.pth.tar`.

The old candidate below is missing and should stay retired:

```text
training/lightzero-curvytron-visual-survival/curvytron-dense-ckpt1-iter10000-sanity-20260512a/checkpoints/lightzero/iteration_32.pth.tar
```

The best current candidates are the r18fresh champion checkpoints that the
current CZ26 lane reused as its shared initial policy and tournament seed
material. These are not mutable aliases, and Modal volume listing confirms the
files and matching `.metadata.json` sidecars exist.

## Top Candidates

| Priority | Checkpoint ref | Why use it | Immutable? | Current-lane relevance | Verification |
| ---: | --- | --- | --- | --- | --- |
| 1 | `training/lightzero-curvytron-visual-survival/curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/attempts/try-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/train/lightzero_exp/ckpt/iteration_250000.pth.tar` | Current local CZ26 rating artifact ranks this checkpoint #1 active, rating about `1671.82`. It is a mature champion checkpoint from the same r18 champion run. | Yes, `iteration_250000.pth.tar`. | High. It is old-r18 lineage, but current CZ26 rating still ranks it top. | `modal volume ls curvyzero-runs-v2 .../lightzero_exp/ckpt` showed the file and `iteration_250000.pth.tar.metadata.json`. |
| 2 | `training/lightzero-curvytron-visual-survival/curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/attempts/try-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/train/lightzero_exp/ckpt/iteration_180000.pth.tar` | This was the pinned shared seed for the 136-row CZ26 batch and was old r18fresh rank #1 in the preserved trainer/public snapshot. Current local CZ26 rating still has it near the top. | Yes, `iteration_180000.pth.tar`. | Very high for parity because it is the exact seed all CZ26 rows inherited. | Same Modal directory listing showed the file and metadata sidecar. |
| 3 | `training/lightzero-curvytron-visual-survival/curvy-r18fresh-survbonusout-blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1i-2115bef336/attempts/try-r18fresh-survbonusout-blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm-972db4c548/train/lightzero_exp_260516_110032/ckpt/iteration_150000.pth.tar` | Top candidate from a different r18fresh run. Useful as a second shape/lineage check after the champion path passes. Current local CZ26 rating has it as active rank #5. | Yes, `iteration_150000.pth.tar`. | Medium-high. It is still source-state LightZero lineage, but not the CZ26 shared seed. | Modal listing of the timestamped `lightzero_exp_260516_110032/ckpt` dir showed the file and metadata sidecar. |

## Current CZ26-Generated Sanity Candidates

These are useful only if we specifically want a checkpoint produced by the
newer CZ26 batch, not just inherited from its r18 seed. They are lower
confidence as policy-strength candidates because the local rating artifact marks
them `provisional`, but they are valid immutable checkpoint files for model
format/parity.

| Candidate | Checkpoint ref | Note | Verification |
| --- | --- | --- | --- |
| CZ26 nonzero, early high provisional | `training/lightzero-curvytron-visual-survival/cz26a-r017-out67-n20-imm0-b20w05r1/attempts/try-cz26a-r017-out67-n20-imm0-b20w05r1/train/lightzero_exp/ckpt/iteration_10000.pth.tar` | Highest-ranked nonzero CZ26-produced checkpoint found locally: rank #34 provisional in `cz26_rating_latest.json`. | Modal listing showed the file and metadata sidecar. |
| CZ26 nonzero, mature | `training/lightzero-curvytron-visual-survival/cz26a-r043-out100-n0-imm0-b10w05r1/attempts/try-cz26a-r043-out100-n0-imm0-b10w05r1/train/lightzero_exp/ckpt/iteration_290000.pth.tar` | Mature CZ26 checkpoint, rank #58 provisional locally. Useful if the probe should stress a late-training CZ26 artifact. | Modal listing showed the file and metadata sidecar. |
| CZ26 Grid B nonzero, mature | `training/lightzero-curvytron-visual-survival/cz26b-r019-out50-n10-imm0-b25w25r1/attempts/try-cz26b-r019-out50-n10-imm0-b25w25r1/train/lightzero_exp/ckpt/iteration_190000.pth.tar` | Grid B mature checkpoint, rank #65 provisional locally. Useful as a different recipe family. | Modal listing showed the file and metadata sidecar. |

## Sources Checked

- `docs/working/training/r18fresh_postmortem_2026-05-16/NEXT_BATCH_SEEDING.md`
- `docs/working/training/r18fresh_postmortem_2026-05-16/TOP10_RAW_REFS_auto-r000032-g22-555c999b.txt`
- `docs/working/training/r18fresh_postmortem_2026-05-16/CURRENT_LAUNCH_DEFAULTS.md`
- `docs/working/training/training_loop_extension_refactor_2026-05-19/EXPERIMENT_BATCH_INVENTORY.md`
- `docs/working/training/r18fresh_postmortem_2026-05-16/CZ26_BATCH_RATIONALE_REORIENTATION_2026-05-18.md`
- `artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.json`
- `artifacts/local/cz26_analysis_2026-05-18/cz26_rating_latest.json`
- Read-only Modal listings from `curvyzero-runs-v2`.

## Confidence And Blockers

Confidence is high that the top three refs exist and are immutable checkpoint
files. Confidence is also high that they are relevant to the current source-state
fixed-opponent lineage.

The only caution is semantic, not filesystem: the top r18fresh refs are older
champion/seed checkpoints, while the CZ26-generated nonzero refs are newer but
currently provisional. For the parity probe, start with `iteration_250000` or
`iteration_180000`. After that passes, run one CZ26 nonzero candidate to confirm
newer checkpoint artifacts have the same model/load surface.
