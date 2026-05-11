# Pong Replication Follow-up Queue - 2026-05-11

Purpose: short prioritized disposition list for historical stock or near-stock
Pong runs that should have worked. Survival versus the same run's
`iteration_0` is the primary readout; score is secondary until survival moves.

Source audit:
`docs/working/training/pong_replication_failure_audit_2026-05-11.md`.

## Priority Queue

| priority | run / lane | disposition | why | next action |
| --- | --- | --- | --- | --- |
| 1 | May11 installed stock64 H100 repeat surface: positive `s122`, repeat `s142` | keep monitoring/eval | `s122` is the cleanest May11 proof that the installed 64x64 stock surface can move: mean survival `761.5 -> 934.5 -> 988.562 -> 1295.06` by `12k`, with score still weak. The most valuable question is repeatability, not another new variant. | Evaluate `s142` on the same stock-only survival panel around `7k/10k/12k`, then `20k/50k` if still running. Compare directly to `s122`; do not relaunch another duplicate before this read lands. |
| 2 | May11 installed stock64 flat comparisons: `s114`, `s120`, `s121` | keep monitoring/eval | These are real unresolved flat/failed seeds because they share the same stock64 surface as positive `s122`: `s114` only had a small late bump, `s120` stayed flat, and `s121` worsened versus its own start. That makes them useful variance evidence, not discardable noise. | If later checkpoints exist, evaluate `20k` and `50k` stock-only survival versus each run's `iteration_0`. After that, archive as negative/weak seeds unless one recovers. |
| 3 | current GitHub upstream MuZero segment lane: `s1-wait`, `s2-long500k` | keep monitoring/eval | This is the clean current-upstream route after the plain config hit action-map drift. The segment path with `ALE/Pong-v5` is alive/evaluable, but it does not yet have a serious `2048` cap, `50` simulation stock survival result. | Wait for visible `5k+` checkpoints, then run strict stock-only survival eval at `0/5k/6k/7k` or latest. Relaunch only if the Volume root/progress truly stalls, not because no final score exists yet. |
| 4 | Agent96 model-card lane: `s125`, `s127`, `s128` | keep monitoring/eval, separate bucket | It is a near-stock visual Pong surface, but not comparable to stock64: `s127` saturated the `512`-step eval cap at every checkpoint and only nudged return from `-13` to `-12.25`. The current panel cannot discriminate learning. | Re-evaluate later checkpoints with a longer cap, preferably `2048`, and keep all Agent96 claims separate from installed stock64 claims. Do not use the HF pretrained key mismatch as a policy verdict. |
| 5 | Wave11 normal proof lane: `s71`, `s73`, `s74`, `s76` | archive as positive controls | These already answer the basic viability question: multiple normal runs show late stock-only survival gains, with `s74` reaching cap and `s73`/`s76` strong. More fresh eval here is lower value than resolving May11 repeatability and flat seeds. | Keep their manifests as reference positive controls. Only revisit if a later summary needs a robustness panel or direct comparison to a new stock64 run. |
| 6 | seed `1` normal L4/T4 65k | archive as positive control | The strict rerun reached `2048/2048` stock steps at `iteration_18459`, stock return `-11`, and `8` positive rewards. It proves the lane can learn, while also showing eval sensitivity. | No relaunch. Cite as historical proof of possibility; use cross-seed May11/Wave11 results for stability instead. |
| 7 | mixed normal 65k seeds: seed `1` H100, seed `2` L4/T4, seed `3` L4/T4 | archive as variance evidence | These had transient or weak bumps and fallback. They are useful for the story that checkpoint and seed variance are real, but they are not the next best use of eval time. | No fresh eval unless exact later checkpoints are already present and a summary needs a curve gap filled. |
| 8 | older installed stock64 early sweeps and repeats: `s10`-`s19`, `s24`-`s27`, repeatB, repeat A seed `0` | archive as undertrained/early historical controls | Early `1k/5k` flat reads were not fair final verdicts, and some rows had small survival bumps (`s19`, `s27`, repeat A latest `+212`). They have been superseded by longer Wave11 and May11 stock64 curves. | No relaunch. Do not cite early flat rows as failure proof. Only use them to document that short-horizon sparse-reward Pong can look flat before later survival movement. |
| 9 | spawned/no-root May11 rows: `s111`, `s112`, `s130`-`s133` | relaunch only after one final poll | These are launch/visibility cases, not policy results. The audit says names were absent or Volume roots were not visible at the check points. | Poll Volume once. If still absent, archive the missing rows as no-evidence launch failures and relaunch only through the detached/wait pattern that has already produced checkpoints. |
| 10 | exact stock controls with slow cadence: `s113`, `s123` | keep monitoring/eval, low priority | They are closer to exact stock settings, but their slow checkpoint cadence makes them less immediately useful than `s122/s142` and the flat comparisons with `1k` cadence. | Check when later checkpoints exist. Run stock-only survival eval at meaningful later points rather than spending effort on `iteration_0`-only inventory. |

## Ignore Or Archive Out Of The Pong Proof Queue

| lane | disposition | why |
| --- | --- | --- |
| survival-shaped Pong: `s30`-`s37`, `s60`-`s61`, `s80`-`s82` | ignore for stock proof; archive as side telemetry | Reward shaping changes the claim. Useful for intuition, not for stock-reward Pong replication. |
| LightZero board controls: `s200`-`s203` | archive | TicTacToe/Connect4 controls prove framework plumbing, not visual Atari Pong. |
| custom dummy Pong | ignore | It was a custom/debug wrapper bridge by design, not stock Atari Pong. |
| custom CurvyTron accumulated replay / two-seat loop | archive as custom diagnostic | It mechanically ran but stayed weak/flat and bypassed native `train_muzero`, collector, replay, and learner lifecycle. |
| non-LightZero controls: OpenSpiel, MCTX, MiniZero | ignore for this queue | External controls can help broader confidence, but they do not decide whether LightZero stock visual Pong learned. |

## Operating Rule

Do not launch more near-duplicate stock64 Pong runs until the `s142` repeat and
the later `s114/s120/s121` flat-seed checks are read. The queue should spend
eval time on direct repeatability and same-surface negatives before adding more
surface variants.
