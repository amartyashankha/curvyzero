# MuZero Library Choice Recheck - 2026-05-09

Scope: short sanity check on whether to rethink LightZero, switch MuZero repos,
or admit we simply did not follow LightZero's setup. Local docs/code only; no
web browsing, no pytest, no training.

## Short Answer

Do not switch away from LightZero in the next 24 hours.

LightZero does contain a full MuZero-family implementation. Our failure so far
is not "LightZero is fake" or "we picked a non-MuZero repo." The sharper truth
is that we have not run an exact full LightZero Pong reproduction yet, and the
off-recipe control rungs that did run are too small and too patched to indict
LightZero as a learner.

Recommended next posture:

1. Keep an exact LightZero reproduction/control lane alive.
2. In parallel, keep the repo-native PPO `[B, P]` architecture probe alive.
3. Do not migrate the immediate MuZero lane to MiniZero, muzero-general, or
   Mctx/owned trainer yet.

That is not a vote of confidence in LightZero as the CurvyTron backbone. It is
a vote to finish the control before replacing it.

## Does LightZero Contain Full MuZero?

Yes.

Evidence already in local docs:

- `docs/working/muzero_implementation_reality_check_2026-05-09.md` records
  that LightZero has model, policy/training, MCTS, collectors, evaluators,
  replay/buffer machinery, configs, examples, logging, and checkpoints.
- The same note records local checkpoints with expected MuZero pieces:
  `representation_network`, `dynamics_network`, `prediction_network`, policy
  head, value head, and reward head.
- Stock LightZero CartPole MuZero and visual Atari Pong smokes called
  LightZero's real `train_muzero`, wrote native `.pth.tar` checkpoints, and
  loaded through LightZero/DI-engine policy surfaces.
- Custom dummy Pong used LightZero's MuZero observation contract:
  `observation`, `action_mask`, and `to_play`.

So the answer is yes: LightZero is a real full MuZero-family framework. The
open question is not existence. The open question is whether its setup and
wrapper semantics are trustworthy enough for our CurvyTron-shaped problem.

## Did We Run Exact Full Replication?

No.

Two exact targets exist, and we have run neither:

- Current GitHub upstream Atari Pong MuZero: about `500000` env steps,
  `50` simulations, `8` collector envs, `8` collect episodes, `3` evaluator
  envs, `3` eval episodes, `batch_size=256`, CUDA, `game_segment_length=400`,
  and stock `update_per_collect=None` / `replay_ratio=0.25`.
- Installed Modal package target, `LightZero==0.2.0`: same broad surface, but
  captured local dry evidence says `max_env_step=200000`.

What has actually run:

- Stock LightZero controls: CartPole MuZero, sparse TicTacToe, and sparse
  Connect4 bot-mode smokes.
- Official Atari Pong mechanics: env/ROM smoke, tiny train smoke,
  checkpoint-load probe, and strict no-fallback eval mechanics.
- Official Atari Pong bounded rungs: CPU/tiny and GPU controls including
  `4096/sim10`, plus the installed-0.2.0 near-upstream `8192/sim25` rung.
  These are infrastructure passes and signal failures, not exact reproduction.
- The `8192/sim25` rung produced many checkpoints and a strict eval curve, but
  periodic checkpoints collapsed to one action and capped return `-6`.
  `ckpt_best` had a manual/stock parity warning, so it is not quality proof.
- Pretrained OpenDILab Pong strict eval has not run successfully because the
  checkpoint/config surface mismatches current `64x64` config assumptions
  versus older `96x96`/downsample weights.
- Custom dummy Pong LightZero runs produced checkpoints, target sidecars,
  independent MCTS scorecards, contact-pressure curriculum smokes, and frozen
  checkpoint opponent plumbing. They still do not show reliable held-out
  improvement.
- Repo-native PPO actor-loop and learner smokes have run as architecture
  probes. They preserve `[T, B, P]` rollout structure and masked actions, but
  they are no-quality smokes.

So yes, we did partially fail to follow LightZero's full setup. More precisely:
we followed the official path mechanically, then changed the scale, collector
shape, evaluator shape, update semantics, segment length, episode caps, batch
size, and checkpoint cadence. That makes the failed learning signal weak
evidence about LightZero itself.

## Alternatives

| Option | Immediate value | Main problem | 24-hour call |
| --- | --- | --- | --- |
| LightZero | Best complete MuZero trainer/control already running in Modal. Has real MuZero machinery, stock Atari path, custom-env path, checkpoints, and MCTS eval hooks. | We have not run exact full upstream/package reproduction; custom dummy Pong wrappers can hide CurvyTron simultaneous semantics; independent eval has not validated learning. | Keep. Run exact/control work or explain setup/eval disagreement before judging it. |
| muzero-general | Readable educational MuZero implementation and useful pseudocode for replay, targets, and game-file shape. | Older/educational posture, Ray surface, limited current control value, no native CurvyTron `[B, P]` fit, no local Modal proof. | Do not switch. Read only. |
| MiniZero | Serious full AlphaZero/MuZero/Gumbel system with Atari support and production-like workers/storage. | New build/runtime stack, new game integration, no local Modal proof, likely more invasive than finishing the LightZero control. | Do not switch now. Consider later as a second external benchmark only. |
| Mctx plus owned trainer | Best clean search substrate if we own MuZero. JAX batched MuZero/Gumbel search is attractive. | Mctx is search only. We would need to write replay, actors, learner, targets, checkpoints, eval, Modal artifacts, and scorecards. | Not immediate. Start only after PPO/env gates prove the task shape. |
| Repo-native PPO baseline | Best way to prove CurvyTron observation/action/reward/reset/scorecard semantics in the native simultaneous `[B, P]` shape. | Not MuZero and not a replacement control for LightZero. Current smoke is no-quality. | Keep in parallel as the architecture probe and environment learnability gate. |

## Next 24 Hours Recommendation

Run two lanes, not a repo switch.

Lane A: exact LightZero reproduction/control.

- Choose one exact target: current GitHub upstream `500000/sim50` or installed
  `LightZero==0.2.0` `200000/sim50`.
- Add or use an exact wrapper that patches only artifact/output paths.
- Preserve stock collector/evaluator counts, batch, search sims, segment
  length, CUDA, `update_per_collect=None`, and replay ratio.
- If full training is too expensive today, run only dry-exact config validation
  and document the cost/timeout requirement. Do not call smaller rungs exact.
- Separately resolve the pretrained checkpoint/config mismatch if a pretrained
  control is desired.

Lane B: repo-native PPO architecture probe.

- Keep the simulator, scorecards, terminal evidence, reset metadata, and
  `[B, P]` rollout shape repo-owned.
- Move from no-quality smoke toward a tiny held-out scorecard/profile.
- Use this to test whether the environment/reward/action contract can support
  learning before investing in owned MuZero.

Do not switch to MiniZero or muzero-general in the next 24 hours. That would
add integration uncertainty before answering the simpler question: can exact
LightZero Pong reproduce, and can our repo-native environment shape learn at
all?

## Critical Bottom Line

LightZero is real. Our exact reproduction is not.

The failed small Pong signals should make us more disciplined, not more
framework-hoppy. Finish the exact LightZero control lane far enough to know
whether setup fidelity explains the failure, while using repo-native PPO to
protect the CurvyTron-specific architecture from being swallowed by an
ego-wrapper framework.

## Local Sources

- `docs/working/training_state_index_2026-05-09.md`
- `docs/working/training_experiment_backlog.md`
- `docs/working/rl_framework_reliability_deep_dive_2026-05-09.md`
- `docs/working/muzero_library_alternatives_2026-05-09.md`
- `docs/working/lightzero_exact_upstream_atari_command_2026-05-09.md`
- `docs/working/muzero_implementation_reality_check_2026-05-09.md`
- `docs/working/lightzero_setup_fidelity_audit_2026-05-09.md`
- `docs/working/training_setup_red_team_2026-05-09.md`
