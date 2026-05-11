# Framework Reassessment For CurvyTron MuZero

Date: 2026-05-11

Status: working worldview, not final decision. This document records what the
optimizer currently believes, what evidence supports it, and what is still
uncertain.

Related full-loop component map:
`docs/working/optimizer/full_training_loop_worldview_2026-05-11.md`.

## Problem Shape

CurvyTron target:

- visual input
- two players first, possible multiplayer later
- simultaneous actions at each physical tick
- delayed/survival reward
- self-play or checkpoint-lagged self-play
- high-throughput searched experience generation

The important correction is that fast AlphaZero/MuZero systems are not just
local learner loops. They are searched self-play data factories: many actors,
batched neural inference, replay, learner, checkpoint publishing, and eval.

Plain English: the hard part is not only "make one trainer faster." The hard
part is "produce lots of searched game data without starving the GPU or the
learner."

## Simultaneous Action Modeling

Current best working stance: simultaneous action is not the main speed
bottleneck. It is mostly a representation and target-semantics problem. A
framework can collect player actions one at a time and advance the physical
environment only after all players have committed actions. That is the right
implementation pattern, not a hack. The hard question is what policy/search is
allowed to know while those pending actions exist.

| Option | Keep real simultaneous env step? | Main issue | Current use |
| --- | --- | --- | --- |
| Sequential commit, no advance | Yes, if world advances only after all commits | Must decide whether pending earlier actions are hidden from later players/search | Viable engineering representation; needs care |
| Centralized joint action | Yes | For 2P action space is `3*3=9`, but this trains a centralized controller, not independent self-play; scales as `3^P` | Small 2P ablation |
| Independent per-seat search from same state, then joint step | Yes | Not a full game-theoretic simultaneous equilibrium, but preserves the physical game and is computationally sane | Best near-term 2P self-play candidate |
| Focal player search vs fixed/frozen/checkpoint opponent | Yes | Not true current-policy simultaneous self-play | Good control/ladder path |
| Full simultaneous-game search | Yes | More complex: matrix-game, regret, information-set, or simultaneous MCTS variants | Research later |

Sequential commit can be equivalent to simultaneous play for data collection if:

- every player chooses from the same physical tick state;
- the world does not move until all committed actions exist;
- later players' observations do not reveal earlier pending actions unless the
  original game would reveal them;
- replay and reward targets are recorded at physical ticks, or artificial
  half-tick records are clearly labeled.

Sequential commit becomes a different game if pending actions are exposed to
later players. It also becomes awkward for standard perfect-information
AlphaZero/MuZero search if the copied/search state contains hidden pending
actions that the current player should not know. That is an algorithmic
semantics issue, not a reason to ignore the representation.

Canaries before trusting any sequential-commit wrapper:

- order-invariance: collecting actions in player order `0,1` versus `1,0` from
  the same pre-tick snapshot must not change learned observations, masks,
  transition, rewards, or action distributions except for explicitly isolated
  RNG;
- leak canary: a toy simultaneous game where seeing player 0's pending action
  lets player 1 always win must not show that advantage under the hidden-commit
  wrapper;
- replay audit: no pending action, synthetic phase, or first-mover diagnostic
  may enter learned observations or learner batches, only post-commit sidecars.

Useful literature pointer: Tron has appeared in simultaneous-move MCTS research,
with decoupled variants compared against sequential and matrix-game variants.
That supports treating independent per-seat search as a practical first
approximation, not a final theory claim.

## Framework Fit

| Framework | What It Gives | Main CurvyTron Gap | Current Read |
| --- | --- | --- | --- |
| LightZero | PyTorch MuZero/EfficientZero/Gumbel variants, C++ tree search, Atari configs, stock replay/learner, vector env collection | No ready distributed actor fleet; stock collector assumes one action per env row; simultaneous `[B,P]` needs custom collector or env-owned opponent | Best short-term machinery because repo already has it working |
| MiniZero | Serious C++ distributed Zero system: server, self-play workers, optimization worker, batched GPU inference, Atari support | New games are C++; visual path is Atari-shaped; two-player limit; simultaneous action is not native | Strongest replacement/reference candidate, but not drop-in |
| EfficientZero | Ray actor/replay/shared-storage/reanalysis architecture, Atari visual path, C++/Cython tree | Single-agent Atari-shaped; no native two-player simultaneous game model | Architecture reference, not likely base |
| muzero-general | Readable Ray `SelfPlay`/`Trainer`/`ReplayBuffer`/`SharedStorage` roles | Educational, Python MCTS, alternating two-player assumptions, slow | Reference only |
| OpenSpiel AlphaZero | Clean simultaneous-game API and actor/learner/replay reference | Trainer is illustrative, not production-scale; visual CurvyTron fit awkward | Semantics reference |
| MCTX/JAX | Batched accelerator-friendly AlphaZero/MuZero/Gumbel search primitive | No env actors, replay, learner system; would require major build/rewrite | Best fast-search candidate if we rebuild |
| RLlib | Real distributed multi-agent infrastructure for PPO/APPO/etc. | No current first-class MuZero/AlphaZero search stack | Baseline/control option, not MuZero drop-in |
| Acme/Launchpad/Reverb | Actor/learner/replay architecture patterns | Not a turnkey visual simultaneous MuZero stack | Design reference |

## Working Recommendation

Do not declare a framework migration yet.

Near-term:

1. Keep LightZero as the shortest path for current CurvyTron visual MuZero
   smokes and for a custom two-seat collector.
2. Build toward repo-owned actor/replay/learner boundaries around that collector.
3. Treat MiniZero as the strongest full-system replacement candidate and inspect
   it more deeply before deciding.
4. Treat MCTX/JAX as the serious fast-search option if LightZero's PyTorch/C++
   MCTS path cannot be made to batch and scale enough.
5. Keep leaky public-turn wrappers and centralized joint-action wrappers as
   ablations only. A private sequential-commit barrier is fine if it preserves
   the real simultaneous env step and does not leak pending actions.

## What Would Change This View

- If MiniZero can accept a CurvyTron visual env with limited C++ work and can
  preserve simultaneous action semantics, it becomes a stronger base candidate.
- If LightZero's collector/GameBuffer can be extended cleanly to store per-seat
  trajectory data while reusing native target builders, staying LightZero gets
  stronger.
- If MCTX can run a minimal CurvyTron-style batched MuZero search quickly with a
  simple JAX model, a JAX rebuild becomes more attractive.
- If centralized joint-action `9` for 2P learns much better/faster than
  independent per-seat search, the algorithmic lane needs review.
- If model-free RLlib baselines learn quickly enough, MuZero may be unnecessary
  for the first useful CurvyTron agent.

## Next Research Tasks

- MiniZero spike: identify the minimum C++ files and network factory changes
  needed for a 2P visual CurvyTron-like env.
- LightZero spike: determine whether native `GameSegment`/`GameBuffer` can store
  two-seat trajectories without losing reward/value semantics.
- MCTX spike: run or sketch a tiny batched search over `B*P` roots with a dummy
  visual model to estimate implementation cost.
- Semantics spike: write the exact training target contract for independent
  per-seat search, including action masks, rewards, terminal handling, and
  player perspective.
- Baseline spike: keep a model-free simultaneous multi-agent baseline in view as
  a sanity check, but do not let it distract the MuZero optimizer lane.

## Short Coach/Captain Handoff

Optimizer is no longer treating single-loop LightZero speed as the whole
problem. Current working view: keep LightZero for near-term CurvyTron MuZero
machinery, but design the real speed path as actor chunks plus replay plus
learner/checkpoint publishing. MiniZero and MCTX are the main alternatives to
keep investigating. Sequential action collection is okay if pending actions are
private and the environment steps once after all players commit.
