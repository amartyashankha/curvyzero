# Optimizer Framework Working Hypotheses

Date: 2026-05-09

Status: working hypotheses and evidence gates for the optimizer lane. This is
not a final architecture decision and not a training-quality claim.

## Current Stance

Treat a project-owned PPO/IPPO-style runner as the leading repo-native
optimizer bench hypothesis now, because it is the shortest path to exposing
CurvyTron's multiplayer `[B, P]` wrapper loop while preserving source-control
semantics.

Keep LightZero alive as a serious replication/control lane for MuZero plumbing,
target audits, stock Atari/Pong-like reproduction, custom dummy Pong bridge
evidence, and comparison scorecards. Do not treat LightZero failures as
evidence against MuZero until the coach lane records a credible reproduction or
a clear blocker.

Tentatively treat Mctx/JAX as a later owned-search module, not the base trainer
today. Revisit TorchRL, Sample Factory, or RLlib if measurements show the owned
bench is hiding bottlenecks, missing needed library behavior, or spending too
much time rebuilding standard machinery.

## Why

The current uncertainty is not whether a public framework can train something.
The uncertainty is whether our source-faithful CurvyTron loop can run the right
thing:

- wrapper/replay action maps for all live players, converted to source control
  state for elapsed-ms frames;
- trainer-facing observations and legal masks;
- sparse payoff rewards plus final observations;
- public reset/autoreset ordering;
- replay/rollout chunking and metadata;
- opponent/checkpoint metadata;
- wall-clock profiling buckets.

An owned PPO/IPPO runner is currently the shortest-looking path to measuring
that without forcing CurvyTron into Atari, turn-based board-game, or single-ego
wrapper semantics. That is a hypothesis to validate, not a conclusion to defend.

## Comparison

| Candidate | Fit | Risk | Optimizer use |
| --- | --- | --- | --- |
| Owned PPO/IPPO or CleanRL-style runner | Best ownership of `[B, P]`, replay, profiler, sparse rewards, and metadata. | We must write the runner, rollout buffer, eval, and checkpoint path. No MCTS. | Leading repo-native bench hypothesis now. |
| LightZero | Ready MuZero/MCTS stack and required controls. | Adapter can hide all-player wrapper metadata; replay/profiler/metadata are harder to keep transparent; current reproduction status is still unresolved. | Serious replication/control and target audit. |
| TorchRL | Good PyTorch ecosystem fit for multi-agent tensors, collectors, replay, and PPO losses. | More framework surface than the owned bench; no native MCTS. | Candidate to test later. |
| Sample Factory | Strong high-throughput PPO with multi-agent, self-play, multi-policy, image/dict obs support. | Optimized architecture may hide the exact bottlenecks we still need to expose. | Later throughput candidate. |
| RLlib | Very flexible multi-agent API and policy mapping. | Heavy Ray stack and less transparent early profiling. | Later distributed/control candidate. |
| Mctx/JAX | Strong batched MuZero/Gumbel MuZero search module. | Search only; we still own env, replay, learner, targets, checkpointing, and profiling. | Later search box. |

## Evidence Gates

- If the owned PPO/IPPO timing bench cannot keep the required CurvyTron
  contracts, fix the contracts before scaling a framework.
- If LightZero produces a credible stock Pong-like reproduction and its
  custom-env bridge preserves the metadata we need, keep it as a serious
  promotion candidate for MuZero experiments.
- If LightZero cannot reproduce a known Pong-like control under bounded,
  documented, near-recipe conditions, record the exact blocker before using that
  result to interpret custom CurvyTron work.
- If the owned bench shows policy/model cost dominates, compare Torch/LightZero/
  Mctx timing on the same ego-row shape.
- If env/obs/autoreset dominates after debug events are off, optimize the
  source-faithful vector runtime.
- If replay/handoff dominates, optimize chunking and writer/learner queues
  before adding more actors.
- If the coach lane proves LightZero targets are useful and the optimizer
  profile says search cost is acceptable, promote LightZero or Mctx from
  control/search-spike into a serious candidate.

## Open Context Risks

- LightZero may look worse than it is if current runs are too far from stock
  recipes or too small to learn.
- An owned PPO bench may look cleaner because it exposes more metadata, not
  because PPO is the better final algorithm.
- A framework with more machinery, such as TorchRL or Sample Factory, may become
  useful once contracts stabilize and throughput, replay, or multi-policy
  support becomes the actual bottleneck.
- Mctx may become relevant sooner if coach-lane evidence shows planning targets
  are the main missing ingredient and the search cost is measurable.
- PPO may be too weak or too myopic for CurvyTron's delayed adversarial
  steering dynamics, making it a poor proxy for the final architecture.
- LightZero may become the better main candidate if stock Pong-like
  reproduction becomes credible and the CurvyTron bridge preserves metadata.

## External Anchors

- LightZero docs: <https://opendilab.github.io/LightZero/>
- LightZero repository: <https://github.com/opendilab/LightZero>
- Mctx repository: <https://github.com/google-deepmind/mctx>
- TorchRL docs: <https://docs.pytorch.org/rl/main/>
- TorchRL collectors: <https://docs.pytorch.org/rl/main/reference/collectors.html>
- Sample Factory docs: <https://www.samplefactory.dev/>
- PettingZoo Parallel API: <https://pettingzoo.farama.org/api/parallel/>
- RLlib multi-agent docs: <https://docs.ray.io/en/latest/rllib/multi-agent-envs.html>
- CleanRL docs: <https://docs.cleanrl.dev/>
