# Open Questions And Hypotheses

Purpose: keep the investigation honest. Each item should eventually move to
evidence, rejected, or next action.

## Confirmed Or Strongly Supported

### H1: The scaled May 12 path was not stock LightZero training.

Status: confirmed.

Evidence:

- `--mode two-seat-selfplay` does not call stock `train_muzero`.
- It owns collection, replay rows, target construction, and checkpointing.
- It calls `MuZeroPolicy.learn_mode.forward` directly.

Consequence: flat curves from that path are evidence against the custom adapter,
not against CurvyTron or LightZero as a whole.

### H2: Fixed/frozen opponent was not wrong, only a different claim.

Status: supported.

Plain version: stock LightZero can train one player if the env supplies the
opponent. That is not exact live two-seat self-play, but it can still be a
useful way to learn and may be the best near-term route if the opponent comes
from recent checkpoints.

Consequence: do not dismiss fixed/frozen as useless. Label it as stock
ego-vs-opponent training or recent-checkpoint opponent training.

### H3: Turn-commit is unsafe for training as currently written.

Status: confirmed by target/replay audit.

Plain version: fake "waiting for player 1" rows enter replay as if they were
real game steps. That can assign credit to the wrong place.

Consequence: keep turn-commit as smoke/profile only until replay stores only
real physical ticks.

### H4: Centralized joint-action is a clean stock control, not self-play.

Status: supported.

Plain version: one scalar action can encode both players' actions. That keeps
one LightZero row equal to one real CurvyTron tick. But one policy controls the
whole joint action, so it is not two competitive policies.

Consequence: useful learning control; do not overclaim.

## Active Questions

### Q1: Can source-state fixed/frozen support recent checkpoint opponents cleanly?

Why it matters: this may let us stay close to stock `train_muzero` while still
training against improving opponents.

Status: yes for tiny CPU and GPU stock canaries, not yet for learning.

Evidence:

- local source-state env and Modal readiness tests pass for frozen checkpoint
  opponent wiring;
- `stock-frozen-canary-source-state-s304-20260512` completed with
  `called_train_muzero=true`, `trainer_entrypoint=lzero.entry.train_muzero`,
  `opponent_provider_load_ok=true`, and `opponent_provider_load_strict=true`.
- `stock-frozen-gpu-base-canary-source-state-s304-20260512b` completed with
  `called_train_muzero=true`, strict checkpoint opponent load, checkpoint save,
  and `torch_cuda_available=true` on an NVIDIA L4, using
  `env_manager_type=base`.

Remaining question: can this stock lane produce a learning curve?

### Q2: What exact LightZero fields matter for native replay parity?

Status: first tiny bridge parity test passed.

Need to map:

- observation stack;
- action;
- reward;
- `to_play`;
- `action_mask`;
- root value;
- visit distribution;
- done/bootstrap masks;
- segment boundary padding.

### Q3: Did the custom two-seat target builder create bad value/reward targets?

Need tiny known trajectories and a comparison against native
`MuZeroGameBuffer.sample(...)`.

Local status: a tiny two-seat physical-tick projection helper and pytest now
exist. The pure projection check passes locally. In Modal/LightZero, native
`GameSegment` construction, `MuZeroGameBuffer.push_game_segments`, and
deterministic reward/value/policy target assertions pass for the hand-authored
three-tick trace. This proves the bridge direction is plausible. It does not
prove the existing custom target builder was correct.

### Q4: Did pure same-policy self-play erase competitive signal?

The no-learning audit found many pure same-policy rows with zero terminal
outcome signal. Need to separate three possibilities:

- game dynamics often produce symmetric/no-winner outcomes;
- reward accounting missed death/outcome;
- same-policy symmetry is a poor first curriculum.

### Q5: Which pieces are slow because of GPU/CPU split?

Current read: model inference and learner run on GPU when configured; env,
rendering, replay storage, and much tree bookkeeping are CPU-side. Need to
record this as a performance fact, not a learning blocker by itself.

### Q6: What common MuZero/AlphaZero implementation pitfalls match this failure?

Potential areas for literature/web review:

- target construction and bootstrapping;
- replay buffer age;
- self-play opponent staleness;
- MCTS temperature and exploration;
- reward scaling and value support;
- action masks and legal actions;
- evaluation using greedy policy vs sampled collection.

Latest read: the useful external-research output is a local checklist, not a
new algorithm recommendation. The highest-priority checks are target parity,
mask parity, player-perspective parity, replay freshness, reward scale, and
separating eval mode from collect mode.

### Q7: Are we repeating the Pong mistake?

Working hypothesis: yes. Pong had several custom or semi-custom attempts that
were hard to trust. The credible signal came from the closer stock LightZero
path plus survival-first eval after enough horizon.

Status: supported by the Pong history pass. Direct lesson: CurvyTron should
repeat the stock-loop discipline that made Pong interpretable, not the custom
trainer path that produced ambiguous artifacts.

## Next Actions

1. Finish stock LightZero dataflow doc.
2. Finish cleanup list and patch the worst stale defaults/docs.
3. Decide whether the next proof is recent-frozen stock training, centralized
   joint-action stock training, or native replay parity first.
4. Run web/literature review only against specific questions, not as a broad
   distraction.
