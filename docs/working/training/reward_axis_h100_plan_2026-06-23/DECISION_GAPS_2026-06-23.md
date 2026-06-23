# Decision Gaps - 2026-06-23

Status: active captain ledger. This is a critical map of what we still do not
know before choosing long-running CurvyTron training architecture, rewards,
RND, observation surfaces, PPO/Puffer, or planning lanes.

Use this as an index into the existing docs, not a replacement for them.

## Evidence Rule

Do not believe a claim just because it appears in a doc, a previous chat, or a
single speed row.

Trust evidence in this order:

1. Source code plus focused tests that exercise the exact behavior.
2. Fresh run artifacts with immutable refs, manifests, and logs.
3. Audited manifests and no-launch packet checks.
4. Prior docs and chat summaries.
5. Intuition.

Docs are useful witnesses. They are not judges. Every promotion-grade choice
needs a runnable contract or measured artifact behind it.

## What We Can Choose Now

These choices are already believable enough:

- Keep canonical learning runs on the stock-ish source-state LightZero path
  until another path proves checkpoint/eval/tournament compatibility.
- Keep compact optimized training as a speed R&D lane, not the default learning
  lane.
- Run RND on the stock-ish path, not compact, and require stock plus
  `rnd_meter_v0` controls for every positive RND claim.
- Treat raw-tick pure MCTS as quarantined. Planning is still important, but it
  should first be tested as macro-actions, dense trajectory planning, selected
  state reanalysis, or planner distillation.
- Treat Flash raycasting as a fast tactical/control baseline, not proof that
  rays are the final CurvyZero representation.
- Prefer the historical best-known learner seed for longer non-RND Wave A rows
  when the anchor audit passes.

See:

- `STOCK_PATH_RND_REORIENTATION.md`
- `LONG_TERM_PLANNING_RND_STRATEGY.md`
- `CHECKPOINT_ANCHOR_POLICY.md`
- `../curvytron_flash_comparison_2026-06-23/RENDERING_RAYCASTING_PROFILE_PLAN.md`
- `../curvytron_flash_comparison_2026-06-23/PUFFERLIB_STRATEGY.md`

## Choices Still Blocked

### 1. Is RND Actually Useful, Or Merely Implemented?

What we know:

- `src/curvyzero/training/exploration_bonus.py` implements
  `CurvyRNDRewardModel`.
- `none`, `rnd_meter_v0`, and `rnd_replay_target_v0` have explicit config
  semantics.
- Focused tests prove local predictor training, frozen target hashes,
  zero-weight reward preservation, positive reward augmentation, and trainer
  entrypoint selection.

What could be wrong:

- The local tests are still not a real long LightZero collector/replay/learner
  run.
- The monkey-patched LightZero reward-model path could be bypassed by an
  upstream resolution change.
- RND could improve novelty metrics while hurting extrinsic survival.
- Per-batch RND normalization could create a brittle or misleading novelty
  scale.
- `state_dict` exists, but full reward-model checkpoint/resume round-trip is
  not promotion-grade proof yet.
- Terminal and final-observation RND behavior may be wrong or unmeasured. A
  death frame can be exactly where novelty is most misleading or most useful.

Evidence that closes the gap:

- A real stock-path `rnd_meter_v0` canary with matched `none` control.
- A low-weight `rnd_replay_target_v0` canary proving finite nonzero target
  reward deltas.
- At least one checkpoint, eval artifact, GIF artifact, and RND metrics JSONL.
- An RND save/resume round trip proving predictor, target, optimizer, counters,
  config hash, and metrics survive the actual training checkpoint path.
- A sampled-batch test or artifact proving latest-frame extraction on terminal
  and final-observation batches.
- Later: survival AUC/best/retention beats both stock and meter controls, then
  transfers beyond blank-canvas opponents.

First signal horizon:

- 0-30 minutes: health only.
- 30k-50k iterations: weak sanity signal.
- 100k-170k: first useful RND read.
- 240k-300k: retention read.

### 2. Which Reward Function Is Actually Best?

What we know:

- Implemented main variants are `sparse_outcome`,
  `dense_survival_plus_outcome`, `survival_plus_bonus_no_outcome`, and
  `survival_plus_bonus_plus_outcome`.
- `survival_plus_bonus_plus_outcome` is the current main hypothesis.
- `survival_plus_bonus_no_outcome` is the clean survival control.
- `sparse_outcome` is the clean game-outcome diagnostic.

What could be wrong:

- Plus-outcome may look good because of reward-scale or support effects, not
  better game play.
- Dense survival may improve average duration while reducing opponent pressure
  or aggression.
- Sparse outcome may produce strong isolated checkpoints but poor retention.
- Trainer reward across variants is not comparable without normalization and
  component telemetry.

Evidence that closes the gap:

- Same seed, same opponents, same horizon reward matrix.
- Survival AUC, best-so-far, latest retention, action collapse, death causes,
  and reward component telemetry.
- Support-cap and `reward_outcome_alpha` controls around plus-outcome.
- Preservation of best checkpoints when latest regresses.

Primary docs:

- `REWARD_INVENTORY.md`
- `MONITORING_SIGNALS.md`
- `CONTINGENCY_PLANS.md`

### 3. What Is The Best Known Checkpoint?

What we know:

- The current policy distinguishes initial learner seed, opponent curriculum
  refs, and promotion candidate refs.
- Historical best seed is the r18fresh plus-outcome `iteration_180000` ref.
- Bestseed Wave A manifests have passed the current no-launch anchor and packet
  audits in the docs.
- Wave A is auditable today as a prelaunch package, not as a learning result.

What could be wrong:

- "Best" can mean tournament champion, eval-best checkpoint, current
  launchable ref, or best seed for a specific curriculum.
- Historical ranking evidence is not the same as current Modal existence.
- `top4nz` is launchable opponent/ref material, not automatically the global
  best learner seed.
- Saved capacity snapshots are volatile. They are context for operator review,
  not launch permission.

Evidence that closes the gap:

- A single generated checkpoint registry joining tournament, eval, immutable
  ref existence, observation contract, reward contract, and loadability.
- Fresh audit before every medium or long launch.
- Clear label for seed role in every manifest.
- After launch: actual nonzero checkpoints, eval curves, RND metrics where
  relevant, and retention readouts at the useful horizons.

Primary doc:

- `CHECKPOINT_ANCHOR_POLICY.md`

### 4. Which Observation Surface Preserves The Right Information?

What we know:

- CurvyZero stock learning uses source-state gray64 visual stacks with the CPU
  oracle backend as the current reliable default.
- That stack is source-state raster, not browser-pixel parity.
- CurvyZero also has a flat 1v1/no-bonus egocentric ray observation path.
- Flash `raycast_v1` is a fast GPU-resident structured observation.
- Fast GPU rendering prototypes exist, but they are profile/canary lanes.

What could be wrong:

- Raycasting may be too local and miss topology, bounded-curvature
  reachability, enclosure, opponent commitment, and delayed dead ends.
- Visual stacks may also omit important state such as trail age, gap state,
  bonus timers, precise heading, or subcell geometry.
- A faster observation can still produce a weaker agent.
- A richer observation can be too expensive or too hard to optimize.
- GPU visual prototypes may look fast while still failing freshness, metadata,
  no-fallback, terminal-observation, or trainer-integration gates.

Evidence that closes the gap:

- Equal physics, reward, opponent, seed, and evaluation comparisons for:
  ray-MLP, ray-RNN, crop/map CNN/RNN, hybrid, privileged critic, and planner
  hybrid.
- Hard scenario evals for near-dead-end escape, corridor entry, opponent cut
  threats, enclosure conversion, gap schedules, and board/spawn generalization.
- Whole-loop cost, not renderer-only cost.

Primary docs:

- `../curvytron_flash_comparison_2026-06-23/RENDERING_RAYCASTING_PROFILE_PLAN.md`
- `CURVYTRON_GAME_MECHANICS_GATES.md`

### 5. Does Puffer Give Us A Better Baseline, Or Just Another Port?

What we know:

- Local repo-native PPO is smoke/plumbing, not a production PPO trainer.
- PufferLib has a serious fixed-buffer recurrent PPO runtime pattern.
- Puffer does not solve MuZero, MCTS, or planning.
- Flash may still be a better substrate for GPU-resident CurvyTron mechanics.

What could be wrong:

- A Puffer C/Ocean CurvyTron env may lose too much to CPU stepping or copies.
- Puffer self-play assumptions may not fit non-1v1 or future game modes.
- A fast ray PPO baseline may learn only tactical clearance.
- Porting Puffer deeply could distract from nearer learning evidence.

Evidence that closes the gap:

- Minimal 2-agent CurvyTron Puffer/Ocean parity spike.
- Raw env, rollout/update, and policy-quality benchmarks reported separately.
- Recurrent PPO eval against scripted/frozen opponents.
- Clear `puffer_ppo_baseline` denominator.

Primary doc:

- `../curvytron_flash_comparison_2026-06-23/PUFFERLIB_STRATEGY.md`

### 6. How Should Long-Term Planning Enter?

What we know:

- Raw-tick MCTS is probably the wrong default because each 60 Hz tick is a tiny
  steering change and ordinary tree search is sequential/irregular.
- Planning should not be thrown away. CurvyTron has real tactical and
  strategic lookahead.
- PPO plus planner actions is not clean PPO unless behavior probabilities are
  correct.

What could be wrong:

- Macro-actions can change death/reward semantics if they skip source frames.
- Dense planners can overfit heuristic scores rather than game outcomes.
- Planners using hidden state can look strong while being undeployable.
- A planner improvement may never amortize into the policy.

Evidence that closes the gap:

- Golden macro-action fidelity tests for frame replay, early terminal, reward
  accumulation, gap/printing state, and trail latency.
- Selected-state policy-only versus planner-action lift.
- Equal GPU-seconds and action-latency comparisons.
- Distillation/reanalysis proof if planner results are claimed as network
  progress.

Primary docs:

- `LONG_TERM_PLANNING_RND_STRATEGY.md`
- `CURVYTRON_GAME_MECHANICS_GATES.md`
- `MONITORING_SIGNALS.md`

### 7. Are We Evaluating The Right Behaviors?

What we know:

- Existing docs correctly warn that leaderboard comes later.
- Current monitoring focuses on health, survival AUC, best/latest retention,
  action collapse, and tournament exposure when mature.

What could be wrong:

- Survival against weak or blank opponents can reward passivity.
- GIF review can miss systematic exploitability.
- Early flat curves may be misread as failure even though prior runs needed
  long horizons.
- Tournament Elo can mislead without exposure and nonzero checkpoints.

Evidence that closes the gap:

- A hard CurvyTron scenario suite independent of training reward.
- Death-cause and pressure/aggression metrics.
- Checkpoint matchup matrices, not just latest or Elo.
- Held-out scripted/frozen opponents and later league exposure.

Primary docs:

- `MONITORING_SIGNALS.md`
- `CONTINGENCY_PLANS.md`

### 8. Are The Game Mechanics Faithful Across Branches?

What we know:

- The mechanics gate doc already lists source timing, action semantics, reward
  accounting, observation contracts, opponent contracts, and seat perspective.

What could be wrong:

- Sequential server resolution, simultaneous-action semantics, trail latency,
  gap processes, bonus effects, and border behavior can silently diverge across
  Flash, CurvyZero, Puffer, macro-action, or planner branches.
- A branch can win by changing the game.

Evidence that closes the gap:

- Golden traces for wall collisions, head-on/crossing collisions, fresh-tail
  latency, gap printing, bonus effects, action repeat, and two-seat reward
  perspective.
- Every branch records the exact physics, reward, observation, action cadence,
  opponent, and information-access contract.

Primary doc:

- `CURVYTRON_GAME_MECHANICS_GATES.md`

## Near-Term Gap-Closing Experiments

Run broad, but keep the rows interpretable.

1. RND real-path health gate:
   `none` versus `rnd_meter_v0` versus low positive `rnd_replay_target_v0`,
   matched seed and horizon, requiring metrics, checkpoints, eval, GIF, and
   explicit save/resume and terminal-batch checks.

2. Wave A bestseed controlled sweep:
   keep non-RND reward/cadence controls alive beside RND; use long tiers only
   within the H100 operating limits in `OPERATING_PATTERNS.md`. For an 8h+
   first read, prefer the capacity-cleared `long17_no_highest_weight_bestseed`
   shape over the full 90-row packet unless the operator explicitly chooses a
   different tier after fresh audits.

3. Observation scenario suite:
   compare ray, visual, and hybrid information on fixed states before declaring
   any representation strategically adequate.

4. Minimal Puffer/PPO feasibility:
   build or isolate a 2-agent parity spike, then measure raw env, PPO
   rollout/update, and policy quality separately.

5. Macro-action fidelity gate:
   prove action-repeat or macro-actions replay source frames exactly before
   using them for planner or training claims.

6. Selected-state planner lift:
   test policy-only, full enumeration, beam/CEM/MPC, and small Gumbel/MuZero
   style planning on fixed states at equal compute before launching long
   planner training.

## Stop Yourself From Over-Choosing

Do not choose a final architecture from:

- one Flash raw-env number;
- one compact speed row;
- one RND metric file;
- one GIF;
- one early survival spike;
- one leaderboard rank without exposure;
- one Puffer/PPO smoke;
- one planning example state.

The current correct posture is broad parallel evidence gathering with clean
denominators. The final architecture should be the smallest stack that survives
hard strategic evals and long-run retention:

```text
faithful game + useful observation + stable learning + retained checkpoints
+ honest evals + acceptable wall-clock
```
