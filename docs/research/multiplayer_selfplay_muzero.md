# Multiplayer self-play and MuZero-style search

Research snapshot: 2026-05-08.

## Short answer

AlphaZero and MuZero are not, by themselves, recipes for multiplayer
general-sum CurvyTron. Their published successes are self-play plus neural
search in two-player zero-sum perfect-information board games, with MuZero also
covering single-agent Atari. Multiplayer real-time systems that did scale
well, such as AlphaStar, OpenAI Five, and Quake Capture the Flag, leaned more
on policy learning, populations, leagues, and checkpoint diversity than on
online tree search.

For CurvyTron v0, use an ego-perspective shared policy/value model, train all
agents from self-play against a checkpoint pool, and if search is included,
search only the focal agent while sampling opponents from policy-only agents.
Do not start with full joint-action search or a full AlphaStar-style league.
Those are useful later experiments, but the first baseline should keep the game
formulation stable and the branching factor under control.

## What the primary sources say

### AlphaZero and MuZero

- AlphaZero trained from self-play with MCTS-guided policy improvement and was
  demonstrated on chess, shogi, and Go. These are two-player, zero-sum,
  perfect-information games, even though the algorithm was presented as
  domain-general within that class. See Silver et al.,
  ["Mastering Chess and Shogi by Self-Play with a General Reinforcement
  Learning Algorithm"](https://arxiv.org/abs/1712.01815), and the later Science
  version, ["A general reinforcement learning algorithm that masters chess,
  shogi, and Go through self-play"](https://doi.org/10.1126/science.aar6404).
- MuZero replaces AlphaZero's known simulator with a learned dynamics model
  that predicts reward, policy, and value for planning. Its reported domains
  were 57 Atari games plus Go, chess, and shogi, again not n-player
  general-sum multiplayer. See Schrittwieser et al.,
  ["Mastering Atari, Go, Chess and Shogi by Planning with a Learned
  Model"](https://arxiv.org/abs/1911.08265) and
  [Nature](https://www.nature.com/articles/s41586-020-03051-4).
- The OpenSpiel project is useful context because it explicitly supports
  n-player, general-sum, simultaneous-move, perfect- and imperfect-information
  games, but its AlphaZero documentation describes AlphaZero as an algorithm
  for perfect-information games and notes its implementation is illustrative,
  not a production-scale reproduction. See
  [OpenSpiel](https://github.com/google-deepmind/open_spiel) and
  [OpenSpiel AlphaZero docs](https://openspiel.readthedocs.io/en/latest/alpha_zero.html).

Practical implication: a CurvyTron implementation must choose its multiplayer
semantics. A scalar value from the ego player's perspective is the simplest
choice. A vector value for every player is possible, but then MCTS selection and
backup are no longer AlphaZero's two-player negamax backup. General-sum search
requires a solution concept, opponent model, or heuristic action-selection rule.

### Self-play with search beyond two-player board games

- Poker is the strongest precedent for self-play plus search outside
  perfect-information board games. Pluribus trained a blueprint strategy by
  self-play against copies of itself and used real-time depth-limited search in
  six-player no-limit Texas hold'em. It is multiplayer and imperfect
  information, but its search is CFR-style game solving, not MuZero MCTS. See
  Brown and Sandholm, ["Superhuman AI for multiplayer
  poker"](https://noambrown.github.io/papers/19-Science-Superhuman.pdf) and
  [Science DOI](https://doi.org/10.1126/science.aay2400).
- ReBeL combines deep RL and search for imperfect-information games, but its
  convergence claims are for two-player zero-sum games. It matters because it
  shows why naive AlphaZero-style state search is not sound when hidden
  information and strategic beliefs matter. See Brown et al.,
  ["Combining Deep Reinforcement Learning and Search for
  Imperfect-Information Games"](https://papers.nips.cc/paper_files/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html).
- Student of Games combines guided search, self-play learning, and
  game-theoretic reasoning across chess, Go, poker, and Scotland Yard. It is a
  better conceptual precedent than MuZero for "search plus learning outside
  board games," but it is far heavier than a CurvyTron v0 and is not simply
  MuZero with more players. See Schmid et al.,
  ["Student of Games"](https://arxiv.org/abs/2112.03178).
- There is direct simultaneous-move MCTS work on Tron. Lanctot et al. model
  simultaneous-move Tron both sequentially and as a stacked matrix game, testing
  Sequential UCT, Decoupled UCT, Exp3, and Regret Matching. In their Tron
  experiments, Decoupled UCB1-Tuned performed best overall, but this was still
  a small-game search study, not learned MuZero self-play at scale. See
  ["Monte Carlo Tree Search for Simultaneous Move Games: A Case Study in the
  Game of Tron"](https://cris.maastrichtuniversity.nl/en/publications/monte-carlo-tree-search-for-simultaneous-move-games-a-case-study-/).
- Broader simultaneous-move MCTS results show the extension is non-trivial and
  game-dependent. Some regret-minimizing variants converge in two-player
  zero-sum simultaneous-move games, while empirical comparisons across games
  show tuning and opponent dependence. See
  ["Convergence of Monte Carlo Tree Search in Simultaneous Move Games"](https://papers.nips.cc/paper/5145-convergence-of-monte-carlo-tree-search-in-simultaneous-move-games),
  ["Monte Carlo Tree Search variants for simultaneous move games"](https://cris.maastrichtuniversity.nl/en/publications/monte-carlo-tree-search-variants-for-simultaneous-move-games/),
  and Sturtevant,
  ["An Analysis of UCT in Multi-Player Games"](https://doi.org/10.3233/ICG-2008-31402).

Practical implication: CurvyTron is closer to simultaneous-move Tron than to
chess or Go. The moment all players move every tick, joint branching becomes
`|A|^N` per tick before considering horizon length. That is the main reason to
prefer searched-ego-only or sampled-opponent search for v0.

### Multi-agent RL lessons without search

- AlphaStar reached grandmaster StarCraft II through imitation learning,
  reinforcement learning, and a diverse league of agents and counter-agents,
  not through online MCTS. See Vinyals et al.,
  ["Grandmaster level in StarCraft II using multi-agent reinforcement
  learning"](https://www.nature.com/articles/s41586-019-1724-z).
- OpenAI Five defeated the Dota 2 world champions using large-scale PPO and
  self-play, explicitly without search. The useful lesson for CurvyTron is that
  self-play can produce long-horizon team behavior when scaled, but it needs
  careful tooling for continual training and evaluation. See
  [OpenAI Five](https://openai.com/research/openai-five) and
  ["Dota 2 with Large Scale Deep Reinforcement
  Learning"](https://arxiv.org/abs/1912.06680).
- DeepMind's Quake III Capture the Flag agents used population-based
  reinforcement learning and tournament-style evaluation in a 3D multiplayer
  game. The relevant lesson is not MCTS, but population diversity, robust
  evaluation, and training against varied co-players/opponents. See Jaderberg
  et al., ["Human-level performance in 3D multiplayer games with
  population-based reinforcement learning"](https://doi.org/10.1126/science.aau6249).
- PSRO and related open-ended learning work explain why single-current-policy
  self-play can overfit or cycle in non-transitive games. See Lanctot et al.,
  ["A Unified Game-Theoretic Approach to Multiagent Reinforcement
  Learning"](https://papers.nips.cc/paper_files/paper/2017/hash/3323fe11e9595c09af38fe67567a9394-Abstract.html),
  and Balduzzi et al.,
  ["Open-ended learning in symmetric zero-sum games"](https://proceedings.mlr.press/v97/balduzzi19a.html).
- Population Based Training is useful once the training system is expensive
  enough to justify automatic hyperparameter schedules. It is not required for
  v0, but is a strong later tool for entropy, learning-rate, reward-shaping, and
  search-budget schedules. See Jaderberg et al.,
  ["Population Based Training of Neural Networks"](https://arxiv.org/abs/1711.09846).

## Practical formulations for CurvyTron

### Ego-perspective shared policy

Use one network for every player:

- Input: observation from player `i`'s frame of reference, with heading,
  nearby walls/trails, living opponents, distances, and optionally a short
  history stack.
- Output: action logits for player `i`, scalar value `V_i`, and optional reward
  prediction if using MuZero-style dynamics.
- Training data: every live player at every tick contributes an ego transition.
- Symmetry: randomize player order, colors, spawn positions, and observation
  channels so the shared policy does not learn seat/color artifacts.

This is the best default. It turns n-player gameplay into many ego-agent
samples while keeping the model small and the policy stationary enough to
debug.

### Checkpoint pool or lightweight league

Maintain a pool of frozen policies:

- Always include the latest policy, recent checkpoints, a few older high-Elo
  checkpoints, random/heuristic baselines, and maybe a "safe driver" baseline.
- For each match, sample opponents from the pool, with a bias toward recent and
  strong policies.
- Evaluate by round-robin against a held-out pool that training does not sample
  from directly.

This is a low-cost version of the AlphaStar/PSRO lesson: train against a
distribution, not just the latest copy of yourself.

### Population-based training

Use PBT later, not first. It can tune entropy, learning rate, reward weights,
search simulations, model unroll length, opponent-pool sampling temperature, and
exploration noise. In v0, hand-pick a small number of stable settings so failure
analysis is readable.

### Policy-only opponents

During data generation and search, let non-focal agents act from frozen policy
networks. This has two variants:

- All agents policy-only: simplest baseline, no search.
- Focal player searched, opponents policy-only: practical MuZero/AlphaZero-like
  improvement without joint branching.

This is the right opponent model for v0 because it makes opponent behavior
sampleable and cheap.

### Searched ego only

At a decision point for player `i`, run MCTS over only `i`'s actions. For each
edge expansion or simulated step, sample other players' actions from their
assigned policy networks, then advance the simulator or learned model. Backup
only the ego utility `u_i`.

Useful details:

- Treat simultaneous opponent moves as chance/policy-sampled events, not as
  actions the ego search controls.
- Store the MCTS visit distribution as the policy target only for the focal
  player.
- Use short horizons and small simulation budgets. CurvyTron is fast and
  reactive; a shallow search may catch tactical collisions without trying to
  solve the whole match.
- Start with the true simulator if it is cheap. Add MuZero learned dynamics only
  when simulator cost, partial observability, or representation learning
  justifies it.

This is the recommended search formulation for v0.

### Joint-action search

Full joint-action search expands every combination of player actions. With
`N` players and `A` discrete actions per player, one tick has `A^N` children.
For example, 6 players with 3 turn actions already gives 729 joint actions per
tick. A 10-step lookahead is not remotely viable without aggressive sampling,
factorization, or abstraction.

Use this only as a research experiment:

- small `N`, small `A`, and short horizon;
- top-k action pruning from policies;
- progressive widening;
- decoupled UCT/regret-matching simultaneous-move search;
- vector values if you actually need general-sum reasoning.

It is not a v0 training backbone.

## Reward and value formulation

Prefer a scalar ego payoff for v0:

- Terminal rank payoff: winner near `+1`, first death near `-1`, intermediate
  ranks linearly spaced; average tied ranks.
- Centered payoff: subtract the mean terminal score in the match so the sum is
  approximately zero. This makes free-for-all evaluation cleaner and reduces
  "everyone survive forever" incentives.
- Small shaping: survival ticks, collision penalty, and optional kill/pressure
  rewards can help exploration, but keep shaping small relative to terminal
  rank so the agent does not optimize stalling or griefing over winning.

Train `V_i` to predict ego return from player `i`'s perspective. Avoid a vector
value head until the scalar baseline works. Vector values make sense later if
you want explicit opponent modeling, alliance/kingmaking analysis, or
general-sum search backups.

## Known failure modes and mitigations

- Non-stationarity: every policy update changes the environment seen by other
  learners. Mitigate with frozen checkpoint opponents, version-tagged replay,
  and evaluation against fixed pools.
- Cycling and non-transitivity: a policy can beat the previous policy while
  losing to an older one. Mitigate with checkpoint reservoirs, exploiters,
  round-robin evaluation, PSRO-style mixtures, or AlphaRank-style analysis.
- Exploitability: latest-self self-play can learn brittle conventions. Mitigate
  by training occasional best-response/exploiter agents against frozen mains and
  by holding out evaluation opponents.
- Collusion, teaming, and kingmaking: free-for-all games can reward implicit
  coordination against a third player or sacrificing one agent to help a shared
  policy copy. Mitigate with anonymous randomized identities, centered
  individual payoffs, no persistent partner identities, and evaluation against
  mixed independent checkpoints. Sequential social-dilemma work is a reminder
  that learned cooperation/conflict is policy-level and environment-dependent;
  see Leibo et al., ["Multi-agent Reinforcement Learning in Sequential Social
  Dilemmas"](https://discovery.ucl.ac.uk/10069053/).
- Credit assignment: delayed death/win rewards are sparse, and kill credit can
  be ambiguous. Use rank payoff plus light shaping, and log death causes,
  nearest threats, and last-contact features for analysis.
- Branching-factor explosion: simultaneous joint search grows as `A^N` per
  tick. Avoid in v0 with searched-ego-only, policy-sampled opponents, short
  horizons, and action pruning.
- Model bias in MuZero: search exploits learned-model errors, especially if
  opponent policies in search differ from those in the replay. Keep search
  shallow, mix in true-simulator rollouts if possible, reanalyze with current
  networks, and monitor value calibration by opponent-pool slice.
- Stale replay: old trajectories may come from obsolete opponent mixtures.
  Keep policy/version metadata, cap replay age, and sample recent data more
  heavily while retaining some older diversity.
- Shared-policy identity leakage: if observations encode stable color/seat
  identity, the shared model may learn conventions that fail under shuffled
  seats. Randomize and canonicalize observations early.

## Recommendation for v0

Build the first CurvyTron self-play system as:

1. Shared ego policy/value network for all players.
2. Policy-only self-play against a checkpoint pool, with random/heuristic
   baselines included.
3. Scalar centered rank payoff with minimal shaping.
4. Optional searched-ego-only MCTS using the true simulator first; opponents are
   sampled from their assigned policy networks.
5. Held-out round-robin evaluation across checkpoints, plus occasional
   best-response/exploiter training to measure brittleness.

Defer these until after a stable policy-only baseline:

- learned MuZero dynamics;
- full joint-action search;
- vector-valued general-sum backups;
- full league/PBT automation.

The main engineering bet is that CurvyTron needs diversity and stable opponent
sampling before it needs sophisticated multiplayer search. Search should be a
tactical booster for the focal player, not the core multiplayer solution in the
first version.

## Source index

- Silver et al., AlphaZero preprint:
  https://arxiv.org/abs/1712.01815
- Silver et al., AlphaZero Science paper:
  https://doi.org/10.1126/science.aar6404
- Schrittwieser et al., MuZero:
  https://arxiv.org/abs/1911.08265 and
  https://www.nature.com/articles/s41586-020-03051-4
- Brown and Sandholm, Pluribus:
  https://noambrown.github.io/papers/19-Science-Superhuman.pdf and
  https://doi.org/10.1126/science.aay2400
- Brown et al., ReBeL:
  https://papers.nips.cc/paper_files/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html
- Schmid et al., Student of Games:
  https://arxiv.org/abs/2112.03178
- Lanctot et al., simultaneous-move MCTS for Tron:
  https://cris.maastrichtuniversity.nl/en/publications/monte-carlo-tree-search-for-simultaneous-move-games-a-case-study-/
- Lisy et al., simultaneous-move MCTS convergence:
  https://papers.nips.cc/paper/5145-convergence-of-monte-carlo-tree-search-in-simultaneous-move-games
- Tak et al., simultaneous-move MCTS variants:
  https://cris.maastrichtuniversity.nl/en/publications/monte-carlo-tree-search-variants-for-simultaneous-move-games/
- Sturtevant, UCT in multi-player games:
  https://doi.org/10.3233/ICG-2008-31402
- OpenSpiel:
  https://github.com/google-deepmind/open_spiel and
  https://openspiel.readthedocs.io/en/latest/alpha_zero.html
- AlphaStar:
  https://www.nature.com/articles/s41586-019-1724-z
- OpenAI Five:
  https://openai.com/research/openai-five and
  https://arxiv.org/abs/1912.06680
- Quake CTF population-based RL:
  https://doi.org/10.1126/science.aau6249
- PSRO:
  https://papers.nips.cc/paper_files/paper/2017/hash/3323fe11e9595c09af38fe67567a9394-Abstract.html
- Open-ended learning / non-transitivity:
  https://proceedings.mlr.press/v97/balduzzi19a.html
- Population Based Training:
  https://arxiv.org/abs/1711.09846
- Sequential social dilemmas:
  https://discovery.ucl.ac.uk/10069053/
