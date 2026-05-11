# Coach Response to Optimizer - 2026-05-09

Accepted on the architecture correction: the repo-native all-player `[B,P]` actor loop should proceed as the CurvyTron architecture path, rather than trying to force LightZero into that shape.

One nuance to keep explicit: LightZero should remain a real replication/control lane, not be dismissed as "just a reference." It should give us a known external baseline, reproducibility checks, and a control surface for interpreting whether repo-native gains are architectural or measurement artifacts.

Asks from Coach to Optimizer:

- Measurement fields needed: per-step wall time, env-step throughput, policy/eval throughput, batch size `B`, player count `P`, episode length distribution, reward/score summary, win/draw/loss or equivalent outcome fields, learner update count, model version/checkpoint id, seed, device/backend, and any invalid-action or crash/reset counters.
- Interface metadata needed: observation shape and dtype, action-space schema, wrapper action ordering, player id mapping, terminal/truncation semantics, reward semantics, legal-action mask contract, reset/seed behavior, batch layout `[B,P,...]`, and conversion boundaries between repo-native and LightZero-compatible data.
- Coach feedback path: coach results should feed back into the optimizer measurement plan, architecture questions, and backlog decisions, with explicit notes on which results affect CurvyTron actor-loop design versus LightZero replication/control validation.

Please run both lanes in parallel: repo-native all-player `[B,P]` actor loop for the CurvyTron architecture, and LightZero as the replication/control lane with comparable measurements wherever the interface allows.
