# Architecture Boundaries

## Clean Ownership

### Environment

Owns game state, reset, step, observation, reward, terminal state, and info. It
should not know about Modal, volume refs, checkpoint mirrors, GIF websites, or
trainer scheduling.

For this refactor, do not edit environment mechanics unless a trainer regression
test exposes a clear interface mismatch. The coach lane is training code.

### LightZero

Owns `train_muzero`, collector, replay, learner updates, target semantics,
policy/search, and the internal experiment directory.

### CurvyZero Training Scaffolding

Owns config construction, env registration, run metadata, progress/status
payloads, checkpoint discovery, resume choice, background eval/GIF scheduling,
and Modal entrypoints.

This is the main refactor surface.

### External Observability

Owns eval summaries, GIF summaries, browser markers, manifests, and dashboards.
It should read artifacts; it should not change trainer semantics.

## Desired Shape After Refactor

The big trainer file should become a thin entrypoint that wires small helpers
together. Pure helpers should be importable in tests without Modal.

Potential modules, pending audit:

```text
src/curvyzero/training/checkpoints.py
src/curvyzero/training/progress.py
src/curvyzero/training/resume.py
src/curvyzero/training/background_eval.py
src/curvyzero/training/lightzero_config.py
```

Do not create these modules until tests and audits agree on the cut.
