# Glossary

Use plain language.

## Stock LightZero Train Path

The current training path we are trying to keep: stock LightZero `train_muzero`
called by the CurvyZero Modal trainer in `--mode train`.

Prefer this over vague uses of "trusted lane" in new docs.

## Trainer Scaffolding

Code around LightZero training: config building, checkpoint discovery, resume,
status, poller, eval/GIF scheduling, manifests, and Modal wrappers.

## Environment Contract

What the trainer expects from the env: reset, step, observation, reward, done,
and info. This refactor should not redesign the environment.

## Checkpoint Discovery

Finding actual saved `iteration_*.pth.tar` files. Correct discovery must scan
all relevant `lightzero_exp*` dirs, not just one fixed path.

For the current bug, prefer the phrase `lightzero_exp* discovery`.

## Status

Human-readable files and CLI output that say where a run is. Status is only as
good as the checkpoint discovery behind it.

## Poller

A background process that watches for new checkpoints and triggers eval/GIF
jobs.

## Eval/GIF

External observability jobs. They help us inspect policies, but they are not
the LightZero learner.
