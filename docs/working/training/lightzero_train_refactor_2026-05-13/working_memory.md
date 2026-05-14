# Working Memory

## Current Mental Model

The core trainer file has accumulated too many responsibilities because it had
to launch, observe, resume, checkpoint, evaluate, and display many experiment
waves. The cleanup should separate responsibilities without changing the
learning path.

Coach scope: focus on the training launcher and the scaffolding around it. Do
not burn time redesigning the environment. Read env code only to understand the
contract the trainer relies on.

## Most Important Known Risk

The first checkpoint-discovery risk is now fixed under focused tests. Keep
watching for the same pattern in less central callers: any new code that reads
only `train/lightzero_exp/ckpt` can undercount or miss the real latest
checkpoint if DI-engine wrote to a timestamped sibling.

## Refactor Principle

First make the current behavior observable with tests. Then move code behind
that tested behavior.

Bugfixes can happen before or during the refactor only after tests lock down the
bug. The first checkpoint-discovery bug was fixed, then the pure path/parsing
helpers were extracted. The next move should be another small tested cut, likely
resume or poller candidate logic, not a broad rewrite.

## User Priorities Captured

- Use simple language and clear names.
- Stay close to stock LightZero.
- Keep environment behavior clean and separate from trainer plumbing.
- Keep the main thread as coach: plan, delegate, synthesize, decide.
- Use subagents aggressively but keep their work bounded.
- Keep docs current because memory will decay.
- Prefer real end-to-end contract tests over random tests.
- Delete irrelevant tests only after replacement coverage exists.
- Future frozen-opponent selection should be externally controlled: tournament
  or another control plane writes opponent registry data, and the trainer
  consumes a resolved opponent spec.
- Do not add tournament policy, Modal Dict plumbing, or hard-coded opponent refs
  directly into the Modal trainer.
