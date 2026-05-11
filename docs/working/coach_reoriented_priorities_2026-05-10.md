# Coach Reoriented Priorities - 2026-05-10

Plain working-memory note. Keep docs ahead of memory.

## Corrected Terms

`Atari-style` means a LightZero-compatible visual env shape. It means stacked
image frames, discrete actions, conv model path, reward/done/info,
reset/seed, checkpoint/eval discipline, and clear run artifacts.

It does not mean literal ALE. ALE is only the Atari emulator for real Atari
ROMs, used by the official Atari Pong control lane.

## Current Priority

The official/control LightZero Pong lane is the control lane, not the project
goal. Read it survival-first: stock evaluator steps survived versus same-run
`iteration_0`, across later checkpoints and multiple eval starts. Return and
score are secondary context. The current positive signal is late stock-survival
lift in multiple normal Pong runs; this supports using the Modal/LightZero
pattern as a control while CurvyTron work proceeds, but it is not solved Pong
or a CurvyTron quality claim.

## Custom Dummy Pong

Custom dummy Pong stays bridge/debug only. Shaped reward plumbing works, but
policy quality is bad or collapsed. Do not use this lane as evidence that
LightZero can or cannot solve official Pong, and do not compare it to the
official/control Pong lane.

## CurvyTron Next Step

CurvyTron does not need ALE. The current CurvyTron path is a non-ALE
LightZero-compatible visual/survival lane with survival-only reward. Keep
adapter, trainer, and eval work labeled as CurvyTron plumbing or survival
signal, not Pong proof. Plan or maintain a LightZero adapter with:

- stacked image frames;
- discrete ego actions;
- `reward`, `done`, and `info`;
- deterministic `reset(seed)`;
- `action_mask`;
- `to_play=-1`;
- full joint-action logging for every environment tick.

The next useful CurvyTron tasks are checkpointed survival eval, frozen-opponent
bridge work, and then current-policy-versus-current-policy design. This is not
a new RL theory or framework-choice rabbit hole.

## Reporting Rule

Every run needs a plain claim and a plain non-claim near the top.

Use this wording:

```text
Claim: ...
Non-claim: ...
```

Example: "strict loading and eval ran" is a claim; "policy quality is proven"
is a non-claim unless the evidence actually proves it.
