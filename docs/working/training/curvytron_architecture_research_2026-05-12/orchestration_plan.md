# Orchestration Plan

Purpose: keep the investigation split into clean lanes while the main thread
does synthesis and decisions.

## Active Lanes

| Lane | Owner | Output | Status |
| --- | --- | --- | --- |
| Stock frozen-opponent canary | Anscombe notes | tiny CPU `train_muzero` proof with strict checkpoint opponent | passed: `stock-frozen-canary-source-state-s304-20260512` |
| GPU stock frozen-opponent canary | Darwin notes | same proof with GPU learner and `env_manager_type=base` | passed: `stock-frozen-gpu-base-canary-source-state-s304-20260512b` |
| Native replay bridge parity | Pasteur / Helmholtz / Ohm / Turing | prove two seat-local `GameSegment`s can push/sample native LightZero targets | passed for tiny hand-authored trace in Modal/LightZero |
| Stock LightZero dataflow | Hubble | exact dataflow and custom-path seam map | integrated |
| Current path discrepancy | Harvey/Pascal notes | exact comparison of fixed/frozen, turn-commit, joint-action, and two-seat code paths | first pass integrated |
| History and paper trail | Dalton plus old notes | timeline of how fixed/frozen, turn-commit, and two-seat were promoted/demoted | deeper pass integrated |
| Cleanup targets | Rawls plus main thread | stale docs/defaults/scripts that still point to wrong lane | first guardrails patched; more cleanup later |
| Pong analogy | Dalton plus Boole notes | how custom Pong attempts failed/inconclusive and stock LightZero Pong showed signal | deeper pass integrated |
| Literature / pitfalls | Ramanujan plus Schrodinger notes | narrow checklist of MuZero/RL pitfalls mapped to our failure | source-backed pass integrated |

## Main-Thread Jobs

1. Keep the high-level worldview simple and current.
2. Merge returned findings into the working docs.
3. Promote stable conclusions into design docs and the Coach index.
4. Avoid launching more large runs until the next gate is explicit.
5. Keep [open_questions_and_hypotheses.md](open_questions_and_hypotheses.md)
   and [known_wrong.md](known_wrong.md) current so the investigation does not
   lose the plot.

## Investigation DAG

```text
history + code path matrix
        -> postmortem facts
        -> stock-loop contract
        -> learning gates
        -> cleanup edits
        -> next small proof runs

stock LightZero dataflow
        -> stock compatibility contract
        -> native replay bridge contract
        -> target/replay parity tests

frozen/recent opponent route
        -> stock-loop practical training plan
        -> opponent refresh/eval panel requirements

Pong history analogy
        -> repeatable lesson from custom Pong -> stock LightZero Pong
        -> CurvyTron cleanup and next-gate checks

literature/pitfalls
        -> local checklist only
        -> architecture questions and learning gates
```

## Near-Term Stopping Condition

This research phase is done when we have:

- one page that says what went wrong;
- one page that says which path is trusted for which claim;
- one page that says how stock LightZero data flows;
- one cleanup list for stale code/docs;
- a short list of next proof runs or tests.

Current remaining blockers:

- one small learning curve on a stock route, after canary plumbing is stable.
- native bridge integration into an actual two-seat trainer if we choose that
  route later.

## Subagent Output Protocol

Every subagent should keep output narrow:

- state whether files were edited;
- write or propose one target doc when possible;
- include exact file/line refs for repo claims;
- avoid broad recommendations unless tied to evidence;
- separate facts, hypotheses, and next checks;
- avoid calling a path "self-play" without naming the opponent source and
  action semantics.

Preferred landing docs:

- history: `history_timeline.md`;
- path comparisons: `path_matrix.md`;
- stock dataflow: `stock_lightzero_dataflow.md`;
- frozen/recent route: `frozen_recent_opponent_route.md`;
- known mistakes: `known_wrong.md`;
- open questions: `open_questions_and_hypotheses.md`;
- cleanup: `cleanup_targets.md`;
- external research: `muzero_training_pitfalls_literature.md`.
