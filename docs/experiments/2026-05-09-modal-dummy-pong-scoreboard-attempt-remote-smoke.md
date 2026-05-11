# 2026-05-09 Modal dummy Pong scoreboard attempt remote smoke

## Question

Can the dummy Pong checkpoint scoreboard run as one CPU Modal job, read
checkpoints from the `curvyzero-runs` Volume, and write eval artifacts back to
the same Volume?

## Setup

- Modal app: `curvyzero-dummy-pong-scoreboard-attempt`
- Volume: `curvyzero-runs`
- Run id: `modal-pong-scoreboard-smoke-20260509`
- Attempt id: `attempt-000001`
- Eval id: `checkpoint-scoreboard`
- Split: `dummy_pong_modal_smoke_v0`
- Split role: `monitor`
- Episodes per seated matchup: `2`
- Checkpoints uploaded to the Volume:
  - latest:
    `training/dummy-pong/manual-checkpoints/imitation-v0-e1000/epoch-001000/checkpoint.npz`
  - previous:
    `training/dummy-pong/manual-checkpoints/imitation-v0-e1000/epoch-000250/checkpoint.npz`
- No pytest.

## Command

```sh
modal volume put curvyzero-runs \
  artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-000250/checkpoint.npz \
  training/dummy-pong/manual-checkpoints/imitation-v0-e1000/epoch-000250/checkpoint.npz
```

```sh
modal volume put curvyzero-runs \
  artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz \
  training/dummy-pong/manual-checkpoints/imitation-v0-e1000/epoch-001000/checkpoint.npz
```

The first launch with plain `modal run` failed before remote execution because
the local CLI process could not import the project package. The working command
uses the repo environment:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --checkpoints latest=ref:training/dummy-pong/manual-checkpoints/imitation-v0-e1000/epoch-001000/checkpoint.npz,previous=ref:training/dummy-pong/manual-checkpoints/imitation-v0-e1000/epoch-000250/checkpoint.npz \
  --episodes 2 \
  --seed 19 \
  --split-id dummy_pong_modal_smoke_v0 \
  --split-role monitor \
  --run-id modal-pong-scoreboard-smoke-20260509 \
  --attempt-id attempt-000001
```

Downloaded copy:

```sh
modal volume get curvyzero-runs \
  training/dummy-pong/modal-pong-scoreboard-smoke-20260509/attempts/attempt-000001/eval/checkpoint-scoreboard \
  artifacts/local/modal-dummy-pong-scoreboard-attempt-remote-smoke-2026-05-09 \
  --force
```

## Results

- The remote Modal job completed and committed the Volume.
- Remote elapsed time was about `1.92` seconds after image setup.
- The Function read both checkpoints from Volume refs.
- Returned refs:
  - `training/dummy-pong/modal-pong-scoreboard-smoke-20260509/run.json`
  - `training/dummy-pong/modal-pong-scoreboard-smoke-20260509/latest_attempt.json`
  - `training/dummy-pong/modal-pong-scoreboard-smoke-20260509/attempts/attempt-000001/attempt.json`
  - `training/dummy-pong/modal-pong-scoreboard-smoke-20260509/attempts/attempt-000001/eval/checkpoint-scoreboard/summary.json`
  - `training/dummy-pong/modal-pong-scoreboard-smoke-20260509/attempts/attempt-000001/eval/checkpoint-scoreboard/episodes.jsonl`
- Scoreboard smoke rows:
  - `track_ball` beat `random_uniform` 4/4.
  - `track_ball` versus `track_ball` truncated 2/2.
  - latest beat random 3/4.
  - latest versus `track_ball` truncated 4/4.
  - previous lost to random 0/4 and lost to `track_ball` 0/4.
  - latest versus previous tied 2/4 to 2/4.

## Interpretation

Pong now has a real CPU Modal scoreboard path with durable Volume artifacts.
This proves remote eval and artifact plumbing, not policy quality.

The next Modal step is not GPU training. After the repair-vs-baseline decision
chooses a learner path, make Pong training attempts write checkpoints into the
same Volume layout directly.

## Artifacts

Volume refs:

- `training/dummy-pong/manual-checkpoints/imitation-v0-e1000/epoch-000250/checkpoint.npz`
- `training/dummy-pong/manual-checkpoints/imitation-v0-e1000/epoch-001000/checkpoint.npz`
- `training/dummy-pong/modal-pong-scoreboard-smoke-20260509/attempts/attempt-000001/eval/checkpoint-scoreboard/summary.json`
- `training/dummy-pong/modal-pong-scoreboard-smoke-20260509/attempts/attempt-000001/eval/checkpoint-scoreboard/episodes.jsonl`

Local downloaded copy:

- `artifacts/local/modal-dummy-pong-scoreboard-attempt-remote-smoke-2026-05-09/checkpoint-scoreboard/summary.json`
- `artifacts/local/modal-dummy-pong-scoreboard-attempt-remote-smoke-2026-05-09/checkpoint-scoreboard/episodes.jsonl`

## Follow-ups

- Add a Pong Modal training-attempt wrapper only after the repair-vs-baseline
  decision chooses a learner path worth running.
- Keep checkpoint refs in `curvyzero-runs`; do not rely on local artifact paths
  for remote eval.
