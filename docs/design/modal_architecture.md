# Modal Architecture

Status: Draft

Modal should make compute easy without putting networked primitives inside the simulation or MCTS hot loop.

## Initial Shape

- Local development runs tests, small benchmarks, and debug renderers.
- Modal CPU jobs run environment benchmarks and non-GPU smoke tests from the beginning.
- Modal GPU jobs run JAX/PyTorch device smokes, MCTS benchmarks, training, and evaluation.
- Storage is chunked: checkpoints, replay, logs, and videos should not create millions of tiny files.

## Code Location

Modal entry points live under `src/curvyzero/infra/modal/`. The first app is `curvyzero.infra.modal.smoke`, which runs remote tests, environment benchmarks, and a minimal GPU visibility smoke.

## Hot-Loop Rule

Environment ticks, MCTS node expansion, model inference batches, and action selection should live inside one process/container whenever possible. Modal Queues, Dicts, and cross-function calls are coordination tools, not per-step tools.

## Primitives To Investigate

- `modal.App` and `modal.Function` for jobs.
- `modal.Image` for pinned runtime environments.
- `modal.Volume` for checkpoints and medium-lived artifacts.
- `modal.CloudBucketMount` for long-term replay and large archives.
- `modal.Queue` for coarse work dispatch.
- `modal.Dict` for tiny metadata such as latest checkpoint pointers.
- `modal.Secret` for API keys and external storage credentials.
- Sandboxes and snapshots for debugging or reproducible exploratory workers.
- Memory snapshots for cold-start-heavy deployed functions, only after checking randomness and GPU limitations.

## Open Questions

- Which artifact layout should be standard across local and Modal?
- Should replay first live on Modal Volume or cloud bucket?
- How should long-running training resume after preemption?
- Which GPU types are cost-effective for early MCTS benchmarks?
