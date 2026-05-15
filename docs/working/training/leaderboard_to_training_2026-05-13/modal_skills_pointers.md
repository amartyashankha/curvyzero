# Modal Skills Pointers

This is a map to the local Modal skills packet for the CurvyZero
training/tournament loop. Use it as an index; do not paste the packet into
working docs.

## Packet Roots

- `/Users/shankha/Downloads/skills-main/modal/SKILL.md`: smaller general Modal
  skill. Good first stop for CLI habits, docs pointers, app structure, function
  types, and development workflow.
- `/private/tmp/modal-auto-research-skills/README.md`: larger auto-research
  packet index. It splits Modal work into basic platform knowledge, interactive
  GPU development, training experiments, and parallel sub-agents.

## Most Useful Files

- `/private/tmp/modal-auto-research-skills/modal-basic-skills/SKILL.md`: base
  operating notes. Prefer `modal --help`, check `modal --version`, use JSON CLI
  output where available, and avoid guessing around changing SDK behavior.
- `/private/tmp/modal-auto-research-skills/modal-basic-skills/references/app-structure.md`:
  package multi-file apps and deploy with module mode (`modal deploy -m ...`).
  Keep global Modal app scope light because it runs both locally and inside
  remote containers.
- `/private/tmp/modal-auto-research-skills/modal-basic-skills/references/development-workflow.md`:
  `modal run` is for ephemeral development apps. Use `modal deploy` for
  concurrent fanout, function lookups, and shared resources like Modal Dicts and
  Queues. Use Modal environments when dev resources must not touch production.
- `/private/tmp/modal-auto-research-skills/modal-basic-skills/references/function-types.md`:
  use `@app.cls()` plus `@modal.enter()` for expensive per-container setup, and
  parameterization or `.with_options()` when each runtime configuration should
  get its own autoscaling pool.
- `/private/tmp/modal-auto-research-skills/modal-gpu-experiment/SKILL.md` and
  `references/training.md`: template for long training jobs: bake code into the
  image, mount volumes for data/checkpoints, add secrets, set generous timeouts,
  and pair retries with checkpoint auto-resume.
- `/private/tmp/modal-auto-research-skills/modal-gpu-experiment/references/volumes.md`:
  persistent storage patterns. Volume files are the durable truth for
  checkpoints, manifests, snapshots, and debug bundles; call `commit()` after
  remote writes that must survive.
- `/private/tmp/modal-auto-research-skills/modal-gpu-experiment/references/compute.md`:
  GPU menu and selection hints. Append `:N` for multi-GPU containers; use
  clustered multi-node only when a job truly needs more than one machine.
- `/private/tmp/modal-auto-research-skills/modal-gpu-dev/SKILL.md` and
  `references/development.md`: interactive GPU sandbox pattern for debugging or
  profiling. It is useful before converting a repro into a proper Modal app,
  but sandbox output should still be reduced to durable artifacts.
- `/private/tmp/modal-auto-research-skills/sub-agents/SKILL.md` and
  `references/sub-agents.md`: parallel research pattern. Deploy the sandbox or
  experiment app once, then call deployed functions many times; do not launch
  one `modal run` per agent.

## CurvyZero Takeaways

- For tournament/intake fanout, prefer one deployed Modal app plus many
  `.remote()` / `.spawn()` calls. Repeated concurrent `modal run` processes are
  the wrong proof surface for production-like scheduling.
- A non-detached `modal run` ephemeral app is not a safe parent for background
  tournament workers. If child game/rating calls must outlive the local command,
  use `modal run --detach`, use a deployed function that waits/keeps the work
  alive correctly, or make the parent wait for child completion.
- Treat Modal Volume paths as authoritative. Dicts and Queues are coordination
  surfaces, not truth; this packet mentions them for environment segregation but
  does not provide a dedicated Queue/Dict repair guide.
- For long training or rating jobs, require checkpoint/resume behavior before
  depending on retries. A retry without a durable checkpoint can repeat work or
  hide partial progress.
- For debugging live failures, use the existing debug-bundle habit first, then
  apply the packet docs to decide whether the issue is app packaging,
  deployment/fanout, storage, GPU allocation, or agent orchestration.

## Official Modal Docs Checked

- Job queue guide: deploy the worker, submit work with `Function.spawn()`, and
  poll by call id. This matches the desired shape for tournament jobs and
  website-triggered background work.
- Queue guide: Queues are good for active communication, but they are not the
  durable record. Our manifest-on-Volume repair path is the right direction.
- Volume guide: commits/reloads are explicit; background commits exist, but a
  reader only sees another container's commit after reload. Reload fails if the
  same container has open files on the Volume. Concurrent writes to the same
  files are last-writer-wins, and too many small commits can create contention.
- Web endpoint guide: deployed FastAPI/ASGI endpoints are persistent web
  surfaces. Cold starts and request-rate limits are normal, so the tournament
  website should keep the first page cheap and lazy-load heavy panels.

Plain conclusion:

- Use Queue/Dict to wake and coordinate.
- Use Volume JSON as truth.
- Write immutable per-round/per-shard files, then move one small pointer last.
- Do not make the website or proof depend on broad Volume reloads during active
  file reads.
- Prove production behavior from a deployed app, not from several overlapping
  ephemeral `modal run` calls.
