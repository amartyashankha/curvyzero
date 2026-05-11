# Modal Reference Hosting Plan

Status: Draft

## Recommendation

Do not start with the public CurvyTron web server as the main reference path.

Start with a headless Node probe on Modal. Use it to boot the old server code or
the bundled game model inside one Modal Function, drive fixed source control
values, and save JSON artifacts. Add Playwright and the public web endpoint
later as a browser smoke, not as the first fidelity gate.

This keeps the work simple:

- Modal launches a coarse reference job.
- The job runs Node and Python locally inside the container.
- The job writes artifacts once per run or shard.
- No Modal Function, Queue, Dict, or web call sits inside the physics loop.

## Why This Shape

The old CurvyTron app is a real-time web game, not a test harness.

- It needs an old Node/Gulp/Bower build.
- The local snapshot does not include `bin/curvytron.js`, `web/index.html`,
  built client JS, or built CSS.
- The server keeps room and game state in one Node process.
- WebSockets can work on Modal, but auto-scaling can split rooms across
  containers unless the endpoint is kept to one container.
- Browser screenshots are useful, but state JSON and websocket logs are easier
  to compare to Python.

## Modal Pieces

Use one Modal app for reference fidelity work, for example
`curvyzero-reference`.

Use two entry points:

- `reference_probe`: a normal Modal Function that runs a batch of scripted cases.
- `reference_web_smoke`: an optional `@modal.web_server` endpoint for manual
  browser and Playwright checks.

Use a Modal Volume for artifacts:

```text
/artifacts/reference/<run_id>/
  manifest.json
  cases/
    <case_id>/
      input.json
      node_state.jsonl
      node_ws.jsonl
      python_state.jsonl
      compare.json
      screenshots/
      videos/
```

Write immutable case folders. Publish `manifest.json` only after payload files
exist. Commit the Volume after each completed case or small batch.

## Node Image

Use a separate image from the main Python simulator image.

Preferred first build attempt:

```python
reference_image = (
    modal.Image.from_registry("node:8-buster", add_python="3.11")
    .apt_install("git")
    .run_commands("npm install -g bower gulp@3")
    .add_local_dir("third_party/curvytron-reference", "/ref/curvytron", copy=True)
    .workdir("/ref/curvytron")
    .run_commands("npm install")
    .run_commands("gulp")
    .pip_install("numpy>=1.26")
    .add_local_dir("src", "/repo/src", copy=True)
    .env({"PYTHONPATH": "/repo/src"})
)
```

Treat this as a probe, not a promise. If `gulp-sass` or `node-gyp` fails, pin an
older Node image or build only the server bundle needed by the headless probe.
Do not modernize the old app until a small Modal build log shows exactly what is
broken.

## Web Server Option

If the raw web server is needed, expose it with `@modal.web_server(8080)`.

The server already uses one HTTP server for both static files and websocket
upgrades. It serves `web/` and listens on the configured port, defaulting to
`8080`. The client opens `ws://` or `wss://` to the same host and path.

Sketch:

```python
@app.function(image=reference_image, max_containers=1, scaledown_window=10 * 60)
@modal.concurrent(max_inputs=50)
@modal.web_server(8080, label="curvytron-reference")
def reference_web_smoke():
    import subprocess

    subprocess.Popen(["node", "bin/curvytron.js"], cwd="/ref/curvytron")
```

Keep `max_containers=1` for reference use. CurvyTron room state lives in memory,
so multiple containers would create separate room worlds.

## Observation Capture

Use these in order:

1. State JSON from Node.
   - Best first signal.
   - Capture tick, elapsed ms, room, round, players, avatar position, angle,
     alive flag, score, trail point count, bonuses, and emitted events.
   - Save as JSONL.

2. Websocket messages.
   - Useful to prove the browser protocol and compression.
   - Record inbound and outbound websocket arrays.
   - Decode compressed positions in the comparer, but keep raw messages too.

3. Playwright screenshots.
   - Useful for visual smoke and debugging.
   - Capture only a few fixed moments per case.

4. Saved videos.
   - Useful for failures and demos.
   - Keep them short and sampled. Do not write videos for every training case.

## Comparison Job

A Modal reference job should compare at case boundaries, not per physics tick
over Modal.

For each case:

1. Load a fixed seed, room config, and action trace from `input.json`.
2. Run the Node reference locally in the Modal container and write
   `node_state.jsonl`.
3. Run the Python simulator locally in the same container, or in a separate
   coarse Modal Function over the same case batch, and write
   `python_state.jsonl`.
4. Compare files locally:
   - death tick and killer
   - score events
   - round start/end events
   - position and angle tolerance by tick
   - trail point counts and bonus events
   - websocket event names and payload shape
5. Write one compact `compare.json`.

The Python physics loop only reads local action arrays and local reference files.
It never calls a Modal endpoint during `step()`.

## First Three Spikes

1. Modal Node build probe.
   - Build old dependencies.
   - Run `gulp`.
   - Confirm `bin/curvytron.js`, `web/index.html`, `web/js/curvytron.js`, and
     `web/css/style.css` exist.

2. Headless websocket probe.
   - Start `node bin/curvytron.js` on localhost inside one Modal Function.
   - Use a small Node websocket client to create a room, add two players, mark
     them ready, send fixed moves, and record messages for 5 seconds.

3. State JSON probe.
   - Add a temporary probe script that serializes server-side game state at a
     fixed cadence.
   - Compare one case to the Python environment offline in the same Modal job.

Promote `reference_web_smoke` only after these pass.

## Sources

- Local CurvyTron source: `third_party/curvytron-reference`.
- Local Modal smoke: `src/curvyzero/infra/modal/smoke.py`.
- Existing Modal policy: `docs/decisions/0002-modal-hot-loop-locality.md`.
- Modal web endpoints: https://modal.com/docs/guide/webhooks
- Modal web server reference: https://modal.com/docs/reference/modal.web_server
- Modal image reference: https://modal.com/docs/reference/modal.Image
- Modal Volumes: https://modal.com/docs/guide/volumes
