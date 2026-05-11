# Modal CurvyTron Hosting Research

Date: 2026-05-08

## Short Answer

Hosting the old CurvyTron server on Modal looks feasible, but it is not the
best first step for fidelity work.

The better first step is a headless Node probe on Modal. It can run the old game
logic or the old server in one container, record state and websocket messages,
and compare those files to Python output. A public Modal web endpoint should
come after that as a browser smoke.

## Local CurvyTron Facts

Files inspected:

- `third_party/curvytron-reference/doc/installation.md`
- `third_party/curvytron-reference/package.json`
- `third_party/curvytron-reference/gulpfile.js`
- `third_party/curvytron-reference/recipes/server.json`
- `third_party/curvytron-reference/src/server/launcher.js`
- `third_party/curvytron-reference/src/server/core/Server.js`
- `third_party/curvytron-reference/src/shared/core/BaseSocketClient.js`
- `third_party/curvytron-reference/src/client/core/SocketClient.js`
- `third_party/curvytron-reference/src/server/controller/RoomsController.js`
- `third_party/curvytron-reference/src/server/controller/GameController.js`
- `third_party/curvytron-reference/src/shared/model/BaseGame.js`

Observed facts:

- The docs say CurvyTron runs on Node `>= v0.10`.
- Local machine has Node `v25.9.0` and npm `11.12.1`, but I did not run
  `npm install` because it would write dependency folders outside the requested
  docs-only scope.
- `package.json` has no `engines` pin and no lockfile.
- Runtime dependencies are `express`, `faye-websocket`, and `influx`.
- Build dependencies are old Gulp 3 era packages, including `gulp-sass` 0.7.
- The install script runs `bower install`, but `bower` is not listed as a local
  dependency. The old install docs also ask for `bower install`.
- The local snapshot has source files and static asset files, but no built
  `bin/curvytron.js`, no `web/index.html`, no built `web/js`, and no built
  `web/css`.
- `gulp` builds the server bundle to `bin/curvytron.js`.
- `gulp` builds browser files under `web/`, including bundled JS, views, and CSS.
- The launcher reads `config.json` if present, otherwise uses port `8080`.
- The server uses Express static hosting for `web/`.
- The same HTTP server handles websocket upgrades.
- The browser opens a websocket to the current host and current path, using
  `ws://` or `wss://`.
- Socket messages are JSON arrays of events. Some events include callback ids.
- Game state is pushed through websocket events such as `position`, `angle`,
  `die`, `score`, `round:new`, `round:end`, `bonus:pop`, `bonus:clear`,
  `game:start`, and `game:stop`.
- The game loop targets 60 Hz with wall-clock elapsed milliseconds as the step.

## Modal Facts Checked

Official Modal docs inspected on 2026-05-08:

- Web endpoints: https://modal.com/docs/guide/webhooks
- Web server decorator: https://modal.com/docs/reference/modal.web_server
- Web endpoint URLs: https://modal.com/docs/guide/webhook-urls
- Images: https://modal.com/docs/guide/images
- Image reference: https://modal.com/docs/reference/modal.Image
- Volumes: https://modal.com/docs/guide/volumes
- Timeouts: https://modal.com/docs/guide/timeouts

Relevant Modal facts:

- `@modal.web_server` can expose a server process that listens on a container
  port.
- Modal says web servers should bind to `0.0.0.0`, not only localhost.
- `@modal.web_server`, `@modal.asgi_app`, and `@modal.wsgi_app` support
  WebSockets.
- Modal WebSockets do not support `permessage-deflate`, and messages can be up
  to 2 MiB.
- A web endpoint gets a URL when served or deployed. Deployed URLs are stable
  enough for jobs and manual testing.
- Modal Images can be built from registries such as Docker Hub, and
  `add_python` can add Python if the base image lacks it.
- Volumes need commit/reload care. They are good for write-once/read-many
  artifacts, but not for many tiny files or concurrent writes to the same file.
- Modal Functions default to 5 minute timeouts and can be configured up to 24
  hours.

Local Modal examples inspected:

- `src/curvyzero/infra/modal/smoke.py`
- `docs/research/modal_patterns.md`
- `docs/research/modal_example_patterns.md`
- `docs/decisions/0002-modal-hot-loop-locality.md`
- `/Users/shankha/modal-examples/07_web_endpoints/basic_web.py`
- `/Users/shankha/modal-examples/07_web_endpoints/http_server.py`
- `/Users/shankha/modal-examples/07_web_endpoints/http_server_sticky.py`
- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/main.py`
- `/Users/shankha/modal-projects/claude-slack-gif-creator/src/runtime_common.py`

The local examples reinforce the same pattern: use Modal for coarse jobs,
web endpoints, and artifact storage. Keep tight loops inside one container.

## Hosting Feasibility

Raw server hosting should work if the old Node build works.

The likely Modal shape is:

- Build a Node image.
- Install `bower` and old `gulp`.
- Copy `third_party/curvytron-reference`.
- Run `npm install`.
- Run `gulp`.
- Start `node bin/curvytron.js`.
- Expose port `8080` with `@modal.web_server(8080)`.
- Add `@modal.concurrent` for multiple websocket connections.
- Cap the endpoint to one container for reference use.

Main risks:

- Modern Node/npm may not build old Gulp 3, Bower, and `gulp-sass`.
- Old npm packages may need Python 2 or old `node-gyp`.
- Public web hosting tests browser behavior and transport, but it does not give
  the cleanest state trace for simulator fidelity.
- Modal autoscaling can create multiple Node processes. That is good for stateless
  services, but bad for one in-memory room world.
- The old game uses wall-clock time and `Math.random`, so exact repeatability
  needs a probe script or instrumentation.

## Better First Step

Run a headless Node probe first.

Good probe goals:

- Prove the old code can build on Modal.
- Start the server on localhost or instantiate game objects directly.
- Drive a tiny scripted game with fixed actions.
- Save JSONL state and websocket messages.
- Run the Python simulator over the same action trace.
- Write one comparison file.

This gives useful evidence before spending time on public endpoint polish.

## Observation Options

State JSON:

- Best first artifact.
- Record tick, step milliseconds, room config, avatar x/y/angle, alive flag,
  score, round score, trail point count, bonus list, and game events.

Websocket messages:

- Best transport artifact.
- Record raw inbound and outbound event arrays.
- Keep compressed values raw, then decode in the comparer.

Playwright screenshots:

- Best browser artifact.
- Use only after the server builds and scripted rooms work.
- Capture fixed moments such as home page, room page, first round start, mid
  round, and death.

Videos:

- Best human-debug artifact.
- Save short videos only for failures, demos, and sampled cases.

## Python Comparison Shape

Do not call Modal from inside `CurvyTronEnv.step()`.

Use Modal like this:

1. A local or Modal launcher creates a batch of cases.
2. One Modal Function gets a whole case batch.
3. Inside that container, Node produces reference JSONL.
4. Inside that container, Python produces simulator JSONL.
5. A local Python comparer reads both files and writes `compare.json`.
6. The Modal Function returns only a compact summary and artifact paths.

That keeps Modal at the job boundary. The physics loop stays local and fast.

## Concrete Recommendation

Build this in order:

1. `reference_node_build_probe`.
   - Goal: prove the old app can run `npm install` and `gulp` on Modal.
   - Output: build log plus a file existence check for `bin/curvytron.js`,
     `web/index.html`, `web/js/curvytron.js`, and `web/css/style.css`.

2. `reference_headless_probe`.
   - Goal: run one tiny scripted game without a browser.
   - Output: `input.json`, `node_state.jsonl`, `node_ws.jsonl`, and a summary.

3. `reference_compare_probe`.
   - Goal: run the same case through Python and compare outcomes.
   - Output: `python_state.jsonl` and `compare.json`.

4. `reference_web_smoke`.
   - Goal: expose the real app with `@modal.web_server` and run a Playwright
     screenshot smoke.
   - Output: screenshots, short video only on failure, and websocket transcript.

Modal is useful here, but the raw web game should be a witness, not the hot
runtime.
