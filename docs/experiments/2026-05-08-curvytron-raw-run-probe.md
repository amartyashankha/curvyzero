# 2026-05-08 curvytron-raw-run-probe

## Question

Can we run the raw CurvyTron reference clone locally right now?

## Setup

Working tree: `/Users/shankha/curvy`

Reference clone: `third_party/curvytron-reference`

Current local tools:

```text
node --version -> v25.9.0
npm --version  -> 11.12.1
```

## Command

Small commands only. No dependency install was run in the reference clone.

Important commands:

```sh
find web -maxdepth 3 -type f
find bin -maxdepth 2 -type f
npm ls --depth=0
node bin/curvytron.js
node src/server/launcher.js
npm install --dry-run --ignore-scripts
command -v bower
command -v gulp
```

## Results

`web` only has images, sounds, and fonts. It does not have generated `index.html`, JS, or CSS.

`bin` does not exist:

```text
find: bin: No such file or directory
```

Dependencies are not installed:

```text
npm error code ELSPROBLEMS
UNMET DEPENDENCY express@^4.13.3
UNMET DEPENDENCY faye-websocket@^0.10.0
UNMET DEPENDENCY gulp@^3.8.*
UNMET DEPENDENCY influx@^4.0.1
```

The documented run command fails because the generated server file is missing:

```text
Error: Cannot find module '/Users/shankha/curvy/third_party/curvytron-reference/bin/curvytron.js'
```

The source launcher is not directly runnable:

```text
Error: Cannot find module '../package.json'
Require stack:
- /Users/shankha/curvy/third_party/curvytron-reference/src/server/launcher.js
```

The networked npm dry-run resolved packages after approval:

```text
add gulp 3.9.1
add gulp-sass 0.7.3
add node-sass 0.9.6
add faye-websocket 0.10.0
add express 4.22.1
added 729 packages in 8s
```

This was only a dry-run. It did not create `node_modules`.

`bower` and `gulp` are not installed globally:

```text
command -v bower -> no output, exit 1
command -v gulp  -> no output, exit 1
```

## Interpretation

We cannot run it now.

The raw command, after a successful install and build, should be:

```sh
node bin/curvytron.js
```

Then visit:

```text
http://localhost:8080/
```

The default game does not appear to need Mongo or Redis. It uses Node, Express static files, and WebSockets. InfluxDB is only for the optional inspector, which is disabled by default.

The main risk is old Node tooling. The project uses Gulp 3, Bower, and `gulp-sass@0.7.3` / `node-sass@0.9.6`. That stack is likely to break on Node 25 even though npm can resolve the package list.

## Artifacts

Detailed notes:

- `docs/research/environment/curvytron_raw_run_probe.md`

## Follow-ups

Smallest next experiment: copy the reference clone to `/private/tmp`, install/build there, and record the first real failure. Prefer an old Node runtime for that experiment if one is available.
