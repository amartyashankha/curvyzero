# Browser Golden Capture Side Lane

Date: 2026-05-13

Scope: side-lane feasibility note. This is not the optimizer parity oracle and
does not block training profiles.

## Plain Status

The actual browser/reference source is vendored at
`third_party/curvytron-reference`. It is an old Node 0.10-era Angular/WebSocket
app. The documented run path is:

```text
npm install
bower install
gulp
node bin/curvytron.js
```

The current checkout does not include generated `bin/curvytron.js` or
`web/js/curvytron.js`, so full-app browser capture requires a disposable old
Node/Gulp/Bower build probe first.

## Useful Capture Shape

The browser view uses stacked canvas layers:

- `background`
- `bonus`
- `game`
- `effect`

For the first parity proof, do not use a page screenshot. Use Playwright and
compose the canvas layers in `page.evaluate` into an offscreen canvas with
`drawImage`, then return `toDataURL("image/png")`.

For comparison against our CPU reference, the first useful composite is:

```text
background + bonus + game
```

Leave `effect` out unless explicitly testing death particles.

## State Control Problem

The full browser app is event-driven. It receives positions, trail points,
bonus pops, and other state through client repositories and WebSocket events.
There is no current deterministic source-state snapshot loader for arbitrary
training states.

A spectator resync gets current positions/properties/active bonuses, but not a
full historical trail polyline. So a full-app golden frame for a rich trail
state needs either event replay from the beginning or a new debug-only snapshot
loader.

## Minimal First Experiment

1. Build a tiny browser harness, not the whole multiplayer flow.
2. Create the four canvas elements at 704x704 with `deviceScaleFactor=1`.
3. Load `third_party/curvytron-reference/web/images/bonus.png`.
4. Draw one deterministic 2P state using the same browser drawing primitives.
5. Compose `background + bonus + game`.
6. Render the same state through
   `render_source_state_rgb_canvas_like(..., frame_size=704)`.
7. Save `browser.png`, `curvyzero.png`, `diff.png`, and `metrics.json` under
   `artifacts/local/browser_golden_probe/...`.

This proves browser-canvas capture mechanics without touching training.

## Modal + Playwright Pattern

Modal has an official web-scraper example that installs Playwright and Chromium
inside the Modal image, so the local machine does not need a browser install:
https://modal.com/docs/examples/webscraper.
Modal's image guide is the right reference for adding Python/system packages and
shipping local files into an image:
https://modal.com/docs/guide/images.

The example shape is:

```python
playwright_image = modal.Image.debian_slim(python_version="3.10").run_commands(
    "apt-get update",
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "pip install playwright==1.42.0",
    "playwright install-deps chromium",
    "playwright install chromium",
)
```

Then a Modal function can use:

```python
from playwright.async_api import async_playwright

async with async_playwright() as p:
    browser = await p.chromium.launch()
    page = await browser.new_page(viewport={"width": 704, "height": 704}, device_scale_factor=1)
```

For the CurvyTron golden probe, the Modal function should:

1. build/load a tiny static HTML canvas harness;
2. load `bonus.png`;
3. draw the deterministic frame;
4. run `page.evaluate(...)` to compose canvas layers and return PNG bytes or a
   data URL;
5. write artifacts only to a local artifact folder or explicit non-training
   Volume path.

Do not run this inside training jobs. Do not require the old CurvyTron full app
until the tiny harness works.

Prefer a capture-only localhost page inside the Modal function, not an external
web endpoint. Modal `@web_server` is only needed if we later want a visible
service:
https://modal.com/docs/guide/webhooks.

Use a separate Volume such as `curvyzero-browser-goldens`, not the training runs
Volume. Modal Volume writes need commit/reload awareness:
https://modal.com/docs/guide/volumes.

Playwright mechanics:

- use `page.evaluate()` to run browser-side JS and compose canvas layers:
  https://playwright.dev/python/docs/evaluating;
- use screenshot APIs only for debug; screenshots include CSS/layout concerns:
  https://playwright.dev/python/docs/screenshots.

Important gotchas:

- Serve `bonus.png` from the same local origin as the HTML; cross-origin images
  can taint the canvas and block `toDataURL()`.
- Pin viewport and `device_scale_factor=1`.
- Freeze or omit animations at first.
- Omit the `effect` layer for the first proof.
- Keep artifact count tiny: `browser.png`, `manifest.json`, and maybe
  `diff.png` after offline comparison.

## Blockers

- Old Node/Gulp/Bower stack.
- Missing generated JS/server artifacts.
- No deterministic snapshot/replay hook.
- Full app depends on rooms/WebSockets/lifecycle.
- Pixel exactness may vary by browser/backend, so start with metrics and saved
  diffs before declaring exact parity.

## Recommendation

Keep this as a side lane. It is valuable for Environment Reconstruction and
human confidence, but Optimizer should use CPU-reference parity for dirty-cache
and GPU renderer promotion.
