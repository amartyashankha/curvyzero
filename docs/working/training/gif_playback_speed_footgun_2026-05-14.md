# GIF Playback Speed Footgun - 2026-05-14

- Browser pages serve GIFs as plain `image/gif` bytes through `<img>` tags. There is no reliable web playback-rate control for an already-encoded GIF.
- Playback speed is encoded when the artifact is written. The central defaults are now `80.0` fps for both checkpoint selfplay GIFs and tournament arena GIF samples, with a 10 ms encoder floor so the 80 fps request is not capped to 50 fps.
- Existing `raw.gif`, `collect_t1.gif`, and tournament `game.gif` artifacts keep their old frame delays until regenerated or re-encoded from saved frame NPZ artifacts.
- Explicit launch/spec values such as `background_gif_fps=8.0` or `gif_fps=8.0` still override the faster defaults. Check generated manifests before assuming new runs inherited the default.
