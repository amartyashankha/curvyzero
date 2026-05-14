# Downsample And Reference Fidelity

Date: 2026-05-13

Scope: optimizer-owned visual fidelity guardrails for the current CurvyTron
training observation.

## Plain Status

The trusted renderer for optimizer work is the CPU reference path, not a real
browser canvas:

1. render source-state RGB at 704x704 with `browser_lines`;
2. draw browser-sprite bonuses;
3. draw live heads;
4. convert RGB to BT.601 luma;
5. average each exact 11x11 source block into one 64x64 grayscale pixel.

Optimized dirty-cache or GPU renderers must match that CPU reference exactly
for the surface they replace. If an optimized path outputs gray64, require
byte-exact gray64. If it keeps or claims RGB cache state, require byte-exact RGB
cache state too.

Browser-pixel exactness is a separate Environment Reconstruction proof. It is
useful later, but it is not the optimizer parity oracle.

Near-term LightZero constraint: the current stock path feeds a local
four-frame grayscale stack to a conv MuZero model. In code this is
`STACKED_SOURCE_STATE_GRAY64_SHAPE = (4, 64, 64)`, and the Modal launcher pins
`policy.model.image_channel = 4`, `policy.model.frame_stack_num = 1`, and
`policy.model.observation_shape = env_spec["observation_shape"]`. So the
overnight-safe path is `[4,64,64]` gray. RGB or semantic planes are possible
research/canary paths, but they are not “just a grayscale tweak”; they change
the model/input contract and need their own smoke test.

## Current Code Facts

- The module explicitly says the source-state renderer is weaker than browser
  pixel parity: `src/curvyzero/env/vector_visual_observation.py:1`.
- The RGB reference frame size is 704:
  `src/curvyzero/env/vector_visual_observation.py:55`.
- The model path renders RGB first, then gray64:
  `src/curvyzero/env/vector_visual_observation.py:570`.
- The stock wrapper observation shape is `(4,64,64)`:
  `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:185`.
- The launcher configures LightZero as a conv model with `image_channel=4` and
  `frame_stack_num=1`:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4520`.
- The actual downsample is exact integer local mean after luma:
  `src/curvyzero/env/vector_visual_observation.py:990`.
- The schema records `downsample_method="integer_area_average_after_luma"`:
  `tests/test_vector_visual_observation.py:507`.
- Dirty-cache parity already has focused gray64 tests and a few RGB-cache tests:
  `tests/test_curvytron_two_seat_render_mode.py:405`,
  `tests/test_curvytron_two_seat_render_mode.py:532`.

## Literature/Library Read

The current downsample is the right family of operation. OpenCV describes
`INTER_AREA` as area-relation resampling and a preferred image decimation method:
https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html.

scikit-image says downsampling should anti-alias, and for integer factors
`downscale_local_mean` averages local blocks:
https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html.
Its API docs describe local-mean resize as area-based/pixel-mixing and say
integer factors should prefer local mean:
https://scikit-image.org/docs/stable/api/skimage.transform.html.

The Atari/DQN visual precedent is also grayscale plus downsample plus frame
stacking. Mnih et al. describe extracting luminance and resizing to 84x84 before
stacking recent frames:
https://faculty.washington.edu/minster/bio_inspired_robotics/research_articles/mnih_atari_deep_reinforcement_learning_nature2015.pdf.
Stable Baselines3's Atari wrapper follows the same broad shape: no-op reset,
frame skip, max-pool recent observations, resize to 84x84, grayscale, optional
sticky actions, and reward clipping:
https://stable-baselines3.readthedocs.io/en/v2.3.2/common/atari_wrappers.html.
LightZero's config docs expose the same relevant knobs directly:
`observation_shape`, `frame_stack_num`, `gray_scale`, `warp_frame`,
`image_channel`, and `downsample`:
https://opendilab.github.io/LightZero/tutorials/config/config.html.
MuZero is relevant because it uses visual Atari observations with planning, but
it does not by itself settle our renderer fidelity contract:
https://arxiv.org/abs/1911.08265.

## What The Current Downsample Preserves

The good part: this is not center sampling. Every source pixel in an 11x11 block
contributes to the final grayscale value. Small objects that cross a target-cell
boundary smear into neighboring cells instead of disappearing due to an arbitrary
sample point. That is what we want for a neural net.

The alignment is also simple: 704 is exactly `64 * 11`, so there are no
fractional block boundaries.

For linear luma, averaging after luma is effectively the same information as
averaging RGB then applying luma, apart from the final rounding. So the main
question is not the math order. The main question is whether one grayscale
channel has enough contrast for the game facts.

## Concrete Contrast Audit

BT.601 luma values for the current palette:

| item | RGB | luma | delta from background |
| --- | --- | ---: | ---: |
| background | `(34,34,34)` | 34 | 0 |
| self/red | `(255,0,0)` | 76 | 42 |
| other/green | `(0,255,0)` | 150 | 116 |
| blue | `(0,80,255)` | 76 | 42 |
| yellow | `(255,240,0)` | 217 | 183 |
| white | `(255,255,255)` | 255 | 221 |

Plain read:

- Current 2P player-perspective input is okay: self is red-like and other is
  green-like, so the model sees a large luma gap.
- Future 3P/4P grayscale is not okay by default: red and blue have almost the
  same luma. A one-channel grayscale observation cannot distinguish them by
  color alone.
- A single red or blue source pixel inside an 11x11 block can round back to the
  background value. Two red/blue pixels are enough to create a one-count signal.
  Real heads, trails, and sprites should cover far more than one source pixel,
  but this is exactly why center-sampling or too-thin approximations are risky.

## Real Risks

1. **Color collapse.** Single-channel grayscale cannot preserve arbitrary color
   identity. The current 2P self/other palette avoids the worst case, but 3P/4P
   would need a better palette or extra channels.

2. **Tiny feature dilution.** Area average preserves subcell evidence, but a
   very tiny low-contrast feature can be rounded away. This is unlikely for
   normal CurvyTron trails/heads, but it should be measured for small sprites,
   thin trails, and edge cases.

3. **Bonus identity.** Browser sprites are visible in gray64, but we have not
   proven that all 12 bonus types remain separable after grayscale and 11x11
   averaging. If exact bonus type matters, a diagnostic/auxiliary channel is
   cleaner than clever grayscale hacks.

4. **Optimized path shortcuts.** Direct-to-64 center sampling is not acceptable
   for the trusted path. Earlier toy probes showed center sampling mismatched
   the CPU reference in most sprite placements. Luma-space sprite alpha blending
   is also not byte-exact because current sprite blending happens in RGB before
   luma.

5. **Reference drift.** The environment is still changing. A downsample or
   parity result must name the code state and renderer semantics.

## Tests We Should Add Or Keep

- Coverage response test: place 1, 2, 3, ... source pixels of each player color
  inside one 11x11 block and record when the gray64 value changes.
- Minimum visibility sweep: render actual trail/head radii across many phases
  and require a stable multi-pixel signature, not just one pixel at `bg+1`.
- Position sweep: move a trail/head/sprite across 11x11 block boundaries and
  verify center-of-mass moves smoothly, not by off-by-one jumps.
- Thin line orientation test: horizontal, vertical, diagonal, and curved trails
  at many phases; measure changed-pixel count, total intensity mass, centroid
  error, and continuity under thresholds such as `bg+1`, `bg+2`, and `bg+4`.
- Player palette matrix: verify 2P self/other luma separation; mark 3P/4P
  grayscale collision as an explicit blocker until solved.
- Bonus sprite signature matrix: render all 12 bonus types at several positions
  and radii, then compute pairwise gray64 distances.
- Tiny bonus classifier probe: crop gray64 patches around each bonus type and
  check whether a simple nearest-template or logistic classifier can decode the
  type above chance. If not, put bonus type in an explicit semantic plane rather
  than hiding it in grayscale.
- Dirty/GPU parity gate: optimized path must match CPU reference byte-for-byte
  on gray64, and RGB too if it stores RGB.
- Long random trajectory parity: compare dirty cache to full CPU reference
  across many append, wrap, bonus spawn/despawn, bonus type, and reset events.

## Current Recommendation

Do not change the downsample method blindly. Keep `gray64 stack4` as the trusted
control for now. The current 704 RGB -> BT.601 luma -> exact 11x11 area average
is a clean, standard, anti-aliased compression path.

Do add the tests above before promoting any shortcut. If the concern is
preserving color/type identity, the right fix is probably extra semantic
channels, not a more exotic grayscale formula and not an immediate blind jump
to RGB.

The strongest canary already exists in the repo:
`curvyzero_source_state_bonus64_stack4_player_perspective/v1`, shape
`(22,64,64)`. It keeps the 4 occupancy history channels and adds bonus mask,
bonus type, self/other status planes, borderless, and TTL. It should be treated
as a diagnostic or A/B canary before changing the main baseline.

Recommended next comparison set:

- `gray64 stack4`: baseline/control;
- `bonus64 semantic`: canary for bonus/status information loss, only after
  model/config compatibility is explicitly smoked;
- optional `RGB64` or `RGB96`: only if we want a visual-only color-preserving
  comparison after the semantic canary is understood, and only as a separate
  model/config lane.

Do not increase resolution before proving a `64x64` visibility failure.
