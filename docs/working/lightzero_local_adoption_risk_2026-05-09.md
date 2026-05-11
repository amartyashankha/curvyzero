# LightZero Local Adoption Risk - 2026-05-09

Scope: local docs/source evidence only. No web, no code edits beyond this
document, and no pytest.

## Bottom Line

LightZero has been useful in this repo, but painful in exactly the places that
matter for adoption: evaluator protocol, checkpoint/config shape, target
visibility, support-scale semantics, Modal runtime packaging, and policy
quality. The strongest current recommendation is **contain**: keep LightZero as
a bounded MuZero control and custom-env experiment lane, but do not make it the
CurvyTron training backbone unless it earns that status with strict stock Pong
parity and reliable custom-env checkpoint curves.

LightZero should not be deleted. It has already paid for itself as a forcing
function around artifacts, strict checkpoint loading, target sidecars,
scorecards, and official-control discipline. But it should also not be allowed
to own the main simulator/training architecture by default.

## Pain Points

### Stock Evaluator Mismatch

The repo hit a real evaluator mismatch by trying DI-engine's generic
`InteractionSerialEvaluator` on a MuZero Atari path. That generic evaluator did
not pass the MuZero-specific `action_mask` argument into
`MuZeroPolicy._forward_eval`, producing the missing-argument failure recorded in
`docs/working/lightzero_stock_evaluator_action_mask_gap_2026-05-09.md`.

The fix was not to patch around the policy call manually. The local note says
the correct stock route is LightZero's `lzero.worker.MuZeroEvaluator`, which
collates `action_mask`, `to_play`, and `timestep` from `env.ready_obs` before
calling the policy. A tiny Modal parity smoke then passed for the matching
64x64 checkpoint: strict model load, no fallback, manual-vs-stock action
sequence matched, and the stock evaluator path recorded the expected mask and
stacked observation shape.

Adoption risk: LightZero's correct path is narrower than it first appears. The
repo has to know when generic DI-engine pieces are the wrong abstraction for
MuZero.

### Pretrained Checkpoint Shape Mismatch

The official/pretrained Pong lane also has a separate checkpoint/config shape
problem. Local state docs say OpenDILab pretrained strict eval remains blocked
by an older 96x96/downsample checkpoint surface versus the current 64x64 stock
config/eval path. The evaluator fix does not solve this: a mismatched
checkpoint can fail before evaluator parity is meaningful.

Local checkpoint round-trip works when the checkpoint and config match. The
stock visual tiny trainer's `iteration_1.pth.tar` strict-loaded into the
matching conv `MuZeroModel` and `MuZeroPolicy`, with input `[1,4,64,64]`,
policy logits `[1,6]`, and value `[1,601]` recorded in
`docs/experiments/2026-05-09-modal-lightzero-pong-checkpoint-load-smoke.md`.

Adoption risk: "use an official checkpoint" is not plug-and-play. The exact
LightZero version, observation preprocessing, model surface, and checkpoint
shape must match.

### Config And Support-Scale Patching

The dummy Pong support calibration audit found a version-specific trap. The
repo pins `LightZero==0.2.0`; that version's MuZero policy/model path still
uses `policy.model.support_scale=300` and 601-class reward/value supports,
while the repo's patched config surface can record
`reward_support_range`/`value_support_range` fields that v0.2.0 may not
actually consume.

For dummy Pong, the environment reward returned to LightZero is sparse
`-1/0/+1`; shaped loss-delay is telemetry only. A 601-atom support is probably
not literally dividing rewards by 300, but it is badly calibrated for a tiny
sparse task and can hide whether the model is learning value/reward at the
right scale.

Adoption risk: config patches can look correct in our summaries while the
pinned LightZero internals still compile a different model/support contract.
The audit recommends logging decisive compiled fields such as `support_scale`,
`reward_support_size`, and `value_support_size`, not only the requested patch
surface.

### Dummy Pong Target Visibility

The target-sidecar work was a major unlock because it changed the question from
"what action did exploration execute?" to "what policy target did MuZero train
against?" Local target notes state that LightZero trains policy logits toward
MCTS root visit distributions, not toward the final exploratory action.

The completed safe sidecar smoke wrote 16 rows from one episode. It separated
executed action from target mass: executed `down` appeared on 9/16 rows, and
target mass for `down` was nonzero on 12/16 rows. But the sidecar still did not
label the oracle-winning action per row, so it could not answer whether the
winning move received target mass in each decisive state.

A separate parity-gap note records the sharper failure: in down-needed states,
low-simulation train-time MCTS can produce visits like `[1,1,0]`, assigning
zero target mass to the winning `down` action. That makes more replay of the
same target actively unhelpful.

Adoption risk: LightZero can be mechanically collecting data while silently
teaching the wrong root policy for the custom task. Target sidecars are not
optional for trust.

### Modal Image And ROM Handling

Modal itself has mostly worked when kept coarse: one whole train/eval job per
Function, framework outputs under `/tmp`, mirrored summaries/checkpoints into
`curvyzero-runs`, and Volume commits after durable artifacts exist. The
LightZero training pattern critique says this shape is right and warns against
putting Modal in the hot loop.

The stock Atari image path still had dependency friction. The env smoke first
failed on missing OpenCV, then reached the Atari ROM gate. It passed only after
adding `opencv-python-headless` and making ROM/license handling explicit via
`AutoROM[accept-rom-license]` plus `AutoROM --accept-license` in
`lightzero_atari_rom_image.py`.

Adoption risk: stock LightZero Atari/Pong is not just Python package install.
It requires visible runtime decisions for OpenCV, ALE/Gym, ROM installation,
and license acceptance. The repo now has a workable pattern, but it is a
project-owned image contract, not something LightZero hides.

### Action Collapse

Action mapping itself is no longer the leading suspect. Local bug-hunt docs
record that dummy Pong actions are `0=up`, `1=stay`, `2=down`, all legal, and
baseline/random policies can emit `down`.

The learned LightZero checkpoints still collapse. Examples from local notes:
the 512/8 MCTS scorecard loaded strictly and ran, but combined LightZero
actions were effectively up-only, `[2060,7,0]`. Later post-seed-fix and
post-deep-seed-fix runs improved seed diversity and showed training could
physically use action `2`, yet independent held-out MCTS scorecards still had
zero `down` across learned rows. The lagged-opponent run collapsed to all
`stay` in both paired and player0-only evals.

Official Atari controls show a related but separate collapse pattern. Stock
Atari train/checkpoint/eval mechanics work, but the 4096/sim10 L4 run stayed at
return `-6` under a 256-step cap and by `iteration_4` selected action `5`
deterministically. Local audits classify this as infrastructure pass / signal
fail, likely undertrained and off-recipe rather than action mapping.

Adoption risk: the repo can make LightZero run, load, and evaluate, but policy
quality remains brittle. More same-config scale is not justified without
target-quality, support-scale, and official-recipe parity checks.

## What Has Worked

- Stock ALE Pong env reset/step now works on Modal through the
  LightZero/DI-engine `atari_lightzero` path after OpenCV and AutoROM handling.
- Tiny stock Atari Pong train/checkpoint/eval mechanics work: checkpoints are
  mirrored, strict-loaded, and evaluated without model fallback when config and
  checkpoint shape match.
- Stock CartPole, TicTacToe, and Connect4 smokes provide useful execution
  controls for the LightZero package/image path.
- Custom dummy Pong can call LightZero's real `train_muzero`, write summaries,
  episodes, training signals, LightZero manifests, and `.pth.tar` checkpoints.
- Full dummy Pong checkpoint loading is no longer the main blocker for matched
  configs: MCTS/eval-mode loader smokes and full scorecards can strict-load the
  mirrored model.
- Independent scorecards have improved the truth boundary: trainer-side rows
  are no longer mistaken for final checkpoint quality.
- Target replay sidecars now expose executed actions, child visit targets,
  rewards, done flags, and config snapshots, making root-target audits possible.
- Modal artifact discipline is emerging: immutable checkpoint refs, hashes,
  run/attempt ids, summaries, eval JSON, and explicit lane labels.

## Recommendation: Contain

Do not replace LightZero immediately, and do not keep scaling it as the default
answer. **Contain it.**

Keep LightZero for:

- stock Pong/ALE reproduction and evaluator parity;
- custom dummy Pong MuZero experiments with strict target/support/eval audits;
- checkpoint/load/scorecard artifact patterns;
- comparison against repo-native learners.

Do not let LightZero become the CurvyTron backbone until both conditions hold:

- a recognizably stock LightZero Pong control works with matching checkpoint
  shape, stock evaluator path, strict no-fallback eval, and a meaningful
  checkpoint curve;
- the custom-env lane shows reliable held-out improvement with visible root
  targets, compiled support-scale proof, action diversity, and independent
  scorecards.

For CurvyTron, keep the simulator and serious first learner repo-owned around
the simultaneous `[B,P]` shape. Use LightZero as a contained MuZero control
lane, not as the framework that defines the environment contract.

## Local Sources

- `docs/working/lightzero_stock_evaluator_action_mask_gap_2026-05-09.md`
- `docs/working/lightzero_checkpoint_loader_probe_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_support_calibration_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_target_sidecar_read_2026-05-09.md`
- `docs/working/lightzero_pong_action_collapse_bug_hunt_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_official_parity_gap_2026-05-09.md`
- `docs/working/modal_lightzero_training_pattern_2026-05-09.md`
- `docs/working/lightzero_official_atari_collapse_investigation_2026-05-09.md`
- `docs/working/training_state_index_2026-05-09.md`
- `docs/working/training_framework_alternatives_2026-05-09.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-env-smoke.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-checkpoint-load-smoke.md`
