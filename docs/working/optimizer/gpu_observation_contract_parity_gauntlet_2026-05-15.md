# GPU Observation Contract Parity Gauntlet

Date: 2026-05-15

Status: red-team design, not a training change.

Purpose: define the smallest adversarial proof that a JAX/H100 observation path
is still the same production policy contract as CPU `cpu_oracle`:

```text
browser_lines + simple_symbols -> controlled-player [4,64,64]
```

The gauntlet should fail loudly before GPU promotion if the candidate only
matches easy real-env smoke rows. It must compare the trainer-visible contract,
not just a single rendered frame.

## Non-Negotiables

- CPU `cpu_oracle` is the reference.
- The gauntlet runs out of band. Do not edit live training defaults or promote a
  backend as part of this proof.
- The candidate must name its backend and must not silently fall back to CPU.
- Promotion requires exact equality for production-contract fields. Any
  tolerance is a research waiver, not a trainer promotion result.
- A renderer-only pass is insufficient. Stack, reset, final observation,
  reward, done, and info fields are part of the contract handed to LightZero.

## Smallest Useful Shape

Run a fixed `B=32` kill corpus through both CPU and GPU paths, for both
controlled-player views where the state supports them.

- `16` hand-built adversarial rows.
- `16` real-rollout rows sampled from CPU production wrappers.
- `2` lifecycle probes around the corpus: reset before first row, and one
  terminal-to-reset transition in a mixed batch.

That is small enough to run on every candidate build, but sharp enough to catch
the bugs that easy smoke rows miss. A broader `B=128` nightly can exist later,
but the `B=32` suite is the promotion blocker.

## Hand-Built Row Categories

Each hand row gets a stable id, a short reason string, and a minimal serialized
source-state fixture. Prefer one row that attacks multiple assumptions over a
large collection of polite cases.

1. **Owner connectivity and cursor traps**
   - visual trail owner sequences `[0, 1, 0]`, `[1, 0, 1]`, and `[0, 1, 1, 0]`;
   - inactive holes between same-owner points;
   - `visual_trail_break_before` set at each possible link boundary;
   - `visual_trail_write_cursor` at `0`, mid-buffer, and capacity;
   - active stale tail slots past the cursor containing bright crossed geometry.

2. **Composition order traps**
   - owner crossing where the later owner draw order wins despite lower luma;
   - bonus stamp over trail;
   - live head over bonus and trail;
   - dead or absent player head not drawn;
   - coincident live heads with deterministic player draw order.

3. **Geometry and downsample traps**
   - line caps exactly on 704-to-64 block boundaries;
   - radius sequence `4, 4, 8, 8, 4` within one owner path;
   - same-position repeated points with different radii;
   - clipped wall and corner symbols;
   - near-zero radius and large-radius rows.

4. **Player perspective and color traps**
   - controlled player `0` and `1`;
   - swapped `avatar_color`, duplicate color indices, and high color indices;
   - `BonusAllColor` active, expired, and just-cleared rows;
   - a 4P row with one dead player, one absent player, and stale invalid-owner
     body slots that must be ignored.

5. **Bonus symbol traps**
   - all 12 simple-symbol bonus types represented at least once;
   - bonus centered on trail crossing, live head, dead head, wall edge, clipped
     corner, and another bonus;
   - stack-affecting bonuses with visible effect metadata present, even though
     the policy image only sees simple symbols.

6. **Lifecycle traps**
   - one terminal row where `final_observation` must be the terminal stack;
   - one autoreset row where the new first observation must not leak terminal
     pixels into the stack;
   - one mixed batch where only selected rows reset.

## Real-Rollout Sampling

The real rows must come from the CPU production wrapper, not from a synthetic
renderer benchmark alone. Use fixed seeds and store the selected rows.

Sample strata:

- source ticks `0, 1, 2, 8, 64, 512, 2048, 8192`;
- rows immediately before terminal, at terminal, and immediately after reset;
- rows with max active trail prefix, max bonus count, max stack depth, and max
  cursor weirdness seen in the rollout;
- both controlled-player views;
- at least one scripted action rollout and one random action rollout;
- at least one bonus-enabled rollout, even if the first trainer target normally
  profiles no-bonus opponents.

Keep the frozen corpus hash in the run output. Real sampling is allowed to find
rows, but the promotion gauntlet consumes pinned rows so failures are stable.

## Fields To Compare

Compare a normalized trainer-step object, not only pixels.

Required reset fields:

- observation stack: shape, dtype, value range, channel order, and zero-fill
  policy;
- `action_mask`, `to_play`, and legal action ids;
- reset metadata: schema ids and hashes, `policy_observation_backend`,
  `trail_render_mode`, `bonus_render_mode`, `controlled_player`,
  `decision_source_frames`, reset seed, reset source, episode id, and round id.

Required step fields:

- `obs["observation"]` full `[4,64,64]` stack, not just the newest channel;
- `reward`;
- `done`, `terminated`, and `truncated`;
- `info["final_observation"]` or equivalent final-observation field;
- `final_reward_map` / final reward fields where present;
- terminal reason, truncation reason, winners, losers, death fields, and
  `needs_reset`;
- reset/autoreset provenance and row mask behavior;
- visual metadata: schema ids and hashes, render modes, backend id, truth level,
  frame stack owner, raw/single-frame schema ids, and source fidelity claim.

Stack-specific assertions:

- reset stack has exactly three zero historical channels plus the first frame,
  or whatever policy metadata states for that backend;
- nonterminal step shifts FIFO exactly once;
- terminal `final_observation` captures the terminal public stack before any
  reset mutation;
- post-reset first stack does not contain any terminal frame;
- mixed-batch reset changes only selected rows.

## Pass And Fail Criteria

Pass:

- exact byte equality for CPU and GPU single frames;
- exact `float32` equality for normalized stacks after the shared conversion;
- exact equality for rewards, done flags, terminal flags, final observation
  fields, reset metadata, and info contract fields;
- same corpus hash, config hash, and schema hashes;
- backend proof shows JAX GPU execution for the candidate and no CPU fallback;
- parity is checked for both controlled-player views where applicable.

Fail immediately:

- any unexplained pixel or stack mismatch;
- any mismatch in reward, done, final observation, reset behavior, or info;
- any avatar-color or controlled-player perspective mismatch;
- any hidden fallback or omitted backend metadata;
- any row whose CPU and GPU paths consume different state, action, seed, or
  reset provenance;
- any promotion claim based only on render-kernel timing without stack, reset,
  and host/device boundary accounting.

An intentionally accepted off-by-one luma difference must be documented as a
research exception with row id, pixel count, cause, and owner approval. It still
does not pass this promotion gauntlet.

## Failure Debug Bundle

Every failed row should write one directory under:

```text
artifacts/local/gpu_observation_contract_parity/<run_id>/<case_id>/
```

Minimum contents:

- `case.json`: row id, category, reason string, seed, tick, action, controlled
  player, render config, stack depth, reset index, and source-state hashes;
- `state.npz`: compact source-state arrays, including visual trail, cursor,
  bonus, avatar color, terminal, and reset fields;
- `cpu_step.json` and `gpu_step.json`: contract objects with large arrays
  replaced by hashes and metadata;
- `cpu.npy`, `gpu.npy`, and `diff.npy` for the newest frame and full stack;
- `diff.png`: CPU, GPU, abs-diff, and enlarged first-difference crop;
- `stage_diffs/`: optional trails-only, bonuses-only, heads-only, and
  downsample-only comparisons;
- `replay.sh`: exact command to rerun just this case.

The first-difference report should include:

- row id and controlled player;
- field name and array path;
- pixel coordinate in `64x64` and corresponding 704-block bounds;
- CPU value, GPU value, absolute diff, and local `11x11` block summary;
- nearby trail slots, owners, radii, `break_before`, cursor, bonus slots, head
  positions, avatar-color LUT, and alive/present flags;
- candidate topmost object under CPU z-order and GPU z-order.

If the report cannot explain the first mismatch without opening a debugger, the
failure tooling is not good enough yet.

## Execution Ladder

P0, promotion blocker:

- run the `B=32` frozen corpus on CPU and H100 candidate;
- compare both views and all trainer-visible fields;
- emit failure bundles;
- no live training changes.

P1, confidence expansion:

- run `B=128` with more real-rollout rows, longer horizons, more seeds, and
  generated mutations around any previous mismatch;
- add L4/T4 only as portability evidence, not as the H100 promotion proof.

P2, systems proof:

- run a mock collector boundary that includes pack, host-to-device transfer,
  render, readback or device-resident handoff, stack update, reset/final
  observation handling, policy-forward stub, and replay copy;
- report compile excluded, warmup excluded, host readback counted, GPU memory,
  shape buckets, and batch size;
- only after this can speed claims be compared to CPU `cpu_oracle`.

## Skeptical Default Verdict

A GPU renderer that matches a few real-env smoke rows is promising, not promoted.
The smallest trustworthy proof is the `B=32` adversarial corpus plus
trainer-visible lifecycle comparison. If it cannot explain and reproduce the
first failure in one command, it is not ready to replace `cpu_oracle`.
