# Known Gaps And Questions

## Gaps

- The central trainer/status paths now scan `lightzero_exp*`, but some scripts
  may still default to `train/lightzero_exp/ckpt`.
- We do not yet know which existing tests are valuable after the refactor and
  which only preserve stale structure.
- We do not yet have one local end-to-end test that connects checkpoint
  discovery, status, resume, poller, and eval/GIF scheduling.
- The large Modal trainer responsibilities are mapped at a first-pass level,
  but extraction is still only partly done.
- The future opponent source should be externally controlled, likely via a
  registry/control-plane interface, but that contract is not implemented yet.

## Questions

- Which functions can be moved to pure modules without importing Modal?
- Which status fields are truth and which are display cache?
- Should stable mirrored checkpoints remain a separate fallback, or should all
  readers directly use broad LightZero discovery first?
- Which tests should be deleted after replacement coverage exists?
- What is the smallest opponent-registry interface the trainer should consume
  without learning tournament policy or hard-coded path strings?
- Should active manifest scripts resolve checkpoint refs through the new helper
  or be replaced by a cleaner opponent-registry writer?
