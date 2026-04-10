## Scripts learnings

- MoE trace NPZ v1 stores arrays as `.npy` members plus a raw `metadata.json` ZIP entry; parse metadata via `zipfile` (or equivalent ZIP reader) rather than relying on `np.load()` keys alone.
- Some trace NPZs can include malformed/non-JSON `metadata.json`; treat metadata as optional and continue with array-derived dimensions after warning.
