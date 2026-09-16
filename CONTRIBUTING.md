# Contributing

This repository is both research software and an audit record for a manuscript.

- Do not silently modify files under `scripts/paper/` after results are frozen.
- Method improvements should normally be made under `src/` with tests.
- Any change that alters a manuscript result should document the affected figure/table, random seed, input data version, and regenerated output.
- Never commit the upstream multi-gigabyte RAW volumes. Update `docs/data_provenance.md` and the downloader if the authoritative upstream record changes.
- Prefer deterministic tests and small synthetic fixtures for CI.
