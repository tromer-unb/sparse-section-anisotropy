# Scripts and provenance layers

This directory intentionally separates **historical manuscript code** from the reusable API in `src/`.

## `figures/`

- `figure1.py` through `figure5.py`: exact scripts recovered from the original analysis workstation and used to regenerate the manuscript figures during the September 2026 audit.
- `figure6_original.py`: exact historical Figure 6 renderer. It is preserved verbatim even though it contains workstation-specific paths.
- `figure6.py`: portable Figure 6 renderer. It reads the frozen CSVs from `results/figure6/` and obtains raw volumes from `SPARSE_SECTION_DATA_DIR`.

The historical scripts are provenance records. Avoid cleanup edits that would make them diverge from the code actually used for the paper. Improvements belong in reusable modules or portable wrappers.

## `validation/`

These are the recovered real-rock development and production scripts:

- `check_real_rock.py`: initial RAW inspection and phase-convention checks.
- `real_bentheimer_test1.py` -- `real_bentheimer_test5.py`: calibration/development sequence.
- `real_bentheimer_final_validation.py`: frozen Bentheimer production protocol.
- `real_edb1_final_validation.py`: Edwards Brown production validation using the frozen protocol.

The final validation scripts are computationally heavy and may create caches. They are preserved for auditability; the lightweight Figure 6 renderer does not rerun them.

## Portable Figure 6

After acquiring the raw volumes with `scripts/data/download_real_rocks.py`:

```bash
export SPARSE_SECTION_DATA_DIR=/path/to/real_rocks
python scripts/figures/figure6.py
```

Expected raw layout:

```text
$SPARSE_SECTION_DATA_DIR/
├── Bentheimer_15A/ROI1/kocurek_15a_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw
└── EdwardsBrown_EdB1/ROI1/edb-1_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw
```

Generated outputs are written to `outputs/figure6/`.
