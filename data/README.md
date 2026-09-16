# Data directory

Large third-party micro-CT volumes are intentionally **not stored in normal Git history**.

For reproducible real-rock validation, create the following local layout after acquiring the original dataset files from their authoritative repository:

```text
data/
  raw/
    bentheimer/
      kocurek_15a_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw
    edwards_brown/
      edb-1_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw
  derived/
    bentheimer/
      bentheimer_final_subvolumes.csv
      bentheimer_final_summary.csv
      bentheimer_final_random.csv
      bentheimer_final_systematic.csv
      bentheimer_final_reference_seeds.csv
    edwards_brown/
      edb1_final_subvolumes.csv
      edb1_final_summary.csv
  manifest.csv
```

## Expected raw representation

The archived production workflow expects both Figure-6 volumes as:

- shape: `2500 x 2500 x 2500` voxels;
- dtype: unsigned 8-bit integer;
- voxel size: `2.25 micrometre`;
- validated analysis convention: `0 = pore`, `1 = solid`.

Never infer a RAW shape from file size alone when adding a new dataset. Record the shape, dtype, endianness, phase convention, voxel size, and source metadata in `manifest.csv`.

## Manifest schema

Recommended columns:

```text
rock,sample_id,role,source_title,source_authors,doi,url,
archive_filename,local_filename,sha256,bytes,shape,dtype,
voxel_size_um,pore_value,solid_value,license,manuscript_citation
```

## Figure 6

The Figure-6 renderer should depend on the compact derived CSV tables plus the two raw volumes only for its representative image panel. It must not silently invoke the expensive Monte Carlo reference calculation.

See `docs/DATA_PROVENANCE.md` for the frozen validation protocol and the distinction between raw data, reference estimation, sparse reconstruction, and figure rendering.
