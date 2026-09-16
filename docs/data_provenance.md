# Data provenance

## Upstream real-rock dataset

The real-rock validation uses the public dataset:

> Ferreira et al., *Full scale, microscopically resolved tomographies of sandstone and carbonate rocks augmented by experimental porosity and permeability values*, Scientific Data (2023).

- Figshare+ article ID: **21375565**
- Frozen dataset version used by the manuscript workflow: **v6**
- Versioned DOI: **10.25452/figshare.plus.21375565.v6**
- Metadata workbook: `Dataset_Information.xlsx`
- Metadata file ID observed from the Figshare API: **40168054**

The downloader queries Figshare metadata rather than relying on manually copied temporary URLs.

## Files used for Figure 6

| Manuscript label | Dataset sample | Exact archive | Figshare file ID |
|---|---|---|---:|
| Bentheimer sandstone | Kocurek 15A | `kocurek_15a_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw.tar.bz2` | 38022670 |
| Edwards Brown carbonate | EdB-1 | `edb-1_2p25um_ir_rec_2500x2500x2500_binary_ROI-1.raw.tar.bz2` | 38022599 |

Both decompressed ROI-1 files are:

- shape: `2500 x 2500 x 2500`;
- dtype: unsigned 8-bit integer;
- expected byte count: `15,625,000,000`;
- nominal voxel size used in the manuscript analysis: **2.25 micrometres**.

The published binary convention used in the final analysis is:

```text
RAW 0 -> pore
RAW 1 -> solid
internal pore indicator chi = 1 - RAW
```

No resizing, smoothing, morphological operation, or re-segmentation was applied to these published binary ROI files before the final validation.

## Reproducible retrieval

```bash
python scripts/data/download_real_rocks.py --rock both --rois 1 --list-only
python scripts/data/download_real_rocks.py --rock both --rois 1
```

The first command checks the current Figshare metadata and identifies the files without downloading them. The second downloads, extracts, checks the expected byte count, and samples binary values.

The historical working tree contained this downloader as `script.py`; its own docstring and command examples identify the intended released filename as `download_real_rocks.py`. The repository therefore publishes the same script under that descriptive name.

## Spatial sampling used for final validation

Each ROI was sampled by a non-overlapping 3 x 3 x 3 grid of 27 subvolumes:

- subvolume size: `512^3` voxels;
- centers on each computational axis: `384, 1250, 2116`;
- physical side length at 2.25 micrometres/voxel: 1.152 mm.

Within each subvolume, three orthogonal section families were precomputed. Each family used 13 candidate offsets:

```text
-192, -160, -128, -96, -64, -32, 0, 32, 64, 96, 128, 160, 192 voxels
```

Thus each subvolume contains 39 cached candidate section measurements. Final sparse configurations select `1, 3, 5, 7` sections per orientation, corresponding to `3, 9, 15, 21` total sections. For each count, 200 random position configurations were drawn without replacement within each orientation family.

This gives 27 x 4 x 200 = **21,600 sparse reconstructions per rock**.

These 21,600 configurations are resampling experiments within fixed subvolumes, not independent geological samples. Likewise, the 27 subvolumes are spatial replicates within one ROI, not 27 independent rock specimens.

## Why raw volumes are not committed

The original RAW files are already permanently archived upstream and are approximately 14.55 GiB each uncompressed. Committing copies would make the code repository harder to audit and would duplicate an authoritative source. This repository stores filenames, file IDs, version information, download/validation code, and compact derived tables instead.

The MIT license in this repository applies to repository code only. The upstream dataset retains its own terms; consult the versioned Figshare record before redistribution.
