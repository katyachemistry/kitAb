<p align="center">
  <img src="kitAb_logo.png" alt="kitAb logo" width="300">
</p>

# kitAb

**kitAb: lightweight molecular descriptors and AutoML integrated framework for rapid antibody developability assessment**

kitAb combines structure-based developability descriptors with automated machine learning for antibody developability prediction. Pipelines are driven by YAML config files; run them with `bash fastab.sh <config>` (entry-point scripts and conda env names still use the legacy `fastab` prefix and will be renamed in a later release).

## Installation

From the repo root (requires **mamba** or **conda**, **git**, **wget** or **curl**):

```bash
./install.sh                    # fastab + abb2 + abb3 + flashabb (default)
./install.sh --fastab-only      # fastab env only (descriptors / AutoML)
./install.sh --no-abb2          # fastab + abb3 + flashabb (skip abb2 / ImmuneBuilder)
./install.sh --no-flashabb      # fastab + abb2 + abb3 (skip FlashABB)
source fastab.local.env
```

The `fastab` env includes [MMseqs2](https://github.com/soedinglab/mmseqs2) (`mmseqs` on PATH via the `mmseqs2` bioconda package) for automatic sequence-identity cross-validation splits.

Optional structure backends (installed by default unless skipped):

| Backend | Conda env | Use case |
|---------|-----------|----------|
| ABB2 (ImmuneBuilder) | `fastab-abb2` | Structure prediction; IMGT renumbering and minimization |
| ABB3 | `fastab-abb3` | Structure prediction |
| FlashABB | `fastab-flashabb` | Fast structure prediction (see `configs/scenario1_flashabb.yaml`) |

## Scenario 1: Sequences, assay measurements, and optional external features → descriptors and predictions

The toolkit is run with config YAML files. See **configs/scenario1.yaml**.

Most often you configure `input_csvs_folder` and `result_folder`. You can pass several datasets by placing one CSV per file under `input_csvs_folder`.

```text
kitAb/   # repo root
├── datasets/       # input_csvs_folder
│   ├── dataset1.csv
│   │       name,heavy,light,target_Tm,target_HIC,feature_isotype
│   │       mAb1,EVQLVES...,DIQMTQ...,55,0.7,lambda
│   │       mAb2,EVQLVES...,DIQMTQ...,57,1.2,kappa
│   ├── dataset2.csv
│   └── …
```

One dataset = one CSV file. It must contain `name`, `heavy`, `light`, and `target_X`, `target_Y`, …, `feature_A`, `feature_B`, … columns. Each `name` must be unique, and the concatenated `heavy`+`light` sequence must be unique across rows (checked when the run config is prepared and again before AutoML merge).

When you do not supply pre-built structures, configure `structure_prediction` so the pipeline can predict Fv structures before descriptors:

```yaml
structure_prediction:
  model: abb2          # abb2 | abb3 | flashabb
  device: 1            # GPU index (0, 1, …) or cuda:0 / cpu
  runs: 2              # replicate runs (useful for stochastic models)
  skip_existing: true  # skip antibodies whose .pdb already exists
  # batch_size: 4      # default 4 for abb2/abb3; default 50 for flashabb
```

Try the example:

```bash
bash fastab.sh configs/scenario1.yaml
```

Default AutoML hyperparameters are used. To customize them, edit **src/automl.yaml** (or set `automl_config:` in your run config).

### Cross-validation splits

Input CSVs are expected to have every row filled in for `name`, `heavy`, and `light` (validated earlier in the pipeline).

By default, the pipeline automatically derives balanced cross-validation folds from the antibody sequences using MMseqs2 sequence-identity clustering. The cascade tries identity thresholds from 0.50 to 0.80 in 0.05 steps, and for each threshold attempts 5-fold then 4-fold splits. A split is accepted when the largest fold is at most twice the size of the smallest fold. If no balanced split is found, AutoML falls back to random 5-fold CV.

CV mode and how many AutoML jobs run are decided in one place (`prepare_run_config.py`): if the dataset uses a `fold` column (`split_col: fold`), one job is run with seed `42`; if there is no `fold` column (`split_col` unset), three jobs run with seeds `42`, `43`, and `44`. That applies equally to MMseqs-derived folds, a pre-existing `fold` column, and random-CV fallback.

When clustering succeeds, a `fold` column is written to `{result_folder}/splits/{dataset_stem}_seqid_folds.csv`. MMseqs2 intermediate files are removed after each attempt; only the final `*_seqid_folds.csv` is kept under `splits/`.

If your CSV already contains a `fold` column, that column is used directly and the clustering step is skipped.

To force random 5-fold CV (three runs with seeds 42, 43, 44) for specific datasets—ignoring any existing `fold` column and skipping MMseqs clustering—list them under `split_randomly`:

```yaml
split_randomly: ab21.csv,hutchinson2023enhancement_top200tm1_igg.csv,pdgf38.csv
```

Entries may be given with or without the `.csv` suffix; matching is by dataset stem.

To skip specific datasets entirely, use `exclude_datasets` (same stem matching as `split_randomly`).

AutoML uses all developability descriptor groups for every target (`surface`, `core`, `general`, `sequence_motives`) unless a run config block sets `developability_features` explicitly.

---

## Scenario 2: Same as Scenario 1 + you already have structures (PDB/CIF)

See **configs/scenario2.yaml**.

Structures must contain Fv fragments; only those are used for the pipeline.

Data organization:

```text
kitAb/
├── datasets/       # input_csvs_folder
│   ├── dataset1.csv
│   │       name,heavy,light,target_Tm,target_HIC,feature_isotype
│   │       mAb1,EVQLVES...,DIQMTQ...,55,0.7,lambda
│   │       mAb2,EVQLVES...,DIQMTQ...,57,1.2,kappa
│   ├── dataset2.csv
│   └── …
│
├── structures/     # input_structures_folder
│   ├── dataset1/   # for dataset1.csv
│   │   ├── mAb1.pdb     # for mAb1 in name column
│   │   ├── mAb2.pdb
│   ├── dataset2/
│   └── …
```

Configure parent folder names in the YAML (`input_csvs_folder` and `input_structures_folder`). Subfolder and file names must match dataset stems and antibody `name` values.

Another important block is `structures_processing`. If your structures are not IMGT-numbered, set **`renumber_imgt: true`** (requires the `fastab-abb2` env). To energy-minimize structures, set **`minimize: true`** (also requires `fastab-abb2`):

```yaml
structures_processing:
  renumber_imgt: false
  minimize: false
```

---

## Scenario 2a: Descriptors only on existing structures (no CSV datasets)

See **configs/scenario2a.yaml**.

Use this when you already have PDB/mmCIF files and only need developability descriptors (no assay CSVs, no AutoML). Set `input_structures_folder` and `result_folder`; omit `input_csvs_folder`.

Supported layouts:

```text
kitAb/
├── structures_abb2_paired_oas_sanity_skip_minimized/   # flat folder
│   ├── abngs_oas_paired_merged_data_paired_5457817.pdb
│   └── …
```

or nested batch folders (same convention as scenario 2):

```text
kitAb/
├── structures_abb3/
│   ├── ab21_abb3_1/
│   │   ├── mAb1.pdb
│   │   └── …
│   └── …
```

Control how jobs are discovered with `structures_layout`:

| Value | Behavior |
|-------|----------|
| `auto` (default) | One job per subdirectory if any contain PDB/mmCIF; otherwise the root folder is a single job |
| `flat` | Root folder is one job (recommended for very large flat PDB collections) |
| `subdirs` | One job per subdirectory; error if none found |

Optional `structures_processing` (`renumber_imgt` / `minimize`) runs in place before descriptors, as in scenario 2.

For large batches, set `cleanup: true` to delete per-structure intermediate dirs (`dssp/`, `propka/`, `sasa/`) after each JSON is written, and optionally `descriptor_batch_size` (e.g. `10000`) to process structures in chunks and bound peak disk and memory use.

```bash
bash fastab.sh configs/scenario2a.yaml
```

---

## Scenario 3: Descriptors only, no AutoML predictions

Depending on whether you have structures or not, prepare input and configure the relevant fields (see scenarios 1 and 2). Example for the no-structures case: **configs/scenario3.yaml**.

This scenario differs only by disabling AutoML:

```yaml
automl: false
```

---

## Scenario 4: AutoML only on pre-calculated descriptors

If you already have developability descriptors (e.g. from a previous run) and want only the AutoML step, set a `predefined_descriptors` block. See **configs/scenario4.yaml**. Descriptor calculation is skipped automatically when `predefined_descriptors` is set (you may also set `calculate_descriptors: false` explicitly).

Provide the folder containing pre-calculated descriptors and list allowed suffixes for run names `{dataset_stem}{suffix}`. For each match, the pipeline finds the corresponding dataset CSV in `input_csvs_folder` by stem. Layout is detected automatically (vendor-agnostic):

- **Subfolders**: `{dataset_stem}{suffix}/features.csv` (optional `features_filename`), or `{dataset_stem}{suffix}/results/*.json` (kitAb-style JSON).
- **Flat CSVs** (when no matching subfolders exist): `{dataset_stem}{suffix}.csv` in the folder root.

If any `{dataset_stem}{suffix}` subfolders exist, flat CSVs in the root are ignored.

```yaml
input_csvs_folder: datasets
result_folder: scenario4

predefined_descriptors:
  folder: descriptors_reproducibility_finalization
  allowed_suffixes:
    - _abb2_4
```

```bash
bash fastab.sh configs/scenario4.yaml
```
