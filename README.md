<p align="center">
  <img src="kitAb_logo.png" alt="kitAb logo" width="160">
</p>

# kitAb

**kitAb: lightweight molecular descriptors and AutoML integrated framework for rapid antibody developability assessment**

kitAb combines structure-based developability descriptors with automated machine learning for antibody developability prediction.

## Installation

From the repo root (requires **mamba** or **conda**, **git**, **wget** or **curl**):

```bash
./install.sh                         # kitab + abb2 + abb3 + flashabb (default)
./install.sh --kitab-only            # kitab env only (descriptors / AutoML / IMGT / OpenMM)
./install.sh --skip abb2             # kitab + abb3 + flashabb
./install.sh --skip abb3 flashabb    # kitab + abb2
source kitab.local.env
```

The `kitab` env also includes [MMseqs2](https://github.com/soedinglab/mmseqs2) for automatic sequence-identity CV splits.

| Role | Conda env |
|------|-----------|
| Descriptors, AutoML, IMGT, OpenMM | `kitab` |
| ABB2 (ImmuneBuilder) prediction | `kitab-abb2` |
| ABB3 prediction | `kitab-abb3` |
| FlashABB prediction | `kitab-flashabb` |


## Quick start


### Input CSV format

One dataset = one CSV under `inputs.datasets_dir` (filename stem is the dataset name, e.g. `ab21.csv`). Required columns: `name`, `heavy`, `light`, and at least one `target_*` assay column. Optional: `feature_*` columns and `fold`. Names and concatenated heavy+light sequences must be unique.

```csv
name,heavy,light,target_viscosity,feature_lc
mAb1,EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS,DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIK,14.4,lambda
mAb2,EVQLVESGGGLVQPGGSLRLSCAASGFTFSDSWIHWVRQAPGKGLEWVAWISPYGGSTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARRHWPGGFDYWGQGTLVTVSA,DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYLYHPATFGQGTKVEIK,20.9,kappa
```

### Structure folders

When using existing structures (`inputs.structures_dir` + `datasets_dir`), put one subdirectory per dataset under `structures_dir`. The folder name must be the CSV stem (`ab21` for `ab21.csv`) or start with that stem plus `_` (`ab21_abb2_1`). Every CSV `name` needs `{name}.pdb`, `{name}.cif`, or `{name}.mmcif` in that folder. Several matching folders are all used (for example both `ab21_abb2_1` and `ab21_abb2_2`).

```text
structures_abb2/
  ab21/                 # datasets/ab21.csv
    mAb1.pdb            # CSV name column
    mAb2.cif
  pdgf38_abb2_1/        # datasets/pdgf38.csv
    ...
```

Descriptors-only runs (no CSVs) use each subdirectory that contains PDB/mmCIF.

### Run

Pick an example yaml, copy it, and edit paths (`datasets_dir`, `output_dir`, …):

| Example | What it does |
|----------|----------------|
| `predict-and-automl.yaml` | Dataset(s) with seqs and targets → predict structures → get descriptors → AutoML |
| `existing-structures.yaml` | Dataset(s) with seqs and targets + existing structures → get descriptors → AutoML |
| `descriptors-only.yaml` | Existing structures → get descriptors |
| `automl-only.yaml` | Existing descriptors → AutoML on precomputed descriptors |
| `full-with-tuning.yaml` | Existing structures → descriptors → AutoML + hyperparameter tuning |

```bash
mkdir -p my_configs
cp examples/configs/predict-and-automl.yaml my_configs/run.yaml # or another starter from the table

./kitab.sh validate my_configs/run.yaml # check the config
./kitab.sh my_configs/run.yaml # run tool from the config
./kitab.sh resume runs/example_predict_automl # run.output_dir from the YAML, if interrupted
```

## Example config

```yaml
inputs:
  datasets_dir: datasets                 # input CSVs, see format below
  # structures_dir: structures           # e.g. ab21 or ab21_abb2_1 (see logic below)
  split_randomly: [ab21, pdgf38]       # CSV stems that use random 5-fold CV (skip MMseqs2)
  exclude_datasets: []                 # dataset stems to skip
  # predefined_descriptors_dir: descriptors  # e.g. ab21 or ab21_abb2_1, usually won't be needed; was used in the study for benchmarking


run:
  output_dir: runs/my_kitab_run          
  resume: false                          # false aborts if output_dir has files; true continues it and skips finished descriptor JSONs
  n_cpu: 8                             # parallelization with GNU Parallel

structure_prediction:
  enabled: true
  model: [abb2, abb3]
  device: cuda:0
  skip_existing: true                    # skip PDBs already predicted in this run (e.g. output_dir/structures/ab21/mAb1.pdb); does not use input structures_dir!

structure_processing: # IMGT renumbering and minimization
  enabled: true
  renumber_imgt: false
  minimize: true
  minimize_attempts: 5                   # retries per structure if minimize fails

descriptors:
  enabled: true
  cleanup: false                         # delete DSSP, PROPKA and freesasa files

automl:
  enabled: true
  eval_models: all                       # regressors: all, or linear,elasticnet,andomforest,svm,knn
  # config_path: src/automl.yaml         # optional: custom AutoML YAML (selectors, features_frac); default is src/automl.yaml

tuning: # HPO; top models are always saved after AutoML
  enabled: false
  # margin: 0.1                          # shortlist models within this CV Spearman of the best
  # max_rank: 3                          # at most this many models enter the shortlist
```

### Cross-validation

By default, balanced CV folds are derived with MMseqs2 sequence-identity clustering (0.50–0.80 cascade). If a CSV already has a `fold` column it is used. Force random 5-fold CV (seeds 42,43,44) with `inputs.split_randomly`.

Shared AutoML defaults live in [`src/automl.yaml`](src/automl.yaml).

### Failure policy

- Per-sequence / per-structure failures are logged; other items continue.
- Minimization retries each failed structure up to `minimize_attempts` (default **5**), then records a clear error and continues.
- AutoML completeness is all-or-nothing per dataset CSV: if **any** antibody is missing its descriptor JSON (or is listed in `failed_structures.tsv`), the whole dataset is **skipped for AutoML** (`skipped_incomplete`). Antibodies that did succeed are not run on their own. Other complete datasets still run.
- The pipeline exits non-zero when any stage had failures, while keeping logs and partial outputs.

### Saved models

After AutoML the top model per dataset/source/target is refit and written to `models/<dataset>__<source>__<target>/` (`estimator.joblib`, `meta.json`) plus `models/model_index.json`. Shortlist is CV Spearman (`tuning.margin` / `tuning.max_rank`). Defaults unless `tuning.enabled: true`; `meta.json` records `tuned`.

Load/predict:

```python
from kitab.models import load_tuned_model, predict_with_tuned_model
est, meta = load_tuned_model("runs/.../models/...")
yhat = predict_with_tuned_model("runs/.../models/...", feature_dataframe)
```

Eval models are `linear`, `elasticnet`, `randomforest`, `svm`, and `knn`.

## Reproduce paper results

Reuse the paper structures (no structure prediction). Descriptors use the default final set (59 features), then AutoML.

| Config | Structures |
|--------|------------|
| `examples/configs/reproduce-paper-abb2.yaml` | `structures_abb2` |
| `examples/configs/reproduce-paper-abb3.yaml` | `structures_abb3_minimized` |
| `examples/configs/reproduce-paper-flashabb.yaml` | `structures_flashabb` |

Needs `datasets/` plus the matching `structures_*` tree under the repo root. Outputs go under `runs/reproduce_paper_<backend>/`.

```bash
source kitab.local.env
./kitab.sh validate examples/configs/reproduce-paper-abb2.yaml
./kitab.sh examples/configs/reproduce-paper-abb2.yaml
# abb3 / flashabb:
# ./kitab.sh examples/configs/reproduce-paper-abb3.yaml
# ./kitab.sh examples/configs/reproduce-paper-flashabb.yaml
```

## Tests

```bash
source kitab.local.env
export PYTHONPATH=src
python -m pytest tests/ -q
```

Disposable smoke configs and fixtures live under `tests/`.
