<p align="center">
  <img src="kitAb_logo.png" alt="kitAb logo" width="160">
</p>

# kitAb

**kitAb: lightweight molecular descriptors and AutoML integrated framework for rapid antibody developability assessment**

kitAb combines structure-based developability descriptors with automated machine learning for antibody developability prediction.

The physicochemical representation and distance enrichment is available through the web server https://kitab-atlas.com/ to support exploration and comparison of antibody profiles.

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

## Quick start

A run is a YAML file you pass to `./kitab.sh`. Copy a starter from [`examples/configs/`](examples/configs/) and edit it. Relative paths in that YAML are from the **repo root** (where `kitab.sh` lives), not from the YAML file.

| Starter in `examples/configs/` | What it does |
|---------|----------------|
| `predict-and-automl.yaml` | Dataset(s) with seqs and targets → predict structures → descriptors → AutoML |
| `existing-structures.yaml` | Dataset(s) with seqs and targets + existing structures → descriptors → AutoML |
| `descriptors-only.yaml` | Existing structures → descriptors |
| `automl-only.yaml` | Existing descriptors → AutoML |

```bash
mkdir -p my_configs
cp examples/configs/predict-and-automl.yaml my_configs/run.yaml   # edit this file

./kitab.sh validate my_configs/run.yaml
./kitab.sh my_configs/run.yaml
./kitab.sh my_configs/run.yaml --techniques elasticnet,sfs_svm --cv-mode nested
./kitab.sh resume runs/example_predict_automl   # run.output_dir from the YAML, if interrupted
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--enable-automl` / `--disable-automl` | YAML `automl.enabled` | turn AutoML on/off |
| `--techniques` | all four | `elasticnet,intercorr_svm,sfs_svm,sfs_knn` |
| `--cv-mode` | `nested` | `nested` or `flat` |
| `--no-final-model` | off | skip full-data refit (`estimator.joblib`) |
| `--cpus` | `run.n_cpu` | CPU workers for descriptors, IMGT, OpenMM, AutoML |

Defaults (SFS fraction, ElasticNet grid, intercorr threshold) live in [`src/automl.yaml`](src/automl.yaml).

## Run YAML

The file you copied (`my_configs/run.yaml`) looks like this. `inputs.datasets_dir` is the folder of sequence/assay CSVs; `run.output_dir` is where results go.

```yaml
inputs:
  datasets_dir: datasets                 # folder of CSVs, relative to repo root (or an absolute path)
  # structures_dir: structures           # existing PDB folders; same path rules
  split_randomly: [ab21, pdgf38]         # dataset names (ab21.csv → ab21) that use random 5-fold CV (skip MMseqs2)
  exclude_datasets: []                   # dataset names to skip
  # predefined_descriptors_dir: descriptors  # AutoML-only; used in the study for benchmarking

run:
  output_dir: runs/my_kitab_run
  resume: false                          # false aborts if output_dir has files; true continues and skips finished descriptor JSONs
  n_cpu: 8                               # descriptors, IMGT, OpenMM, AutoML (override with --cpus)

structure_prediction:
  enabled: true
  model: [abb2, abb3]
  device: cuda:0
  runs: 3                                # independent predictions per sequence (folders _1, _2, _3)
  skip_existing: true                    # skip PDBs already in this run (output_dir/structures/…); does not use structures_dir
  # batch_size: 4                        # GPU sequences per forward (default: 50 flashabb, 4 abb2/abb3)

structure_processing:                    # IMGT renumbering and minimization
  enabled: true
  renumber_imgt: false
  minimize: true
  minimize_attempts: 5                   # retries per structure if minimize fails

descriptors:
  enabled: true
  cleanup: false                         # delete DSSP, PROPKA and freesasa files

automl:
  enabled: true
```

## Input CSV

In that YAML, `inputs.datasets_dir` is a folder of CSVs. Relative paths are from the repo root; an absolute path is also fine.

Each file is one dataset, named after the file without `.csv`: `ab21.csv` is dataset `ab21`. That name is what `exclude_datasets`, `split_randomly`, and structure / descriptor folders use.

Required columns: `name`, `heavy`, `light`, and at least one `target_*` assay column (fill with 0 if you only need descriptors). Optional: `feature_*` columns and `fold`. Names and concatenated heavy+light sequences must be unique.

```csv
name,heavy,light,target_viscosity,feature_lc,fold
mAb1,EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS,DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIK,14.4,lambda,0
mAb2,EVQLVESGGGLVQPGGSLRLSCAASGFTFSDSWIHWVRQAPGKGLEWVAWISPYGGSTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARRHWPGGFDYWGQGTLVTVSA,DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYLYHPATFGQGTKVEIK,20.9,kappa,1
```

Example: with `datasets_dir: datasets` in the YAML, put the folder at the repo root:

```text
kitAb/                    # repo root
  kitab.sh
  my_configs/run.yaml     # the YAML you pass to kitab.sh
  datasets/               # inputs.datasets_dir: datasets
    ab21.csv
    pdgf38.csv
```

## Structure folders

If you already have PDB files, set `inputs.structures_dir` in the same YAML. One folder per dataset, named after the dataset (`ab21` for `ab21.csv`) or that name plus `_` (`ab21_abb2_1`). Every CSV `name` needs `{name}.pdb` in that folder. Several matching folders are all used (for example both `ab21_abb2_1` and `ab21_abb2_2`).

```text
kitAb/
  structures_abb2/      # inputs.structures_dir: structures_abb2
    ab21/               # dataset ab21 (ab21.csv)
      mAb1.pdb          # CSV name column
      mAb2.pdb
    pdgf38_abb2_1/      # dataset pdgf38 (pdgf38.csv)
      ...
```

Descriptors-only runs (no CSVs) use each subdirectory that contains PDB files.

## Precomputed descriptors

For AutoML-only runs, set `inputs.predefined_descriptors_dir` in the YAML. One folder (or a flat `{dataset}.csv`) per dataset, named like the dataset (`ab21`) or `{dataset}_...` (`ab21_abb2_1`). Each folder needs `features.csv` or `results/*.json`.

```text
kitAb/
  descriptors/          # inputs.predefined_descriptors_dir: descriptors
    ab21_abb2_1/        # dataset ab21 (ab21.csv)
      results/
        mAb1.json
        mAb2.json
    pdgf38_abb2_1/
      features.csv
```

## Cross-validation and AutoML

By default, balanced CV folds are derived with MMseqs2 sequence-identity clustering (0.50–0.80 cascade). If a CSV already has a `fold` column it is used. Force random 5-fold CV (seeds 42, 43, 44) with `inputs.split_randomly`.

AutoML evaluates four techniques in parallel (`elasticnet`, intercorrelation+SVM, SFS+SVM, SFS+KNN). Nested CV chooses the technique from **inner-fold** Spearman (never from outer-test) and pools those mixed out-of-fold predictions as the procedure score. The saved model is that inner-chosen technique refit on **all labelled rows**, using the mode of its per-fold inner-CV hyperparameters (ElasticNet `(alpha, l1_ratio)`; SFS eval model among `svm`, `knn`, `linear`, `randomforest`). Flat CV cannot choose among techniques; use nested CV or a single `--techniques` value.

## Failure policy

- Per-sequence / per-structure failures are logged; other items continue.
- Minimization retries each failed structure up to `minimize_attempts` (default **5**), then records a clear error and continues.
- AutoML completeness is all-or-nothing per dataset CSV: if **any** antibody is missing its descriptor JSON (or is listed in `failed_structures.tsv`), the whole dataset is **skipped for AutoML** (`skipped_incomplete`). Antibodies that did succeed are not run on their own. Other complete datasets still run.
- The pipeline exits non-zero when any stage had failures, while keeping logs and partial outputs.

## Saved models

After AutoML the deployed technique per dataset/source/target is refit on all labelled rows and written under `models/` (`estimator.joblib`, `meta.json`) plus `models/model_index.json`. The technique is the highest **mean inner** Spearman from nested CV. `meta.json` `cv_spearman_pooled_oof` is the nested-procedure score (mixed techniques across outer folds), not the CV score of the fitted model. Pass `--no-final-model` to compare techniques without saving fitted models.

```text
runs/my_kitab_run/
  descriptors/ab21_abb2_1/results/mAb1.json
  automl/metrics/technique_comparison.csv
  automl/metrics/best_technique.csv
  automl/metrics/fold_winners.csv
  models/
    model_index.json
    ab21__ab21_abb2_1/
      target_viscosity/
        estimator.joblib
        meta.json
```

```python
from kitab.models import load_tuned_model, predict_with_tuned_model
est, meta = load_tuned_model("runs/.../models/...")
yhat = predict_with_tuned_model("runs/.../models/...", feature_dataframe)
```

## Reproduce paper results

Reuse the paper structures (no structure prediction). Descriptors use the default final set (59 features), then AutoML.

| Config | Structures |
|--------|------------|
| `examples/configs/reproduce-paper-abb2.yaml` | `structures_abb2` |
| `examples/configs/reproduce-paper-abb3.yaml` | `structures_abb3` |
| `examples/configs/reproduce-paper-flashabb.yaml` | `structures_flashabb` |

Needs `datasets/` plus the matching `structures_*` tree under the repo root. Outputs go under `runs/reproduce_paper_<backend>/`.

```bash
source kitab.local.env
./kitab.sh validate examples/configs/reproduce-paper-abb2.yaml
./kitab.sh examples/configs/reproduce-paper-abb2.yaml
# ./kitab.sh examples/configs/reproduce-paper-abb3.yaml
# ./kitab.sh examples/configs/reproduce-paper-flashabb.yaml
```