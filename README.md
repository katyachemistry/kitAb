# FASTAb

## Installation

From the repo root (requires **mamba** or **conda**, **git**, **wget** or **curl**):

```bash
./install.sh                    # fastab + abb2 + abb3 (default)
./install.sh --fastab-only      # fastab only (descriptors / automl)
./install.sh --no-abb2            # fastab + abb3 (skip abb2)
source fastab.local.env
```

The `fastab` env includes [MMseqs2](https://github.com/soedinglab/mmseqs2) (`mmseqs` on PATH via the `mmseqs2` bioconda package) for automatic sequence-identity cross-validation splits.

## Scenario 1: You have antibodies sequences, assay measurements and, optionally, some external features. You want to get descriptors and predictions. 

The toolkit is run with the help of config yaml files. See **configs/scenario1.yaml**

Most often, in this scenario you would like to configure two fields in the yaml: ```input_csvs_folder``` and ```result_folder```. You can pass several datasets by putting them under ```input_csvs_folder``` root.

```text
FASTAb/
├── datasets/       # input_csvs_folder                                    
│   ├── dataset1.csv                      
│   │       name,heavy,light,target_Tm,target_HIC,feature_isotype
│   │       mAb1,EVQLVES...,DIQMTQ...,55,0.7,lambda
│   │       mAb2,EVQLVES...,DIQMTQ...,57,1.2,kappa
│   ├── dataset2.csv                                    
│   └── …
```

One dataset = one csv file. It must contain "name", "heavy", "light" and "target_X", "target_Y", ..., "feature_A", "feature_B", ... columns. Each `name` must be unique, and the concatenated `heavy`+`light` sequence must be unique across rows (checked when the run config is prepared and again before AutoML merge).

Try out the scenario using example. 

```
bash fastab.sh configs/scenario1.yaml
``` 

Default AutoML hyperparameters are used. To customize them, edit **src/automl.yaml** (or set ``automl_config:`` in your run config).

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

AutoML uses all developability descriptor groups for every target (`surface`, `core`, `general`, `sequence_motives`) unless a run config block sets `developability_features` explicitly.

---

## Scenario 2: Same as Scenario 1 + you already have structures (PDB/CIF).

See **configs/scenario2.yaml**.

Structures must contain Fv-fragments. Only they will be used for the pipeline.  

Data organization must follow:

```text
FASTAb/
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
Configure parent folder names in yaml (```input_csvs_folder``` and ```input_structures_folder``` fields).
Other namings are essential to the work of the program - they must coincide.

Another important field in scenario 2 is ```structures_processing```.
If your structures are not IMGT-numbered, you **must** set **renumber: True**. It requires abb2 env installed. If you want to minimize your structures, set **minimize: True** (also needs abb2).

---

## Scenario 2a: Descriptors only on existing structures (no CSV datasets)

See **configs/scenario2a.yaml**.

Use this when you already have PDB/mmCIF files and only need developability descriptors (no assay CSVs, no AutoML). Set ```input_structures_folder``` and ```result_folder```; omit ```input_csvs_folder```.

Supported layouts:

```text
FASTAb/
├── structures_abb2_paired_oas_sanity_skip_minimized/   # flat folder
│   ├── abngs_oas_paired_merged_data_paired_5457817.pdb
│   └── …
```

or nested batch folders (same convention as scenario 2):

```text
FASTAb/
├── structures_abb3/
│   ├── ab21_abb3_1/
│   │   ├── mAb1.pdb
│   │   └── …
│   └── …
```

Optional ```structures_processing``` (minimize / IMGT renumber) runs in place before descriptors, as in scenario 2.

```bash
bash fastab.sh configs/scenario2a.yaml
```

---

## Scenario 3: You only want to run descriptors calculation, no predictions from AutoML part. 

Depending on whether you have structures or not, prepare input and configure relevant fields in yaml (see scenarios 1 and 2). We give example **scenario3.yaml** for no-structures case. 
This scenario only differs from previous in 1 field: ```no_automl```. Configure it to True.

## Scenario 4: You only want to run AutoML part on your own existing descriptors. 

If you already have calculated developability descriptors (e.g. from previous runs) and want to run only the AutoML part, configure ```calculate_descriptors: False``` in your config YAML file. See **configs/scenario4.yaml**.

Provide the path to the folder containing your pre-calculated descriptors under ```predefined_descriptors```, and list the allowed suffixes for run names ``{dataset_stem}{suffix}``. For each match, the pipeline finds the corresponding dataset CSV in ```input_csvs_folder``` by stem. Layout is detected automatically (vendor-agnostic):

- **Subfolders**: ``{dataset_stem}{suffix}/features.csv`` (optional ```features_filename```), or ``{dataset_stem}{suffix}/results/*.json`` (FASTAb-style JSON).
- **Flat CSVs** (when no matching subfolders exist): ``{dataset_stem}{suffix}.csv`` in the folder root.

If any ``{dataset_stem}{suffix}`` subfolders exist, flat CSVs in the root are ignored.

```yaml
input_csvs_folder: datasets
result_folder: scenario4

calculate_descriptors: False

predefined_descriptors:
  folder: descriptors_reproducibility_finalization
  allowed_suffixes:
    - _abb2_4
```

To run this scenario, execute:

```bash
bash fastab.sh configs/scenario4.yaml
```