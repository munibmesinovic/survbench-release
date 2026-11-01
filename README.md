# SurvBench

Code for SurvBench. Five models
(Cox, DeepHit, Dynamic-DeepHit, DySurv, a survival transformer) across four
ICU datasets and two admission-anchored arms, five seeds each, plus a
landmarked arm on all four datasets and a cross-dataset transfer arm.
SurvTRACE, SurvLatent ODE, and a summary input for Cox, DeepHit, and SurvTRACE
are also included.

## Setup

    conda env create -f environment.yml
    conda activate survbench

## Reproduce the tables

    for c in configs/*_adm*.yaml configs/*_lmk_cr.yaml; do python preprocess.py --config $c; done

Builds the arrays for the admission-anchored and landmarked arms from the raw data (see below).

    python build_cross_dataset.py --source-config configs/eicu_adm_v1clock.yaml --target-config configs/mimiciv_adm_v1clock.yaml --output-dir arrays/transfer_eicu_to_mimiciv --prefix eicu_to_mimiciv

Builds the transfer arrays from the censored-arm eICU and MIMIC-IV arrays, on the 25 features the two share.

    python train_all.py --arm all

Trains every cell in `configs/hparams.yaml` into `runs/` at the seeds the paper used and skips runs already there.

    python evaluate.py --arm all --runs runs

Writes the mean and SD over seeds of every metric to `metrics.csv` and `metrics.json`; `--arm`, `--dataset`, and `--model` narrow it.

    python bootstrap.py --workers 8

Writes `bootstrap.csv` and `bootstrap.json` with 95% intervals from 1,000 resamples of test patients for each cell and each paired difference in `configs/bootstrap.yaml`.

### The landmarked arm

Metrics come back per landmark and as their mean.

| cohort | landmarks L | reporting windows | horizon H |
| --- | --- | --- | --- |
| MC-MED | 2, 3, 6 h | 2, 6, 12 h | 24 h |
| MIMIC-IV | 6, 12, 24, 48, 72 h | 24, 72, 168 h | 240 h |
| eICU | 6, 12, 24, 48, 72 h | 24, 72, 168 h | 240 h |
| HiRID | 24, 48, 72 h | 24, 72, 168 h | 240 h |

### Patient re-splits

    for s in 1 2 3; do
      for c in configs/*_adm*.yaml configs/*_lmk_cr.yaml; do python preprocess.py --config $c --split-seed $s; done
      python train_all.py --arm both --split-seed $s
      python train_all.py --arm cr_lmk --split-seed $s
      python evaluate.py --arm all --runs runs/s$s --out runs/s$s/metrics
    done

`--split-seed` redraws the patient split and trains seed 42 into `runs/s<N>` at the paper split's hyperparameters, except the summary-input Cox penaliser, which is chosen on validation concordance for each split.

## Retrain or re-tune a cell

    python train.py --arm sr --dataset hirid --model deephit --seed 42
    python tune.py  --arm sr --dataset hirid --model cox

`train.py` trains one cell at the hyperparameters in `configs/hparams.yaml`
and writes to `runs/`. `tune.py` runs
the 20-trial search that picked those hyperparameters and writes to `tuning/`.

### SurvTRACE, SurvLatent ODE, and the summary input

    python train.py --arm cr_lmk --dataset mimiciv --model survtrace-summary --seed 42 --defaults
    python train.py --arm cr_lmk --dataset mcmed --model survlatent_ode --seed 42 --defaults
    python train.py --arm cr_lmk --dataset hirid --model cox-summary --seed 42 --defaults
    python train.py --arm sr --dataset hirid --model deephit-summary --seed 42 --defaults

`-summary` works with `cox`, `deephit`, and `survtrace`. `--defaults` trains at
the model's default hyperparameters instead of `configs/hparams.yaml`.

### Modality ablation

    python train.py --arm sr --dataset mimiciv --model transformer-static --seed 42

`-static`, `-ts`, `-icd`, `-rad`, and `-mmbase` train the transformer on static inputs only, on the time series only, on time series and static plus ICD or plus radiology, and on time series and static at the multi-modal hyperparameters.

## Build the arrays from raw data

All four databases are on PhysioNet, each behind its own credentialing and data
use agreement:

| database | PhysioNet |
| --- | --- |
| MIMIC-IV v3.1 (+ MIMIC-IV-Note v2.2) | `content/mimiciv/3.1/`, `content/mimic-iv-note/2.2/` |
| eICU-CRD v2.0 | `content/eicu-crd/2.0/` |
| HiRID v1.1.1 | `content/hirid/1.1.1/` |
| MC-MED v1.0.1 | `content/mc-med/1.0.1/` |

Put the files under `raw/`, keeping each database's own layout:

    raw/mimiciv/icu/icustays.csv.gz          raw/eicu/patient.csv.gz
    raw/mimiciv/icu/chartevents.csv.gz       raw/eicu/lab.csv.gz
    raw/mimiciv/hosp/patients.csv.gz         raw/eicu/vitalPeriodic.csv.gz
    raw/mimiciv/hosp/admissions.csv.gz       raw/eicu/vitalAperiodic.csv.gz
    raw/mimiciv/hosp/labevents.csv.gz
    raw/mimiciv/hosp/diagnoses_icd.csv.gz    raw/mcmed/visits.csv
    raw/mimiciv/radiology.csv                raw/mcmed/numerics.csv
                                             raw/mcmed/labs.csv
    raw/hirid/general_table.csv              raw/mcmed/pmh.csv
    raw/hirid/imputed_stage/csv/             raw/mcmed/rads.csv
    raw/hirid/benchmark_output/hirid_patient_labels.csv          <- see below
    raw/hirid/benchmark_output/general_table_extended.parquet    <- see below

**MIMIC-IV's `radiology.csv`** comes from the separate MIMIC-IV-Note module. Unzip it into `raw/mimiciv/`, or pass
`--radiology-csv` to point somewhere else.

The two files under `raw/hirid/benchmark_output/` are not PhysioNet files.
They are produced by the HiRID-ICU-Benchmark preprocessing pipeline (Yèche et
al., NeurIPS Datasets & Benchmarks 2021,
<https://github.com/ratschlab/HIRID-ICU-Benchmark>).

To check a database is in place before committing to a long run:

    python preprocess.py --config configs/mimiciv_adm.yaml --check

## Add a dataset

    from data.base_loader import BaseDataLoader
    from survbench.registry import register_dataset


    class MyLoader(BaseDataLoader):
        def load_labels(self): ...
        def load_static_features(self): ...
        def load_timeseries(self): ...
        def apply_cohort_criteria(self, df): ...
        def build_episodes(self): ...


    register_dataset("mydata", MyLoader,
                     {"sr": "mydata_sr.yaml", "cr": "mydata_cr.yaml", "cr_lmk": "mydata_lmk.yaml"},
                     landmarks=[6, 12, 24], windows=[24, 72])

Save it as `mydata.py` and write one config per arm from a shipped one, with `dataset.name: mydata` and its own `output.dir`; `build_episodes()`, `landmarks`, and `windows` are needed only for the landmarked arm.

    for c in mydata_*.yaml; do python preprocess.py --plugin mydata.py --config $c; done
    python train_all.py --plugin mydata.py --dataset mydata --arm all --defaults
    python evaluate.py --plugin mydata.py --dataset mydata --arm all --runs runs
    python bootstrap.py --plugin mydata.py --groups cr/mydata --models cox deephit transformer
    python tune.py --plugin mydata.py --arm cr --dataset mydata --model deephit

`--defaults` trains every registered model, and the `-summary` variants, at default hyperparameters.

## Add a model

    import sys

    import numpy as np
    from sklearn.linear_model import Ridge

    from survbench import registry


    def run(config, *, alpha=1.0, input_mode="summary", seed=42, save_dir=None, save_tag=None,
            skip_test=False):
        train, val, test = registry.load_data(config, input_mode)
        causes, cuts = registry.causes(train), train["cuts"]
        y = np.stack([(train["durations"][:, None] <= cuts) & (train["events"][:, None] == c)
                      for c in causes], 1)
        fit = Ridge(alpha=alpha).fit(train["x"], y.reshape(len(y), -1))

        def cif(split):
            p = fit.predict(split["x"]).reshape(len(split["x"]), len(causes), len(cuts))
            return np.maximum.accumulate(np.clip(p, 0, 1), axis=-1)

        return registry.result(config, "ridge", seed, (train, val, None if skip_test else test),
                               cif, {"alpha": alpha}, save_dir=save_dir, save_tag=save_tag,
                               input_mode=input_mode)


    registry.register_model("ridge", sys.modules[__name__], selection="val.ctd")

    python train_all.py --plugin ridge.py --model ridge --arm all --defaults
    python evaluate.py --plugin ridge.py --model ridge --arm all --runs runs

`registry.result` scores the cumulative incidence on every split, writes `<tag>_preds.npz`, and returns the run record. Pass `space=` to tune it with `tune.py`, `predictor=` to re-score saved models with `evaluate.py --checkpoints`, and `summary=True` to add a `-summary` variant, which calls `run` with `input_mode="summary"` and the base model with `input_mode="static"`.
