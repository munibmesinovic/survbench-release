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

    python evaluate.py --arm both

Writes `metrics.csv` and `metrics.json`. Runs all 220 cells in a few minutes
on one GPU. Use `--arm sr` or `--arm cr` for one arm, and `--dataset hirid`
or `--model cox` to narrow it further.

`--arm both` is the two admission-anchored arms, the 220 cells above.
`--arm all` adds the transfer arm and the landmarked arm.

### The landmarked arm

    python evaluate.py --arm cr_lmk
    python evaluate.py --arm cr_lmk --dataset mimiciv

Metrics come back per landmark and as their mean.

| cohort | landmarks L | reporting windows | horizon H |
| --- | --- | --- | --- |
| MC-MED | 2, 3, 6 h | 2, 6, 12 h | 24 h |
| MIMIC-IV | 6, 12, 24, 48, 72 h | 24, 72, 168 h | 240 h |
| eICU | 6, 12, 24, 48, 72 h | 24, 72, 168 h | 240 h |
| HiRID | 24, 48, 72 h | 24, 72, 168 h | 240 h |

    python preprocess.py --config configs/mcmed_lmk_cr.yaml
    python preprocess.py --config configs/mimiciv_lmk_cr.yaml
    python preprocess.py --config configs/eicu_lmk_cr.yaml
    python preprocess.py --config configs/hirid_lmk_cr.yaml

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

Subclass `data.base_loader.BaseDataLoader` and implement four methods:

    load_labels()          load_static_features()
    load_timeseries()      apply_cohort_criteria(df)

Cohort assembly, patient-level splitting, competing-risks stratification,
time-windowing, imputation, scaling and the missingness masks are handled by
the base class and the pipeline, so an adapter only describes how to read one
database.

Register it before building the pipeline, then write a config with a matching
`dataset.name`:

    from preprocessing.pipeline import register_loader
    register_loader("mydataset", MyLoader)
