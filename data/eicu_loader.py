import gc
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from data.base_loader import BaseDataLoader
from data.schema import validate_schema
from preprocessing.ranges import (apply_ranges_long, hourly_mean_with_ranges,
                                  merge_range_reports)
from preprocessing.units import normalize_dataframe


_STAY_BATCH = 5_000

EICU_FEATURE_TO_KIND = {
    "glucose": "glucose",
    "hemoglobin": "hemoglobin",
    "creatinine": "creatinine",
}


class eICUDataLoader(BaseDataLoader):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.files = config['dataset']['files']
        self.max_horizon = config['preprocessing']['max_horizon_hours']
        self.max_hours_input = config['preprocessing']['max_hours']


        self._cohort_df_cache = None

        self._label_report = None


        self._unit_conversion_summary: dict = {}


        self._range_filter_report: dict = {}

    def _get_cohort_df(self) -> pd.DataFrame:
        if self._cohort_df_cache is not None:
            return self._cohort_df_cache

        patient_path = self.base_dir / self.files['patient']
        print(f"Loading core cohort from: {patient_path}")


        cols_to_load = [
            'patientunitstayid', 'patienthealthsystemstayid', 'uniquepid',
            'gender', 'age', 'ethnicity', 'unitstaytype',
            'unitdischargeoffset', 'unitdischargestatus',


            'hospitaladmitoffset', 'hospitaldischargeoffset',
            'hospitaldischargestatus',
        ]

        cohort_df = pd.read_csv(patient_path, usecols=cols_to_load)
        validate_schema(cohort_df, "eicu", "patient", self.dataset_version)


        n_all_stays = len(cohort_df)
        cohort_df = (cohort_df
                     .sort_values(['patienthealthsystemstayid',
                                   'hospitaladmitoffset', 'patientunitstayid'],
                                  ascending=[True, False, True])
                     .groupby('patienthealthsystemstayid', sort=False)
                     .head(1))
        n_non_index = n_all_stays - len(cohort_df)


        cohort_cfg = self.config['dataset'].get('cohort', {})
        min_age = cohort_cfg.get('min_age', 18)
        max_age = cohort_cfg.get('max_age', 89)
        cohort_df = cohort_df[cohort_df['age'] != '> 89']
        cohort_df['age'] = pd.to_numeric(cohort_df['age'], errors='coerce')
        cohort_df = cohort_df.dropna(subset=['age'])
        cohort_df = cohort_df[(cohort_df['age'] >= min_age) & (cohort_df['age'] <= max_age)]


        cohort_df['duration_hours'] = cohort_df['unitdischargeoffset'] / 60.0


        min_hours = self.config['dataset']['cohort']['min_stay_hours']
        cohort_df = cohort_df[cohort_df['duration_hours'] >= min_hours]


        cohort_df = cohort_df.rename(columns={
            'patientunitstayid': 'admission_id',
            'uniquepid': 'subject_id'
        })

        cohort_df = cohort_df.set_index('admission_id')

        print(f"Loaded {len(cohort_df)} index ICU stays ({n_non_index:,d} later stays dropped), "
              f"{cohort_df['subject_id'].nunique():,d} patients.")
        self._cohort_df_cache = cohort_df
        return cohort_df

    def _build_episodes(self, cohort_df: pd.DataFrame) -> pd.DataFrame:
        from data.episodes import build_icu_episodes

        minutes_to_seconds = 60
        start = (-cohort_df['hospitaladmitoffset'].to_numpy(dtype='int64')
                 * minutes_to_seconds)
        end = start + (cohort_df['unitdischargeoffset'].to_numpy(dtype='int64')
                       * minutes_to_seconds)

        died_in_icu = (cohort_df['unitdischargestatus'] == 'Expired').to_numpy()
        died_in_hospital = (
            cohort_df['hospitaldischargestatus'] == 'Expired').to_numpy()
        hospital_discharge = start + (
            cohort_df['hospitaldischargeoffset'].to_numpy(dtype='int64')
            * minutes_to_seconds)


        hospital_death = np.maximum(hospital_discharge, end)

        death = pd.Series(pd.NA, index=range(len(cohort_df)), dtype='Int64')


        death[died_in_hospital] = hospital_death[died_in_hospital]
        death[died_in_icu] = end[died_in_icu]

        return build_icu_episodes(
            patient_id=cohort_df['patienthealthsystemstayid'].to_numpy(),
            stay_id=cohort_df.index.to_numpy(),
            start=start,
            end=end,
            death_time=death,
        )

    def _cache_dir(self) -> Path:
        explicit = self.config.get('dataset', {}).get('cache_dir')
        if explicit:
            return Path(explicit)
        return Path(self.base_dir) / ".cache"

    def build_episodes(self) -> pd.DataFrame:
        return self._build_episodes(self._get_cohort_df())

    def _death_clock_and_observation_end(self, cohort_df):
        from survbench.core.labels import EICU_IN_HOSPITAL_DEATH

        declared = (self.config.get('dataset', {}).get('competing_risks', {})
                    .get('post_icu_death_source', EICU_IN_HOSPITAL_DEATH))
        if declared is None:


            return None, None
        if declared != EICU_IN_HOSPITAL_DEATH:
            raise ValueError(
                f"dataset.competing_risks.post_icu_death_source={declared!r} is not valid for eICU; "
                f"use {EICU_IN_HOSPITAL_DEATH!r} or null")
        if cohort_df is None:
            return declared, None


        minutes_to_seconds = 60
        start = (-cohort_df['hospitaladmitoffset'].to_numpy(dtype='int64')
                 * minutes_to_seconds)
        hospital_discharge = start + (
            cohort_df['hospitaldischargeoffset'].to_numpy(dtype='int64')
            * minutes_to_seconds)


        icu_end = start + (
            cohort_df['unitdischargeoffset'].to_numpy(dtype='int64')
            * minutes_to_seconds)
        observation_end = pd.Series(
            np.maximum(hospital_discharge, icu_end),
            index=cohort_df['patienthealthsystemstayid'].to_numpy())
        return declared, observation_end[~observation_end.index.duplicated()]

    def load_labels(self) -> pd.DataFrame:
        from survbench.core.labels import (EICU_IN_HOSPITAL_DEATH,
                                           EICU_IN_ICU_DEATH,
                                           label_icu_mortality)

        cohort_df = self._get_cohort_df()


        source, observation_end = self._death_clock_and_observation_end(
            cohort_df)

        labels = label_icu_mortality(
            self._build_episodes(cohort_df),
            post_icu_death_source=source,
            horizon_seconds=np.int64(int(self.max_horizon * 3600)),
            observation_end=observation_end,


            in_icu_death_time_source=(EICU_IN_HOSPITAL_DEATH,
                                      EICU_IN_ICU_DEATH),
        )
        self._label_report = labels.to_dict()

        rows = labels.rows
        stay_ids = rows['stay_id'].to_numpy()
        labels_df = pd.DataFrame({
            'duration': rows['duration_seconds'].to_numpy() / 3600.0,
            'event': rows['cause'].to_numpy(),

            'subject_id': cohort_df.loc[stay_ids, 'subject_id'].to_numpy(),

            'censoring_reason': rows['censoring_reason'].to_numpy(),

        }, index=pd.Index(stay_ids, name=cohort_df.index.name))

        from survbench.core.labels import format_label_report

        for line in format_label_report(self._label_report):
            print(line)
        return labels_df

    def load_static_features(self) -> pd.DataFrame:
        cohort_df = self._get_cohort_df()

        static_cols = ['age', 'gender', 'ethnicity', 'unitstaytype']
        static_df = cohort_df[static_cols].copy()


        static_df = pd.get_dummies(
            static_df,
            columns=['gender', 'ethnicity', 'unitstaytype'],
            dtype=int
        )


        print(f"Loaded {len(static_df.columns)} static features.")
        return static_df

    def load_timeseries(self) -> pd.DataFrame:
        cohort_ids = self.cohort_df.index.unique()
        train_ids = self.create_splits()['train']


        from preprocessing.landmarking import observation_horizon_hours
        load_hours = observation_horizon_hours(self.config, 'eicu')
        max_offset_minutes = load_hours * 60


        vital_p_path = self.base_dir / self.files['vital_periodic']
        print(f"Loading and aggregating vitalPeriodic from: {vital_p_path}...")
        vital_p_cols = [
            'patientunitstayid', 'observationoffset', 'temperature', 'sao2',
            'heartrate', 'respiration', 'systemicsystolic', 'systemicdiastolic'
        ]
        vital_p_df = pd.read_csv(vital_p_path, usecols=vital_p_cols)
        validate_schema(vital_p_df, "eicu", "vitalPeriodic", self.dataset_version)
        vital_p_df = vital_p_df[vital_p_df['patientunitstayid'].isin(cohort_ids)]
        vital_p_df = vital_p_df[
            (vital_p_df['observationoffset'] >= 0) &
            (vital_p_df['observationoffset'] < max_offset_minutes)
            ]


        vital_p_df['hour_bin'] = (vital_p_df['observationoffset'] / 60.0).astype(int)
        vital_p_cols_to_agg = [col for col in vital_p_cols if col not in ['patientunitstayid', 'observationoffset']]


        vitals_p_hourly, report = hourly_mean_with_ranges(
            vital_p_df, by=['patientunitstayid', 'hour_bin'],
            value_cols=vital_p_cols_to_agg)
        self._range_filter_report = merge_range_reports(
            self._range_filter_report, report)
        print(f"  Aggregated {len(vital_p_df)} rows to {len(vitals_p_hourly)} hourly records.")
        del vital_p_df
        gc.collect()


        vital_a_path = self.base_dir / self.files['vital_aperiodic']
        print(f"Loading and aggregating vitalAperiodic from: {vital_a_path}...")
        vital_a_cols = [
            'patientunitstayid', 'observationoffset', 'noninvasivesystolic',
            'noninvasivediastolic'
        ]
        vital_a_df = pd.read_csv(vital_a_path, usecols=vital_a_cols)
        validate_schema(vital_a_df, "eicu", "vitalAperiodic", self.dataset_version)
        vital_a_df = vital_a_df[vital_a_df['patientunitstayid'].isin(cohort_ids)]
        vital_a_df = vital_a_df[
            (vital_a_df['observationoffset'] >= 0) &
            (vital_a_df['observationoffset'] < max_offset_minutes)
            ]


        vital_a_df['hour_bin'] = (vital_a_df['observationoffset'] / 60.0).astype(int)
        vital_a_cols_to_agg = [col for col in vital_a_cols if col not in ['patientunitstayid', 'observationoffset']]

        vitals_a_hourly, report = hourly_mean_with_ranges(
            vital_a_df, by=['patientunitstayid', 'hour_bin'],
            value_cols=vital_a_cols_to_agg)
        self._range_filter_report = merge_range_reports(
            self._range_filter_report, report)
        print(f"  Aggregated {len(vital_a_df)} rows to {len(vitals_a_hourly)} hourly records.")
        del vital_a_df
        gc.collect()


        lab_path = self.base_dir / self.files['lab']
        print(f"Loading and aggregating labs from: {lab_path}...")
        labs_df = pd.read_csv(lab_path, usecols=[
            'patientunitstayid', 'labresultoffset', 'labname', 'labresult',
            'labmeasurenamesystem',
        ], on_bad_lines='skip')
        validate_schema(labs_df, "eicu", "lab", self.dataset_version)
        labs_df = labs_df[labs_df['patientunitstayid'].isin(cohort_ids)]
        labs_df = labs_df.dropna(subset=['labresult'])
        labs_df = labs_df[
            (labs_df['labresultoffset'] >= 0) &
            (labs_df['labresultoffset'] < max_offset_minutes)
            ]


        unit_col = 'labmeasurenamesystem' if 'labmeasurenamesystem' in labs_df.columns else None
        labs_df, lab_summary = normalize_dataframe(
            labs_df,
            feature_col='labname',
            value_col='labresult',
            unit_col=unit_col,
            feature_to_kind=EICU_FEATURE_TO_KIND,
        )
        for kind, payload in lab_summary.items():
            cur = self._unit_conversion_summary.setdefault(
                kind,
                {"converted_rows": 0, "from_units": {}, "unrecognized": {}},
            )
            cur["converted_rows"] += payload["converted_rows"]
            for k, v in payload["from_units"].items():
                cur["from_units"][k] = cur["from_units"].get(k, 0) + v
            for k, v in payload["unrecognized"].items():
                cur["unrecognized"][k] = cur["unrecognized"].get(k, 0) + v


        labs_df, lab_range_report = apply_ranges_long(
            labs_df, feature_col='labname', value_col='labresult')
        self._range_filter_report = merge_range_reports(
            self._range_filter_report, lab_range_report)

        labs_wide_raw = labs_df.pivot_table(
            index=['patientunitstayid', 'labresultoffset'],
            columns='labname',
            values='labresult',
            aggfunc='mean',
        )
        del labs_df
        gc.collect()


        labs_wide_raw = labs_wide_raw.reset_index()
        labs_wide_raw['hour_bin'] = (labs_wide_raw['labresultoffset'] / 60.0).astype(int)
        lab_cols = [col for col in labs_wide_raw.columns if
                    col not in ['patientunitstayid', 'labresultoffset', 'hour_bin']]


        labs_hourly, report = hourly_mean_with_ranges(
            labs_wide_raw, by=['patientunitstayid', 'hour_bin'],
            value_cols=lab_cols)
        self._range_filter_report = merge_range_reports(
            self._range_filter_report, report)
        print(f"  Aggregated and pivoted lab rows to {len(labs_hourly)} hourly records.")
        del labs_wide_raw
        gc.collect()


        vitals_p_hourly = vitals_p_hourly.astype("float32")
        vitals_a_hourly = vitals_a_hourly.astype("float32")
        labs_hourly = labs_hourly.astype("float32")


        print(f"Merging aggregated hourly data in batches of "
              f"{_STAY_BATCH:,d} stays...")
        for frame in (vitals_p_hourly, vitals_a_hourly, labs_hourly):
            frame.sort_index(inplace=True)

        all_ids = sorted(
            set(vitals_p_hourly.index.get_level_values(0))
            | set(vitals_a_hourly.index.get_level_values(0))
            | set(labs_hourly.index.get_level_values(0)))


        spill_dir = Path(self._cache_dir()) / "eicu_hourly_batches"
        if spill_dir.exists():
            shutil.rmtree(spill_dir)
        spill_dir.mkdir(parents=True, exist_ok=True)

        n_batches = 0
        for start in range(0, len(all_ids), _STAY_BATCH):
            batch_ids = all_ids[start:start + _STAY_BATCH]
            lo, hi = batch_ids[0], batch_ids[-1]
            part = (vitals_p_hourly.loc[lo:hi]
                    .join(vitals_a_hourly.loc[lo:hi], how='outer')
                    .join(labs_hourly.loc[lo:hi], how='outer'))
            grid = pd.MultiIndex.from_product(
                [batch_ids, range(load_hours)],
                names=['patientunitstayid', 'hour_bin'])


            part = part.reindex(grid)
            part.to_parquet(spill_dir / f"batch_{n_batches:04d}.parquet")
            n_batches += 1
            del part
            gc.collect()

        del vitals_p_hourly, vitals_a_hourly, labs_hourly
        gc.collect()


        import resource

        def _rss_gib() -> float:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20

        print(f"  Reading back {n_batches} hourly batches "
              f"(peak RSS {_rss_gib():.1f} GiB before)...")
        loaded = []
        for i, path in enumerate(sorted(spill_dir.glob("batch_*.parquet"))):
            loaded.append(pd.read_parquet(path))
            if (i + 1) % 5 == 0 or i + 1 == n_batches:
                print(f"    {i + 1}/{n_batches} read, peak RSS "
                      f"{_rss_gib():.1f} GiB")
        print(f"  Concatenating {len(loaded)} batches "
              f"(peak RSS {_rss_gib():.1f} GiB before concat)...")
        ts_df = pd.concat(loaded, copy=False)
        del loaded
        gc.collect()
        print(f"  Concatenated: {ts_df.shape}, peak RSS {_rss_gib():.1f} GiB")
        shutil.rmtree(spill_dir, ignore_errors=True)
        gc.collect()


        ts_df.index = ts_df.index.set_names(['admission_id', 'time_hours'])
        if not ts_df.index.is_monotonic_increasing:
            ts_df = ts_df.sort_index()


        print("Filtering features by missingness...")
        missingness_threshold = self.config['preprocessing']['missingness_threshold']


        is_train = np.asarray(
            ts_df.index.get_level_values('admission_id').isin(train_ids),
            dtype=bool)


        from preprocessing.imputation import presence_after_ffill


        cohort_for_los = self._get_cohort_df()
        los = cohort_for_los['duration_hours']
        if cohort_for_los.index.name != 'admission_id':
            los = cohort_for_los.set_index('admission_id')['duration_hours']
        los = los[~los.index.duplicated()]
        hours = ts_df.index.get_level_values('time_hours').to_numpy(float)
        stays = ts_df.index.get_level_values('admission_id')
        in_unit = hours < los.reindex(stays).to_numpy(dtype=float)
        print(f"  presence denominator: {in_unit[is_train].mean():.1%} of grid "
              f"rows are within the stay")
        missingness = presence_after_ffill(ts_df, level='admission_id',
                                           valid_rows=in_unit & is_train)
        cols_to_keep = missingness[missingness >= missingness_threshold].index
        ts_df = ts_df[cols_to_keep]

        print(f"Kept {len(cols_to_keep)} dynamic features (>= {missingness_threshold * 100}% present in training)")

        del hours, stays, in_unit, is_train
        gc.collect()

        return ts_df

    def apply_cohort_criteria(self, df: pd.DataFrame) -> pd.DataFrame:

        if 'subject_id' not in df.columns:
            raise ValueError("Critical Error: 'subject_id' was not found.")
        print(f"Cohort criteria applied: {len(df)} stays remaining.")
        return df
