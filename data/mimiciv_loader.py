import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data.base_loader import BaseDataLoader
from data.schema import validate_schema
from leakage_guards import assert_icd_history_is_causal
from preprocessing.ranges import apply_ranges_long, merge_range_reports
from preprocessing.units import normalize_dataframe


VITAL_ITEMIDS = {
    220045: 'Heart_Rate', 220179: 'Systolic_BP', 220180: 'Diastolic_BP',
    220210: 'Respiratory_Rate', 223761: 'Temperature_F', 220277: 'SpO2',
    223900: 'GCS_Total', 220050: 'Arterial_BP_Systolic', 220051: 'Arterial_BP_Diastolic',
}
LAB_ITEMIDS = {
    50912: 'Creatinine', 50902: 'Chloride', 50882: 'Bicarbonate', 50893: 'Calcium_Total',
    51006: 'BUN', 51222: 'Hemoglobin', 51221: 'Hematocrit', 51265: 'Platelet_Count',
    51301: 'WBC', 50931: 'Glucose', 50983: 'Sodium', 50971: 'Potassium',
    50960: 'Magnesium', 50813: 'Lactate', 50868: 'Anion_Gap',
}


MIMIC_FEATURE_TO_KIND = {

    "Temperature_F": "temperature",

    "Glucose": "glucose",
    "Hemoglobin": "hemoglobin",
    "Creatinine": "creatinine",
}


class MIMICIVDataLoader(BaseDataLoader):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.files = config['dataset']['files']
        self.modalities_config = config.get('modalities', {})


        self._load_timeseries = self.modalities_config.get('timeseries', True)
        self._load_static = self.modalities_config.get('static', True)
        self._load_icd = self.modalities_config.get('icd', False)
        self._load_radiology = self.modalities_config.get('radiology', False)


        self.n_top_codes = config['icd_processing'].get('top_n_codes', 500)
        self.bert_batch_size = config['radiology_processing'].get('bert_batch_size', 32)
        self.bert_max_length = config['radiology_processing'].get('bert_max_length', 1024)


        self.n_input_steps_hourly = config['preprocessing']['max_hours']
        self.n_windows = config['preprocessing']['num_windows']
        self.window_size_hours = config['preprocessing']['window_size_hours']
        self.max_label_hours = config['preprocessing']['max_horizon_hours']
        self.seed = config['splits']['seed']


        self._cohort_df_cache = None


        self._label_report: Optional[dict] = None


        self._unit_conversion_summary: dict = {}


        self._range_filter_report: dict = {}

        print("\nMIMIC-IV modalities:")
        print(f"  Time-series: {self._load_timeseries}")
        print(f"  Static (Demo/Admin): {self._load_static}")
        print(f"  ICD codes (Causal): {self._load_icd}")
        print(f"  Radiology (Live BERT): {self._load_radiology}")

    def _get_cohort_df(self) -> pd.DataFrame:
        if self._cohort_df_cache is not None:
            return self._cohort_df_cache

        print("\nLoading core tables (icustays, patients, admissions)...")
        icustays = pd.read_csv(os.path.join(self.base_dir, self.files['icustays']),
                               parse_dates=['intime', 'outtime'])
        validate_schema(icustays, "mimiciv", "icustays", self.dataset_version)
        patients = pd.read_csv(os.path.join(self.base_dir, self.files['patients']),
                               parse_dates=['dod'])
        validate_schema(patients, "mimiciv", "patients", self.dataset_version)
        admissions = pd.read_csv(os.path.join(self.base_dir, self.files['admissions']),
                                 parse_dates=['admittime', 'dischtime', 'deathtime'])
        validate_schema(admissions, "mimiciv", "admissions", self.dataset_version)

        cohort_cfg = self.config['dataset'].get('cohort', {})
        min_age = cohort_cfg.get('min_age', 18)
        max_age = cohort_cfg.get('max_age', 120)
        min_stay_hours = cohort_cfg.get('min_stay_hours', 24)

        print(f"Filtering cohort (first stay, age {min_age}-{max_age}, >={min_stay_hours}h)...")
        cohort = icustays.merge(patients, on='subject_id', how='left')
        cohort = cohort.merge(admissions, on=['subject_id', 'hadm_id'], how='left')


        cohort = cohort[cohort.groupby('subject_id')['intime'].rank(method='first') == 1]

        cohort = cohort[(cohort['anchor_age'] >= min_age)
                        & (cohort['anchor_age'] <= max_age)]

        los_hours = (cohort['outtime'] - cohort['intime']).dt.total_seconds() / 3600
        cohort = cohort[los_hours >= min_stay_hours]

        print(f"Full cohort size: {len(cohort)} ICU stays")
        self._cohort_df_cache = cohort
        return self._cohort_df_cache

    def _cache_dir(self) -> Path:
        explicit = self.config.get('dataset', {}).get('cache_dir')
        if explicit:
            return Path(explicit)
        return Path(self.base_dir) / ".cache"

    def _death_clock(
            self, cohort_df: pd.DataFrame
    ) -> Tuple[Optional[pd.Series], Optional[str]]:
        from survbench.core.labels import (
            MIMICIV_ANY_RECORDED_DEATH, MIMICIV_IN_HOSPITAL_DEATH)

        declared = (self.config.get('dataset', {}).get('competing_risks', {})
                    .get('post_icu_death_source', MIMICIV_IN_HOSPITAL_DEATH))
        if declared is None:


            return None, None
        if declared == MIMICIV_IN_HOSPITAL_DEATH:
            return cohort_df['deathtime'], declared
        if declared == MIMICIV_ANY_RECORDED_DEATH:
            return cohort_df['deathtime'].combine_first(cohort_df['dod']), declared
        raise ValueError(
            f"dataset.competing_risks.post_icu_death_source={declared!r} is not valid for MIMIC-IV; "
            f"use {MIMICIV_IN_HOSPITAL_DEATH!r}, {MIMICIV_ANY_RECORDED_DEATH!r} or null")

    def _episode_death_times(self, cohort_df: pd.DataFrame) -> pd.Series:
        clock, source = self._death_clock(cohort_df)
        if source is None:
            return cohort_df['deathtime']
        return clock

    def _build_episodes(self, cohort_df: pd.DataFrame) -> pd.DataFrame:
        from data.episodes import build_icu_episodes, epoch_seconds_from_datetime

        deaths = self._episode_death_times(cohort_df)
        return build_icu_episodes(
            patient_id=cohort_df['subject_id'].to_numpy(),
            stay_id=cohort_df['stay_id'].to_numpy(),
            start=epoch_seconds_from_datetime(
                cohort_df['intime'], column="icustays.intime"),
            end=epoch_seconds_from_datetime(
                cohort_df['outtime'], column="icustays.outtime",
                allow_absent=True),
            death_time=epoch_seconds_from_datetime(
                deaths, column="admissions.deathtime", allow_absent=True),
        )

    def build_episodes(self) -> pd.DataFrame:
        return self._build_episodes(self._get_cohort_df())

    def _create_survival_labels(self, cohort_df: pd.DataFrame) -> pd.DataFrame:
        from survbench.core.labels import label_icu_mortality

        from survbench.core.labels import (MIMICIV_ANY_RECORDED_DEATH,
                                           MIMICIV_IN_HOSPITAL_DEATH)
        from data.episodes import epoch_seconds_from_datetime

        _, source = self._death_clock(cohort_df)
        episodes = self._build_episodes(cohort_df)


        observation_end = None
        if source == MIMICIV_IN_HOSPITAL_DEATH:
            discharge = epoch_seconds_from_datetime(
                cohort_df['dischtime'], column="admissions.dischtime",
                allow_absent=True)
            icu_out = epoch_seconds_from_datetime(
                cohort_df['outtime'], column="icustays.outtime",
                allow_absent=True)


            observation_end = pd.Series(
                np.maximum(
                    discharge.to_numpy(dtype="float64"),
                    icu_out.to_numpy(dtype="float64"),
                ),
                index=cohort_df['subject_id'].to_numpy()).dropna()


        k1_sources = (MIMICIV_IN_HOSPITAL_DEATH,)
        if source == MIMICIV_ANY_RECORDED_DEATH:
            k1_sources = (MIMICIV_IN_HOSPITAL_DEATH, MIMICIV_ANY_RECORDED_DEATH)

        labels = label_icu_mortality(
            episodes,
            post_icu_death_source=source,
            horizon_seconds=np.int64(int(self.max_label_hours * 3600)),
            observation_end=observation_end,
            in_icu_death_time_source=k1_sources,
        )
        self._label_report = labels.to_dict()

        rows = labels.rows
        return pd.DataFrame({
            'stay_id': rows['stay_id'].to_numpy(),
            'duration_hours': rows['duration_seconds'].to_numpy() / 3600.0,
            'event': rows['cause'].to_numpy(),


            'censoring_reason': rows['censoring_reason'].to_numpy(),
        })

    def load_labels(self) -> pd.DataFrame:
        print(f"Creating competing-risks labels (horizon: {int(self.max_label_hours)}h)...")
        cohort = self._get_cohort_df()
        labels_df = self._create_survival_labels(cohort)

        from survbench.core.labels import format_label_report

        for line in format_label_report(self._label_report):
            print(line)


        labels_df = labels_df.merge(cohort[['stay_id', 'subject_id']], on='stay_id')
        labels_df = labels_df.rename(columns={
            'duration_hours': 'duration',
            'stay_id': 'admission_id'
        }).set_index('admission_id')

        return labels_df[['duration', 'event', 'subject_id',
                          'censoring_reason']]

    def load_static_features(self) -> pd.DataFrame:
        if not self._load_static:
            return pd.DataFrame(index=pd.Index([], name='admission_id'))

        print("Extracting static features...")
        cohort = self._get_cohort_df()
        static_df = cohort[
            ['stay_id', 'anchor_age', 'gender', 'first_careunit', 'admission_type', 'admission_location']
        ].copy()

        static_df = pd.get_dummies(
            static_df,
            columns=['gender', 'first_careunit', 'admission_type', 'admission_location'],
            drop_first=False
        )

        static_df = static_df.rename(columns={'stay_id': 'admission_id'}).set_index('admission_id')
        print(f"Loaded {len(static_df)} stays with {len(static_df.columns)} static features.")
        return static_df

    def load_timeseries(self) -> pd.DataFrame:
        if not self._load_timeseries:
            return pd.DataFrame()

        cohort = self._get_cohort_df()

        from preprocessing.chartevents_cache import stream_filter_events

        cache_dir = self._cache_dir()
        hadm_set = set(int(h) for h in cohort['hadm_id'].tolist())

        print("Loading chartevents (vitals) via chartevents_cache ...")
        chartevents = stream_filter_events(
            csv_path=os.path.join(self.base_dir, self.files['timeseries_vitals']),
            hadm_ids=hadm_set,
            itemids=list(VITAL_ITEMIDS.keys()),
            cache_dir=cache_dir,
            source_label="chartevents",
        )
        validate_schema(chartevents, "mimiciv", "chartevents", self.dataset_version)

        chartevents = chartevents.merge(
            cohort[['hadm_id', 'stay_id', 'intime']], on='hadm_id', how='left',
        )
        chartevents['feature'] = chartevents['itemid'].map(VITAL_ITEMIDS)
        chartevents, ce_summary = normalize_dataframe(
            chartevents, feature_col='feature', value_col='valuenum',
            unit_col='valueuom', feature_to_kind=MIMIC_FEATURE_TO_KIND,
        )

        print("Loading labevents (labs) via chartevents_cache ...")
        labevents = stream_filter_events(
            csv_path=os.path.join(self.base_dir, self.files['timeseries_lab']),
            hadm_ids=hadm_set,
            itemids=list(LAB_ITEMIDS.keys()),
            cache_dir=cache_dir,
            source_label="labevents",
        )
        validate_schema(labevents, "mimiciv", "labevents", self.dataset_version)
        labevents = labevents.merge(
            cohort[['hadm_id', 'stay_id', 'intime']], on='hadm_id', how='left',
        )
        labevents['feature'] = labevents['itemid'].map(LAB_ITEMIDS)
        labevents, le_summary = normalize_dataframe(
            labevents, feature_col='feature', value_col='valuenum',
            unit_col='valueuom', feature_to_kind=MIMIC_FEATURE_TO_KIND,
        )

        for s in (ce_summary, le_summary):
            for kind, payload in s.items():
                cur = self._unit_conversion_summary.setdefault(
                    kind,
                    {"converted_rows": 0, "from_units": {}, "unrecognized": {}},
                )
                cur["converted_rows"] += payload["converted_rows"]
                for k, v in payload["from_units"].items():
                    cur["from_units"][k] = cur["from_units"].get(k, 0) + v
                for k, v in payload["unrecognized"].items():
                    cur["unrecognized"][k] = cur["unrecognized"].get(k, 0) + v


        timeseries_raw = pd.concat([
            chartevents[['stay_id', 'charttime', 'feature', 'valuenum', 'intime']],
            labevents[['stay_id', 'charttime', 'feature', 'valuenum', 'intime']]
        ], ignore_index=True)
        timeseries_raw = timeseries_raw.dropna(subset=['stay_id'])
        timeseries_raw['stay_id'] = timeseries_raw['stay_id'].astype(int)

        print("Filtering and binning time-series data...")
        timeseries_raw['hours_since_admit'] = (
                (timeseries_raw['charttime'] - timeseries_raw['intime']).dt.total_seconds() / 3600
        )


        from preprocessing.landmarking import observation_horizon_hours
        load_hours = observation_horizon_hours(self.config, 'mimiciv')
        timeseries_raw = timeseries_raw[
            (timeseries_raw['hours_since_admit'] >= 0) &
            (timeseries_raw['hours_since_admit'] < load_hours)
            ]


        timeseries_raw['hour_bin'] = (timeseries_raw['hours_since_admit']).astype(int)


        timeseries_raw, range_report = apply_ranges_long(
            timeseries_raw, feature_col='feature', value_col='valuenum')
        self._range_filter_report = merge_range_reports(
            self._range_filter_report, range_report)


        ts_wide = timeseries_raw.pivot_table(
            index=['stay_id', 'hour_bin'],
            columns='feature',
            values='valuenum',
            aggfunc='mean'
        )


        all_hours = pd.MultiIndex.from_product([
            cohort['stay_id'].unique(),
            range(load_hours)
        ], names=['stay_id', 'hour_bin'])
        ts_hourly = ts_wide.reindex(all_hours)


        ts_windowed = ts_hourly.rename_axis(['admission_id', 'time_hours'])

        print(f"Final dynamic dataframe shape (Hourly): {ts_windowed.shape}")


        train_ids = self.create_splits()['train']
        missingness_threshold = self.config['preprocessing']['missingness_threshold']
        print(f"Filtering features by missingness (threshold: {missingness_threshold})...")


        ts_train = ts_windowed[ts_windowed.index.get_level_values('admission_id').isin(train_ids)]
        missingness = ts_train.notna().mean()
        cols_to_keep = missingness[missingness >= missingness_threshold].index
        ts_windowed = ts_windowed[cols_to_keep]
        ts_windowed = ts_windowed.sort_index()

        print(f"Kept {len(cols_to_keep)} dynamic features (≥{missingness_threshold * 100}% present in training)")
        return ts_windowed

    def load_icd_codes(self) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        if not self._load_icd:
            return pd.DataFrame(index=pd.Index([], name='admission_id')), None

        cohort = self._get_cohort_df()

        train_stays = self.create_splits()['train']

        print("\nExtracting causally-correct ICD codes...")
        diagnoses = pd.read_csv(os.path.join(self.base_dir, self.files['icd_codes']))
        validate_schema(diagnoses, "mimiciv", "diagnoses_icd", self.dataset_version)


        admissions = pd.read_csv(
            os.path.join(self.base_dir, self.files['admissions']),
            usecols=['hadm_id', 'admittime'],
            parse_dates=['admittime'],
        ).drop_duplicates('hadm_id')
        diagnoses = diagnoses.merge(admissions, on='hadm_id', how='left')

        diagnoses_merged = diagnoses.merge(
            cohort[['stay_id', 'subject_id', 'intime', 'hadm_id']].rename(
                columns={'hadm_id': 'current_hadm_id', 'intime': 'current_intime'}
            ),
            on='subject_id'
        )


        past_diagnoses = diagnoses_merged[
            (diagnoses_merged['hadm_id'] != diagnoses_merged['current_hadm_id'])
            & (diagnoses_merged['admittime'] < diagnoses_merged['current_intime'])
            ]
        assert_icd_history_is_causal(
            past_diagnoses['hadm_id'],
            past_diagnoses['current_hadm_id'],
            past_diagnoses['admittime'],
            past_diagnoses['current_intime'],
        )


        train_diagnoses = past_diagnoses[past_diagnoses['stay_id'].isin(train_stays)]
        top_codes = train_diagnoses['icd_code'].value_counts().head(self.n_top_codes).index.tolist()

        past_diagnoses = past_diagnoses[past_diagnoses['icd_code'].isin(top_codes)]
        icd_matrix = past_diagnoses.groupby(['stay_id', 'icd_code']).size().unstack(fill_value=0)
        icd_matrix = (icd_matrix > 0).astype(int)

        icd_matrix = icd_matrix.reindex(cohort['stay_id'], fill_value=0)
        icd_matrix = icd_matrix.reindex(columns=top_codes, fill_value=0)
        icd_matrix.columns = [f'ICD_{code}' for code in icd_matrix.columns]

        icd_df = icd_matrix.rename_axis(index='admission_id')


        cohort_ids = self.cohort_df.index.unique()
        icd_df = icd_df.loc[icd_df.index.isin(cohort_ids)]

        print(f"Created ({len(icd_df)}, {len(icd_df.columns)}) visit-ICD matrix.")
        return icd_df, None

    def load_radiology_reports(self) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        if not self._load_radiology:
            return pd.DataFrame(index=pd.Index([], name='admission_id')), None

        cohort = self._get_cohort_df()


        rad_path = self.base_dir / self.files['radiology_embeddings']

        print(f"\nExtracting {self.files['radiology_embeddings']} (Radiology Reports)...")


        radiology = pd.read_csv(rad_path, parse_dates=['charttime', 'storetime'])
        validate_schema(radiology, "mimiciv", "radiology", self.dataset_version)

        radiology = radiology.merge(
            cohort[['stay_id', 'subject_id', 'hadm_id', 'intime']],
            on=['subject_id', 'hadm_id'],
            how='inner'
        )


        radiology['hours_since_intime'] = (
                (radiology['storetime'] - radiology['intime']).dt.total_seconds() / 3600
        )
        from preprocessing.landmarking import observation_horizon_hours
        load_horizon = observation_horizon_hours(self.config, 'mimiciv')
        radiology = radiology[
            (radiology['hours_since_intime'] >= 0) &
            (radiology['hours_since_intime'] < load_horizon)
            ]

        radiology = radiology.dropna(subset=['text'])

        if len(radiology) == 0:
            print("No valid radiology reports found for this cohort.")
            return pd.DataFrame(index=pd.Index([], name='admission_id')), None

        from preprocessing.text_embedding_cache import get_or_compute_embeddings

        rad_cfg = self.config.get('radiology_processing', {})
        model_name = rad_cfg.get('model_name', 'yikuan8/Clinical-Longformer')
        max_length = rad_cfg.get('max_length', self.bert_max_length)
        batch_size = rad_cfg.get('batch_size', self.bert_batch_size)
        pooling = rad_cfg.get('pooling', 'mean')


        emb = get_or_compute_embeddings(
            texts=radiology['text'].tolist(),
            cache_dir=self._cache_dir() / "embeddings",
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            pooling=pooling,
        )


        embedding_cols = [f'Radio_{i}' for i in range(emb.shape[1])]
        radio_df = pd.DataFrame(emb, columns=embedding_cols)
        radio_df['admission_id'] = radiology['stay_id'].values
        radio_df['hours_since_intime'] = radiology['hours_since_intime'].values

        print(f"Embedded {len(radio_df):,} reports across "
              f"{radio_df['admission_id'].nunique():,} stays "
              f"(pooled per landmark downstream).")
        return radio_df, None

    def apply_cohort_criteria(self, df: pd.DataFrame) -> pd.DataFrame:


        if 'subject_id' not in df.columns:
            raise ValueError("Critical Error: 'subject_id' was not found in cohort_df before splitting.")

        print(f"Cohort criteria applied: {len(df)} stays remaining.")
        return df
