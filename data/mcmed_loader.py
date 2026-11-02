import gc
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data.base_loader import BaseDataLoader
from data.schema import validate_schema
from preprocessing.ranges import hourly_mean_with_ranges, merge_range_reports
from preprocessing.units import normalize_dataframe

_UNIT_PATTERN = re.compile(r"\(([^)]+)\)\s*$")


_SECONDS_PER_HOUR = 3600
_SECONDS_PER_DAY = 86400


_READ_CHUNK_ROWS = 2_000_000


def _to_epoch_seconds_or_drop(frame: pd.DataFrame, column: str,
                              *, table: str) -> tuple[pd.DataFrame, int]:
    from survbench.ir.time import absent_mask, to_epoch_seconds

    absent = absent_mask(frame[column])
    n_absent = int(absent.sum())
    if n_absent:
        frame = frame.loc[~absent]
    if frame.empty:
        return frame.assign(**{column: pd.Series(dtype="int64")}), n_absent
    converted = to_epoch_seconds(frame[column], column=f"{table}.{column}")
    return frame.assign(**{column: converted.to_numpy(dtype="int64")}), n_absent


def _to_epoch_seconds_masked(frame: pd.DataFrame, column: str,
                             *, table: str) -> pd.DataFrame:
    from survbench.ir.time import to_epoch_seconds_masked

    converted = to_epoch_seconds_masked(frame[column], column=f"{table}.{column}")
    return frame.assign(**{column: converted.to_numpy(dtype="object")}).astype(
        {column: "Int64"})


def _read_cohort_chunks(path, *, table: str, version, cohort_csns,
                        time_column: str,
                        chunksize: int = _READ_CHUNK_ROWS) -> pd.DataFrame:
    kept: list[pd.DataFrame] = []
    n_rows_read = 0
    n_absent_total = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        n_rows_read += len(chunk)
        validate_schema(chunk, "mcmed", table, version)
        chunk = chunk[chunk['CSN'].isin(cohort_csns)]
        if chunk.empty:
            continue
        chunk, n_absent = _to_epoch_seconds_or_drop(
            chunk, time_column, table=table)
        n_absent_total += n_absent
        if not chunk.empty:
            kept.append(chunk)

    frame = (pd.concat(kept, ignore_index=True) if kept
             else pd.DataFrame(columns=['CSN', time_column]))
    print(f"  {table}: read {n_rows_read:,d} rows, kept {len(frame):,d} in "
          f"cohort; dropped {n_absent_total:,d} with no {time_column}.")
    return frame


def _split_measure_unit(measure):
    m = _UNIT_PATTERN.search(str(measure))
    if not m:
        return measure, None
    name = _UNIT_PATTERN.sub("", measure).strip()
    return name, m.group(1).strip()


MCMED_FEATURE_TO_KIND = {
    "Glucose": "glucose",
    "Hemoglobin": "hemoglobin",
    "Creatinine": "creatinine",
    "Temp": "temperature",
    "Temperature": "temperature",
}


class MCMEDDataLoader(BaseDataLoader):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.files = config['dataset']['files']
        self.modalities_config = config.get('modalities', {})


        self._label_report: Optional[Dict] = None


        self._load_timeseries = self.modalities_config.get('timeseries', True)
        self._load_static = self.modalities_config.get('static', True)
        self._load_icd = self.modalities_config.get('icd', False)
        self._load_radiology = self.modalities_config.get('radiology', False)


        self.n_top_codes = config['icd_processing'].get('top_n_codes', 500)
        self.bert_batch_size = config['radiology_processing'].get('bert_batch_size', 32)
        self.bert_max_length = config['radiology_processing'].get('bert_max_length', 1024)


        self._visits_df_cache = None
        self._pmh_df_cache = None
        self._rad_df_cache = None


        self._unit_conversion_summary: dict = {}


        self._range_filter_report: dict = {}

        print("\nMC-MED modalities:")
        print(f"  Time-series: {self._load_timeseries}")
        print(f"  Static (Triage/Demo): {self._load_static}")
        print(f"  ICD codes: {self._load_icd}")
        print(f"  Radiology (Live BERT): {self._load_radiology}")

    def _get_visits_df(self) -> pd.DataFrame:
        if self._visits_df_cache is not None:
            return self._visits_df_cache

        label_path = self.base_dir / self.files['labels']
        print(f"\nProcessing {self.files['labels']} (Demographics, Triage, & Outcomes)...")
        visits = pd.read_csv(label_path)
        validate_schema(visits, "mcmed", "visits", self.dataset_version)


        core_cols = ['Race', 'Ethnicity', 'Triage_Temp', 'Triage_HR', 'Triage_RR',
                     'Triage_SpO2', 'Triage_SBP', 'Triage_DBP', 'Triage_acuity', 'ED_LOS', 'ED_dispo']
        visits = visits.dropna(subset=core_cols)


        keep_cols = ['MRN', 'CSN', 'Age', 'Gender', 'Race', 'Ethnicity',
                     'Triage_Temp', 'Triage_HR', 'Triage_RR', 'Triage_SpO2',
                     'Triage_SBP', 'Triage_DBP', 'Triage_acuity', 'ED_LOS', 'ED_dispo',
                     'Arrival_time', 'Dispo_time']
        visits = visits[[c for c in keep_cols if c in visits.columns]]
        if 'Arrival_time' in visits.columns:


            visits, n_no_arrival = _to_epoch_seconds_or_drop(
                visits, 'Arrival_time', table='visits')
            if n_no_arrival:
                print(f"  Dropped {n_no_arrival:,d} visits with no Arrival_time.")
        if 'Dispo_time' in visits.columns:
            visits = _to_epoch_seconds_masked(visits, 'Dispo_time',
                                              table='visits')


        ed_dispo_mapping = {'ICU': 1, 'Inpatient': 2, 'Observation': 3,
                            'Discharge': 4}
        visits['Outcome'] = visits['ED_dispo'].map(ed_dispo_mapping)
        visits = visits.dropna(subset=['Outcome'])
        visits['Outcome'] = visits['Outcome'].astype(int)


        visits.set_index('CSN', inplace=True)


        if visits.index.has_duplicates:
            duplicated = visits.index[visits.index.duplicated()].unique()
            raise ValueError(
                f"CSN is not unique in visits.csv: {len(duplicated):,d} repeated value(s), first {duplicated[0]!r}")


        max_label_hours = self.config['preprocessing']['max_horizon_hours']
        n_truncated = int((self._event_hours(visits) > max_label_hours).sum())

        print(f"Loaded {len(visits)} valid visits "
              f"({n_truncated:,d} still in the ED at {max_label_hours}h, kept "
              f"and truncated).")

        self._visits_df_cache = visits
        return self._visits_df_cache

    def _cache_dir(self) -> Path:
        explicit = self.config.get('dataset', {}).get('cache_dir')
        if explicit:
            return Path(explicit)
        return Path(self.base_dir) / ".cache"

    def _event_hours(self, visits: pd.DataFrame) -> pd.Series:
        clock = self.config['dataset'].get('event_clock', 'departure')
        departure = visits['ED_LOS'].astype('float64')
        if clock == 'departure':
            return departure
        if clock != 'decision':
            raise ValueError(
                f"dataset.event_clock = {clock!r}; expected 'departure' or "
                f"'decision'.")
        dispo = visits['Dispo_time'].astype('Float64')
        arrival = visits['Arrival_time'].astype('float64')
        decision = ((dispo - arrival) / _SECONDS_PER_HOUR).astype('float64')
        decision = decision.fillna(departure)
        return decision.clip(lower=0.0).where(decision <= departure, departure)

    def load_labels(self) -> pd.DataFrame:
        visits = self._get_visits_df().copy()


        labels = visits[['Outcome', 'MRN']].rename(columns={
            'Outcome': 'event',
            'MRN': 'subject_id'
        })
        labels.insert(0, 'duration', self._event_hours(visits))

        horizon = self.config['preprocessing']['max_horizon_hours']
        truncated = labels['duration'] > horizon
        labels.loc[truncated, 'event'] = 0
        labels.loc[truncated, 'duration'] = horizon


        min_hours = self.config['dataset']['cohort']['min_stay_hours']
        labels = labels[labels['duration'] >= min_hours]

        counts = labels['event'].value_counts().sort_index()
        mapping = (self.config['dataset'].get('competing_risks', {})
                   .get('outcome_mapping', {}))
        print(f"  Disposition causes over {len(labels):,d} visits:")
        for code, count in counts.items():
            print(f"    k={code} {mapping.get(code, '?')}: {count:,d} "
                  f"({count / max(1, len(labels)):.1%})")


        n_truncated = int(truncated.sum())
        self._label_report = {
            "horizon_seconds": int(horizon * 3600),
            "n_causes": int(len(counts)),
            "cause_codes": [int(c) for c in counts.index],
            "cause_labels": {str(int(c)): str(mapping.get(int(c), "?"))
                             for c in counts.index},
            "n_rows": int(len(labels)),
            "n_index_patients": int(len(labels)),
            "cause_counts": {str(int(c)): int(n) for c, n in counts.items()},


            "n_censored_administratively": n_truncated,
            "n_censored_at_observation_end": 0,
        }

        return labels

    def build_episodes(self) -> pd.DataFrame:
        from data.episodes import build_icu_episodes
        from survbench.core.landmarks import MCMED_MAX_HORIZON_HOURS

        horizon = self.config['preprocessing']['max_horizon_hours']


        if float(horizon) != float(MCMED_MAX_HORIZON_HOURS):
            raise ValueError(
                f"preprocessing.max_horizon_hours={horizon!r} does not match "
                f"survbench.core.landmarks.MCMED_MAX_HORIZON_HOURS={MCMED_MAX_HORIZON_HOURS}")

        visits = self._get_visits_df()
        arrival = visits['Arrival_time'].to_numpy(dtype='int64')


        observed_hours = np.minimum(
            self._event_hours(visits).to_numpy(dtype='float64'), float(horizon))
        end = arrival + np.rint(
            observed_hours * _SECONDS_PER_HOUR).astype('int64')

        min_stay = float(self.config['dataset']['cohort']['min_stay_hours'])
        labelled = observed_hours >= min_stay

        stay_id = visits.index.to_numpy()[labelled]


        n_people = int(pd.Series(visits['MRN'].to_numpy()[labelled]).nunique())
        print(f"  Episodes: {len(stay_id):,d} ED visits from {n_people:,d} people, "
              f"arrival to min(event time, {int(horizon)}h)")

        return build_icu_episodes(
            patient_id=stay_id,
            stay_id=stay_id,
            start=arrival[labelled],
            end=end[labelled],
            death_time=None,
        )

    def load_static_features(self) -> pd.DataFrame:
        if not self._load_static:
            return pd.DataFrame(index=pd.Index([], name='CSN'))

        visits = self._get_visits_df().copy()


        static_df = pd.get_dummies(visits, columns=['Race', 'Gender', 'Ethnicity'], dtype=int)


        triage_acuity_mapping = {'1-Resuscitation': 5, '2-Emergent': 4, '3-Urgent': 3,
                                 '4-Semi-Urgent': 2, '5-Non-Urgent': 1}
        static_df['Triage_acuity_ordinal'] = static_df['Triage_acuity'].map(triage_acuity_mapping)
        static_df['Triage_acuity_ordinal'] = static_df['Triage_acuity_ordinal'].fillna(-1)


        drop_cols = ['ED_dispo', 'Triage_acuity', 'ED_LOS', 'Outcome', 'MRN',
                     'Arrival_time', 'Dispo_time']
        static_df = static_df.drop(columns=[c for c in drop_cols if c in static_df.columns])

        print(f"Loaded {len(static_df)} stays with {len(static_df.columns)} static features.")
        return static_df

    def load_timeseries(self) -> pd.DataFrame:
        if not self._load_timeseries:
            return pd.DataFrame()

        cohort_csns = self.cohort_df.index.unique()


        from preprocessing.landmarking import observation_horizon_hours
        load_hours = observation_horizon_hours(self.config, 'mcmed')

        print("Processing dynamic data (Labs & Vitals)...")


        labs_path = self.base_dir / self.files['timeseries_lab']
        print(f"  Loading {self.files['timeseries_lab']}...")
        labs = _read_cohort_chunks(
            labs_path, table="labs", version=self.dataset_version,
            cohort_csns=cohort_csns, time_column='Result_time')


        labs['Component_value'] = pd.to_numeric(labs['Component_value'], errors='coerce')


        labs = labs.dropna(subset=['Component_name', 'Component_value'])


        splits = labs['Component_name'].apply(_split_measure_unit)
        labs = labs.copy()
        labs['Component_name'] = [s[0] for s in splits]
        labs['_unit'] = [s[1] for s in splits]
        labs, labs_summary = normalize_dataframe(
            labs, feature_col='Component_name', value_col='Component_value',
            unit_col='_unit', feature_to_kind=MCMED_FEATURE_TO_KIND,
        )
        labs = labs.drop(columns=['_unit'])
        for kind, payload in labs_summary.items():
            cur = self._unit_conversion_summary.setdefault(
                kind,
                {"converted_rows": 0, "from_units": {}, "unrecognized": {}},
            )
            cur["converted_rows"] += payload["converted_rows"]
            for k, v in payload["from_units"].items():
                cur["from_units"][k] = cur["from_units"].get(k, 0) + v
            for k, v in payload["unrecognized"].items():
                cur["unrecognized"][k] = cur["unrecognized"].get(k, 0) + v

        df_labs = labs.pivot_table(index=['CSN', 'Result_time'],
                                   columns='Component_name',
                                   values='Component_value',
                                   aggfunc='first')

        del labs
        gc.collect()

        threshold = 0.95
        df_labs = df_labs.loc[:, df_labs.isnull().sum() / len(df_labs) < threshold]
        for col in df_labs.columns:
            df_labs[col] = pd.to_numeric(df_labs[col], errors='coerce')
        df_labs = df_labs.dropna(how='all')


        vitals_path = self.base_dir / self.files['timeseries_vitals']
        print(f"  Loading {self.files['timeseries_vitals']}...")
        vitals = _read_cohort_chunks(
            vitals_path, table="numerics", version=self.dataset_version,
            cohort_csns=cohort_csns, time_column='Time')


        vitals['Value'] = pd.to_numeric(vitals['Value'], errors='coerce')
        vitals = vitals.dropna(subset=['Measure', 'Value'])


        splits = vitals['Measure'].apply(_split_measure_unit)
        vitals = vitals.copy()
        vitals['Measure'] = [s[0] for s in splits]
        vitals['_unit'] = [s[1] for s in splits]
        vitals, vitals_summary = normalize_dataframe(
            vitals, feature_col='Measure', value_col='Value',
            unit_col='_unit', feature_to_kind=MCMED_FEATURE_TO_KIND,
        )
        vitals = vitals.drop(columns=['_unit'])
        for kind, payload in vitals_summary.items():
            cur = self._unit_conversion_summary.setdefault(
                kind,
                {"converted_rows": 0, "from_units": {}, "unrecognized": {}},
            )
            cur["converted_rows"] += payload["converted_rows"]
            for k, v in payload["from_units"].items():
                cur["from_units"][k] = cur["from_units"].get(k, 0) + v
            for k, v in payload["unrecognized"].items():
                cur["unrecognized"][k] = cur["unrecognized"].get(k, 0) + v

        df_vitals = vitals.pivot_table(index=['CSN', 'Time'],
                                       columns='Measure',
                                       values='Value',
                                       aggfunc='first')

        del vitals
        gc.collect()
        df_vitals = df_vitals.loc[:, df_vitals.isnull().sum() / len(df_vitals) < threshold]
        df_vitals = df_vitals.dropna(how='all')


        print("  Aligning and resampling to hourly grid...")
        df_dynamic = pd.concat([df_labs, df_vitals.rename_axis(index={'Time': 'Result_time'})])

        df_dynamic.reset_index(level=1, inplace=True)


        arrival = self._get_visits_df()['Arrival_time']
        df_dynamic = df_dynamic.merge(
            arrival.rename('arrival_time'), left_on='CSN', right_index=True,
            how='left')
        n_no_arrival = int(df_dynamic['arrival_time'].isna().sum())
        if n_no_arrival:
            df_dynamic = df_dynamic[df_dynamic['arrival_time'].notna()]


        df_dynamic['Time_delta'] = (
            df_dynamic['Result_time'] - df_dynamic['arrival_time'])


        n_pre_arrival = int((df_dynamic['Time_delta'] < 0).sum())
        df_dynamic = df_dynamic[
            (df_dynamic['Time_delta'] >= 0)
            & (df_dynamic['Time_delta'] <= load_hours * _SECONDS_PER_HOUR)]
        print(f"  Window origin = arrival: dropped {n_pre_arrival:,d} "
              f"measurements timestamped before arrival"
              + (f", {n_no_arrival:,d} with no arrival time" if n_no_arrival
                 else "") + ".")


        df_dynamic['Time'] = (
            -(-df_dynamic['Time_delta'] // _SECONDS_PER_HOUR)
        ) * _SECONDS_PER_HOUR
        df_dynamic = df_dynamic.drop(
            columns=['Result_time', 'arrival_time', 'Time_delta'])


        df_dynamic, report = hourly_mean_with_ranges(
            df_dynamic, by=['CSN', 'Time'],
            value_cols=[c for c in df_dynamic.columns if c != 'Time'])
        self._range_filter_report = merge_range_reports(
            self._range_filter_report, report)


        new_index = pd.MultiIndex.from_product(
            [df_dynamic.index.get_level_values(0).unique(),
             [h * _SECONDS_PER_HOUR for h in range(load_hours + 1)]],
            names=['CSN', 'Time']
        )
        df_dynamic = df_dynamic.reindex(new_index)


        df_dynamic = df_dynamic.loc[
            df_dynamic.index.get_level_values('Time') < load_hours * _SECONDS_PER_HOUR
            ]


        df_dynamic.index = df_dynamic.index.set_levels(
            df_dynamic.index.levels[1] / float(_SECONDS_PER_HOUR), level=1
        )
        df_dynamic.index = df_dynamic.index.rename(['stay_id', 'time_hours'])

        print(f"Final dynamic dataframe shape: {df_dynamic.shape}")


        train_ids = self.create_splits()['train']
        missingness_threshold = self.config['preprocessing']['missingness_threshold']
        print(f"Filtering features by missingness (threshold: {missingness_threshold})...")
        ts_train = df_dynamic[df_dynamic.index.get_level_values('stay_id').isin(train_ids)]


        from preprocessing.imputation import presence_after_ffill


        visits = self._get_visits_df()
        los = (visits.set_index('CSN')['ED_LOS']
               if 'CSN' in visits.columns else visits['ED_LOS'])
        los = los[~los.index.duplicated()]


        hours = ts_train.index.get_level_values('time_hours').to_numpy(float)
        stays = ts_train.index.get_level_values('stay_id')
        in_ed = hours < los.reindex(stays).to_numpy(dtype=float)
        print(f"  presence denominator: {in_ed.mean():.1%} of grid rows are "
              f"within the ED visit")
        missingness = presence_after_ffill(ts_train, level='stay_id',
                                           valid_rows=in_ed)
        cols_to_keep = missingness[missingness >= missingness_threshold].index
        df_dynamic = df_dynamic[cols_to_keep]
        df_dynamic = df_dynamic.sort_index()

        print(f"Kept {len(cols_to_keep)} dynamic features (≥{missingness_threshold * 100}% present in training)")
        return df_dynamic

    def load_icd_codes(self) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        if not self._load_icd:
            return pd.DataFrame(index=pd.Index([], name='CSN')), None

        visits_df = self._get_visits_df()
        if 'Arrival_time' not in visits_df.columns:
            raise ValueError(
                "MC-MED PMH causal filter requires Arrival_time in visits.csv. "
                "Re-run _get_visits_df with Arrival_time in keep_cols."
            )

        icd_path = self.base_dir / self.files['icd_codes']
        print(f"\nProcessing {self.files['icd_codes']} (PMH ICD codes, causal filter)...")
        pmh = pd.read_csv(icd_path, usecols=['MRN', 'Noted_date', 'Code'])
        validate_schema(pmh, "mcmed", "pmh", self.dataset_version)
        pmh = pmh.dropna(subset=['MRN', 'Code'])


        pmh, n_no_date = _to_epoch_seconds_or_drop(pmh, 'Noted_date',
                                                   table='pmh')
        if n_no_date:
            print(f"  Dropped {n_no_date:,d} PMH codes with no Noted_date.")


        visits_for_join = visits_df.reset_index()[['MRN', 'CSN', 'Arrival_time']]
        merged = pmh.merge(visits_for_join, on='MRN', how='inner')
        arrival_day = (merged['Arrival_time'] // _SECONDS_PER_DAY) * _SECONDS_PER_DAY
        causal = merged[merged['Noted_date'] < arrival_day].copy()


        cohort_csns = set(self.cohort_df.index.unique())
        causal = causal[causal['CSN'].isin(cohort_csns)]

        if causal.empty:
            print("  No PMH codes survived the causal filter; ICD modality will be all-zero.")
            empty = pd.DataFrame(index=pd.Index(sorted(cohort_csns), name='CSN'))
            return empty, None


        train_csns = set(int(c) for c in self.create_splits()['train'])
        causal_train = causal[causal['CSN'].isin(train_csns)]
        top_codes = (causal_train['Code'].value_counts()
                     .nlargest(self.n_top_codes).index.tolist())


        kept = causal[causal['Code'].isin(top_codes)]
        icd_matrix = pd.crosstab(kept['CSN'], kept['Code'])
        icd_matrix = icd_matrix.reindex(columns=top_codes, fill_value=0)
        icd_matrix.columns = [f'icd_{c}' for c in icd_matrix.columns]
        icd_matrix = (icd_matrix > 0).astype(int)


        icd_matrix = icd_matrix.reindex(
            pd.Index(sorted(cohort_csns), name='CSN'), fill_value=0,
        )

        print(f"  Causal filter: {len(causal):,d} codes survived (of {len(merged):,d} merged).")
        print(f"  Created ({len(icd_matrix)}, {len(icd_matrix.columns)}) visit-ICD matrix "
              f"(top-{self.n_top_codes} by training fold).")
        return icd_matrix, None

    def _filter_radiology_to_observable(self, rads: pd.DataFrame) -> pd.DataFrame:
        from preprocessing.landmarking import observation_horizon_hours
        window_hours = observation_horizon_hours(self.config, 'mcmed')
        window_seconds = window_hours * _SECONDS_PER_HOUR
        visits = self._get_visits_df()

        n_input = len(rads)
        rads = _to_epoch_seconds_masked(rads, 'Result_time', table='rads')
        times = visits[['Arrival_time', 'Dispo_time']]
        merged = rads.merge(times, left_on='CSN', right_index=True, how='left')

        result = merged['Result_time']
        undetermined = result.isna() | merged['Dispo_time'].isna()\
            | merged['Arrival_time'].isna()
        after_dispo = (result > merged['Dispo_time']).fillna(False)
        after_window = (
            result > merged['Arrival_time'] + window_seconds).fillna(False)

        keep = ~(undetermined | after_dispo | after_window)
        print(f"  Radiology temporal filter: kept {int(keep.sum()):,d} of "
              f"{n_input:,d} reports "
              f"({int(after_dispo.sum()):,d} resulted after disposition, "
              f"{int(after_window.sum()):,d} after the "
              f"{window_hours}h window, "
              f"{int(undetermined.sum()):,d} undetermined).")
        return merged.loc[keep.to_numpy(dtype=bool)].drop(
            columns=['Arrival_time', 'Dispo_time'])

    def load_radiology_reports(self) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        if not self._load_radiology:
            return pd.DataFrame(index=pd.Index([], name='CSN')), None

        cohort_csns = self.cohort_df.index.unique()
        rad_path = self.base_dir / self.files['radiology_embeddings']
        print(f"\nProcessing {self.files['radiology_embeddings']} (Radiology Reports) for {len(cohort_csns)} visits...")
        rads = pd.read_csv(rad_path)
        validate_schema(rads, "mcmed", "rads", self.dataset_version)
        rads = rads[rads['CSN'].isin(cohort_csns)]
        rads = rads.dropna(subset=['Study', 'Impression'])
        rads = self._filter_radiology_to_observable(rads)
        if rads.empty:
            print("No radiology reports remain after the temporal filter.")
            return pd.DataFrame(index=pd.Index([], name='CSN')), None
        rads['Text'] = rads['Study'] + ' ' + rads['Impression']

        if len(rads) == 0:
            print("No valid radiology reports found for this cohort.")
            return pd.DataFrame(index=pd.Index([], name='CSN')), None

        from preprocessing.text_embedding_cache import get_or_compute_embeddings

        rad_cfg = self.config.get('radiology_processing', {})
        model_name = rad_cfg.get('model_name', 'yikuan8/Clinical-Longformer')
        max_length = rad_cfg.get('max_length', self.bert_max_length)
        batch_size = rad_cfg.get('batch_size', self.bert_batch_size)
        pooling = rad_cfg.get('pooling', 'mean')

        cache_dir = self._cache_dir() / "embeddings"

        emb = get_or_compute_embeddings(
            texts=rads['Text'].tolist(),
            cache_dir=cache_dir,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            pooling=pooling,
        )


        emb_cols = [f'rad_{i}' for i in range(emb.shape[1])]
        radio_df = pd.DataFrame(emb, columns=emb_cols)
        radio_df['admission_id'] = rads['CSN'].to_numpy()
        arrival = (self._get_visits_df()['Arrival_time']
                   .reindex(rads['CSN'].to_numpy()).to_numpy(dtype='float64'))
        radio_df['hours_since_intime'] = (
            rads['Result_time'].to_numpy(dtype='float64') - arrival
        ) / _SECONDS_PER_HOUR

        print(f"Embedded {len(radio_df):,} reports across "
              f"{radio_df['admission_id'].nunique():,} visits "
              f"(pooled per landmark downstream).")
        return radio_df, None

    def apply_cohort_criteria(self, df: pd.DataFrame) -> pd.DataFrame:


        print(f"Cohort criteria applied: {len(df)} stays remaining.")
        return df

    def get_modality_info(self) -> Dict:
        return {
            'timeseries': self.load_timeseries,
            'static': self.load_static,
            'icd': self.load_icd,
            'radiology': self.load_radiology,
        }
