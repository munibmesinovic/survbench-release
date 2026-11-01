
from __future__ import annotations

import gc
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.base_loader import BaseDataLoader
from data.schema import validate_schema, clip_to_ranges, HIRID_RANGE_BOUNDS
from preprocessing.ranges import hourly_mean_with_ranges, merge_range_reports


HIRID_FEATURE_COLS = [
    "vm1", "vm3", "vm4", "vm5", "vm13", "vm20", "vm28", "vm62",
    "vm136", "vm146", "vm172", "vm174", "vm176",
    "pm41", "pm42", "pm43", "pm44", "pm87",
]
HIRID_DYNAMIC_FEATURE_NAMES = [
    "HR", "ABP_sys", "ABP_dia", "MAP", "CO", "SpO2", "RASS", "PeakPressure",
    "lactate_arterial", "lactate_venous", "INR", "glucose", "CRP",
    "dobutamine", "milrinone", "levosimendan", "theophylline", "analgesics",
]


class HIRIDDataLoader(BaseDataLoader):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.files = config['dataset']['files']
        self.modalities_config = config.get('modalities', {})
        self._load_timeseries_flag = self.modalities_config.get('timeseries', True)
        self._load_static_flag = self.modalities_config.get('static', True)
        self._load_icd_flag = self.modalities_config.get('icd', False)
        self._load_radiology_flag = self.modalities_config.get('radiology', False)


        self._range_filter_report: dict = {}
        if self._load_icd_flag:
            print("  HIRID warning: 'icd' modality enabled but HIRID has no ICD codes; ignored.")
        if self._load_radiology_flag:
            print("  HIRID warning: 'radiology' modality enabled but HIRID has no notes; ignored.")


        from preprocessing.landmarking import observation_horizon_hours
        self.n_input_steps_hourly = observation_horizon_hours(config, 'hirid')
        self.max_horizon = config['preprocessing']['max_horizon_hours']

        self._stage1_cache: Optional[Dict] = None


        self._label_report: Optional[Dict] = None

    def _cache_dir(self) -> Path:
        explicit = self.config.get('dataset', {}).get('cache_dir')
        if explicit:
            return Path(explicit)
        return Path(self.base_dir) / ".cache"

    def _build_static_rich(self, labels_df: pd.DataFrame, apache_top_k: int = 5) -> Tuple[np.ndarray, List[str]]:
        ext_path_rel = self.files.get('general_table_extended')
        ext_path = Path(self.base_dir) / ext_path_rel if ext_path_rel else None
        if ext_path is None or not ext_path.exists():
            print("  [HIRID] general_table_extended.parquet not found; falling back to age+sex only.")
            static = np.stack([
                labels_df['age'].values.astype(np.float32),
                labels_df['sex_is_male'].values.astype(np.float32),
            ], axis=1)
            return static, ['age', 'sex_is_male']

        ext = pd.read_parquet(ext_path)[
            ['patientid', 'height', 'APACHE II Group', 'APACHE IV Group']
        ]
        validate_schema(ext, "hirid", "general_table_extended", self.dataset_version)
        merged = labels_df.merge(ext, on='patientid', how='left')
        h_mean = merged['height'].mean()
        merged['height'] = merged['height'].fillna(h_mean)

        feats = [
            merged['age'].values.astype(np.float32),
            merged['sex_is_male'].values.astype(np.float32),
            merged['height'].values.astype(np.float32),
        ]
        names = ['age', 'sex_is_male', 'height']
        for label, col in [('apII', 'APACHE II Group'), ('apIV', 'APACHE IV Group')]:
            vals = merged[col]
            top = vals.value_counts(dropna=True).head(apache_top_k).index.tolist()
            for i, code in enumerate(top):
                feats.append((vals == code).astype(np.float32).values)
                names.append(f"{label}_top{i+1}_code{int(code)}")
            is_other = vals.notna() & ~vals.isin(top)
            feats.append(is_other.astype(np.float32).values)
            names.append(f"{label}_other")
            feats.append(vals.notna().astype(np.float32).values)
            names.append(f"{label}_present")
        return np.stack(feats, axis=1).astype(np.float32), names

    def _build_stage1_cache(self) -> Dict:
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)


        T = self.n_input_steps_hourly


        feat_path = cache_dir / f"hirid_hourly_features_T{T}_raw_ranged.npy"
        los_path = cache_dir / "hirid_los_hours.npy"
        event_path = cache_dir / "hirid_event.npy"
        static_path = cache_dir / "hirid_static.npy"
        names_path = cache_dir / "hirid_static_names.json"
        pids_path = cache_dir / "hirid_pids.npy"

        if all(p.exists() for p in [feat_path, los_path, event_path, static_path, pids_path, names_path]):
            print(f"  [HIRID Stage 1] cache hit at {cache_dir}")
            with open(names_path) as f:
                static_names = json.load(f)
            x_hourly = np.load(feat_path)
            assert x_hourly.shape[1] == T, (
                f"HIRID cache shape mismatch: file has T={x_hourly.shape[1]}, "
                f"config requires T={T}. Delete {feat_path} and re-run."
            )
            return {
                "x_hourly": x_hourly,
                "los_hours": np.load(los_path),
                "event": np.load(event_path),
                "static": np.load(static_path),
                "static_names": static_names,
                "pids": np.load(pids_path),
            }

        print("  [HIRID Stage 1] cache miss; building from raw (~30 min on first run)")


        labels_csv = Path(self.base_dir) / self.files['labels']
        gen_csv = Path(self.base_dir) / self.files['general_table']
        labels_df = pd.read_csv(labels_csv)
        validate_schema(labels_df, "hirid", "hirid_patient_labels", self.dataset_version)
        gen = pd.read_csv(gen_csv)[['patientid', 'age', 'sex', 'discharge_status']].copy()
        validate_schema(gen, "hirid", "general_table", self.dataset_version)
        gen['sex_is_male'] = gen['sex'].map({'M': 1.0, 'F': 0.0})
        gen['event'] = (gen['discharge_status'] == 'dead').astype(np.int64)
        labels_df = labels_df.merge(
            gen[['patientid', 'age', 'sex_is_male', 'event']], on='patientid', how='left',
        )
        n_before = len(labels_df)
        labels_df = labels_df.dropna(subset=['age', 'sex_is_male']).reset_index(drop=True)
        labels_df = labels_df.sort_values('patientid').reset_index(drop=True)
        print(f"    Cohort after age/sex non-null: {len(labels_df)} (dropped {n_before - len(labels_df)})")

        pids = labels_df['patientid'].values.astype(np.int64)
        pid_to_idx = {int(p): i for i, p in enumerate(pids)}
        n_patients = len(pids)

        static, static_names = self._build_static_rich(labels_df)
        event = labels_df['event'].values.astype(np.int64)


        imputed_dir = Path(self.base_dir) / self.files['imputed_dir']
        csv_files = sorted(glob.glob(str(imputed_dir / "*.csv")))
        print(f"    Streaming {len(csv_files)} imputed-stage CSVs from {imputed_dir}")

        n_feat = len(HIRID_FEATURE_COLS)
        x_hourly = np.full((n_patients, self.n_input_steps_hourly, n_feat),
                            np.nan, dtype=np.float32)
        max_reldatetime = np.full(n_patients, np.nan, dtype=np.float64)
        cohort_set = set(int(p) for p in pids)

        usecols = ['patientid', 'reldatetime'] + HIRID_FEATURE_COLS
        for file_i, csv_path in enumerate(csv_files):
            df = pd.read_csv(csv_path, usecols=usecols)
            if file_i == 0:
                validate_schema(df, "hirid", "imputed_stage", self.dataset_version)
            df = df[df['patientid'].isin(cohort_set)]
            if len(df) == 0:
                continue
            mx = df.groupby('patientid')['reldatetime'].max()
            for pid_, rmax in mx.items():
                idx = pid_to_idx[int(pid_)]
                if np.isnan(max_reldatetime[idx]) or rmax > max_reldatetime[idx]:
                    max_reldatetime[idx] = rmax


            cutoff = (self.n_input_steps_hourly * 3600) - 300
            df24 = df[df['reldatetime'] <= cutoff].copy()
            if len(df24):
                df24['hour_bin'] = (df24['reldatetime'] // 3600).astype(np.int64)
                df24 = df24[df24['hour_bin'] < self.n_input_steps_hourly]


                grouped, report = hourly_mean_with_ranges(
                    df24, by=['patientid', 'hour_bin'],
                    value_cols=HIRID_FEATURE_COLS,
                    names=HIRID_DYNAMIC_FEATURE_NAMES)
                self._range_filter_report = merge_range_reports(
                    self._range_filter_report, report)
                for (pid_, h), row in grouped.iterrows():
                    x_hourly[pid_to_idx[int(pid_)], int(h)] = row.values.astype(np.float32)
            if (file_i + 1) % 50 == 0:
                print(f"    [HIRID Stage 1] processed {file_i + 1}/{len(csv_files)} batches")
            del df


        has_data = ~np.isnan(max_reldatetime)
        if not has_data.all():
            drop_n = int((~has_data).sum())
            print(f"    Dropping {drop_n} patients with no raw observations")
            pids = pids[has_data]
            x_hourly = x_hourly[has_data]
            max_reldatetime = max_reldatetime[has_data]
            static = static[has_data]
            event = event[has_data]
            n_patients = len(pids)


        n, t, f = x_hourly.shape
        flat = pd.DataFrame(x_hourly.reshape(n * t, f),
                            columns=HIRID_DYNAMIC_FEATURE_NAMES)
        flat, clip_counts = clip_to_ranges(flat, HIRID_RANGE_BOUNDS)
        if clip_counts:
            print(f"  [HIRID Stage 1] range-clipped values: {clip_counts}")
        x_hourly = flat.values.reshape(n, t, f).astype(np.float32)

        los_hours = (max_reldatetime / 3600.0).astype(np.float32)
        print(f"    LOS stats: min={np.nanmin(los_hours):.1f}h "
              f"median={np.nanmedian(los_hours):.1f}h max={np.nanmax(los_hours):.1f}h")

        np.save(feat_path, x_hourly)
        np.save(los_path, los_hours)
        np.save(event_path, event)
        np.save(static_path, static)
        np.save(pids_path, pids)
        with open(names_path, "w") as f:
            json.dump(static_names, f, indent=2)
        gc.collect()

        return {
            "x_hourly": x_hourly, "los_hours": los_hours, "event": event,
            "static": static, "static_names": static_names, "pids": pids,
        }

    def _ensure_stage1(self) -> Dict:
        if self._stage1_cache is None:
            self._stage1_cache = self._build_stage1_cache()
        return self._stage1_cache

    def build_episodes(self) -> pd.DataFrame:
        from data.episodes import build_icu_episodes

        cache = self._ensure_stage1()
        pids = cache['pids'].astype(np.int64)
        end_seconds = np.rint(
            cache['los_hours'].astype(np.float64) * 3600).astype(np.int64)
        died_in_icu = cache['event'].astype(bool)

        return build_icu_episodes(
            patient_id=pids,
            stay_id=pids,
            start=np.zeros(len(pids), dtype=np.int64),
            end=end_seconds,
            death_time=pd.Series(
                np.where(died_in_icu, end_seconds, np.nan)).astype("Int64"),
        )

    def load_labels(self) -> pd.DataFrame:
        from survbench.core.labels import label_icu_mortality

        from survbench.core.labels import HIRID_IN_ICU_DEATH

        episodes = self.build_episodes()
        labels = label_icu_mortality(
            episodes,
            post_icu_death_source=None,
            horizon_seconds=np.int64(int(self.max_horizon * 3600)),


            in_icu_death_time_source=(HIRID_IN_ICU_DEATH,),
        )
        self._label_report = labels.to_dict()

        from survbench.core.labels import format_label_report

        rows = labels.rows
        for line in format_label_report(self._label_report):
            print(line)

        stay = rows['stay_id'].to_numpy().astype(np.int64)
        return pd.DataFrame({
            'duration': (rows['duration_seconds'].to_numpy() / 3600.0
                         ).astype(np.float32),
            'event': rows['cause'].to_numpy().astype(np.int64),
            'subject_id': rows['patient_id'].to_numpy(),


            'censoring_reason': rows['censoring_reason'].to_numpy(),
        }, index=pd.Index(stay, name='admission_id'))

    def load_static_features(self) -> pd.DataFrame:
        cache = self._ensure_stage1()
        df = pd.DataFrame(
            cache['static'], columns=cache['static_names'],
            index=pd.Index(cache['pids'].astype(np.int64), name='admission_id'),
        )
        return df

    def load_timeseries(self) -> pd.DataFrame:
        cache = self._ensure_stage1()
        x = cache['x_hourly']
        n, t, f = x.shape
        pids = cache['pids'].astype(np.int64)


        idx = pd.MultiIndex.from_product(
            [pids, range(t)], names=['admission_id', 'time_hours'],
        )
        df = pd.DataFrame(
            x.reshape(n * t, f),
            index=idx,
            columns=HIRID_DYNAMIC_FEATURE_NAMES,
        )


        train_ids = self.create_splits()['train']
        thresh = self.config['preprocessing'].get('missingness_threshold', 0.0)
        if thresh > 0.0:
            train_df = df.loc[df.index.get_level_values(0).isin(train_ids)]


            from preprocessing.imputation import presence_after_ffill
            keep = presence_after_ffill(train_df, level=0)
            df = df[keep[keep >= thresh].index]
        return df.sort_index()

    def apply_cohort_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'subject_id' not in df.columns:
            raise ValueError("HIRIDDataLoader.apply_cohort_criteria: 'subject_id' missing.")
        min_los = self.config['dataset']['cohort'].get('min_stay_hours', 24.0)
        kept = df[df['duration'] >= min_los]
        print(f"  HIRID cohort: LOS >= {min_los}h -> {len(kept)} stays (was {len(df)})")
        return kept
