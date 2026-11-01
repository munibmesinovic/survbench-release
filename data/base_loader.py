from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path


_CAUSE_CODE_DOC = ("ICU mortality {0 censored, 1 in-ICU death, "
                   "2 discharged/died by H, 3 discharged alive} or ED "
                   "disposition {0 truncated, 1 ICU, 2 hospital, "
                   "3 observation, 4 home}")


def _dominant_cause_per_patient(cohort_df: pd.DataFrame) -> pd.Series:
    if 'event' not in cohort_df.columns:
        raise ValueError(
            "Stratified split requires an 'event' column in cohort_df."
        )
    events = cohort_df[['subject_id', 'event']]


    if (events['event'] < 0).any():
        bad = sorted({int(v) for v in events['event'].unique() if v < 0})
        raise ValueError(
            f"negative cause code(s) {bad} in event; valid codes: {_CAUSE_CODE_DOC}"
        )
    per_patient = (
        events.assign(_resolved=events['event'].where(events['event'] > 0))
        .groupby('subject_id')['_resolved'].min()
        .fillna(0)
        .astype('int64')
    )
    per_patient.name = 'stratum'
    return per_patient


from survbench.core.arms import arm_of as _arm_of


def _v1_binary_stratum(cohort_df: pd.DataFrame,
                       censor_causes: List[int]) -> pd.Series:
    events = cohort_df[['subject_id', 'event']].copy()
    events['event'] = events['event'].where(
        ~events['event'].isin(censor_causes), 0)
    per_patient = (
        events.groupby('subject_id')['event'].max().gt(0).astype('int64')
    )
    per_patient.name = 'stratum'
    return per_patient


def _min_stratum_size(train_size: float, val_size: float,
                      test_size: float) -> int:
    total = train_size + val_size + test_size
    f_temp = (val_size + test_size) / total if total > 0 else 0.0
    if f_temp <= 0:
        raise ValueError(
            f"splits.val + splits.test must be > 0 to hold out anything; got "
            f"val={val_size!r}, test={test_size!r}."
        )
    return max(2, int(np.ceil(2.0 / f_temp)))


def _pool_thin_strata(strata: np.ndarray, min_per_class: int,
                      pooled_label: int = -1) -> np.ndarray:
    values, counts = np.unique(strata, return_counts=True)
    thin = [(int(v), int(c)) for v, c in zip(values, counts)
            if c < min_per_class]
    if not thin:
        return strata

    pooled = strata.copy()
    pooled[np.isin(strata, [v for v, _ in thin])] = pooled_label
    n_pooled = int((pooled == pooled_label).sum())

    print(f"\n--- Split strata: pooling causes with fewer than {min_per_class} patients ---")
    for value, count in thin:
        print(f"  cause k={value}: {count} patients -> pooled stratum "
              f"{pooled_label}")
    print(f"  pooled stratum {pooled_label}: {n_pooled} patients from {len(thin)} cause(s) {[v for v, _ in thin]}")

    if n_pooled < min_per_class:
        survivors = pooled[pooled != pooled_label]
        if survivors.size == 0:
            print("  every class is thin; the split is effectively unstratified")
            return pooled
        s_values, s_counts = np.unique(survivors, return_counts=True)
        largest = int(s_values[int(np.argmax(s_counts))])
        pooled[pooled == pooled_label] = largest
        print(f"  pooled stratum {pooled_label} has fewer than {min_per_class} patients; "
              f"merged into class k={largest} ({n_pooled} patients unstratified)")
    return pooled


class BaseDataLoader(ABC):

    def __init__(self, config: Dict):
        self.config = config
        self.dataset_name = config['dataset']['name']
        self.base_dir = Path(config['dataset']['base_dir'])
        self.seed = config['splits']['seed']
        np.random.seed(self.seed)

        from data.schema import DEFAULT_VERSIONS
        self.dataset_version = (
            config['dataset'].get('version')
            or DEFAULT_VERSIONS.get(self.dataset_name)
        )

        self.labels_df = None
        self.static_df = None
        self.timeseries_df = None
        self.cohort_df = None
        self._splits_cache: dict | None = None

    @abstractmethod
    def load_labels(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_static_features(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_timeseries(self) -> pd.DataFrame:
        pass

    def build_cohort(self) -> pd.DataFrame:
        self.labels_df = self.load_labels()
        self.static_df = self.load_static_features()

        cohort_df = self.static_df.merge(
            self.labels_df,
            left_index=True,
            right_index=True
        )

        cohort_df = self.apply_cohort_criteria(cohort_df)
        self.cohort_df = cohort_df
        return cohort_df

    @abstractmethod
    def apply_cohort_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def create_splits(self) -> Dict[str, pd.Index]:
        if self._splits_cache is not None:
            return self._splits_cache

        from sklearn.model_selection import train_test_split

        if self.cohort_df is None:
            raise ValueError("Must build cohort before creating splits")

        if 'subject_id' not in self.cohort_df.columns:
            raise ValueError(
                "cohort_df has no 'subject_id' column; the loader must keep it for splitting."
            )

        split_config = self.config['splits']


        split_by = split_config.get('split_by', 'patient')
        if split_by != 'patient':
            raise ValueError(
                f"splits.split_by={split_by!r} is not supported; splits are always by subject_id ('patient')")

        train_size = split_config['train']
        val_size = split_config['val']
        test_size = split_config['test']


        arm, censor_causes = _arm_of(self.config)
        if arm == "censored":
            per_patient_stratum = _v1_binary_stratum(self.cohort_df,
                                                     censor_causes)
        else:
            per_patient_stratum = _dominant_cause_per_patient(self.cohort_df)
        unique_patient_ids = per_patient_stratum.index.values


        patient_strata = _pool_thin_strata(
            per_patient_stratum.values,
            _min_stratum_size(train_size, val_size, test_size),
        )

        train_patient_ids, temp_patient_ids, _, temp_strata = train_test_split(
            unique_patient_ids,
            patient_strata,
            test_size=(val_size + test_size),
            random_state=self.seed,
            stratify=patient_strata,
        )
        val_patient_ids, test_patient_ids = train_test_split(
            temp_patient_ids,
            test_size=(test_size / (val_size + test_size)),
            random_state=self.seed,
            stratify=temp_strata,
        )


        from leakage_guards import assert_splits_disjoint
        assert_splits_disjoint(train_patient_ids, val_patient_ids, test_patient_ids)

        train_ids = self.cohort_df[
            self.cohort_df['subject_id'].isin(train_patient_ids)
        ].index
        val_ids = self.cohort_df[
            self.cohort_df['subject_id'].isin(val_patient_ids)
        ].index
        test_ids = self.cohort_df[
            self.cohort_df['subject_id'].isin(test_patient_ids)
        ].index

        splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}


        strata_values, strata_counts = np.unique(patient_strata,
                                                 return_counts=True)
        observed_causes = [int(c) for c in
                           np.unique(self.cohort_df['event'].to_numpy())]

        def _cause_rates(ids: pd.Index) -> str:
            events = self.cohort_df.loc[ids, 'event']
            return " ".join(f"k={c}:{(events == c).mean():.1%}"
                            for c in observed_causes)


        if arm == "censored":
            how = (f"single-risk arm: binary stratum, causes {censor_causes} set to 0")
        else:
            how = "competing-risks arm: per-patient cause code, one class per cause"
        print(f"\n--- Data Splits ({how}; split on subject_id) ---")
        print(f"Total unique patients: {len(unique_patient_ids)}")
        print(f"Strata used ({len(strata_values)} classes): "
              + ", ".join(f"{int(v)}:{int(n)}"
                          for v, n in zip(strata_values, strata_counts)))
        print(f"Train: {len(train_patient_ids)} patients ({len(train_ids)} stays, "
              f"cause rates {_cause_rates(train_ids)})")
        print(f"Val:   {len(val_patient_ids)} patients ({len(val_ids)} stays, "
              f"cause rates {_cause_rates(val_ids)})")
        print(f"Test:  {len(test_patient_ids)} patients ({len(test_ids)} stays, "
              f"cause rates {_cause_rates(test_ids)})")

        self._splits_cache = splits
        return splits
