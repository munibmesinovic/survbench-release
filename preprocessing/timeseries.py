import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm


_OBSERVED_PREFIX = "__observed__"


class TimeSeriesAggregator:

    def __init__(self, config: Dict):
        self.config = config
        self.num_windows = config['preprocessing']['num_windows']
        self.window_size = config['preprocessing']['window_size_hours']
        self.max_hours = config['preprocessing']['max_hours']


        self.max_carry_forward_hours = config['preprocessing'].get(
            'max_carry_forward_hours', 24)

        assert self.num_windows * self.window_size == self.max_hours

    def aggregate_to_windows(
            self,
            ts_df: pd.DataFrame,
            cohort_dfs: Dict[str, pd.DataFrame],
            static_df: pd.DataFrame,
            return_observed: bool = False,
    ):
        dynamic_cols = list(ts_df.columns)
        static_cols = list(static_df.columns) if static_df is not None else []

        print("\\n--- Time-Series Aggregation ---")
        print(f"Windows: {self.num_windows} × {self.window_size}h = {self.max_hours}h")
        print(f"Dynamic features: {len(dynamic_cols)}")
        print(f"Static features: {len(static_cols)}")

        all_feature_names = dynamic_cols + static_cols
        num_features = len(all_feature_names)

        split_data = {}
        split_observed = {}

        for split in ['train', 'val', 'test']:
            print(f"\\nProcessing {split} split...")
            cohort = cohort_dfs[split]

            N = len(cohort)
            T = self.num_windows
            F = num_features

            data_array = np.full((N, T, F), np.nan, dtype=np.float32)


            observed_array = np.zeros((N, T, F), dtype=bool)
            observed_array[:, :, len(dynamic_cols):] = True


            if isinstance(cohort.index, pd.MultiIndex):
                stay_ids = cohort.index.get_level_values(0)
                landmarks = cohort.index.get_level_values(1).to_numpy(
                    dtype=np.float64)
            else:
                stay_ids = cohort.index
                landmarks = None


            static_data_split = (static_df.reindex(stay_ids)
                                 if static_df is not None else None)


            cached_id, cached_ts = None, None

            for idx, patient_id in enumerate(tqdm(stay_ids, desc=f"{split}")):

                if patient_id in ts_df.index:
                    if patient_id == cached_id:
                        patient_ts = cached_ts
                    else:
                        patient_ts = ts_df.loc[patient_id]

                        if isinstance(patient_ts, pd.Series):
                            patient_ts = patient_ts.to_frame().T
                            patient_ts.index.name = 'time_hours'

                        patient_ts = patient_ts.reset_index()


                        observed_block = patient_ts[dynamic_cols].notna()
                        observed_block.columns = [_OBSERVED_PREFIX + c
                                                  for c in dynamic_cols]
                        patient_ts[dynamic_cols] = patient_ts[dynamic_cols].ffill()
                        patient_ts = pd.concat([patient_ts, observed_block],
                                               axis=1)


                        if self.max_carry_forward_hours is not None:
                            t = patient_ts['time_hours'].to_numpy(dtype=float)
                            for col in dynamic_cols:
                                seen = patient_ts[_OBSERVED_PREFIX + col].to_numpy()
                                last = np.where(seen, t, np.nan)
                                last = pd.Series(last).ffill().to_numpy()
                                stale = (t - last) > self.max_carry_forward_hours
                                if stale.any():
                                    patient_ts.loc[stale, col] = np.nan
                        cached_id, cached_ts = patient_id, patient_ts


                    patient_ts = patient_ts.copy()


                    origin = 0.0 if landmarks is None else (
                        landmarks[idx] - self.max_hours)
                    offset = patient_ts['time_hours'] - origin
                    patient_ts['window'] = np.floor(
                        offset / self.window_size).astype(int)
                    patient_ts = patient_ts[(patient_ts['window'] >= 0) & (patient_ts['window'] < self.num_windows)]
                    ts_grouped = patient_ts.groupby('window')[dynamic_cols].mean()


                    obs_cols = [_OBSERVED_PREFIX + c for c in dynamic_cols]
                    obs_grouped = patient_ts.groupby('window')[obs_cols].any()

                    for window_idx in ts_grouped.index:
                        data_array[idx, window_idx, :len(dynamic_cols)] = ts_grouped.loc[window_idx].values
                        observed_array[idx, window_idx, :len(dynamic_cols)] =\
                            obs_grouped.loc[window_idx].values

                if static_data_split is not None:
                    static_vector = static_data_split.iloc[idx].values
                    data_array[idx, :, len(dynamic_cols):] = static_vector

            split_data[split] = data_array
            split_observed[split] = observed_array
            n_dyn = len(dynamic_cols)
            if n_dyn:
                rate = float(observed_array[:, :, :n_dyn].mean())
                print(f"{split} shape: {data_array.shape}  "
                      f"dynamic cells measured (not carried forward): {rate:.2%}")
            else:
                print(f"{split} shape: {data_array.shape}")

        if return_observed:
            return split_data, split_observed
        return split_data

    def get_feature_structure(
            self,
            dynamic_cols: List[str],
            static_cols: List[str]
    ) -> Dict:
        return {
            'num_dynamic': len(dynamic_cols),
            'num_static': len(static_cols),
            'num_total': len(dynamic_cols) + len(static_cols),
            'dynamic_names': dynamic_cols,
            'static_names': static_cols,
            'all_names': dynamic_cols + static_cols,
            'dynamic_indices': list(range(len(dynamic_cols))),
            'static_indices': list(range(len(dynamic_cols), len(dynamic_cols) + len(static_cols)))
        }
