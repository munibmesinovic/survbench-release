
from __future__ import annotations

import numpy as np
import pandas as pd


def ffill_within_patient(df: pd.DataFrame, group_level: int = 0) -> pd.DataFrame:
    return df.groupby(level=group_level).ffill()


def observed_after_ffill(df: pd.DataFrame, level=0) -> pd.DataFrame:
    return df.notna().groupby(level=level).cummax()


def presence_after_ffill(df: pd.DataFrame, level=0,
                         chunk_cols: int = 8,
                         valid_rows=None) -> pd.Series:
    out = {}
    columns = list(df.columns)
    keep = None if valid_rows is None else np.asarray(valid_rows, dtype=bool)
    for start in range(0, len(columns), max(1, chunk_cols)):
        block = columns[start:start + max(1, chunk_cols)]


        observed = df[block].notna().groupby(level=level).cummax()
        for col in block:
            series = observed[col].to_numpy()
            out[col] = float(series[keep].mean()) if keep is not None\
                else float(series.mean())
        del observed
    return pd.Series(out, index=columns, dtype=float)


def ffill_along_time(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    n, t, f = out.shape
    for i in range(n):
        for j in range(f):
            col = out[i, :, j]
            nan_mask = np.isnan(col)
            if nan_mask.all() or not nan_mask.any():
                continue

            idx_arr = np.where(~nan_mask, np.arange(t), 0)
            np.maximum.accumulate(idx_arr, out=idx_arr)

            ffilled = col[idx_arr]

            first_obs = np.argmax(~nan_mask)
            ffilled[:first_obs] = np.nan
            out[i, :, j] = ffilled
    return out


def train_mean_fallback(
    x: np.ndarray,
    train_idx: np.ndarray,
    dynamic_indices: list[int],
) -> np.ndarray:
    out = x.copy()
    for j in dynamic_indices:
        col_train = out[train_idx, :, j]
        if np.isnan(col_train).all():
            mean = 0.0
        else:
            mean = float(np.nanmean(col_train))
        if np.isnan(mean):
            mean = 0.0
        nan_mask = np.isnan(out[:, :, j])
        out[:, :, j] = np.where(nan_mask, mean, out[:, :, j])
    return out
