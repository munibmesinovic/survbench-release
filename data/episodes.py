from __future__ import annotations

import numpy as np
import pandas as pd


from survbench.core.landmarks import DEAD_STATE, ICU_STATE

EPISODE_COLUMNS: tuple[str, ...] = (
    "patient_id", "stay_id", "state", "start", "end")


def epoch_seconds_from_datetime(values, *, column: str,
                                allow_absent: bool = False) -> pd.Series:
    series = pd.Series(values).reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise TypeError(
            f"{column}: expected a parsed datetime column, got {series.dtype}; "
            f"convert with survbench.ir.time.to_epoch_seconds")

    absent = series.isna()
    if bool(absent.any()) and not allow_absent:
        raise ValueError(
            f"{column}: {int(absent.sum())} of {len(series)} values missing; column must be complete")

    seconds = series.astype("int64") // 10 ** 9
    if not allow_absent:
        return seconds.astype("int64")


    return seconds.astype("Int64").mask(absent.to_numpy(dtype=bool))


def build_icu_episodes(*, patient_id, stay_id, start, end,
                       death_time=None) -> pd.DataFrame:
    frame = pd.DataFrame({
        "patient_id": np.asarray(patient_id),
        "stay_id": np.asarray(stay_id),
        "state": ICU_STATE,
        "start": pd.Series(start).reset_index(drop=True).astype("int64"),
        "end": pd.Series(end).reset_index(drop=True).astype("Int64"),
    })

    if death_time is None:
        return frame[list(EPISODE_COLUMNS)].reset_index(drop=True)

    deaths = pd.Series(death_time).reset_index(drop=True).astype("Int64")
    present = deaths.notna().to_numpy(dtype=bool)
    if not present.any():
        return frame[list(EPISODE_COLUMNS)].reset_index(drop=True)

    dead = pd.DataFrame({
        "patient_id": frame.loc[present, "patient_id"].to_numpy(),


        "stay_id": frame.loc[present, "stay_id"].to_numpy(),
        "state": DEAD_STATE,
        "start": deaths[present].to_numpy(dtype="int64"),
        "end": pd.array([pd.NA] * int(present.sum()), dtype="Int64"),
    })

    out = pd.concat([frame, dead], ignore_index=True)


    out["start"] = out["start"].astype("int64")
    out["end"] = out["end"].astype("Int64")
    return out[list(EPISODE_COLUMNS)].reset_index(drop=True)
