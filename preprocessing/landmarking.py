from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from survbench.core.landmarks import (
    MCMED_GRID_SECONDS, MORTALITY_GRID_SECONDS, expand_grid,
)

_SECONDS_PER_HOUR = 3600


LANDMARK_INDEX_NAMES: Tuple[str, str] = ("admission_id", "landmark_hours")


GRIDS: Dict[str, Tuple[np.int64, ...]] = {
    "mimiciv": MORTALITY_GRID_SECONDS,
    "eicu": MORTALITY_GRID_SECONDS,
    "hirid": MORTALITY_GRID_SECONDS,
    "mcmed": MCMED_GRID_SECONDS,
}


def landmarking_enabled(config: dict) -> bool:
    return bool(((config.get('preprocessing') or {})
                 .get('landmarking') or {}).get('enabled', False))


def grid_for(dataset: str) -> Tuple[np.int64, ...]:
    try:
        return GRIDS[dataset]
    except KeyError:
        raise ValueError(
            f"no landmark grid for dataset {dataset!r}; add one to GRIDS") from None


def observation_horizon_hours(config: dict, dataset: str) -> int:
    landmarking = landmarking_enabled(config)
    max_hours = int(config['preprocessing']['max_hours'])
    if not landmarking:
        return max_hours
    largest_landmark = max(int(seconds) for seconds in grid_for(dataset))
    return max(max_hours, largest_landmark // _SECONDS_PER_HOUR)


def assert_min_stay_does_not_delete_the_early_landmarks(
        config: dict, dataset: str) -> None:
    landmarking = landmarking_enabled(config)
    if not landmarking:
        return
    cohort_cfg = config.get('dataset', {}).get('cohort', {}) or {}
    min_stay = float(cohort_cfg.get('min_stay_hours', 0) or 0)
    smallest = min(int(s) for s in grid_for(dataset)) / _SECONDS_PER_HOUR
    if min_stay > smallest:
        raise ValueError(
            f"dataset.cohort.min_stay_hours={min_stay:g}h is above the first {dataset} landmark "
            f"({smallest:g}h); set it to 0 or shorten the grid")


def expand_cohort(cohort_df: pd.DataFrame, episodes: pd.DataFrame,
                  *, dataset: str,
                  duration_col: str = "duration") -> pd.DataFrame:
    expansion = expand_grid(episodes, grid_seconds=grid_for(dataset))
    rows = expansion.rows
    if rows.empty:
        empty = cohort_df.iloc[:0].copy()
        empty.index = pd.MultiIndex.from_arrays(
            [[], []], names=LANDMARK_INDEX_NAMES)
        return empty

    landmark_hours = (rows["landmark_seconds"].to_numpy(dtype="int64")
                      / _SECONDS_PER_HOUR)
    stay_ids = rows["stay_id"].to_numpy()


    known = np.isin(stay_ids, cohort_df.index.to_numpy())
    stay_ids, landmark_hours = stay_ids[known], landmark_hours[known]

    out = cohort_df.loc[stay_ids].copy()
    out.index = pd.MultiIndex.from_arrays(
        [stay_ids, landmark_hours], names=LANDMARK_INDEX_NAMES)
    out[duration_col] = out[duration_col].to_numpy() - landmark_hours
    return out.sort_index()


def assert_landmark_rows_are_at_risk(cohort: pd.DataFrame,
                                     *, duration_col: str = "duration") -> None:
    if cohort.empty:
        return
    durations = cohort[duration_col].to_numpy(dtype="float64")
    bad = durations <= 0
    if bad.any():
        first = cohort.index[bad][0]
        raise ValueError(
            f"{int(bad.sum())} of {len(cohort)} landmark rows have {duration_col} <= 0; "
            f"first is {first} at {durations[bad][0]:.3f}h")
