from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from survbench.ir.schema import declared_time_dtype


SECONDS_PER_HOUR: int = 3600


ICU_STATE: str = "ICU"
DEAD_STATE: str = "dead"


def _seconds(*hours: int) -> tuple[np.int64, ...]:
    return tuple(np.int64(h) * np.int64(SECONDS_PER_HOUR) for h in hours)


def _landmark_seconds(value, argument: str) -> np.int64:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{argument}: expected int64 seconds, got {type(value).__name__} ({value!r})")
    seconds = np.int64(value)
    if seconds < 0:
        raise ValueError(
            f"{argument}: {int(seconds)} seconds is negative")
    return seconds


MORTALITY_GRID_HOURS: tuple[int, ...] = (6, 12, 24, 48, 72, 120, 168)


MORTALITY_HORIZONS_HOURS: tuple[int, ...] = (24, 72, 168)


MORTALITY_MAX_HORIZON_HOURS: int = 240

MORTALITY_GRID_SECONDS: tuple[np.int64, ...] = _seconds(*MORTALITY_GRID_HOURS)
MORTALITY_HORIZONS_SECONDS: tuple[np.int64, ...] = _seconds(
    *MORTALITY_HORIZONS_HOURS)
MORTALITY_MAX_HORIZON_SECONDS: np.int64 = _seconds(
    MORTALITY_MAX_HORIZON_HOURS)[0]


READMISSION_HORIZONS_HOURS: tuple[int, ...] = (24, 72, 168)
READMISSION_HORIZONS_SECONDS: tuple[np.int64, ...] = _seconds(
    *READMISSION_HORIZONS_HOURS)


MCMED_GRID_HOURS: tuple[int, ...] = (2, 3, 6)
MCMED_HORIZONS_HOURS: tuple[int, ...] = (2, 6, 12)
MCMED_MAX_HORIZON_HOURS: int = 24

MCMED_GRID_SECONDS: tuple[np.int64, ...] = _seconds(*MCMED_GRID_HOURS)
MCMED_HORIZONS_SECONDS: tuple[np.int64, ...] = _seconds(*MCMED_HORIZONS_HOURS)
MCMED_MAX_HORIZON_SECONDS: np.int64 = _seconds(MCMED_MAX_HORIZON_HOURS)[0]


LANDMARK_COLUMNS: tuple[str, ...] = (
    "patient_id", "stay_id", "landmark_seconds", "origin_time",
    "landmark_time", "end_observed",
)


GRID_LANDMARK_SOURCE_COLUMN: str = "episodes.start"
ANCHOR_LANDMARK_SOURCE_COLUMN: str = "episodes.end"


_EPISODE_TIME_COLUMNS: tuple[str, ...] = ("start", "end")


def valid_cells(
    grid_seconds: Sequence[np.int64],
    horizons_seconds: Sequence[np.int64],
    max_horizon_seconds: np.int64,
) -> list[tuple[np.int64, np.int64]]:
    cells: list[tuple[np.int64, np.int64]] = []
    cap = _landmark_seconds(max_horizon_seconds, "max_horizon_seconds")
    for raw_l in grid_seconds:
        landmark = _landmark_seconds(raw_l, "grid_seconds")
        for raw_w in horizons_seconds:
            horizon = _landmark_seconds(raw_w, "horizons_seconds")
            if landmark + horizon <= cap:
                cells.append((landmark, horizon))
    return cells


MORTALITY_CELLS: tuple[tuple[np.int64, np.int64], ...] = tuple(valid_cells(
    MORTALITY_GRID_SECONDS, MORTALITY_HORIZONS_SECONDS,
    MORTALITY_MAX_HORIZON_SECONDS))


MCMED_CELLS: tuple[tuple[np.int64, np.int64], ...] = tuple(valid_cells(
    MCMED_GRID_SECONDS, MCMED_HORIZONS_SECONDS, MCMED_MAX_HORIZON_SECONDS))


def _require_declared_dtype(frame: pd.DataFrame, table: str,
                            column: str) -> pd.Series:
    qualified = f"{table}.{column}"
    declared = declared_time_dtype(qualified)
    if column not in frame.columns:
        raise ValueError(
            f"{qualified}: column missing; landmark expansion needs "
            f"{list(_EPISODE_TIME_COLUMNS)}, patient_id, stay_id and state")
    series = frame[column]
    if str(series.dtype) != declared:
        raise TypeError(
            f"{qualified}: dtype is {str(series.dtype)!r}, declared {declared!r}; convert through survbench.ir.time")
    return series


def index_icu_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    for column in ("patient_id", "stay_id", "state"):
        if column not in episodes.columns:
            raise ValueError(
                f"episodes.{column}: column missing; landmark expansion reads "
                f"patient_id, stay_id, state, start and end.")
    for column in _EPISODE_TIME_COLUMNS:
        _require_declared_dtype(episodes, "episodes", column)

    icu = episodes.loc[episodes["state"] == ICU_STATE,
                       ["patient_id", "stay_id", "start", "end"]]
    idx = (icu.sort_values(["patient_id", "start", "stay_id"])
              .drop_duplicates(subset="patient_id", keep="first")
              .reset_index(drop=True))
    idx["duration_seconds"] = idx["end"] - idx["start"]
    return idx


def _death_records(episodes: pd.DataFrame) -> pd.DataFrame:
    dead = episodes.loc[episodes["state"] == DEAD_STATE,
                        ["patient_id", "start"]].copy()
    dead = dead.rename(columns={"start": "death_time"})
    dead["death_time"] = dead["death_time"].astype("Int64")
    return dead


def _at_or_after_index_admission(frame: pd.DataFrame) -> np.ndarray:
    if not len(frame):
        return np.zeros(0, dtype=bool)
    death = frame["death_time"].to_numpy(dtype="int64")
    start = frame["start"].to_numpy(dtype="int64")
    return death >= start


def _index_death_time(joined: pd.DataFrame) -> pd.DataFrame:
    coherent = joined.loc[_at_or_after_index_admission(joined)]
    return (coherent.sort_values(["patient_id", "death_time"])
                    .drop_duplicates(subset="patient_id", keep="first"))


def _end_before_index_start(idx: pd.DataFrame) -> np.ndarray:
    observed = idx["end"].notna().to_numpy(dtype=bool)
    out = np.zeros(len(idx), dtype=bool)
    if observed.any():
        ends = idx["end"][observed].to_numpy(dtype="int64")
        starts = idx["start"].to_numpy(dtype="int64")[observed]
        out[observed] = ends < starts
    return out


@dataclass(frozen=True)
class _IndexFrame:

    rows: pd.DataFrame
    n_death_records_before_index_admission: int
    n_patients_with_no_coherent_death_record: int
    end_before_start: np.ndarray


def _index_with_death(episodes: pd.DataFrame) -> _IndexFrame:
    idx = index_icu_episodes(episodes)
    joined = _death_records(episodes).merge(
        idx[["patient_id", "start"]], on="patient_id", how="inner")
    n_before = int((~_at_or_after_index_admission(joined)).sum())
    earliest = _index_death_time(joined)


    n_no_coherent = len(set(joined["patient_id"]) - set(earliest["patient_id"]))
    rows = idx.merge(earliest[["patient_id", "death_time"]], on="patient_id",
                     how="left")
    return _IndexFrame(
        rows=rows,
        n_death_records_before_index_admission=n_before,
        n_patients_with_no_coherent_death_record=n_no_coherent,
        end_before_start=_end_before_index_start(rows),
    )


def _still_in_episode(idx: pd.DataFrame,
                      landmark_time: np.ndarray) -> np.ndarray:
    observed = idx["end"].notna().to_numpy(dtype=bool)
    ends_after = np.zeros(len(idx), dtype=bool)
    if observed.any():
        ends = idx["end"][observed].to_numpy(dtype="int64")
        ends_after[observed] = ends > landmark_time[observed]


    return ends_after | ~observed


def _dead_by(idx: pd.DataFrame, landmark_time: np.ndarray) -> np.ndarray:
    death = idx["death_time"]
    has_death = death.notna().to_numpy(dtype=bool)
    dead = np.zeros(len(idx), dtype=bool)
    if has_death.any():
        times = death[has_death].to_numpy(dtype="int64")
        dead[has_death] = times <= landmark_time[has_death]
    return dead


def _at_risk(idx: pd.DataFrame, landmark: np.int64) -> np.ndarray:
    landmark_time = idx["start"].to_numpy(dtype="int64") + landmark
    return _still_in_episode(idx, landmark_time) & ~_dead_by(idx, landmark_time)


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": pd.Series([], dtype=object),
        "stay_id": pd.Series([], dtype=object),
        "landmark_seconds": pd.Series([], dtype="int64"),
        "origin_time": pd.Series([], dtype="int64"),
        "landmark_time": pd.Series([], dtype="int64"),
        "end_observed": pd.Series([], dtype=bool),
    })


def _finish(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return _empty_rows()
    out = pd.concat(frames, ignore_index=True)
    return (out.sort_values(["patient_id", "landmark_seconds"])
               .reset_index(drop=True))


@dataclass(frozen=True)
class GridExpansion:

    rows: pd.DataFrame
    landmark_source_column: str
    n_index_patients: int
    n_open_index_episodes: int
    n_rows_from_open_episodes: int
    n_death_records_before_index_admission: int
    n_patients_with_no_coherent_death_record: int
    n_end_before_index_start: int
    at_risk_by_landmark: Mapping[int, int]

    def to_dict(self) -> dict:
        return {
            "landmark_source_column": self.landmark_source_column,
            "n_rows": int(len(self.rows)),
            "n_index_patients": int(self.n_index_patients),
            "n_open_index_episodes": int(self.n_open_index_episodes),
            "n_rows_from_open_episodes": int(self.n_rows_from_open_episodes),
            "n_death_records_before_index_admission":
                int(self.n_death_records_before_index_admission),
            "n_patients_with_no_coherent_death_record":
                int(self.n_patients_with_no_coherent_death_record),
            "n_end_before_index_start": int(self.n_end_before_index_start),
            "at_risk_by_landmark": {int(k): int(v)
                                    for k, v in self.at_risk_by_landmark.items()},
        }


@dataclass(frozen=True)
class AnchorExpansion:

    rows: pd.DataFrame
    landmark_source_column: str
    n_index_patients: int
    n_open_index_episodes: int
    n_died_in_index_icu: int
    n_death_records_before_index_admission: int
    n_patients_with_no_coherent_death_record: int
    n_end_before_index_start: int

    def to_dict(self) -> dict:
        return {
            "landmark_source_column": self.landmark_source_column,
            "n_rows": int(len(self.rows)),
            "n_index_patients": int(self.n_index_patients),
            "n_open_index_episodes": int(self.n_open_index_episodes),
            "n_died_in_index_icu": int(self.n_died_in_index_icu),
            "n_death_records_before_index_admission":
                int(self.n_death_records_before_index_admission),
            "n_patients_with_no_coherent_death_record":
                int(self.n_patients_with_no_coherent_death_record),
            "n_end_before_index_start": int(self.n_end_before_index_start),
        }


def expand_grid(
    episodes: pd.DataFrame,
    grid_seconds: Sequence[np.int64] = MORTALITY_GRID_SECONDS,
) -> GridExpansion:
    seen: set[int] = set()
    grid: list[np.int64] = []
    for raw in grid_seconds:
        landmark = _landmark_seconds(raw, "grid_seconds")
        if int(landmark) in seen:
            raise ValueError(
                f"grid_seconds: {int(landmark)} appears more than once")
        seen.add(int(landmark))
        grid.append(landmark)

    index = _index_with_death(episodes)
    idx = index.rows
    open_episode = (~idx["end"].notna()).to_numpy(dtype=bool)

    frames: list[pd.DataFrame] = []
    at_risk_by_landmark: dict[int, int] = {}
    n_open_rows = 0
    for landmark in grid:
        mask = _at_risk(idx, landmark)
        at_risk_by_landmark[int(landmark)] = int(mask.sum())
        n_open_rows += int((mask & open_episode).sum())
        rows = idx.loc[mask, ["patient_id", "stay_id", "start", "end"]].copy()
        rows["landmark_seconds"] = landmark
        rows["origin_time"] = rows["start"]
        rows["landmark_time"] = rows["start"] + landmark
        rows["end_observed"] = rows["end"].notna().to_numpy(dtype=bool)
        frames.append(rows[list(LANDMARK_COLUMNS)])

    return GridExpansion(
        rows=_finish(frames),
        landmark_source_column=GRID_LANDMARK_SOURCE_COLUMN,
        n_index_patients=int(len(idx)),
        n_open_index_episodes=int(open_episode.sum()),
        n_rows_from_open_episodes=n_open_rows,
        n_death_records_before_index_admission=(
            index.n_death_records_before_index_admission),
        n_patients_with_no_coherent_death_record=(
            index.n_patients_with_no_coherent_death_record),


        n_end_before_index_start=int(index.end_before_start.sum()),
        at_risk_by_landmark=at_risk_by_landmark,
    )


def expand_discharge_anchor(episodes: pd.DataFrame) -> AnchorExpansion:
    index = _index_with_death(episodes)
    idx = index.rows
    observed = idx["end"].notna().to_numpy(dtype=bool)


    anchored = idx.loc[observed & ~index.end_before_start].copy()
    death = anchored["death_time"]
    has_death = death.notna().to_numpy(dtype=bool)
    died_in_icu = np.zeros(len(anchored), dtype=bool)
    if has_death.any():
        times = death[has_death].to_numpy(dtype="int64")
        ends = anchored["end"].to_numpy(dtype="int64")[has_death]


        died_in_icu[has_death] = times <= ends

    alive = anchored.loc[~died_in_icu].copy()


    origin = alive["start"].to_numpy(dtype="int64")
    landmark_time = alive["end"].to_numpy(dtype="int64")
    rows = pd.DataFrame({
        "patient_id": alive["patient_id"].to_numpy(),
        "stay_id": alive["stay_id"].to_numpy(),
        "landmark_seconds": landmark_time - origin,
        "origin_time": origin,
        "landmark_time": landmark_time,
        "end_observed": np.ones(len(alive), dtype=bool),
    })


    rows = (rows.sort_values(["patient_id", "landmark_seconds"])
                .reset_index(drop=True))

    return AnchorExpansion(
        rows=rows,
        landmark_source_column=ANCHOR_LANDMARK_SOURCE_COLUMN,
        n_index_patients=int(len(idx)),
        n_open_index_episodes=int((~observed).sum()),
        n_died_in_index_icu=int(died_in_icu.sum()),
        n_death_records_before_index_admission=(
            index.n_death_records_before_index_admission),
        n_patients_with_no_coherent_death_record=(
            index.n_patients_with_no_coherent_death_record),
        n_end_before_index_start=int(index.end_before_start.sum()),
    )


__all__ = [
    "ANCHOR_LANDMARK_SOURCE_COLUMN",
    "AnchorExpansion",
    "GRID_LANDMARK_SOURCE_COLUMN",
    "GridExpansion",
    "LANDMARK_COLUMNS",
    "MCMED_CELLS",
    "MCMED_GRID_HOURS",
    "MCMED_GRID_SECONDS",
    "MCMED_HORIZONS_HOURS",
    "MCMED_HORIZONS_SECONDS",
    "MCMED_MAX_HORIZON_HOURS",
    "MCMED_MAX_HORIZON_SECONDS",
    "MORTALITY_CELLS",
    "MORTALITY_GRID_HOURS",
    "MORTALITY_GRID_SECONDS",
    "MORTALITY_HORIZONS_HOURS",
    "MORTALITY_HORIZONS_SECONDS",
    "MORTALITY_MAX_HORIZON_HOURS",
    "MORTALITY_MAX_HORIZON_SECONDS",
    "READMISSION_HORIZONS_HOURS",
    "READMISSION_HORIZONS_SECONDS",
    "SECONDS_PER_HOUR",
    "expand_discharge_anchor",
    "expand_grid",
    "index_icu_episodes",
    "valid_cells",
]
