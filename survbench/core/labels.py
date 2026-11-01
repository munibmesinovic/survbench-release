from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


from survbench.core.landmarks import (
    SECONDS_PER_HOUR,
    _index_with_death,
    _landmark_seconds,
)


CAUSE_CENSORED: int = 0
CAUSE_ICU_DEATH: int = 1
CAUSE_DISCHARGED_DIED: int = 2
CAUSE_DISCHARGED_ALIVE: int = 3


CAUSE_UNKNOWN: int = -1


CAUSE_LABELS: Mapping[int, str] = {
    CAUSE_CENSORED: "censored (truncated at H, or observation ended earlier)",
    CAUSE_ICU_DEATH: "in-ICU death",
    CAUSE_DISCHARGED_DIED: "ICU discharge alive, died by H",
    CAUSE_DISCHARGED_ALIVE: "ICU discharge alive, alive at H",
    CAUSE_UNKNOWN: "unknown outcome, excluded and counted",
}


FOUR_CAUSE_CODES: tuple[int, ...] = (
    CAUSE_CENSORED, CAUSE_ICU_DEATH, CAUSE_DISCHARGED_DIED,
    CAUSE_DISCHARGED_ALIVE)


THREE_CAUSE_CODES: tuple[int, ...] = (
    CAUSE_CENSORED, CAUSE_ICU_DEATH, CAUSE_DISCHARGED_ALIVE)


LABEL_COLUMNS: tuple[str, ...] = (
    "patient_id", "stay_id", "duration_seconds", "cause", "censoring_reason")


CENSORING_NOT_CENSORED: str = ""
CENSORING_ADMINISTRATIVE: str = "administrative_truncation_at_H"
CENSORING_OBSERVATION_END: str = "observation_ended_before_H"


@dataclass(frozen=True)
class PostIcuDeathSource:

    name: str
    dataset: str
    measures: str
    extends_past_hospital_discharge: bool


MIMICIV_IN_HOSPITAL_DEATH: str = "mimiciv:admissions.deathtime"


MIMICIV_ANY_RECORDED_DEATH: str = "mimiciv:patients.dod"


EICU_IN_HOSPITAL_DEATH: str = "eicu:hospitaldischargestatus"


POST_ICU_DEATH_UNOBSERVABLE = None

POST_ICU_DEATH_SOURCES: Mapping[str, PostIcuDeathSource] = {
    MIMICIV_IN_HOSPITAL_DEATH: PostIcuDeathSource(
        name=MIMICIV_IN_HOSPITAL_DEATH, dataset="mimiciv",
        measures="in-hospital death after ICU discharge",
        extends_past_hospital_discharge=False),
    MIMICIV_ANY_RECORDED_DEATH: PostIcuDeathSource(
        name=MIMICIV_ANY_RECORDED_DEATH, dataset="mimiciv",
        measures="any recorded death, including after hospital discharge",
        extends_past_hospital_discharge=True),
    EICU_IN_HOSPITAL_DEATH: PostIcuDeathSource(
        name=EICU_IN_HOSPITAL_DEATH, dataset="eicu",
        measures="in-hospital death after ICU discharge",
        extends_past_hospital_discharge=False),
}


EICU_IN_ICU_DEATH: str = "eicu:unitdischargestatus"


HIRID_IN_ICU_DEATH: str = "hirid:general_table.discharge_status"


DATASET_POST_ICU_DEATH_SOURCE: Mapping[str, str | None] = {
    "mimiciv": MIMICIV_IN_HOSPITAL_DEATH,
    "eicu": EICU_IN_HOSPITAL_DEATH,
    "hirid": POST_ICU_DEATH_UNOBSERVABLE,
}


DEFAULT_HORIZON_HOURS: int = 240
DEFAULT_HORIZON_SECONDS: np.int64 = (
    np.int64(DEFAULT_HORIZON_HOURS) * np.int64(SECONDS_PER_HOUR))


EXTENDED_HORIZON_HOURS: int = 365 * 24
EXTENDED_HORIZON_SECONDS: np.int64 = (
    np.int64(EXTENDED_HORIZON_HOURS) * np.int64(SECONDS_PER_HOUR))


def _resolve_source(post_icu_death_source: str | None) -> PostIcuDeathSource | None:
    if post_icu_death_source is POST_ICU_DEATH_UNOBSERVABLE:
        return None
    if not isinstance(post_icu_death_source, str):
        raise TypeError(
            f"post_icu_death_source: expected one of {sorted(POST_ICU_DEATH_SOURCES)} or None, "
            f"got {type(post_icu_death_source).__name__} ({post_icu_death_source!r})")
    try:
        return POST_ICU_DEATH_SOURCES[post_icu_death_source]
    except KeyError:
        raise ValueError(
            f"post_icu_death_source: {post_icu_death_source!r} is not one of {sorted(POST_ICU_DEATH_SOURCES)} or None") from None


def _open_episode_death_is_attributable(
        source: PostIcuDeathSource | None) -> bool:
    return source is None or not source.extends_past_hospital_discharge


def _unknown_outcome_mask(patient_id: pd.Series,
                          unknown_outcome_patients: Iterable | None) -> np.ndarray:
    if unknown_outcome_patients is None:
        return np.zeros(len(patient_id), dtype=bool)
    if isinstance(unknown_outcome_patients, (str, bytes)):
        raise TypeError(
            f"unknown_outcome_patients must be a collection of patient ids, not a {type(unknown_outcome_patients).__name__}")


    declared = set(unknown_outcome_patients)
    return np.array([pid in declared for pid in patient_id.to_numpy()],
                    dtype=bool)


@dataclass(frozen=True)
class MortalityLabels:

    rows: pd.DataFrame
    horizon_seconds: int
    n_causes: int
    cause_codes: tuple[int, ...]
    post_icu_death_source: str | None
    n_index_patients: int
    cause_counts: Mapping[int, int]


    n_post_icu_deaths_collapsed: int


    n_excluded_end_before_index_start: int
    n_excluded_unknown_outcome_declared: int
    n_excluded_open_with_no_outcome: int


    n_open_index_episodes: int
    n_death_records_before_index_admission: int
    n_patients_with_no_coherent_death_record: int


    n_censored_at_observation_end: int = 0


    n_censored_administratively: int = 0


    in_icu_death_time_source: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "horizon_seconds": int(self.horizon_seconds),
            "n_causes": int(self.n_causes),
            "cause_codes": [int(k) for k in self.cause_codes],
            "cause_labels": {int(k): CAUSE_LABELS[int(k)]
                             for k in self.cause_codes},
            "post_icu_death_source": self.post_icu_death_source,


            "in_icu_death_time_source": [str(s)
                                         for s in self.in_icu_death_time_source],
            "n_rows": int(len(self.rows)),
            "n_index_patients": int(self.n_index_patients),
            "cause_counts": {int(k): int(v)
                             for k, v in self.cause_counts.items()},
            "n_post_icu_deaths_collapsed": int(self.n_post_icu_deaths_collapsed),
            "n_censored_at_observation_end":
                int(self.n_censored_at_observation_end),
            "n_censored_administratively":
                int(self.n_censored_administratively),
            "n_excluded_end_before_index_start":
                int(self.n_excluded_end_before_index_start),
            "n_excluded_unknown_outcome_declared":
                int(self.n_excluded_unknown_outcome_declared),
            "n_excluded_open_with_no_outcome":
                int(self.n_excluded_open_with_no_outcome),
            "n_open_index_episodes": int(self.n_open_index_episodes),
            "n_death_records_before_index_admission":
                int(self.n_death_records_before_index_admission),
            "n_patients_with_no_coherent_death_record":
                int(self.n_patients_with_no_coherent_death_record),
        }


def _observation_end(patient_id: pd.Series,
                     observation_end) -> tuple[np.ndarray, np.ndarray]:
    series = (observation_end if isinstance(observation_end, pd.Series)
              else pd.Series(observation_end))
    aligned = patient_id.map(series)
    known = aligned.notna().to_numpy(dtype=bool)
    times = np.zeros(len(aligned), dtype="int64")
    if known.any():
        times[known] = aligned[known].to_numpy(dtype="int64")
    return times, known


def label_icu_mortality(
    episodes: pd.DataFrame,
    *,
    post_icu_death_source: str | None,
    horizon_seconds: np.int64 = DEFAULT_HORIZON_SECONDS,
    unknown_outcome_patients: Iterable | None = None,
    observation_end=None,
    in_icu_death_time_source: Iterable[str] = (),
) -> MortalityLabels:
    source = _resolve_source(post_icu_death_source)
    observable = source is not None
    open_episode_death_attributable = _open_episode_death_is_attributable(source)
    horizon = _landmark_seconds(horizon_seconds, "horizon_seconds")
    if (int(horizon) > int(DEFAULT_HORIZON_SECONDS)
            and not (source is not None
                     and source.extends_past_hospital_discharge)):
        raise ValueError(
            f"horizon_seconds {int(horizon)} exceeds H={int(DEFAULT_HORIZON_SECONDS)}s, but "
            f"post_icu_death_source={post_icu_death_source!r} stops at hospital discharge")

    index = _index_with_death(episodes)
    idx = index.rows
    n = len(idx)

    start = idx["start"].to_numpy(dtype="int64")
    horizon_time = start + horizon


    end_observed = idx["end"].notna().to_numpy(dtype=bool)
    end_time = np.zeros(n, dtype="int64")
    if end_observed.any():
        end_time[end_observed] = idx["end"][end_observed].to_numpy(dtype="int64")

    has_death = idx["death_time"].notna().to_numpy(dtype=bool)
    death_time = np.zeros(n, dtype="int64")
    if has_death.any():
        death_time[has_death] = (
            idx["death_time"][has_death].to_numpy(dtype="int64"))


    died_in_index_icu = has_death & (
        (~end_observed & open_episode_death_attributable)
        | (end_observed & (death_time <= end_time)))


    death_by_horizon = has_death & (death_time <= horizon_time)
    end_by_horizon = end_observed & (end_time <= horizon_time)

    cause = np.full(n, CAUSE_UNKNOWN, dtype="int64")
    event_time = np.zeros(n, dtype="int64")


    icu_death = died_in_index_icu & death_by_horizon
    cause[icu_death] = CAUSE_ICU_DEATH
    event_time[icu_death] = death_time[icu_death]


    in_icu_at_horizon = (
        (died_in_index_icu & ~death_by_horizon)
        | (end_observed & ~died_in_index_icu & ~end_by_horizon))
    cause[in_icu_at_horizon] = CAUSE_CENSORED
    event_time[in_icu_at_horizon] = horizon_time[in_icu_at_horizon]


    discharged_alive = end_by_horizon & ~died_in_index_icu
    post_icu_death = discharged_alive & death_by_horizon
    cause[discharged_alive] = CAUSE_DISCHARGED_ALIVE
    if observable:
        cause[post_icu_death] = CAUSE_DISCHARGED_DIED
    event_time[discharged_alive] = end_time[discharged_alive]


    if observation_end is not None:
        obs_end, obs_known = _observation_end(idx["patient_id"], observation_end)
        lost = (discharged_alive & ~post_icu_death & obs_known
                & (obs_end < horizon_time))
        cause[lost] = CAUSE_CENSORED
        event_time[lost] = obs_end[lost]
        n_censored_at_observation_end = int(lost.sum())
    else:
        n_censored_at_observation_end = 0


    unresolved = cause == CAUSE_UNKNOWN


    excluded_incoherent = index.end_before_start
    declared_unknown = _unknown_outcome_mask(idx["patient_id"],
                                             unknown_outcome_patients)
    excluded_declared = declared_unknown & ~excluded_incoherent
    excluded_open = unresolved & ~excluded_incoherent & ~excluded_declared
    keep = ~(excluded_incoherent | excluded_declared | excluded_open)

    codes = FOUR_CAUSE_CODES if observable else THREE_CAUSE_CODES
    kept_cause = cause[keep]


    reason = np.full(n, CENSORING_NOT_CENSORED, dtype=object)
    reason[cause == CAUSE_CENSORED] = CENSORING_ADMINISTRATIVE
    if observation_end is not None:
        reason[lost] = CENSORING_OBSERVATION_END
    rows = pd.DataFrame({
        "patient_id": idx["patient_id"].to_numpy()[keep],
        "stay_id": idx["stay_id"].to_numpy()[keep],
        "duration_seconds": event_time[keep] - start[keep],
        "cause": kept_cause,
        "censoring_reason": reason[keep],
    })


    return MortalityLabels(
        rows=rows,
        horizon_seconds=int(horizon),
        n_causes=len(codes),
        cause_codes=codes,
        post_icu_death_source=post_icu_death_source,
        n_index_patients=n,
        cause_counts={int(k): int((kept_cause == k).sum()) for k in codes},
        n_post_icu_deaths_collapsed=(
            0 if observable else int((post_icu_death & keep).sum())),
        n_censored_at_observation_end=n_censored_at_observation_end,
        n_censored_administratively=(
            int(((cause == CAUSE_CENSORED) & keep).sum())
            - n_censored_at_observation_end),
        n_excluded_end_before_index_start=int(excluded_incoherent.sum()),
        n_excluded_unknown_outcome_declared=int(excluded_declared.sum()),
        n_excluded_open_with_no_outcome=int(excluded_open.sum()),
        n_open_index_episodes=int((~end_observed).sum()),
        n_death_records_before_index_admission=(
            index.n_death_records_before_index_admission),
        n_patients_with_no_coherent_death_record=(
            index.n_patients_with_no_coherent_death_record),
        in_icu_death_time_source=tuple(str(s)
                                       for s in in_icu_death_time_source),
    )


def unlabelled_landmark_rows(landmark_rows: pd.DataFrame,
                             labels: MortalityLabels) -> np.ndarray:
    if "patient_id" not in landmark_rows.columns:
        raise ValueError(
            "landmark_rows has no patient_id column; see survbench.core.landmarks.LANDMARK_COLUMNS")
    labelled = set(labels.rows["patient_id"].to_numpy())
    return np.array([pid not in labelled
                     for pid in landmark_rows["patient_id"].to_numpy()],
                    dtype=bool)


def format_label_report(report: dict) -> list[str]:
    lines: list[str] = []
    n_causes = report["n_causes"]
    if report.get("post_icu_death_source") is None:


        lines.append(f"  Causes ({n_causes}, three-cause schema):")
    else:
        lines.append(f"  Causes ({n_causes}), source="
                     f"{report['post_icu_death_source']}:")


    k1_source = report.get("in_icu_death_time_source") or []
    if k1_source:
        lines.append(f"    k=1 death times read from: {', '.join(k1_source)}")

    n = max(1, report["n_rows"])


    admin = report.get("n_censored_administratively", 0)
    lost = report.get("n_censored_at_observation_end", 0)
    for code, count in sorted(report["cause_counts"].items()):
        lines.append(f"    k={code} {report['cause_labels'][code]}: "
                     f"{count:,d} ({count / n:.1%})")
        if code == CAUSE_CENSORED and lost:
            lines.append(f"        of which {admin:,d} ({admin / n:.1%}) truncated at H (in ICU, alive at the horizon)")
            lines.append(f"                 {lost:,d} ({lost / n:.1%}) "
                         f"censored when observation ended before H")

    dropped = report["n_index_patients"] - report["n_rows"]
    if dropped:
        lines.append(
            f"  Excluded {dropped:,d} of {report['n_index_patients']:,d} "
            f"index patients: "
            f"{report['n_excluded_end_before_index_start']:,d} incoherent, "
            f"{report['n_excluded_open_with_no_outcome']:,d} open with no "
            f"outcome, "
            f"{report['n_excluded_unknown_outcome_declared']:,d} declared "
            f"unknown")
    return lines


__all__ = [
    "CAUSE_CENSORED",
    "CAUSE_DISCHARGED_ALIVE",
    "CAUSE_DISCHARGED_DIED",
    "CAUSE_ICU_DEATH",
    "CAUSE_LABELS",
    "CAUSE_UNKNOWN",
    "CENSORING_ADMINISTRATIVE",
    "CENSORING_NOT_CENSORED",
    "CENSORING_OBSERVATION_END",
    "format_label_report",
    "DATASET_POST_ICU_DEATH_SOURCE",
    "DEFAULT_HORIZON_HOURS",
    "DEFAULT_HORIZON_SECONDS",
    "EICU_IN_HOSPITAL_DEATH",
    "EICU_IN_ICU_DEATH",
    "EXTENDED_HORIZON_HOURS",
    "EXTENDED_HORIZON_SECONDS",
    "FOUR_CAUSE_CODES",
    "HIRID_IN_ICU_DEATH",
    "LABEL_COLUMNS",
    "MIMICIV_ANY_RECORDED_DEATH",
    "MIMICIV_IN_HOSPITAL_DEATH",
    "MortalityLabels",
    "POST_ICU_DEATH_SOURCES",
    "POST_ICU_DEATH_UNOBSERVABLE",
    "PostIcuDeathSource",
    "THREE_CAUSE_CODES",
    "label_icu_mortality",
    "unlabelled_landmark_rows",
]
