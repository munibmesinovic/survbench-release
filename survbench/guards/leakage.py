from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from survbench.ir.schema import (
    TIME_COLUMN_DTYPES,
    declared_time_dtype,
    iter_time_columns,
)
from survbench.ir.time import TIME_UNIT, ConversionLog


SOURCE_COUNT_KEYS: tuple[str, str] = ("n_source_values", "n_ir_values")


@dataclass(frozen=True)
class ColumnResult:

    column: str
    declared_dtype: str
    dtype: str | None
    n_rows: int | None
    n_values: int | None
    n_source_values: int | None
    n_ir_values_declared: int | None
    recorded_time_origin: str | None = None
    contributors: dict[str, dict[str, int]] = field(default_factory=dict)
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "declared_dtype": self.declared_dtype,
            "dtype": self.dtype,
            "n_rows": self.n_rows,
            "n_values": self.n_values,
            "n_source_values": self.n_source_values,
            "n_ir_values_declared": self.n_ir_values_declared,
            "recorded_time_origin": self.recorded_time_origin,
            "contributors": {k: dict(v) for k, v in self.contributors.items()},
            "passed": self.passed,
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class TimeRepresentationReport:

    time_unit: str | None
    time_origin: str | None
    columns: dict[str, ColumnResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.columns.values())

    def failures(self) -> list[str]:
        out: list[str] = []
        for name in self.columns:
            out.extend(self.columns[name].failures)
        return out

    def require(self, column: str) -> ColumnResult:
        result = self.columns.get(column)
        if result is None:
            raise ValueError(
                f"G7 has no result for {column!r}; G4 and G5 need it first")
        if not result.passed:
            raise ValueError(
                f"G7 failed for {column!r}: {'; '.join(result.failures)}")
        return result

    def to_dict(self) -> dict:
        return {
            "time_unit": self.time_unit,
            "time_origin": self.time_origin,
            "passed": self.passed,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
        }


def require_g7(report: TimeRepresentationReport | None,
               *columns: str) -> None:
    if report is None:
        raise ValueError(
            f"no G7 result supplied for {list(columns)}; run G7 before G4/G5")
    for column in columns:
        report.require(column)


def _validate_source_counts(
    source_counts: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> None:
    for column, contributors in source_counts.items():
        if column not in TIME_COLUMN_DTYPES:
            raise ValueError(
                f"source counts supplied for {column!r}, which is not a "
                f"declared IR time column; declared: "
                f"{sorted(TIME_COLUMN_DTYPES)}")
        if not contributors:
            raise ValueError(
                f"{column}: no contributing source column named")
        for name, counts in contributors.items():
            missing = [k for k in SOURCE_COUNT_KEYS if k not in counts]
            if missing:
                raise ValueError(
                    f"{column}: contributor {name!r} is missing {missing}; source and IR counts are both required")


def check_time_representation(
    tables: Mapping[str, pd.DataFrame],
    *,
    time_unit: str | None,
    time_origin: str | None,
    source_counts: Mapping[str, Mapping[str, Mapping[str, int]]],
    conversions: ConversionLog | None = None,
) -> TimeRepresentationReport:
    _validate_source_counts(source_counts)

    declared_failures: list[str] = []
    if time_unit is None or not str(time_unit).strip():
        declared_failures.append(
            "time_unit is not declared")
    elif time_unit != TIME_UNIT:


        declared_failures.append(
            f"time_unit is {time_unit!r} but survbench.ir.time produces {TIME_UNIT!r}")
    if time_origin is None or not str(time_origin).strip():
        declared_failures.append(
            "time_origin is not declared")

    results: dict[str, ColumnResult] = {}
    for column, declared, series in iter_time_columns(tables):
        failures: list[str] = list(declared_failures)
        contributors = {k: dict(v)
                        for k, v in source_counts.get(column, {}).items()}


        record = None if conversions is None else conversions.get(column)
        recorded_origin = None if record is None else record.time_origin
        if record is None:
            failures.append(
                f"{column}: no conversion record; pass the ConversionLog from survbench.ir.time")
        elif time_origin is not None and recorded_origin != time_origin:
            failures.append(
                f"{column}: declared time_origin={time_origin!r} but the "
                f"conversion recorded {recorded_origin!r}")

        if series is None:
            failures.append(
                f"{column}: absent from the IR, so no value count can be taken")
            results[column] = ColumnResult(
                column=column, declared_dtype=declared, dtype=None,
                n_rows=None, n_values=None, n_source_values=None,
                n_ir_values_declared=None,
                recorded_time_origin=recorded_origin,
                contributors=contributors, failures=tuple(failures))
            continue

        dtype = str(series.dtype)
        n_rows = int(len(series))


        n_values = int(series.notna().sum())

        if dtype != declared:
            failures.append(
                f"{column}: dtype is {dtype!r}, declared {declared!r}")

        n_source_values: int | None = None
        n_ir_declared: int | None = None
        if not contributors:
            failures.append(
                f"{column}: no source counts supplied")
        else:
            n_source_values = sum(int(c["n_source_values"])
                                  for c in contributors.values())
            n_ir_declared = sum(int(c["n_ir_values"])
                                for c in contributors.values())


            if n_values != n_source_values:
                failures.append(
                    f"{column}: n_values={n_values} != n_non_null(source)={n_source_values}; "
                    f"{abs(n_values - n_source_values)} value(s) dropped, nulled or added")


            if n_ir_declared != n_values:
                failures.append(
                    f"{column}: emitted per-contributor IR counts sum to "
                    f"{n_ir_declared} but the column holds {n_values} value(s)")


            for name, counts in contributors.items():
                got, want = int(counts["n_ir_values"]), int(counts["n_source_values"])
                if got != want:
                    failures.append(
                        f"{column}: contributor {name!r} n_ir_values={got} != "
                        f"n_source_values={want}")

        results[column] = ColumnResult(
            column=column, declared_dtype=declared, dtype=dtype,
            n_rows=n_rows, n_values=n_values,
            n_source_values=n_source_values,
            n_ir_values_declared=n_ir_declared,
            recorded_time_origin=recorded_origin,
            contributors=contributors, failures=tuple(failures))

    return TimeRepresentationReport(time_unit=time_unit,
                                    time_origin=time_origin, columns=results)


def assert_time_representation(
    tables: Mapping[str, pd.DataFrame],
    *,
    time_unit: str | None,
    time_origin: str | None,
    source_counts: Mapping[str, Mapping[str, Mapping[str, int]]],
    conversions: ConversionLog | None = None,
) -> TimeRepresentationReport:
    report = check_time_representation(
        tables, time_unit=time_unit, time_origin=time_origin,
        source_counts=source_counts, conversions=conversions)
    if not report.passed:
        raise ValueError(
            "G7 assert_time_representation failed:\n  - "
            + "\n  - ".join(report.failures()))
    return report


FOLDS: tuple[str, ...] = ("train", "val", "test")


TRAIN_FOLD: str = "train"


LANDMARK_INDEPENDENT: str = "landmark_independent"


FOLD_PAIRS: tuple[tuple[str, str], ...] = tuple(itertools.combinations(FOLDS, 2))


@dataclass(frozen=True)
class SplitDisjointnessReport:

    patient_id_field: str | None
    n_ids: dict[str, int]
    n_patients: dict[str, int]
    overlaps: dict[str, int]
    n_subject_patients: int = 0
    n_not_in_subjects: int = 0
    overlapping_examples: dict[str, list[str]] = field(default_factory=dict)
    not_in_subjects_examples: list[str] = field(default_factory=list)
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "patient_id_field": self.patient_id_field,
            "n_ids": dict(self.n_ids),
            "n_patients": dict(self.n_patients),
            "overlaps": dict(self.overlaps),
            "n_subject_patients": self.n_subject_patients,
            "n_not_in_subjects": self.n_not_in_subjects,
            "overlapping_examples": {k: list(v)
                                     for k, v in self.overlapping_examples.items()},
            "not_in_subjects_examples": list(self.not_in_subjects_examples),
            "passed": self.passed,
            "failures": list(self.failures),
        }


def check_splits_disjoint(
    train_ids: Iterable[Hashable],
    val_ids: Iterable[Hashable],
    test_ids: Iterable[Hashable],
    *,
    patient_id_field: str | None,
    subject_patient_ids: Iterable[Hashable] | None,
) -> SplitDisjointnessReport:
    ids: dict[str, list[Hashable]] = {
        "train": list(train_ids), "val": list(val_ids), "test": list(test_ids)}
    sets = {fold: set(members) for fold, members in ids.items()}

    overlaps: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for left, right in FOLD_PAIRS:
        shared = sets[left] & sets[right]
        overlaps[f"{left}_{right}"] = len(shared)
        if shared:
            examples[f"{left}_{right}"] = sorted(map(str, shared))[:5]

    subjects: set[Hashable] = (
        set() if subject_patient_ids is None else set(subject_patient_ids))
    outside = (sets["train"] | sets["val"] | sets["test"]) - subjects
    outside_examples = sorted(map(str, outside))[:5]

    failures: list[str] = []
    if patient_id_field is None or not str(patient_id_field).strip():
        failures.append(
            "no patient_id_field declared")


    if not subjects:
        failures.append(
            f"no subjects table to check the split against "
            f"(subject_patient_ids is {'None' if subject_patient_ids is None else 'empty'})")
    bad = {pair: n for pair, n in overlaps.items() if n}
    if bad:
        failures.append(
            f"patient-level split leakage: {bad} (checked on "
            f"patient_id_field={patient_id_field!r}); examples: {examples}")
    if outside:
        failures.append(
            f"{len(outside)} split id(s) not in subjects.patient_id: {outside_examples}")

    return SplitDisjointnessReport(
        patient_id_field=patient_id_field,
        n_ids={fold: len(members) for fold, members in ids.items()},
        n_patients={fold: len(members) for fold, members in sets.items()},
        overlaps=overlaps,
        n_subject_patients=len(subjects),
        n_not_in_subjects=len(outside),
        overlapping_examples=examples,
        not_in_subjects_examples=outside_examples,
        failures=tuple(failures))


def assert_splits_disjoint(
    train_ids: Iterable[Hashable],
    val_ids: Iterable[Hashable],
    test_ids: Iterable[Hashable],
    *,
    patient_id_field: str | None,
    subject_patient_ids: Iterable[Hashable] | None,
) -> SplitDisjointnessReport:
    report = check_splits_disjoint(train_ids, val_ids, test_ids,
                                   patient_id_field=patient_id_field,
                                   subject_patient_ids=subject_patient_ids)
    if not report.passed:
        raise ValueError("G1 assert_splits_disjoint failed:\n  - "
                         + "\n  - ".join(report.failures))
    return report


@dataclass(frozen=True)
class RowFoldReport:

    n_rows: dict[str, int]
    n_patients: dict[str, int]
    n_rows_by_landmark: dict[str, dict[str, int]] = field(default_factory=dict)
    spanning: dict[str, list[str]] = field(default_factory=dict)
    disagreeing: dict[str, list[str]] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)
    undeclared_folds: dict[str, dict[str, int]] = field(default_factory=dict)
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "n_rows": dict(self.n_rows),
            "n_patients": dict(self.n_patients),
            "n_rows_by_landmark": {k: dict(v)
                                   for k, v in self.n_rows_by_landmark.items()},
            "spanning": {k: list(v) for k, v in self.spanning.items()},
            "disagreeing": {k: list(v) for k, v in self.disagreeing.items()},
            "unknown": list(self.unknown),
            "undeclared_folds": {k: dict(v)
                                 for k, v in self.undeclared_folds.items()},
            "passed": self.passed,
            "failures": list(self.failures),
        }


def check_rows_respect_patient_folds(
    row_patient_ids: Sequence[Hashable],
    fold_of_row: Sequence[str],
    fold_of_patient: Mapping[Hashable, str],
    *,
    landmark_of_row: Sequence[Hashable] | None = None,
) -> RowFoldReport:
    if len(row_patient_ids) != len(fold_of_row):


        raise ValueError(
            f"length mismatch: {len(row_patient_ids)} rows, "
            f"{len(fold_of_row)} fold labels")
    if landmark_of_row is not None and len(landmark_of_row) != len(row_patient_ids):
        raise ValueError(
            f"length mismatch: {len(row_patient_ids)} rows, "
            f"{len(landmark_of_row)} landmark labels")

    n_rows: dict[str, int] = {}
    patients_in_fold: dict[str, set[Hashable]] = {}
    by_landmark: dict[str, dict[str, int]] = {}
    seen: dict[Hashable, set[str]] = {}
    for position, (pid, fold) in enumerate(zip(row_patient_ids, fold_of_row)):
        n_rows[fold] = n_rows.get(fold, 0) + 1
        patients_in_fold.setdefault(fold, set()).add(pid)
        seen.setdefault(pid, set()).add(fold)
        if landmark_of_row is not None:
            landmark = str(landmark_of_row[position])
            cell = by_landmark.setdefault(fold, {})
            cell[landmark] = cell.get(landmark, 0) + 1

    failures: list[str] = []

    undeclared: dict[str, dict[str, int]] = {}
    for fold, count in n_rows.items():
        if fold not in FOLDS:
            undeclared.setdefault(fold, {"n_rows": 0, "n_patients": 0})
            undeclared[fold]["n_rows"] = count
    for pid, fold in fold_of_patient.items():
        if fold not in FOLDS:
            undeclared.setdefault(fold, {"n_rows": 0, "n_patients": 0})
            undeclared[fold]["n_patients"] += 1
    if undeclared:
        failures.append(
            f"{len(undeclared)} fold label(s) outside {list(FOLDS)}: {undeclared}")

    unknown = sorted(str(p) for p in seen if p not in fold_of_patient)
    if unknown:
        failures.append(
            f"{len(unknown)} patient(s) with rows but no fold assignment: "
            f"{unknown[:5]}")

    spanning = {str(p): sorted(folds) for p, folds in seen.items()
                if len(folds) > 1}
    if spanning:
        failures.append(
            f"{len(spanning)} patient(s) have rows span folds: "
            f"{dict(list(spanning.items())[:5])}")

    disagreeing = {
        str(p): sorted(folds - {fold_of_patient[p]})
        for p, folds in seen.items()
        if p in fold_of_patient and folds - {fold_of_patient[p]}
    }
    if disagreeing:
        failures.append(
            f"{len(disagreeing)} row fold(s) disagrees with its patient's "
            f"fold: {dict(list(disagreeing.items())[:5])}")

    return RowFoldReport(
        n_rows=n_rows,
        n_patients={fold: len(members)
                    for fold, members in patients_in_fold.items()},
        n_rows_by_landmark=by_landmark, spanning=spanning,
        disagreeing=disagreeing, unknown=unknown, undeclared_folds=undeclared,
        failures=tuple(failures))


def assert_rows_respect_patient_folds(
    row_patient_ids: Sequence[Hashable],
    fold_of_row: Sequence[str],
    fold_of_patient: Mapping[Hashable, str],
    *,
    landmark_of_row: Sequence[Hashable] | None = None,
) -> RowFoldReport:
    report = check_rows_respect_patient_folds(
        row_patient_ids, fold_of_row, fold_of_patient,
        landmark_of_row=landmark_of_row)
    if not report.passed:
        raise ValueError("G2 assert_rows_respect_patient_folds failed:\n  - "
                         + "\n  - ".join(report.failures))
    return report


@dataclass(frozen=True)
class FitPopulationResult:

    name: str
    n_fitted: int
    n_train: int
    n_leaked: int
    leaked_examples: list[str] = field(default_factory=list)
    fold: str | None = None
    fit_landmark: str | None = None
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_fitted": self.n_fitted,
            "n_train": self.n_train,
            "n_leaked": self.n_leaked,
            "leaked_examples": list(self.leaked_examples),
            "fold": self.fold,
            "fit_landmark": self.fit_landmark,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def check_fit_on_train_only(
    fitted_on_patient_ids: Iterable[Hashable],
    train_patient_ids: Iterable[Hashable],
    name: str,
    *,
    fold: str | None,
    fit_landmark: Hashable | None,
) -> FitPopulationResult:
    fitted = set(fitted_on_patient_ids)
    train = set(train_patient_ids)
    leaked = fitted - train

    failures: list[str] = []
    if not fitted:
        failures.append(f"{name}: no recorded fit population")
    if leaked:
        failures.append(
            f"{name}: fit on {len(leaked)} non-training patient(s): "
            f"{sorted(map(str, leaked))[:5]}")
    if fold is None or not str(fold).strip():
        failures.append(
            f"{name}: no recorded fit fold")
    elif fold != TRAIN_FOLD:
        failures.append(
            f"{name}: records fold {fold!r}, not the training fold "
            f"{TRAIN_FOLD!r}")
    if fit_landmark is None:
        failures.append(
            f"{name}: no recorded fit landmark; declare fit_landmark={LANDMARK_INDEPENDENT!r}")
    elif fit_landmark != LANDMARK_INDEPENDENT:
        failures.append(
            f"{name}: fit_landmark={fit_landmark!r}; fold-fit objects must be fit once, landmark-independently")

    return FitPopulationResult(
        name=name, n_fitted=len(fitted), n_train=len(train),
        n_leaked=len(leaked),
        leaked_examples=sorted(map(str, leaked))[:5],
        fold=fold,
        fit_landmark=None if fit_landmark is None else str(fit_landmark),
        failures=tuple(failures))


def assert_fit_on_train_only(
    fitted_on_patient_ids: Iterable[Hashable],
    train_patient_ids: Iterable[Hashable],
    name: str,
    *,
    fold: str | None,
    fit_landmark: Hashable | None,
) -> FitPopulationResult:
    result = check_fit_on_train_only(fitted_on_patient_ids, train_patient_ids,
                                     name, fold=fold, fit_landmark=fit_landmark)
    if not result.passed:
        raise ValueError("G3 assert_fit_on_train_only failed:\n  - "
                         + "\n  - ".join(result.failures))
    return result


@dataclass(frozen=True)
class BootstrapUnitReport:

    n_source_rows: int
    n_resampled_rows: int
    n_source_patients: int
    n_patients_drawn: int
    n_patients_absent: int
    min_draws_per_patient: int | None = None
    max_draws_per_patient: int | None = None
    offenders: dict[str, dict[str, int]] = field(default_factory=dict)
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "n_source_rows": self.n_source_rows,
            "n_resampled_rows": self.n_resampled_rows,
            "n_source_patients": self.n_source_patients,
            "n_patients_drawn": self.n_patients_drawn,
            "n_patients_absent": self.n_patients_absent,
            "min_draws_per_patient": self.min_draws_per_patient,
            "max_draws_per_patient": self.max_draws_per_patient,
            "offenders": {k: dict(v) for k, v in self.offenders.items()},
            "passed": self.passed,
            "failures": list(self.failures),
        }


def check_bootstrap_unit_is_patient(
    resampled_row_index: Sequence[int],
    row_patient_ids: Sequence[Hashable],
) -> BootstrapUnitReport:
    ids = np.asarray(row_patient_ids, dtype=object)
    idx = np.asarray(resampled_row_index, dtype=int)
    if idx.size and (idx.min() < 0 or idx.max() >= ids.size):


        raise ValueError("resample index out of range for row_patient_ids")

    draws_per_row = np.bincount(idx, minlength=int(ids.size))

    rows_of: dict[Hashable, list[int]] = {}
    for position, pid in enumerate(ids.tolist()):
        rows_of.setdefault(pid, []).append(position)

    offenders: dict[str, dict[str, int]] = {}
    clean_draws: dict[Hashable, int] = {}
    n_drawn = 0
    for pid, rows in rows_of.items():
        multiplicities = {int(draws_per_row[r]) for r in rows}
        if any(m > 0 for m in multiplicities):
            n_drawn += 1
        if len(multiplicities) > 1:
            offenders[str(pid)] = {str(r): int(draws_per_row[r]) for r in rows}
        else:
            clean_draws[pid] = multiplicities.pop()

    failures: list[str] = []
    if offenders:
        failures.append(
            f"{len(offenders)} patient(s) not resampled as whole patients "
            f"(row -> times drawn): {dict(list(offenders.items())[:5])}")

    drawn_counts = [k for k in clean_draws.values() if k > 0]
    return BootstrapUnitReport(
        n_source_rows=int(ids.size),
        n_resampled_rows=int(idx.size),
        n_source_patients=len(rows_of),
        n_patients_drawn=n_drawn,
        n_patients_absent=len(rows_of) - n_drawn,
        min_draws_per_patient=min(drawn_counts) if drawn_counts else None,
        max_draws_per_patient=max(drawn_counts) if drawn_counts else None,
        offenders=offenders, failures=tuple(failures))


def assert_bootstrap_unit_is_patient(
    resampled_row_index: Sequence[int],
    row_patient_ids: Sequence[Hashable],
) -> BootstrapUnitReport:
    report = check_bootstrap_unit_is_patient(resampled_row_index,
                                             row_patient_ids)
    if not report.passed:
        raise ValueError("G6 assert_bootstrap_unit_is_patient failed:\n  - "
                         + "\n  - ".join(report.failures))
    return report


IR_TIME_DTYPE: str = "int64"


DOCUMENT_TIME_COLUMN: str = "documents.time"


MEASUREMENT_TIME_COLUMN: str = "measurements.time"


def _int64_times(values, argument: str) -> np.ndarray:
    dtype = getattr(values, "dtype", None)
    if dtype is None:
        raise TypeError(
            f"{argument}: expected {IR_TIME_DTYPE} seconds, got {type(values).__name__}; "
            f"pass the IR column or np.{IR_TIME_DTYPE}(L) for a scalar")
    if str(dtype) != IR_TIME_DTYPE:
        raise TypeError(
            f"{argument}: dtype is {str(dtype)!r}, expected {IR_TIME_DTYPE!r}; convert through survbench.ir.time")
    return np.asarray(values)


def _broadcast(array: np.ndarray, n: int, argument: str) -> np.ndarray:
    if array.ndim == 0:
        return np.broadcast_to(array, (n,))
    if array.shape != (n,):
        raise ValueError(
            f"{argument}: shape {array.shape} does not match the {n} row(s) of the first argument; "
            f"use length {n} or a 0-d scalar")
    return array


def _ids(values, n: int, argument: str) -> np.ndarray:
    return _broadcast(np.asarray(values, dtype=object), n, argument)


def _landmark_column(name: str | None) -> str:
    if name is None or not str(name).strip():
        raise ValueError(
            "no landmark_source_column declared")
    declared_time_dtype(name)
    return name


@dataclass(frozen=True)
class _LatenessFinding:

    mask: np.ndarray
    n: int
    positions: list[int]
    worst_seconds: int | None
    message: str


def _after_landmark(times: np.ndarray, landmarks: np.ndarray,
                    noun: str) -> _LatenessFinding:
    at_or_after = ~(times < landmarks)
    lateness = (times - landmarks)[at_or_after]
    n = int(at_or_after.sum())
    positions = [int(i) for i in np.flatnonzero(at_or_after)[:5]]
    worst = int(lateness.max()) if n else None
    return _LatenessFinding(
        mask=at_or_after, n=n, positions=positions, worst_seconds=worst,
        message=(f"{n} {noun}(s) at or after the landmark (t < L required); worst is {worst}s past it; at row(s) {positions}"))


@dataclass(frozen=True)
class DocumentCausalityReport:

    n_documents: int
    n_at_or_after_landmark: int
    n_from_index_encounter: int
    n_offending_documents: int
    max_seconds_after_landmark: int | None = None
    at_or_after_positions: list[int] = field(default_factory=list)
    index_encounter_positions: list[int] = field(default_factory=list)
    g7_columns: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "n_documents": self.n_documents,
            "n_at_or_after_landmark": self.n_at_or_after_landmark,
            "n_from_index_encounter": self.n_from_index_encounter,
            "n_offending_documents": self.n_offending_documents,
            "max_seconds_after_landmark": self.max_seconds_after_landmark,
            "at_or_after_positions": list(self.at_or_after_positions),
            "index_encounter_positions": list(self.index_encounter_positions),
            "g7_columns": list(self.g7_columns),
            "passed": self.passed,
            "failures": list(self.failures),
        }


def check_documents_are_causal(
    doc_times,
    landmark_times,
    source_encounter_ids: Sequence[Hashable],
    index_encounter_ids: Sequence[Hashable],
    *,
    g7_report: TimeRepresentationReport | None,
    landmark_source_column: str | None,
) -> DocumentCausalityReport:
    columns = (DOCUMENT_TIME_COLUMN, _landmark_column(landmark_source_column))
    require_g7(g7_report, *columns)

    t = _int64_times(doc_times, "doc_times")
    if t.ndim != 1:
        raise ValueError(
            f"doc_times must be a 1-D column of IR seconds, got shape {t.shape}")
    n = int(t.shape[0])
    landmarks = _broadcast(_int64_times(landmark_times, "landmark_times"), n,
                           "landmark_times")
    source = _ids(source_encounter_ids, n, "source_encounter_ids")
    index = _ids(index_encounter_ids, n, "index_encounter_ids")


    late = _after_landmark(t, landmarks, "document")
    from_index = np.asarray(source == index, dtype=bool)
    offending = late.mask | from_index
    n_from_index = int(from_index.sum())
    index_positions = [int(i) for i in np.flatnonzero(from_index)[:5]]

    failures: list[str] = []
    if late.n:
        failures.append(f"causal document leakage: {late.message}")
    if n_from_index:
        failures.append(
            f"causal document leakage: {n_from_index} document(s) from the index encounter; at row(s) {index_positions}")

    return DocumentCausalityReport(
        n_documents=n,
        n_at_or_after_landmark=late.n,
        n_from_index_encounter=n_from_index,
        n_offending_documents=int(offending.sum()),
        max_seconds_after_landmark=late.worst_seconds,
        at_or_after_positions=late.positions,
        index_encounter_positions=index_positions,
        g7_columns=columns,
        failures=tuple(failures))


def assert_documents_are_causal(
    doc_times,
    landmark_times,
    source_encounter_ids: Sequence[Hashable],
    index_encounter_ids: Sequence[Hashable],
    *,
    g7_report: TimeRepresentationReport | None,
    landmark_source_column: str | None,
) -> DocumentCausalityReport:
    report = check_documents_are_causal(
        doc_times, landmark_times, source_encounter_ids, index_encounter_ids,
        g7_report=g7_report, landmark_source_column=landmark_source_column)
    if not report.passed:
        raise ValueError("G4 assert_documents_are_causal failed:\n  - "
                         + "\n  - ".join(report.failures))
    return report


@dataclass(frozen=True)
class PostLandmarkMeasurementReport:

    n_measurements: int
    n_at_or_after_landmark: int
    max_seconds_after_landmark: int | None = None
    at_or_after_positions: list[int] = field(default_factory=list)
    g7_columns: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "n_measurements": self.n_measurements,
            "n_at_or_after_landmark": self.n_at_or_after_landmark,
            "max_seconds_after_landmark": self.max_seconds_after_landmark,
            "at_or_after_positions": list(self.at_or_after_positions),
            "g7_columns": list(self.g7_columns),
            "passed": self.passed,
            "failures": list(self.failures),
        }


def check_no_post_landmark_measurements(
    meas_times,
    landmark_times,
    *,
    g7_report: TimeRepresentationReport | None,
    landmark_source_column: str | None,
) -> PostLandmarkMeasurementReport:
    columns = (MEASUREMENT_TIME_COLUMN, _landmark_column(landmark_source_column))
    require_g7(g7_report, *columns)

    t = _int64_times(meas_times, "meas_times")
    if t.ndim != 1:
        raise ValueError(
            f"meas_times must be a 1-D column of IR seconds, got shape {t.shape}")
    n = int(t.shape[0])
    landmarks = _broadcast(_int64_times(landmark_times, "landmark_times"), n,
                           "landmark_times")

    late = _after_landmark(t, landmarks, "measurement")

    failures: list[str] = []
    if late.n:
        failures.append(f"post-landmark leakage: {late.message}")

    return PostLandmarkMeasurementReport(
        n_measurements=n,
        n_at_or_after_landmark=late.n,
        max_seconds_after_landmark=late.worst_seconds,
        at_or_after_positions=late.positions,
        g7_columns=columns,
        failures=tuple(failures))


def assert_no_post_landmark_measurements(
    meas_times,
    landmark_times,
    *,
    g7_report: TimeRepresentationReport | None,
    landmark_source_column: str | None,
) -> PostLandmarkMeasurementReport:
    report = check_no_post_landmark_measurements(
        meas_times, landmark_times, g7_report=g7_report,
        landmark_source_column=landmark_source_column)
    if not report.passed:
        raise ValueError("G5 assert_no_post_landmark_measurements failed:\n  - "
                         + "\n  - ".join(report.failures))
    return report


__all__: Sequence[str] = [
    "BootstrapUnitReport",
    "ColumnResult",
    "DOCUMENT_TIME_COLUMN",
    "DocumentCausalityReport",
    "FOLDS",
    "FOLD_PAIRS",
    "FitPopulationResult",
    "IR_TIME_DTYPE",
    "LANDMARK_INDEPENDENT",
    "MEASUREMENT_TIME_COLUMN",
    "PostLandmarkMeasurementReport",
    "RowFoldReport",
    "SplitDisjointnessReport",
    "TRAIN_FOLD",
    "TimeRepresentationReport",
    "assert_bootstrap_unit_is_patient",
    "assert_documents_are_causal",
    "assert_fit_on_train_only",
    "assert_no_post_landmark_measurements",
    "assert_rows_respect_patient_folds",
    "assert_splits_disjoint",
    "assert_time_representation",
    "check_bootstrap_unit_is_patient",
    "check_documents_are_causal",
    "check_fit_on_train_only",
    "check_no_post_landmark_measurements",
    "check_rows_respect_patient_folds",
    "check_splits_disjoint",
    "check_time_representation",
    "require_g7",
]
