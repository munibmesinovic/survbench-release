from __future__ import annotations

from typing import Iterator, Mapping

import pandas as pd



TABLES: tuple[str, ...] = ("subjects", "episodes", "measurements", "documents")


REQUIRED_COLUMNS: dict[str, set[str]] = {
    "subjects": {"patient_id"},
    "episodes": {"patient_id", "stay_id", "state", "start", "end"},
    "measurements": {"patient_id", "stay_id", "time", "code", "value", "unit"},
    "documents": {"patient_id", "time", "kind", "source_encounter_id",
                  "payload"},
}

VALID_STATES: frozenset[str] = frozenset(
    {"ICU", "ward", "hospital", "dead", "discharged"}
)


TERMINAL_STATES: frozenset[str] = frozenset({"dead", "discharged"})


TIME_COLUMN_DTYPES: dict[str, str] = {
    "episodes.start": "int64",
    "episodes.end": "Int64",
    "measurements.time": "int64",
    "documents.time": "int64",
}


TIME_COLUMNS: tuple[str, ...] = tuple(TIME_COLUMN_DTYPES)


TOTAL_TIME_COLUMNS: tuple[str, ...] = tuple(
    c for c, dtype in TIME_COLUMN_DTYPES.items() if dtype == "int64"
)


MASKED_TIME_COLUMNS: tuple[str, ...] = tuple(
    c for c, dtype in TIME_COLUMN_DTYPES.items() if dtype == "Int64"
)


def split_column(qualified: str) -> tuple[str, str]:
    table, _, column = qualified.partition(".")
    if not table or not column:
        raise ValueError(
            f"expected a qualified 'table.column' name, got {qualified!r}")
    return table, column


def declared_time_dtype(qualified: str) -> str:
    try:
        return TIME_COLUMN_DTYPES[qualified]
    except KeyError:
        raise ValueError(
            f"{qualified!r} is not a declared IR time column; the declared "
            f"ones are {sorted(TIME_COLUMN_DTYPES)}") from None


def iter_time_columns(
    tables: Mapping[str, pd.DataFrame],
) -> Iterator[tuple[str, str, pd.Series | None]]:
    for qualified, declared in TIME_COLUMN_DTYPES.items():
        table, column = split_column(qualified)
        frame = tables.get(table)
        if frame is None or column not in frame.columns:
            yield qualified, declared, None
        else:
            yield qualified, declared, frame[column]


def time_dtype_violations(
    tables: Mapping[str, pd.DataFrame],
) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for qualified, declared, series in iter_time_columns(tables):
        actual = "<absent>" if series is None else str(series.dtype)
        if actual != declared:
            out.append((qualified, declared, actual))
    return out
