from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TIME_UNIT: str = "s"


UNIX_EPOCH: str = "unix_epoch"


PROHIBITED_IDIOMS: tuple[tuple[str, str], ...] = (
    ("pd.read_csv(..., parse_dates=[...])",
     "degrades the whole column to object on an out-of-range value"),
    ("pd.to_datetime(..., errors='coerce')",
     "returns NaT for out-of-range values without warning"),
    ("np.asarray(..., dtype='datetime64[ns]')",
     "wraps out-of-range int64 values to wrong dates"),
)


_NS_DTYPE = re.compile(r"[<>|]?(?:datetime64|M8)\[ns(?:,\s*[^\]]+)?\]")


_ISO = re.compile(
    r"^\s*(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?:[ T](?P<hh>\d{2}):(?P<mm>\d{2})(?::(?P<ss>\d{2}))?(?P<frac>\.\d+)?)?"
    r"(?P<utc>Z)?"
    r"\s*$"
)


OFFSET_UNIT_SECONDS: dict[str, int] = {"s": 1, "min": 60}


_EXACT_FLOAT_INT = 2 ** 53

_MAX_REPORTED = 5


def _examples(values: Iterable) -> str:
    shown = [repr(v) for v in list(values)[:_MAX_REPORTED]]
    return ", ".join(shown)


def _as_series(values) -> pd.Series:
    if isinstance(values, pd.Series):
        return values
    if isinstance(values, pd.Index):
        return pd.Series(values)
    return pd.Series(list(values) if not isinstance(values, np.ndarray) else values)


def _refuse_datetime64(series: pd.Series, column: str) -> None:
    if pd.api.types.is_datetime64_any_dtype(series):
        raise TypeError(
            f"{column}: datetime64 input (dtype {series.dtype}) not accepted; pass the source text. "
            f"Prohibited: {', '.join(name for name, _ in PROHIBITED_IDIOMS)}")


def absent_mask(values) -> pd.Series:
    series = _as_series(values)
    missing = series.isna()
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return missing.astype(bool)
    blank = series.astype("string").str.strip().eq("").fillna(False)
    return (missing | blank).astype(bool)


def n_non_null_source(values) -> int:
    return int((~absent_mask(values)).sum())


@dataclass(frozen=True)
class ConversionRecord:

    column: str
    time_unit: str
    time_origin: str

    def to_dict(self) -> dict:
        return {"column": self.column, "time_unit": self.time_unit,
                "time_origin": self.time_origin}


class ConversionLog:

    def __init__(self) -> None:
        self._records: dict[str, ConversionRecord] = {}

    def record(self, record: ConversionRecord) -> None:
        previous = self._records.get(record.column)
        if previous is not None and previous != record:
            raise ValueError(
                f"{record.column}: already converted with time_unit={previous.time_unit!r}, "
                f"time_origin={previous.time_origin!r}; now {record.time_unit!r}, {record.time_origin!r}")
        self._records[record.column] = record

    def get(self, column: str) -> ConversionRecord | None:
        return self._records.get(column)

    def __contains__(self, column: object) -> bool:
        return column in self._records

    def __len__(self) -> int:
        return len(self._records)

    def to_dict(self) -> dict[str, dict]:
        return {k: v.to_dict() for k, v in sorted(self._records.items())}


def _log(log: ConversionLog | None, column: str, origin: str) -> None:
    if log is not None:
        log.record(ConversionRecord(column=column, time_unit=TIME_UNIT,
                                    time_origin=origin))


def _parse_iso(series: pd.Series, column: str, present: pd.Series) -> np.ndarray:
    text = series.astype("string")
    parts = text.str.extract(_ISO)

    unparseable = present & parts["date"].isna()
    if bool(unparseable.any()):
        raise ValueError(
            f"{column}: {int(unparseable.sum())} value(s) are not ISO-8601 date[ Thh:mm[:ss]]: "
            f"{_examples(series[unparseable])}")

    fractional = present & parts["frac"].notna()
    if bool(fractional.any()):
        raise ValueError(
            f"{column}: {int(fractional.sum())} value(s) have fractional seconds: "
            f"{_examples(series[fractional])}; time_unit is {TIME_UNIT!r}")

    joined = (parts["date"] + "T" + parts["hh"].fillna("00")
              + ":" + parts["mm"].fillna("00")
              + ":" + parts["ss"].fillna("00"))
    literals = [x if isinstance(x, str) else None for x in joined.tolist()]
    try:


        stamps = np.array(literals, dtype="datetime64[s]")
    except ValueError as exc:
        raise ValueError(
            f"{column}: invalid calendar date ({exc})") from exc
    return stamps.astype("int64")


def to_epoch_seconds(values, *, column: str,
                     log: ConversionLog | None = None) -> pd.Series:
    series = _as_series(values)
    _refuse_datetime64(series, column)
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raise TypeError(
            f"{column}: {series.dtype} input not accepted; pass source text "
            f"(.astype('string') for an all-empty column) or use offset_to_seconds")
    absent = absent_mask(series)
    if bool(absent.any()):
        raise ValueError(
            f"{column}: {int(absent.sum())} value(s) missing in a total column; "
            f"drop them or use to_epoch_seconds_masked for episodes.end")
    present = pd.Series(True, index=series.index)
    seconds = _parse_iso(series, column, present)
    _log(log, column, UNIX_EPOCH)
    return pd.Series(seconds, index=series.index, dtype="int64")


def to_epoch_seconds_masked(values, *, column: str,
                            log: ConversionLog | None = None) -> pd.Series:
    series = _as_series(values)
    _refuse_datetime64(series, column)
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raise TypeError(
            f"{column}: {series.dtype} input not accepted; cast an all-empty column with .astype('string')")
    absent = absent_mask(series)
    seconds = _parse_iso(series, column, ~absent)
    _log(log, column, UNIX_EPOCH)
    return _masked_series(seconds, absent, series.index)


def _masked_series(seconds: np.ndarray, absent: pd.Series,
                   index: pd.Index) -> pd.Series:
    mask = absent.to_numpy(dtype=bool)
    safe = np.where(mask, 0, seconds).astype("int64")
    return pd.Series(pd.arrays.IntegerArray(safe, mask.copy()), index=index)


def _scale_offsets(series: pd.Series, unit: str, column: str,
                   present: pd.Series) -> np.ndarray:
    if unit not in OFFSET_UNIT_SECONDS:
        raise ValueError(
            f"{column}: unknown offset unit {unit!r}; known units are {sorted(OFFSET_UNIT_SECONDS)}")
    _refuse_datetime64(series, column)
    if not (pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)):
        raise TypeError(
            f"{column}: offsets must be numeric, got {series.dtype}; use to_epoch_seconds for ISO text")
    factor = OFFSET_UNIT_SECONDS[unit]
    keep = present.to_numpy(dtype=bool)


    raw = series.astype("float64").to_numpy()
    raw = np.where(keep, raw, 0.0)
    scaled = raw * factor
    too_big = keep & (np.abs(scaled) >= _EXACT_FLOAT_INT)
    if bool(too_big.any()):
        raise ValueError(
            f"{column}: {int(too_big.sum())} offset(s) exceed 2**53 seconds and cannot be converted exactly")
    fractional = keep & (scaled != np.rint(scaled))
    if bool(fractional.any()):
        raise ValueError(
            f"{column}: {int(fractional.sum())} offset(s) are not whole seconds in unit {unit!r}: "
            f"{_examples(series[fractional])}")
    return np.rint(np.where(keep, scaled, 0.0)).astype("int64")


def _check_origin(origin: str, column: str) -> str:
    if origin is None or not str(origin).strip():
        raise ValueError(
            f"{column}: offsets need an explicit origin (eICU: 'unit_admission'; HiRID: 'icu_admission')")
    return origin


def offset_to_seconds(values, *, unit: str, column: str, origin: str,
                      log: ConversionLog | None = None) -> pd.Series:
    series = _as_series(values)
    _check_origin(origin, column)
    absent = absent_mask(series)
    if bool(absent.any()):
        raise ValueError(
            f"{column}: {int(absent.sum())} offset(s) missing in a total column; "
            f"drop them or use offset_to_seconds_masked")
    present = pd.Series(True, index=series.index)
    seconds = _scale_offsets(series, unit, column, present)
    _log(log, column, origin)
    return pd.Series(seconds, index=series.index, dtype="int64")


def offset_to_seconds_masked(values, *, unit: str, column: str, origin: str,
                             log: ConversionLog | None = None) -> pd.Series:
    series = _as_series(values)
    _check_origin(origin, column)
    absent = absent_mask(series)
    seconds = _scale_offsets(series, unit, column, ~absent)
    _log(log, column, origin)
    return _masked_series(seconds, absent, series.index)


@dataclass(frozen=True)
class IdiomViolation:

    path: str
    lineno: int
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: {self.kind}: {self.detail}"


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _scan_module(source: str, path: str) -> list[IdiomViolation]:
    out: list[IdiomViolation] = []
    tree = ast.parse(source, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):


            if _NS_DTYPE.fullmatch(node.value):
                out.append(IdiomViolation(
                    path, node.lineno, "datetime64[ns] dtype",
                    f"{node.value!r} wraps out-of-range int64 values; the IR uses int64 seconds"))
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        kwargs = {kw.arg for kw in node.keywords if kw.arg}
        if name == "read_csv" and "parse_dates" in kwargs:
            out.append(IdiomViolation(
                path, node.lineno, "read_csv(parse_dates=)",
                "degrades the column to object out of range; parse through "
                "survbench.ir.time instead"))
        if name == "to_datetime":
            coerce = any(
                kw.arg == "errors" and isinstance(kw.value, ast.Constant)
                and kw.value.value == "coerce" for kw in node.keywords)
            if coerce:
                out.append(IdiomViolation(
                    path, node.lineno, "to_datetime(errors='coerce')",
                    "returns NaT for out-of-range values without warning"))
            else:
                out.append(IdiomViolation(
                    path, node.lineno, "to_datetime",
                    "produces datetime64[ns], which the IR does not allow"))
    return out


def scan_for_prohibited_idioms(
    roots: str | Path | Sequence[str | Path],
) -> list[IdiomViolation]:
    if isinstance(roots, (str, Path)):
        roots = [roots]
    out: list[IdiomViolation] = []
    for root in roots:
        base = Path(root)
        files = sorted(base.rglob("*.py")) if base.is_dir() else [base]
        for file in files:
            out.extend(_scan_module(file.read_text(), str(file)))
    return out
