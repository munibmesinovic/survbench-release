from __future__ import annotations

import pandas as pd

from survbench.ir.schema import (
    REQUIRED_COLUMNS,
    TABLES,
    VALID_STATES,
    time_dtype_violations,
)


def validate_ir(tables: dict[str, pd.DataFrame]) -> None:
    for name in TABLES:
        if name not in tables:
            raise ValueError(f"missing IR table: {name!r}")

    for name in TABLES:
        missing = REQUIRED_COLUMNS[name] - set(tables[name].columns)
        if missing:
            raise ValueError(
                f"IR table {name!r} missing column(s): {sorted(missing)}"
            )

    bad_dtypes = time_dtype_violations(tables)
    if bad_dtypes:
        detail = "; ".join(f"{c}: declared {d!r}, got {a!r}"
                           for c, d, a in bad_dtypes)
        raise ValueError(
            f"IR time column dtype differs from the declared one: {detail}"
        )


    ep = tables["episodes"]
    unknown = set(ep["state"].unique()) - VALID_STATES
    if unknown:
        raise ValueError(f"unknown episode state(s): {sorted(unknown)}")


    precedes = (ep["end"] < ep["start"]).fillna(False)
    if bool(precedes.any()):
        raise ValueError(
            f"episode end precedes start in {int(precedes.sum())} row(s)")

    subj = tables["subjects"]
    if subj["patient_id"].duplicated().any():
        n = int(subj["patient_id"].duplicated().sum())
        raise ValueError(f"duplicate patient_id in subjects: {n} row(s)")

    known = set(subj["patient_id"])
    for name in ("episodes", "measurements", "documents"):
        orphans = set(tables[name]["patient_id"]) - known
        if orphans:
            raise ValueError(
                f"{name}: patient_id not in subjects: {sorted(orphans)[:5]}"
            )


__all__ = ["validate_ir"]
