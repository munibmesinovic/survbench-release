
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import pandas as pd


CANONICAL_UNITS: Dict[str, str] = {
    "temperature": "celsius",
    "weight": "kg",
    "height": "cm",
    "glucose": "mg/dL",
    "hemoglobin": "g/dL",
    "creatinine": "mg/dL",
}


_SYNONYMS: Dict[str, Dict[str, str]] = {
    "temperature": {
        "f": "°F", "deg f": "°F", "degf": "°F", "fahrenheit": "°F", "°f": "°F",
        "c": "°C", "deg c": "°C", "degc": "°C", "celsius": "°C", "°c": "°C",
    },
    "weight": {
        "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
        "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    },
    "height": {
        "in": "in", "inch": "in", "inches": "in", '"': "in",
        "cm": "cm", "centimeter": "cm", "centimeters": "cm",
    },
    "glucose": {
        "mg/dl": "mg/dL", "mg dl": "mg/dL",
        "mmol/l": "mmol/L", "mmol l": "mmol/L",
    },
    "hemoglobin": {
        "g/dl": "g/dL", "g dl": "g/dL",
        "g/l": "g/L", "g l": "g/L",
    },
    "creatinine": {
        "mg/dl": "mg/dL",
        "umol/l": "μmol/L", "μmol/l": "μmol/L", "micromol/l": "μmol/L",
    },
}


_CONVERTERS = {
    ("temperature", "°F", "celsius"): lambda x: (x - 32.0) * 5.0 / 9.0,
    ("temperature", "°C", "celsius"): lambda x: x,
    ("weight", "lb", "kg"): lambda x: x * 0.453592,
    ("weight", "kg", "kg"): lambda x: x,
    ("height", "in", "cm"): lambda x: x * 2.54,
    ("height", "cm", "cm"): lambda x: x,
    ("glucose", "mmol/L", "mg/dL"): lambda x: x * 18.0156,
    ("glucose", "mg/dL", "mg/dL"): lambda x: x,
    ("hemoglobin", "g/L", "g/dL"): lambda x: x / 10.0,
    ("hemoglobin", "g/dL", "g/dL"): lambda x: x,
    ("creatinine", "μmol/L", "mg/dL"): lambda x: x / 88.4,
    ("creatinine", "mg/dL", "mg/dL"): lambda x: x,
}


_IDENTITY_RESOLVED: Dict[str, str] = {
    "temperature": "°C",
    "weight": "kg",
    "height": "cm",
    "glucose": "mg/dL",
    "hemoglobin": "g/dL",
    "creatinine": "mg/dL",
}


def _resolve_unit(kind: str, unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    s = str(unit).strip().lower()
    return _SYNONYMS.get(kind, {}).get(s)


def to_canonical(kind: str, value: float, unit: Optional[str]) -> Tuple[float, Optional[str]]:
    if kind not in CANONICAL_UNITS:
        return value, unit
    canonical = CANONICAL_UNITS[kind]
    resolved = _resolve_unit(kind, unit)
    if resolved is None:
        return value, unit
    fn = _CONVERTERS.get((kind, resolved, canonical))
    if fn is None:
        return value, unit
    return float(fn(value)), canonical


def normalize_dataframe(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    unit_col: Optional[str] = None,
    feature_to_kind: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if unit_col is None or feature_to_kind is None:
        return df.copy(), {}

    out = df.copy()
    summary: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"converted_rows": 0,
                 "from_units": defaultdict(int),
                 "unrecognized": defaultdict(int)}
    )

    feat_arr = out[feature_col].astype(str).values
    val_arr = out[value_col].astype(float).values
    unit_arr = out[unit_col].astype(object).values

    new_vals = val_arr.copy()
    for i in range(len(out)):
        feat = feat_arr[i]
        kind = feature_to_kind.get(feat)
        if kind is None or kind not in CANONICAL_UNITS:
            continue

        if pd.isna(val_arr[i]):
            continue
        unit = unit_arr[i]
        canonical = CANONICAL_UNITS[kind]
        resolved = _resolve_unit(kind, unit)
        if resolved is None:
            if unit is not None and not pd.isna(unit):
                summary[kind]["unrecognized"][str(unit)] += 1
            continue
        if resolved == _IDENTITY_RESOLVED.get(kind):
            continue
        fn = _CONVERTERS.get((kind, resolved, canonical))
        if fn is None:
            continue
        new_vals[i] = float(fn(val_arr[i]))
        summary[kind]["converted_rows"] += 1
        summary[kind]["from_units"][resolved] += 1

    out[value_col] = new_vals

    summary_plain = {
        k: {"converted_rows": v["converted_rows"],
            "from_units": dict(v["from_units"]),
            "unrecognized": dict(v["unrecognized"])}
        for k, v in summary.items()
    }
    return out, summary_plain
