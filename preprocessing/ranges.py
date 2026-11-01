from __future__ import annotations

import numpy as np


PHYSIOLOGICAL_RANGES = {
    "Heart rate":        (0.0, 300.0),
    "Respiratory rate":  (0.0, 120.0),
    "SpO2":              (0.0, 100.0),
    "Systolic BP":       (0.0, 300.0),
    "Diastolic BP":      (0.0, 200.0),
    "Mean BP":           (0.0, 250.0),
    "pH":                (6.5, 8.0),
    "Glucose":           (0.0, 2000.0),


    "Vent rate":         (0.0, 100.0),
    "PEEP":              (0.0, 50.0),
}


UNIT_BANDS = {
    "Temperature": [
        (25.0, 45.0, lambda x: x),
        (77.0, 113.0, lambda x: (x - 32.0) * 5.0 / 9.0),
    ],


    "FiO2": [
        (21.0, 100.0, lambda x: x),
        (0.21, 1.0, lambda x: x * 100.0),
    ],
}


CONCEPT_ALIASES = {
    "Heart rate":       {"HR", "Heart_Rate", "heart rate", "heartrate"},
    "Respiratory rate": {"RR", "Respiratory_Rate", "respiratory rate",
                         "respiration", "resprate"},
    "SpO2":             {"SpO2", "SaO2", "O2 Saturation", "o2sat"},
    "Systolic BP":      {"Systolic_BP", "systolic bp", "SBP", "sysbp",
                         "Arterial_BP_Systolic", "noninvasivesystolic",
                         "systemicsystolic"},
    "Diastolic BP":     {"Diastolic_BP", "diastolic bp", "DBP", "diasbp",
                         "Arterial_BP_Diastolic", "noninvasivediastolic",
                         "systemicdiastolic"},
    "Mean BP":          {"MAP", "Mean_BP", "mean bp", "meanbp",
                         "noninvasivemean", "systemicmean"},
    "pH":               {"pH", "ph", "arterial ph"},
    "Glucose":          {"Glucose", "bedside glucose", "glucose (serum)"},


    "Temperature":      {"Temperature", "Temperature_F", "Temperature_C",
                         "temp", "temperature (c)"},
    "FiO2":             {"FiO2", "fio2", "FIO2", "fraction inspired oxygen"},
    "Vent rate":        {"Vent Rate", "vent rate", "ventrate",
                         "respiratory rate (set)", "set rate"},
    "PEEP":             {"PEEP", "peep", "peep/cpap"},
}


def _normalise(name: str) -> str:
    return str(name).lower().replace("_", "").replace(" ", "")


def _build_lookup():
    lookup = {}
    for concept, aliases in CONCEPT_ALIASES.items():
        for alias in aliases:
            lookup[_normalise(alias)] = concept
    return lookup


def resolve_ranges(feature_names):
    lookup = _build_lookup()
    known = set(PHYSIOLOGICAL_RANGES) | set(UNIT_BANDS)
    resolved = {}
    for i, name in enumerate(feature_names):
        concept = lookup.get(_normalise(name))
        if concept is not None and concept in known:
            resolved[i] = concept
    return resolved


def _filter_values(column, concept):


    present = np.isfinite(column)
    n_converted = 0

    if concept in UNIT_BANDS:


        claimed = np.zeros_like(column, dtype=bool)
        for low, high, convert in UNIT_BANDS[concept]:
            in_band = present & (column >= low) & (column <= high) & ~claimed
            if in_band.any():
                original = column[in_band]
                column[in_band] = convert(original)

                n_converted += int(np.sum(convert(original) != original))
            claimed |= in_band
        bad = present & ~claimed
    else:
        low, high = PHYSIOLOGICAL_RANGES[concept]
        bad = present & ((column < low) | (column > high))

    n_bad = int(bad.sum())
    if n_bad:
        column[bad] = np.nan
    return n_bad, n_converted


def _empty_report():
    return {"filtered": {}, "converted": {}, "no_range": [],
            "n_filtered": 0, "n_converted": 0}


def _finish_report(filtered, converted, no_range):
    return {"filtered": filtered,
            "converted": converted,
            "no_range": no_range,
            "n_filtered": int(sum(filtered.values())),
            "n_converted": int(sum(converted.values()))}


def merge_range_reports(*reports):
    filtered, converted, no_range = {}, {}, []
    for report in reports:
        if not report:
            continue
        for concept, n in (report.get("filtered") or {}).items():
            filtered[concept] = filtered.get(concept, 0) + int(n)
        for concept, n in (report.get("converted") or {}).items():
            converted[concept] = converted.get(concept, 0) + int(n)
        for name in report.get("no_range") or []:
            if name not in no_range:
                no_range.append(name)
    return _finish_report(filtered, converted, no_range)


def apply_ranges_wide(frame, columns=None, names=None):
    import pandas as pd

    candidates = list(frame.columns) if columns is None else list(columns)
    labels = candidates if names is None else list(names)
    if len(labels) != len(candidates):
        raise ValueError(
            f"names has {len(labels)} entries for {len(candidates)} columns")
    resolved = resolve_ranges(labels)
    out = frame.copy()

    filtered, converted = {}, {}
    for position, concept in resolved.items():
        name = candidates[position]
        column = pd.to_numeric(out[name], errors="coerce").to_numpy(dtype=float)
        n_bad, n_conv = _filter_values(column, concept)
        out[name] = column
        filtered[concept] = filtered.get(concept, 0) + n_bad
        converted[concept] = converted.get(concept, 0) + n_conv

    no_range = sorted(label for i, label in enumerate(labels)
                      if i not in resolved)
    return out, _finish_report(filtered, converted, no_range)


def apply_ranges_long(frame, feature_col, value_col):
    out = frame.copy()
    names = out[feature_col].astype(str)
    unique_names = list(dict.fromkeys(names.tolist()))
    resolved = resolve_ranges(unique_names)
    concept_of = {unique_names[i]: concept for i, concept in resolved.items()}

    values = out[value_col].to_numpy(dtype=float, copy=True)
    name_array = names.to_numpy()

    filtered, converted = {}, {}
    for name, concept in concept_of.items():
        rows = name_array == name
        if not rows.any():
            continue


        subset = values[rows]
        n_bad, n_conv = _filter_values(subset, concept)
        values[rows] = subset
        filtered[concept] = filtered.get(concept, 0) + n_bad
        converted[concept] = converted.get(concept, 0) + n_conv

    out[value_col] = values
    no_range = sorted(n for n in unique_names if n not in concept_of)
    return out, _finish_report(filtered, converted, no_range)


def hourly_mean_with_ranges(frame, by, value_cols, names=None):
    value_cols = list(value_cols)
    filtered_frame, report = apply_ranges_wide(
        frame, columns=value_cols, names=names)
    return filtered_frame.groupby(list(by))[value_cols].mean(), report


def apply_physiological_ranges(data, feature_names):
    out = np.array(data, dtype=float, copy=True)
    resolved = resolve_ranges(feature_names)

    filtered, converted = {}, {}
    for idx, concept in resolved.items():
        column = np.array(out[..., idx], copy=True)
        n_bad, n_converted = _filter_values(column, concept)
        out[..., idx] = column
        filtered[concept] = filtered.get(concept, 0) + n_bad
        converted[concept] = converted.get(concept, 0) + n_converted

    no_range = sorted(name for i, name in enumerate(feature_names)
                      if i not in resolved)
    return out, _finish_report(filtered, converted, no_range)


def scale_inflation_report(data, feature_names, *, mask=None,
                           threshold: float = 5.0):
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    observed = None
    if mask is not None:
        observed = np.asarray(mask).astype(bool)
        if observed.ndim == 3:
            observed = observed.reshape(-1, observed.shape[-1])

    flagged = {}
    for i, name in enumerate(feature_names):
        column = arr[:, i]
        if observed is not None:
            column = column[observed[:, i]]
        column = column[np.isfinite(column)]
        if column.size < 100:
            continue
        fitted = float(np.std(column))
        q25, q75 = np.percentile(column, [25, 75])
        robust = float((q75 - q25) / 1.349)
        if robust <= 0 or fitted <= 0:
            continue
        ratio = fitted / robust
        if ratio > threshold:
            flagged[name] = {"fitted": fitted, "robust": robust,
                             "ratio": ratio, "max": float(column.max())}
    return flagged


def constant_feature_report(data, feature_names, *, mask=None,
                            tolerance: float = 0.0):
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    observed = None
    if mask is not None:
        observed = np.asarray(mask).astype(bool)
        if observed.ndim == 3:
            observed = observed.reshape(-1, observed.shape[-1])

    flagged = {}
    for i, name in enumerate(feature_names):
        column = arr[:, i]
        if observed is not None:
            column = column[observed[:, i]]
        column = column[np.isfinite(column)]
        if column.size == 0:
            continue
        spread = float(column.max() - column.min())
        if spread <= tolerance:
            flagged[name] = {"value": float(column[0]), "n": int(column.size)}
    return flagged


def format_constant_report(flagged) -> str:
    if not flagged:
        return "  Constant-feature check: every dynamic feature varies."
    lines = [f"  Constant-feature check: {len(flagged)} feature(s) with a single observed value:"]
    for name, d in sorted(flagged.items()):
        lines.append(f"    {name:32s} = {d['value']:g} in all {d['n']:,} "
                     f"observed cells")
    return "\n".join(lines)


def format_inflation_report(flagged) -> str:
    if not flagged:
        return "  Scale check: no dynamic feature's fitted SD exceeds 5x its robust SD."
    lines = ["  Scale check: fitted SD vs robust SD (IQR/1.349):"]
    for name, d in sorted(flagged.items(), key=lambda kv: -kv[1]["ratio"]):
        lines.append(f"    {name:26s} {d['ratio']:8.0f}x  "
                     f"(fitted {d['fitted']:12.2f} vs robust {d['robust']:9.2f}, "
                     f"max {d['max']:.1f})")
    lines.append("    Heavy-tailed labs give large ratios; 100x or more suggests a unit or range problem.")
    return "\n".join(lines)


def format_range_report(report, *, total_cells=None) -> str:
    lines = ["  Physiological range filter:"]
    acted = {k: v for k, v in report["filtered"].items() if v}
    if acted:
        for concept, n in sorted(acted.items(), key=lambda kv: -kv[1]):
            if concept in UNIT_BANDS:
                bands = "/".join(f"[{lo:g},{hi:g}]" for lo, hi, _ in UNIT_BANDS[concept])
            else:
                low, high = PHYSIOLOGICAL_RANGES[concept]
                bands = f"[{low:g}, {high:g}]"
            frac = f" ({n / total_cells:.4%})" if total_cells else ""
            lines.append(f"    {concept:20s} {bands}  nulled {n:,}{frac}")
    else:
        lines.append("    nothing nulled")


    for concept, n in sorted(report.get("converted", {}).items(), key=lambda kv: -kv[1]):
        if n:
            lines.append(f"    {concept:20s} converted to canonical units: {n:,}")

    covered = sorted(report["filtered"])
    lines.append(f"    concepts with a range in this dataset: "
                 f"{', '.join(covered) if covered else 'NONE'}")
    lines.append(f"    features with no declared range: {len(report['no_range'])}")
    return "\n".join(lines)
