from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


SHARED_DYNAMIC_FEATURES: List[str] = [
    "heart_rate",
    "systolic_bp_invasive",
    "diastolic_bp_invasive",
    "systolic_bp_noninvasive",
    "diastolic_bp_noninvasive",
    "respiratory_rate",
    "spo2",
    "temperature",
    "glucose",
    "sodium",
    "potassium",
    "chloride",
    "bicarbonate",
    "bun",
    "creatinine",
    "calcium",
    "magnesium",
    "lactate",
    "anion_gap",
    "hemoglobin",
    "hematocrit",
    "platelets",
    "wbc",
]

EICU_DYNAMIC_ALIASES: Dict[str, List[str]] = {
    "heart_rate": ["heartrate"],
    "systolic_bp_invasive": ["systemicsystolic"],
    "diastolic_bp_invasive": ["systemicdiastolic"],
    "systolic_bp_noninvasive": ["noninvasivesystolic"],
    "diastolic_bp_noninvasive": ["noninvasivediastolic"],
    "respiratory_rate": ["respiration"],
    "spo2": ["sao2"],
    "temperature": ["temperature"],
    "glucose": ["glucose", "bedside glucose"],
    "sodium": ["sodium"],
    "potassium": ["potassium"],
    "chloride": ["chloride"],
    "bicarbonate": ["bicarbonate", "HCO3"],
    "bun": ["BUN"],
    "creatinine": ["creatinine"],
    "calcium": ["calcium"],
    "magnesium": ["magnesium"],
    "lactate": ["lactate"],
    "anion_gap": ["anion gap"],
    "hemoglobin": ["Hgb", "hemoglobin"],
    "hematocrit": ["Hct", "hematocrit"],
    "platelets": ["platelets x 1000", "platelet count x 1000"],
    "wbc": ["WBC x 1000", "WBC"],
}

MIMIC_DYNAMIC_ALIASES: Dict[str, List[str]] = {
    "heart_rate": ["Heart_Rate"],
    "systolic_bp_invasive": ["Arterial_BP_Systolic"],
    "diastolic_bp_invasive": ["Arterial_BP_Diastolic"],
    "systolic_bp_noninvasive": ["Systolic_BP"],
    "diastolic_bp_noninvasive": ["Diastolic_BP"],
    "respiratory_rate": ["Respiratory_Rate"],
    "spo2": ["SpO2"],
    "temperature": ["Temperature_F", "Temperature"],
    "glucose": ["Glucose"],
    "sodium": ["Sodium"],
    "potassium": ["Potassium"],
    "chloride": ["Chloride"],
    "bicarbonate": ["Bicarbonate"],
    "bun": ["BUN"],
    "creatinine": ["Creatinine"],
    "calcium": ["Calcium_Total", "Calcium"],
    "magnesium": ["Magnesium"],
    "lactate": ["Lactate"],
    "anion_gap": ["Anion_Gap"],
    "hemoglobin": ["Hemoglobin"],
    "hematocrit": ["Hematocrit"],
    "platelets": ["Platelet_Count"],
    "wbc": ["WBC"],
}


SHARED_STATIC_FEATURES: List[str] = ["age", "is_male"]

EICU_STATIC_ALIASES: Dict[str, List[List[str]]] = {


    "age": [["age", "passthrough"]],
    "is_male": [["gender_Male", "binary_indicator"]],
}

MIMIC_STATIC_ALIASES: Dict[str, List[List[str]]] = {
    "age": [["anchor_age", "passthrough"], ["age", "passthrough"]],
    "is_male": [["gender_M", "binary_indicator"]],
}


def _normalise(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


@dataclass
class FeatureMatch:

    canonical: str
    matched_alias: Optional[str]
    column_index: Optional[int]

    @property
    def found(self) -> bool:
        return self.column_index is not None


def match_dynamic(
    canonical: str, available_names: List[str], dataset: str
) -> FeatureMatch:
    if dataset == "eicu":
        aliases = EICU_DYNAMIC_ALIASES.get(canonical, [])
    elif dataset == "mimiciv":
        aliases = MIMIC_DYNAMIC_ALIASES.get(canonical, [])
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    norm_avail: Dict[str, int] = {}
    for i, n in enumerate(available_names):
        k = _normalise(n)
        if k not in norm_avail:
            norm_avail[k] = i
    for alias in aliases:
        key = _normalise(alias)
        if key in norm_avail:
            idx = norm_avail[key]
            return FeatureMatch(canonical, available_names[idx], idx)
    return FeatureMatch(canonical, None, None)


def match_static(
    canonical: str, available_names: List[str], dataset: str
) -> Tuple[FeatureMatch, Optional[str]]:
    if dataset == "eicu":
        recipes = EICU_STATIC_ALIASES.get(canonical, [])
    elif dataset == "mimiciv":
        recipes = MIMIC_STATIC_ALIASES.get(canonical, [])
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    norm_avail: Dict[str, int] = {}
    for i, n in enumerate(available_names):
        k = _normalise(n)
        if k not in norm_avail:
            norm_avail[k] = i
    for name, op in recipes:
        key = _normalise(name)
        if key in norm_avail:
            idx = norm_avail[key]
            return (FeatureMatch(canonical, available_names[idx], idx), op)
    return (FeatureMatch(canonical, None, None), None)


def resolve_dynamic_indices(
    feature_info: dict, dataset: str
) -> Tuple[List[int], List[int], List[str]]:
    dynamic_names = list(feature_info.get("dynamic_names", []))
    dynamic_indices = list(feature_info.get("dynamic_indices", []))
    if len(dynamic_names) != len(dynamic_indices):
        raise ValueError(
            "feature_info.dynamic_{names,indices} have mismatched lengths"
        )

    canonical_idx: List[int] = []
    tensor_col_idx: List[int] = []
    names: List[str] = []
    for i, canonical in enumerate(SHARED_DYNAMIC_FEATURES):
        m = match_dynamic(canonical, dynamic_names, dataset)
        if not m.found:
            continue
        canonical_idx.append(i)
        tensor_col_idx.append(dynamic_indices[m.column_index])
        names.append(canonical)
    return canonical_idx, tensor_col_idx, names


def resolve_static_indices(
    feature_info: dict, dataset: str
) -> Tuple[List[int], List[int], List[str], List[str]]:
    static_names = list(feature_info.get("static_names", []))
    static_indices = list(feature_info.get("static_indices", []))
    if len(static_names) != len(static_indices):
        raise ValueError(
            "feature_info.static_{names,indices} have mismatched lengths"
        )

    canonical_idx: List[int] = []
    tensor_col_idx: List[int] = []
    names: List[str] = []
    ops: List[str] = []
    for i, canonical in enumerate(SHARED_STATIC_FEATURES):
        m, op = match_static(canonical, static_names, dataset)
        if not m.found:
            continue
        canonical_idx.append(i)
        tensor_col_idx.append(static_indices[m.column_index])
        names.append(canonical)
        ops.append(op)
    return canonical_idx, tensor_col_idx, names, ops


def compute_shared_schema(
    source_feature_info: dict, target_feature_info: dict,
    source_dataset: str, target_dataset: str,
) -> dict:
    s_can, s_dyn, s_names = resolve_dynamic_indices(source_feature_info, source_dataset)
    t_can, t_dyn, t_names = resolve_dynamic_indices(target_feature_info, target_dataset)

    common_can_dyn = sorted(set(s_can) & set(t_can))
    s_can_to_col = dict(zip(s_can, s_dyn))
    t_can_to_col = dict(zip(t_can, t_dyn))
    source_dyn_cols = [s_can_to_col[c] for c in common_can_dyn]
    target_dyn_cols = [t_can_to_col[c] for c in common_can_dyn]
    dyn_names = [SHARED_DYNAMIC_FEATURES[c] for c in common_can_dyn]

    s_can_st, s_stc, s_names_st, s_ops = resolve_static_indices(
        source_feature_info, source_dataset
    )
    t_can_st, t_stc, t_names_st, t_ops = resolve_static_indices(
        target_feature_info, target_dataset
    )
    common_can_st = sorted(set(s_can_st) & set(t_can_st))
    s_st_map = dict(zip(s_can_st, zip(s_stc, s_ops)))
    t_st_map = dict(zip(t_can_st, zip(t_stc, t_ops)))
    src_static_cols = [s_st_map[c][0] for c in common_can_st]
    tgt_static_cols = [t_st_map[c][0] for c in common_can_st]
    src_static_ops = [s_st_map[c][1] for c in common_can_st]
    tgt_static_ops = [t_st_map[c][1] for c in common_can_st]
    static_names = [SHARED_STATIC_FEATURES[c] for c in common_can_st]

    return {
        "dynamic_names": dyn_names,
        "source_dynamic_cols": source_dyn_cols,
        "target_dynamic_cols": target_dyn_cols,
        "static_names": static_names,
        "source_static_cols": src_static_cols,
        "target_static_cols": tgt_static_cols,
        "source_static_ops": src_static_ops,
        "target_static_ops": tgt_static_ops,
    }
