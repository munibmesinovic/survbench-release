
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class TableSchema:
    dataset: str
    version: str
    table: str
    required_columns: Dict[str, str]
    optional_columns: Dict[str, str] = field(default_factory=dict)


class SchemaValidationError(Exception):
    pass


DEFAULT_VERSIONS: Dict[str, str] = {
    "mimiciv": "3.1",
    "eicu": "2.0",
    "mcmed": "1.0.1",
    "hirid": "1.1.1",
}


SCHEMAS: Dict[Tuple[str, str, str], TableSchema] = {

    ("mimiciv", "3.1", "icustays"): TableSchema(
        dataset="mimiciv", version="3.1", table="icustays",
        required_columns={"subject_id": "int64", "hadm_id": "int64",
                          "stay_id": "int64", "intime": "datetime64[ns]",
                          "outtime": "datetime64[ns]"},
    ),
    ("mimiciv", "3.1", "patients"): TableSchema(
        dataset="mimiciv", version="3.1", table="patients",
        required_columns={"subject_id": "int64", "anchor_age": "int64",
                          "gender": "object"},
    ),
    ("mimiciv", "3.1", "admissions"): TableSchema(
        dataset="mimiciv", version="3.1", table="admissions",
        required_columns={"hadm_id": "int64", "subject_id": "int64",
                          "admittime": "datetime64[ns]",
                          "dischtime": "datetime64[ns]"},
        optional_columns={"deathtime": "datetime64[ns]",
                          "hospital_expire_flag": "int64"},
    ),
    ("mimiciv", "3.1", "chartevents"): TableSchema(
        dataset="mimiciv", version="3.1", table="chartevents",
        required_columns={"hadm_id": "int64", "charttime": "datetime64[ns]",
                          "itemid": "int64", "valuenum": "float64",
                          "valueuom": "object"},
        optional_columns={"stay_id": "int64"},
    ),
    ("mimiciv", "3.1", "labevents"): TableSchema(
        dataset="mimiciv", version="3.1", table="labevents",
        required_columns={"hadm_id": "int64", "charttime": "datetime64[ns]",
                          "itemid": "int64", "valuenum": "float64",
                          "valueuom": "object"},
    ),
    ("mimiciv", "3.1", "diagnoses_icd"): TableSchema(
        dataset="mimiciv", version="3.1", table="diagnoses_icd",
        required_columns={"subject_id": "int64", "hadm_id": "int64",
                          "icd_code": "object"},
    ),
    ("mimiciv", "3.1", "radiology"): TableSchema(
        dataset="mimiciv", version="3.1", table="radiology",
        required_columns={"hadm_id": "int64", "charttime": "datetime64[ns]",
                          "text": "object"},
    ),


    ("eicu", "2.0", "patient"): TableSchema(
        dataset="eicu", version="2.0", table="patient",
        required_columns={"patientunitstayid": "int64",
                          "patienthealthsystemstayid": "int64",


                          "uniquepid": "object",
                          "age": "object",
                          "gender": "object",
                          "unitdischargestatus": "object",
                          "unitdischargeoffset": "int64",


                          "hospitaladmitoffset": "int64",
                          "hospitaldischargeoffset": "int64",
                          "hospitaldischargestatus": "object"},
    ),
    ("eicu", "2.0", "lab"): TableSchema(
        dataset="eicu", version="2.0", table="lab",
        required_columns={"patientunitstayid": "int64",
                          "labresultoffset": "int64",
                          "labname": "object",
                          "labresult": "float64"},
        optional_columns={"labmeasurenamesystem": "object"},
    ),
    ("eicu", "2.0", "vitalPeriodic"): TableSchema(
        dataset="eicu", version="2.0", table="vitalPeriodic",
        required_columns={"patientunitstayid": "int64",
                          "observationoffset": "int64"},
    ),
    ("eicu", "2.0", "vitalAperiodic"): TableSchema(
        dataset="eicu", version="2.0", table="vitalAperiodic",
        required_columns={"patientunitstayid": "int64",
                          "observationoffset": "int64"},
    ),


    ("mcmed", "1.0.1", "visits"): TableSchema(
        dataset="mcmed", version="1.0.1", table="visits",
        required_columns={"CSN": "int64", "MRN": "int64",
                          "Arrival_time": "object", "ED_dispo": "object",


                          "Dispo_time": "object"},
        optional_columns={"ED_LOS": "float64"},
    ),
    ("mcmed", "1.0.1", "numerics"): TableSchema(
        dataset="mcmed", version="1.0.1", table="numerics",
        required_columns={"CSN": "int64", "Time": "object",
                          "Measure": "object", "Value": "float64"},
    ),
    ("mcmed", "1.0.1", "labs"): TableSchema(
        dataset="mcmed", version="1.0.1", table="labs",
        required_columns={"CSN": "int64", "Result_time": "object",
                          "Component_name": "object", "Component_value": "float64"},
    ),
    ("mcmed", "1.0.1", "pmh"): TableSchema(
        dataset="mcmed", version="1.0.1", table="pmh",
        required_columns={"MRN": "int64", "Code": "object",
                          "Noted_date": "object"},
    ),
    ("mcmed", "1.0.1", "rads"): TableSchema(
        dataset="mcmed", version="1.0.1", table="rads",


        required_columns={"CSN": "int64", "Study": "object",
                          "Impression": "object", "Result_time": "object"},
    ),


    ("hirid", "1.1.1", "general_table"): TableSchema(
        dataset="hirid", version="1.1.1", table="general_table",
        required_columns={"patientid": "int64", "age": "float64",
                          "sex": "object", "discharge_status": "object"},
    ),
    ("hirid", "1.1.1", "hirid_patient_labels"): TableSchema(
        dataset="hirid", version="1.1.1", table="hirid_patient_labels",
        required_columns={"patientid": "int64"},
    ),
    ("hirid", "1.1.1", "general_table_extended"): TableSchema(
        dataset="hirid", version="1.1.1", table="general_table_extended",
        required_columns={"patientid": "int64"},
        optional_columns={"height": "float64",
                          "APACHE II Group": "object",
                          "APACHE IV Group": "object"},
    ),
    ("hirid", "1.1.1", "imputed_stage"): TableSchema(
        dataset="hirid", version="1.1.1", table="imputed_stage",
        required_columns={"patientid": "int64", "reldatetime": "float64"},
    ),
}


HIRID_RANGE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "HR": (20.0, 250.0),
    "ABP_sys": (30.0, 250.0),
    "ABP_dia": (10.0, 200.0),
    "MAP": (20.0, 230.0),
    "SpO2": (50.0, 100.0),
    "PeakPressure": (0.0, 80.0),
    "lactate_arterial": (0.0, 30.0),
    "lactate_venous": (0.0, 30.0),
    "INR": (0.5, 20.0),
    "glucose": (1.0, 50.0),
    "CRP": (0.0, 600.0),


}


def validate_schema(
    df: pd.DataFrame,
    dataset: str,
    table: str,
    version: Optional[str] = None,
) -> None:
    version = version or DEFAULT_VERSIONS.get(dataset)
    if version is None:
        raise SchemaValidationError(
            f"Unknown dataset {dataset!r}; no default version registered."
        )
    key = (dataset, version, table)
    schema = SCHEMAS.get(key)
    if schema is None:
        raise SchemaValidationError(
            f"No schema registered for dataset={dataset!r}, "
            f"version={version!r}, table={table!r}. "
            f"Add an entry in data/schema.py:SCHEMAS."
        )
    missing = [col for col in schema.required_columns if col not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"{dataset} {version} {table}: missing required columns {missing}. "
            f"Got columns: {list(df.columns)[:20]}"
            + ("..." if len(df.columns) > 20 else "")
        )


def clip_to_ranges(
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    counts: Dict[str, int] = {}
    for col, (lo, hi) in bounds.items():
        if col not in out.columns:
            continue
        s = out[col]
        n_clipped = int(((s < lo) | (s > hi)).sum())
        if n_clipped:
            out[col] = s.clip(lower=lo, upper=hi)
            counts[col] = n_clipped
    return out, counts
