from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from preprocessing.landmarking import landmarking_enabled


def patch_scipy_simps() -> None:
    import scipy.integrate
    if not hasattr(scipy.integrate, "simps"):
        scipy.integrate.simps = scipy.integrate.simpson


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


_ROOT = Path(__file__).resolve().parents[2]


def output_dir_of(config: dict) -> Path:
    out = Path(config["output"]["dir"])
    return out if out.is_absolute() else _ROOT / out


def prefix_of(config: dict) -> str:
    return config["output"]["prefix"]


def dataset_of(config: dict) -> str:
    return str(config["dataset"]["name"]).lower()


def load_feature_info(config: dict) -> dict:
    out = output_dir_of(config)
    prefix = prefix_of(config)
    candidates = [
        out / f"feature_names_{prefix}.pkl",
        out / "feature_names.pkl",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"No feature_names pickle found in {out}")


def load_modality_info(config: dict) -> dict:
    out = output_dir_of(config)
    prefix = prefix_of(config)
    candidates = [
        out / f"modality_info_{prefix}.pkl",
        out / "modality_info.pkl",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"No modality_info pickle found in {out}")


def landmarks_for_split(config: dict, split: str, prefix: str):
    enabled = landmarking_enabled(config)
    path = output_dir_of(config) / f"landmarks_{split}_{prefix}.npy"
    on_disk = path.exists()
    name = config.get("dataset", {}).get("name", "?")
    if enabled and not on_disk:
        raise FileNotFoundError(
            f"{name}: preprocessing.landmarking.enabled is true but {path.name} is not in {path.parent}")
    if on_disk and not enabled:
        raise ValueError(
            f"{name}: {path.name} found in {path.parent} but preprocessing.landmarking.enabled is false")
    if not enabled:
        return None
    return np.load(path, allow_pickle=True)


def assert_arrays_are_not_landmarked(config: dict, prefix: str) -> None:
    for split in ("val", "test"):
        landmarks_for_split(config, split, prefix)


def censor_beyond_grid(durations, events, cuts, *, landmarked: bool):
    durations = np.asarray(durations)
    events = np.asarray(events)
    if not landmarked:
        return durations, events
    top = float(cuts[-1])
    beyond = durations > top
    if not beyond.any():
        return durations, events
    durations, events = durations.copy(), events.copy()
    durations[beyond] = top
    events[beyond] = 0
    return durations, events


def static_input_kwargs(input_mode: str) -> dict:
    if input_mode == "static":
        return {"static_only": True}
    if input_mode == "summary":
        return {"summary": True}
    raise ValueError(
        f"input_mode = {input_mode!r}; expected 'static' or 'summary'.")


def load_split(
    config: dict,
    split: str,
    *,
    static_only: bool = False,
    mean_pool_time: bool = False,
    flatten: bool = False,
    summary: bool = False,
) -> dict:


    discharge_as, censor_causes = arm_of(config)

    out = output_dir_of(config)
    prefix = prefix_of(config)
    feature_info = load_feature_info(config)
    modality_info = load_modality_info(config)

    x = np.load(out / f"x_{split}_{prefix}.npy").astype(np.float32)

    if split in ("train", "val"):
        with open(out / f"y_{split}_surv_{prefix}.p", "rb") as f:
            durations_raw, events = pickle.load(f)
    else:

        durations_raw = np.load(out / f"durations_test_{prefix}.npy")
        events = np.load(out / f"events_test_{prefix}.npy")

    cuts = np.load(out / f"cuts_{prefix}.npy")

    landmarks = landmarks_for_split(config, split, prefix)
    if landmarks is not None and len(landmarks) != len(x):


        raise ValueError(
            f"landmarks_{split}_{prefix}.npy has {len(landmarks)} rows but x_{split}_{prefix}.npy has {len(x)}")
    subject_ids = np.load(out / f"subject_ids_{split}_{prefix}.npy",
                          allow_pickle=True)

    dyn_idx = feature_info["dynamic_indices"]
    stat_idx = feature_info["static_indices"]


    m_out: Optional[np.ndarray] = None

    if summary:
        mask_path = out / f"x_{split}_{prefix}_mask.npy"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"{mask_path.name} missing from {out}; the summary input needs the observation mask")
        obs = np.load(mask_path, mmap_mode="r")[:, :, dyn_idx]
        dyn = x[:, :, dyn_idx]
        x_out = np.concatenate([
            dyn[:, -1, :],
            dyn.mean(axis=1),
            np.asarray(obs, dtype=np.float32).mean(axis=1),
            x[:, :, stat_idx].mean(axis=1),
        ], axis=1).astype(np.float32)
    elif static_only:

        x_out = x[:, :, stat_idx].mean(axis=1).astype(np.float32)
    else:


        keep = list(dyn_idx) + list(stat_idx)
        x_out = x[:, :, keep].astype(np.float32)
        if mean_pool_time:
            x_out = x_out.mean(axis=1).astype(np.float32)
        elif flatten:
            n, t, f = x_out.shape
            x_out = x_out.reshape(n, t * f).astype(np.float32)
        else:


            mask_path = out / f"x_{split}_{prefix}_mask.npy"
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"{mask_path.name} missing from {out}; rebuild the corpus to get the observation mask")
            m_out = np.load(mask_path).astype(np.float32)[:, :, keep]
            if m_out.shape != x_out.shape:
                raise ValueError(
                    f"{mask_path.name} has shape {m_out.shape} after column selection but x_{split}_{prefix}.npy has {x_out.shape}")

    durations = np.asarray(durations_raw).astype(np.float64)
    events = np.asarray(events).astype(np.int64)


    durations, events = censor_beyond_grid(
        durations, events, cuts, landmarked=landmarking_enabled(config))


    dur_idx = np.searchsorted(cuts, durations, side="right") - 1
    dur_idx = np.clip(dur_idx, 0, len(cuts) - 1).astype(np.int64)


    events, modality_info = apply_discharge_as(
        events, modality_info,
        discharge_as=discharge_as, censor_causes=censor_causes)

    return {
        "x": x_out,
        "m": m_out,
        "durations": durations,
        "events": events,
        "dur_idx": dur_idx,
        "cuts": cuts,
        "landmarks": landmarks,
        "subject_ids": subject_ids,
        "feature_info": feature_info,
        "modality_info": modality_info,
        "discharge_as": discharge_as,
        "censored_causes": censor_causes,
    }


def remove_constant_cols(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    if x_train.ndim == 2:
        std = x_train.std(axis=0)
        keep = std > 1e-10
        return (x_train[:, keep], x_val[:, keep],
                None if x_test is None else x_test[:, keep], keep)
    elif x_train.ndim == 3:
        std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
        keep = std > 1e-10
        return (x_train[:, :, keep], x_val[:, :, keep],
                None if x_test is None else x_test[:, :, keep], keep)
    raise ValueError(f"Unexpected x_train.ndim={x_train.ndim}")


def cause_codes_of(modality_info: dict) -> list[int]:
    n_events = int(modality_info.get("n_events", 1) or 1)
    if not modality_info.get("competing_risks", False):
        return [1]

    declared = modality_info.get("cause_codes")
    if not declared:


        warnings.warn(
            f"modality_info has no cause_codes; assuming 1..{n_events}. Regenerate it to record them.",
            UserWarning, stacklevel=2)
        return list(range(1, n_events + 1))

    codes = sorted({int(c) for c in declared if int(c) != 0})
    if len(codes) != n_events:
        raise ValueError(
            f"modality_info declares n_events={n_events} but cause_codes={codes} ({len(codes)} causes)")
    return codes


def head_index_of(modality_info: dict) -> dict[int, int]:
    return {code: i for i, code in enumerate(cause_codes_of(modality_info))}


def cause_code_of_head(modality_info: dict) -> dict[int, int]:
    return {i: code for i, code in enumerate(cause_codes_of(modality_info))}


def head_index_lookup(modality_info: dict) -> np.ndarray:
    codes = cause_codes_of(modality_info)
    table = np.full(max(codes) + 1, -1, dtype=np.int64)
    for i, code in enumerate(codes):
        table[code] = i
    return table


from survbench.core.arms import arm_of


def apply_discharge_as(events, modality_info: dict, *,
                       discharge_as: str,
                       censor_causes) -> tuple[np.ndarray, dict]:
    if discharge_as not in ("censored", "competing"):
        raise ValueError(
            f"discharge_as={discharge_as!r}; expected 'censored' or 'competing'")

    events = np.asarray(events, dtype=np.int64).copy()
    if discharge_as == "competing":
        return events, modality_info

    declared = cause_codes_of(modality_info)
    censor = sorted({int(c) for c in censor_causes})
    if not censor:
        raise ValueError(
            f"discharge_as='censored' needs non-empty censor_causes; this dataset declares {declared}")

    unknown = [c for c in censor if c not in declared]
    if unknown:
        raise ValueError(
            f"censor_causes={censor} includes {unknown}, not in cause_codes={declared}")

    remaining = [c for c in declared if c not in censor]
    if not remaining:
        raise ValueError(
            f"censor_causes={censor} covers every declared cause {declared}; no cause left to predict")

    if len(remaining) == 1 and remaining[0] != 1:
        raise ValueError(
            f"censor_causes={censor} leaves cause {remaining[0]}; the single-risk path expects cause 1")

    events[np.isin(events, censor)] = 0

    info = dict(modality_info)
    info["cause_codes"] = remaining
    info["n_events"] = len(remaining)
    info["competing_risks"] = len(remaining) > 1
    info["discharge_as"] = "censored"
    info["censored_causes"] = censor
    return events, info
