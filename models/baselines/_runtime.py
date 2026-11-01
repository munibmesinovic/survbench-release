from __future__ import annotations

import hashlib
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


def git_commit(repo_root: "Path | str" = ".") -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root), stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def config_hash(config: dict) -> str:
    blob = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def param_count(model: Any) -> int:
    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            return int(sum(p.numel() for p in model.parameters()))
    except ImportError:
        pass
    return 0


def reject_unexpected_kwargs(ignored: dict) -> None:
    if ignored:
        raise TypeError(
            f"run() got unexpected keyword argument(s): {sorted(ignored)}")


def layer_widths(module) -> list:
    import torch.nn as nn
    return [int(m.out_features) for m in module.modules()
            if isinstance(m, nn.Linear)]


def dropout_rate(module) -> float:
    import torch.nn as nn
    rates = sorted({float(m.p) for m in module.modules()
                    if isinstance(m, nn.Dropout)})
    if len(rates) > 1:
        raise ValueError(
            f"module has nn.Dropout layers at different rates {rates}; cannot report one effective dropout")
    return rates[0] if rates else 0.0


def selection_losses(flat_metrics: dict, best_val_loss: float,
                     best_val_nll: float) -> dict:
    return {**flat_metrics,
            "loss": float(best_val_loss),
            "nll": float(best_val_nll)}


RISK_AXIS_KEY = "risk_axis_cause_codes"


def _validated_axis(cause_codes, artifact: str) -> list[int]:
    codes = [int(c) for c in np.asarray(list(cause_codes)).ravel()]
    if not codes:
        raise ValueError(
            f"{artifact}: `{RISK_AXIS_KEY}` is empty")
    if codes[0] <= 0:
        raise ValueError(
            f"{artifact}: `{RISK_AXIS_KEY}` is {codes}; cause codes must be positive")
    if sorted(set(codes)) != codes:
        raise ValueError(
            f"{artifact}: `{RISK_AXIS_KEY}` is {codes}; codes must be ascending and distinct")
    return codes


def risk_axis_of(modality_info: dict) -> list[int]:
    from models.baselines._data import cause_codes_of
    return _validated_axis(cause_codes_of(modality_info), "modality_info")


def read_risk_axis(artifact_contents, *, artifact: str) -> list[int]:
    if RISK_AXIS_KEY not in artifact_contents:
        raise ValueError(
            f"{artifact} has no `{RISK_AXIS_KEY}`; re-run the model to regenerate the npz")
    return _validated_axis(artifact_contents[RISK_AXIS_KEY], artifact)


def save_predictions(
    path: "Path | str",
    *,
    cuts: np.ndarray,
    cause_codes,
    surv_val: Optional[np.ndarray] = None,
    surv_test: Optional[np.ndarray] = None,
    cif_val: Optional[np.ndarray] = None,
    cif_test: Optional[np.ndarray] = None,
    pids_val: Optional[np.ndarray] = None,
    pids_test: Optional[np.ndarray] = None,
    cuts_val: Optional[np.ndarray] = None,
    cuts_test: Optional[np.ndarray] = None,
) -> None:
    payload: dict[str, np.ndarray] = {
        "cuts": np.asarray(cuts),
        RISK_AXIS_KEY: np.asarray(_validated_axis(cause_codes, str(path)),
                                  dtype=np.int64),
    }
    for name, arr in (
        ("surv_val", surv_val),
        ("surv_test", surv_test),
        ("cif_val", cif_val),
        ("cif_test", cif_test),
        ("pids_val", pids_val),
        ("pids_test", pids_test),
        ("cuts_val", cuts_val),
        ("cuts_test", cuts_test),
    ):
        if arr is not None:
            payload[name] = np.asarray(arr)
    np.savez(path, **payload)


def save_checkpoint(path: "Path | str", obj: Any) -> None:
    p = Path(path)
    if p.suffix == ".pt":
        import torch
        if hasattr(obj, "state_dict"):
            torch.save(obj.state_dict(), p)
        else:
            torch.save(obj, p)
    else:

        with open(p, "wb") as f:
            pickle.dump(obj, f)


class WallClock:

    def __enter__(self) -> "WallClock":
        self._t0 = time.time()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.seconds = time.time() - self._t0


def _model_modules():
    from models.baselines import cox, deephit, dynamic_deephit, dysurv, transformer
    from models.baselines import survlatent_ode, survtrace
    return {
        "cox": cox,
        "deephit": deephit,
        "dynamic_deephit": dynamic_deephit,
        "dysurv": dysurv,
        "transformer": transformer,
        "survtrace": survtrace,
        "survlatent_ode": survlatent_ode,
    }


MULTIMODAL_SUFFIX = "-mm"
SUMMARY_SUFFIX = "-summary"
SUMMARY_MODELS = ("cox", "deephit", "survtrace")


def __getattr__(name):
    if name == "_MODULES":
        value = _model_modules()
    elif name == "REGISTRY":
        value = {n: m.run for n, m in _model_modules().items()}
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


def registry_key(model: str) -> str:
    if is_multimodal(model):
        return model[: -len(MULTIMODAL_SUFFIX)]
    if is_summary(model):
        return model[: -len(SUMMARY_SUFFIX)]
    return model


def is_multimodal(model: str) -> bool:
    return model.endswith(MULTIMODAL_SUFFIX)


def is_summary(model: str) -> bool:
    return model.endswith(SUMMARY_SUFFIX)


def input_mode_of(model: str) -> Optional[str]:
    base = registry_key(model)
    if is_summary(model):
        if base not in SUMMARY_MODELS:
            raise ValueError(
                f"{model}: the summary input is defined for "
                f"{', '.join(SUMMARY_MODELS)} only; {base} reads the time "
                f"series itself.")
        return "summary"
    return "static" if base in SUMMARY_MODELS else None


def file_model_name(model: str, multimodal: bool) -> str:
    if model == "transformer" and multimodal:
        return "transformer" + MULTIMODAL_SUFFIX
    return model


def invoke(model: str, config: dict, **kwargs) -> dict:
    return getattr(_model_modules()[registry_key(model)], "run")(config, **kwargs)


def _set_seed(seed: int) -> None:
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


ROOT = Path(__file__).resolve().parents[2]

IN_DOMAIN_ARMS = ("sr", "cr")
LANDMARK_ARMS = ("cr_lmk",)
TRANSFER_ARMS = ("transfer",)
ARMS = IN_DOMAIN_ARMS + LANDMARK_ARMS + TRANSFER_ARMS

DATASETS = ("mimiciv", "eicu", "hirid", "mcmed")
TRANSFER_DATASETS = ("eicu_to_mimiciv", "eicu_to_mimiciv_target")
ALL_DATASETS = DATASETS + TRANSFER_DATASETS

SEEDS = (1, 2, 13, 42, 2024)

CONFIGS = {
    ("sr", "mimiciv"): "configs/mimiciv_adm_v1clock.yaml",
    ("sr", "eicu"): "configs/eicu_adm_v1clock.yaml",
    ("sr", "hirid"): "configs/hirid_adm.yaml",
    ("sr", "mcmed"): "configs/mcmed_adm.yaml",
    ("cr", "mimiciv"): "configs/mimiciv_adm.yaml",
    ("cr", "eicu"): "configs/eicu_adm.yaml",
    ("cr", "hirid"): "configs/hirid_adm_cr.yaml",
    ("cr", "mcmed"): "configs/mcmed_adm_cr.yaml",
    ("cr_lmk", "mcmed"): "configs/mcmed_lmk_cr.yaml",
    ("cr_lmk", "mimiciv"): "configs/mimiciv_lmk_cr.yaml",
    ("cr_lmk", "eicu"): "configs/eicu_lmk_cr.yaml",
    ("cr_lmk", "hirid"): "configs/hirid_lmk_cr.yaml",
    ("transfer", "eicu_to_mimiciv"): "configs/eicu_to_mimiciv.yaml",
    ("transfer", "eicu_to_mimiciv_target"): "configs/eicu_to_mimiciv_target.yaml",
}

_ECHO_ALL = ("num_bins", "num_risks")

_ECHO_ONLY = {
    "cox": ("l1_ratio", "n_features"),
    "deephit": (),
    "dynamic_deephit": (),
    "dysurv": ("alpha_ae", "alpha_kl", "alpha_surv", "dropout", "pos_weight"),
    "transformer": ("input_width",),
    "survtrace": ("max_grad_norm", "n_features", "n_intervals", "optimizer"),
    "survlatent_ode": ("decoder_atol", "decoder_ode_method", "decoder_rtol",
                       "encoder_ode_method", "event_weights",
                       "lr_decay_per_iteration", "lr_floor", "n_features",
                       "obsrv_std", "optimizer", "time_steps",
                       "wait_until_kl_inc"),
}


def config_path(arm: str, dataset: str) -> Path:
    return ROOT / CONFIGS[(arm, dataset)]


def load_hparams() -> dict:
    with open(ROOT / "configs" / "hparams.yaml") as fh:
        return yaml.safe_load(fh)


def cells(hparams: dict, arms, dataset=None, model=None):
    for key in sorted(hparams):
        arm, ds, mdl = key.split("/")
        if arm not in arms:
            continue
        if dataset and ds != dataset:
            continue
        if model and mdl != model:
            continue
        yield arm, ds, mdl


def checkpoint_path(arm: str, dataset: str, model: str, seed: int) -> Path:
    suffix = "pkl" if registry_key(model) == "cox" else "pt"
    return ROOT / "checkpoints" / arm / f"{model}_{dataset}_seed{seed}.{suffix}"


def runner_kwargs(model: str, effective_hparams: Optional[dict]) -> dict:
    import inspect
    base = registry_key(model)
    accepted = inspect.signature(
        getattr(_model_modules()[base], "run")).parameters
    echo = set(_ECHO_ALL) | set(_ECHO_ONLY[base])
    out: dict = {}
    unknown = []
    for key, value in (effective_hparams or {}).items():
        if key in accepted:
            out[key] = value
        elif key not in echo:
            unknown.append(key)
    if unknown:
        raise ValueError(
            f"configs/hparams.yaml records {sorted(unknown)} for {model}, which {base}.run() does not accept")
    return out
