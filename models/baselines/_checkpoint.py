
from __future__ import annotations

import copy
import json
import pickle
import random
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np




def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_config_pointed_at_bundle(config_path: str,
                                   data_root: Optional[str]) -> dict:
    from models.baselines import _data

    config = copy.deepcopy(_data.load_config(config_path))
    if data_root is not None:
        out_dir = Path(config["output"]["dir"])


        config["output"]["dir"] = str(Path(data_root) / out_dir.name)
    return config


def _seed_from_tag(tag: str, default: int = 42) -> int:
    m = re.search(r"seed(\d+)", tag)
    return int(m.group(1)) if m else default


def _maybe_load_json(checkpoint: Path, json_path: Optional[str]) -> Optional[dict]:
    if json_path:
        p = Path(json_path)
    else:

        p = checkpoint.with_suffix(".json")
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _expected_param_count(shipped_json: Optional[dict]) -> Optional[int]:
    if not shipped_json:
        return None
    pc = shipped_json.get("training", {}).get("param_count")
    if pc:
        return int(pc)
    pc = shipped_json.get("model_info", {}).get("n_params")
    if pc:
        return int(pc)
    return None


def _assert_param_count(model, shipped_json: Optional[dict]) -> None:
    import torch.nn as nn
    if not isinstance(model, nn.Module):
        return
    rebuilt = int(sum(p.numel() for p in model.parameters()))
    expected = _expected_param_count(shipped_json)
    if expected is not None:
        assert rebuilt == expected, (
            f"rebuilt model has {rebuilt} params but the shipped JSON records {expected}"
        )
        print(f"  param_count OK: {rebuilt} == shipped {expected}")
    else:
        print(f"  param_count: {rebuilt} (no shipped JSON to compare against)")


def _arch_from_artifact(shipped_json: Optional[dict],
                        defaults: dict) -> dict:
    echo = (shipped_json or {}).get("effective_hparams") or {}
    arch = dict(defaults)
    for name, fallback in defaults.items():
        if name not in echo:
            continue
        value = echo[name]


        if isinstance(fallback, list) and isinstance(value, str):
            value = [int(part) for part in value.split(",") if part.strip()]
        arch[name] = value
    return arch


def _NAN_BLOCK() -> dict:
    return {"ctd": float("nan"), "ibs": float("nan"), "ibll": float("nan")}


def _horizon_of(config: dict) -> float:
    return float(config.get("preprocessing", {}).get("max_horizon_hours", 240))


def _num_risks_of(modality_info: dict) -> int:
    competing = bool(modality_info.get("competing_risks", False))
    return int(modality_info.get("n_events", 1)) if competing else 1


def _score_split(module, quantity, split, train, cuts, num_risks, horizon,
                 modality_info, dataset) -> dict:
    return module.evaluate_by_landmark(
        quantity, split["durations"], split["events"], cuts, num_risks,
        train_durations=train["durations"],
        train_events_full=train["events"],
        horizon=horizon,
        modality_info=modality_info,
        landmarks=split["landmarks"],
        dataset=dataset,
    )


def _predict_cox(config: dict, checkpoint: Path, seed: int,
                 shipped_json: Optional[dict]) -> dict:
    from models.baselines._data import (dataset_of, load_split,
                                        remove_constant_cols)
    from models.baselines._runtime import risk_axis_of
    from models.baselines.cox import (_CS_KEY, _cif_refusal_reason,
                                      _coxph_survival, _landmark_block,
                                      _mean_over_causes, aalen_johansen_cif,
                                      evaluate_survival)
    from models.baselines._eval import (_pool_landmark_means,
                                        compute_competing_risks_metrics,
                                        evaluation_times, landmark_key,
                                        landmark_strata, rebased_horizon)

    dataset = dataset_of(config)

    with open(checkpoint, "rb") as f:
        obj = pickle.load(f)

    train = load_split(config, "train", static_only=True)
    val = load_split(config, "val", static_only=True)
    test = load_split(config, "test", static_only=True)

    modality_info = train["modality_info"]
    competing = bool(modality_info.get("competing_risks", False))
    num_risks = int(modality_info.get("n_events", 1))

    x_train, x_val, x_test = train["x"], val["x"], test["x"]


    x_train, x_val, x_test, _keep = remove_constant_cols(x_train, x_val, x_test)
    n_features = x_train.shape[1]
    col_names = [f"f{i}" for i in range(n_features)]

    cause_codes = risk_axis_of(modality_info)
    horizon = _horizon_of(config)
    bin_cuts = val["cuts"]

    splits = {"val": (val, x_val), "test": (test, x_test)}
    strata = {name: landmark_strata(s["landmarks"], len(s["durations"]))
              for name, (s, _x) in splits.items()}

    if competing and num_risks > 1:
        models_dict = obj if isinstance(obj, dict) else {"risk1": obj}
        results: dict = {name: {} for name in splits}
        surv_arrays: dict = {name: {} for name in splits}
        surv_on_cuts: dict = {name: {} for name in splits}

        for code in cause_codes:
            key = f"risk{code}"
            model_r = models_dict.get(key)
            n_events_r = int((train["events"] == code).sum())
            if model_r is None or n_events_r < 10:
                for name in splits:
                    for landmark, _rows in strata[name]:
                        _landmark_block(results[name], landmark)[
                            _CS_KEY.format(code=code)] = _NAN_BLOCK()
                continue
            for name, (split, x_s) in splits.items():
                dur_s = split["durations"]
                evt_s_r = (split["events"] == code).astype(int)
                if evt_s_r.sum() < 5:
                    for landmark, _rows in strata[name]:
                        _landmark_block(results[name], landmark)[
                            _CS_KEY.format(code=code)] = _NAN_BLOCK()
                    continue
                time_grid = np.linspace(dur_s.min(), dur_s.max(), 100)
                surv = _coxph_survival(model_r, x_s, col_names, time_grid)
                surv_arrays[name][key] = surv
                surv_on_cuts[name][code] = np.asarray(
                    _coxph_survival(model_r, x_s, col_names,
                                    np.asarray(bin_cuts, dtype=float)),
                    dtype=float)
                for landmark, rows in strata[name]:
                    _landmark_block(results[name], landmark)[
                        _CS_KEY.format(code=code)] = evaluate_survival(
                            surv.loc[:, rows], dur_s[rows], evt_s_r[rows],
                            train_durations=train["durations"],
                            train_events_bin=(train["events"] == code).astype(int),
                            horizon=rebased_horizon(horizon, landmark),
                            auc_times=evaluation_times(horizon, landmark, dataset))

        for name, (split, _x) in splits.items():
            reason = _cif_refusal_reason(cause_codes, surv_on_cuts[name],
                                         bin_cuts)
            if reason is not None:
                results[name]["cif_status"] = {"emitted": False,
                                               "reason": reason}
                continue
            saturation: dict = {}
            cif, _overall = aalen_johansen_cif(surv_on_cuts[name],
                                               diagnostics=saturation)
            cif_times = np.asarray(bin_cuts, dtype=float)
            for landmark, rows in strata[name]:
                block = _landmark_block(results[name], landmark)
                for code in cause_codes:
                    block[f"risk{code}"] = compute_competing_risks_metrics(
                        cif=cif[code][:, rows],
                        cif_times=cif_times,
                        durations=split["durations"][rows],
                        events=split["events"][rows],
                        cause=code,
                        times=evaluation_times(horizon, landmark, dataset),
                        horizon=rebased_horizon(horizon, landmark))
            results[name]["cif_status"] = {
                "emitted": True,
                "reason": None,
                "causes": [int(c) for c in cause_codes],
                "grid": "dataset bin cuts (cif_times), as passed to the "
                        "neural runners",
                "saturation": saturation,
            }

        for name in splits:
            emitted = results[name].get("cif_status", {}).get("emitted", False)
            for landmark, _rows in strata[name]:
                block = _landmark_block(results[name], landmark)
                block["mean_causespecific"] = _mean_over_causes(
                    block, cause_codes, _CS_KEY)
                if emitted:
                    block["mean"] = _mean_over_causes(block, cause_codes,
                                                      "risk{code}")
            pooled = [lm for lm, _ in strata[name] if lm is not None]
            if pooled:
                results[name]["mean"] = _pool_landmark_means(results[name],
                                                             pooled)

        out = {"cuts": bin_cuts, "cause_codes": cause_codes}
        for name in splits:
            stack = [surv_arrays[name][f"risk{c}"].values.T
                     for c in cause_codes if f"risk{c}" in surv_arrays[name]]
            out[f"surv_{name}"] = np.stack(stack, axis=1) if stack else None
            first = next((surv_arrays[name][f"risk{c}"] for c in cause_codes
                          if f"risk{c}" in surv_arrays[name]), None)
            out[f"cuts_{name}"] = (np.asarray(first.index, dtype=float)
                                   if first is not None else np.array([]))
            out[f"metrics_{name}"] = results[name]
        return out

    model = obj if not isinstance(obj, dict) else obj.get("risk1")
    out = {"cuts": bin_cuts, "cause_codes": cause_codes}
    for name, (split, x_s) in splits.items():
        dur_s = split["durations"]
        time_grid = np.linspace(dur_s.min(), dur_s.max(), 100)
        surv = _coxph_survival(model, x_s, col_names, time_grid)
        block: dict = {}
        for landmark, rows in strata[name]:
            _landmark_block(block, landmark)["risk1"] = evaluate_survival(
                surv.loc[:, rows], dur_s[rows],
                (split["events"][rows] > 0).astype(int),
                train_durations=train["durations"],
                train_events_bin=(train["events"] > 0).astype(int),
                horizon=rebased_horizon(horizon, landmark),
                auc_times=evaluation_times(horizon, landmark, dataset))
        pooled = [lm for lm, _ in strata[name] if lm is not None]
        if pooled:
            for landmark in pooled:
                lb = block[landmark_key(landmark)]
                lb["mean"] = dict(lb["risk1"])
            block["mean"] = _pool_landmark_means(block, pooled)
        out[f"surv_{name}"] = surv.values.T
        out[f"cuts_{name}"] = np.asarray(surv.index, dtype=float)
        out[f"metrics_{name}"] = block
    return out


def _torch_device(device: str):
    import torch
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _predict_deephit(config: dict, checkpoint: Path, device: str,
                     shipped_json: Optional[dict]) -> dict:
    import torch
    from models.baselines import deephit as deephit_mod
    from models.baselines.deephit import DeepHit
    from models.baselines._data import (dataset_of, load_split,
                                        remove_constant_cols)
    from models.baselines._runtime import risk_axis_of

    dev = _torch_device(device)
    train = load_split(config, "train", static_only=True)
    val = load_split(config, "val", static_only=True)
    test = load_split(config, "test", static_only=True)

    modality_info = train["modality_info"]
    num_risks = _num_risks_of(modality_info)
    horizon = _horizon_of(config)
    cuts = train["cuts"]
    num_bins = len(cuts)

    x_train, x_val, x_test, _keep = remove_constant_cols(
        train["x"], val["x"], test["x"]
    )
    in_features = x_train.shape[1]

    arch = _arch_from_artifact(shipped_json, {
        "shared_dims": [100, 100], "cs_dims": [100], "dropout": 0.4})
    model = DeepHit(in_features, num_risks, num_bins, **arch).to(dev)
    _assert_param_count(model, shipped_json)
    model.load_state_dict(torch.load(checkpoint, map_location=dev), strict=True)
    model.eval()

    out: dict = {"cuts": cuts, "cause_codes": risk_axis_of(modality_info)}
    for split_name, x_s, split in [("val", x_val, val), ("test", x_test, test)]:
        with torch.no_grad():
            pmf = model(torch.from_numpy(x_s).float().to(dev)).cpu().numpy()
        out[f"surv_{split_name}"] = 1.0 - pmf.cumsum(axis=-1)
        out[f"metrics_{split_name}"] = _score_split(
            deephit_mod, pmf, split, train, cuts, num_risks, horizon,
            modality_info, dataset_of(config))
    return out


def _predict_dynamic_deephit(config: dict, checkpoint: Path, device: str,
                             shipped_json: Optional[dict]) -> dict:
    import torch
    from models.baselines import dynamic_deephit as ddh_mod
    from models.baselines.dynamic_deephit import DynamicDeepHit
    from models.baselines._data import dataset_of, load_split
    from models.baselines._runtime import risk_axis_of

    dev = _torch_device(device)
    train = load_split(config, "train")
    val = load_split(config, "val")
    test = load_split(config, "test")

    modality_info = train["modality_info"]
    num_risks = _num_risks_of(modality_info)
    horizon = _horizon_of(config)
    cuts = train["cuts"]
    num_bins = len(cuts)

    x_train = train["x"]
    x_val = val["x"]
    x_test = test["x"]


    mu = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0
    x_val = (x_val - mu) / std
    x_test = (x_test - mu) / std

    in_features = x_train.shape[-1]
    arch = _arch_from_artifact(shipped_json, {
        "rnn_hidden": 100, "rnn_layers": 2, "cs_hidden": 100, "dropout": 0.4})
    model = DynamicDeepHit(in_features, num_risks, num_bins, **arch).to(dev)
    _assert_param_count(model, shipped_json)
    model.load_state_dict(torch.load(checkpoint, map_location=dev), strict=True)
    model.eval()

    out: dict = {"cuts": cuts, "cause_codes": risk_axis_of(modality_info)}
    for split_name, x_s, split in [("val", x_val, val), ("test", x_test, test)]:
        with torch.no_grad():
            pmf, _ = model(torch.from_numpy(x_s).float().to(dev),
                           torch.from_numpy(split["m"]).float().to(dev))
            pmf = pmf.cpu().numpy()
        out[f"surv_{split_name}"] = 1.0 - pmf.cumsum(axis=-1)
        out[f"metrics_{split_name}"] = _score_split(
            ddh_mod, pmf, split, train, cuts, num_risks, horizon,
            modality_info, dataset_of(config))
    return out


def _predict_dysurv(config: dict, checkpoint: Path, device: str, seed: int,
                    shipped_json: Optional[dict]) -> dict:
    import torch
    from models.baselines import dysurv as dysurv_mod
    from models.baselines.dysurv import DySurv
    from models.baselines._data import dataset_of, load_split
    from models.baselines._runtime import risk_axis_of

    dev = _torch_device(device)
    train = load_split(config, "train")
    val = load_split(config, "val")
    test = load_split(config, "test")

    modality_info = train["modality_info"]
    num_risks = _num_risks_of(modality_info)
    horizon = _horizon_of(config)
    cuts = train["cuts"]
    num_bins = len(cuts)

    x_train = train["x"]
    x_val = val["x"]
    x_test = test["x"]
    seq_len = x_train.shape[1]

    mu = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0
    x_val = (x_val - mu) / std
    x_test = (x_test - mu) / std

    in_features = x_train.shape[-1]
    arch = _arch_from_artifact(shipped_json, {"encoded_features": 32})
    model = DySurv(in_features, out_features=num_bins, seq_len=seq_len,
                   num_risks=num_risks, **arch).to(dev)
    _assert_param_count(model, shipped_json)
    model.load_state_dict(torch.load(checkpoint, map_location=dev), strict=True)
    model.eval()


    _set_seed(seed)
    out: dict = {"cuts": cuts, "cause_codes": risk_axis_of(modality_info)}
    for split_name, x_s, split in [("val", x_val, val), ("test", x_test, test)]:
        with torch.no_grad():
            surv = model.predict_surv(
                torch.from_numpy(x_s).float().to(dev)
            ).cpu().numpy()
        out[f"surv_{split_name}"] = surv
        out[f"metrics_{split_name}"] = _score_split(
            dysurv_mod, surv, split, train, cuts, num_risks, horizon,
            modality_info, dataset_of(config))
    return out


def _predict_transformer(config: dict, checkpoint: Path, device: str, seed: int,
                         modalities: Sequence[str],
                         shipped_json: Optional[dict]) -> dict:
    import torch
    from models.baselines.transformer import (
        TransformerSurv, MultiModalDataset, evaluate, _arm_schema,
    )
    from models.baselines._data import output_dir_of, prefix_of
    from models.baselines._runtime import risk_axis_of

    dev = _torch_device(device)
    out_dir = output_dir_of(config)
    prefix = prefix_of(config)


    modality_info = _arm_schema(config)
    num_risks = int(modality_info["n_events"])

    cuts = np.load(out_dir / f"cuts_{prefix}.npy")
    out_features = int(np.load(out_dir / f"out_features_{prefix}.npy"))

    val_ds = MultiModalDataset(config, "val", tuple(modalities))
    test_ds = MultiModalDataset(config, "test", tuple(modalities))
    n_features = test_ds.x.shape[-1]

    arch = _arch_from_artifact(shipped_json, {
        "d_model": 128, "n_layers": 3, "n_heads": 4, "ff_dim": 256,
        "encoder_dropout": 0.1, "head_dropout": 0.3, "head_hidden": 32,
        "aux_recon": False, "pooling": "mean"})


    echo = (shipped_json or {}).get("effective_hparams") or {}
    widths = {name: echo[f"{name}_bottleneck"]
              for name in ("icd", "radiology")
              if echo.get(f"{name}_bottleneck", "none") != "none"}
    model = TransformerSurv(
        n_features=n_features, n_risks=num_risks, n_bins=out_features,
        modality_slots=(test_ds.modality_keys if widths else None),
        bottleneck_widths=(widths or None),
        **arch,
    ).to(dev)
    _assert_param_count(model, shipped_json)
    model.load_state_dict(torch.load(checkpoint, map_location=dev), strict=True)
    model.eval()


    with open(out_dir / f"y_val_surv_{prefix}.p", "rb") as f:
        y_val = pickle.load(f)
    durations_val = np.asarray(y_val[0], dtype=np.float64)
    events_val = np.asarray(y_val[1], dtype=np.int64)
    durations_test = np.load(out_dir / f"durations_test_{prefix}.npy").astype(np.float64)
    events_test = np.load(out_dir / f"events_test_{prefix}.npy").astype(np.int64)
    with open(out_dir / f"y_train_surv_{prefix}.p", "rb") as f:
        y_train = pickle.load(f)
    train_durations = np.asarray(y_train[0], dtype=np.float64)
    train_events_full = np.asarray(y_train[1], dtype=np.int64)

    from models.baselines._data import load_split as _load_split
    _te = _load_split(config, "test")
    _va = _load_split(config, "val")
    _tr = _load_split(config, "train")
    durations_test = np.asarray(_te["durations"], dtype=np.float64)
    events_test = np.asarray(_te["events"], dtype=np.int64)
    events_val = np.asarray(_va["events"], dtype=np.int64)
    durations_val = np.asarray(_va["durations"], dtype=np.float64)
    train_durations = np.asarray(_tr["durations"], dtype=np.float64)
    train_events_full = np.asarray(_tr["events"], dtype=np.int64)
    horizon = float(config.get("preprocessing", {}).get("max_horizon_hours", 240))

    x_val = val_ds.x.astype(np.float32)
    m_val = val_ds.m.astype(np.float32)
    x_test = test_ds.x.astype(np.float32)
    m_test = test_ds.m.astype(np.float32)


    from models.baselines._data import dataset_of as _dataset_of
    _dataset = _dataset_of(config)

    _set_seed(seed)
    _vm, surv_val_np = evaluate(
        model, x_val, m_val, durations_val, events_val, cuts,
        num_risks, train_durations, train_events_full, dev, 128,
        horizon=horizon, modality_info=modality_info,
        landmarks=_va["landmarks"], dataset=_dataset,
    )
    _tm, surv_test_np = evaluate(
        model, x_test, m_test, durations_test, events_test, cuts,
        num_risks, train_durations, train_events_full, dev, 128,
        horizon=horizon, modality_info=modality_info,
        landmarks=_te["landmarks"], dataset=_dataset,
    )
    return {"cuts": cuts, "surv_val": surv_val_np, "surv_test": surv_test_np,
            "cause_codes": risk_axis_of(modality_info),
            "metrics_val": _vm, "metrics_test": _tm}


_MM_MODALITIES = {
    "mimiciv": ("dynamic", "static", "icd", "radiology"),
    "mcmed":   ("dynamic", "static", "icd", "radiology"),
}
