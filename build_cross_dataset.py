
from __future__ import annotations

import argparse
import json
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

project_root = str(Path(__file__).resolve().parent)
sys.path.insert(0, project_root)

from cross_dataset_features import compute_shared_schema
from survbench.core.arms import arm_of
from sklearn.preprocessing import StandardScaler
from pycox.preprocessing.label_transforms import LabTransDiscreteTime


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pkl(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _output_paths(config: dict, split: str) -> dict:
    out = Path(config["output"]["dir"])
    files = config["output"]["files"]
    prefix_path = files.get(f"x_{split}")
    if prefix_path is None:
        raise KeyError(f"output.files.x_{split} missing in config")
    return {
        "x": out / prefix_path,
        "mask": out / prefix_path.replace(".npy", "_mask.npy"),
        "y": out / files.get(f"y_{split}", f"y_{split}_surv.p"),
        "feature_names": out / files.get("feature_names", "feature_names.pkl"),
        "modality_info": out / files.get("modality_info", "modality_info.pkl"),
    }


def _find_scaler_paths(config: dict) -> dict:
    out = Path(config["output"]["dir"])
    files = config["output"]["files"]
    prefix = config["output"]["prefix"]


    def _resolve(key: str, suffixed: str, bare: str) -> Path:
        if key in files:
            return out / files[key]
        if (out / suffixed).exists():
            return out / suffixed
        return out / bare
    return {
        "dynamic": _resolve("scaler_dynamic", f"scaler_dynamic_{prefix}.pkl", "scaler_dynamic.pkl"),
        "static": _resolve("scaler_static", f"scaler_static_{prefix}.pkl", "scaler_static.pkl"),
        "legacy": _resolve("scaler", f"scaler_{prefix}.pkl", "scaler.pkl"),
    }


def assert_arms_agree(source_config: dict, target_config: dict) -> None:
    src = arm_of(source_config)
    tgt = arm_of(target_config)
    if src != tgt:
        raise ValueError(
            f"source and target disagree about the arm: source={src}, target={tgt}")


def load_dataset(config: dict) -> dict:
    paths_train = _output_paths(config, "train")
    paths_val = _output_paths(config, "val")
    paths_test = _output_paths(config, "test")

    x_train = np.load(paths_train["x"])
    x_val = np.load(paths_val["x"])
    x_test = np.load(paths_test["x"])

    def _maybe_mask(p):
        return np.load(p) if p.exists() else np.ones_like(x_train, dtype=np.float32)

    m_train = _maybe_mask(paths_train["mask"])
    m_val = _maybe_mask(paths_val["mask"])
    m_test = _maybe_mask(paths_test["mask"])

    with open(paths_train["y"], "rb") as f:
        d_train, e_train = pickle.load(f)
    with open(paths_val["y"], "rb") as f:
        d_val, e_val = pickle.load(f)


    out = Path(config["output"]["dir"])
    files = config["output"]["files"]
    durations_test_path = out / files.get("durations_test", "durations_test.npy")
    events_test_path = out / files.get("events_test", "events_test.npy")
    if durations_test_path.exists() and events_test_path.exists():
        d_test = np.load(durations_test_path)
        e_test = np.load(events_test_path)
    elif paths_test["y"].exists():
        with open(paths_test["y"], "rb") as f:
            d_test, e_test = pickle.load(f)
    else:
        raise FileNotFoundError(
            f"Test labels missing. Tried {durations_test_path}, "
            f"{events_test_path}, and {paths_test['y']}."
        )


    subject_ids = {}
    sid_prefix = config["output"]["prefix"]
    for split in ("train", "val", "test"):
        p = out / f"subject_ids_{split}_{sid_prefix}.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing; subject ids are needed to split the transfer corpus by patient")


        subject_ids[split] = np.load(p, allow_pickle=True)

    feature_info = _load_pkl(paths_train["feature_names"])
    modality_info = _load_pkl(paths_train["modality_info"])

    spaths = _find_scaler_paths(config)
    if spaths["dynamic"].exists() and spaths["static"].exists():
        scaler_dyn = _load_pkl(spaths["dynamic"])
        scaler_sta = _load_pkl(spaths["static"])
    elif spaths["legacy"].exists():

        scaler_dyn = _load_pkl(spaths["legacy"])
        scaler_sta = None
    else:
        raise FileNotFoundError(
            f"No scaler found at {spaths['dynamic']} / {spaths['static']} "
            f"or legacy {spaths['legacy']}"
        )

    return {
        "x_train": x_train, "x_val": x_val, "x_test": x_test,
        "m_train": m_train, "m_val": m_val, "m_test": m_test,
        "d_train": np.asarray(d_train, dtype=np.float64),
        "e_train": np.asarray(e_train, dtype=np.int64),
        "d_val": np.asarray(d_val, dtype=np.float64),
        "e_val": np.asarray(e_val, dtype=np.int64),
        "d_test": np.asarray(d_test, dtype=np.float64),
        "e_test": np.asarray(e_test, dtype=np.int64),
        "subject_ids": subject_ids,
        "feature_info": feature_info,
        "modality_info": modality_info,
        "scaler_dynamic": scaler_dyn,
        "scaler_static": scaler_sta,
    }


def _inverse_scale_block(
    x: np.ndarray, cols: list[int], scaler, n_features_block: int,
    block_offset: int,
) -> np.ndarray:
    if scaler is None:

        return x[:, :, cols].copy()
    N, T, _ = x.shape


    block = x[:, :, block_offset : block_offset + n_features_block]
    flat = block.reshape(N * T, n_features_block)
    inv = scaler.inverse_transform(flat).reshape(N, T, n_features_block)


    local_cols = [c - block_offset for c in cols]
    return inv[:, :, local_cols]


def _legacy_inverse_block(
    x: np.ndarray, cols: list[int], scaler, n_dynamic: int, n_static: int,
) -> np.ndarray:
    N, T, F = x.shape
    flat = x.reshape(N * T, F)
    inv = scaler.inverse_transform(flat).reshape(N, T, F)
    return inv[:, :, cols]


def carve_and_inverse(ds: dict, dataset_name: str, schema: dict) -> dict:
    feature_info = ds["feature_info"]
    dyn_indices = list(feature_info.get("dynamic_indices", []))
    sta_indices = list(feature_info.get("static_indices", []))
    n_dyn = len(dyn_indices)
    n_sta = len(sta_indices)


    dyn_block_offset = min(dyn_indices) if dyn_indices else 0
    sta_block_offset = min(sta_indices) if sta_indices else n_dyn

    cols_dyn = (
        schema["source_dynamic_cols"]
        if dataset_name == "source"
        else schema["target_dynamic_cols"]
    )
    cols_sta = (
        schema["source_static_cols"]
        if dataset_name == "source"
        else schema["target_static_cols"]
    )

    use_legacy = ds["scaler_static"] is None

    out = {}
    for split in ("train", "val", "test"):
        x = ds[f"x_{split}"]
        m = ds[f"m_{split}"]
        if use_legacy:
            dyn_inv = _legacy_inverse_block(x, cols_dyn, ds["scaler_dynamic"], n_dyn, n_sta)
            sta_inv = _legacy_inverse_block(x, cols_sta, ds["scaler_dynamic"], n_dyn, n_sta)
        else:
            dyn_inv = _inverse_scale_block(
                x, cols_dyn, ds["scaler_dynamic"], n_dyn, dyn_block_offset,
            )
            sta_inv = _inverse_scale_block(
                x, cols_sta, ds["scaler_static"], n_sta, sta_block_offset,
            )

        ops = (
            schema["source_static_ops"]
            if dataset_name == "source"
            else schema["target_static_ops"]
        )
        for j, op in enumerate(ops):
            if op == "binary_indicator":
                sta_inv[:, :, j] = (sta_inv[:, :, j] > 0.5).astype(np.float32)
        out[f"dyn_{split}"] = dyn_inv.astype(np.float32)
        out[f"sta_{split}"] = sta_inv.astype(np.float32)
        out[f"mask_dyn_{split}"] = m[:, :, cols_dyn]
        out[f"mask_sta_{split}"] = m[:, :, cols_sta]
    return out


def refit_and_standardise(
    source_blocks: dict, target_blocks: dict, z_clip: float = 10.0,
) -> tuple[dict, dict, StandardScaler, StandardScaler]:
    src_train_dyn = source_blocks["dyn_train"]
    src_train_sta = source_blocks["sta_train"]

    N, T, F_dyn = src_train_dyn.shape
    F_sta = src_train_sta.shape[2]


    src_dyn_mean = np.nanmean(src_train_dyn.reshape(-1, F_dyn), axis=0)
    src_sta_mean = np.nanmean(src_train_sta.reshape(-1, F_sta), axis=0)
    src_dyn_mean = np.nan_to_num(src_dyn_mean, nan=0.0)
    src_sta_mean = np.nan_to_num(src_sta_mean, nan=0.0)

    def _fill_split(blocks: dict, kind: str, mean: np.ndarray, F: int):
        for split in ("train", "val", "test"):
            arr = blocks[f"{kind}_{split}"]
            mask = np.isnan(arr)
            if mask.any():

                broadcast = np.broadcast_to(mean, arr.shape).copy()
                arr[mask] = broadcast[mask]
            blocks[f"{kind}_{split}"] = arr

    _fill_split(source_blocks, "dyn", src_dyn_mean, F_dyn)
    _fill_split(source_blocks, "sta", src_sta_mean, F_sta)
    _fill_split(target_blocks, "dyn", src_dyn_mean, F_dyn)
    _fill_split(target_blocks, "sta", src_sta_mean, F_sta)

    scaler_dyn = StandardScaler().fit(src_train_dyn.reshape(-1, F_dyn))
    scaler_sta = StandardScaler().fit(src_train_sta.reshape(-1, F_sta))

    def _apply(blocks: dict):
        for split in ("train", "val", "test"):
            d = blocks[f"dyn_{split}"]
            s = blocks[f"sta_{split}"]
            Nd, Td, Fd = d.shape
            Ns, Ts, Fs = s.shape
            d = scaler_dyn.transform(d.reshape(-1, Fd)).reshape(Nd, Td, Fd)
            s = scaler_sta.transform(s.reshape(-1, Fs)).reshape(Ns, Ts, Fs)
            d = np.clip(d, -z_clip, z_clip)
            s = np.clip(s, -z_clip, z_clip)
            blocks[f"dyn_{split}"] = d.astype(np.float32)
            blocks[f"sta_{split}"] = s.astype(np.float32)

    _apply(source_blocks)
    _apply(target_blocks)
    return source_blocks, target_blocks, scaler_dyn, scaler_sta


def stack_blocks(blocks: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    d = blocks[f"dyn_{split}"]
    s = blocks[f"sta_{split}"]
    x = np.concatenate([d, s], axis=2)
    m = np.concatenate(
        [blocks[f"mask_dyn_{split}"], blocks[f"mask_sta_{split}"]], axis=2,
    )
    return x.astype(np.float32), m.astype(np.float32)


def cuts_from_target(target_config: dict) -> np.ndarray:
    from models.baselines._data import output_dir_of, prefix_of
    return np.load(
        output_dir_of(target_config) / f"cuts_{prefix_of(target_config)}.npy")


def refit_cuts_from_source(
    durations_train: np.ndarray, events_train: np.ndarray, n_time_bins: int,
    scheme: str = "quantiles",
) -> np.ndarray:
    lab_trans = LabTransDiscreteTime(cuts=n_time_bins, scheme=scheme)
    lab_trans.fit(durations_train.astype(np.float64), events_train.astype(np.int64))
    cuts = np.asarray(lab_trans.cuts, dtype=np.float64)
    return cuts


def write_outputs(
    output_dir: Path, prefix: str,
    source_blocks: dict, target_blocks: dict,
    source_ds: dict, target_ds: dict,
    schema: dict, cuts: np.ndarray,
    scaler_dynamic: StandardScaler, scaler_static: StandardScaler,
    source_config: dict, target_config: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)


    x_train, m_train = stack_blocks(source_blocks, "train")
    x_val, m_val = stack_blocks(source_blocks, "val")
    x_test, m_test = stack_blocks(target_blocks, "test")

    np.save(output_dir / f"x_train_{prefix}.npy", x_train)
    np.save(output_dir / f"x_val_{prefix}.npy", x_val)
    np.save(output_dir / f"x_test_{prefix}.npy", x_test)
    np.save(output_dir / f"x_train_{prefix}_mask.npy", m_train)
    np.save(output_dir / f"x_val_{prefix}_mask.npy", m_val)
    np.save(output_dir / f"x_test_{prefix}_mask.npy", m_test)


    _save_pkl(
        (source_ds["d_train"], source_ds["e_train"]),
        output_dir / f"y_train_surv_{prefix}.p",
    )
    _save_pkl(
        (source_ds["d_val"], source_ds["e_val"]),
        output_dir / f"y_val_surv_{prefix}.p",
    )
    np.save(output_dir / f"durations_test_{prefix}.npy", target_ds["d_test"])
    np.save(output_dir / f"events_test_{prefix}.npy", target_ds["e_test"])


    tgt_prefix = f"{prefix}_target"
    tx_train, tm_train = stack_blocks(target_blocks, "train")
    tx_val, tm_val = stack_blocks(target_blocks, "val")

    np.save(output_dir / f"x_train_{tgt_prefix}.npy", tx_train)
    np.save(output_dir / f"x_val_{tgt_prefix}.npy", tx_val)
    np.save(output_dir / f"x_test_{tgt_prefix}.npy", x_test)
    np.save(output_dir / f"x_train_{tgt_prefix}_mask.npy", tm_train)
    np.save(output_dir / f"x_val_{tgt_prefix}_mask.npy", tm_val)
    np.save(output_dir / f"x_test_{tgt_prefix}_mask.npy", m_test)
    _save_pkl((target_ds["d_train"], target_ds["e_train"]),
              output_dir / f"y_train_surv_{tgt_prefix}.p")
    _save_pkl((target_ds["d_val"], target_ds["e_val"]),
              output_dir / f"y_val_surv_{tgt_prefix}.p")
    np.save(output_dir / f"durations_test_{tgt_prefix}.npy", target_ds["d_test"])
    np.save(output_dir / f"events_test_{tgt_prefix}.npy", target_ds["e_test"])
    for split in ("train", "val", "test"):
        np.save(output_dir / f"subject_ids_{split}_{tgt_prefix}.npy",
                target_ds["subject_ids"][split])
    np.save(output_dir / f"cuts_{tgt_prefix}.npy", cuts)
    np.save(output_dir / f"out_features_{tgt_prefix}.npy", np.int64(cuts.shape[0]))


    from survbench.core.arms import tuning_arm_of
    marker = {
        "source_arm": tuning_arm_of(source_config),
        "target_arm": tuning_arm_of(target_config),
        "direction": prefix,
        "cuts_from": str(Path(target_config["output"]["dir"])),
        "n_shared_features": int(x_train.shape[-1]),
        "n_shared_dynamic": int(len(schema["dynamic_names"])),
        "n_shared_static": int(len(schema["static_names"])),
    }
    with open(output_dir / "SURVBENCH_VERSION.json", "w") as f:
        json.dump(marker, f, indent=2)


    for split in ("train", "val"):
        np.save(output_dir / f"subject_ids_{split}_{prefix}.npy",
                source_ds["subject_ids"][split])
    np.save(output_dir / f"subject_ids_test_{prefix}.npy",
            target_ds["subject_ids"]["test"])


    np.save(output_dir / f"cuts_{prefix}.npy", cuts)
    np.save(output_dir / f"out_features_{prefix}.npy", np.int64(cuts.shape[0]))

    dyn_names = schema["dynamic_names"]
    sta_names = schema["static_names"]
    n_dyn = len(dyn_names)
    n_sta = len(sta_names)
    feature_info = {
        "dynamic_names": dyn_names,
        "static_names": sta_names,
        "dynamic_indices": list(range(0, n_dyn)),
        "static_indices": list(range(n_dyn, n_dyn + n_sta)),
    }
    _save_pkl(feature_info, output_dir / f"feature_names_{prefix}.pkl")


    src_mi = source_ds["modality_info"]
    modality_info = {
        "cause_codes": list(src_mi["cause_codes"]),
        "competing_risks": bool(src_mi["competing_risks"]),
        "n_events": int(src_mi["n_events"]),
        "active_modalities": ["dynamic", "static"],
        "n_features": n_dyn + n_sta,
        "n_dynamic": n_dyn,
        "n_static": n_sta,
        "source_dataset": source_config["dataset"].get("name", "source"),
        "target_dataset": target_config["dataset"].get("name", "target"),
    }
    _save_pkl(modality_info, output_dir / f"modality_info_{prefix}.pkl")


    _save_pkl(feature_info, output_dir / f"feature_names_{tgt_prefix}.pkl")
    _save_pkl(modality_info, output_dir / f"modality_info_{tgt_prefix}.pkl")


    _save_pkl(scaler_dynamic, output_dir / f"scaler_dynamic_{prefix}.pkl")
    _save_pkl(scaler_static, output_dir / f"scaler_static_{prefix}.pkl")


    with open(output_dir / f"shared_schema_{prefix}.json", "w") as f:
        json.dump(
            {
                "dynamic_names": dyn_names,
                "static_names": sta_names,
                "source_dynamic_cols": schema["source_dynamic_cols"],
                "target_dynamic_cols": schema["target_dynamic_cols"],
                "source_static_cols": schema["source_static_cols"],
                "target_static_cols": schema["target_static_cols"],
                "source_dataset": source_config["dataset"].get("name", "source"),
                "target_dataset": target_config["dataset"].get("name", "target"),
                "n_train": int(x_train.shape[0]),
                "n_val": int(x_val.shape[0]),
                "n_test": int(x_test.shape[0]),
                "n_time_bins": int(cuts.shape[0]),
                "cuts_hours": cuts.tolist(),
            },
            f,
            indent=2,
        )


    new_config = deepcopy(source_config)
    new_config["dataset"]["name"] = f"{source_config['dataset']['name']}_to_{target_config['dataset']['name']}"
    new_config["output"] = {
        "dir": str(output_dir),
        "prefix": prefix,
        "files": {
            "x_train": f"x_train_{prefix}.npy",
            "x_val": f"x_val_{prefix}.npy",
            "x_test": f"x_test_{prefix}.npy",
            "train_mask": f"x_train_{prefix}_mask.npy",
            "val_mask": f"x_val_{prefix}_mask.npy",
            "test_mask": f"x_test_{prefix}_mask.npy",
            "y_train": f"y_train_surv_{prefix}.p",
            "y_val": f"y_val_surv_{prefix}.p",
            "y_test": f"y_test_surv_{prefix}.p",
            "durations_test": f"durations_test_{prefix}.npy",
            "events_test": f"events_test_{prefix}.npy",
            "cuts": f"cuts_{prefix}.npy",
            "feature_names": f"feature_names_{prefix}.pkl",
            "modality_info": f"modality_info_{prefix}.pkl",
            "scaler_dynamic": f"scaler_dynamic_{prefix}.pkl",
            "scaler_static": f"scaler_static_{prefix}.pkl",
        },
    }


    new_config["modalities"] = {
        "timeseries": True, "static": True, "icd": False, "radiology": False,
    }
    config_path = output_dir / f"{prefix}_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(new_config, f, sort_keys=False)


    target_config_out = deepcopy(new_config)
    target_config_out["dataset"]["name"] = f"{new_config['dataset']['name']}_target"
    target_config_out["output"] = {
        "dir": str(output_dir),
        "prefix": tgt_prefix,
        "files": {
            "x_train": f"x_train_{tgt_prefix}.npy",
            "x_val": f"x_val_{tgt_prefix}.npy",
            "x_test": f"x_test_{tgt_prefix}.npy",
            "train_mask": f"x_train_{tgt_prefix}_mask.npy",
            "val_mask": f"x_val_{tgt_prefix}_mask.npy",
            "test_mask": f"x_test_{tgt_prefix}_mask.npy",
            "y_train": f"y_train_surv_{tgt_prefix}.p",
            "y_val": f"y_val_surv_{tgt_prefix}.p",
            "durations_test": f"durations_test_{tgt_prefix}.npy",
            "events_test": f"events_test_{tgt_prefix}.npy",
            "cuts": f"cuts_{tgt_prefix}.npy",
            "feature_names": f"feature_names_{tgt_prefix}.pkl",
            "modality_info": f"modality_info_{tgt_prefix}.pkl",
            "scaler_dynamic": f"scaler_dynamic_{prefix}.pkl",
            "scaler_static": f"scaler_static_{prefix}.pkl",
        },
    }
    target_config_path = output_dir / f"{tgt_prefix}_config.yaml"
    with open(target_config_path, "w") as f:
        yaml.safe_dump(target_config_out, f, sort_keys=False)

    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--n-time-bins", type=int, default=10)
    parser.add_argument("--z-clip", type=float, default=10.0)
    args = parser.parse_args()

    print(f"Source config:  {args.source_config}")
    print(f"Target config:  {args.target_config}")
    src_cfg = _load_config(args.source_config)
    tgt_cfg = _load_config(args.target_config)


    assert_arms_agree(src_cfg, tgt_cfg)

    print(f"\n[1/5] Loading source ({src_cfg['dataset']['name']})...")
    src = load_dataset(src_cfg)
    print(f"      x_train: {src['x_train'].shape}  x_val: {src['x_val'].shape}  "
          f"x_test: {src['x_test'].shape}")

    print(f"\n[1/5] Loading target ({tgt_cfg['dataset']['name']})...")
    tgt = load_dataset(tgt_cfg)
    print(f"      x_train: {tgt['x_train'].shape}  x_val: {tgt['x_val'].shape}  "
          f"x_test: {tgt['x_test'].shape}")

    src_name = src_cfg["dataset"]["name"]
    tgt_name = tgt_cfg["dataset"]["name"]

    print("\n[2/5] Computing shared schema...")
    schema = compute_shared_schema(
        src["feature_info"], tgt["feature_info"],
        source_dataset=src_name, target_dataset=tgt_name,
    )
    print(f"      Shared dynamic features ({len(schema['dynamic_names'])}): "
          f"{schema['dynamic_names']}")
    print(f"      Shared static features ({len(schema['static_names'])}): "
          f"{schema['static_names']}")

    print("\n[3/5] Inverse-scaling to physical units + carving shared columns...")
    src_blocks = carve_and_inverse(src, "source", schema)
    tgt_blocks = carve_and_inverse(tgt, "target", schema)

    print("\n[4/5] Refitting StandardScaler on source train + applying everywhere...")
    src_blocks, tgt_blocks, scaler_dyn, scaler_sta = refit_and_standardise(
        src_blocks, tgt_blocks, z_clip=args.z_clip,
    )

    print("\n[4/5] Taking the discretisation grid from the target arm...")
    cuts = cuts_from_target(tgt_cfg)
    print(f"      cuts: {cuts}")
    if len(cuts) - 1 != args.n_time_bins:
        print(f"      --n-time-bins={args.n_time_bins} ignored; the target grid has {len(cuts) - 1} bins")

    print(f"\n[5/5] Writing outputs to {args.output_dir} (prefix={args.prefix})...")
    config_path = write_outputs(
        args.output_dir, args.prefix,
        src_blocks, tgt_blocks, src, tgt, schema, cuts,
        scaler_dyn, scaler_sta, src_cfg, tgt_cfg,
    )
    print(f"\nDone. Cross-dataset config: {config_path}")


if __name__ == "__main__":
    main()
