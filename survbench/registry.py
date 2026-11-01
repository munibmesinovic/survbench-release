import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

DATASETS = {}
MODELS = {}


def _resolve(ref):
    if not isinstance(ref, str):
        return ref
    module, _, attr = ref.partition(":")
    obj = importlib.import_module(module)
    return getattr(obj, attr) if attr else obj


def register_dataset(name, loader=None, configs=None, *, landmarks=None,
                     windows=None):
    if name in DATASETS:
        raise ValueError(f"dataset {name} is already registered")
    if loader is not None:
        from preprocessing.pipeline import register_loader
        register_loader(name, loader)
    DATASETS[name] = {
        "configs": {arm: Path(path).resolve() for arm, path in (configs or {}).items()},
        "landmarks": None if landmarks is None else tuple(float(h) for h in landmarks),
        "windows": None if windows is None else tuple(float(h) for h in windows),
    }


def register_model(name, module, *, predictor=None, space=None, echo=(),
                   summary=False, selection=None, checkpoint="pt"):
    if name in MODELS:
        raise ValueError(f"model {name} is already registered")
    MODELS[name] = {"module": module, "predictor": predictor, "space": space,
                    "echo": tuple(echo), "summary": bool(summary),
                    "selection": selection, "checkpoint": checkpoint}


def model_spec(name):
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name}; registered: {sorted(MODELS)}") from None


def model_module(name):
    spec = model_spec(name)
    spec["module"] = _resolve(spec["module"])
    return spec["module"]


def model_predictor(name):
    spec = MODELS.get(name)
    if spec is None or spec["predictor"] is None:
        return None
    spec["predictor"] = _resolve(spec["predictor"])
    return spec["predictor"]


def model_space(name):
    spec = MODELS.get(name)
    if spec is None or spec["space"] is None:
        return None
    spec["space"] = _resolve(spec["space"])
    return spec["space"]


def default_models():
    out = []
    for name, spec in MODELS.items():
        out.append(name)
        if spec["summary"]:
            out.append(f"{name}-summary")
    return out


def landmark_hours(dataset):
    return (DATASETS.get(dataset) or {}).get("landmarks")


def window_hours(dataset):
    return (DATASETS.get(dataset) or {}).get("windows")


def load_plugin(spec):
    path = Path(spec)
    if path.suffix == ".py" and path.exists():
        name = f"survbench_plugin_{path.stem}"
        mod_spec = importlib.util.spec_from_file_location(name, path.resolve())
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[name] = module
        mod_spec.loader.exec_module(module)
        return module
    return importlib.import_module(spec)


def load_plugins(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--plugin", action="append", default=[])
    known, _ = ap.parse_known_args(argv)
    return [load_plugin(spec) for spec in known.plugin]


def add_plugin_argument(ap):
    ap.add_argument("--plugin", action="append", default=[])


def load_data(config, input_mode=None):
    from models.baselines._data import load_split, static_input_kwargs
    kw = static_input_kwargs(input_mode) if input_mode else {}
    return tuple(load_split(config, s, **kw) for s in ("train", "val", "test"))


def causes(split):
    from models.baselines._runtime import risk_axis_of
    return risk_axis_of(split["modality_info"])


def score(config, train, split, cif):
    from models.baselines._data import dataset_of
    from models.baselines.deephit import evaluate_by_landmark
    cif = np.asarray(cif, dtype=float)
    if cif.ndim == 2:
        cif = cif[:, None, :]
    pmf = np.diff(cif, axis=-1, prepend=0.0)
    return evaluate_by_landmark(
        pmf, split["durations"], split["events"], train["cuts"], cif.shape[1],
        train_durations=train["durations"], train_events_full=train["events"],
        horizon=float(config["preprocessing"]["max_horizon_hours"]),
        dataset=dataset_of(config), modality_info=train["modality_info"],
        landmarks=split["landmarks"])


def result(config, name, seed, data, predict, effective_hparams, *,
           save_dir=None, save_tag=None, model=None, input_mode=None):
    from models.baselines._data import dataset_of
    from models.baselines._runtime import (RISK_AXIS_KEY, save_checkpoint,
                                           save_predictions)
    train, val, test = data
    axis = causes(train)
    splits = {"val": val} if test is None else {"val": val, "test": test}
    cif = {s: np.asarray(predict(split), dtype=float) for s, split in splits.items()}
    blocks = {s: score(config, train, split, cif[s]) for s, split in splits.items()}
    out = {"model": name, "dataset": dataset_of(config), "seed": seed,
           "n_train": len(train["durations"]), "n_val": len(val["durations"]),
           "n_test": None if test is None else len(test["durations"]),
           "horizon_hours": float(config["preprocessing"]["max_horizon_hours"]),
           "effective_hparams": dict(effective_hparams), RISK_AXIS_KEY: axis,
           "input_mode": input_mode, "model_info": {"type": name},
           "training": {"param_count": None},
           "metrics": {s: b.get("mean", b) for s, b in blocks.items()}}
    out.update(blocks)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = save_tag or f"{name}_{dataset_of(config)}_seed{seed}"
        save_predictions(save_dir / f"{tag}_preds.npz", cuts=train["cuts"],
                         cause_codes=axis,
                         surv_val=1.0 - cif["val"].reshape(len(cif["val"]), len(axis), -1),
                         surv_test=(None if test is None else 1.0 - cif["test"].reshape(
                             len(cif["test"]), len(axis), -1)))
        if model is not None:
            suffix = (MODELS.get(name) or {}).get("checkpoint", "pt")
            save_checkpoint(save_dir / f"{tag}.{suffix}", model)
    return out
