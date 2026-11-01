import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.baselines import _data
from models.baselines._runtime import (ARMS, SEEDS, _set_seed, all_datasets,
                                       cell_config, cells, input_mode_of,
                                       invoke, load_hparams, modalities_of,
                                       registry_key, run_path, runner_kwargs,
                                       runs_root)
from survbench import registry

DEFAULT_MODALITIES = ("dynamic", "static")
PENALIZER_GRID = (0.01, 0.1, 1.0)


def stamp_arm(result: dict, config: dict) -> dict:
    arm, censored = _data.arm_of(config)
    result["discharge_as"] = arm
    result["censored_causes"] = censored
    return result


def cell_modalities(model: str, entry: dict) -> tuple:
    if registry_key(model) != "transformer":
        return DEFAULT_MODALITIES
    if entry.get("modalities"):
        return tuple(entry["modalities"])
    return modalities_of(model) or DEFAULT_MODALITIES


def assert_echo_matches(model: str, wanted: dict, echo) -> None:
    if echo is None:
        raise AssertionError(
            f"{model}.run() returned no effective_hparams; cannot check them against configs/hparams.yaml")
    bad = []
    for key, want in wanted.items():
        got = echo.get(key)
        if isinstance(want, float) or isinstance(got, float):
            ok = got is not None and abs(float(got) - float(want)) <= 1e-9
        elif isinstance(want, (list, tuple)) or isinstance(got, (list, tuple)):
            ok = list(got or []) == list(want)
        else:
            ok = got == want
        if not ok:
            bad.append((key, want, got))
    if bad:
        raise AssertionError(
            f"{model} trained at a configuration that disagrees with "
            f"configs/hparams.yaml: "
            + "; ".join(f"{k}: wanted {w!r}, echoed {g!r}" for k, w, g in bad)
            + ".")


def train_cell(arm, dataset, model, seed, entry, out_dir, split_seed=None):
    config = cell_config(arm, dataset, split_seed)
    winner = runner_kwargs(model, entry.get("effective_hparams"))
    modalities = cell_modalities(model, entry)
    save_dir = Path(out_dir) / arm
    if save_dir.resolve() == (ROOT / "checkpoints" / arm).resolve():
        raise SystemExit(
            "checkpoints/ holds the shipped artifacts; train into --out-dir runs/ (the default) instead.")
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model}_{dataset}_seed{seed}"

    kwargs = dict(winner)
    kwargs["seed"] = seed
    kwargs["save_dir"] = save_dir
    kwargs["save_tag"] = tag
    if registry_key(model) == "transformer":
        kwargs["modalities"] = modalities
    input_mode = input_mode_of(model)
    if input_mode is not None:
        kwargs["input_mode"] = input_mode
    if entry.get("device"):
        kwargs["device"] = entry["device"]

    threads = entry.get("threads")
    if threads:
        import torch
        previous = torch.get_num_threads()
        torch.set_num_threads(threads)
    _set_seed(seed)
    try:
        result = invoke(model, config, **kwargs)
    finally:
        if threads:
            torch.set_num_threads(previous)
    assert_echo_matches(model, winner, result.get("effective_hparams"))
    if input_mode is not None and result.get("input_mode") != input_mode:
        raise AssertionError(
            f"{model} was asked for input_mode={input_mode!r} and reported "
            f"{result.get('input_mode')!r}.")
    stamp_arm(result, config)
    result["model"] = model
    result["seed"] = seed
    with open(save_dir / f"{tag}.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    return result


def read_run(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def with_penalizer(entry, penalizer):
    hp = dict(entry.get("effective_hparams") or {}, penalizer=penalizer)
    return dict(entry, effective_hparams=hp)


def select_penalizer(arm, dataset, model, entry, out_dir, split_seed):
    val = {}
    for pen in PENALIZER_GRID:
        sub = Path(out_dir) / "coxpen" / f"pen{pen:g}"
        result = read_run(run_path(sub, arm, dataset, model, 42))
        if result is None:
            result = train_cell(arm, dataset, model, 42,
                                with_penalizer(entry, pen), sub, split_seed)
        ctd = float(result["metrics"]["val"]["ctd"])
        val[pen] = -math.inf if math.isnan(ctd) else ctd
    chosen = max(PENALIZER_GRID, key=val.get)
    print(f"{arm}/{dataset}/{model} split {split_seed}: penalizer {chosen:g}, "
          f"val ctd {val[chosen]:.6f}")
    return with_penalizer(entry, chosen)


def cell_entry(arm, dataset, model, entry, out_dir, split_seed):
    if split_seed is not None and model == "cox-summary":
        return select_penalizer(arm, dataset, model, entry, out_dir, split_seed)
    return entry


def main():
    registry.load_plugins()
    ap = argparse.ArgumentParser()
    registry.add_plugin_argument(ap)
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--dataset", choices=list(all_datasets()), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42, choices=list(SEEDS))
    ap.add_argument("--split-seed", type=int)
    ap.add_argument("--out-dir")
    ap.add_argument("--defaults", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir or runs_root(args.split_seed)

    hparams = load_hparams()
    key = f"{args.arm}/{args.dataset}/{args.model}"
    if args.defaults:
        entry = {}
    elif key in hparams:
        entry = hparams[key]
    else:
        available = sorted(m for _a, _d, m in cells(
            hparams, (args.arm,), args.dataset))
        ap.error(f"configs/hparams.yaml has no entry for {key}. Models for {args.arm}/{args.dataset}: {available}. "
                 f"Use --defaults to train at default hyperparameters.")

    if not args.defaults:
        entry = cell_entry(args.arm, args.dataset, args.model, entry, out_dir,
                           args.split_seed)
    result = train_cell(args.arm, args.dataset, args.model, args.seed,
                        entry, out_dir, args.split_seed)
    test = result.get("test") or {}
    flat = test.get("mean") or test.get("risk1") or {}
    print(f"{key} seed{args.seed}: "
          + "  ".join(f"{m}={flat[m]:.6f}" for m in ("ctd", "ibs", "ibll")
                      if isinstance(flat.get(m), float)))


if __name__ == "__main__":
    main()
