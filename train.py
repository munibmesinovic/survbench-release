import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.baselines import _data
from models.baselines._runtime import (ALL_DATASETS, ARMS, SEEDS, _set_seed,
                                       cells, config_path, file_model_name,
                                       input_mode_of, invoke, is_multimodal,
                                       is_summary, load_hparams, registry_key,
                                       runner_kwargs)

DEFAULT_MODALITIES = ("dynamic", "static")
MULTIMODAL_MODALITIES = ("dynamic", "static", "icd", "radiology")


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
    return MULTIMODAL_MODALITIES if is_multimodal(model) else DEFAULT_MODALITIES


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


def train_cell(arm, dataset, model, seed, entry, out_dir):
    config = _data.load_config(str(config_path(arm, dataset)))
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

    _set_seed(seed)
    result = invoke(model, config, **kwargs)
    assert_echo_matches(model, winner, result.get("effective_hparams"))
    if input_mode is not None and result.get("input_mode") != input_mode:
        raise AssertionError(
            f"{model} was asked for input_mode={input_mode!r} and reported "
            f"{result.get('input_mode')!r}.")
    stamp_arm(result, config)
    if is_summary(model):
        result["model"] = model
    else:
        result["model"] = file_model_name(registry_key(model),
                                          is_multimodal(model))
    result["seed"] = seed
    with open(save_dir / f"{tag}.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--dataset", choices=list(ALL_DATASETS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42, choices=list(SEEDS))
    ap.add_argument("--out-dir", default=str(ROOT / "runs"))
    ap.add_argument("--defaults", action="store_true")
    args = ap.parse_args()

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

    result = train_cell(args.arm, args.dataset, args.model, args.seed,
                        entry, args.out_dir)
    test = result.get("test") or {}
    flat = test.get("mean") or test.get("risk1") or {}
    print(f"{key} seed{args.seed}: "
          + "  ".join(f"{m}={flat[m]:.6f}" for m in ("ctd", "ibs", "ibll")
                      if isinstance(flat.get(m), float)))


if __name__ == "__main__":
    main()
