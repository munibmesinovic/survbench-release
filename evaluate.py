import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.baselines import _checkpoint, _data
from models.baselines._runtime import (ALL_DATASETS, ARMS, IN_DOMAIN_ARMS,
                                       SEEDS, _set_seed, cells,
                                       checkpoint_path, config_path,
                                       is_summary, load_hparams,
                                       registry_key)

PREDICTORS = {
    "cox": _checkpoint._predict_cox,
    "deephit": _checkpoint._predict_deephit,
    "dynamic_deephit": _checkpoint._predict_dynamic_deephit,
    "dysurv": _checkpoint._predict_dysurv,
    "transformer": _checkpoint._predict_transformer,
}


def evaluate_cell(arm, dataset, model, seed, entry, device="cpu"):
    config = _data.load_config(str(config_path(arm, dataset)))
    ckpt = checkpoint_path(arm, dataset, model, seed)
    if not ckpt.exists():
        return None
    _set_seed(seed)
    shipped = {"effective_hparams": entry.get("effective_hparams") or {}}
    if entry.get("param_count") is not None:
        shipped["training"] = {"param_count": entry["param_count"]}
    base = registry_key(model)
    common = {"config": config, "checkpoint": ckpt, "shipped_json": shipped}
    if base == "cox":
        payload = PREDICTORS[base](seed=seed, **common)
    elif base in ("deephit", "dynamic_deephit"):
        payload = PREDICTORS[base](device=device, **common)
    elif base == "dysurv":
        payload = PREDICTORS[base](device=device, seed=seed, **common)
    else:
        modalities = entry.get("modalities") or ["dynamic", "static"]
        payload = PREDICTORS[base](device=device, seed=seed,
                                   modalities=tuple(modalities), **common)
    return payload["metrics_test"]


def flatten(block, prefix=""):
    out = {}
    if not isinstance(block, dict):
        return out
    for key, value in block.items():
        name = f"{prefix}{key}"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value)
        elif isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
    return out


def aggregate(per_seed):
    flat = {s: flatten(b) for s, b in per_seed.items()}
    out = {}
    for k in sorted({k for f in flat.values() for k in f}):
        vals = [f[k] for f in flat.values() if k in f and np.isfinite(f[k])]
        if not vals:
            continue
        out[k] = {"mean": float(np.mean(vals)),
                  "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                  "n_seeds": len(vals)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=[*ARMS, "both", "all"], default="both")
    ap.add_argument("--dataset", choices=list(ALL_DATASETS))
    ap.add_argument("--model")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="metrics")
    args = ap.parse_args()

    if args.arm == "both":
        arms = IN_DOMAIN_ARMS
    elif args.arm == "all":
        arms = ARMS
    else:
        arms = (args.arm,)
    hparams = load_hparams()
    results = {}
    for arm, dataset, model in cells(hparams, arms, args.dataset, args.model):
        if registry_key(model) not in PREDICTORS or is_summary(model):
            print(f"{arm}/{dataset}/{model}: no checkpoint predictor, skipped")
            continue
        entry = hparams[f"{arm}/{dataset}/{model}"]
        per_seed = {}
        for seed in args.seeds:
            block = evaluate_cell(arm, dataset, model, seed, entry,
                                  device=args.device)
            if block is not None:
                per_seed[seed] = block
        if not per_seed:
            continue
        results.setdefault(arm, {}).setdefault(dataset, {})[model] = {
            "per_seed": {str(s): b for s, b in per_seed.items()},
            "aggregate": aggregate(per_seed),
        }
        print(f"{arm}/{dataset}/{model}: {len(per_seed)} seed(s)")

    with open(f"{args.out}.json", "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    rows = ["arm,dataset,model,metric,mean,sd,n_seeds"]
    for arm in sorted(results):
        for ds in sorted(results[arm]):
            for mdl in sorted(results[arm][ds]):
                agg = results[arm][ds][mdl]["aggregate"]
                for metric in sorted(agg):
                    a = agg[metric]
                    rows.append(f"{arm},{ds},{mdl},{metric},"
                                f"{a['mean']!r},{a['sd']!r},{a['n_seeds']}")
    with open(f"{args.out}.csv", "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"wrote {args.out}.json and {args.out}.csv")


if __name__ == "__main__":
    main()
