import argparse
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.baselines._runtime import (ARMS, IN_DOMAIN_ARMS, SEEDS, _set_seed,
                                       all_datasets, cell_config, cells,
                                       checkpoint_path, is_summary,
                                       load_hparams, registry_key, run_path)
from survbench import registry


def has_predictor(model):
    return (registry.model_predictor(registry_key(model)) is not None
            and not is_summary(model))


def evaluate_cell(arm, dataset, model, seed, entry, device="cpu", root=None,
                  split_seed=None):
    config = cell_config(arm, dataset, split_seed)
    ckpt = checkpoint_path(arm, dataset, model, seed, root)
    if not ckpt.exists():
        return None
    _set_seed(seed)
    shipped = {"effective_hparams": entry.get("effective_hparams") or {}}
    if entry.get("param_count") is not None:
        shipped["training"] = {"param_count": entry["param_count"]}
    predictor = registry.model_predictor(registry_key(model))
    optional = {"device": device, "seed": seed,
                "modalities": tuple(entry.get("modalities") or ["dynamic", "static"])}
    accepted = inspect.signature(predictor).parameters
    payload = predictor(config=config, checkpoint=ckpt, shipped_json=shipped,
                        **{k: v for k, v in optional.items() if k in accepted})
    return payload["metrics_test"]


def evaluate_run(arm, dataset, model, seed, runs):
    path = run_path(runs, arm, dataset, model, seed)
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)["test"]


def run_cells(runs, arms, dataset=None, model=None):
    names = "|".join(re.escape(d) for d in sorted(all_datasets(), key=len, reverse=True))
    pattern = re.compile(rf"^(.+?)_({names})_seed(\d+)\.json$")
    found = set()
    for arm in arms:
        for path in (Path(runs) / arm).glob("*_seed*.json"):
            m = pattern.match(path.name)
            if m is None:
                continue
            if (dataset and m[2] != dataset) or (model and m[1] != model):
                continue
            found.add((arm, m[2], m[1]))
    return sorted(found)


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
    registry.load_plugins()
    ap = argparse.ArgumentParser()
    registry.add_plugin_argument(ap)
    ap.add_argument("--arm", choices=[*ARMS, "both", "all"], default="both")
    ap.add_argument("--dataset", choices=list(all_datasets()))
    ap.add_argument("--model")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs")
    ap.add_argument("--checkpoints")
    ap.add_argument("--split-seed", type=int)
    ap.add_argument("--out", default="metrics")
    args = ap.parse_args()
    if args.runs and args.checkpoints:
        ap.error("use --runs or --checkpoints, not both")

    if args.arm == "both":
        arms = IN_DOMAIN_ARMS
    elif args.arm == "all":
        arms = ARMS
    else:
        arms = (args.arm,)
    hparams = load_hparams()
    if args.runs or args.checkpoints:
        todo = run_cells(args.runs or args.checkpoints, arms, args.dataset,
                         args.model)
    else:
        todo = list(cells(hparams, arms, args.dataset, args.model))
    results = {}
    for arm, dataset, model in todo:
        if not args.runs and not has_predictor(model):
            print(f"{arm}/{dataset}/{model}: no checkpoint predictor, skipped")
            continue
        entry = hparams.get(f"{arm}/{dataset}/{model}") or {}
        per_seed = {}
        for seed in args.seeds or entry.get("seeds") or SEEDS:
            if args.runs:
                block = evaluate_run(arm, dataset, model, seed, args.runs)
            else:
                block = evaluate_cell(arm, dataset, model, seed, entry,
                                      device=args.device,
                                      root=args.checkpoints,
                                      split_seed=args.split_seed)
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
