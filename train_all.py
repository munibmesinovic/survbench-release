import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.baselines._runtime import (ARMS, IN_DOMAIN_ARMS, SEEDS,
                                       all_datasets, cells, load_hparams,
                                       run_path, runs_root)
from survbench import registry
from train import cell_entry, read_run, train_cell


def default_cells(arms, dataset=None, model=None):
    models = [model] if model else registry.default_models()
    for ds, spec in registry.DATASETS.items():
        if dataset and ds != dataset:
            continue
        for arm in arms:
            if arm in spec["configs"]:
                for m in models:
                    yield arm, ds, m


def main():
    registry.load_plugins()
    ap = argparse.ArgumentParser()
    registry.add_plugin_argument(ap)
    ap.add_argument("--arm", choices=[*ARMS, "both", "all"], default="both")
    ap.add_argument("--dataset", choices=list(all_datasets()))
    ap.add_argument("--model")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--split-seed", type=int)
    ap.add_argument("--out-dir")
    ap.add_argument("--defaults", action="store_true")
    args = ap.parse_args()

    if args.arm == "both":
        arms = IN_DOMAIN_ARMS
    elif args.arm == "all":
        arms = ARMS
    else:
        arms = (args.arm,)
    out_dir = Path(args.out_dir) if args.out_dir else runs_root(args.split_seed)
    hparams = {} if args.defaults else load_hparams()
    if args.defaults:
        todo_cells = list(default_cells(arms, args.dataset, args.model))
    else:
        todo_cells = list(cells(hparams, arms, args.dataset, args.model))
    failed = []
    for arm, dataset, model in todo_cells:
        key = f"{arm}/{dataset}/{model}"
        entry = hparams.get(key, {})
        if args.seeds:
            seeds = args.seeds
        elif args.split_seed is not None:
            seeds = [42]
        else:
            seeds = entry.get("seeds") or list(SEEDS)
        todo = [s for s in seeds
                if read_run(run_path(out_dir, arm, dataset, model, s)) is None]
        if not todo:
            print(f"{key}: done")
            continue
        if not args.defaults:
            try:
                entry = cell_entry(arm, dataset, model, entry, out_dir,
                                   args.split_seed)
            except Exception:
                traceback.print_exc()
                failed.append(key)
                continue
        for seed in todo:
            print(f"{key} seed{seed}")
            try:
                train_cell(arm, dataset, model, seed, entry, out_dir,
                           args.split_seed)
            except Exception:
                traceback.print_exc()
                failed.append(f"{key} seed{seed}")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
