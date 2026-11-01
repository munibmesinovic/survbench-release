import argparse
import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd
import yaml
from pycox.evaluation import EvalSurv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.baselines._data import (load_split, remove_constant_cols,
                                    static_input_kwargs)
from models.baselines._eval import _cause_specific_ctd
from models.baselines._runtime import (SEEDS, cell_config, input_mode_of,
                                       read_risk_axis, registry_key, run_path,
                                       runs_root)
from survbench import registry
from models.baselines.cox import _coxph_survival, aalen_johansen_cif

GROUP = {}


def clusters(subject_ids):
    subjects, inverse = np.unique(np.asarray(subject_ids), return_inverse=True)
    inverse = inverse.astype(np.int64).ravel()
    order = np.argsort(inverse, kind="stable")
    sizes = np.bincount(inverse, minlength=len(subjects))
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    return order, offsets, sizes


def resample(group, b):
    order, offsets, sizes = group["clusters"]
    n = len(sizes)
    rng = np.random.default_rng(
        np.random.SeedSequence([group["master_seed"], group["index"], b]))
    draw = rng.integers(0, n, size=n)
    counts = sizes[draw]
    first = np.cumsum(counts) - counts
    within = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(first, counts)
    return order[np.repeat(offsets[draw], counts) + within]


def cox_design(config, model, designs):
    mode = input_mode_of(model)
    if mode not in designs:
        kw = static_input_kwargs(mode)
        train, val, test = (load_split(config, s, **kw)
                            for s in ("train", "val", "test"))
        x_train, _, x_test, _ = remove_constant_cols(train["x"], val["x"],
                                                     test["x"])
        cols = [f"f{i}" for i in range(x_train.shape[1])]
        designs[mode] = (x_test, cols, np.asarray(val["cuts"], dtype=float))
    return designs[mode]


def cox_cif(design, pkl, causes):
    x_test, cols, grid = design
    with open(pkl, "rb") as fh:
        fits = pickle.load(fh)
    surv = {c: np.asarray(_coxph_survival(fits[f"risk{c}"], x_test, cols, grid),
                          dtype=float) for c in causes}
    cif, _ = aalen_johansen_cif(surv)
    return {c: np.ascontiguousarray(cif[c]) for c in causes}, grid


def predictions(config, model, path, designs):
    npz = np.load(path.with_name(path.stem + "_preds.npz"))
    causes = read_risk_axis(npz, artifact=path.name)
    if registry_key(model) == "cox" and len(causes) > 1:
        curves, grid = cox_cif(cox_design(config, model, designs),
                               path.with_suffix(".pkl"), causes)
        return "cif", causes, curves, grid
    surv = npz["surv_test"]
    grid = np.asarray(npz["cuts_test"] if "cuts_test" in npz.files else npz["cuts"],
                      dtype=float)
    curves = {}
    for i, c in enumerate(causes):
        if surv.ndim == 2:
            risk = surv
        elif registry_key(model) == "dysurv":
            risk = surv[:, :, i]
        else:
            risk = surv[:, i, :]
        if len(causes) > 1:
            curves[c] = np.ascontiguousarray((1.0 - risk).T)
        else:
            curves[c] = np.ascontiguousarray(risk)
    return ("cif" if len(causes) > 1 else "surv"), causes, curves, grid


def stored(record, landmarks):
    test = record["test"]
    if landmarks:
        return [float(test[f"L{L:g}"]["mean"]["ctd"]) for L in landmarks]
    block = test["mean"] if "mean" in test else test["risk1"]
    return [float(block["ctd"])]


def ctd(unit, rows, durations, events):
    kind, causes, curves, grid = unit
    d, e = durations[rows], events[rows]
    vals = []
    for c in causes:
        if kind == "cif":
            vals.append(_cause_specific_ctd(curves[c][:, rows], grid, d, e, c)[0])
        elif len(d) == 0 or (e > 0).sum() == 0:
            vals.append(float("nan"))
        else:
            surv = pd.DataFrame(curves[c][rows].T, index=grid)
            vals.append(float(EvalSurv(surv, d, (e > 0).astype(np.int64))
                              .concordance_td()))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def score(group, rows):
    out = {}
    lm = group["landmarks"]
    for key, unit in group["units"].items():
        if lm is None:
            out[key] = [ctd(unit, rows, group["durations"], group["events"])]
        else:
            out[key] = [ctd(unit, rows[lm[rows] == L], group["durations"],
                            group["events"]) for L in group["landmark_list"]]
    return out


def work(span):
    group = GROUP["current"]
    lo, hi = span
    out = {k: np.full((hi - lo, len(group["names"])), np.nan) for k in group["units"]}
    for j, b in enumerate(range(lo, hi)):
        for key, vals in score(group, resample(group, b)).items():
            out[key][j] = vals
    return lo, hi, out


def load_group(name, spec, master_seed, runs, split_seed, seeds):
    arm, dataset = name.split("/")
    config = cell_config(arm, dataset, split_seed)
    test = load_split(config, "test", static_only=True)
    if "landmarks" in spec or test["landmarks"] is None:
        landmark_list = [float(L) for L in spec.get("landmarks", [])]
    else:
        landmark_list = [float(L) for L in np.unique(test["landmarks"])]
    group = {"name": name, "index": int(spec["index"]), "master_seed": master_seed,
             "durations": np.asarray(test["durations"], dtype=np.float64),
             "events": np.asarray(test["events"], dtype=np.int64),
             "landmarks": (None if test["landmarks"] is None
                           else np.asarray(test["landmarks"], dtype=np.float64)),
             "landmark_list": landmark_list,
             "names": [f"L{L:g}" for L in landmark_list] or ["all"],
             "clusters": clusters(test["subject_ids"]),
             "units": {}, "stored": {}, "models": {}}
    designs = {}
    for model, model_seeds in spec["models"].items():
        model_seeds = seeds or model_seeds
        paths = [run_path(runs, arm, dataset, model, s) for s in model_seeds]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"{name}/{model}: {len(missing)} of {len(paths)} runs missing, skipped")
            continue
        group["models"][model] = list(model_seeds)
        for seed, path in zip(model_seeds, paths):
            with open(path) as fh:
                record = json.load(fh)
            group["units"][(model, seed)] = predictions(config, model, path,
                                                        designs)
            group["stored"][(model, seed)] = stored(record, landmark_list)
    return group


def interval(dist):
    if not np.isfinite(dist).all():
        return float("nan"), float("nan")
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return float(lo), float(hi)


def run_group(group, B, workers, chunk):
    GROUP["current"] = group
    spans = [(lo, min(B, lo + chunk)) for lo in range(0, B, chunk)]
    boot = {k: np.full((B, len(group["names"])), np.nan) for k in group["units"]}
    if workers > 1:
        with mp.get_context("fork").Pool(workers, initializer=numba.set_num_threads,
                                         initargs=(1,)) as pool:
            parts = pool.imap_unordered(work, spans)
            for lo, hi, out in parts:
                for k, v in out.items():
                    boot[k][lo:hi] = v
    else:
        for span in spans:
            lo, hi, out = work(span)
            for k, v in out.items():
                boot[k][lo:hi] = v
    return boot


def summarise(group, boot, differences, B):
    cells, diffs = [], []
    seed_mean = {m: np.mean([boot[(m, s)] for s in seeds], axis=0)
                 for m, seeds in group["models"].items()}
    point = {m: np.mean([group["stored"][(m, s)] for s in seeds], axis=0)
             for m, seeds in group["models"].items()}
    for i, name in enumerate(group["names"]):
        landmark = group["landmark_list"][i] if group["landmark_list"] else None
        for m, seeds in group["models"].items():
            lo, hi = interval(seed_mean[m][:, i])
            cells.append({"group": group["name"], "landmark": landmark, "model": m,
                          "seeds": seeds, "point": float(point[m][i]),
                          "ci95_low": lo, "ci95_high": hi, "B": B})
        for d in differences:
            if d["group"] != group["name"]:
                continue
            if landmark is not None and "landmarks" in d and landmark not in d["landmarks"]:
                continue
            a, b = d["a"], d["b"]
            if a not in group["models"] or b not in group["models"]:
                print(f"{group['name']}: {a} - {b} skipped, a model has no runs")
                continue
            dist = seed_mean[a][:, i] - seed_mean[b][:, i]
            lo, hi = interval(dist)
            diffs.append({"group": group["name"], "landmark": landmark, "a": a, "b": b,
                          "point": float(point[a][i] - point[b][i]),
                          "ci95_low": lo, "ci95_high": hi,
                          "share_gt_0": float(np.mean(dist > 0)), "B": B})
    return cells, diffs


def main():
    registry.load_plugins()
    ap = argparse.ArgumentParser()
    registry.add_plugin_argument(ap)
    ap.add_argument("--spec", default=str(ROOT / "configs" / "bootstrap.yaml"))
    ap.add_argument("--runs")
    ap.add_argument("--split-seed", type=int)
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--out", default="bootstrap")
    args = ap.parse_args()

    with open(args.spec) as fh:
        spec = yaml.safe_load(fh)
    runs = Path(args.runs) if args.runs else runs_root(args.split_seed)
    differences = [dict(d, landmarks=[float(L) for L in d["landmarks"]])
                   if "landmarks" in d else d for d in spec["differences"]]
    groups = dict(spec["groups"])
    for name in args.groups or []:
        if name not in groups and not args.models:
            ap.error(f"{name} is not in {args.spec}; pass --models")
    if args.models:
        for name in args.groups or list(groups):
            groups[name] = dict(groups.get(name, {"index": 0}),
                                models={m: list(SEEDS) for m in args.models})
    cells, diffs = [], []
    for name, gspec in groups.items():
        if args.groups and name not in args.groups:
            continue
        group = load_group(name, gspec, int(spec["master_seed"]), runs,
                           args.split_seed, args.seeds)
        if not group["units"]:
            print(f"{name}: no runs, skipped")
            continue
        print(f"{name}: {len(group['units'])} runs, B={args.B}")
        boot = run_group(group, args.B, args.workers, args.chunk)
        c, d = summarise(group, boot, differences, args.B)
        cells += c
        diffs += d

    with open(f"{args.out}.json", "w") as fh:
        json.dump({"master_seed": spec["master_seed"], "cells": cells,
                   "differences": diffs}, fh, indent=1)
    rows = ["group,landmark,model,baseline,point,ci95_low,ci95_high"]
    for c in cells:
        rows.append(f"{c['group']},{'' if c['landmark'] is None else format(c['landmark'], 'g')},"
                    f"{c['model']},,{c['point']!r},{c['ci95_low']!r},{c['ci95_high']!r}")
    for d in diffs:
        rows.append(f"{d['group']},{'' if d['landmark'] is None else format(d['landmark'], 'g')},"
                    f"{d['a']},{d['b']},{d['point']!r},{d['ci95_low']!r},{d['ci95_high']!r}")
    with open(f"{args.out}.csv", "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"wrote {args.out}.json and {args.out}.csv")


if __name__ == "__main__":
    main()
