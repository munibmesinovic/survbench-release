from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from models.baselines._data import (
    dataset_of,
    load_split,
    patch_scipy_simps,
    prefix_of,
    remove_constant_cols,
    static_input_kwargs,
)
from models.baselines._runtime import (
    WallClock,
    config_hash,
    git_commit,
    reject_unexpected_kwargs as _reject_unexpected_kwargs,
    RISK_AXIS_KEY,
    risk_axis_of,
    save_checkpoint,
    save_predictions,
)

patch_scipy_simps()

from lifelines import CoxPHFitter
from models.baselines._eval import (
    _pool_landmark_means, compute_competing_risks_metrics, compute_metrics,
    evaluation_times, landmark_key, landmark_strata, rebased_horizon,
)


_CS_KEY = "risk{code}_causespecific"


def _mean_over_causes(block: dict, codes: Sequence[int],
                      key_template: str = _CS_KEY) -> dict:
    valid = [key_template.format(code=code) for code in codes
             if not np.isnan(block.get(key_template.format(code=code), {})
                             .get("ctd", float("nan")))]
    auc_vals = [block[k].get("auc_integrated", float("nan")) for k in valid
                if not np.isnan(block[k].get("auc_integrated", float("nan")))]
    from collections import defaultdict as _dd
    _aat_pool = _dd(list)
    for risk_key in valid:
        for t_key, val in (block[risk_key].get("auc_at_times") or {}).items():
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            _aat_pool[t_key].append(float(val))
    return {
        "ctd": float(np.mean([block[k]["ctd"] for k in valid]))
        if valid else float("nan"),
        "ibs": float(np.mean([block[k]["ibs"] for k in valid]))
        if valid else float("nan"),
        "ibll": float(np.mean([block[k]["ibll"] for k in valid]))
        if valid else float("nan"),
        "auc_integrated": float(np.mean(auc_vals)) if auc_vals else float("nan"),
        "auc_mean": float(np.mean(auc_vals)) if auc_vals else float("nan"),
        "auc_at_times": {k: float(np.mean(v)) for k, v in _aat_pool.items()},
    }


def _landmark_block(split_results: dict, landmark) -> dict:
    if landmark is None:
        return split_results
    return split_results.setdefault(landmark_key(landmark), {})


def evaluate_survival(
    surv_df: pd.DataFrame,
    durations: np.ndarray,
    events_binary: np.ndarray,
    *,
    train_durations: np.ndarray | None = None,
    train_events_bin: np.ndarray | None = None,
    horizon: float | None = None,
    auc_times: Sequence[float] | None = None,
) -> dict:
    return compute_metrics(
        surv_df, durations, events_binary,
        train_durations=train_durations,
        train_events_bin=train_events_bin,
        horizon=horizon,
        auc_times=auc_times,
    )


def _fit_coxph(
    x_train: np.ndarray,
    dur_train: np.ndarray,
    evt_binary: np.ndarray,
    penalizer: float = 0.01,
):
    n_features = x_train.shape[1]
    col_names = [f"f{i}" for i in range(n_features)]
    df_train = pd.DataFrame(x_train, columns=col_names)
    df_train["duration"] = dur_train
    df_train["event"] = evt_binary

    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    cph.fit(df_train, duration_col="duration", event_col="event",
            show_progress=True)
    return cph, col_names


def _coxph_survival(
    model,
    x_eval: np.ndarray,
    col_names: list[str],
    time_grid: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame(x_eval, columns=col_names)
    return model.predict_survival_function(df, times=time_grid)


def _stack_surv_by_cause(split_arrays, codes):
    if split_arrays is None:
        return None, []
    if isinstance(split_arrays, pd.DataFrame):
        return split_arrays.values.T, list(codes)

    present = [c for c in codes if f"risk{c}" in split_arrays]
    if not present:
        return None, []
    mats = [split_arrays[f"risk{c}"].values.T for c in present]
    if len(codes) == 1:
        return mats[0], present
    return np.stack(mats, axis=1), present


def _cuts_of(split_arrays, codes):
    if split_arrays is None:
        return np.array([])
    if isinstance(split_arrays, pd.DataFrame):
        return np.asarray(split_arrays.index, dtype=float)
    for c in codes:
        key = f"risk{c}"
        if key in split_arrays:
            return np.asarray(split_arrays[key].index, dtype=float)
    return np.array([])


def run_single_risk(
    x_train, dur_train, evt_train,
    x_val, dur_val, evt_val,
    x_test, dur_test, evt_test,
    penalizer: float = 0.01,
    horizon: float | None = None,
    dataset: str | None = None,
    landmarks_val: np.ndarray | None = None,
    landmarks_test: np.ndarray | None = None,
    skip_test: bool = False,
) -> tuple:
    n_features = x_train.shape[1]

    print(f"  Fitting Cox PH: {len(x_train)} patients, {n_features} features, "
          f"penalizer={penalizer} ...")
    t0 = time.time()
    model, col_names = _fit_coxph(
        x_train, dur_train, (evt_train > 0).astype(int), penalizer
    )
    print(f"  Fitted in {time.time() - t0:.1f}s  "
          f"(concordance_train={model.concordance_index_:.4f})")

    results: dict = {}
    surv_arrays: dict[str, pd.DataFrame] = {}
    scored = [("val", x_val, dur_val, evt_val, landmarks_val)]
    if not skip_test:
        scored.append(("test", x_test, dur_test, evt_test, landmarks_test))
    for split_name, x_s, dur_s, evt_s, lm_s in scored:
        time_grid = np.linspace(dur_s.min(), dur_s.max(), 100)
        surv = _coxph_survival(model, x_s, col_names, time_grid)
        surv_arrays[split_name] = surv
        results[split_name] = {}
        strata = landmark_strata(lm_s, len(dur_s))
        for landmark, rows in strata:
            metrics = evaluate_survival(
                surv.loc[:, rows], dur_s[rows], (evt_s[rows] > 0).astype(int),
                train_durations=dur_train,
                train_events_bin=(evt_train > 0).astype(int),
                horizon=rebased_horizon(horizon, landmark),
                auc_times=evaluation_times(horizon, landmark, dataset),
            )
            _landmark_block(results[split_name], landmark)["risk1"] = metrics
            label = "" if landmark is None else f" {landmark_key(landmark)}"
            print(f"  {split_name}{label}: Ctd={metrics['ctd']:.4f}  "
                  f"IBS={metrics['ibs']:.4f}  IBLL={metrics['ibll']:.4f}"
                  f"  AUC={metrics['auc_integrated']:.4f}")
        pooled = [landmark for landmark, _ in strata if landmark is not None]
        if pooled:
            for landmark in pooled:
                block = results[split_name][landmark_key(landmark)]
                block["mean"] = dict(block["risk1"])
            results[split_name]["mean"] = _pool_landmark_means(
                results[split_name], pooled)

    results["model_info"] = {
        "type": "CoxPH",
        "penalizer": penalizer,
        "n_features": n_features,
        "train_concordance": float(model.concordance_index_),
    }
    return results, model, col_names, surv_arrays


_SURV_FLOOR = 1e-12


def aalen_johansen_cif(surv_by_cause, diagnostics=None):
    if not surv_by_cause:
        raise ValueError(
            "aalen_johansen_cif needs at least one cause")

    codes = sorted(surv_by_cause)
    mats: dict = {}
    shape = None
    for code in codes:
        mat = np.asarray(surv_by_cause[code], dtype=float)
        if shape is None:
            shape = mat.shape
        elif mat.shape != shape:
            raise ValueError(
                f"all causes must share one shape and grid: cause {codes[0]} has {shape}, cause {code} has {mat.shape}")
        mats[code] = mat


    dlam = {code: np.clip(np.diff(-np.log(np.clip(mats[code], _SURV_FLOOR, 1.0)),
                                  axis=0, prepend=0.0), 0.0, None)
            for code in codes}

    total = sum(dlam.values())
    hot = total >= 1.0


    if diagnostics is not None:
        diagnostics["n_saturated_cells"] = int(np.count_nonzero(hot))
        diagnostics["n_cells"] = int(total.size)
        diagnostics["max_bin_hazard"] = float(total.max()) if total.size else 0.0
    if np.any(hot):
        scale = np.ones_like(total)
        scale[hot] = (1.0 - _SURV_FLOOR) / total[hot]
        dlam = {code: d * scale for code, d in dlam.items()}
        total = total * scale

    overall = np.cumprod(1.0 - total, axis=0)
    prior = np.concatenate([np.ones((1,) + shape[1:]), overall[:-1]], axis=0)
    cif = {code: np.cumsum(prior * dlam[code], axis=0) for code in codes}
    return cif, overall


def _cif_refusal_reason(codes, surv_on_cuts_for_split, cuts):
    if cuts is None or len(np.asarray(cuts)) == 0:
        return ("no bin grid supplied; the CIF needs the dataset cuts")
    missing = [int(c) for c in codes if c not in surv_on_cuts_for_split]
    if missing:
        return (f"cause(s) {missing} of {[int(c) for c in codes]} not fitted or not scored on this split (too few events)")
    return None


def run_competing_risk(
    x_train, dur_train, evt_train,
    x_val, dur_val, evt_val,
    x_test, dur_test, evt_test,
    num_risks: int,
    cause_codes: list | None = None,
    penalizer: float = 0.01,
    horizon: float | None = None,
    dataset: str | None = None,
    landmarks_val: np.ndarray | None = None,
    landmarks_test: np.ndarray | None = None,
    skip_test: bool = False,
    cuts: np.ndarray | None = None,
) -> tuple:
    n_features = x_train.shape[1]
    scored_splits = ("val",) if skip_test else ("val", "test")
    results: dict = {s: {} for s in scored_splits}
    strata = {"val": landmark_strata(landmarks_val, len(dur_val))}
    if not skip_test:
        strata["test"] = landmark_strata(landmarks_test, len(dur_test))
    models_dict: dict = {}
    col_names_dict: dict = {}
    surv_arrays: dict[str, dict] = {s: {} for s in scored_splits}


    surv_on_cuts: dict[str, dict] = {s: {} for s in scored_splits}


    raw_events = {"val": evt_val} if skip_test else {"val": evt_val,
                                                     "test": evt_test}
    raw_durations = {"val": dur_val} if skip_test else {"val": dur_val,
                                                        "test": dur_test}


    codes = list(cause_codes) if cause_codes else list(range(1, num_risks + 1))
    for r in codes:
        print(f"\n  --- Risk k={r} ({codes.index(r) + 1}/{len(codes)}) ---")
        evt_train_r = (evt_train == r).astype(int)
        evt_val_r = (evt_val == r).astype(int)


        evt_test_r = None if skip_test else (evt_test == r).astype(int)

        n_events = evt_train_r.sum()
        print(f"  Training events for risk {r}: {n_events}/{len(evt_train_r)} "
              f"({n_events / len(evt_train_r) * 100:.1f}%)")

        if n_events < 10:
            print(f"  Skipping risk {r}: too few events ({n_events})")
            for s in scored_splits:
                for landmark, _rows in strata[s]:
                    _landmark_block(results[s], landmark)[_CS_KEY.format(code=r)] = {
                        "ctd": float("nan"), "ibs": float("nan"),
                        "ibll": float("nan"),
                    }
            continue

        t0 = time.time()
        model_r, col_names = _fit_coxph(x_train, dur_train, evt_train_r, penalizer)
        models_dict[f"risk{r}"] = model_r
        col_names_dict[f"risk{r}"] = col_names
        print(f"  Fitted in {time.time() - t0:.1f}s  "
              f"(concordance_train={model_r.concordance_index_:.4f})")

        scored = [("val", x_val, dur_val, evt_val_r)]
        if not skip_test:
            scored.append(("test", x_test, dur_test, evt_test_r))
        for split_name, x_s, dur_s, evt_s_r in scored:
            if evt_s_r.sum() < 5:
                print(f"  {split_name} risk{r}: skipped (only {evt_s_r.sum()} events)")
                for landmark, _rows in strata[split_name]:
                    block = _landmark_block(results[split_name], landmark)
                    block[_CS_KEY.format(code=r)] = {
                        "ctd": float("nan"), "ibs": float("nan"),
                        "ibll": float("nan"),
                    }
                continue
            time_grid = np.linspace(dur_s.min(), dur_s.max(), 100)
            surv = _coxph_survival(model_r, x_s, col_names, time_grid)
            surv_arrays[split_name][f"risk{r}"] = surv
            if cuts is not None:


                surv_on_cuts[split_name][r] = np.asarray(
                    _coxph_survival(model_r, x_s, col_names,
                                    np.asarray(cuts, dtype=float)),
                    dtype=float)
            for landmark, rows in strata[split_name]:


                metrics = evaluate_survival(
                    surv.loc[:, rows], dur_s[rows], evt_s_r[rows],
                    train_durations=dur_train,
                    train_events_bin=(evt_train == r).astype(int),
                    horizon=rebased_horizon(horizon, landmark),
                    auc_times=evaluation_times(horizon, landmark, dataset),
                )
                _landmark_block(results[split_name],
                                landmark)[_CS_KEY.format(code=r)] = metrics
                label = "" if landmark is None else f" {landmark_key(landmark)}"
                print(f"  {split_name}{label} risk{r}: "
                      f"Ctd={metrics['ctd']:.4f}  "
                      f"IBS={metrics['ibs']:.4f}  IBLL={metrics['ibll']:.4f}"
                      f"  AUC={metrics['auc_integrated']:.4f}")


    for split_name in scored_splits:
        reason = _cif_refusal_reason(codes, surv_on_cuts[split_name], cuts)
        if reason is not None:
            results[split_name]["cif_status"] = {"emitted": False,
                                                 "reason": reason}
            print(f"  {split_name}: no CIF ({reason})")
            continue

        saturation: dict = {}
        cif, _overall = aalen_johansen_cif(surv_on_cuts[split_name],
                                           diagnostics=saturation)
        cif_times = np.asarray(cuts, dtype=float)
        dur_s = raw_durations[split_name]
        evt_s = raw_events[split_name]
        for landmark, rows in strata[split_name]:
            block = _landmark_block(results[split_name], landmark)
            for r in codes:
                block[f"risk{r}"] = compute_competing_risks_metrics(
                    cif=cif[r][:, rows],
                    cif_times=cif_times,
                    durations=dur_s[rows],
                    events=evt_s[rows],
                    cause=r,
                    times=evaluation_times(horizon, landmark, dataset),
                    horizon=rebased_horizon(horizon, landmark),
                )
                label = "" if landmark is None else f" {landmark_key(landmark)}"
                print(f"  {split_name}{label} risk{r} (CIF): "
                      f"Ctd={block[f'risk{r}']['ctd']:.4f}  "
                      f"IBS={block[f'risk{r}']['ibs']:.4f}")
        results[split_name]["cif_status"] = {
            "emitted": True,
            "reason": None,
            "causes": [int(c) for c in codes],
            "grid": "dataset bin cuts (cif_times), as passed to the neural runners",


            "saturation": saturation,
        }
        if saturation.get("n_saturated_cells"):
            print(f"  {split_name}: {saturation['n_saturated_cells']} of "
                  f"{saturation['n_cells']} (subject, bin) cells saturated "
                  f"(max bin hazard {saturation['max_bin_hazard']:.3f})")

    for split_name in scored_splits:
        emitted = results[split_name].get("cif_status", {}).get("emitted", False)
        for landmark, _rows in strata[split_name]:
            block = _landmark_block(results[split_name], landmark)


            block["mean_causespecific"] = _mean_over_causes(block, codes, _CS_KEY)
            if emitted:
                block["mean"] = _mean_over_causes(block, codes, "risk{code}")
        pooled = [landmark for landmark, _ in strata[split_name]
                  if landmark is not None]
        if pooled:
            results[split_name]["mean"] = _pool_landmark_means(
                results[split_name], pooled)

    results["model_info"] = {
        "type": "CauseSpecificCoxPH",
        "penalizer": penalizer,
        "n_features": n_features,
        "num_risks": num_risks,
    }
    return results, models_dict, col_names_dict, surv_arrays


def run(
    config: dict,
    *,
    penalizer: float = 0.01,
    input_mode: str = "static",
    seed: int = 42,
    skip_test: bool = False,
    save_dir: Optional[Path] = None,
    save_tag: Optional[str] = None,
    **_ignored,
) -> dict:
    _reject_unexpected_kwargs(_ignored)
    model_name = "cox"

    prefix = prefix_of(config)
    dataset = dataset_of(config)
    horizon = float(config.get("preprocessing", {}).get("max_horizon_hours", 240))


    train = load_split(config, "train", **static_input_kwargs(input_mode))
    val = load_split(config, "val", **static_input_kwargs(input_mode))


    test = None if skip_test else load_split(config, "test", **static_input_kwargs(input_mode))

    modality_info = train["modality_info"]
    competing = bool(modality_info.get("competing_risks", False))
    num_risks = int(modality_info.get("n_events", 1))


    codes = risk_axis_of(modality_info)

    x_train, dur_train, evt_train = train["x"], train["durations"], train["events"]
    x_val, dur_val, evt_val = val["x"], val["durations"], val["events"]
    if test is None:
        x_test = dur_test = evt_test = None
        landmarks_test = None
    else:
        x_test, dur_test, evt_test = test["x"], test["durations"], test["events"]
        landmarks_test = test["landmarks"]

    n_train = len(dur_train)
    n_val = len(dur_val)


    n_test = None if test is None else len(dur_test)
    n_events_train = int((evt_train > 0).sum())
    n_events_val = int((evt_val > 0).sum())
    n_events_test = None if test is None else int((evt_test > 0).sum())

    print(f"  Shapes: train={x_train.shape} val={x_val.shape} "
          f"test={'not loaded' if x_test is None else x_test.shape}")
    print(f"  Event rate (train): {(evt_train > 0).mean() * 100:.1f}%")


    x_train, x_val, x_test, keep_mask = remove_constant_cols(
        x_train, x_val, x_test
    )
    n_dropped = int((~keep_mask).sum())
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} constant features "
              f"-> {x_train.shape[1]} remaining")

    print(f"\n{'=' * 65}")
    print(f"  Cox PH baseline: {prefix.upper()} ({num_risks} risk{'s' if num_risks > 1 else ''})")
    print(f"{'=' * 65}")


    with WallClock() as wc:
        if competing and num_risks > 1:
            inner, models_dict, col_names_dict, surv_arrays = run_competing_risk(
                x_train, dur_train, evt_train,
                x_val, dur_val, evt_val,
                x_test, dur_test, evt_test,
                num_risks=num_risks,
                cause_codes=codes,
                penalizer=penalizer,
                horizon=horizon,
                dataset=dataset,
                landmarks_val=val["landmarks"],
                landmarks_test=landmarks_test,
                skip_test=skip_test,


                cuts=val["cuts"],
            )

            _param_count = 0
        else:
            inner, fitted_model, col_names, surv_arrays = run_single_risk(
                x_train, dur_train, evt_train,
                x_val, dur_val, evt_val,
                x_test, dur_test, evt_test,
                penalizer=penalizer,
                horizon=horizon,
                dataset=dataset,
                landmarks_val=val["landmarks"],
                landmarks_test=landmarks_test,
                skip_test=skip_test,
            )
            models_dict = {"risk1": fitted_model}
            _param_count = 0

    wall_time_seconds = wc.seconds


    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = save_tag if save_tag is not None else f"{model_name}_{prefix}_seed{seed}"


        surv_val_np, codes_val = _stack_surv_by_cause(surv_arrays.get("val"), codes)
        surv_test_np, codes_test = _stack_surv_by_cause(surv_arrays.get("test"), codes)
        cuts_val = _cuts_of(surv_arrays.get("val"), codes)
        cuts_test = _cuts_of(surv_arrays.get("test"), codes)


        cuts = cuts_val


        saved_codes = codes_val or codes_test or codes
        if codes_val and codes_test and codes_val != codes_test:
            raise ValueError(
                f"val stacked causes {codes_val} but test stacked {codes_test}; one npz cannot hold two risk axes")
        if list(saved_codes) != list(codes):
            print(f"  risk axis written as {saved_codes}, not {codes}; missing causes had too few events")


        save_predictions(
            save_dir / f"{tag}_preds.npz",
            cuts=cuts,
            cause_codes=saved_codes,
            surv_val=surv_val_np,
            surv_test=surv_test_np,
            cuts_val=cuts_val if len(cuts_val) else None,
            cuts_test=cuts_test if len(cuts_test) else None,
        )


        obj_to_save = (models_dict if len(models_dict) > 1
                       else models_dict.get("risk1", models_dict))
        save_checkpoint(save_dir / f"{tag}.pkl", obj_to_save)


    val_metrics = inner.get("val", {})
    test_metrics = inner.get("test", {})


    def _flat_metrics(split_dict: dict) -> dict:
        if "risk1" in split_dict and "mean" not in split_dict:
            return split_dict["risk1"]
        if "mean" in split_dict:
            return split_dict["mean"]
        return split_dict


    _fitted = models_dict.get("risk1") or next(iter(models_dict.values()), None)
    if _fitted is None:


        effective_hparams: dict = {"penalizer": None, "l1_ratio": None,
                                   "n_features": None}
    else:
        effective_hparams = {
            "penalizer": float(_fitted.penalizer),
            "l1_ratio": float(_fitted.l1_ratio),
            "n_features": int(len(_fitted.params_)),
        }


    _repo_root = Path(__file__).resolve().parents[2]

    result: dict = {

        "model": model_name,
        "dataset": config.get("dataset", {}).get("name", prefix),
        "config_hash": config_hash(config),
        "git_commit": git_commit(_repo_root),
        "seed": seed,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_events_train": n_events_train,
        "n_events_val": n_events_val,
        "n_events_test": n_events_test,
        "horizon_hours": horizon,
        "n_time_bins": None,
        "training": {
            "optimiser": "lifelines",
            "lr": None,
            "batch_size": None,
            "epochs_run": None,
            "early_stop_epoch": None,
            "early_stop_patience": None,
            "early_stop_metric": None,
            "wall_time_seconds": wall_time_seconds,
            "param_count": _param_count,
            "device": "cpu",
        },
        "metrics": {"val": _flat_metrics(val_metrics)},


        "effective_hparams": effective_hparams,
        "notes": "",


        RISK_AXIS_KEY: codes,
        "input_mode": input_mode,

        "val": inner.get("val", {}),
        "model_info": inner.get("model_info", {}),
    }


    if not skip_test:
        result["metrics"]["test"] = _flat_metrics(test_metrics)
        result["test"] = inner.get("test", {})
    return result
