from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pycox.evaluation import EvalSurv


from survbench import registry
from survbench.core.landmarks import (
    MCMED_HORIZONS_HOURS, MORTALITY_HORIZONS_HOURS,
)

try:
    from sksurv.metrics import cumulative_dynamic_auc
    _SKSURV_OK = True
except ImportError:
    _SKSURV_OK = False


_CLINICAL_GRIDS: dict[float, list[float]] = {
    240.0: [24.0, 72.0, 168.0, 240.0],
    24.0:  [6.0, 12.0, 18.0, 24.0],
}


def _clinical_auc_times(horizon: float) -> list[float]:
    if horizon in _CLINICAL_GRIDS:
        return list(_CLINICAL_GRIDS[horizon])
    return list(np.linspace(0.05 * horizon, 0.95 * horizon, 5))


_LANDMARK_WINDOWS_BY_DATASET: dict[str, tuple[float, ...]] = {
    "mimiciv": tuple(float(h) for h in MORTALITY_HORIZONS_HOURS),
    "eicu": tuple(float(h) for h in MORTALITY_HORIZONS_HOURS),
    "hirid": tuple(float(h) for h in MORTALITY_HORIZONS_HOURS),
    "mcmed": tuple(float(h) for h in MCMED_HORIZONS_HOURS),
}


def landmark_windows_for(dataset: str) -> tuple[float, ...]:
    registered = registry.window_hours(dataset)
    if registered is not None:
        return registered
    try:
        return _LANDMARK_WINDOWS_BY_DATASET[dataset]
    except KeyError:
        raise KeyError(
            f"no reporting windows for dataset {dataset!r}; add to _LANDMARK_WINDOWS_BY_DATASET "
            f"(ICU {tuple(MORTALITY_HORIZONS_HOURS)}, ED {tuple(MCMED_HORIZONS_HOURS)})") from None


def landmark_key(landmark: Optional[float]) -> str:
    return f"L{float(landmark):g}"


def time_key(t: float) -> str:
    return f"{float(t):g}"


def parse_time_key(key, *, source: str = "<result>") -> float:
    if not isinstance(key, str):
        raise ValueError(
            f"{source}: evaluation-time key {key!r} is a {type(key).__name__}, not a str; build keys with time_key")
    try:
        value = float(key)
    except ValueError:
        raise ValueError(
            f"{source}: evaluation-time key {key!r} is not a number; build keys with time_key") from None
    if key != time_key(value):
        raise ValueError(
            f"{source}: evaluation-time key {key!r} should be spelled {time_key(value)!r}")
    return value


_LANDMARK_KEY_RE = re.compile(r"^L\d+(?:\.\d+)?$")


def landmark_keys_of(block) -> list[str]:
    if not isinstance(block, dict):
        return []
    return [k for k in block
            if isinstance(k, str) and _LANDMARK_KEY_RE.match(k)]


def rebased_horizon(horizon: float, landmark: Optional[float]) -> float:
    if landmark is None:
        return float(horizon)
    return float(horizon) - float(landmark)


def evaluation_times(horizon: float, landmark: Optional[float],
                     dataset: str) -> list[float]:
    if landmark is None:
        return _clinical_auc_times(float(horizon))
    attainable = rebased_horizon(horizon, landmark)
    return [w for w in landmark_windows_for(dataset) if w <= attainable + 1e-9]


def landmark_strata(landmarks, n_rows: int) -> list[tuple]:
    if landmarks is None:
        return [(None, np.ones(int(n_rows), dtype=bool))]
    values = np.asarray(landmarks, dtype=float)
    if len(values) != int(n_rows):
        raise ValueError(
            f"{len(values)} landmarks for {n_rows} rows")
    return [(float(v), values == v) for v in np.unique(values)]


def _pool_landmark_means(results: dict, landmarks: Sequence[float]) -> dict:
    keys = [landmark_key(landmark) for landmark in landmarks]
    blocks = [results[key].get("mean", {}) for key in keys]
    out: dict = {"landmarks_pooled": [float(landmark) for landmark in landmarks]}

    fields: set = set()
    for block in blocks:
        fields |= {k for k, v in block.items()
                   if isinstance(v, float) or isinstance(v, int)}
    for field in sorted(fields):
        values = [float(block[field]) for block in blocks
                  if field in block and not np.isnan(float(block[field]))]
        out[field] = float(np.mean(values)) if values else float("nan")

    pooled_at_times: dict[str, list] = {}
    for block in blocks:
        for time_key, value in (block.get("auc_at_times") or {}).items():
            if value is None or np.isnan(float(value)):
                continue
            pooled_at_times.setdefault(time_key, []).append(float(value))
    out["auc_at_times"] = {k: float(np.mean(v))
                           for k, v in pooled_at_times.items()}
    return out


def stratify_by_landmark(evaluate_stratum, landmarks, n_rows: int, *,
                         horizon: float,
                         dataset: Optional[str] = None) -> dict:
    results: dict = {}
    pooled: list[float] = []
    for landmark, rows in landmark_strata(landmarks, n_rows):
        block = evaluate_stratum(rows,
                                 rebased_horizon(horizon, landmark),
                                 evaluation_times(horizon, landmark, dataset))
        if landmark is None:
            return block
        results[landmark_key(landmark)] = block
        pooled.append(float(landmark))
    results["mean"] = _pool_landmark_means(results, pooled)
    return results


def _to_struct(durations: np.ndarray, events_bin: np.ndarray) -> np.ndarray:
    arr = np.empty(len(durations),
                   dtype=[("event", "?"), ("time", "f8")])
    arr["event"] = events_bin.astype(bool)
    arr["time"] = durations.astype(float)
    return arr


def _risk_score_from_surv(surv_df: pd.DataFrame, horizon: float) -> np.ndarray:
    times = np.asarray(surv_df.index, dtype=float)

    pos = int(np.searchsorted(times, horizon, side="right") - 1)
    pos = max(0, min(pos, len(times) - 1))
    surv_at_h = surv_df.iloc[pos].to_numpy()
    return 1.0 - surv_at_h


def compute_metrics(
    surv_df: pd.DataFrame,
    durations_eval: np.ndarray,
    events_eval_bin: np.ndarray,
    *,
    train_durations: Optional[np.ndarray] = None,
    train_events_bin: Optional[np.ndarray] = None,
    n_time_points: int = 100,
    horizon: Optional[float] = None,
    auc_times: Optional[Sequence[float]] = None,
) -> dict:
    out: dict = {
        "ctd": float("nan"),
        "ibs": float("nan"),
        "ibll": float("nan"),
        "auc_integrated": float("nan"),
        "auc_mean": float("nan"),
        "auc_per_time": [],
        "auc_at_times": {},
        "eval_times": [],
    }

    if len(durations_eval) == 0 or events_eval_bin.sum() == 0:
        return out

    try:
        ev = EvalSurv(surv_df, durations_eval, events_eval_bin, censor_surv="km")
        out["ctd"] = float(ev.concordance_td())
    except Exception as exc:
        print(f"  EvalSurv ctd failed: {exc}")

    t_min, t_max = float(durations_eval.min()), float(durations_eval.max())
    time_grid = np.linspace(t_min, t_max, n_time_points)
    try:
        out["ibs"] = float(ev.integrated_brier_score(time_grid))
        out["ibll"] = float(ev.integrated_nbll(time_grid))
    except Exception as exc:
        print(f"  EvalSurv ibs/ibll failed: {exc}")

    if not _SKSURV_OK:
        return out
    if train_durations is None or train_events_bin is None:

        return out

    try:


        max_train_time = float(np.max(train_durations))
        in_support = durations_eval < max_train_time
        if in_support.sum() < 5 or events_eval_bin[in_support].sum() == 0:
            return out
        durations_auc = durations_eval[in_support]
        events_auc = events_eval_bin[in_support]


        max_supported = min(max_train_time, float(np.max(durations_auc)))
        event_times_auc = durations_auc[events_auc == 1]
        t_start = float(np.min(event_times_auc)) * 1.001
        if t_start >= max_supported * 0.99:
            return out


        if auc_times is not None:
            desired_times = list(auc_times)
        elif horizon is not None:
            desired_times = _clinical_auc_times(float(horizon))
        else:
            desired_times = list(
                np.linspace(t_start, max_supported * 0.99,
                            max(5, n_time_points // 10))
            )


        safe_ceiling = max_supported * 0.99
        filtered = [
            t for t in desired_times if t_start <= t <= safe_ceiling
        ]
        if not filtered:
            if auc_times is not None:


                return out

            filtered = list(
                np.linspace(t_start, safe_ceiling,
                            max(5, n_time_points // 10))
            )
        eval_times = np.asarray(sorted(set(filtered)), dtype=float)


        train_y = _to_struct(train_durations, train_events_bin)
        eval_y = _to_struct(durations_auc, events_auc)

        risk_horizon = safe_ceiling
        risk_scores = _risk_score_from_surv(surv_df, risk_horizon)
        risk_scores = risk_scores[in_support]

        auc, mean_auc = cumulative_dynamic_auc(
            train_y, eval_y, risk_scores, eval_times,
        )
        out["auc_per_time"] = [float(a) for a in auc]
        out["auc_integrated"] = float(mean_auc)
        out["auc_mean"] = float(mean_auc)
        out["eval_times"] = [float(t) for t in eval_times]


        out["auc_at_times"] = {time_key(t_val): float(a_val)
                               for t_val, a_val in zip(eval_times, auc)}

    except Exception as exc:
        print(f"  cumulative_dynamic_auc failed: {exc}")

    return out


def assert_complete_followup(durations: np.ndarray, events: np.ndarray,
                             horizon: float) -> int:
    censored_early = (events == 0) & (durations < horizon - 1e-9)
    n_bad = int(censored_early.sum())
    if n_bad:
        raise ValueError(
            f"{n_bad} of {len(durations)} rows censored before {horizon}h "
            f"(earliest {float(durations[censored_early].min()):.3f}h); metrics need full follow-up to H")
    return int(len(durations))


def compute_competing_risks_metrics(
    cif: np.ndarray,
    cif_times: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    cause: int,
    *,
    times: Sequence[float],
    horizon: float,
) -> dict:
    out: dict = {"brier_at_times": {}, "auc_at_times": {},
                 "nbll_at_times": {}, "ibs": float("nan"),
                 "auc_mean": float("nan"), "ctd": float("nan"),
                 "ibll": float("nan"), "ctd_n_pairs": 0, "n_scored": 0,
                 "n_events_of_cause": 0, "censoring": "none",
                 "censoring_is_informative": False}
    if len(durations) == 0:
        return out

    out["n_events_of_cause"] = int((events == cause).sum())


    try:
        out["n_scored"] = assert_complete_followup(durations, events, horizon)
        out["censoring"] = "none"
        g_times = g_values = None
    except ValueError:
        out["n_scored"] = int(len(durations))
        out["censoring"] = "ipcw"
        g_times, g_values = _censoring_survival(durations, events)


        out["censoring_is_informative"] = True

    ctd, n_pairs = _cause_specific_ctd(cif, cif_times, durations, events,
                                       cause)
    out["ctd"] = ctd
    out["ctd_n_pairs"] = n_pairs
    if np.isnan(ctd):
        print(f"  ctd (cause {cause}): no comparable pairs "
              f"({out['n_events_of_cause']} events of this cause); NaN")

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None

    briers, aucs, nblls = [], [], []
    for t in times:
        if t > horizon + 1e-9:
            continue

        y = ((durations <= t) & (events == cause)).astype(float)
        row = int(np.searchsorted(cif_times, t, side="right")) - 1
        row = min(max(row, 0), len(cif_times) - 1)
        p = np.clip(np.asarray(cif[row], dtype=float), 0.0, 1.0)

        if g_times is None:
            w = np.ones(len(y))
        else:


            settled = durations <= t
            w = np.where(settled, 0.0, 1.0 / _g_at(g_times, g_values, t))
            if settled.any():
                w[settled] = [1.0 / _g_at(g_times, g_values, float(ti))
                              for ti in durations[settled]]
            w[(events == 0) & settled] = 0.0

        denom = w.sum()
        if denom <= 0:
            continue
        brier = float(np.sum(w * (p - y) ** 2) / denom)
        out["brier_at_times"][time_key(t)] = brier
        briers.append(brier)


        p_safe = np.clip(p, 1e-7, 1.0 - 1e-7)
        nbll = float(-np.sum(w * (y * np.log(p_safe)
                                  + (1.0 - y) * np.log1p(-p_safe))) / denom)
        out["nbll_at_times"][time_key(t)] = nbll
        nblls.append(nbll)

        keep = w > 0
        if (roc_auc_score is not None and keep.any()
                and 0 < y[keep].sum() < keep.sum()):
            auc = float(roc_auc_score(y[keep], p[keep],
                                      sample_weight=w[keep]))
            out["auc_at_times"][time_key(t)] = auc
            aucs.append(auc)

    if briers:
        out["ibs"] = float(np.mean(briers))
    else:
        print(f"  competing-risks metrics (cause {cause}): no evaluation time in {list(times)} "
              f"within {horizon}h; ibs/ibll NaN")
    if nblls:
        out["ibll"] = float(np.mean(nblls))
    if aucs:
        out["auc_mean"] = float(np.mean(aucs))
    return out


def _cause_specific_ctd(cif: np.ndarray, cif_times: np.ndarray,
                        durations: np.ndarray, events: np.ndarray,
                        cause: int) -> tuple[float, int]:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events)
    is_cause = events == cause
    if not is_cause.any():
        return float("nan"), 0
    is_competing = (events != 0) & ~is_cause

    concordant = 0.0
    n_pairs = 0
    for t_i in np.unique(durations[is_cause]):
        row = int(np.searchsorted(cif_times, t_i, side="right")) - 1
        row = min(max(row, 0), len(cif_times) - 1)
        p = np.asarray(cif[row], dtype=float)
        comparable = ((durations > t_i) | is_competing
                      | ((events == 0) & (durations == t_i)))
        p_cmp = p[comparable]
        if p_cmp.size == 0:
            continue
        for i in np.where(is_cause & (durations == t_i))[0]:
            n_pairs += int(p_cmp.size)
            concordant += float(np.count_nonzero(p_cmp < p[i]))
            concordant += 0.5 * float(np.count_nonzero(p_cmp == p[i]))
    if n_pairs == 0:
        return float("nan"), 0
    return concordant / n_pairs, n_pairs


def _censoring_survival(durations: np.ndarray, events: np.ndarray) -> tuple:
    order = np.argsort(durations, kind="mergesort")
    t, is_cens = durations[order], (events[order] == 0)
    n = len(t)
    times, values, surv, at_risk = [], [], 1.0, n
    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        d = int(is_cens[i:j].sum())
        if d and at_risk > 0:
            surv *= 1.0 - d / at_risk
        times.append(float(t[i])); values.append(surv)
        at_risk -= (j - i)
        i = j
    return np.asarray(times), np.asarray(values)


def _g_at(times: np.ndarray, values: np.ndarray, t: float) -> float:
    if len(times) == 0:
        return 1.0
    idx = int(np.searchsorted(times, t, side="right")) - 1
    g = 1.0 if idx < 0 else float(values[idx])
    return max(g, 1e-3)
