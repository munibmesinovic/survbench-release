from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import optuna


_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.baselines._runtime import REGISTRY, _set_seed
from survbench.tuning.spaces import SearchSpace, space_for

__all__ = [
    "TrialRecord", "SearchResult", "run_search",
    "DEFAULT_SELECTION", "selection_for", "sampler_for", "carve_out_reason_for",
]


_MINIMISED = frozenset({"nll", "loss", "ibs", "ibll", "brier"})
_MAXIMISED = frozenset({"ctd", "auc", "auc_integrated", "cindex", "c_index"})


DEFAULT_SELECTION = "val.nll"


_DEFAULT_SELECTION = {"cox": "val.ctd"}


_GRID_MODELS = frozenset({"cox"})


_CARVE_OUT_REASON = {
    "cox": (
        "cox is deterministic and has no validation likelihood, so it is searched on a "
        "log-spaced penalizer grid of n_trials points and selected on validation Ctd."
    ),
}


_DRIVER_OWNED = ("config", "model", "seed", "skip_test", "modalities")


@dataclass(frozen=True)
class TrialRecord:
    number: int
    config: Dict[str, Any]
    status: str
    score: Optional[float]
    message: str = ""


@dataclass(frozen=True)
class SearchResult:
    trials: Tuple[TrialRecord, ...] = ()
    selected: Optional[Dict[str, Any]] = None
    n_successful: int = 0
    under_budget: bool = False


    selection: str = DEFAULT_SELECTION


    sampler: str = "tpe"
    selected_score: Optional[float] = None


    carve_out_reason: str = ""


def selection_for(model: str, selection: Optional[str] = None) -> str:
    if selection is not None:
        return selection
    return _DEFAULT_SELECTION.get(model, DEFAULT_SELECTION)


def sampler_for(model: str) -> str:
    return "grid" if model in _GRID_MODELS else "tpe"


def carve_out_reason_for(model: str, selection: Optional[str] = None) -> str:
    reason = _CARVE_OUT_REASON.get(model, "")
    if not reason:
        return ""
    declared = _DEFAULT_SELECTION.get(model, DEFAULT_SELECTION)
    resolved = selection_for(model, selection)
    if resolved != declared:
        reason += (
            f" Overridden in this run: selection is {resolved!r}, not the carve-out's {declared!r}.")
    return reason


def _invoke_runner(model: str, config: dict, **kwargs) -> dict:
    return REGISTRY[model](config, **kwargs)


def _direction_of(selection: str) -> str:
    leaf = selection.rsplit(".", 1)[-1]
    if leaf in _MINIMISED:
        return "minimize"
    if leaf in _MAXIMISED:
        return "maximize"
    raise ValueError(
        f"no declared direction for selection metric {selection!r}; add it to _MINIMISED or _MAXIMISED")


def _suggest(trial: optuna.Trial, space: SearchSpace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dim in space.dims:
        if dim.kind == "categorical":
            out[dim.name] = trial.suggest_categorical(dim.name,
                                                      list(dim.choices))
        elif dim.kind == "int":
            out[dim.name] = trial.suggest_int(dim.name, int(dim.low),
                                              int(dim.high), log=dim.log)
        elif dim.kind == "float":
            out[dim.name] = trial.suggest_float(dim.name, float(dim.low),
                                                float(dim.high), log=dim.log)
        else:
            raise ValueError(f"{dim.name}: unknown dim kind {dim.kind!r}")
    return out


def _grid_configs(model: str, space: SearchSpace,
                  n_trials: int) -> List[Dict[str, Any]]:
    dims = tuple(space.dims)
    shape = [(d.name, d.kind, d.log) for d in dims]
    if len(dims) != 1 or dims[0].kind != "float" or not dims[0].log:
        raise ValueError(
            f"{model} uses a grid search, which needs exactly one log-spaced float dimension; got {shape}")
    dim = dims[0]
    low, high = float(dim.low), float(dim.high)
    if not 0.0 < low <= high:
        raise ValueError(
            f"{model}'s grid dim {dim.name!r} is log-spaced over [{low}, {high}]; bounds must be positive and increasing")
    return [{dim.name: float(value)}
            for value in np.geomspace(low, high, int(n_trials))]


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    if any(cls.__name__ == "OutOfMemoryError" for cls in type(exc).__mro__):
        return True
    return "out of memory" in str(exc).lower()


def _classify_failure(exc: BaseException) -> Tuple[str, None, str]:
    if _is_oom(exc):
        return "oom", None, f"{type(exc).__name__}: {exc}"
    return "error", None, f"{type(exc).__name__}: {exc}"


def _score_of(result: Any, selection: str) -> Tuple[str, Optional[float], str]:
    node: Any = result
    try:
        node = node["metrics"]
        for part in selection.split("."):
            node = node[part]
    except (KeyError, TypeError, IndexError) as exc:
        available = ""
        try:
            available = f" Present under result['metrics']: {sorted(result['metrics'])}."
        except Exception:
            pass
        return "error", None, (
            f"selection metric {selection!r} is not in the runner's result "
            f"({type(exc).__name__}: {exc}).{available} cox has no validation likelihood.")
    try:
        value = float(node)
    except (TypeError, ValueError) as exc:
        return "error", None, f"selection metric {selection!r} is not a number: {exc}"
    if not math.isfinite(value):
        return "nan", None, f"selection metric {selection!r} was {value!r}"
    return "ok", value, ""


def _reject_shadowed(fixed: Mapping[str, Any], space: SearchSpace) -> None:
    searched = {dim.name for dim in space.dims}
    clash = sorted(set(fixed) & searched)
    if clash:
        raise ValueError(
            f"fixed_kwargs {clash} are also search dimensions")
    owned = sorted(set(fixed) & set(_DRIVER_OWNED))
    if owned:
        raise ValueError(
            f"fixed_kwargs {owned} are set by the driver and cannot be overridden")


def _assemble(trials: Sequence[TrialRecord],
              n_successful: int,
              n_trials: int,
              selection: str,
              direction: str,
              sampler: str,
              carve_out_reason: str) -> SearchResult:
    ok = [t for t in trials if t.status == "ok"]
    best = None
    if ok:
        pick = min if direction == "minimize" else max
        best = pick(ok, key=lambda t: t.score)
    return SearchResult(
        trials=tuple(trials),
        selected=dict(best.config) if best is not None else None,
        n_successful=n_successful,
        under_budget=n_successful < n_trials,
        selection=selection,
        sampler=sampler,
        selected_score=best.score if best is not None else None,
        carve_out_reason=carve_out_reason,
    )


def _attempt(model: str,
             config: dict,
             number: int,
             params: Dict[str, Any],
             *,
             fixed: Mapping[str, Any],
             modalities: Sequence[str],
             search_seed: int,
             selection: str,
             on_trial_start: Optional[Callable[[int, Dict[str, Any]], None]],
             ) -> TrialRecord:
    kwargs = dict(params)
    kwargs.update(fixed)
    kwargs["seed"] = search_seed


    kwargs["skip_test"] = True
    if model == "transformer":
        kwargs["modalities"] = tuple(modalities)


    if on_trial_start is not None:
        on_trial_start(number, dict(params))

    try:


        _set_seed(search_seed)
        result = _invoke_runner(model=model, config=config, **kwargs)
    except Exception as exc:
        status, score, message = _classify_failure(exc)
    else:
        status, score, message = _score_of(result, selection)

    return TrialRecord(number=number, config=params, status=status,
                       score=score, message=message)


def run_search(model: str,
               config: dict,
               modalities: Sequence[str] = ("dynamic", "static"),
               *,
               n_trials: int = 20,
               sampler_seed: int = 0,
               search_seed: int = 42,
               selection: Optional[str] = None,
               fixed_kwargs: Optional[Mapping[str, Any]] = None,
               on_trial_start: Optional[Callable[[int, Dict[str, Any]], None]] = None,
               on_trial: Optional[Callable[[SearchResult], None]] = None,
               ) -> SearchResult:
    if n_trials < 1:


        raise ValueError(f"n_trials must be at least 1, got {n_trials}")

    space = space_for(model, modalities=modalities)
    fixed = dict(fixed_kwargs or {})
    _reject_shadowed(fixed, space)
    selection = selection_for(model, selection)
    sampler = sampler_for(model)
    carve_out_reason = carve_out_reason_for(model, selection)
    if sampler != "tpe" and not carve_out_reason:


        raise ValueError(
            f"{model} uses sampler {sampler!r} but has no entry in _CARVE_OUT_REASON")
    direction = _direction_of(selection)

    trial_args = dict(fixed=fixed, modalities=modalities,
                      search_seed=search_seed, selection=selection,
                      on_trial_start=on_trial_start)

    if sampler == "grid":


        trials: List[TrialRecord] = []
        n_successful = 0
        for number, params in enumerate(_grid_configs(model, space, n_trials)):
            record = _attempt(model, config, number, params, **trial_args)
            trials.append(record)
            if record.status == "ok":
                n_successful += 1
            if on_trial is not None:
                on_trial(_assemble(trials, n_successful, n_trials, selection,
                                   direction, sampler, carve_out_reason))
        return _assemble(trials, n_successful, n_trials, selection, direction,
                         sampler, carve_out_reason)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed))

    trials = []
    n_successful = 0
    max_attempts = 2 * n_trials

    while n_successful < n_trials and len(trials) < max_attempts:
        trial = study.ask()
        params = _suggest(trial, space)

        record = _attempt(model, config, trial.number, params, **trial_args)
        trials.append(record)
        if record.status == "ok":
            n_successful += 1
            study.tell(trial, record.score)
        else:


            study.tell(trial, state=optuna.trial.TrialState.FAIL)

        if on_trial is not None:
            on_trial(_assemble(trials, n_successful, n_trials, selection,
                               direction, sampler, carve_out_reason))

    return _assemble(trials, n_successful, n_trials, selection, direction,
                     sampler, carve_out_reason)
