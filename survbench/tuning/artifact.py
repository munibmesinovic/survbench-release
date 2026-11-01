from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ARMS",
    "TRIAL_STATUSES",
    "artifact_key",
    "artifact_key_for_config",
    "published_dims_of",
    "searched_dims_of",
    "write_artifact",
    "load_artifact",
]


ARTIFACT_SCHEMA_VERSION = 1


ARMS = ("SR-adm", "CR-adm", "CR-lmk")


TRIAL_STATUSES = ("ok", "nan", "oom", "error")

_KEY_SEP = "__"
_MODALITY_SEP = "+"


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


_ROOT = Path(__file__).resolve().parents[2]


def _check_component(value: Any, field: str) -> str:
    text = str(value)
    if not _SAFE_COMPONENT.match(text) or _KEY_SEP in text:
        raise ValueError(
            f"{field}={value!r} is not a valid key component: use [A-Za-z0-9._-], start alphanumeric, "
            f"no {_KEY_SEP!r} or {_MODALITY_SEP!r}")
    return text


def canonical_out_dir(out_dir: Any) -> str:
    if out_dir is None or not str(out_dir).strip():
        raise ValueError(
            "out_dir is required (output.dir from the dataset config)")
    text = str(out_dir)
    absolute = Path(os.path.abspath(text))


    for candidate in (absolute, Path(os.path.realpath(text))):
        try:
            return candidate.relative_to(_ROOT).as_posix()
        except ValueError:
            continue
    return absolute.as_posix()


def _dir_slug(canonical: str) -> str:
    import hashlib

    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    safe = _UNSAFE_CHARS.sub("_", canonical)[-48:].strip("_") or "dir"
    return f"{safe}-{digest}"


def artifact_key(dataset: str,
                 model: str,
                 modalities: Sequence[str],
                 arm: str,
                 out_dir: Any) -> str:
    dataset = _check_component(dataset, "dataset")
    model = _check_component(model, "model")
    if arm not in ARMS:
        raise ValueError(
            f"arm={arm!r} is not one of {ARMS}; use survbench.core.arms.tuning_arm_of")
    mods = tuple(sorted(_check_component(m, "modality") for m in modalities))
    if not mods:
        raise ValueError("modalities is empty; a cell trains on something.")
    if len(set(mods)) != len(mods):
        raise ValueError(f"modalities {tuple(modalities)!r} repeats a block.")
    return _KEY_SEP.join([
        dataset, model, _MODALITY_SEP.join(mods), arm,
        _dir_slug(canonical_out_dir(out_dir)),
    ])


def artifact_key_for_config(config: Mapping[str, Any],
                            model: str,
                            modalities: Sequence[str]) -> str:
    from survbench.core.arms import tuning_arm_of

    try:
        dataset = config["dataset"]["name"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"config has no dataset.name: {exc}") from None
    try:
        out_dir = config["output"]["dir"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"config has no output.dir: {exc}") from None
    return artifact_key(dataset, model, modalities, tuning_arm_of(config),
                        out_dir)


def published_dims_of(space: Any) -> List[str]:
    return sorted(d.name for d in space.dims if not d.repo_declared)


def searched_dims_of(space: Any) -> List[str]:
    return sorted(d.name for d in space.dims)


def _as_trial_dict(trial: Any, index: int) -> Dict[str, Any]:
    if dataclasses.is_dataclass(trial) and not isinstance(trial, type):
        record = dataclasses.asdict(trial)
    elif isinstance(trial, Mapping):
        record = dict(trial)
    else:
        raise TypeError(
            f"trials[{index}] is a {type(trial).__name__}; expected a "
            f"TrialRecord or a mapping.")

    missing = [k for k in ("number", "status", "config") if k not in record]
    if missing:
        raise ValueError(f"trials[{index}] is missing {missing}.")
    status = record["status"]
    if status not in TRIAL_STATUSES:
        raise ValueError(
            f"trials[{index}] has status {status!r}; expected one of {TRIAL_STATUSES}")

    score = record.get("score")
    if score is not None:
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(
                f"trials[{index}] has non-finite score={record['score']!r}; record it as status='nan', score=None")
    record["score"] = score
    record["message"] = record.get("message", "")
    record["config"] = dict(record["config"] or {})
    return record


def write_artifact(path: Any,
                   *,
                   dataset: str,
                   model: str,
                   modalities: Sequence[str],
                   arm: str,
                   out_dir: Any,
                   n_trials: int,
                   sampler_seed: int,
                   search_seed: int,
                   selection: str,
                   trials: Iterable[Any],
                   selected: Optional[Mapping[str, Any]],
                   under_budget: bool,
                   space: Any = None,
                   space_source: Optional[str] = None,
                   repo_declared: Optional[bool] = None,
                   published_dims: Optional[Sequence[str]] = None,
                   searched_dims: Optional[Sequence[str]] = None,
                   sampler: str = "tpe",
                   selected_score: Optional[float] = None,
                   n_successful: Optional[int] = None,
                   read_test_during_search: bool = False,
                   carve_out_reason: str = "",
                   ) -> Dict[str, Any]:
    derived = space is not None
    if derived:
        conflicting = [name for name, value in
                       (("space_source", space_source),
                        ("repo_declared", repo_declared),
                        ("published_dims", published_dims),
                        ("searched_dims", searched_dims))
                       if value is not None]
        if conflicting:
            raise ValueError(
                f"pass either space= or {conflicting}, not both")
        space_source = space.source
        repo_declared = bool(space.repo_declared)
        published_dims = published_dims_of(space)
        searched_dims = searched_dims_of(space)
    else:
        if space_source is None or repo_declared is None:
            raise ValueError(
                "space_source and repo_declared are required when space= is not given")

    key = artifact_key(dataset, model, modalities, arm, out_dir)
    records = [_as_trial_dict(t, i) for i, t in enumerate(trials)]
    counted = sum(1 for r in records if r["status"] == "ok")
    if n_successful is not None and int(n_successful) != counted:
        raise ValueError(
            f"n_successful={n_successful} but {counted} of the {len(records)} trials have status 'ok'")

    payload: Dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "key": key,
        "dataset": dataset,
        "model": model,


        "modalities": sorted(str(m) for m in modalities),
        "arm": arm,
        "out_dir": str(out_dir),
        "out_dir_canonical": canonical_out_dir(out_dir),
        "n_trials": int(n_trials),
        "n_successful": counted,
        "under_budget": bool(under_budget),
        "sampler": sampler,
        "sampler_seed": int(sampler_seed),
        "search_seed": int(search_seed),
        "selection": selection,


        "carve_out_reason": str(carve_out_reason or ""),
        "selected": dict(selected) if selected is not None else None,
        "selected_score": (None if selected_score is None
                           else float(selected_score)),
        "space_source": space_source,
        "repo_declared": bool(repo_declared),
        "published_dims": (None if published_dims is None
                           else sorted(str(d) for d in published_dims)),
        "searched_dims": (None if searched_dims is None
                          else sorted(str(d) for d in searched_dims)),


        "read_test_during_search": bool(read_test_during_search),
        "trials": records,
    }

    destination = Path(path)
    if destination.parent and not destination.parent.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)


    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    destination.write_text(text + "\n")
    return payload


def load_artifact(path: Any,
                  expected_arm: str,
                  expected_key: Optional[str] = None) -> Dict[str, Any]:
    destination = Path(path)
    with open(destination) as handle:
        data = json.load(handle)

    stored = data.get("arm")
    if stored != expected_arm:
        raise ValueError(
            f"{destination} was searched under arm {stored!r} but is being read for arm {expected_arm!r}")
    if expected_key is not None and data.get("key") != expected_key:
        raise ValueError(
            f"{destination} is keyed {data.get('key')!r} but was read for {expected_key!r}")
    return data
