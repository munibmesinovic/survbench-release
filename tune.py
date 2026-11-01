import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.baselines import _data
from models.baselines._runtime import (ALL_DATASETS, ARMS, cells, config_path,
                                       load_hparams, registry_key)
from survbench.core.arms import tuning_arm_of
from survbench.tuning.artifact import artifact_key_for_config, write_artifact
from survbench.tuning.search import (SearchResult, carve_out_reason_for,
                                     run_search, sampler_for, selection_for)
from survbench.tuning.spaces import space_for

DEFAULT_MODALITIES = ("dynamic", "static")
NO_VAL_NLL = {"cox"}
CARVE_OUT_PENDING: set = set()
KEPT_INFLIGHT_LIMIT = 3


class _Interrupted(BaseException):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


class CellWriter:

    def __init__(self, out_dir: Path, key: str, config: dict, model: str,
                 modalities: Sequence[str], space: Any, args,
                 selection: str, carve_out_reason: str,
                 provisional: bool = False) -> None:
        self.key = key
        self.provisional = provisional
        self.canonical = out_dir / f"{key}.json"
        self.final = (out_dir / f"{key}.provisional.json" if provisional
                      else self.canonical)
        self.partial = out_dir / f"{key}.partial.json"
        self.no_winner = out_dir / f"{key}.no-winner.json"
        self.inflight = out_dir / f"{key}.inflight.json"
        self._fixed = dict(
            dataset=config["dataset"]["name"],
            model=model,
            modalities=list(modalities),
            arm=tuning_arm_of(config),
            out_dir=config["output"]["dir"],
            n_trials=args.n_trials,
            sampler_seed=args.sampler_seed,
            search_seed=args.search_seed,
            selection=selection,
            carve_out_reason=carve_out_reason,
            space=space,
            read_test_during_search=False,
        )

    def save(self, path: Path, result: SearchResult) -> None:
        tmp = path.with_name(path.name + ".tmp")
        write_artifact(
            tmp,
            trials=result.trials,
            selected=result.selected,
            under_budget=result.under_budget,
            sampler=result.sampler,
            selected_score=result.selected_score,
            **self._fixed,
        )
        os.replace(tmp, path)

    def mark_inflight(self, number: int, config: Dict[str, Any]) -> None:
        tmp = self.inflight.with_name(self.inflight.name + ".tmp")
        tmp.write_text(json.dumps({
            "key": self.key,
            "trial_number": number,
            "config": config,
            "started_utc": _now(),
            "pid": os.getpid(),
            "note": ("Trial running when the process died; it has no record in the artifact."),
        }, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self.inflight)

    def clear_inflight(self) -> None:
        _unlink_quiet(self.inflight)

    def preserve_prior_inflight(self) -> Optional[Path]:
        if not self.inflight.exists():
            return None
        _unlink_quiet(self._kept(KEPT_INFLIGHT_LIMIT))
        for slot in range(KEPT_INFLIGHT_LIMIT - 1, 0, -1):
            if self._kept(slot).exists():
                os.replace(self._kept(slot), self._kept(slot + 1))
        os.replace(self.inflight, self._kept(1))
        return self._kept(1)

    def _kept(self, slot: int) -> Path:
        return self.inflight.with_name(f"{self.key}.inflight.{slot}.kept")


def tune_cell(model: str, config: dict, modalities: Sequence[str],
              out_dir: Path, args) -> Dict[str, Any]:
    base = registry_key(model)
    space = space_for(base, modalities=modalities)
    key = artifact_key_for_config(config, base, modalities)
    provisional = base in CARVE_OUT_PENDING
    selection = selection_for(base, args.selection)
    sampler = sampler_for(base)
    carve_out_reason = carve_out_reason_for(base, args.selection)
    writer = CellWriter(out_dir, key, config, base, modalities, space, args,
                        selection=selection,
                        carve_out_reason=carve_out_reason,
                        provisional=provisional)

    kept = writer.preserve_prior_inflight()
    if kept is not None:
        print(f"  kept the previous run's inflight record -> {kept.name}",
              flush=True)

    writer.save(writer.partial, SearchResult(
        trials=(), selected=None, n_successful=0, under_budget=True,
        selection=selection, sampler=sampler, selected_score=None,
        carve_out_reason=carve_out_reason))

    started = time.monotonic()
    try:
        result = run_search(
            base, config, modalities,
            n_trials=args.n_trials,
            sampler_seed=args.sampler_seed,
            search_seed=args.search_seed,
            selection=selection,
            on_trial_start=writer.mark_inflight,
            on_trial=lambda partial: (writer.save(writer.partial, partial),
                                      writer.clear_inflight()),
        )
    except _Interrupted:
        print(f"  interrupted; completed trials are in {writer.partial}", flush=True)
        raise

    elapsed = time.monotonic() - started
    writer.clear_inflight()

    if result.selected is None:
        os.replace(writer.partial, writer.no_winner)
        print(f"  no winner after {len(result.trials)} attempts ({elapsed:.1f}s) -> {writer.no_winner}", flush=True)
        return {"model": model, "key": key, "status": "no-winner",
                "path": str(writer.no_winner), "selected": None}

    writer.save(writer.final, result)
    _unlink_quiet(writer.partial)
    _unlink_quiet(writer.no_winner)
    print(f"  {result.n_successful}/{args.n_trials} trials ok ({len(result.trials)} attempts, {elapsed:.1f}s), "
          f"{result.selection}={result.selected_score:.6g} -> {writer.final}", flush=True)
    return {"model": model, "key": key,
            "status": "provisional" if provisional else "ok",
            "path": str(writer.final), "selected": dict(result.selected)}


def cell_modalities(model: str, entry: dict) -> tuple:
    if registry_key(model) != "transformer":
        return DEFAULT_MODALITIES
    return tuple(entry.get("modalities") or DEFAULT_MODALITIES)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--dataset", choices=list(ALL_DATASETS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--sampler-seed", type=int, default=0)
    ap.add_argument("--search-seed", type=int, default=42)
    ap.add_argument("--selection", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "tuning"))
    args = ap.parse_args(argv)

    hparams = load_hparams()
    key = f"{args.arm}/{args.dataset}/{args.model}"
    if key not in hparams:
        available = sorted(m for _a, _d, m in cells(
            hparams, (args.arm,), args.dataset))
        ap.error(f"no such cell {key}. Models for {args.arm}/{args.dataset}: "
                 f"{available}")

    base = registry_key(args.model)
    if args.selection and args.selection.endswith("nll") and base in NO_VAL_NLL:
        ap.error(f"{base} has no held-out likelihood, so --selection {args.selection} is not available; leave --selection unset.")

    config = _data.load_config(str(config_path(args.arm, args.dataset)))
    modalities = cell_modalities(args.model, hparams[key])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _on_signal(signum, _frame):
        raise _Interrupted(f"signal {signal.Signals(signum).name}")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"tuning {key} ({args.n_trials} trials, modalities={list(modalities)})",
          flush=True)
    try:
        outcome = tune_cell(args.model, config, modalities, out_dir, args)
    except _Interrupted as exc:
        print(f"interrupted: {exc}", flush=True)
        return 130

    if outcome["selected"] is not None:
        print("selected: " + json.dumps(outcome["selected"], sort_keys=True))
    return 0 if outcome["status"] in ("ok", "provisional") else 1


if __name__ == "__main__":
    sys.exit(main())
