import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.baselines._data import load_config
from models.baselines._runtime import split_config
from survbench import registry


def _resolve(value, override):
    if override is not None:
        return str(Path(override).expanduser().resolve())
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (ROOT / path))


def main():
    registry.load_plugins()
    ap = argparse.ArgumentParser()
    registry.add_plugin_argument(ap)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--radiology-csv", default=None)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    config = split_config(load_config(args.config), args.split_seed)
    if args.radiology_csv is not None:
        config.setdefault("dataset", {}).setdefault("files", {})[
            "radiology_embeddings"] = str(
                Path(args.radiology_csv).expanduser().resolve())
    if args.out_dir is not None:
        config.setdefault("output", {})["dir"] = str(
            Path(args.out_dir).expanduser())
    dataset = config.setdefault("dataset", {})
    dataset["base_dir"] = _resolve(dataset.get("base_dir", ""), args.base_dir)
    if args.cache_dir is not None or dataset.get("cache_dir"):
        dataset["cache_dir"] = _resolve(dataset.get("cache_dir", ""),
                                        args.cache_dir)

    base_dir = Path(dataset["base_dir"])
    missing = []
    for key, rel in sorted((dataset.get("files") or {}).items()):
        path = base_dir / rel
        status = "ok" if path.exists() else "MISSING"
        if status == "MISSING":
            missing.append((key, path))
        if args.check:
            print(f"  {status:8s} {key:24s} {path}")
    if args.check:
        print(f"{dataset.get('name')}: {len(missing)} missing of "
              f"{len(dataset.get('files') or {})}")
        return 1 if missing else 0
    if not base_dir.exists() or missing:
        detail = "\n".join(f"  {k}: {p}" for k, p in missing)
        raise SystemExit(
            f"raw data for {dataset.get('name')} not found under {base_dir}:\n{detail or '  (the directory itself)'}\n"
            f"Run with --check to list expected files or set --base-dir.")

    os.environ["SURVBENCH_BACKEND"] = config.get("compute", {}).get(
        "backend", "auto")

    from preprocessing.pipeline import PreprocessingPipeline

    print(f"preprocessing {dataset.get('name')} from {base_dir}")
    pipeline = PreprocessingPipeline(config)
    try:
        pipeline.run()
    except Exception as exc:
        print(f"failed: {type(exc).__name__}: {exc}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    print(f"wrote arrays to {pipeline.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
