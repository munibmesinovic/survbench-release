import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.baselines._runtime import CONFIGS

MANIFEST = ROOT / "configs" / "arrays_manifest.json"

PATH_KEYS = ("dir", "base_dir", "cache_dir", "cuts_from")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_paths(obj):
    if isinstance(obj, dict):
        return {k: _strip_paths(v) for k, v in obj.items() if k not in PATH_KEYS}
    if isinstance(obj, list):
        return [_strip_paths(v) for v in obj]
    return obj


def digest_of(path):
    if path.name == "SURVBENCH_VERSION.json" or path.name.endswith("_config.yaml"):
        try:
            if path.suffix == ".json":
                loaded = json.loads(path.read_text())
            else:
                loaded = yaml.safe_load(path.read_text())
        except Exception:
            return sha256(path)
        canonical = json.dumps(_strip_paths(loaded), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
    return sha256(path)


def array_dirs():
    dirs = {}
    for (_arm, _dataset), rel in sorted(CONFIGS.items()):
        cfg = yaml.safe_load((ROOT / rel).read_text())
        out = Path(cfg["output"]["dir"])
        path = out if out.is_absolute() else ROOT / out
        dirs.setdefault(path.name, path)
    return dirs


def write_manifest():
    manifest = {}
    for name, path in array_dirs().items():
        if not path.is_dir():
            print(f"  skipped {name}: {path} is not present")
            continue
        manifest[name] = {p.name: digest_of(p)
                          for p in sorted(path.iterdir()) if p.is_file()}
        print(f"  {name}: {len(manifest[name])} files")
    if not manifest:
        raise SystemExit("no array directories are present; nothing to record")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {MANIFEST} covering {len(manifest)} set(s)")
    return 0


def check(only=None, reference=None):
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} is missing; there is nothing to check "
                         "these arrays against")
    manifest = json.loads(MANIFEST.read_text())
    dirs = array_dirs()
    if only:
        unknown = sorted(set(only) - set(manifest))
        if unknown:
            raise SystemExit(f"no manifest entry for {unknown}; known sets are "
                             f"{sorted(manifest)}")
        wanted = list(only)
    else:
        wanted = sorted(manifest)

    checked = skipped = failed = 0
    for name in wanted:
        path = Path(reference) / name if reference else dirs.get(name)
        if path is None or not path.is_dir():
            print(f"{name:28s} SKIPPED   not present at {path}")
            skipped += 1
            continue
        recorded = manifest[name]
        on_disk = {p.name: p for p in sorted(path.iterdir()) if p.is_file()}
        identical = differ = missing = extra = 0
        worst = []
        for fname, digest in sorted(recorded.items()):
            p = on_disk.get(fname)
            if p is None:
                missing += 1
                worst.append(f"missing {fname}")
                continue
            if digest_of(p) == digest:
                identical += 1
            else:
                differ += 1
                worst.append(f"differs {fname}")
        extra = len(set(on_disk) - set(recorded))
        ok = differ == 0 and missing == 0 and extra == 0
        status = "OK" if ok else "MISMATCH"
        print(f"{name:28s} {status:9s} {identical}/{len(recorded)} identical"
              + (f", {differ} differ" if differ else "")
              + (f", {missing} missing" if missing else "")
              + (f", {extra} unexpected" if extra else ""))
        for w in worst[:4]:
            print(f"    {w}")
        checked += 1
        if not ok:
            failed += 1

    print(f"\n{checked - failed}/{checked} set(s) match the manifest"
          + (f"; {skipped} not present" if skipped else ""))
    if checked == 0:
        raise SystemExit("no array directories were present to check")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="sets", nargs="*", default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if args.write:
        return write_manifest()
    return check(args.sets, args.reference)


if __name__ == "__main__":
    sys.exit(main())
