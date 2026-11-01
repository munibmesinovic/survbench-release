from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


MARKER_NAME = "SURVBENCH_VERSION.json"


CURRENT_VERSION = "v2"

ADM_VERSION = "v2-adm"

ADM_V1CLOCK_VERSION = "v2-adm-v1clock"


DECLARABLE_VERSIONS = (CURRENT_VERSION, ADM_VERSION, ADM_V1CLOCK_VERSION)


_ARTIFACT_SUFFIXES = (".npy", ".p", ".pkl")


def resolve_version(config: dict) -> str:
    declared = (config.get("output") or {}).get(
        "benchmark_version", CURRENT_VERSION)
    if not declared:
        return CURRENT_VERSION
    if declared not in DECLARABLE_VERSIONS:
        raise ValueError(
            f"unknown output.benchmark_version {declared!r}; valid: "
            f"{', '.join(repr(v) for v in DECLARABLE_VERSIONS)}. Omit it to write {CURRENT_VERSION!r}.")
    return declared


def read_version(directory: Path) -> Optional[str]:
    directory = Path(directory)
    marker = directory / MARKER_NAME
    if marker.is_file():
        try:
            return str(json.loads(marker.read_text())["benchmark_version"])
        except (ValueError, KeyError, OSError):


            raise ValueError(
                f"{marker} exists but is not a readable version marker; delete it if the directory is empty")
    if not directory.is_dir():
        return None
    for path in directory.iterdir():
        if path.is_file() and path.suffix in _ARTIFACT_SUFFIXES:
            return "v1"
    return None


def read_arm(directory: Path) -> Optional[str]:
    directory = Path(directory)
    marker = directory / MARKER_NAME
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    arm = payload.get("discharge_as")
    return arm if isinstance(arm, str) and arm else None


def assert_writable(directory: Path, version: str = CURRENT_VERSION,
                    arm: Optional[str] = None) -> None:
    existing_arm = read_arm(directory)
    if arm is not None and existing_arm is not None and existing_arm != arm:
        raise ValueError(
            f"{directory} holds the {existing_arm!r} arm but this run is the {arm!r} arm; "
            f"point output.dir at this arm's own directory")

    existing = read_version(directory)
    if existing is None or existing == version:
        return
    raise ValueError(
        f"{directory} holds SurvBench {existing} artifacts but this run writes {version}; "
        f"set output.dir to a {version} directory or move the existing one aside")


def write_marker(directory: Path, version: str = CURRENT_VERSION,
                 **details) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"benchmark_version": version, **details}
    marker = directory / MARKER_NAME
    marker.write_text(json.dumps(payload, indent=2) + "\n")
    return marker
