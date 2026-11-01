
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

from preprocessing.backend import dataframe_module as pd


def _cache_path(cache_dir: Path, source_label: str) -> Path:
    return Path(cache_dir) / f"events_filtered_{source_label}.pkl"


def stream_filter_events(
    csv_path: str | Path,
    hadm_ids: Iterable[int],
    itemids: Iterable[int],
    cache_dir: str | Path,
    source_label: str,
    chunksize: int = 5_000_000,
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(cache_dir, source_label)
    hadm_set = set(int(h) for h in hadm_ids)
    item_set = set(int(i) for i in itemids)
    usecols = ["hadm_id", "charttime", "itemid", "valuenum", "valueuom"]

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                blob = pickle.load(f)
            if (set(blob.get("hadm_ids", [])) == hadm_set
                    and set(blob.get("itemids", [])) == item_set
                    and set(blob.get("df").columns) == set(usecols)):
                df = blob["df"]
                print(f"  [chartevents_cache] hit: {cache_path.name} ({len(df):,d} rows)")
                return df
            print("  [chartevents_cache] key mismatch: rebuilding")
        except Exception as exc:
            print(f"  [chartevents_cache] error reading cache ({exc}); rebuilding")

    print(f"  [chartevents_cache] streaming {csv_path} (chunksize={chunksize})")
    chunks = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["hadm_id", "valuenum"])
        chunk = chunk[chunk["hadm_id"].isin(hadm_set)]
        chunk = chunk[chunk["itemid"].isin(item_set)]
        if not chunk.empty:
            chunks.append(chunk)
    df = (pd.concat(chunks, ignore_index=True)
          if chunks else pd.DataFrame(
              columns=usecols))
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    print(f"  [chartevents_cache] kept {len(df):,d} rows from {csv_path}")

    blob = {"hadm_ids": list(hadm_set), "itemids": list(item_set), "df": df}
    with open(cache_path, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [chartevents_cache] saved: {cache_path.name}")
    return df
