from __future__ import annotations

import numpy as np
import pandas as pd


def pool_reports_to_rows(reports: pd.DataFrame, stay_ids, landmarks,
                         embedding_cols) -> tuple[np.ndarray, np.ndarray]:
    n_rows, dim = len(stay_ids), len(embedding_cols)
    pooled = np.full((n_rows, dim), np.nan, dtype=np.float32)
    n_seen = np.zeros(n_rows, dtype=np.int64)
    if reports is None or len(reports) == 0:
        return pooled, n_seen

    values = reports[list(embedding_cols)].to_numpy(dtype=np.float64)
    hours = reports["hours_since_intime"].to_numpy(dtype=np.float64)


    by_stay: dict = {}
    for pos, stay in enumerate(reports["admission_id"].to_numpy()):
        by_stay.setdefault(stay, []).append(pos)
    order = {stay: np.asarray(sorted(idx, key=lambda p: hours[p]))
             for stay, idx in by_stay.items()}

    for row, stay in enumerate(np.asarray(stay_ids)):
        idx = order.get(stay)
        if idx is None:
            continue
        if landmarks is not None:


            keep = idx[hours[idx] < float(landmarks[row])]
        else:
            keep = idx
        if len(keep) == 0:
            continue
        pooled[row] = values[keep].mean(axis=0)
        n_seen[row] = len(keep)

    return pooled, n_seen
