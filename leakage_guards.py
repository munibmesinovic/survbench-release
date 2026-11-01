from __future__ import annotations

import numpy as np


def assert_splits_disjoint(train_ids, val_ids, test_ids) -> None:
    tr = set(np.asarray(train_ids).tolist())
    va = set(np.asarray(val_ids).tolist())
    te = set(np.asarray(test_ids).tolist())
    overlaps = {"train_val": tr & va, "train_test": tr & te, "val_test": va & te}
    bad = {k: len(v) for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(f"patient-level split leakage: {bad}")


_TIME_KINDS = {"M": "a temporal dtype (any unit)", "i": "int64 epoch seconds"}


def _comparable_times(values, argument: str) -> np.ndarray:
    array = np.asarray(values)
    kind = array.dtype.kind
    if kind == "i" and array.dtype.itemsize == 8:
        return array
    if kind == "M":
        return array
    raise TypeError(
        f"{argument}: unsupported dtype {str(array.dtype)!r}; pass {' or '.join(sorted(_TIME_KINDS.values()))}"
    )


def _assert_no_promotion_overflow(admit: np.ndarray, index: np.ndarray) -> None:
    if admit.dtype.kind != "M" or admit.dtype == index.dtype:
        return
    common = np.promote_types(admit.dtype, index.dtype)
    for array, argument in ((admit, "admittimes"), (index, "index_times")):
        if array.dtype == common:
            continue
        round_tripped = array.astype(common).astype(array.dtype)
        lost = (round_tripped != array) & ~np.isnat(array)
        if lost.any():
            raise TypeError(
                f"{argument}: {int(lost.sum())} values overflow {str(array.dtype)!r} -> {str(common)!r} (index_times {str(index.dtype)!r}), "
                f"e.g. {array[lost][0]} -> {array[lost][0].astype(common)}"
            )


def assert_icd_history_is_causal(hadm_ids, current_hadm_ids, admittimes, index_times) -> None:
    hadm = np.asarray(hadm_ids)
    current = np.asarray(current_hadm_ids)
    n_index_encounter = int((hadm == current).sum())

    admit = _comparable_times(admittimes, "admittimes")
    index = _comparable_times(index_times, "index_times")
    if admit.dtype.kind != index.dtype.kind:
        raise TypeError(
            f"admittimes is {_TIME_KINDS[admit.dtype.kind]} but index_times is {_TIME_KINDS[index.dtype.kind]}; convert both first"
        )
    _assert_no_promotion_overflow(admit, index)

    n_not_preceding = int((~(admit < index)).sum())

    if n_index_encounter or n_not_preceding:
        raise ValueError(
            f"causal ICD filter leakage: {n_index_encounter} rows from the index encounter, {n_not_preceding} rows not before the index time"
        )


def mask_post_event_windows(data_array, durations, window_size_hours, num_dynamic):
    out = data_array.copy()
    n, t, f = out.shape
    ends = (np.arange(t) + 1) * window_size_hours
    post = ends[None, :] > np.asarray(durations)[:, None]
    rows, wins = np.where(post)
    out[rows, wins, :num_dynamic] = np.nan
    return out


def assert_no_observation_after_the_event(mask_array, durations,
                                          window_size_hours, num_dynamic,
                                          *, context: str = "") -> None:
    mask = np.asarray(mask_array)
    if mask.ndim != 3:
        raise ValueError(
            f"mask_array must be (N, T, F); got {mask.shape}")
    durations = np.asarray(durations, dtype=float)
    n, t, _ = mask.shape
    if durations.shape != (n,):
        raise ValueError(
            f"durations has shape {durations.shape} but the mask has {n} rows")

    ends = (np.arange(t) + 1) * window_size_hours
    post = ends[None, :] > durations[:, None]


    observed_after = (mask[:, :, :num_dynamic] > 0) & post[:, :, None]
    n_bad = int(observed_after.sum())
    if not n_bad:
        return

    rows = np.unique(np.where(observed_after)[0])
    per_window = observed_after.sum(axis=(0, 2))
    where = ", ".join(f"w{i}:{int(c):,d}" for i, c in enumerate(per_window) if c)
    raise ValueError(
        f"post-event observation leak{f' [{context}]' if context else ''}: {n_bad:,d} cells in {len(rows):,d} rows observed after event/censoring; "
        f"by window: {where}")
