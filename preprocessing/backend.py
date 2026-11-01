
from __future__ import annotations

import os
from typing import Literal

import numpy as _np


class BackendUnavailableError(RuntimeError):
    pass


_REQUESTED = os.environ.get("SURVBENCH_BACKEND", "auto").lower()
if _REQUESTED not in ("cpu", "gpu", "auto"):
    raise ValueError(
        f"SURVBENCH_BACKEND={_REQUESTED!r}; expected one of "
        f"'cpu', 'gpu', 'auto'."
    )


def _try_import_gpu():
    import cudf
    import cupy
    from cuml.preprocessing import StandardScaler as _gpu_scaler
    return cudf, cupy, _gpu_scaler


def _resolve():
    if _REQUESTED == "cpu":
        import pandas, numpy
        from sklearn.preprocessing import StandardScaler
        return "cpu", pandas, numpy, StandardScaler

    if _REQUESTED == "gpu":
        try:
            cudf, cupy, scaler = _try_import_gpu()
            return "gpu", cudf, cupy, scaler
        except ImportError as e:
            raise BackendUnavailableError(
                "compute.backend=gpu needs cuDF, CuPy, and cuML"
            ) from e


    try:
        cudf, cupy, scaler = _try_import_gpu()
        return "gpu", cudf, cupy, scaler
    except ImportError:
        import pandas, numpy
        from sklearn.preprocessing import StandardScaler
        return "cpu", pandas, numpy, StandardScaler


BACKEND: Literal["cpu", "gpu"]
BACKEND, dataframe_module, array_module, StandardScaler = _resolve()
assert BACKEND in ("cpu", "gpu"), f"Unexpected backend resolved: {BACKEND!r}"


def to_numpy(arr) -> _np.ndarray:
    if BACKEND == "gpu":
        return arr.get()
    return arr
