from typing import Dict, List, Tuple


def arm_of(config: Dict) -> Tuple[str, List[int]]:
    ev = (config.get("evaluation") or {})
    mode = ev.get("discharge_as", "competing")
    causes = sorted({int(c) for c in (ev.get("censor_causes") or [])})

    if mode not in ("censored", "competing"):
        raise ValueError(
            f"evaluation.discharge_as={mode!r}; expected 'censored' or "
            "'competing'.")
    if mode == "censored" and not causes:
        raise ValueError(
            "evaluation.discharge_as='censored' needs evaluation.censor_causes; it has no default.")
    if mode == "competing" and causes:
        raise ValueError(
            f"evaluation.discharge_as='competing' conflicts with censor_causes={causes}")
    return mode, causes


def tuning_arm_of(config: Dict) -> str:
    mode, _ = arm_of(config)
    landmarked = bool(
        ((config.get("preprocessing") or {}).get("landmarking") or {})
        .get("enabled", False))

    if mode == "censored":
        if landmarked:
            raise ValueError(
                "evaluation.discharge_as='censored' with landmarking enabled (SR-lmk) is not supported")
        return "SR-adm"
    return "CR-lmk" if landmarked else "CR-adm"


def collapse_censored(events, censor_causes):
    import numpy as np

    events = np.asarray(events, dtype=np.int64).copy()
    censor = sorted({int(c) for c in (censor_causes or [])})
    if censor:
        events[np.isin(events, censor)] = 0
    return events
