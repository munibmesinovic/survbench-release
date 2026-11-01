import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, Optional
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

from survbench.core.arms import arm_of, collapse_censored
from preprocessing.landmarking import landmarking_enabled


def _segment_interior(inside: np.ndarray, lo: float, hi: float,
                      k: int) -> list:
    if k <= 0:
        return []
    fractions = np.linspace(0.0, 1.0, k + 2)[1:-1]
    if inside.size:
        qs = np.asarray(np.quantile(inside, fractions), dtype=float)
        if (np.unique(qs).size == k and qs.min() > lo and qs.max() < hi):
            return [float(q) for q in np.sort(qs)]
    return [float(x) for x in np.linspace(lo, hi, k + 2)[1:-1]]


def conformant_cuts(durations: np.ndarray, events: np.ndarray,
                    anchors: Tuple[float, ...], n_bins: int) -> np.ndarray:
    if int(n_bins) != n_bins:
        raise ValueError(f"n_bins must be a whole number of bins, got {n_bins!r}")
    n_bins = int(n_bins)

    anchors = tuple(sorted(float(a) for a in anchors))
    if len(anchors) + 1 > n_bins:
        raise ValueError(
            f"{len(anchors)} anchors need at least {len(anchors) + 1} bins; got n_bins={n_bins}")
    if anchors[0] <= 0.0:
        raise ValueError(
            f"anchors must be positive hours after the landmark; smallest is {anchors[0]:g}")
    if any(b <= a for a, b in zip(anchors, anchors[1:])):
        raise ValueError(
            f"anchors must be strictly increasing, got {anchors}")

    d = np.asarray(durations, dtype=float)
    e = np.asarray(events)
    if e.shape[0] != d.shape[0]:
        raise ValueError(
            f"durations has {d.shape[0]} rows but events has {e.shape[0]}")

    max_w = anchors[-1]
    d = d[(d >= 0.0) & (d <= max_w)]

    bounds = (0.0,) + anchors
    segments = list(zip(bounds[:-1], bounds[1:]))
    n_interior = n_bins - len(anchors)

    counts = np.array([((d > lo) & (d <= hi)).sum() for lo, hi in segments],
                      dtype=float)
    if counts.sum() == 0:
        counts = np.ones(len(segments))
    share = counts / counts.sum()
    alloc = np.floor(share * n_interior).astype(int)
    while alloc.sum() < n_interior:
        alloc[int(np.argmax(share * n_interior - alloc))] += 1

    edges = [0.0]
    for (lo, hi), k in zip(segments, alloc):
        edges.extend(_segment_interior(d[(d > lo) & (d <= hi)],
                                       lo, hi, int(k)))
        edges.append(float(hi))

    cuts = np.asarray(edges, dtype=float)
    if cuts.shape[0] != n_bins + 1 or not np.all(np.diff(cuts) > 0):
        raise RuntimeError(
            f"conformant_cuts built {cuts.shape[0]} edges for n_bins={n_bins} and anchors {anchors}; "
            f"expected n_bins + 1 increasing edges: {cuts}")
    return cuts


class SurvivalLabelsProcessor:

    def __init__(self, config: Dict):
        self.config = config
        self.max_horizon = config['preprocessing']['max_horizon_hours']
        self.n_bins = config['preprocessing']['n_time_bins']
        self.discretisation = config['preprocessing']['discretisation_method']
        self.event_col = config['preprocessing'].get('event_col', 'event')
        self.duration_col = config['preprocessing'].get('duration_col', 'duration')


        self.discharge_as, self.censor_causes = arm_of(config)


        self.competing_risks_enabled = config['dataset'].get('competing_risks', {}).get('enabled', False)
        self.outcome_mapping = config['dataset'].get('competing_risks', {}).get('outcome_mapping', {})


        self.cause_codes = (
            sorted(k for k in self.outcome_mapping.keys() if k != 0)
            if self.competing_risks_enabled and self.outcome_mapping else [1])
        self.n_events = len(self.cause_codes)

        self.cuts = None
        self.labtrans = None

        if self.competing_risks_enabled:
            print(f"\nCompeting Risks Mode: {self.n_events} competing outcomes")
            for code, name in self.outcome_mapping.items():
                print(f"  {code}: {name}")
        else:
            print("\nSingle Risk Mode")

    def process_labels(
            self,
            cohort_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:

        y_data = {}
        all_durations_train = None
        all_events_train = None

        for split in ['train', 'val', 'test']:
            df = cohort_dfs[split]

            durations = df[self.duration_col].values
            events = df[self.event_col].values.astype(int)


            events[durations > self.max_horizon] = 0
            durations = np.clip(durations, a_min=None, a_max=self.max_horizon)

            y_data[split] = (durations, events)

            if split == 'train':
                all_durations_train = durations
                all_events_train = events


        self._create_time_bins(all_durations_train, all_events_train)


        self._log_statistics(y_data)

        return y_data

    def _landmark_anchors(self) -> Optional[Tuple[float, ...]]:
        if not landmarking_enabled(self.config):
            return None
        from models.baselines._data import dataset_of
        from models.baselines._eval import landmark_windows_for
        return landmark_windows_for(dataset_of(self.config))

    def _create_time_bins(self, durations: np.ndarray, events: np.ndarray):
        if self.competing_risks_enabled:


            scored = collapse_censored(events, self.censor_causes)
            indicator = (scored > 0).astype(int)
        else:

            indicator = events

        anchors = self._landmark_anchors()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            if anchors is None:
                print(f"\nCreating {self.n_bins} time bins using "
                      f"{self.discretisation} method...")
                self.labtrans = LabTransDiscreteTime(self.n_bins,
                                                     self.discretisation)
                self.cuts = self.labtrans.fit(durations, indicator).cuts
            else:


                print(f"\nCreating {self.n_bins} landmark-relative time bins "
                      f"with the reporting windows {anchors} on edges...")
                self.cuts = conformant_cuts(durations, indicator, anchors,
                                            self.n_bins - 1)
                self.labtrans = LabTransDiscreteTime(self.cuts)

        print(f"Time bins: Min={self.cuts.min():.1f}h, Max={self.cuts.max():.1f}h")
        print(f"Bin edges: {self.cuts}")

    def _log_statistics(self, y_data: Dict):
        print("\n" + "=" * 80)
        print("LABEL STATISTICS")
        print("=" * 80)

        for split, (durations, events) in y_data.items():
            print(f"\n{split.upper()}:")
            print(f"  Total N: {len(durations)}")

            if self.competing_risks_enabled:

                censored_count = (events == 0).sum()
                censored_pct = censored_count / len(events) * 100
                print(f"  Censored (0): {censored_count} ({censored_pct:.1f}%)")

                for event_code, event_name in self.outcome_mapping.items():
                    if event_code == 0:
                        continue
                    count = (events == event_code).sum()
                    pct = count / len(events) * 100
                    mean_time = durations[events == event_code].mean() if count > 0 else 0
                    print(f"  {event_name} ({event_code}): {count} ({pct:.1f}%), mean time: {mean_time:.1f}h")
            else:

                event_rate = events.mean() * 100
                censoring_rate = (1 - events).mean() * 100
                mean_duration_event = durations[events == 1].mean() if events.sum() > 0 else 0
                mean_duration_censored = durations[events == 0].mean() if (events == 0).sum() > 0 else 0

                print(f"  Event rate: {event_rate:.2f}%")
                print(f"  Censoring rate: {censoring_rate:.2f}%")
                print(f"  Mean duration (events): {mean_duration_event:.1f}h")
                print(f"  Mean duration (censored): {mean_duration_censored:.1f}h")

            print(f"  Overall mean duration: {durations.mean():.1f}h")
            print(f"  Overall median duration: {np.median(durations):.1f}h")

    def transform_labels(self, durations: np.ndarray, events: np.ndarray) -> Tuple:
        if self.labtrans is None:
            raise ValueError("Must fit labtrans before transforming labels")

        return self.labtrans.transform(durations, events)

    def get_cuts(self) -> np.ndarray:
        if self.cuts is None:
            raise ValueError("Must process labels before accessing cuts")
        return self.cuts

    def get_n_bins(self) -> int:
        return self.n_bins

    def get_n_events(self) -> int:
        return self.n_events

    def get_cause_codes(self) -> list:
        return list(self.cause_codes)

    def is_competing_risks(self) -> bool:
        return self.competing_risks_enabled
