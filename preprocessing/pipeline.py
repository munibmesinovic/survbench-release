import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional

from preprocessing.backend import dataframe_module as pd, array_module as np
from preprocessing.backend import StandardScaler, to_numpy

from data.eicu_loader import eICUDataLoader
from data.mcmed_loader import MCMEDDataLoader
from data.mimiciv_loader import MIMICIVDataLoader
from data.hirid_loader import HIRIDDataLoader
from preprocessing.labels import SurvivalLabelsProcessor
from preprocessing.landmarking import LANDMARK_INDEX_NAMES
from preprocessing.timeseries import TimeSeriesAggregator
from preprocessing.imputation import train_mean_fallback

LOADERS: Dict[str, type] = {
    "eicu": eICUDataLoader,
    "mcmed": MCMEDDataLoader,
    "mimiciv": MIMICIVDataLoader,
    "hirid": HIRIDDataLoader,
}


def register_loader(name: str, loader_cls: type) -> None:
    from data.base_loader import BaseDataLoader
    key = name.lower()
    if not issubclass(loader_cls, BaseDataLoader):
        raise TypeError(
            f"{loader_cls.__name__} must subclass BaseDataLoader")
    existing = LOADERS.get(key)
    if existing is not None and existing is not loader_cls:
        raise ValueError(
            f"dataset name {key!r} is already registered to {existing.__name__}")
    LOADERS[key] = loader_cls


_ROOT = Path(__file__).resolve().parents[1]


def _resolve_output_dir(config: Dict) -> Path:
    out = Path(config['output']['dir'])
    return out if out.is_absolute() else _ROOT / out


class PreprocessingPipeline:

    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = _resolve_output_dir(config)
        self.output_dir.mkdir(parents=True, exist_ok=True)


        dataset_name = config['dataset']['name'].lower()
        try:
            loader_cls = LOADERS[dataset_name]
        except KeyError:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Registered: {sorted(LOADERS)}. "
                f"Use register_loader({dataset_name!r}, YourLoader) to add one.") from None
        self.loader = loader_cls(config)

        self.label_processor = SurvivalLabelsProcessor(config)
        self.ts_aggregator = TimeSeriesAggregator(config)

        self.cohort_dfs = None
        self.split_data = {}
        self.split_masks = {}


        self.timeseries_observed: dict = {}
        self.y_data = None
        self.scaler = None
        self.scaler_static = None
        self.feature_structure = None
        self.unit_conversion_summary: dict = {}


        self.modalities = config.get('modalities', {})
        self.active_modalities = [k for k, v in self.modalities.items() if v]


        self.landmarking = bool(
            config.get('preprocessing', {})
            .get('landmarking', {}).get('enabled', False))

        print(f"\n{'=' * 80}")
        print("MULTI-MODAL PREPROCESSING PIPELINE")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"Active modalities: {', '.join(self.active_modalities)}")
        print(f"{'=' * 80}")

    def run(self):

        print("\n[1/9] Building cohort...")
        cohort_df = self.loader.build_cohort()

        print("\n[2/9] Creating data splits...")
        splits = self.loader.create_splits()
        self.cohort_dfs = {
            split: cohort_df.loc[patient_ids]
            for split, patient_ids in splits.items()
        }

        if self.landmarking:
            print("\n[2b/9] Expanding to landmark rows...")
            self._expand_to_landmarks()

        print("\n[3/9] Processing survival labels...")
        self.y_data = self.label_processor.process_labels(self.cohort_dfs)

        print("\n[4/9] Processing modalities...")
        self._process_all_modalities()

        print("\n[4b/9] Masking post-event windows (leakage guard)...")
        self._mask_post_event_windows()

        print("\n[4c/9] Applying physiological range filter...")
        self._apply_physiological_ranges()

        if self.landmarking:
            self._assert_every_landmark_has_dynamic_signal()

        print("\n[5/9] Creating missingness masks...")
        self._create_all_masks()
        self._compute_modality_masks()

        print("\n[6/9] Applying imputation...")
        self._apply_imputation()

        print("\n[7/9] Applying feature scaling...")
        self._apply_scaling()

        print("\n[8/9] Combining modalities...")
        self._combine_modalities()

        print("\n[9/9] Saving outputs...")
        self._save_outputs()

        print(f"\n{'=' * 80}")
        print("PREPROCESSING COMPLETE!")
        print(f"{'=' * 80}")

    @staticmethod
    def save_row_identity(cohort: pd.DataFrame, output_dir, split: str,
                          prefix: str) -> list:
        output_dir = Path(output_dir)
        written = []


        pids_path = output_dir / f"pids_{split}_{prefix}.npy"
        np.save(pids_path, to_numpy(np.asarray(
            PreprocessingPipeline._stay_ids(cohort).values)))
        written.append(pids_path)

        if isinstance(cohort.index, pd.MultiIndex):
            lm_path = output_dir / f"landmarks_{split}_{prefix}.npy"
            np.save(lm_path, to_numpy(np.asarray(
                cohort.index.get_level_values(LANDMARK_INDEX_NAMES[1]).values,
                dtype=np.float64)))
            written.append(lm_path)


        if 'subject_id' in cohort.columns:
            sid_path = output_dir / f"subject_ids_{split}_{prefix}.npy"
            np.save(sid_path, to_numpy(np.asarray(cohort['subject_id'].values)))
            written.append(sid_path)


        if 'censoring_reason' in cohort.columns:
            cr_path = output_dir / f"censoring_reason_{split}_{prefix}.npy"
            np.save(cr_path,
                    np.asarray(cohort['censoring_reason'].to_numpy(),
                               dtype=object),
                    allow_pickle=True)
            written.append(cr_path)

        return written

    @staticmethod
    def _stay_ids(cohort: pd.DataFrame) -> pd.Index:
        if isinstance(cohort.index, pd.MultiIndex):
            return cohort.index.get_level_values(LANDMARK_INDEX_NAMES[0])
        return cohort.index

    def _expand_to_landmarks(self):
        from preprocessing.landmarking import (
            assert_landmark_rows_are_at_risk,
            assert_min_stay_does_not_delete_the_early_landmarks, expand_cohort,
        )


        assert_min_stay_does_not_delete_the_early_landmarks(
            self.config, self.config['dataset']['name'].lower())

        if not hasattr(self.loader, 'build_episodes'):
            raise ValueError(
                f"{type(self.loader).__name__} has no build_episodes(); landmarking needs episode start and end times")

        episodes = self.loader.build_episodes()
        dataset = self.config['dataset']['name'].lower()
        total_before = sum(len(c) for c in self.cohort_dfs.values())

        expanded = {}
        for split, cohort in self.cohort_dfs.items():
            rows = expand_cohort(cohort, episodes, dataset=dataset)
            assert_landmark_rows_are_at_risk(rows)
            expanded[split] = rows
            n_stays = rows.index.get_level_values(0).nunique()
            print(f"    {split}: {n_stays:,d} stays -> {len(rows):,d} "
                  f"landmark rows")

        self.cohort_dfs = expanded
        total_after = sum(len(c) for c in self.cohort_dfs.values())
        stays_after = sum(c.index.get_level_values(0).nunique()
                          for c in self.cohort_dfs.values())
        print(f"    total: {stays_after:,d} stays -> {total_after:,d} "
              f"landmark rows")


        if stays_after != total_before:
            print(f"    dropped {total_before - stays_after:,d} of {total_before:,d} stays that ended before the first landmark")

    def _process_all_modalities(self):


        if self.modalities.get('timeseries', False):
            print("\n  -> Processing time-series modality...")
            self.loader.timeseries_df = self.loader.load_timeseries()
            if hasattr(self.loader, '_unit_conversion_summary'):
                self.unit_conversion_summary.update(self.loader._unit_conversion_summary)


            static_for_agg = (
                self.loader.static_df
                if (hasattr(self.loader, 'static_df')
                    and self.loader.static_df is not None
                    and self.modalities.get('static', False))
                else None)
            self.split_data['timeseries'], self.timeseries_observed = (
                self.ts_aggregator.aggregate_to_windows(
                    self.loader.timeseries_df,
                    self.cohort_dfs,
                    static_for_agg,
                    return_observed=True,
                ))

            dynamic_cols = list(self.loader.timeseries_df.columns) if len(self.loader.timeseries_df) > 0 else []
            static_cols = list(self.loader.static_df.columns) if self.modalities.get('static', False) and hasattr(
                self.loader, 'static_df') else []

            self.feature_structure = self.ts_aggregator.get_feature_structure(
                dynamic_cols, static_cols
            )

            print(f"    Time-series shape: {self.split_data['timeseries']['train'].shape}")


        elif self.modalities.get('static', False):
            print("\n  -> Processing static-only modality...")


            if self.loader.static_df is None:
                self.loader.static_df = self.loader.load_static_features()

            static_cols = list(self.loader.static_df.columns)
            self.feature_structure = {
                'num_dynamic': 0,
                'num_static': len(static_cols),
                'num_total': len(static_cols),
                'dynamic_names': [],
                'static_names': static_cols,
                'all_names': static_cols,
                'dynamic_indices': [],
                'static_indices': list(range(len(static_cols)))
            }

            self.split_data['static'] = {}
            for split, cohort in self.cohort_dfs.items():
                stay_ids = self._stay_ids(cohort)
                split_static = self.loader.static_df.reindex(stay_ids)
                self.split_data['static'][split] = split_static.values.astype(np.float32)
                print(f"    {split} static shape: {self.split_data['static'][split].shape}")


        if self.modalities.get('icd', False):
            print("\n  -> Processing ICD codes modality...")
            icd_df, icd_embeddings = self.loader.load_icd_codes()

            if icd_embeddings is not None:

                self.split_data['icd'] = self._create_icd_splits(icd_df, icd_embeddings)
            elif len(icd_df) > 0:

                self.split_data['icd'] = self._create_icd_splits(icd_df, None)
            else:
                print("    Warning: No ICD data available")


        if self.modalities.get('radiology', False):
            print("\n  -> Processing radiology modality...")
            reports_df, rad_embeddings = self.loader.load_radiology_reports()

            if rad_embeddings is not None or len(reports_df) > 0:
                self.split_data['radiology'] = self._create_radiology_splits(reports_df, rad_embeddings)
            else:
                print("    Warning: No radiology data available")

    def _create_icd_splits(self, icd_df: pd.DataFrame, embeddings: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        icd_splits = {}

        for split in ['train', 'val', 'test']:
            cohort = self.cohort_dfs[split]


            split_ids = self._stay_ids(cohort)

            split_icd = icd_df.reindex(split_ids, fill_value=0)

            icd_splits[split] = split_icd.values.astype(np.float32)
            print(f"    {split} ICD shape: {icd_splits[split].shape}")

        return icd_splits

    def _create_radiology_splits(self, reports_df: pd.DataFrame, embeddings: Optional[np.ndarray]) -> Dict[
        str, np.ndarray]:
        from preprocessing.radiology import pool_reports_to_rows

        embedding_cols = [c for c in reports_df.columns
                          if c not in ("admission_id", "hours_since_intime")]
        per_landmark = "hours_since_intime" in reports_df.columns

        rad_splits = {}
        for split in ['train', 'val', 'test']:
            cohort = self.cohort_dfs[split]
            split_ids = np.asarray(self._stay_ids(cohort).values)

            if not per_landmark:


                rad_splits[split] = (reports_df.reindex(split_ids)
                                     .values.astype(np.float32))
                print(f"    {split} radiology shape: {rad_splits[split].shape}")
                continue

            landmarks = None
            if isinstance(cohort.index, pd.MultiIndex):
                landmarks = cohort.index.get_level_values(
                    LANDMARK_INDEX_NAMES[1]).to_numpy(dtype=np.float64)

            pooled, n_seen = pool_reports_to_rows(
                reports_df, split_ids, landmarks, embedding_cols)
            rad_splits[split] = pooled
            with_any = float((n_seen > 0).mean())
            print(f"    {split} radiology shape: {pooled.shape}  "
                  f"rows with >=1 report available: {with_any:.1%}  "
                  f"mean pooled per row: {n_seen.mean():.2f}")

        return rad_splits

    def _mask_post_event_windows(self):
        if not self.config.get('preprocessing', {}).get('mask_post_event_windows', True):
            return
        ts = self.split_data.get('timeseries')
        if ts is None:
            return
        if self.landmarking:
            print("  Post-event masking skipped: with landmarking all input windows precede L.")
            return
        from leakage_guards import mask_post_event_windows
        window_size = self.config['preprocessing']['window_size_hours']
        num_dynamic = self.feature_structure['num_dynamic']
        for split in ['train', 'val', 'test']:
            if split not in ts:
                continue
            ts[split] = mask_post_event_windows(
                ts[split], self.y_data[split][0], window_size, num_dynamic)
        print("  Post-event dynamic windows masked (mask_post_event_windows=True)")

    def _assert_every_landmark_has_dynamic_signal(self, min_observed=0.5):
        ts = self.split_data.get('timeseries')
        if ts is None or self.feature_structure is None:
            return
        num_dynamic = self.feature_structure['num_dynamic']
        if not num_dynamic:
            return

        failures = []
        for split, cohort in self.cohort_dfs.items():
            array = ts.get(split)
            if array is None or len(cohort) == 0:
                continue
            landmarks = cohort.index.get_level_values(LANDMARK_INDEX_NAMES[1])
            dynamic = np.asarray(to_numpy(array)[:, :, :num_dynamic])
            observed = ~np.isnan(dynamic)
            for landmark in sorted(set(np.asarray(landmarks).tolist())):
                rows = np.asarray(landmarks) == landmark
                fraction = float(observed[rows].mean())
                if fraction < min_observed * 0.02:
                    failures.append(
                        f"{split} L={landmark:g}h: {fraction:.4%} of the "
                        f"dynamic block observed over {int(rows.sum()):,d} rows")
        if failures:
            raise ValueError(
                "landmark rows with no dynamic signal:\n  - "
                + "\n  - ".join(failures)
                + "\n\nThe time series does not cover these landmarks; check "
                  "preprocessing.landmarking.observation_horizon_hours.")
        print("  Every landmark has dynamic signal.")

    def _observed_cause_codes(self) -> list:
        report = getattr(self.loader, '_label_report', None) or {}
        codes = report.get('cause_codes')
        if codes:
            return [int(c) for c in codes if int(c) != 0]

        if self.y_data is None:
            return []
        print("  no label report; cause codes taken from the label arrays.")
        seen = set()
        for split in ('train', 'val', 'test'):
            if split in self.y_data:
                seen.update(int(c) for c in np.unique(self.y_data[split][1]))
        return sorted(c for c in seen if c != 0)

    def _apply_physiological_ranges(self):
        cfg = self.config.get('preprocessing', {}).get('range_filter', {}) or {}
        if not cfg.get('enabled', True):
            print("  Range filter disabled by config; impossible values will affect the fitted scale.")
            return
        if 'timeseries' not in self.split_data or not self.feature_structure:
            print("  No timeseries modality; range filter not applicable.")
            return

        from preprocessing.ranges import (
            apply_physiological_ranges, format_range_report,
        )

        num_dynamic = self.feature_structure['num_dynamic']
        dynamic_names = self.feature_structure.get('dynamic_names', [])
        if num_dynamic <= 0 or not dynamic_names:
            print("  No dynamic features; range filter not applicable.")
            return

        totals = {'filtered': {}, 'converted': {}, 'no_range': [],
                  'n_filtered': 0, 'n_converted': 0}
        n_cells = 0
        for split in ['train', 'val', 'test']:
            if split not in self.split_data['timeseries']:
                continue
            data = self.split_data['timeseries'][split]
            block = data[:, :, :num_dynamic]
            n_cells += int(block.size)
            filtered, report = apply_physiological_ranges(block, dynamic_names)
            data[:, :, :num_dynamic] = filtered
            self.split_data['timeseries'][split] = data

            for key in ('filtered', 'converted'):
                for concept, n in report[key].items():
                    totals[key][concept] = totals[key].get(concept, 0) + n
            totals['no_range'] = report['no_range']
        totals['n_filtered'] = int(sum(totals['filtered'].values()))
        totals['n_converted'] = int(sum(totals['converted'].values()))


        upstream = getattr(getattr(self, 'loader', None),
                           '_range_filter_report', None)
        if upstream:
            print("  Physiological range filter (per measurement, before hourly aggregation):")
            print(format_range_report(upstream))
        else:
            print("  warning: loader reports no per-measurement range filter; the bounds below act on aggregated values.")

        print("  Backstop pass over the assembled (N, T, F) block "
              "(expected to do nothing):")
        print(format_range_report(totals, total_cells=n_cells or None))


        if totals['n_filtered'] or totals['n_converted']:
            print(f"    warning: backstop nulled {totals['n_filtered']:,d} cells and converted {totals['n_converted']:,d}; "
                  f"some concept bypasses the per-measurement filter.")
        self.range_filter_report = totals
        self.range_filter_report_upstream = upstream or {}

    def _create_all_masks(self):
        for modality, data_dict in self.split_data.items():
            self.split_masks[modality] = {}
            for split in ['train', 'val', 'test']:
                if split not in data_dict:
                    continue
                data = data_dict[split]
                observed = (self.timeseries_observed.get(split)
                            if modality == 'timeseries' else None)
                if observed is not None:
                    survived = np.isfinite(to_numpy(data))
                    self.split_masks[modality][split] = (
                        np.asarray(observed) & survived).astype(np.float32)
                else:
                    self.split_masks[modality][split] = (~np.isnan(data)).astype(np.float32)

        print("Binary missingness masks created for all modalities")

    def _compute_modality_masks(self):
        self.split_modality_masks = {}


        self.modality_mask_columns: list = []
        for split in ['train', 'val', 'test']:
            cols, names = [], []
            for mod_key in self.active_modalities:
                if mod_key not in self.split_masks:
                    continue
                mask = self.split_masks[mod_key].get(split)
                if mask is None:
                    continue
                names.append(mod_key)


                axes = tuple(range(1, mask.ndim))
                if axes:
                    per_patient = (mask.sum(axis=axes) > 0).astype(np.uint8)
                else:
                    per_patient = mask.astype(np.uint8)
                cols.append(per_patient)
            if self.modality_mask_columns and names != self.modality_mask_columns:
                raise ValueError(
                    f"modality mask columns differ between splits: {self.modality_mask_columns} vs {names}")
            self.modality_mask_columns = names
            if cols:
                self.split_modality_masks[split] = np.stack(cols, axis=1).astype(np.uint8)
            else:
                self.split_modality_masks[split] = np.zeros((0, 0), dtype=np.uint8)

    def _apply_imputation(self):
        if 'timeseries' in self.split_data and self.feature_structure:
            dynamic_indices = self.feature_structure['dynamic_indices']
            static_indices = self.feature_structure['static_indices']


            train = self.split_data['timeseries']['train']
            val = self.split_data['timeseries']['val']
            test = self.split_data['timeseries']['test']
            n_train, n_val = train.shape[0], val.shape[0]
            stacked = np.concatenate([train, val, test], axis=0)
            train_idx_in_stack = np.arange(n_train)

            print(f"Imputation: aggregator ffill, then train-fold means for residual NaN "
                  f"({len(dynamic_indices)} dynamic, {len(static_indices)} static features)")

            stacked = train_mean_fallback(stacked, train_idx_in_stack, dynamic_indices)
            if static_indices:
                stacked = train_mean_fallback(stacked, train_idx_in_stack, static_indices)

            self.split_data['timeseries']['train'] = stacked[:n_train].astype(np.float32)
            self.split_data['timeseries']['val'] = stacked[n_train:n_train + n_val].astype(np.float32)
            self.split_data['timeseries']['test'] = stacked[n_train + n_val:].astype(np.float32)


        for modality in ['icd', 'radiology']:
            if modality in self.split_data:
                for split in ['train', 'val', 'test']:
                    self.split_data[modality][split] = np.nan_to_num(
                        self.split_data[modality][split], nan=0.0,
                    )


        impute_cfg = self.config.get('preprocessing', {}).get('imputation', {})
        if impute_cfg:
            warnings.warn(
                "preprocessing.imputation config keys are ignored; imputation is always ffill + train mean.",
                DeprecationWarning,
                stacklevel=2,
            )

        print("Imputation complete for all modalities")

    def _apply_scaling(self):
        scaling_cfg = self.config['preprocessing'].get('scaling', {})
        method = scaling_cfg.get('method', 'standard')
        z_clip = scaling_cfg.get('z_clip', 10.0)

        if method == 'none':
            print("Scaling disabled (method=none)")
            return
        if method != 'standard':
            print(f"  warning: scaling.method={method!r} not supported; using 'standard'.")


        if 'apply_to' in scaling_cfg:
            warnings.warn(
                "preprocessing.scaling.apply_to is ignored; scaling is always per modality.",
                DeprecationWarning,
                stacklevel=2,
            )

        if 'timeseries' not in self.split_data or not self.feature_structure:
            print("Scaling skipped (no timeseries modality)")
            return

        num_dynamic = self.feature_structure['num_dynamic']
        num_static = self.feature_structure['num_static']
        train = self.split_data['timeseries']['train']
        N_train, T, F = train.shape


        if num_dynamic > 0:
            self.scaler = StandardScaler()
            train_dyn = train[:, :, :num_dynamic].reshape(-1, num_dynamic)
            self.scaler.fit(train_dyn)
            print(f"Fitted dynamic StandardScaler on {num_dynamic} features ({N_train * T} train rows)")


            from preprocessing.ranges import (
                constant_feature_report, format_constant_report,
                format_inflation_report, scale_inflation_report,
            )


            train_mask = None
            ts_masks = self.split_masks.get('timeseries', {})
            if ts_masks.get('train') is not None:
                train_mask = (to_numpy(ts_masks['train'])[:, :, :num_dynamic]
                              .reshape(-1, num_dynamic))
            flagged = scale_inflation_report(
                to_numpy(train_dyn),
                self.feature_structure.get('dynamic_names', []),
                mask=train_mask,
            )
            print(format_inflation_report(flagged))
            self.scale_inflation = flagged


            constants = constant_feature_report(
                to_numpy(train_dyn),
                self.feature_structure.get('dynamic_names', []),
                mask=train_mask,
            )
            print(format_constant_report(constants))
            self.constant_features = constants


        if num_static > 0:
            self.scaler_static = StandardScaler()
            train_stat = train[:, :, num_dynamic:num_dynamic + num_static].reshape(-1, num_static)
            self.scaler_static.fit(train_stat)
            print(f"Fitted static StandardScaler on {num_static} features ({N_train * T} train rows)")
        else:
            self.scaler_static = None


        for split in ['train', 'val', 'test']:
            data = self.split_data['timeseries'][split]
            n_split = data.shape[0]

            if num_dynamic > 0:
                dyn = data[:, :, :num_dynamic].reshape(-1, num_dynamic)
                dyn = self.scaler.transform(dyn).reshape(n_split, T, num_dynamic)
                data[:, :, :num_dynamic] = dyn
            if num_static > 0:
                stat = data[:, :, num_dynamic:num_dynamic + num_static].reshape(-1, num_static)
                stat = self.scaler_static.transform(stat).reshape(n_split, T, num_static)
                data[:, :, num_dynamic:num_dynamic + num_static] = stat

            if z_clip is not None:
                z = float(z_clip)
                clipped_block = data[:, :, :num_dynamic + num_static]
                clipped_block = np.clip(clipped_block, -z, z)
                data[:, :, :num_dynamic + num_static] = clipped_block

            self.split_data['timeseries'][split] = data.astype(np.float32)

        if z_clip is not None:
            print(f"Z-clipped dynamic + static blocks at ±{z_clip}")
        print("Scaling complete (per-modality)")

    def _combine_modalities(self):

        if len(self.active_modalities) == 1 and 'timeseries' in self.active_modalities:
            self.split_data['combined'] = self.split_data['timeseries']
            return


        print("Multiple modalities detected - keeping separate")


    def _save_outputs(self):
        print(f"\nSaving outputs to: {self.output_dir}")


        from preprocessing.output_version import (
            assert_writable, resolve_version, write_marker,
        )


        version = resolve_version(self.config)


        from survbench.core.arms import arm_of
        arm, _censor_causes = arm_of(self.config)
        assert_writable(self.output_dir, version, arm=arm)

        output_files = self.config['output']['files']


        data_key = 'combined' if 'combined' in self.split_data else 'timeseries'


        for split in ['train', 'val', 'test']:
            x_path = self.output_dir / output_files[f'x_{split}']
            np.save(x_path, to_numpy(self.split_data[data_key][split]))
            print(f"  Saved {x_path.name}")


            mask_key = 'combined' if 'combined' in self.split_masks else 'timeseries'

            mask_file_key = f"{split}_mask"
            if mask_file_key in output_files:
                mask_path = self.output_dir / output_files[mask_file_key]
            else:

                mask_path = str(x_path).replace('.npy', '_mask.npy')
                print(f"  Warning: '{mask_file_key}' not in config. "
                      f"Saving mask to {Path(mask_path).name}")
            np.save(mask_path, to_numpy(self.split_masks[mask_key][split]))
            print(f"  Saved {Path(mask_path).name}")


        for modality in ['icd', 'radiology']:
            if modality in self.split_data:
                for split in ['train', 'val', 'test']:
                    mod_path = self.output_dir / f"x_{split}_{modality}.npy"
                    np.save(mod_path, to_numpy(self.split_data[modality][split]))
                    print(f"  Saved {mod_path.name}")


        for split in ['train', 'val']:
            y_path = self.output_dir / output_files[f'y_{split}']
            with open(y_path, 'wb') as f:
                pickle.dump(self.y_data[split], f)
            print(f"  Saved {y_path.name}")


        durations_path = self.output_dir / output_files['durations_test']
        events_path = self.output_dir / output_files['events_test']
        np.save(durations_path, to_numpy(self.y_data['test'][0]))
        np.save(events_path, to_numpy(self.y_data['test'][1]))
        print(f"  Saved {durations_path.name}")
        print(f"  Saved {events_path.name}")


        cuts_path = self.output_dir / output_files['cuts']
        out_features_path = self.output_dir / output_files['out_features']
        np.save(cuts_path, self.label_processor.get_cuts())
        np.save(out_features_path, np.int64(self.label_processor.get_n_bins()))
        print(f"  Saved {cuts_path.name}")
        print(f"  Saved {out_features_path.name}")


        if 'n_events' in output_files:
            n_events_path = self.output_dir / output_files['n_events']
            np.save(n_events_path, np.int64(self.label_processor.get_n_events()))
            print(f"  Saved {n_events_path.name}")


        feature_names_path = self.output_dir / output_files['feature_names']
        with open(feature_names_path, 'wb') as f:
            pickle.dump(self.feature_structure, f)
        print(f"  Saved {feature_names_path.name}")


        if 'modality_info' in output_files:
            modality_info_path = self.output_dir / output_files['modality_info']
            modality_info = {
                'active_modalities': self.active_modalities,
                'competing_risks': self.label_processor.is_competing_risks(),
                'n_events': self.label_processor.get_n_events(),


                'cause_codes': self.label_processor.get_cause_codes(),


                'modality_mask_columns': list(
                    getattr(self, 'modality_mask_columns', [])),
                'dataset_version': (
                    self.config.get('dataset', {}).get('version')
                    or getattr(getattr(self, 'loader', None), 'dataset_version', None)
                ),
            }
            with open(modality_info_path, 'wb') as f:
                pickle.dump(modality_info, f)
            print(f"  Saved {modality_info_path.name}")


        if self.scaler is not None:
            scaler_path = self.output_dir / output_files['scaler']
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"  Saved {scaler_path.name}")


        prefix = self.config['output'].get('prefix', '')

        if self.scaler is not None:
            scaler_dynamic_path = self.output_dir / f"scaler_dynamic_{prefix}.pkl"
            with open(scaler_dynamic_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"  Saved {scaler_dynamic_path.name}")

        if getattr(self, 'scaler_static', None) is not None:
            scaler_static_path = self.output_dir / f"scaler_static_{prefix}.pkl"
            with open(scaler_static_path, 'wb') as f:
                pickle.dump(self.scaler_static, f)
            print(f"  Saved {scaler_static_path.name}")

        if hasattr(self, 'split_modality_masks'):
            for split in ['train', 'val', 'test']:
                mm = self.split_modality_masks.get(split)
                if mm is None or mm.size == 0:
                    continue
                mm_path = self.output_dir / f"modality_mask_{split}_{prefix}.npy"
                np.save(mm_path, to_numpy(mm))
                print(f"  Saved {mm_path.name} {mm.shape}")

        if self.cohort_dfs is not None:
            for split in ['train', 'val', 'test']:
                for path in self.save_row_identity(
                        self.cohort_dfs[split], self.output_dir, split, prefix):
                    print(f"  Saved {path.name}")

        for split in ['train', 'val']:
            dur, _ = self.y_data[split]
            dur_path = self.output_dir / f"durations_{split}_{prefix}.npy"
            np.save(dur_path, to_numpy(np.asarray(dur, dtype=np.float64)))
            print(f"  Saved {dur_path.name}")


        conv_path = self.output_dir / f"unit_conversions_{prefix}.json"
        with open(conv_path, "w") as f:
            json.dump(getattr(self, 'unit_conversion_summary', {}) or {}, f, indent=2)
        print(f"  Saved {conv_path.name}")


        marker = write_marker(
            self.output_dir,
            version,
            dataset=self.config['dataset']['name'],


            discharge_as=arm,
            censor_causes=list(_censor_causes),


            dataset_version=(
                self.config.get('dataset', {}).get('version')
                or getattr(getattr(self, 'loader', None), 'dataset_version', None)
            ),
            landmarking=self.landmarking,
            n_rows={split: int(len(c)) for split, c in self.cohort_dfs.items()}
                    if self.cohort_dfs is not None else None,
            competing_risks=self.label_processor.is_competing_risks(),
            n_events=int(self.label_processor.get_n_events()),
            label_report=getattr(self.loader, '_label_report', None),
        )


        print(f"  Saved {marker.name} ({version})")

        self._print_summary()

    def _print_summary(self):
        print(f"\n{'-' * 80}")
        print("SUMMARY")
        print(f"{'-' * 80}")


        data_key = 'combined' if 'combined' in self.split_data else 'timeseries'

        for split in ['train', 'val', 'test']:
            data = self.split_data[data_key][split]
            mask_key = 'combined' if 'combined' in self.split_masks else 'timeseries'
            mask = self.split_masks[mask_key][split]
            durations, events = self.y_data[split]

            print(f"\n{split.upper()}:")
            print(f"  Shape: {data.shape} [N, T, F]")
            print(f"  N (patients): {data.shape[0]}")

            if len(data.shape) == 3:
                print(f"  T (time windows): {data.shape[1]}")
                print(f"  F (features): {data.shape[2]}")
                if self.feature_structure:
                    print(f"    - Dynamic: {self.feature_structure['num_dynamic']}")
                    print(f"    - Static: {self.feature_structure['num_static']}")


            n_dyn = (self.feature_structure or {}).get('num_dynamic', 0)
            if n_dyn and mask.ndim == 3 and mask.shape[2] >= n_dyn:
                dyn_missing = (mask[:, :, :n_dyn] == 0).mean() * 100
                print(f"  Missingness (dynamic block, {n_dyn} features): "
                      f"{dyn_missing:.2f}%")
                print(f"  Missingness (all {mask.shape[2]} columns, static included): {(mask == 0).mean() * 100:.2f}%")
            else:
                print(f"  Missingness: {(mask == 0).mean() * 100:.2f}%")

            if self.label_processor.is_competing_risks():
                print("  Competing risks distribution:")
                for event_code in np.unique(events):
                    count = (events == event_code).sum()
                    pct = count / len(events) * 100
                    outcome_name = self.config['dataset']['competing_risks']['outcome_mapping'].get(int(event_code),
                                                                                                    f"Event {event_code}")
                    print(f"    {outcome_name}: {count} ({pct:.1f}%)")
            else:
                print(f"  Event rate: {events.mean() * 100:.2f}%")


        for modality in ['icd', 'radiology']:
            if modality in self.split_data:
                print(f"\n{modality.upper()} modality:")
                for split in ['train', 'val', 'test']:
                    shape = self.split_data[modality][split].shape
                    print(f"  {split}: {shape}")
