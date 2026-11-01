from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional, Sequence

from models.baselines._runtime import (
    config_hash,
    git_commit,
    RISK_AXIS_KEY,
    risk_axis_of,
    save_checkpoint,
    save_predictions,
    selection_losses,
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from models.baselines._data import (
    apply_discharge_as,
    arm_of,
    cause_code_of_head,
    censor_beyond_grid,
    dataset_of,
    head_index_lookup,
    landmarks_for_split,
    load_feature_info,
    load_modality_info,
    output_dir_of,
    patch_scipy_simps,
    prefix_of,
)
from preprocessing.landmarking import landmarking_enabled

patch_scipy_simps()

from models.baselines._eval import (
    _clinical_auc_times, compute_competing_risks_metrics, compute_metrics,
    stratify_by_landmark,
)
from survbench.tuning.bottleneck import (
    PerModalityBottleneck, Width, echo_widths, requested_widths,
)


ACTIVE_TO_LOCAL: dict[str, str] = {
    "timeseries": "dynamic",
    "static": "static",
    "icd": "icd",
    "radiology": "radiology",
}


class TransformerSurv(nn.Module):

    def __init__(self, n_features: int, n_risks: int, n_bins: int,
                 d_model: int = 128, n_layers: int = 3, n_heads: int = 4,
                 ff_dim: int = 256, max_len: int = 100,
                 encoder_dropout: float = 0.1,
                 head_dropout: float = 0.3, head_hidden: int = 32,
                 aux_recon: bool = False,
                 pooling: str = "mean",
                 modality_slots: Optional[
                     Sequence[tuple[str, tuple[int, int]]]] = None,
                 bottleneck_widths: Optional[dict] = None):
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls'; got {pooling!r}")
        self.pooling = pooling


        widths = requested_widths(**dict(bottleneck_widths or {}))
        if widths and modality_slots is None:
            raise ValueError(
                f"bottleneck widths {sorted(widths)} need modality_slots (pass MultiModalDataset.modality_keys)")
        self.bottleneck = (PerModalityBottleneck(modality_slots, widths)
                           if widths else None)
        proj_in = n_features if self.bottleneck is None\
            else self.bottleneck.out_features
        self.input_proj = nn.Linear(proj_in, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        if pooling == "cls":

            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=encoder_dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.shared = nn.Linear(d_model, d_model)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.ReLU(),
                nn.BatchNorm1d(head_hidden),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, n_bins),
            ) for _ in range(n_risks)
        ])
        self.aux_recon = aux_recon


        self.in_features = n_features
        if aux_recon:
            self.next_step_head = nn.Linear(d_model, n_features)

    def forward(self, x: Tensor, mask: Tensor):

        B, T, _ = x.shape


        h = self.input_proj(x if self.bottleneck is None
                            else self.bottleneck(x))
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_embedding(positions)

        padding = (mask.sum(dim=-1) == 0)

        if self.pooling == "cls":

            cls = self.cls_token.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)
            cls_pad = torch.zeros(B, 1, dtype=padding.dtype, device=padding.device)
            padding = torch.cat([cls_pad, padding], dim=1)

        h = self.encoder(h, src_key_padding_mask=padding)


        if self.aux_recon and T > 1:
            tail = h[:, 1:, :] if self.pooling == "cls" else h
            x_next_pred = self.next_step_head(tail[:, :-1, :])
        else:
            x_next_pred = None

        if self.pooling == "cls":
            h_pool = h[:, 0, :]
        else:
            valid = (~padding).float().unsqueeze(-1)
            h_pool = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        z = self.shared(h_pool)
        phi = torch.stack([head(z) for head in self.heads], dim=1)
        return phi, x_next_pred


def pad_col(input, val=0, where="end"):
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == "end":
        return torch.cat([input, pad], dim=1)
    return torch.cat([pad, input], dim=1)


def class_weighted_nll_pmf_cr(phi, idx_durations, events, class_weights,
                               modality_info=None,
                               reduction="mean", epsilon=1e-7):


    table = torch.as_tensor(head_index_lookup(modality_info or {}),
                            device=events.device)
    raw_events = events.view(-1).long()
    events_remapped = table[raw_events]
    undeclared = (raw_events > 0) & (events_remapped < 0)
    if undeclared.any():
        bad = sorted({int(c) for c in raw_events[undeclared].tolist()})
        raise ValueError(
            f"events contain undeclared cause code(s) {bad}; modality_info['cause_codes'] is "
            f"{(modality_info or {}).get('cause_codes')}")
    events_remapped = events_remapped.int()
    event_01 = (events_remapped != -1).float()
    idx_durations = idx_durations.view(-1).int()
    B = phi.size(0)
    sm = pad_col(phi.view(B, -1)).softmax(1)[:, :-1].view(phi.shape)
    index = torch.arange(B, device=phi.device)
    idx_durations = idx_durations.clamp(0, sm.shape[2] - 1)


    prob_event = sm[index, events_remapped, idx_durations]
    loss_events = -prob_event.relu().add(epsilon).log().mul(event_01)
    S_t = (1 - sm.cumsum(2)[index, :, idx_durations].sum(1)).relu()
    loss_censored = -S_t.add(epsilon).log().mul(1 - event_01)
    loss = loss_events + loss_censored


    weights = class_weights[(events_remapped + 1).long()]
    loss = loss * weights
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def _diff_cdf_at_time_i(pmf, y):
    n = pmf.shape[0]
    ones = torch.ones((n, 1), device=pmf.device)
    r = pmf.cumsum(1).matmul(y.transpose(0, 1))
    diag_r = r.diag().view(1, -1)
    r = ones.matmul(diag_r) - r
    return r.transpose(0, 1)


def rank_loss_deephit_cr(phi, idx_durations, events, rank_mat, sigma,
                          modality_info=None, reduction="mean"):
    if events.dtype is torch.bool:
        events = events.float()
    phi = phi.float()


    table = torch.as_tensor(head_index_lookup(modality_info or {}),
                            device=events.device)
    raw_events = events.long().view(-1)
    events = table[raw_events]
    if ((raw_events > 0) & (events < 0)).any():


        bad = sorted({int(c) for c in
                      raw_events[(raw_events > 0) & (events < 0)].tolist()})
        raise ValueError(
            f"events contain undeclared cause code(s) {bad}; modality_info['cause_codes'] is "
            f"{(modality_info or {}).get('cause_codes')}")
    idx_durations = idx_durations.view(-1)
    B, n_risks = phi.shape[:2]
    pmf = pad_col(phi.view(B, -1)).softmax(1)[:, :-1].view(phi.shape)
    y = torch.zeros_like(pmf)
    idx_d = idx_durations.clamp(0, y.shape[2] - 1).long()
    y[torch.arange(B, device=y.device), :, idx_d] = 1.0
    losses = []
    for i in range(n_risks):
        r = _diff_cdf_at_time_i(pmf[:, i], y[:, i])
        loss_i = (rank_mat * torch.exp(-r / sigma)).mean(1, keepdim=True)
        losses.append(loss_i.view(-1) * (events == i).float())
    if reduction == "mean":
        return sum(lo.mean() for lo in losses)
    if reduction == "sum":
        return sum(lo.sum() for lo in losses)
    return sum(losses)


def pair_rank_mat(idx_durations, events, dtype="float32"):
    idx_durations = np.asarray(idx_durations).reshape(-1)
    events = np.asarray(events).reshape(-1)
    dur_i = idx_durations[:, None]
    dur_j = idx_durations[None, :]
    ev_i = events[:, None]
    ev_j = events[None, :]
    mat = ((ev_i > 0) & ((dur_i < dur_j) | ((dur_i == dur_j) & (ev_j == 0))))
    return mat.astype(dtype)


class SurvivalLossCR(nn.Module):
    def __init__(self, class_weights, alpha=0.5, sigma=0.1, rank_scale=3.0,
                 modality_info=None):
        super().__init__()
        self.register_buffer("cw", class_weights)
        self.alpha = alpha
        self.sigma = sigma
        self.rank_scale = rank_scale
        self.modality_info = modality_info or {}

    def forward(self, phi, durations, events, rank_mat=None):
        nll = class_weighted_nll_pmf_cr(phi, durations, events, self.cw,
                                        self.modality_info)
        if rank_mat is not None:
            rk_raw = rank_loss_deephit_cr(phi, durations, events, rank_mat,
                                           self.sigma, self.modality_info)
            rk = rk_raw * self.rank_scale
        else:
            rk_raw = torch.tensor(0.0, device=phi.device)
            rk = torch.tensor(0.0, device=phi.device)
        loss = self.alpha * nll + (1 - self.alpha) * rk
        return loss, nll.detach(), rk_raw.detach()


def _require_modality_file(path, modality: str, split: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"modality {modality!r} requested but {path} does not exist; "
            f"drop it from --modalities or regenerate the {split} split")


def _arm_schema(config: dict) -> dict:
    arm, censored = arm_of(config)
    _, modality_info = apply_discharge_as(
        np.zeros(0, dtype=np.int64), load_modality_info(config),
        discharge_as=arm, censor_causes=censored)
    return modality_info


class MultiModalDataset(Dataset):

    def __init__(self, config: dict, split: str,
                 modalities: Sequence[str] = ("dynamic", "static")):
        out = output_dir_of(config)
        prefix = prefix_of(config)
        feature_info = load_feature_info(config)
        modality_info = load_modality_info(config)


        x_full = np.load(out / f"x_{split}_{prefix}.npy").astype(np.float32)
        N, T, _ = x_full.shape


        keep: list[int] = []
        if "dynamic" in modalities:
            keep.extend(feature_info["dynamic_indices"])
        if "static" in modalities:
            keep.extend(feature_info["static_indices"])
        x_main = x_full[:, :, keep]

        n_dyn = len(feature_info["dynamic_indices"]) if "dynamic" in modalities else 0
        n_stat = len(feature_info["static_indices"]) if "static" in modalities else 0


        x_icd_path = out / f"x_{split}_icd.npy"
        n_icd = 0
        x_icd_t: Optional[np.ndarray] = None
        if "icd" in modalities:
            _require_modality_file(x_icd_path, "icd", split)
            x_icd = np.load(x_icd_path).astype(np.float32)
            n_icd = x_icd.shape[1]
            x_icd_t = np.tile(x_icd[:, None, :], (1, T, 1))


        x_rad_path = out / f"x_{split}_radiology.npy"
        n_rad = 0
        x_rad_t: Optional[np.ndarray] = None
        if "radiology" in modalities:
            _require_modality_file(x_rad_path, "radiology", split)
            x_rad = np.load(x_rad_path).astype(np.float32)
            n_rad = x_rad.shape[1]
            x_rad_t = np.tile(x_rad[:, None, :], (1, T, 1))


        parts = [x_main]
        if x_icd_t is not None:
            parts.append(x_icd_t)
        if x_rad_t is not None:
            parts.append(x_rad_t)
        x = np.concatenate(parts, axis=-1)


        modality_keys: list[tuple[str, tuple[int, int]]] = []
        cursor = 0
        for name, dim in [("dynamic", n_dyn), ("static", n_stat),
                           ("icd", n_icd), ("radiology", n_rad)]:
            if name in modalities and dim > 0:
                modality_keys.append((name, (cursor, cursor + dim)))
                cursor += dim
        self.modality_keys = modality_keys


        F_total = x.shape[-1]
        feat_mask = np.ones((N, T, F_total), dtype=np.float32)

        mod_mask_path = out / f"modality_mask_{split}_{prefix}.npy"
        if mod_mask_path.exists():
            mod_mask = np.load(mod_mask_path).astype(np.float32)
            active = modality_info.get("active_modalities", [])

            local_to_col: dict[str, int] = {}
            for col_idx, active_name in enumerate(active):
                local_name = ACTIVE_TO_LOCAL.get(active_name, active_name)
                local_to_col[local_name] = col_idx

            for local_name, (start, end) in modality_keys:
                col = local_to_col.get(local_name)
                if col is not None and col < mod_mask.shape[1]:

                    feat_mask[:, :, start:end] *= mod_mask[:, col, None, None]

        self.x = x
        self.m = feat_mask


        if split in ("train", "val"):
            label_path = out / f"y_{split}_surv_{prefix}.p"
            with open(label_path, "rb") as f:
                y = pickle.load(f)
            durations_raw = np.asarray(y[0], dtype=np.float64)
            events = np.asarray(y[1], dtype=np.int64)
        else:
            durations_raw = np.load(out / f"durations_test_{prefix}.npy").astype(np.float64)
            events = np.load(out / f"events_test_{prefix}.npy").astype(np.int64)

        cuts = np.load(out / f"cuts_{prefix}.npy")


        durations_raw, events = censor_beyond_grid(
            durations_raw, events, cuts,
            landmarked=landmarking_enabled(config))
        idx = np.searchsorted(cuts, durations_raw, side="right") - 1
        self.idx_durations = np.clip(idx, 0, len(cuts) - 1).astype(np.int64)


        arm, censored = arm_of(config)
        self.events, self.modality_info = apply_discharge_as(
            events, modality_info, discharge_as=arm, censor_causes=censored)
        self.durations = durations_raw


        self.landmarks = landmarks_for_split(config, split, prefix)
        if self.landmarks is not None and len(self.landmarks) != N:
            raise ValueError(
                f"landmarks_{split}_{prefix}.npy has {len(self.landmarks)} rows but x_{split}_{prefix}.npy has {N}")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.x[i]),
            torch.from_numpy(self.m[i]),
            torch.tensor(self.idx_durations[i], dtype=torch.long),
            torch.tensor(self.events[i], dtype=torch.long),
        )


def train_epoch(model, loader, opt, loss_fn, device, *, aux_weight: float = 0.0):
    model.train()
    n, totals = 0, dict(loss=0.0, nll=0.0, rank=0.0, aux=0.0)
    for x, m, dur, ev in loader:
        x, m, dur, ev = x.to(device), m.to(device), dur.to(device), ev.to(device)
        rm = pair_rank_mat(dur.cpu().numpy(), ev.cpu().numpy())
        rm = torch.from_numpy(rm).to(device)
        opt.zero_grad()
        phi, x_next_pred = model(x, m)
        loss, nll, rk = loss_fn(phi, dur, ev, rm)
        aux = torch.tensor(0.0, device=phi.device)
        if x_next_pred is not None and aux_weight > 0:


            target_mask = (m[:, 1:, :].sum(dim=-1) > 0).float().unsqueeze(-1)
            sq_err = (x_next_pred - x[:, 1:, :]).pow(2) * target_mask
            denom = target_mask.sum().clamp(min=1.0) * x.shape[-1]
            aux = sq_err.sum() / denom
            loss = loss + aux_weight * aux
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        totals["loss"] += loss.item()
        totals["nll"] += nll.item()
        totals["rank"] += rk.item()
        totals["aux"] += aux.item()
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def validate_epoch(model, loader, loss_fn, device, *, aux_weight: float = 0.0):
    model.eval()
    n, totals = 0, dict(loss=0.0, nll=0.0, rank=0.0, aux=0.0)
    nll_row_sum = 0.0
    n_rows_scored = 0
    n_rows_skipped = 0
    for x, m, dur, ev in loader:
        x, m, dur, ev = x.to(device), m.to(device), dur.to(device), ev.to(device)
        rm = pair_rank_mat(dur.cpu().numpy(), ev.cpu().numpy())
        rm = torch.from_numpy(rm).to(device)
        phi, x_next_pred = model(x, m)
        loss, nll, rk = loss_fn(phi, dur, ev, rm)
        aux = torch.tensor(0.0, device=phi.device)
        if x_next_pred is not None and aux_weight > 0:
            target_mask = (m[:, 1:, :].sum(dim=-1) > 0).float().unsqueeze(-1)
            sq_err = (x_next_pred - x[:, 1:, :]).pow(2) * target_mask
            denom = target_mask.sum().clamp(min=1.0) * x.shape[-1]
            aux = sq_err.sum() / denom
            loss = loss + aux_weight * aux
        if torch.isnan(loss):
            n_rows_skipped += len(dur)
            continue
        totals["loss"] += loss.item()
        totals["nll"] += nll.item()
        totals["rank"] += rk.item()
        totals["aux"] += aux.item()


        nll_row_sum += nll.item() * len(dur)
        n_rows_scored += len(dur)
        n += 1
    out = {k: v / max(n, 1) for k, v in totals.items()}
    out["nll_per_row"] = nll_row_sum / max(n_rows_scored, 1)
    out["n_rows_scored"] = n_rows_scored
    out["n_rows_skipped"] = n_rows_skipped
    return out


def cif_from_logits(preds: np.ndarray) -> np.ndarray:
    B = preds.shape[0]
    flat = torch.from_numpy(preds).view(B, -1)
    pad = torch.zeros_like(flat[:, :1])
    pmf_flat = torch.cat([flat, pad], dim=1).softmax(1)[:, :-1]
    pmf = pmf_flat.view(preds.shape).numpy()
    return np.cumsum(pmf, axis=2)


def evaluate(model, x_test, m_test, durations_test, events_test, cuts,
             num_risks, train_durations, train_events_full, device,
             batch_size=128, horizon: Optional[float] = None,
             modality_info: Optional[dict] = None,
             landmarks: Optional[np.ndarray] = None,
             dataset: Optional[str] = None):
    model.eval()
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(m_test))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for x, m in test_loader:
            phi, _ = model(x.to(device), m.to(device))
            preds.append(phi.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    cif = cif_from_logits(preds)
    surv_np = 1.0 - cif

    def _stratum(rows, stratum_horizon, times):
        return _score_cif(
            cif[rows], durations_test[rows], events_test[rows], cuts, num_risks,
            train_durations=train_durations,
            train_events_full=train_events_full,
            horizon=stratum_horizon, auc_times=times,
            modality_info=modality_info,
        )

    results = stratify_by_landmark(_stratum, landmarks, len(events_test),
                                   horizon=horizon, dataset=dataset)
    return results, surv_np


def _score_cif(cif, durations_test, events_test, cuts, num_risks, *,
               train_durations, train_events_full,
               horizon: Optional[float] = None,
               auc_times: Optional[Sequence[float]] = None,
               modality_info: Optional[dict] = None) -> dict:
    head_to_code = cause_code_of_head(modality_info or {})
    time_index = cuts.astype(float)
    results: dict = {}
    ctds, ibss, iblls, aucs = [], [], [], []
    for r in range(num_risks):
        e_label = head_to_code[r]
        if num_risks == 1:
            mask = np.ones(len(events_test), dtype=bool)
            events_bin = (events_test > 0).astype(int)
            train_events_bin = (train_events_full > 0).astype(int)
        else:


            m = compute_competing_risks_metrics(
                cif=cif[:, r, :].T,
                cif_times=np.asarray(time_index, dtype=float),
                durations=durations_test, events=events_test, cause=e_label,
                times=(auc_times if auc_times is not None
                       else _clinical_auc_times(horizon)),
                horizon=horizon,
            )
            results[f"risk{e_label}"] = m
            if not np.isnan(m["ibs"]):
                ibss.append(m["ibs"])
            if not np.isnan(m["ctd"]):
                ctds.append(m["ctd"])
            if not np.isnan(m["ibll"]):
                iblls.append(m["ibll"])
            if not np.isnan(m["auc_mean"]):
                aucs.append(m["auc_mean"])
            continue


        surv_r = 1.0 - cif[:, r, :]
        surv_df = pd.DataFrame(surv_r[mask].T, index=time_index)

        m = compute_metrics(
            surv_df, durations_test[mask], events_bin,
            train_durations=train_durations,
            train_events_bin=train_events_bin,
            horizon=horizon,
            auc_times=auc_times,
        )
        results[f"risk{e_label}"] = m
        if not np.isnan(m["ctd"]):
            ctds.append(m["ctd"])
            ibss.append(m["ibs"])
            iblls.append(m["ibll"])
            aucs.append(m["auc_integrated"])


    from collections import defaultdict as _dd
    _aat_pool = _dd(list)
    for r in range(num_risks):
        e_label = head_to_code[r]
        risk_key = f"risk{e_label}"
        if risk_key not in results:
            continue
        aat = results[risk_key].get("auc_at_times") or {}
        for t_key, val in aat.items():
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            _aat_pool[t_key].append(float(val))
    mean_auc_at_times = {k: float(np.mean(v)) for k, v in _aat_pool.items()}

    results["mean"] = {
        "ctd": float(np.mean(ctds)) if ctds else float("nan"),
        "ibs": float(np.mean(ibss)) if ibss else float("nan"),
        "ibll": float(np.mean(iblls)) if iblls else float("nan"),
        "auc_integrated": float(np.mean([a for a in aucs if not np.isnan(a)])) if aucs else float("nan"),
        "auc_mean": float(np.mean([a for a in aucs if not np.isnan(a)])) if aucs else float("nan"),
        "auc_at_times": mean_auc_at_times,
    }
    return results


def run(config: dict, *, modalities: Sequence[str] = ("dynamic", "static"),
        seed: int = 42, epochs: int = 100, patience: int = 12,
        batch_size: int = 128, lr: float = 1e-4, weight_decay: float = 1e-5,
        alpha: float = 0.5, rank_scale: float = 3.0, sigma: float = 0.1,
        d_model: int = 128, n_layers: int = 3, n_heads: int = 4,
        ff_dim: int = 256, encoder_dropout: float = 0.1,
        head_dropout: float = 0.3, head_hidden: int = 32,
        class_weights=None,
        aux_recon: bool = False,
        aux_weight: float = 0.0,
        pooling: str = "mean",
        icd_bottleneck: Width = "none",
        radiology_bottleneck: Width = "none",
        skip_test: bool = False,
        save_dir: Optional[Path] = None,
        save_tag: Optional[str] = None,
        init_from: Optional[Path] = None,
        train_subset: Optional[np.ndarray] = None) -> dict:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = tuple(modalities)

    out = output_dir_of(config)
    prefix = prefix_of(config)
    modality_info = _arm_schema(config)
    num_risks: int = modality_info["n_events"]


    risk_axis = risk_axis_of(modality_info)

    cuts = np.load(out / f"cuts_{prefix}.npy")
    out_features = int(np.load(out / f"out_features_{prefix}.npy"))


    train_ds = MultiModalDataset(config, "train", modalities)
    val_ds = MultiModalDataset(config, "val", modalities)

    n_features = train_ds.x.shape[-1]
    T = train_ds.x.shape[1]
    print(
        f"[transformer/{prefix}] "
        f"{len(train_ds)}/{len(val_ds)} train/val "
        f"| F={n_features} T={T} bins={out_features} risks={num_risks} "
        f"modalities={list(modalities)}"
    )


    n_train_rows = len(train_ds)
    train_view = train_ds
    if train_subset is not None:
        sub = np.asarray(train_subset, dtype=np.int64)
        if sub.size == 0:
            raise ValueError(
                "train_subset selects no rows")
        train_view = torch.utils.data.Subset(train_ds, sub.tolist())
        n_train_rows = len(train_view)

    train_loader = DataLoader(train_view, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


    if len(train_loader) == 0:
        raise ValueError(
            f"batch_size={batch_size} with {n_train_rows} training rows and drop_last=True gives no batches")


    if class_weights is None:


        class_weights = [1.0] * (num_risks + 1)

    cw = torch.tensor(class_weights, dtype=torch.float32).to(device)


    bottleneck_widths = {"icd": icd_bottleneck,
                         "radiology": radiology_bottleneck}
    model = TransformerSurv(
        n_features=n_features, n_risks=num_risks, n_bins=out_features,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        ff_dim=ff_dim, encoder_dropout=encoder_dropout,
        head_dropout=head_dropout, head_hidden=head_hidden,
        aux_recon=aux_recon,
        pooling=pooling,
        modality_slots=train_ds.modality_keys,
        bottleneck_widths=bottleneck_widths,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  TransformerSurv: {n_params:,} params  "
          f"layers={n_layers} heads={n_heads} d_model={d_model}")


    if model.bottleneck is not None:
        print(f"  bottleneck: {model.bottleneck.widths} -> input_proj sees "
              f"{model.input_proj.in_features} of {n_features} columns")

    if init_from is not None:
        state = torch.load(Path(init_from), map_location=device)
        model.load_state_dict(state.get("model", state)
                              if isinstance(state, dict) else state)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                     factor=0.5, patience=7)
    loss_fn = SurvivalLossCR(cw, alpha=alpha, sigma=sigma,
                              rank_scale=rank_scale,
                              modality_info=modality_info).to(device)


    effective_hparams: dict = {
        "lr": float(opt.param_groups[0]["lr"]),
        "weight_decay": float(opt.param_groups[0]["weight_decay"]),
        "batch_size": int(train_loader.batch_size),
        "d_model": int(model.input_proj.out_features),
        "n_layers": len(model.encoder.layers),
        "n_heads": int(model.encoder.layers[0].self_attn.num_heads),
        "ff_dim": int(model.encoder.layers[0].linear1.out_features),
        "encoder_dropout": float(model.encoder.layers[0].dropout.p),
        "head_hidden": int(model.heads[0][0].out_features),
        "head_dropout": float(
            next(m.p for m in model.heads[0] if isinstance(m, nn.Dropout))),
        "alpha": float(loss_fn.alpha),
        "sigma": float(loss_fn.sigma),
        "rank_scale": float(loss_fn.rank_scale),


        "pooling": "cls" if hasattr(model, "cls_token") else "mean",
        "aux_recon": hasattr(model, "next_step_head"),
        "num_risks": len(model.heads),
        "num_bins": int(model.heads[0][-1].out_features),


        **echo_widths(model.bottleneck),


        "input_width": int(model.input_proj.in_features),
    }


    best_val = float("inf")


    best_val_nll = float("inf")


    best_val_nll_rows_skipped = 0
    best_epoch = 0
    best_state: Optional[dict] = None
    no_improve = 0
    t0 = time.time()


    init_was_best = False
    init_val_loss = None
    if init_from is not None:
        va0 = validate_epoch(model, val_loader, loss_fn, device,
                             aux_weight=aux_weight)
        best_val = va0["loss"]
        best_val_nll = va0["nll_per_row"]
        best_val_nll_rows_skipped = va0["n_rows_skipped"]
        best_epoch = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        init_was_best = True


        init_val_loss = best_val

    for epoch in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, opt, loss_fn, device,
                         aux_weight=aux_weight)
        va = validate_epoch(model, val_loader, loss_fn, device,
                            aux_weight=aux_weight)
        sch.step(va["loss"])
        saved = False


        if va["loss"] < best_val:
            best_val = va["loss"]
            best_val_nll = va["nll_per_row"]
            best_val_nll_rows_skipped = va["n_rows_skipped"]
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}


            init_was_best = False
            saved = True
        else:
            no_improve += 1
        if epoch == 1 or saved or epoch % 5 == 0:
            aux_str = f" Aux={tr['aux']:.4f}" if aux_weight > 0 else ""
            print(
                f"  E{epoch:03d} TrL={tr['loss']:.4f} NLL={tr['nll']:.4f} "
                f"R={tr['rank']:.4f}{aux_str} VL={va['loss']:.4f} "
                f"lr={opt.param_groups[0]['lr']:.2e}"
                + (" *" if saved else "")
            )
        if no_improve >= patience:
            print(f"  Early stop @ E{epoch} (best E{best_epoch})")
            break

    elapsed = time.time() - t0
    epochs_run = epoch
    print(f"  Training: {elapsed:.0f}s  best_epoch={best_epoch}")


    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})


    test_ds = None if skip_test else MultiModalDataset(config, "test", modalities)
    if test_ds is None:
        x_test = m_test = None
    else:
        x_test = test_ds.x.astype(np.float32)
        m_test = test_ds.m.astype(np.float32)


    if test_ds is None:
        durations_test = events_test = None
    else:
        durations_test = test_ds.durations.astype(np.float64)
        events_test = test_ds.events.astype(np.int64)

    x_val = val_ds.x.astype(np.float32)
    m_val = val_ds.m.astype(np.float32)
    durations_val = val_ds.durations.astype(np.float64)
    events_val = val_ds.events.astype(np.int64)

    train_durations = train_ds.durations.astype(np.float64)
    train_events_full = train_ds.events.astype(np.int64)

    horizon = float(config.get("preprocessing", {}).get("max_horizon_hours", 240))
    dataset = dataset_of(config)


    val_metrics, surv_val_np = evaluate(
        model, x_val, m_val, durations_val, events_val, cuts,
        num_risks, train_durations, train_events_full, device, batch_size,
        horizon=horizon, modality_info=modality_info,
        landmarks=val_ds.landmarks, dataset=dataset,
    )
    test_metrics: Optional[dict] = None
    surv_test_np: Optional[np.ndarray] = None
    if not skip_test:
        test_metrics, surv_test_np = evaluate(
            model, x_test, m_test, durations_test, events_test, cuts,
            num_risks, train_durations, train_events_full, device, batch_size,
            horizon=horizon, modality_info=modality_info,
            landmarks=test_ds.landmarks, dataset=dataset,
        )

    inner: dict = {
        "model_info": {
            "type": "TransformerSurv",
            "n_features": n_features,
            "num_risks": num_risks,
            "num_bins": out_features,
            "modalities": list(modalities),
            "best_val_loss": float(best_val),
            "best_val_nll": float(best_val_nll),
            "best_val_nll_rows_skipped": int(best_val_nll_rows_skipped),
            "epochs_run": epochs_run,
            "n_params": n_params,
        },
        "val": val_metrics,
        "test": test_metrics,
    }


    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = save_tag if save_tag is not None else f"transformer_{prefix}_seed{seed}"
        save_predictions(
            save_dir / f"{tag}_preds.npz",
            cuts=cuts,
            cause_codes=risk_axis,
            surv_val=surv_val_np,
            surv_test=surv_test_np,
        )
        save_checkpoint(save_dir / f"{tag}.pt", model)

    def _flat_metrics(split_dict: dict) -> dict:
        if "mean" in split_dict:
            return split_dict["mean"]
        if "risk1" in split_dict:
            return split_dict["risk1"]
        return split_dict


    n_train_samples = n_train_rows
    n_val_samples = len(val_ds)
    n_events_train = int((train_ds.events > 0).sum())
    n_events_val = int((val_ds.events > 0).sum())


    n_test_samples = None if test_ds is None else len(test_ds)
    n_events_test = (None if test_ds is None
                     else int((test_ds.events > 0).sum()))

    _repo_root = Path(__file__).resolve().parents[2]

    result: dict = {
        "model": "transformer",
        "dataset": config.get("dataset", {}).get("name", prefix),
        "config_hash": config_hash(config),
        "git_commit": git_commit(_repo_root),
        "seed": seed,
        "n_train": n_train_samples,
        "n_val": n_val_samples,
        "n_test": n_test_samples,
        "n_events_train": n_events_train,
        "n_events_val": n_events_val,
        "n_events_test": n_events_test,
        "horizon_hours": horizon,
        "n_time_bins": out_features,
        "training": {
            "optimiser": "AdamW",
            "lr": lr,
            "batch_size": batch_size,
            "epochs_run": epochs_run,
            "early_stop_epoch": best_epoch,
            "early_stop_patience": patience,
            "early_stop_metric": "val_loss",
            "wall_time_seconds": elapsed,
            "param_count": n_params,
            "device": str(device),
            "aux_recon": aux_recon,
            "aux_weight": aux_weight,
        },


        "metrics": {"val": selection_losses(_flat_metrics(val_metrics),
                                            best_val, best_val_nll)},


        "effective_hparams": effective_hparams,
        "notes": "",


        RISK_AXIS_KEY: risk_axis,


        "n_batches": int(len(train_loader)),
        "init_was_best": bool(init_was_best),
        "init_val_loss": init_val_loss,
        "best_val_loss": float(best_val),

        "val": val_metrics,
        "model_info": inner["model_info"],
    }
    if not skip_test:
        result["metrics"]["test"] = _flat_metrics(test_metrics)
        result["test"] = test_metrics
    return result
