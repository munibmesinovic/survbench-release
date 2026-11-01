from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.baselines._data import (
    cause_code_of_head,
    dataset_of,
    head_index_lookup,
    load_split,
    patch_scipy_simps,
    prefix_of,
)
from models.baselines._runtime import (
    WallClock,
    config_hash,
    dropout_rate,
    git_commit,
    param_count,
    reject_unexpected_kwargs as _reject_unexpected_kwargs,
    RISK_AXIS_KEY,
    risk_axis_of,
    save_checkpoint,
    save_predictions,
    selection_losses,
)

patch_scipy_simps()

from models.baselines._eval import (
    _clinical_auc_times, compute_competing_risks_metrics, compute_metrics,
    stratify_by_landmark,
)


class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, rnn_outputs: torch.Tensor, last_hidden: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:

        last_exp = last_hidden.unsqueeze(1).expand_as(rnn_outputs)
        combined = torch.cat([rnn_outputs, last_exp], dim=-1)

        e = self.attn(combined).squeeze(-1)

        e = e.masked_fill(mask == 0, -1e9)
        a = F.softmax(e, dim=-1)

        context = (a.unsqueeze(-1) * rnn_outputs).sum(dim=1)
        return context


class DynamicDeepHit(nn.Module):

    def __init__(self, in_features: int, num_risks: int, num_bins: int,
                 rnn_hidden: int = 128, rnn_layers: int = 2,
                 cs_hidden: int = 128, cs_layers: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.num_risks = num_risks
        self.num_bins = num_bins
        self.in_features = in_features
        self.rnn_hidden = rnn_hidden


        self.rnn = nn.RNN(
            input_size=in_features,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )


        self.attention = TemporalAttention(rnn_hidden)


        self.rnn_pred = nn.Linear(rnn_hidden, in_features)


        combine_dim = in_features + rnn_hidden
        self.combine = nn.Sequential(
            nn.Linear(combine_dim, cs_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )


        self.cs_nets = nn.ModuleList()
        for _ in range(num_risks):
            layers = []
            prev = cs_hidden
            for _ in range(cs_layers):
                layers.extend([
                    nn.Linear(prev, cs_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ])
                prev = cs_hidden
            layers.append(nn.Linear(prev, num_bins))
            self.cs_nets.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, S, n_feat = x.shape
        if mask.shape != x.shape:
            raise ValueError(
                f"observation mask has shape {tuple(mask.shape)} but x has {tuple(x.shape)}")


        rnn_out, _ = self.rnn(x)
        rnn_last = rnn_out[:, -1, :]
        x_last = x[:, -1, :]

        if S > 1:

            rnn_history = rnn_out[:, :-1, :]


            history_mask = (mask[:, :-1, :].sum(dim=-1) > 0).float()
            context = self.attention(rnn_history, rnn_last, history_mask)
            z_pred = self.rnn_pred(rnn_history)
        else:

            context = torch.zeros_like(rnn_last)
            z_pred = torch.zeros(B, 0, n_feat, device=x.device)


        combined = torch.cat([x_last, context], dim=-1)
        shared = self.combine(combined)


        cs_outputs = [net(shared) for net in self.cs_nets]
        logits = torch.stack(cs_outputs, dim=1)


        flat_logits = logits.reshape(B, -1)
        flat_pmf = F.softmax(flat_logits, dim=-1)
        pmf = flat_pmf.reshape(B, self.num_risks, self.num_bins)

        return pmf, z_pred


class DynamicDeepHitLoss(nn.Module):

    def __init__(self, alpha: float = 0.1, gamma: float = 1.0, sigma: float = 0.1,
                 modality_info: dict | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma


        self.modality_info = modality_info or {}

    def forward(self, pmf: torch.Tensor, z_pred: torch.Tensor,
                x: torch.Tensor, durations_idx: torch.Tensor,
                events: torch.Tensor) -> tuple:
        B, E, K = pmf.shape


        ll_loss = torch.zeros(B, device=pmf.device)

        uncensored = events > 0
        censored = events == 0

        if uncensored.any():


            table = torch.as_tensor(head_index_lookup(self.modality_info),
                                    device=events.device)
            risk_idx = table[events[uncensored].long()]
            if (risk_idx < 0).any():
                bad = sorted({int(c) for c in
                              events[uncensored][risk_idx < 0].tolist()})
                raise ValueError(
                    f"events contain undeclared cause code(s) {bad}; modality_info['cause_codes'] is "
                    f"{self.modality_info.get('cause_codes')}")
            time_idx = durations_idx[uncensored].long()
            p_event = pmf[uncensored, risk_idx, time_idx]
            ll_loss[uncensored] = -torch.log(p_event.clamp(min=1e-8))

        if censored.any():
            time_idx_c = durations_idx[censored].long()
            cum_pmf = pmf[censored].cumsum(dim=-1)
            total_cif = cum_pmf.sum(dim=1)
            cif_at_t = total_cif.gather(1, time_idx_c.unsqueeze(1)).squeeze(1)
            surv_at_t = (1.0 - cif_at_t).clamp(min=1e-8)
            ll_loss[censored] = -torch.log(surv_at_t)

        nll = ll_loss.mean()


        rank_loss = torch.tensor(0.0, device=pmf.device)
        if self.alpha > 0 and uncensored.sum() > 1:
            rank_loss = self._ranking_loss(pmf, durations_idx, events)


        x_target = x[:, 1:, :]
        if z_pred.numel() > 0:
            rnn_loss = F.mse_loss(z_pred, x_target)
        else:
            rnn_loss = torch.tensor(0.0, device=pmf.device)

        return (nll + self.alpha * rank_loss + self.gamma * rnn_loss,
                nll.detach(), rank_loss.detach())

    def _ranking_loss(self, pmf: torch.Tensor, durations_idx: torch.Tensor,
                      events: torch.Tensor) -> torch.Tensor:
        B, E, K = pmf.shape
        cif = pmf.cumsum(dim=-1)
        unc_idx = torch.where(events > 0)[0]
        n_unc = len(unc_idx)
        if n_unc < 2:
            return torch.tensor(0.0, device=pmf.device)
        n_pairs = min(n_unc * 20, 5000)

        i_sel = unc_idx[torch.randint(n_unc, (n_pairs,), device=pmf.device)]
        j_sel = torch.randint(B, (n_pairs,), device=pmf.device)
        valid = (i_sel != j_sel) & (durations_idx[i_sel] < durations_idx[j_sel])
        if valid.sum() < 1:
            return torch.tensor(0.0, device=pmf.device)

        i_v, j_v = i_sel[valid], j_sel[valid]
        t_i = durations_idx[i_v].long()


        rank_table = torch.as_tensor(
            head_index_lookup(self.modality_info), device=events.device)
        k_i = rank_table[events[i_v].long()]
        if (k_i < 0).any():


            bad = sorted({int(c) for c in events[i_v][k_i < 0].tolist()})
            raise ValueError(
                f"events contain undeclared cause code(s) {bad}; modality_info['cause_codes'] is "
                f"{self.modality_info.get('cause_codes')}")
        return torch.exp(torch.clamp(
            (cif[j_v, k_i, t_i] - cif[i_v, k_i, t_i]) / self.sigma, max=10.0
        )).mean()


def evaluate(
    pmf: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    cuts: np.ndarray,
    num_risks: int,
    *,
    train_durations: np.ndarray | None = None,
    train_events_full: np.ndarray | None = None,
    horizon: float | None = None,
    auc_times: Sequence[float] | None = None,
    modality_info: dict | None = None,
) -> dict:
    time_index = cuts.astype(float)
    head_to_code = cause_code_of_head(modality_info or {})
    results: dict = {}
    ctds, ibss, iblls, aucs = [], [], [], []

    for r in range(num_risks):
        e_label = head_to_code[r]
        cif_r = pmf[:, r, :].cumsum(axis=-1)
        surv_r = 1.0 - cif_r

        if num_risks == 1:
            mask = np.ones(len(events), dtype=bool)
            events_bin = (events > 0).astype(int)
            train_events_bin = (
                (train_events_full > 0).astype(int)
                if train_events_full is not None else None
            )
        else:


            m = compute_competing_risks_metrics(
                cif=1.0 - surv_r.T,
                cif_times=np.asarray(time_index, dtype=float),
                durations=durations, events=events, cause=e_label,
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

        surv_df = pd.DataFrame(surv_r[mask].T, index=time_index)
        dur_r = durations[mask]

        m = compute_metrics(
            surv_df, dur_r, events_bin,
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
        if not np.isnan(m["auc_integrated"]):
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
        "auc_integrated": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_at_times": mean_auc_at_times,
    }
    return results


def evaluate_by_landmark(
    pmf: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    cuts: np.ndarray,
    num_risks: int,
    *,
    landmarks: np.ndarray | None = None,
    horizon: float | None = None,
    dataset: str | None = None,
    **kwargs,
) -> dict:
    def _stratum(rows, stratum_horizon, times):
        return evaluate(
            pmf[rows], durations[rows], events[rows], cuts, num_risks,
            horizon=stratum_horizon, auc_times=times, **kwargs,
        )

    return stratify_by_landmark(_stratum, landmarks, len(events),
                                horizon=horizon, dataset=dataset)


def _require_mask(split: dict, split_name: str, prefix: str) -> np.ndarray:
    m = split.get("m")
    if m is None:
        raise ValueError(
            f"the {split_name} split has no observation mask (x_{split_name}_{prefix}_mask.npy); "
            f"Dynamic-DeepHit needs a temporal load")
    return m


def _subset_train_rows(train_subset, *arrays):
    sub = np.asarray(train_subset, dtype=np.int64)
    if sub.size == 0:
        raise ValueError(
            "train_subset selects no rows")
    return tuple(a[sub] for a in arrays)


def train_model(
    x_train, m_train, didx_train, evt_train,
    x_val, m_val, didx_val, evt_val,
    num_risks: int, num_bins: int,
    rnn_hidden: int = 128, rnn_layers: int = 2,
    cs_hidden: int = 128,
    dropout: float = 0.3, lr: float = 1e-3,
    weight_decay: float = 1e-4, alpha: float = 0.1,
    gamma: float = 1.0, epochs: int = 100,
    patience: int = 15, batch_size: int = 256,
    device: str = "auto",
    modality_info: dict | None = None,
    init_from: Optional[Path] = None,
) -> tuple:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    in_features = x_train.shape[-1]
    model = DynamicDeepHit(
        in_features, num_risks, num_bins,
        rnn_hidden=rnn_hidden, rnn_layers=rnn_layers,
        cs_hidden=cs_hidden, dropout=dropout,
    ).to(device)
    criterion = DynamicDeepHitLoss(alpha=alpha, gamma=gamma,
                                   modality_info=modality_info)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    effective: dict = {
        "lr": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "rnn_hidden": int(model.rnn.hidden_size),
        "rnn_layers": int(model.rnn.num_layers),
        "cs_hidden": int(model.combine[0].out_features),
        "dropout": dropout_rate(model),
        "alpha": float(criterion.alpha),
        "gamma": float(criterion.gamma),
        "num_risks": int(model.num_risks),
        "num_bins": int(model.num_bins),
    }
    observed_batch_size = 0

    x_t = torch.from_numpy(x_train).float().to(device)
    m_t = torch.from_numpy(m_train).float().to(device)
    d_t = torch.from_numpy(didx_train).long().to(device)
    e_t = torch.from_numpy(evt_train).long().to(device)
    x_v = torch.from_numpy(x_val).float().to(device)
    m_v = torch.from_numpy(m_val).float().to(device)
    d_v = torch.from_numpy(didx_val).long().to(device)
    e_v = torch.from_numpy(evt_val).long().to(device)

    best_val_loss = float("inf")


    best_val_nll = float("inf")
    best_state = None
    wait = 0
    n_train = len(x_train)


    init_was_best = False
    init_val_loss = None
    if init_from is not None:
        state = torch.load(Path(init_from), map_location=device)
        model.load_state_dict(state.get("model", state))
        model.eval()
        with torch.no_grad():
            v_pmf, v_z = model(x_v, m_v)
            v_loss_t, v_nll_t, _ = criterion(v_pmf, v_z, x_v, d_v, e_v)
        best_val_loss = v_loss_t.item()
        best_val_nll = v_nll_t.item()
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        init_was_best = True


        init_val_loss = best_val_loss

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 8:
                continue
            observed_batch_size = max(observed_batch_size, len(idx))
            optimizer.zero_grad()


            pmf, z_pred = model(x_t[idx], m_t[idx])

            loss, _, _ = criterion(pmf, z_pred, x_t[idx], d_t[idx], e_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_pmf, val_z = model(x_v, m_v)


            val_loss_t, val_nll_t, _ = criterion(val_pmf, val_z, x_v, d_v, e_v)
            val_loss = val_loss_t.item()
            val_nll = val_nll_t.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={avg_train_loss:.4f}/{val_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_nll = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}


            init_was_best = False
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val loss={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    effective["batch_size"] = int(observed_batch_size)
    return (model, {"best_val_loss": float(best_val_loss),
                    "best_val_nll": float(best_val_nll),
                    "epochs_run": epoch,
                    "init_was_best": bool(init_was_best),
                    "init_val_loss": init_val_loss,
                    "n_train": int(n_train)},
            effective)


def run(
    config: dict,
    *,
    rnn_hidden: int = 100,
    rnn_layers: int = 2,
    cs_hidden: int = 100,
    dropout: float = 0.4,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 32,
    seed: int = 42,
    skip_test: bool = False,
    save_dir: Optional[Path] = None,
    save_tag: Optional[str] = None,
    init_from: Optional[Path] = None,
    train_subset: Optional[np.ndarray] = None,
    **_ignored,
) -> dict:
    _reject_unexpected_kwargs(_ignored)
    model_name = "dynamic_deephit"
    prefix = prefix_of(config)
    horizon = float(config.get("preprocessing", {}).get("max_horizon_hours", 240))
    dataset = dataset_of(config)


    train = load_split(config, "train")
    val = load_split(config, "val")


    test = None if skip_test else load_split(config, "test")

    modality_info = train["modality_info"]
    competing = bool(modality_info.get("competing_risks", False))
    num_risks = int(modality_info.get("n_events", 1)) if competing else 1


    risk_axis = risk_axis_of(modality_info)
    cuts = train["cuts"]
    num_bins = len(cuts)

    x_train = train["x"]
    m_train = _require_mask(train, "train", prefix)
    didx_train = train["dur_idx"]
    evt_train = train["events"]
    dur_train = train["durations"]


    if train_subset is not None:
        x_train, m_train, didx_train, evt_train, dur_train = _subset_train_rows(
            train_subset, x_train, m_train, didx_train, evt_train, dur_train)

    x_val = val["x"]
    m_val = _require_mask(val, "val", prefix)
    didx_val = val["dur_idx"]
    evt_val = val["events"]
    dur_val = val["durations"]

    if test is None:
        x_test = m_test = evt_test = dur_test = None
        lm_test = None
    else:
        x_test = test["x"]
        m_test = _require_mask(test, "test", prefix)
        evt_test = test["events"]
        dur_test = test["durations"]
        lm_test = test["landmarks"]

    n_train = len(dur_train)
    n_val = len(dur_val)


    n_test = None if test is None else len(dur_test)
    n_events_train = int((evt_train > 0).sum())
    n_events_val = int((evt_val > 0).sum())
    n_events_test = None if test is None else int((evt_test > 0).sum())


    mu = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0
    x_train = (x_train - mu) / std
    x_val = (x_val - mu) / std
    if x_test is not None:
        x_test = (x_test - mu) / std

    seq_len = x_train.shape[1]
    print(f"  Shapes: train={x_train.shape} val={x_val.shape} "
          f"test={'not loaded' if x_test is None else x_test.shape}")
    print(f"  Event rate (train): {(evt_train > 0).mean() * 100:.1f}%")
    print(f"  Bins (K): {num_bins}  Risks: {num_risks}  Time steps (S): {seq_len}")

    with WallClock() as wc:
        model, train_info, effective_hparams = train_model(
            x_train, m_train, didx_train, evt_train,
            x_val, m_val, didx_val, evt_val,
            num_risks=num_risks, num_bins=num_bins,
            rnn_hidden=rnn_hidden, rnn_layers=rnn_layers,
            cs_hidden=cs_hidden, dropout=dropout,
            lr=lr, weight_decay=weight_decay,
            alpha=alpha, gamma=gamma,
            epochs=epochs, patience=patience,
            batch_size=batch_size, modality_info=modality_info,
            init_from=init_from,
        )
    wall_time_seconds = wc.seconds
    device = next(model.parameters()).device
    _param_count = param_count(model)

    inner: dict = {
        "model_info": {
            "type": "DynamicDeepHit",
            **train_info,
            "n_features": x_train.shape[-1],
            "num_risks": num_risks,
            "num_bins": num_bins,
            "time_steps": seq_len,
        }
    }

    surv_val_np: Optional[np.ndarray] = None
    surv_test_np: Optional[np.ndarray] = None

    scored = [("val", x_val, m_val, dur_val, evt_val, val["landmarks"])]
    if not skip_test:
        scored.append(("test", x_test, m_test, dur_test, evt_test, lm_test))
    for split_name, x_s, m_s, dur_s, evt_s, lm_s in scored:
        with torch.no_grad():
            pmf, _ = model(torch.from_numpy(x_s).float().to(device),
                           torch.from_numpy(m_s).float().to(device))
            pmf = pmf.cpu().numpy()
        surv = 1.0 - pmf.cumsum(axis=-1)
        if split_name == "val":
            surv_val_np = surv
        else:
            surv_test_np = surv

        metrics = evaluate_by_landmark(
            pmf, dur_s, evt_s, cuts, num_risks,
            train_durations=dur_train,
            train_events_full=evt_train,
            horizon=horizon,
            dataset=dataset,
            modality_info=modality_info,
            landmarks=lm_s,
        )
        inner[split_name] = metrics
        m = metrics.get("mean", metrics.get("risk1", {}))
        print(f"  {split_name}: Ctd={m.get('ctd', 'n/a'):.4f}  "
              f"IBS={m.get('ibs', 'n/a'):.4f}  IBLL={m.get('ibll', 'n/a'):.4f}"
              f"  AUC={m.get('auc_integrated', float('nan')):.4f}")


    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = save_tag if save_tag is not None else f"{model_name}_{prefix}_seed{seed}"
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

    _repo_root = Path(__file__).resolve().parents[2]

    result: dict = {
        "model": model_name,
        "dataset": config.get("dataset", {}).get("name", prefix),
        "config_hash": config_hash(config),
        "git_commit": git_commit(_repo_root),
        "seed": seed,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_events_train": n_events_train,
        "n_events_val": n_events_val,
        "n_events_test": n_events_test,
        "horizon_hours": horizon,
        "n_time_bins": num_bins,
        "training": {
            "optimiser": "Adam",
            "lr": lr,
            "batch_size": batch_size,
            "epochs_run": train_info.get("epochs_run"),
            "early_stop_epoch": train_info.get("epochs_run"),
            "early_stop_patience": patience,
            "early_stop_metric": "val_loss",
            "wall_time_seconds": wall_time_seconds,
            "param_count": _param_count,
            "device": str(device),
        },


        "metrics": {"val": selection_losses(
            _flat_metrics(inner.get("val", {})),
            train_info["best_val_loss"], train_info["best_val_nll"])},


        "effective_hparams": effective_hparams,
        "notes": "",


        RISK_AXIS_KEY: risk_axis,


        "init_was_best": bool(train_info["init_was_best"]),
        "init_val_loss": train_info["init_val_loss"],
        "best_val_loss": float(train_info["best_val_loss"]),

        "val": inner.get("val", {}),
        "model_info": inner.get("model_info", {}),
    }
    if not skip_test:
        result["metrics"]["test"] = _flat_metrics(inner.get("test", {}))
        result["test"] = inner.get("test", {})
    return result
