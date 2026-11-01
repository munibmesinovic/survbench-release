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
    remove_constant_cols,
    static_input_kwargs,
)
from models.baselines._runtime import (
    WallClock,
    config_hash,
    dropout_rate,
    git_commit,
    layer_widths,
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


class DeepHit(nn.Module):

    def __init__(self, in_features: int, num_risks: int, num_bins: int,
                 shared_dims: list[int] = (256, 128),
                 cs_dims: list[int] = (128,),
                 dropout: float = 0.3, batch_norm: bool = True):
        super().__init__()
        self.num_risks = num_risks
        self.num_bins = num_bins


        shared_layers = []
        prev = in_features
        for h in shared_dims:
            shared_layers.append(nn.Linear(prev, h))
            if batch_norm:
                shared_layers.append(nn.BatchNorm1d(h))
            shared_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                shared_layers.append(nn.Dropout(dropout))
            prev = h
        self.shared = nn.Sequential(*shared_layers)
        shared_out_dim = prev


        cs_input_dim = shared_out_dim + in_features
        self.cs_nets = nn.ModuleList()
        for _ in range(num_risks):
            cs_layers = []
            prev = cs_input_dim
            for h in cs_dims:
                cs_layers.append(nn.Linear(prev, h))
                if batch_norm:
                    cs_layers.append(nn.BatchNorm1d(h))
                cs_layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    cs_layers.append(nn.Dropout(dropout))
                prev = h
            cs_layers.append(nn.Linear(prev, num_bins))
            self.cs_nets.append(nn.Sequential(*cs_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_out = self.shared(x)

        cs_input = torch.cat([shared_out, x], dim=-1)


        cs_outputs = [net(cs_input) for net in self.cs_nets]

        logits = torch.stack(cs_outputs, dim=1)


        B = logits.shape[0]
        flat_logits = logits.reshape(B, -1)
        flat_pmf = F.softmax(flat_logits, dim=-1)
        pmf = flat_pmf.reshape(B, self.num_risks, self.num_bins)
        return pmf


class DeepHitLoss(nn.Module):

    def __init__(self, alpha: float = 0.1, sigma: float = 0.1,
                 modality_info: dict | None = None):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.modality_info = modality_info or {}

    def forward(self, pmf: torch.Tensor, durations_idx: torch.Tensor,
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


        if self.alpha > 0 and uncensored.sum() > 1:
            rank_loss = self._ranking_loss(pmf, durations_idx, events)
        else:
            rank_loss = torch.tensor(0.0, device=pmf.device)

        return (nll + self.alpha * rank_loss, nll.detach(), rank_loss.detach())

    def _ranking_loss(self, pmf: torch.Tensor, durations_idx: torch.Tensor,
                      events: torch.Tensor) -> torch.Tensor:
        B, E, K = pmf.shape


        cif = pmf.cumsum(dim=-1)

        uncensored_idx = torch.where(events > 0)[0]
        n_unc = len(uncensored_idx)
        if n_unc < 2:
            return torch.tensor(0.0, device=pmf.device)


        n_pairs = min(n_unc * 20, 5000)
        i_sel = uncensored_idx[torch.randint(n_unc, (n_pairs,), device=pmf.device)]
        j_sel = torch.randint(B, (n_pairs,), device=pmf.device)


        valid = (i_sel != j_sel) & (durations_idx[i_sel] < durations_idx[j_sel])
        if valid.sum() < 1:
            return torch.tensor(0.0, device=pmf.device)

        i_v = i_sel[valid]
        j_v = j_sel[valid]
        t_i = durations_idx[i_v].long()


        table = torch.as_tensor(head_index_lookup(self.modality_info),
                                device=events.device)
        k_i = table[events[i_v].long()]
        if (k_i < 0).any():


            bad = sorted({int(c) for c in events[i_v][k_i < 0].tolist()})
            raise ValueError(
                f"events contain undeclared cause code(s) {bad}; modality_info['cause_codes'] is "
                f"{self.modality_info.get('cause_codes')}")


        cif_i = cif[i_v, k_i, t_i]
        cif_j = cif[j_v, k_i, t_i]
        loss = torch.exp(torch.clamp((cif_j - cif_i) / self.sigma, max=10.0)).mean()
        return loss


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
                cif=1.0 - surv_r.T, cif_times=np.asarray(time_index, dtype=float),
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


def train_model(
    x_train, dur_idx_train, evt_train,
    x_val, dur_idx_val, evt_val,
    num_risks: int, num_bins: int,
    shared_dims=(256, 128), cs_dims=(128,),
    dropout: float = 0.3, lr: float = 1e-3,
    weight_decay: float = 1e-4, alpha_rank: float = 0.1,
    epochs: int = 100, patience: int = 15,
    batch_size: int = 256, device: str = "auto",
    modality_info: dict | None = None,
) -> tuple:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    in_features = x_train.shape[1]
    model = DeepHit(
        in_features, num_risks, num_bins,
        shared_dims=shared_dims, cs_dims=cs_dims,
        dropout=dropout,
    ).to(device)
    criterion = DeepHitLoss(alpha=alpha_rank, modality_info=modality_info)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    effective: dict = {
        "lr": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "shared_dims": layer_widths(model.shared),
        "cs_dims": layer_widths(model.cs_nets[0])[:-1],
        "dropout": dropout_rate(model),
        "alpha_rank": float(criterion.alpha),
        "num_risks": int(model.num_risks),
        "num_bins": int(model.num_bins),
    }


    observed_batch_size = 0

    x_t = torch.from_numpy(x_train).float().to(device)
    d_t = torch.from_numpy(dur_idx_train).long().to(device)
    e_t = torch.from_numpy(evt_train).long().to(device)
    x_v = torch.from_numpy(x_val).float().to(device)
    d_v = torch.from_numpy(dur_idx_val).long().to(device)
    e_v = torch.from_numpy(evt_val).long().to(device)

    best_val_loss = float("inf")


    best_val_nll = float("inf")
    best_state = None
    wait = 0
    n_train = len(x_train)

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
            pmf = model(x_t[idx])


            loss, _, _ = criterion(pmf, d_t[idx], e_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_pmf = model(x_v)


            val_loss_t, val_nll_t, _ = criterion(val_pmf, d_v, e_v)
            val_loss = val_loss_t.item()
            val_nll = val_nll_t.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={avg_train_loss:.4f}/{val_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_nll = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
                    "epochs_run": epoch},
            effective)


def run(
    config: dict,
    *,
    shared_dims: str = "100,100",
    cs_dims: str = "100",
    dropout: float = 0.4,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    alpha_rank: float = 0.1,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 64,
    input_mode: str = "static",
    seed: int = 42,
    skip_test: bool = False,
    save_dir: Optional[Path] = None,
    save_tag: Optional[str] = None,
    **_ignored,
) -> dict:
    _reject_unexpected_kwargs(_ignored)
    model_name = "deephit"
    prefix = prefix_of(config)
    horizon = float(config.get("preprocessing", {}).get("max_horizon_hours", 240))
    dataset = dataset_of(config)


    train = load_split(config, "train", **static_input_kwargs(input_mode))
    val = load_split(config, "val", **static_input_kwargs(input_mode))


    test = None if skip_test else load_split(config, "test", **static_input_kwargs(input_mode))

    modality_info = train["modality_info"]
    competing = bool(modality_info.get("competing_risks", False))
    num_risks = int(modality_info.get("n_events", 1)) if competing else 1


    risk_axis = risk_axis_of(modality_info)
    cuts = train["cuts"]
    num_bins = len(cuts)

    x_train = train["x"]
    dur_idx_train = train["dur_idx"]
    evt_train = train["events"]
    dur_train = train["durations"]

    x_val = val["x"]
    dur_idx_val = val["dur_idx"]
    evt_val = val["events"]
    dur_val = val["durations"]

    if test is None:
        x_test = evt_test = dur_test = None
        lm_test = None
    else:
        x_test = test["x"]
        evt_test = test["events"]
        dur_test = test["durations"]
        lm_test = test["landmarks"]

    n_train = len(dur_train)
    n_val = len(dur_val)


    n_test = None if test is None else len(dur_test)
    n_events_train = int((evt_train > 0).sum())
    n_events_val = int((evt_val > 0).sum())
    n_events_test = None if test is None else int((evt_test > 0).sum())

    print(f"  Shapes: train={x_train.shape} val={x_val.shape} "
          f"test={'not loaded' if x_test is None else x_test.shape}")
    print(f"  Event rate (train): {(evt_train > 0).mean() * 100:.1f}%")
    print(f"  Bins (K): {num_bins}  Risks: {num_risks}")

    x_train, x_val, x_test, keep_mask = remove_constant_cols(
        x_train, x_val, x_test
    )
    n_dropped = int((~keep_mask).sum())
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} constant features "
              f"-> {x_train.shape[1]} remaining")


    shared_dims_list = (
        [int(d) for d in shared_dims.split(",")]
        if isinstance(shared_dims, str)
        else list(shared_dims)
    )
    cs_dims_list = (
        [int(d) for d in cs_dims.split(",")]
        if isinstance(cs_dims, str)
        else list(cs_dims)
    )

    with WallClock() as wc:
        model, train_info, effective_hparams = train_model(
            x_train, dur_idx_train, evt_train,
            x_val, dur_idx_val, evt_val,
            num_risks=num_risks, num_bins=num_bins,
            shared_dims=shared_dims_list, cs_dims=cs_dims_list,
            dropout=dropout, lr=lr, weight_decay=weight_decay,
            alpha_rank=alpha_rank, epochs=epochs, patience=patience,
            batch_size=batch_size, modality_info=modality_info,
        )
    wall_time_seconds = wc.seconds
    device = next(model.parameters()).device
    _param_count = param_count(model)

    inner: dict = {
        "model_info": {
            "type": "DeepHit",
            **train_info,
            "n_features": x_train.shape[1],
            "num_risks": num_risks,
            "num_bins": num_bins,
        }
    }

    surv_val_np: Optional[np.ndarray] = None
    surv_test_np: Optional[np.ndarray] = None

    scored = [("val", x_val, dur_val, evt_val, val["landmarks"])]
    if not skip_test:
        scored.append(("test", x_test, dur_test, evt_test, lm_test))
    for split_name, x_s, dur_s, evt_s, lm_s in scored:
        with torch.no_grad():
            pmf = model(
                torch.from_numpy(x_s).float().to(device)
            ).cpu().numpy()

        cif = pmf.cumsum(axis=-1)
        surv = 1.0 - cif
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
        "input_mode": input_mode,

        "val": inner.get("val", {}),
        "model_info": inner.get("model_info", {}),
    }
    if not skip_test:
        result["metrics"]["test"] = _flat_metrics(inner.get("test", {}))
        result["test"] = inner.get("test", {})
    return result
