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


class DySurv(nn.Module):

    def __init__(self, in_features: int, encoded_features: int,
                 out_features: int, seq_len: int, num_risks: int = 1):
        super().__init__()
        self.num_risks = num_risks
        self.in_features = in_features
        self.encoded_features = encoded_features


        self.lstm_enc = nn.LSTM(in_features, in_features, batch_first=True)
        self.fc_enc1 = nn.Linear(in_features, 3 * in_features)
        self.fc_enc2 = nn.Linear(3 * in_features, 5 * in_features)
        self.fc_enc3 = nn.Linear(5 * in_features, 3 * in_features)
        self.fc_mu = nn.Linear(3 * in_features, encoded_features)
        self.fc_logvar = nn.Linear(3 * in_features, encoded_features)


        self.surv_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoded_features, 3 * in_features), nn.ReLU(),
                nn.Linear(3 * in_features, 5 * in_features), nn.ReLU(),
                nn.Linear(5 * in_features, 3 * in_features), nn.ReLU(),
                nn.Linear(3 * in_features, out_features),
            )
            for _ in range(num_risks)
        ])


        self.decoder_lstm = nn.LSTM(encoded_features, 2 * encoded_features, batch_first=True)
        self.decoder_fc1 = nn.Linear(2 * encoded_features, 3 * encoded_features)
        self.decoder_fc2 = nn.Linear(3 * encoded_features, in_features)
        self.seq_len = seq_len

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def encode(self, x: torch.Tensor):
        out, _ = self.lstm_enc(x)

        h = out.mean(dim=1)
        h = self.relu(self.fc_enc1(h))
        h = self.relu(self.fc_enc2(h))
        h = self.relu(self.fc_enc3(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):

        z_seq = z.unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.decoder_lstm(z_seq)
        out = self.dropout(self.decoder_fc1(out))
        out = self.decoder_fc2(out)
        return out

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)


        if self.num_risks == 1:
            phi = self.surv_nets[0](z)
        else:
            phi = torch.stack([net(z) for net in self.surv_nets], dim=-1)


        x_recon = self.decode(z)

        return phi, x_recon, mu, logvar

    def predict_surv(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        if self.num_risks == 1:
            phi = self.surv_nets[0](z)
            hazard = torch.sigmoid(phi)
            surv = (1 - hazard).cumprod(dim=-1)
            return surv
        else:

            survs = []
            for net in self.surv_nets:
                phi = net(z)
                hazard = torch.sigmoid(phi)
                surv = (1 - hazard).cumprod(dim=-1)
                survs.append(surv)
            return torch.stack(survs, dim=-1)


class DySurvLoss(nn.Module):

    def __init__(self, alpha_surv: float = 1.0, alpha_ae: float = 1.0,
                 alpha_kl: float = 0.01, num_risks: int = 1,
                 pos_weight: float | None = None,
                 modality_info: dict | None = None):
        super().__init__()
        self.alpha_surv = alpha_surv
        self.alpha_ae = alpha_ae
        self.alpha_kl = alpha_kl
        self.num_risks = num_risks
        self.pos_weight = pos_weight


        self.modality_info = modality_info or {}

    def forward(self, phi, x_recon, mu, logvar, x_orig,
                durations_idx, events):

        if self.num_risks == 1:
            loss_surv = self._nll_logistic_hazard(phi, durations_idx, events)
        else:
            loss_surv = self._nll_competing_risk(phi, durations_idx, events)


        loss_ae = F.mse_loss(x_recon, x_orig)


        loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return (self.alpha_surv * loss_surv + self.alpha_ae * loss_ae + self.alpha_kl * loss_kl,
                loss_surv.detach(),
                torch.zeros((), device=phi.device))

    def _nll_logistic_hazard(self, phi, idx_durations, events):
        y_bce = torch.zeros_like(phi)
        events_f = events.float().view(-1, 1)
        idx = idx_durations.view(-1, 1).clamp(max=phi.shape[1] - 1)
        y_bce.scatter_(1, idx, events_f)

        pw = torch.tensor([self.pos_weight if self.pos_weight else 1.0], device=phi.device)
        bce = F.binary_cross_entropy_with_logits(phi, y_bce, pos_weight=pw, reduction='none')
        loss = bce.cumsum(1).gather(1, idx).view(-1)
        return loss.mean()

    def _nll_competing_risk(self, phi, idx_durations, events):

        B, K, E = phi.shape
        total_loss = torch.tensor(0.0, device=phi.device)
        pw = torch.tensor([self.pos_weight if self.pos_weight else 1.0], device=phi.device)

        head_to_code = cause_code_of_head(self.modality_info)


        table = torch.as_tensor(head_index_lookup(self.modality_info),
                                device=events.device)
        undeclared = (events > 0) & (table[events.long()] < 0)
        if undeclared.any():
            bad = sorted({int(c) for c in events[undeclared].tolist()})
            raise ValueError(
                f"events contain undeclared cause code(s) {bad}; modality_info['cause_codes'] is "
                f"{self.modality_info.get('cause_codes')}")

        for r in range(E):
            phi_r = phi[:, :, r]


            code_r = head_to_code[r]
            events_r = (events == code_r).float()
            y_bce = torch.zeros_like(phi_r)
            idx = idx_durations.view(-1, 1).clamp(max=K - 1)
            y_bce.scatter_(1, idx, events_r.view(-1, 1))
            bce = F.binary_cross_entropy_with_logits(phi_r, y_bce, pos_weight=pw, reduction='none')
            loss_r = bce.cumsum(1).gather(1, idx).view(-1).mean()
            total_loss = total_loss + loss_r

        return total_loss / E


def evaluate(
    surv_np: np.ndarray,
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
        if num_risks == 1:
            surv_r = surv_np
            mask = np.ones(len(events), dtype=bool)
            events_bin = (events > 0).astype(int)
            train_events_bin = (
                (train_events_full > 0).astype(int)
                if train_events_full is not None else None
            )
        else:
            surv_r = surv_np[:, :, r]


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
    surv_np: np.ndarray,
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
            surv_np[rows], durations[rows], events[rows], cuts, num_risks,
            horizon=stratum_horizon, auc_times=times, **kwargs,
        )

    return stratify_by_landmark(_stratum, landmarks, len(events),
                                horizon=horizon, dataset=dataset)


def train_model(
    x_train, didx_train, evt_train,
    x_val, didx_val, evt_val,
    num_risks: int, num_bins: int, seq_len: int,
    encoded_features: int = 20, lr: float = 1e-3,
    weight_decay: float = 1e-4, epochs: int = 100,
    patience: int = 15, batch_size: int = 256,
    device: str = "auto",
    modality_info: dict | None = None,
) -> tuple:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    in_features = x_train.shape[-1]
    model = DySurv(
        in_features, encoded_features, num_bins, seq_len,
        num_risks=num_risks,
    ).to(device)


    event_rate = (evt_train > 0).mean()
    pw = 1.0
    print(f"  pos_weight={pw:.1f} (event_rate={event_rate:.3f}, uniform default)")
    criterion = DySurvLoss(num_risks=num_risks, pos_weight=pw,
                           modality_info=modality_info)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    effective: dict = {
        "lr": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "encoded_features": int(model.fc_mu.out_features),
        "dropout": dropout_rate(model),
        "alpha_surv": float(criterion.alpha_surv),
        "alpha_ae": float(criterion.alpha_ae),
        "alpha_kl": float(criterion.alpha_kl),
        "pos_weight": (None if criterion.pos_weight is None
                       else float(criterion.pos_weight)),
        "num_risks": int(model.num_risks),
        "num_bins": int(model.surv_nets[0][-1].out_features),
    }
    observed_batch_size = 0

    x_t = torch.from_numpy(x_train).float().to(device)
    d_t = torch.from_numpy(didx_train).long().to(device)
    e_t = torch.from_numpy(evt_train).long().to(device)
    x_v = torch.from_numpy(x_val).float().to(device)
    d_v = torch.from_numpy(didx_val).long().to(device)
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
            phi, x_recon, mu, logvar = model(x_t[idx])

            loss, _, _ = criterion(phi, x_recon, mu, logvar, x_t[idx],
                                   d_t[idx], e_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            phi_v, xr_v, mu_v, lv_v = model(x_v)
            val_loss_t, val_nll_t, _ = criterion(phi_v, xr_v, mu_v, lv_v,
                                                 x_v, d_v, e_v)
            val_loss = val_loss_t.item()
            val_nll = val_nll_t.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}/{val_loss:.4f}  "
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
    encoded_features: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 64,
    seed: int = 42,
    skip_test: bool = False,
    save_dir: Optional[Path] = None,
    save_tag: Optional[str] = None,
    **_ignored,
) -> dict:
    _reject_unexpected_kwargs(_ignored)
    model_name = "dysurv"
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
    didx_train = train["dur_idx"]
    evt_train = train["events"]
    dur_train = train["durations"]

    x_val = val["x"]
    didx_val = val["dur_idx"]
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

    seq_len = x_train.shape[1]


    mu = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0
    x_train = (x_train - mu) / std
    x_val = (x_val - mu) / std
    if x_test is not None:
        x_test = (x_test - mu) / std

    print(f"  Shapes: train={x_train.shape} val={x_val.shape} "
          f"test={'not loaded' if x_test is None else x_test.shape}")
    print(f"  Event rate (train): {(evt_train > 0).mean() * 100:.1f}%")
    print(f"  Bins (K): {num_bins}  Risks: {num_risks}  Seq len: {seq_len}")

    with WallClock() as wc:
        model, train_info, effective_hparams = train_model(
            x_train, didx_train, evt_train,
            x_val, didx_val, evt_val,
            num_risks=num_risks, num_bins=num_bins, seq_len=seq_len,
            encoded_features=encoded_features,
            lr=lr, weight_decay=weight_decay,
            epochs=epochs, patience=patience,
            batch_size=batch_size, modality_info=modality_info,
        )
    wall_time_seconds = wc.seconds
    device = next(model.parameters()).device
    _param_count = param_count(model)

    inner: dict = {
        "model_info": {
            "type": "DySurv",
            **train_info,
            "n_features": x_train.shape[-1],
            "num_risks": num_risks,
            "num_bins": num_bins,
            "seq_len": seq_len,
        }
    }

    surv_val_np: Optional[np.ndarray] = None
    surv_test_np: Optional[np.ndarray] = None

    scored = [("val", x_val, dur_val, evt_val, val["landmarks"])]
    if not skip_test:
        scored.append(("test", x_test, dur_test, evt_test, lm_test))
    for split_name, x_s, dur_s, evt_s, lm_s in scored:
        with torch.no_grad():
            surv = model.predict_surv(
                torch.from_numpy(x_s).float().to(device)
            ).cpu().numpy()
        if split_name == "val":
            surv_val_np = surv
        else:
            surv_test_np = surv

        metrics = evaluate_by_landmark(
            surv, dur_s, evt_s, cuts, num_risks,
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

        "val": inner.get("val", {}),
        "model_info": inner.get("model_info", {}),
    }
    if not skip_test:
        result["metrics"]["test"] = _flat_metrics(inner.get("test", {}))
        result["test"] = inner.get("test", {})
    return result
