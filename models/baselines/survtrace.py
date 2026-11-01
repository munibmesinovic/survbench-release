from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from models.baselines._data import (
    cause_code_of_head,
    cause_codes_of,
    dataset_of,
    load_split,
    patch_scipy_simps,
    prefix_of,
    remove_constant_cols,
    static_input_kwargs,
)
from models.baselines._runtime import (
    WallClock,
    config_hash,
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


from models.baselines.deephit import evaluate_by_landmark


UPSTREAM = ("https://github.com/RyanWangZf/SurvTRACE@"
            "e6b354fd61d9505f4026a92df7bd83a5dfd24ff4")

UPSTREAM_LICENSE = """\
MIT License

Copyright (c) 2021 Maximum

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


_PREDICT_BATCH = 10000


class BertEmbeddings(nn.Module):

    def __init__(self, num_feature: int, hidden_size: int,
                 hidden_dropout_prob: float, initializer_range: float):
        super().__init__()
        self.num_embeddings = nn.Parameter(
            torch.randn(1, num_feature, hidden_size), requires_grad=True)
        self.num_embeddings.data.normal_(mean=0.0, std=initializer_range)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_x_num: torch.Tensor) -> torch.Tensor:
        embeddings = torch.unsqueeze(input_x_num, 2) * self.num_embeddings
        return self.dropout(embeddings)


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int,
                 attention_probs_dropout_prob: float):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the "
                f"number of attention heads ({num_attention_heads})")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(*new_shape)


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size: int, hidden_dropout_prob: float,
                 layer_norm_eps: float):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.LayerNorm(hidden_states + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                 attention_probs_dropout_prob, hidden_dropout_prob,
                 layer_norm_eps):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads,
                                      attention_probs_dropout_prob)
        self.output = BertSelfOutput(hidden_size, hidden_dropout_prob,
                                     layer_norm_eps)

    def forward(self, hidden_states):
        return self.output(self.self(hidden_states), hidden_states)


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        return F.gelu(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self, intermediate_size: int, hidden_size: int,
                 hidden_dropout_prob: float, layer_norm_eps: float):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.LayerNorm(hidden_states + input_tensor)


class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size,
                 attention_probs_dropout_prob, hidden_dropout_prob,
                 layer_norm_eps):
        super().__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads,
                                       attention_probs_dropout_prob,
                                       hidden_dropout_prob, layer_norm_eps)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(intermediate_size, hidden_size,
                                 hidden_dropout_prob, layer_norm_eps)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        return self.output(self.intermediate(attention_output),
                           attention_output)


class DenseVanillaBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.kaiming_normal_(self.linear.weight.data, nonlinearity="relu")
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.batch_norm(self.activation(self.linear(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class BertCLSMulti(nn.Module):

    def __init__(self, hidden_size: int, num_feature: int,
                 intermediate_size: int, out_feature: int, num_event: int,
                 hidden_dropout_prob: float):
        super().__init__()
        self.net = nn.Sequential(DenseVanillaBlock(
            hidden_size * num_feature, intermediate_size,
            dropout=hidden_dropout_prob))
        self.net_out = nn.ModuleList(
            [nn.Linear(intermediate_size, out_feature)
             for _ in range(num_event)])

    def forward(self, hidden_states, event: int = 0):
        hidden_states = self.net(hidden_states.flatten(start_dim=1))
        return self.net_out[event](hidden_states)


class SurvTraceMulti(nn.Module):

    def __init__(self, num_feature: int, num_event: int, out_feature: int,
                 hidden_size: int = 16, num_hidden_layers: int = 2,
                 num_attention_heads: int = 2, intermediate_size: int = 64,
                 hidden_dropout_prob: float = 0.0,
                 attention_probs_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12):
        super().__init__()
        self.num_feature = num_feature
        self.num_event = num_event
        self.out_feature = out_feature
        self.initializer_range = initializer_range
        self.embeddings = BertEmbeddings(num_feature, hidden_size,
                                         hidden_dropout_prob, initializer_range)
        self.encoder = nn.ModuleList([
            BertLayer(hidden_size, num_attention_heads, intermediate_size,
                      attention_probs_dropout_prob, hidden_dropout_prob,
                      layer_norm_eps)
            for _ in range(num_hidden_layers)])
        self.cls = BertCLSMulti(hidden_size, num_feature, intermediate_size,
                                out_feature, num_event, hidden_dropout_prob)


        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.embeddings(x)
        for layer in self.encoder:
            hidden = layer(hidden)
        return hidden

    def forward(self, x: torch.Tensor, event: Optional[int] = None):
        hidden = self.encode(x)
        if event is not None:
            return self.cls(hidden, event=event)
        return torch.stack([self.cls(hidden, event=k)
                            for k in range(self.num_event)], dim=1)


def pc_hazard_labels(durations, events, cuts):
    cuts = np.asarray(cuts, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64).copy()
    if not (np.diff(cuts) > 0).all():
        raise ValueError("Cuts are not unique.")
    top = cuts.max()
    clipped = durations.copy()
    beyond = clipped > top
    clipped[beyond] = top
    events[beyond] = 0
    idx = np.searchsorted(cuts, clipped, side="left")
    dur_disc = cuts[idx]
    cut_diff = np.diff(cuts)
    frac = 1.0 - (dur_disc - durations) / cut_diff[idx - 1]
    at_start = idx == 0
    frac[at_start] = 0.0
    events[at_start] = 0
    idx = idx - 1
    idx[idx < 0] = 0
    return (idx.astype(np.int64), events.astype(np.int64),
            frac.astype(np.float32))


def log_softplus(input: torch.Tensor, threshold: float = -15.0) -> torch.Tensor:
    output = input.clone()
    above = input >= threshold
    output[above] = F.softplus(input[above]).log()
    return output


def nll_pc_hazard_loss(phi: torch.Tensor, idx_durations: torch.Tensor,
                       events: torch.Tensor, interval_frac: torch.Tensor,
                       reduction: str = "mean") -> torch.Tensor:
    events = events.float().view(-1)
    idx_durations = idx_durations.view(-1, 1)
    interval_frac = interval_frac.view(-1)
    keep = idx_durations.view(-1) >= 0
    phi = phi[keep, :]
    idx_durations = idx_durations[keep, :]
    events = events[keep]
    interval_frac = interval_frac[keep]

    log_h_e = log_softplus(phi.gather(1, idx_durations).view(-1)).mul(events)
    haz = F.softplus(phi)
    scaled_h_e = haz.gather(1, idx_durations).view(-1).mul(interval_frac)
    haz = torch.cat([torch.zeros_like(haz[:, :1]), haz], dim=1)
    sum_haz = haz.cumsum(1).gather(1, idx_durations).view(-1)
    loss = -log_h_e.sub(scaled_h_e).sub(sum_haz)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid.")


def pc_hazard_loss(phis, idx, events, frac, cause_codes) -> torch.Tensor:
    total = None
    for k, code in enumerate(cause_codes):
        term = nll_pc_hazard_loss(phis[k], idx, (events == code).float(), frac)
        total = term if total is None else total + term
    return total


def pmf_from_hazards(eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=np.float64)
    lam = eta.sum(axis=1)
    cum_before = np.cumsum(lam, axis=1) - lam
    surv_start = np.exp(-cum_before)

    safe = lam > 1e-12
    ratio = np.ones_like(lam)
    ratio[safe] = -np.expm1(-lam[safe]) / lam[safe]
    mass = surv_start[:, None, :] * eta * ratio[:, None, :]
    pad = np.zeros(mass.shape[:2] + (1,), dtype=np.float64)
    return np.concatenate([mass, pad], axis=2)


def predict_pmf(model: SurvTraceMulti, x: np.ndarray, device,
                batch_size: int = _PREDICT_BATCH) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).float().to(device)
            phi = model(xb)
            out.append(F.softplus(phi).double().cpu().numpy())
    eta = np.concatenate(out, axis=0)
    return pmf_from_hazards(eta)


class BERTAdam(Optimizer):

    def __init__(self, params, lr, warmup=-1, t_total=-1,
                 schedule="warmup_linear", b1=0.9, b2=0.999, e=1e-6,
                 weight_decay_rate=0.01, max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup,
                        t_total=t_total, b1=b1, b2=b2, e=e,
                        weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["next_m"] = torch.zeros_like(p.data)
                    state["next_v"] = torch.zeros_like(p.data)
                next_m, next_v = state["next_m"], state["next_v"]
                beta1, beta2 = group["b1"], group["b2"]
                if group["max_grad_norm"] > 0:
                    clip_grad_norm_(p, group["max_grad_norm"])
                next_m.mul_(beta1).add_(grad, alpha=1 - beta1)
                next_v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = next_m / (next_v.sqrt() + group["e"])
                if group["weight_decay_rate"] > 0.0:
                    update += group["weight_decay_rate"] * p.data
                if group["t_total"] != -1:
                    raise NotImplementedError(
                        "warmup schedules are not used by SurvTRACE's "
                        "published configuration")
                p.data.add_(-group["lr"] * update)
                state["step"] += 1
        return loss


def make_optimizer(model: nn.Module, lr: float,
                   weight_decay: float) -> BERTAdam:
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    named = list(model.named_parameters())
    groups = [
        {"params": [p for n, p in named if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in named if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    return BERTAdam(groups, lr, weight_decay_rate=weight_decay)


def _hidden_dropout(model: SurvTraceMulti) -> float:
    rates = {float(model.embeddings.dropout.p)}
    for layer in model.encoder:
        rates.add(float(layer.attention.output.dropout.p))
        rates.add(float(layer.output.dropout.p))
    block = model.cls.net[0]
    rates.add(float(block.dropout.p) if block.dropout is not None else 0.0)
    if len(rates) != 1:
        raise ValueError(f"hidden dropout built at disagreeing rates {rates}")
    return rates.pop()


def _val_loss(model, x_v, idx_v, evt_v, frac_v, cause_codes, device) -> float:
    model.eval()
    with torch.no_grad():
        phis = []
        for k in range(len(cause_codes)):
            chunks = [model(x_v[i:i + _PREDICT_BATCH], event=k)
                      for i in range(0, len(x_v), _PREDICT_BATCH)]
            phis.append(torch.cat(chunks, dim=0))
        return float(pc_hazard_loss(phis, idx_v, evt_v, frac_v,
                                    cause_codes).item())


def train_model(
    x_train, dur_train, evt_train,
    x_val, dur_val, evt_val,
    cuts: np.ndarray,
    cause_codes,
    hidden_size: int = 16, num_hidden_layers: int = 2,
    num_attention_heads: int = 2, intermediate_size: int = 64,
    hidden_dropout_prob: float = 0.0,
    attention_probs_dropout_prob: float = 0.1,
    initializer_range: float = 0.02,
    lr: float = 1e-4, weight_decay: float = 0.0,
    epochs: int = 100, patience: int = 5, batch_size: int = 1024,
    device: str = "auto",
) -> tuple:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    cause_codes = [int(c) for c in cause_codes]

    idx_t, ev_t, fr_t = pc_hazard_labels(dur_train, evt_train, cuts)
    idx_v, ev_v, fr_v = pc_hazard_labels(dur_val, evt_val, cuts)
    n_intervals = len(cuts) - 1

    model = SurvTraceMulti(
        num_feature=x_train.shape[1], num_event=len(cause_codes),
        out_feature=n_intervals, hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        initializer_range=initializer_range,
    ).to(device)
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)

    first_layer = model.encoder[0]
    effective: dict = {
        "lr": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay_rate"]),
        "optimizer": type(optimizer).__name__,
        "max_grad_norm": float(optimizer.param_groups[0]["max_grad_norm"]),
        "hidden_size": int(model.embeddings.num_embeddings.shape[-1]),
        "num_hidden_layers": int(len(model.encoder)),
        "num_attention_heads": int(
            first_layer.attention.self.num_attention_heads),
        "intermediate_size": int(first_layer.intermediate.dense.out_features),
        "hidden_dropout_prob": _hidden_dropout(model),
        "attention_probs_dropout_prob": float(
            first_layer.attention.self.dropout.p),
        "initializer_range": float(model.initializer_range),
        "n_features": int(model.num_feature),
        "num_risks": int(model.num_event),
        "num_bins": int(len(cuts)),
        "n_intervals": int(model.out_feature),
    }
    observed_batch_size = 0

    x_t = torch.from_numpy(x_train).float().to(device)
    i_t = torch.from_numpy(idx_t).long().to(device)
    e_t = torch.from_numpy(ev_t).long().to(device)
    f_t = torch.from_numpy(fr_t).float().to(device)
    x_v = torch.from_numpy(x_val).float().to(device)
    i_v = torch.from_numpy(idx_v).long().to(device)
    e_v = torch.from_numpy(ev_v).long().to(device)
    f_v = torch.from_numpy(fr_v).float().to(device)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    n_train = len(x_train)
    epoch = 0
    epoch_seconds: list[float] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            observed_batch_size = max(observed_batch_size, len(idx))
            optimizer.zero_grad()
            xb = x_t[idx]

            phis = [model(xb, event=k) for k in range(len(cause_codes))]
            loss = pc_hazard_loss(phis, i_t[idx], e_t[idx], f_t[idx],
                                  cause_codes)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_loss = _val_loss(model, x_v, i_v, e_v, f_v, cause_codes, device)
        epoch_seconds.append(time.time() - t0)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={epoch_loss / max(n_batches, 1):.4f}"
                  f"/{val_loss:.4f}  {epoch_seconds[-1]:.1f}s")


        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
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
                    "best_val_nll": float(best_val_loss),
                    "epochs_run": epoch,
                    "epoch_seconds": [float(s) for s in epoch_seconds]},
            effective)


def run(
    config: dict,
    *,
    hidden_size: int = 16,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 2,
    intermediate_size: int = 64,
    hidden_dropout_prob: float = 0.0,
    attention_probs_dropout_prob: float = 0.1,
    initializer_range: float = 0.02,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    epochs: int = 100,
    patience: int = 5,
    batch_size: int = 1024,
    input_mode: str = "static",
    seed: int = 42,
    skip_test: bool = False,
    save_dir: Optional[Path] = None,
    save_tag: Optional[str] = None,
    **_ignored,
) -> dict:
    _reject_unexpected_kwargs(_ignored)
    model_name = "survtrace"
    prefix = prefix_of(config)
    horizon = float(config.get("preprocessing", {}).get("max_horizon_hours", 240))
    dataset = dataset_of(config)

    train = load_split(config, "train", **static_input_kwargs(input_mode))
    val = load_split(config, "val", **static_input_kwargs(input_mode))
    test = (None if skip_test
            else load_split(config, "test", **static_input_kwargs(input_mode)))

    modality_info = train["modality_info"]
    competing = bool(modality_info.get("competing_risks", False))
    num_risks = int(modality_info.get("n_events", 1)) if competing else 1
    risk_axis = risk_axis_of(modality_info)
    cause_codes = cause_codes_of(modality_info)
    if len(cause_codes) != num_risks:
        raise ValueError(f"{len(cause_codes)} cause codes for {num_risks} risks")
    if [cause_code_of_head(modality_info)[r] for r in range(num_risks)]\
            != list(risk_axis):
        raise ValueError("head -> cause mapping disagrees with the risk axis")


    cuts = train["cuts"]
    for other in (val, test):
        if other is not None and not np.array_equal(other["cuts"], cuts):
            raise ValueError("splits disagree on the bin grid `cuts`")
    num_bins = len(cuts)

    x_train, dur_train, evt_train = train["x"], train["durations"], train["events"]
    x_val, dur_val, evt_val = val["x"], val["durations"], val["events"]
    if test is None:
        x_test = evt_test = dur_test = lm_test = None
    else:
        x_test, dur_test, evt_test = test["x"], test["durations"], test["events"]
        lm_test = test["landmarks"]

    n_train, n_val = len(dur_train), len(dur_val)
    n_test = None if test is None else len(dur_test)
    n_events_train = int((evt_train > 0).sum())
    n_events_val = int((evt_val > 0).sum())
    n_events_test = None if test is None else int((evt_test > 0).sum())

    x_train, x_val, x_test, keep_mask = remove_constant_cols(x_train, x_val, x_test)


    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    x_train = ((x_train - mu) / sd).astype(np.float32)
    x_val = ((x_val - mu) / sd).astype(np.float32)
    if x_test is not None:
        x_test = ((x_test - mu) / sd).astype(np.float32)

    print(f"  Shapes: train={x_train.shape} val={x_val.shape} "
          f"test={'not loaded' if x_test is None else x_test.shape}")
    print(f"  Event rate (train): {(evt_train > 0).mean() * 100:.1f}%")
    print(f"  Bins (K): {num_bins}  Risks: {num_risks}  input_mode={input_mode}")

    with WallClock() as wc:
        model, train_info, effective_hparams = train_model(
            x_train, dur_train, evt_train, x_val, dur_val, evt_val,
            cuts=cuts, cause_codes=cause_codes,
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            lr=lr, weight_decay=weight_decay, epochs=epochs,
            patience=patience, batch_size=batch_size,
        )
    wall_time_seconds = wc.seconds
    device = next(model.parameters()).device

    inner: dict = {"model_info": {
        "type": "SurvTRACE", "upstream": UPSTREAM, **train_info,
        "n_features": x_train.shape[1], "num_risks": num_risks,
        "num_bins": num_bins,
    }}
    surv_val_np: Optional[np.ndarray] = None
    surv_test_np: Optional[np.ndarray] = None

    scored = [("val", x_val, dur_val, evt_val, val["landmarks"])]
    if not skip_test:
        scored.append(("test", x_test, dur_test, evt_test, lm_test))
    for split_name, x_s, dur_s, evt_s, lm_s in scored:
        pmf = predict_pmf(model, x_s, device)
        surv = 1.0 - pmf.cumsum(axis=-1)
        if split_name == "val":
            surv_val_np = surv
        else:
            surv_test_np = surv
        metrics = evaluate_by_landmark(
            pmf, dur_s, evt_s, cuts, num_risks,
            train_durations=dur_train, train_events_full=evt_train,
            horizon=horizon, dataset=dataset, modality_info=modality_info,
            landmarks=lm_s,
        )
        inner[split_name] = metrics
        m = metrics.get("mean", metrics.get("risk1", {}))
        print(f"  {split_name}: Ctd={m.get('ctd', float('nan')):.4f}  "
              f"IBS={m.get('ibs', float('nan')):.4f}  "
              f"IBLL={m.get('ibll', float('nan')):.4f}"
              f"  AUC={m.get('auc_integrated', float('nan')):.4f}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = save_tag if save_tag is not None else f"{model_name}_{prefix}_seed{seed}"
        save_predictions(save_dir / f"{tag}_preds.npz", cuts=cuts,
                         cause_codes=risk_axis, surv_val=surv_val_np,
                         surv_test=surv_test_np)
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
            "optimiser": "BERTAdam",
            "lr": lr,
            "batch_size": batch_size,
            "epochs_run": train_info.get("epochs_run"),
            "early_stop_epoch": train_info.get("epochs_run"),
            "early_stop_patience": patience,
            "early_stop_metric": "val_loss",
            "wall_time_seconds": wall_time_seconds,
            "param_count": param_count(model),
            "device": str(device),
        },
        "metrics": {"val": selection_losses(
            _flat_metrics(inner.get("val", {})),
            train_info["best_val_loss"], train_info["best_val_nll"])},
        "effective_hparams": effective_hparams,
        "notes": f"ported from {UPSTREAM}",
        RISK_AXIS_KEY: risk_axis,
        "input_mode": input_mode,
        "val": inner.get("val", {}),
        "model_info": inner.get("model_info", {}),
    }
    if not skip_test:
        result["metrics"]["test"] = _flat_metrics(inner.get("test", {}))
        result["test"] = inner.get("test", {})
    return result
