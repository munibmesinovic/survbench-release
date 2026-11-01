from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from torchdiffeq import odeint

from models.baselines._data import (
    cause_codes_of,
    dataset_of,
    head_index_lookup,
    load_split,
    patch_scipy_simps,
    prefix_of,
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
from preprocessing.landmarking import landmarking_enabled

patch_scipy_simps()


from models.baselines.deephit import evaluate_by_landmark
from models.baselines.dynamic_deephit import _require_mask


UPSTREAM = ("https://github.com/itmoon7/survlatent_ode@"
            "c712bdc0865dc3293fa4f74c0dd499bbeb7152fc")

UPSTREAM_LICENSE = """\
MIT License

Copyright (c) 2022 Intae Moon

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


SURV_EPS = 1e-5
STD_EPS = 1e-7
OBSRV_STD = 1.0
ODEINT_RTOL = 1e-3
ODEINT_ATOL = 1e-4
LR_DECAY = 0.999
LR_FLOOR_FRACTION = 0.1
WAIT_UNTIL_KL_INC = 10
SURV_SCALE_START = 0.01
SELECT_FROM_EPOCH = 3
EARLY_STOP_WINDOW = 3
EVAL_BATCH = 200


def init_network_weights(net: nn.Module, std: float = 0.1) -> None:
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def create_net(n_inputs, n_outputs, n_layers=1, n_units=100,
               nonlinear=nn.Tanh) -> nn.Sequential:
    layers = [nn.Linear(n_inputs, n_units)]
    for _ in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))
    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


def split_last_dim(data: torch.Tensor):
    half = data.size()[-1] // 2
    return data[..., :half], data[..., half:]


class ODEFunc(nn.Module):
    def __init__(self, ode_func_net: nn.Module):
        super().__init__()
        init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards: bool = False):
        grad = self.gradient_net(y)
        return -grad if backwards else grad


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func: ODEFunc, method: str,
                 odeint_rtol: float = ODEINT_RTOL,
                 odeint_atol: float = ODEINT_ATOL):
        super().__init__()
        self.ode_method = method
        self.ode_func = ode_func
        self.odeint_rtol = odeint_rtol * 1e-5
        self.odeint_atol = odeint_atol * 1e-5

    def forward(self, first_point, time_steps_to_predict):
        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol,
                        method=self.ode_method)
        return pred_y.permute(1, 2, 0, 3)


class GRUUnit(nn.Module):

    def __init__(self, latent_dim: int, input_dim: int, n_units: int = 100):
        super().__init__()
        in_dim = latent_dim * 2 + input_dim
        self.update_gate = nn.Sequential(
            nn.Linear(in_dim, n_units), nn.Tanh(),
            nn.Linear(n_units, latent_dim), nn.Sigmoid())
        init_network_weights(self.update_gate)
        self.reset_gate = nn.Sequential(
            nn.Linear(in_dim, n_units), nn.Tanh(),
            nn.Linear(n_units, latent_dim), nn.Sigmoid())
        init_network_weights(self.reset_gate)
        self.new_state_net = nn.Sequential(
            nn.Linear(in_dim, n_units), nn.Tanh(),
            nn.Linear(n_units, latent_dim * 2))
        init_network_weights(self.new_state_net)

    def forward(self, y_mean, y_std, x, masked_update: bool = True):
        y_concat = torch.cat([y_mean, y_std, x], -1)
        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)
        new_state, new_state_std = split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()
        new_y = (1 - update_gate) * new_state + update_gate * y_mean
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std
        if masked_update:
            n_data_dims = x.size(-1) // 2
            mask = x[:, :, n_data_dims:]
            mask = (torch.sum(mask, -1, keepdim=True) > 0).float()
            new_y = mask * new_y + (1 - mask) * y_mean
            new_y_std = mask * new_y_std + (1 - mask) * y_std
        return new_y, new_y_std.abs()


class EncoderZ0ODERNN(nn.Module):

    def __init__(self, latent_dim: int, input_dim: int,
                 z0_diffeq_solver: DiffeqSolver, z0_dim: int,
                 n_gru_units: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.z0_dim = z0_dim
        self.input_dim = input_dim
        self.GRU_update = GRUUnit(latent_dim, input_dim, n_units=n_gru_units)
        self.z0_diffeq_solver = z0_diffeq_solver
        self.transform_z0 = nn.Sequential(
            nn.Linear(latent_dim * 2, 100), nn.Tanh(),
            nn.Linear(100, self.z0_dim * 2))
        init_network_weights(self.transform_z0)

    def forward(self, data, time_steps, run_backwards: bool = True):
        n_traj = data.size(0)
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim), device=data.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim), device=data.device)
            xi = data[:, 0, :].unsqueeze(0)
            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
        else:
            last_yi, last_yi_std = self.run_odernn(data, time_steps,
                                                   run_backwards=run_backwards)
        means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)
        mean_z0, std_z0 = split_last_dim(
            self.transform_z0(torch.cat((means_z0, std_z0), -1)))
        return mean_z0, std_z0.abs()

    def run_odernn(self, data, time_steps, run_backwards: bool = True):
        n_traj = data.size(0)
        device = data.device
        prev_y = torch.zeros((1, n_traj, self.latent_dim), device=device)
        prev_std = torch.zeros((1, n_traj, self.latent_dim), device=device)
        prev_t, t_i = time_steps[-1] + 0.01, time_steps[-1]
        minimum_step = (time_steps[-1] - time_steps[0]) / 50
        order = range(0, len(time_steps))
        if run_backwards:
            order = reversed(order)
        yi, yi_std = prev_y, prev_std
        for i in order:
            if (prev_t - t_i) < minimum_step:
                inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)
                yi_ode = prev_y + inc
            else:
                n_tp = max(2, int(((prev_t - t_i) / minimum_step).item()))
                time_points = torch.linspace(float(prev_t), float(t_i), n_tp,
                                             device=device)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)
                yi_ode = ode_sol[:, :, -1, :]
            xi = data[:, i, :].unsqueeze(0)
            yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)
            prev_y, prev_std = yi, yi_std
            prev_t, t_i = time_steps[i], time_steps[i - 1]
        return yi, yi_std


def _hazard_decoder(latent_dim: int, num_layer: int, out_units: int,
                    sigmoid: bool) -> nn.Sequential:
    half = int(latent_dim / 2)
    if num_layer == 2:
        layers = [nn.Linear(latent_dim, half), nn.Tanh(),
                  nn.Linear(half, out_units)]
    elif num_layer == 3:
        layers = [nn.Linear(latent_dim, half), nn.Tanh(),
                  nn.Linear(half, half), nn.Tanh(),
                  nn.Linear(half, out_units)]
    else:
        raise ValueError(f"haz_dec_layers={num_layer}; upstream builds 2 or 3")
    if sigmoid:
        layers.append(nn.Sigmoid())
    net = nn.Sequential(*layers)
    init_network_weights(net)
    return net


class SurvLatentODE(nn.Module):

    def __init__(self, input_dim: int, n_events: int,
                 dec_latent_dim: int = 40, enc_latent_dim: int = 50,
                 enc_f_nn_layers: int = 5, dec_g_nn_layers: int = 7,
                 num_units_ode: int = 70, num_units_gru: int = 80,
                 haz_dec_layers: int = 3, haz_dec_final_units: int = 5,
                 noise_seed: int = 0):
        super().__init__()
        self.n_events = n_events
        self.input_dim = input_dim
        self.latent_dim = dec_latent_dim
        self.noise_seed = int(noise_seed)
        self.obsrv_std = OBSRV_STD

        gen_net = create_net(dec_latent_dim, dec_latent_dim,
                             n_layers=dec_g_nn_layers, n_units=num_units_ode,
                             nonlinear=nn.Tanh)
        gen_ode_func = ODEFunc(gen_net)
        enc_input_dim = int(input_dim) * 2
        rec_net = create_net(enc_latent_dim, enc_latent_dim,
                             n_layers=enc_f_nn_layers, n_units=num_units_ode,
                             nonlinear=nn.Tanh)
        rec_ode_func = ODEFunc(rec_net)
        z0_diffeq_solver = DiffeqSolver(rec_ode_func, "euler")
        self.encoder_z0 = EncoderZ0ODERNN(enc_latent_dim, enc_input_dim,
                                          z0_diffeq_solver, z0_dim=dec_latent_dim,
                                          n_gru_units=num_units_gru)
        self.diffeq_solver = DiffeqSolver(gen_ode_func, "dopri5")

        self.decoder = nn.Sequential(
            nn.Linear(dec_latent_dim, dec_latent_dim), nn.Sigmoid(),
            nn.Linear(dec_latent_dim, input_dim))
        init_network_weights(self.decoder)

        if n_events == 1:

            self.hazard_decoders = nn.ModuleList(
                [_hazard_decoder(dec_latent_dim, 2, 1, sigmoid=True)])
            self.merging_decoder = None
        else:
            self.hazard_decoders = nn.ModuleList(
                [_hazard_decoder(dec_latent_dim, haz_dec_layers,
                                 haz_dec_final_units, sigmoid=False)
                 for _ in range(n_events)])

            self.merging_decoder = nn.Linear(haz_dec_final_units * n_events,
                                             n_events + 1)

    def sample_z0(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        gen = torch.Generator(device=mu.device)
        gen.manual_seed(self.noise_seed)
        eps = torch.randn(mu.size(), generator=gen, device=mu.device,
                          dtype=mu.dtype)
        return eps * sigma + mu

    def hazards(self, sol_y: torch.Tensor) -> torch.Tensor:
        if self.n_events == 1:
            h = self.hazard_decoders[0](sol_y)
            return torch.cat([h, 1.0 - h], dim=-1)
        stacked = torch.cat([dec(sol_y) for dec in self.hazard_decoders], dim=-1)
        return torch.softmax(self.merging_decoder(stacked), dim=-1)

    def forward(self, x: torch.Tensor, m: torch.Tensor, grid: "TimeGrid"):
        if m.shape != x.shape:
            raise ValueError(
                f"observation mask has shape {tuple(m.shape)} but x has "
                f"{tuple(x.shape)}")
        truth_w_mask = torch.cat((x, m), -1)
        mu, sigma = self.encoder_z0(truth_w_mask, grid.enc_times.to(x.device))
        z0 = self.sample_z0(mu, sigma)
        sigma = sigma.abs() + STD_EPS
        sol_y = self.diffeq_solver(z0, grid.dec_times.to(x.device))
        pred_y = self.decoder(sol_y)[0][:, grid.obs_pos.to(x.device)]
        hazards = self.hazards(sol_y)[0][:, grid.pred_pos.to(x.device)]
        return pred_y, hazards, mu, sigma


class TimeGrid:

    def __init__(self, num_windows: int, window_hours: float, cuts,
                 landmarked: bool):
        cuts = np.asarray(cuts, dtype=np.float64)
        obs = (np.arange(num_windows, dtype=np.float64) + 1.0) * float(window_hours)
        offset = num_windows * float(window_hours) if landmarked else 0.0
        pred = offset + cuts
        union = np.unique(np.concatenate([obs, pred]))
        self.obs_hours = obs
        self.pred_hours = pred
        self.union_hours = union
        self.obs_pos = torch.as_tensor(np.searchsorted(union, obs), dtype=torch.long)
        self.pred_pos = torch.as_tensor(np.searchsorted(union, pred), dtype=torch.long)
        if num_windows > 1:
            enc = (obs - obs[0]) / (obs[-1] - obs[0])
        else:
            enc = np.zeros(1)
        self.enc_times = torch.as_tensor(enc, dtype=torch.float32)
        self.dec_times = torch.as_tensor((union - union[0]) / (union[-1] - union[0]),
                                         dtype=torch.float32)


def window_hours_of(config: dict, num_windows: int) -> float:
    pre = config.get("preprocessing", {}) or {}
    ws = pre.get("window_size_hours")
    span = pre.get("max_hours")
    if ws is None and span is None:
        raise ValueError(
            "config needs preprocessing.window_size_hours or preprocessing.max_hours")
    if ws is None:
        return float(span) / num_windows
    if span is not None and not math.isclose(float(ws) * num_windows, float(span)):
        raise ValueError(
            f"{num_windows} windows of {ws}h do not cover max_hours={span}h")
    return float(ws)


def minmax_stats(x_train: np.ndarray, m_train: np.ndarray):
    obs = m_train > 0
    flat_x = x_train.reshape(-1, x_train.shape[-1])
    flat_o = obs.reshape(-1, obs.shape[-1])
    mn = np.where(flat_o, flat_x, np.inf).min(axis=0)
    mx = np.where(flat_o, flat_x, -np.inf).max(axis=0)
    never = ~np.isfinite(mn)
    mn[never], mx[never] = 0.0, 1.0
    mx[mx == 0.0] = 1.0
    rng = mx - mn
    rng[rng == 0.0] = 1.0
    return mn.astype(np.float32), rng.astype(np.float32)


def normalize_masked(x: np.ndarray, m: np.ndarray, mn, rng) -> np.ndarray:
    out = ((x - mn) / rng).astype(np.float32)
    out[m == 0] = 0.0
    return out


def event_weights_of(head_train: np.ndarray, n_events: int) -> Optional[np.ndarray]:
    if n_events == 1:
        return None
    total = int((head_train >= 0).sum())
    raw = np.array([total - int((head_train == k).sum())
                    for k in range(n_events)], dtype=np.float64)
    if raw.min() <= 0:
        raise ValueError(
            f"event weights undefined: {raw.tolist()}; one cause holds every event")
    return raw / raw.min()


def survival_loglik(hazards: torch.Tensor, dur_idx: torch.Tensor,
                    head: torch.Tensor, eps: float = SURV_EPS) -> torch.Tensor:
    B, K, E1 = hazards.shape
    n_events = E1 - 1
    d = dur_idx.long()
    log_none = torch.log(hazards[..., n_events] + eps)
    steps = torch.arange(K, device=hazards.device)
    before = (steps[None, :] < d[:, None]).to(log_none.dtype)
    surv_part = (log_none * before).sum(dim=1)
    at_d_none = log_none.gather(1, d[:, None]).squeeze(1)
    censored = head < 0
    safe_head = torch.where(censored, torch.zeros_like(head), head).long()
    h_at_d = hazards[torch.arange(B, device=hazards.device), d, safe_head]
    event_part = torch.log(h_at_d + eps)
    return surv_part + torch.where(censored, at_d_none, event_part)


def survival_loss(hazards, dur_idx, head, event_weights=None,
                  eps: float = SURV_EPS) -> torch.Tensor:
    ll = survival_loglik(hazards, dur_idx, head, eps=eps)
    if event_weights is not None:
        w = torch.as_tensor(event_weights, dtype=ll.dtype, device=ll.device)
        row_w = torch.where(head >= 0, w[head.clamp(min=0).long()],
                            torch.ones_like(ll))
        ll = ll * row_w
    return -ll.sum()


def gaussian_reconstruction_loglik(pred_y, truth, mask,
                                   obsrv_std: float = OBSRV_STD) -> torch.Tensor:
    log_density = (-0.5 * math.log(2 * math.pi) - math.log(obsrv_std)
                   - (truth - pred_y) ** 2 / (2 * obsrv_std ** 2))
    n_obs = mask.sum(dim=1)
    total = (log_density * mask).sum(dim=1)
    per_feature = torch.where(n_obs > 0, total / n_obs.clamp(min=1),
                              torch.zeros_like(total))
    return per_feature.mean(dim=-1).mean()


def kl_z0(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    prior = Normal(torch.zeros(1, device=mu.device), torch.ones(1, device=mu.device))
    return kl_divergence(Normal(mu, sigma), prior).mean()


def kl_coef_at(itr: int, num_batches: int,
               wait_until_kl_inc: int = WAIT_UNTIL_KL_INC) -> float:
    epoch = itr // num_batches
    if epoch < wait_until_kl_inc:
        return 0.0
    return 1.0 - 0.99 ** (epoch - wait_until_kl_inc)


def surv_scale_at(itr: int, num_batches: int, surv_loss_scale: float,
                  wait_until_full_surv_loss: int) -> float:
    epoch = itr // num_batches
    if epoch == 0:
        return SURV_SCALE_START
    if epoch < wait_until_full_surv_loss:
        return SURV_SCALE_START * math.exp(
            1.0 / wait_until_full_surv_loss
            * math.log(surv_loss_scale / SURV_SCALE_START) * epoch)
    return float(surv_loss_scale)


def composed_loss(rec_ll, kl, surv_loss_value, kl_coef, surv_scale):
    return -(rec_ll - kl_coef * kl) + surv_loss_value * surv_scale


def decay_learning_rate(optimizer, lr_floor: float,
                        decay_rate: float = LR_DECAY) -> None:
    for group in optimizer.param_groups:
        group["lr"] = max(group["lr"] * decay_rate, lr_floor)


def pmf_from_step_probs(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum(axis=-1, keepdims=True)
    n_events = p.shape[-1] - 1
    none = p[..., n_events]
    surv_before = np.cumprod(np.concatenate(
        [np.ones_like(none[:, :1]), none[:, :-1]], axis=1), axis=1)
    pmf = p[..., :n_events] * surv_before[..., None]
    return np.transpose(pmf, (0, 2, 1))


def _forward_chunks(model, x_t, m_t, grid, batch: int = EVAL_BATCH):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x_t), batch):
            outs.append(model(x_t[i:i + batch], m_t[i:i + batch], grid))
    return outs


def _validate(model, x_v, m_v, d_v, h_v, grid, event_weights,
              kl_coef, surv_scale) -> tuple:
    rec_sum = kl_sum = 0.0
    surv_w_sum = surv_sum = 0.0
    n = len(x_v)
    for (pred_y, hazards, mu, sigma), i in zip(
            _forward_chunks(model, x_v, m_v, grid), range(0, n, EVAL_BATCH)):
        sl = slice(i, i + EVAL_BATCH)
        b = pred_y.shape[0]
        rec_sum += float(gaussian_reconstruction_loglik(pred_y, x_v[sl], m_v[sl])) * b
        kl_sum += float(kl_z0(mu, sigma)) * b
        surv_w_sum += float(survival_loss(hazards, d_v[sl], h_v[sl], event_weights))
        surv_sum += float(survival_loss(hazards, d_v[sl], h_v[sl], None))
    loss = composed_loss(rec_sum / n, kl_sum / n, surv_w_sum, kl_coef, surv_scale)
    return float(loss), surv_sum / n


def predict_pmf(model, x_t, m_t, grid) -> np.ndarray:
    probs = [h.double().cpu().numpy()
             for _, h, _, _ in _forward_chunks(model, x_t, m_t, grid)]
    return pmf_from_step_probs(np.concatenate(probs, axis=0))


def _linear_count(module: nn.Module) -> int:
    return sum(isinstance(m, nn.Linear) for m in module.modules())


def train_model(
    x_train, m_train, didx_train, head_train,
    x_val, m_val, didx_val, head_val,
    grid: TimeGrid, n_events: int,
    dec_latent_dim: int = 40, enc_latent_dim: int = 50,
    enc_f_nn_layers: int = 5, dec_g_nn_layers: int = 7,
    num_units_ode: int = 70, num_units_gru: int = 80,
    haz_dec_layers: int = 3, haz_dec_final_units: int = 5,
    lr: float = 0.01, surv_loss_scale: float = 100.0,
    wait_until_full_surv_loss: int = 3, epochs: int = 30,
    batch_size: int = 100, early_stopping: bool = True,
    seed: int = 0, device: str = "auto",
) -> tuple:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = SurvLatentODE(
        input_dim=x_train.shape[-1], n_events=n_events,
        dec_latent_dim=dec_latent_dim, enc_latent_dim=enc_latent_dim,
        enc_f_nn_layers=enc_f_nn_layers, dec_g_nn_layers=dec_g_nn_layers,
        num_units_ode=num_units_ode, num_units_gru=num_units_gru,
        haz_dec_layers=haz_dec_layers, haz_dec_final_units=haz_dec_final_units,
        noise_seed=seed,
    ).to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    lr_floor = lr * LR_FLOOR_FRACTION
    event_weights = event_weights_of(np.asarray(head_train), n_events)

    gen_net = model.diffeq_solver.ode_func.gradient_net
    rec_net = model.encoder_z0.z0_diffeq_solver.ode_func.gradient_net
    haz0 = model.hazard_decoders[0]
    effective: dict = {
        "lr": float(optimizer.param_groups[0]["lr"]),
        "optimizer": type(optimizer).__name__,
        "lr_decay_per_iteration": LR_DECAY,
        "lr_floor": float(lr_floor),
        "dec_latent_dim": int(gen_net[0].in_features),
        "dec_g_nn_layers": _linear_count(gen_net) - 2,
        "num_units_ode": int(gen_net[0].out_features),
        "enc_latent_dim": int(model.encoder_z0.latent_dim),
        "enc_f_nn_layers": _linear_count(rec_net) - 2,
        "num_units_gru": int(model.encoder_z0.GRU_update.update_gate[0].out_features),
        "haz_dec_layers": _linear_count(haz0),
        "haz_dec_final_units": (None if n_events == 1
                                else int(model.merging_decoder.in_features // n_events)),
        "obsrv_std": float(model.obsrv_std),
        "encoder_ode_method": model.encoder_z0.z0_diffeq_solver.ode_method,
        "decoder_ode_method": model.diffeq_solver.ode_method,
        "decoder_rtol": float(model.diffeq_solver.odeint_rtol),
        "decoder_atol": float(model.diffeq_solver.odeint_atol),
        "surv_loss_scale": float(surv_loss_scale),
        "wait_until_full_surv_loss": int(wait_until_full_surv_loss),
        "wait_until_kl_inc": WAIT_UNTIL_KL_INC,
        "early_stopping": bool(early_stopping),
        "event_weights": (None if event_weights is None
                          else [float(w) for w in event_weights]),
        "n_features": int(model.input_dim),
        "num_risks": int(model.n_events),
        "num_bins": int(len(grid.pred_pos)),
        "time_steps": int(len(grid.obs_pos)),
    }
    observed_batch_size = 0

    x_t = torch.from_numpy(x_train).float().to(device)
    m_t = torch.from_numpy(m_train).float().to(device)
    d_t = torch.from_numpy(np.asarray(didx_train)).long().to(device)
    h_t = torch.from_numpy(np.asarray(head_train)).long().to(device)
    x_v = torch.from_numpy(x_val).float().to(device)
    m_v = torch.from_numpy(m_val).float().to(device)
    d_v = torch.from_numpy(np.asarray(didx_val)).long().to(device)
    h_v = torch.from_numpy(np.asarray(head_val)).long().to(device)

    n_train = len(x_train)
    num_batches = int(math.ceil(n_train / batch_size))
    select_from = min(SELECT_FROM_EPOCH, epochs)
    history: list[float] = []
    epoch_seconds: list[float] = []
    best_val_nll = float("inf")
    best_val_loss = float("inf")
    best_epoch = None
    best_state = None
    itr = 0
    epoch = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for b in range(num_batches):
            itr += 1
            idx = perm[b * batch_size:(b + 1) * batch_size]
            observed_batch_size = max(observed_batch_size, len(idx))
            optimizer.zero_grad()
            decay_learning_rate(optimizer, lr_floor)
            kl_coef = kl_coef_at(itr, num_batches)
            surv_scale = surv_scale_at(itr, num_batches, surv_loss_scale,
                                       wait_until_full_surv_loss)
            pred_y, hazards, mu, sigma = model(x_t[idx], m_t[idx], grid)
            loss = composed_loss(
                gaussian_reconstruction_loglik(pred_y, x_t[idx], m_t[idx]),
                kl_z0(mu, sigma),
                survival_loss(hazards, d_t[idx], h_t[idx], event_weights),
                kl_coef, surv_scale)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        val_loss, val_nll = _validate(model, x_v, m_v, d_v, h_v, grid,
                                      event_weights, kl_coef, surv_scale)
        history.append(val_nll)
        epoch_seconds.append(time.time() - t0)
        print(f"  Epoch {epoch:3d}: loss={epoch_loss / num_batches:.4f}  "
              f"val_nll={val_nll:.4f}  val_loss={val_loss:.4f}  "
              f"kl_coef={kl_coef:.3f}  surv_scale={surv_scale:.3g}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"{epoch_seconds[-1]:.1f}s")

        if epoch >= select_from and val_nll < best_val_nll:
            best_val_nll = val_nll
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (early_stopping
                and epoch >= max(10, wait_until_full_surv_loss)
                and epoch > EARLY_STOP_WINDOW
                and all(history[-i - 2] < history[-i - 1]
                        for i in range(EARLY_STOP_WINDOW))):
            print(f"  Early stopping at epoch {epoch}: validation NLL rose "
                  f"{EARLY_STOP_WINDOW} epochs running (best epoch {best_epoch})")
            stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    effective["batch_size"] = int(observed_batch_size)
    return (model, {"best_val_loss": float(best_val_loss),
                    "best_val_nll": float(best_val_nll),
                    "best_epoch": best_epoch,
                    "epochs_run": epoch,
                    "stopped_early": stopped_early,
                    "val_nll_history": [float(v) for v in history],
                    "epoch_seconds": [float(s) for s in epoch_seconds]},
            effective)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot read {value!r} as a boolean")
    return bool(value)


def _head_of(events: np.ndarray, modality_info: dict) -> np.ndarray:
    table = head_index_lookup(modality_info)
    events = np.asarray(events, dtype=np.int64)
    head = np.full(len(events), -1, dtype=np.int64)
    has = events > 0
    if (events[has] >= len(table)).any():
        raise ValueError(f"event codes {sorted(set(events[has].tolist()))} exceed "
                         f"the declared causes {modality_info.get('cause_codes')}")
    head[has] = table[events[has]]
    if (head[has] < 0).any():
        bad = sorted({int(c) for c in events[has][head[has] < 0]})
        raise ValueError(f"events contain cause code(s) {bad} that this dataset "
                         f"does not declare")
    return head


def run(
    config: dict,
    *,
    dec_latent_dim: int = 40,
    enc_latent_dim: int = 50,
    enc_f_nn_layers: int = 5,
    dec_g_nn_layers: int = 7,
    num_units_ode: int = 70,
    num_units_gru: int = 80,
    haz_dec_layers: int = 3,
    haz_dec_final_units: int = 5,
    lr: float = 0.01,
    surv_loss_scale: float = 100.0,
    wait_until_full_surv_loss: int = 3,
    epochs: int = 30,
    batch_size: int = 100,
    early_stopping: bool = True,
    seed: int = 42,
    skip_test: bool = False,
    save_dir: Optional[Path] = None,
    save_tag: Optional[str] = None,
    **_ignored,
) -> dict:
    _reject_unexpected_kwargs(_ignored)
    early_stopping = _as_bool(early_stopping)
    model_name = "survlatent_ode"
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
    if len(cause_codes_of(modality_info)) != num_risks:
        raise ValueError("cause codes and n_events disagree")
    cuts = train["cuts"]
    for other in (val, test):
        if other is not None and not np.array_equal(other["cuts"], cuts):
            raise ValueError("splits disagree on the bin grid `cuts`")
    num_bins = len(cuts)

    x_train = train["x"]
    m_train = _require_mask(train, "train", prefix)
    didx_train, evt_train, dur_train = train["dur_idx"], train["events"], train["durations"]
    x_val = val["x"]
    m_val = _require_mask(val, "val", prefix)
    didx_val, evt_val, dur_val = val["dur_idx"], val["events"], val["durations"]
    if test is None:
        x_test = m_test = evt_test = dur_test = lm_test = None
    else:
        x_test = test["x"]
        m_test = _require_mask(test, "test", prefix)
        evt_test, dur_test, lm_test = test["events"], test["durations"], test["landmarks"]

    n_train, n_val = len(dur_train), len(dur_val)
    n_test = None if test is None else len(dur_test)
    n_events_train = int((evt_train > 0).sum())
    n_events_val = int((evt_val > 0).sum())
    n_events_test = None if test is None else int((evt_test > 0).sum())

    head_train = _head_of(evt_train, modality_info)
    head_val = _head_of(evt_val, modality_info)

    mn, rng = minmax_stats(x_train, m_train)
    x_train = normalize_masked(x_train, m_train, mn, rng)
    x_val = normalize_masked(x_val, m_val, mn, rng)
    if x_test is not None:
        x_test = normalize_masked(x_test, m_test, mn, rng)

    seq_len = x_train.shape[1]
    landmarked = landmarking_enabled(config)
    grid = TimeGrid(seq_len, window_hours_of(config, seq_len), cuts, landmarked)

    print(f"  Shapes: train={x_train.shape} val={x_val.shape} "
          f"test={'not loaded' if x_test is None else x_test.shape}")
    print(f"  Event rate (train): {(evt_train > 0).mean() * 100:.1f}%")
    print(f"  Bins (K): {num_bins}  Risks: {num_risks}  Time steps (S): {seq_len}"
          f"  landmarked={landmarked}")

    with WallClock() as wc:
        model, train_info, effective_hparams = train_model(
            x_train, m_train, didx_train, head_train,
            x_val, m_val, didx_val, head_val,
            grid=grid, n_events=num_risks,
            dec_latent_dim=dec_latent_dim, enc_latent_dim=enc_latent_dim,
            enc_f_nn_layers=enc_f_nn_layers, dec_g_nn_layers=dec_g_nn_layers,
            num_units_ode=num_units_ode, num_units_gru=num_units_gru,
            haz_dec_layers=haz_dec_layers,
            haz_dec_final_units=haz_dec_final_units,
            lr=lr, surv_loss_scale=surv_loss_scale,
            wait_until_full_surv_loss=wait_until_full_surv_loss,
            epochs=epochs, batch_size=batch_size,
            early_stopping=early_stopping, seed=seed,
        )
    wall_time_seconds = wc.seconds
    device = next(model.parameters()).device

    inner: dict = {"model_info": {
        "type": "SurvLatentODE", "upstream": UPSTREAM, **train_info,
        "n_features": x_train.shape[-1], "num_risks": num_risks,
        "num_bins": num_bins, "time_steps": seq_len,
        "grid_obs_hours": grid.obs_hours.tolist(),
        "grid_hazard_hours": grid.pred_hours.tolist(),
    }}
    surv_val_np: Optional[np.ndarray] = None
    surv_test_np: Optional[np.ndarray] = None

    scored = [("val", x_val, m_val, dur_val, evt_val, val["landmarks"])]
    if not skip_test:
        scored.append(("test", x_test, m_test, dur_test, evt_test, lm_test))
    for split_name, x_s, m_s, dur_s, evt_s, lm_s in scored:
        pmf = predict_pmf(model, torch.from_numpy(x_s).float().to(device),
                          torch.from_numpy(m_s).float().to(device), grid)
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
            "optimiser": "Adamax",
            "lr": lr,
            "batch_size": batch_size,
            "epochs_run": train_info.get("epochs_run"),
            "early_stop_epoch": train_info.get("epochs_run"),
            "early_stop_patience": EARLY_STOP_WINDOW,
            "early_stop_metric": "val_survival_nll",
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
        "val": inner.get("val", {}),
        "model_info": inner.get("model_info", {}),
    }
    if not skip_test:
        result["metrics"]["test"] = _flat_metrics(inner.get("test", {}))
        result["test"] = inner.get("test", {})
    return result
