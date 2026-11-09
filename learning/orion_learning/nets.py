"""PyTorch modules for the teacher, student adapter, critic and B2 (exported to ONNX).

Every module that sees raw observations carries its own frozen normalisation buffers, so the
exported ONNX graphs take the obs_spec vectors exactly as orion_policy builds them.
"""

from __future__ import annotations

import torch
from torch import nn

from orion_learning import obs_spec as S

HISTORY_WIDTH = S.OBS_DIM + S.ACTION_DIM     # 22 floats per history row: obs_k, a_(k-1)


class RunningNorm(nn.Module):
    """Running mean / variance normaliser; buffers are saved and exported (frozen at export).

    update() is called by the trainer between PPO iterations only, so a rollout and the update
    that consumes it see identical statistics.
    """

    def __init__(self, dim: int, clip: float = 5.0, eps: float = 1e-4):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.tensor(eps, dtype=torch.float64))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).double()
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean.double()
        tot = self.count + b_n
        mean = self.mean.double() + delta * b_n / tot
        m2 = self.var.double() * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        self.mean.copy_(mean.float())
        self.var.copy_((m2 / tot).float())
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class PrivEncoder(nn.Module):
    """Teacher's privileged encoder mu: (10,) traction descriptors -> (8,) latent z."""

    def __init__(self, hidden: list[int]):
        super().__init__()
        self.norm = RunningNorm(S.PRIVILEGED_DIM)
        self.net = mlp(S.PRIVILEGED_DIM, hidden, S.LATENT_DIM)

    def forward(self, priv: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(priv))


class Policy(nn.Module):
    """pi(obs, latent) -> deterministic action mean in [-1, 1] (tanh); Gaussian for training.

    use_latent=False (B2) multiplies the latent by zero inside the graph, so the exported
    model keeps the contract's `latent` input but ignores it.
    """

    def __init__(self, hidden: list[int], init_log_std: float = -0.5, use_latent: bool = True):
        super().__init__()
        self.obs_norm = RunningNorm(S.OBS_DIM)
        self.net = mlp(S.OBS_DIM + S.LATENT_DIM, hidden, S.ACTION_DIM)
        self.log_std = nn.Parameter(torch.full((S.ACTION_DIM,), float(init_log_std)))
        self.register_buffer('latent_gain', torch.tensor(1.0 if use_latent else 0.0))

    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.obs_norm(obs), latent * self.latent_gain], dim=-1)
        return torch.tanh(self.net(x))

    def dist(self, obs, latent) -> torch.distributions.Normal:
        return torch.distributions.Normal(self(obs, latent), self.log_std.exp())


class Value(nn.Module):
    """Training-only value function V(obs, privileged) (asymmetric: sees the privileged)."""

    def __init__(self, hidden: list[int]):
        super().__init__()
        self.obs_norm = RunningNorm(S.OBS_DIM)
        self.priv_norm = RunningNorm(S.PRIVILEGED_DIM)
        self.net = mlp(S.OBS_DIM + S.PRIVILEGED_DIM, hidden, 1)

    def forward(self, obs, priv):
        return self.net(torch.cat([self.obs_norm(obs), self.priv_norm(priv)], -1))[..., 0]


class TCNAdapter(nn.Module):
    """Student adaptation module: history (B, 40, 22), oldest first -> latent (B, 8).

    Three dilated causal 1-D convolutions (kernel 3, dilations 1, 2, 4, receptive field 15
    steps) followed by a linear read-out of all time steps. Obs columns are normalised with
    the policy's frozen statistics; action columns are already in [-1, 1].
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        self.obs_norm = RunningNorm(S.OBS_DIM)
        layers, c_in = [], HISTORY_WIDTH
        for dil in (1, 2, 4):
            layers += [nn.ConstantPad1d((2 * dil, 0), 0.0),
                       nn.Conv1d(c_in, channels, kernel_size=3, dilation=dil), nn.ELU()]
            c_in = channels
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels * S.HISTORY_LEN, 64),
                                  nn.ELU(), nn.Linear(64, S.LATENT_DIM))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        obs = self.obs_norm(history[..., :S.OBS_DIM])
        x = torch.cat([obs, history[..., S.OBS_DIM:]], dim=-1).transpose(1, 2)
        return self.head(self.conv(x))


class RiskCritic(nn.Module):
    """Entrapment risk rho = sigmoid(f(obs, latent) / T); T fitted on held-out seeds."""

    def __init__(self, hidden: list[int]):
        super().__init__()
        self.obs_norm = RunningNorm(S.OBS_DIM)
        self.net = mlp(S.OBS_DIM + S.LATENT_DIM, hidden, 1)
        self.register_buffer('temperature', torch.tensor(1.0))

    def logits(self, obs, latent) -> torch.Tensor:
        """Return the uncalibrated logits, shape (B, 1)."""
        return self.net(torch.cat([self.obs_norm(obs), latent], dim=-1))

    def forward(self, obs, latent) -> torch.Tensor:
        return torch.sigmoid(self.logits(obs, latent) / self.temperature)
