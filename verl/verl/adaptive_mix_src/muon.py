"""Muon (Keller Jordan 2024) with the RMS-matched scaling of 'Muon is Scalable for LLM Training' (Moonlight 2025).

Math and persistent state only; the FSDP plumbing (full-param summon, AdamW for the non-target parameters) lives
in mix_actor.MIXDataParallelPPOActor._muon_optimizer_step. Targets are selected with logra.is_target.

Per selected weight W in R^{d x k} with its reduced, clipped fp32 gradient G:
  M <- momentum * M + G                                   (fp32 buffer, zero-initialised)
  U <- G + momentum * M  (nesterov)  |  M
  O <- NewtonSchulz5(U)                                   (fp32, TF32 off)
  W <- W * (1 - lr * wd) - lr * scale * O,   scale = 0.2 * sqrt(max(d, k)) ('moonlight') | 1 ('none')
lr / wd are the actor AdamW's current (scheduled) lr and weight decay.
"""

from __future__ import annotations

import math

import torch

# Defaults; every key may be missing or None in meta_info['muon'] (None == missing).
DEFAULTS = {
    'enable': False,
    'momentum': 0.95,
    'nesterov': True,
    'ns_steps': 5,
    'scale_mode': 'moonlight',  # 'moonlight': 0.2 * sqrt(max(d, k)) (reuses AdamW lr / wd); 'none': raw O
    'targets': 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
    'step': 1,
    'total_steps': None,
}

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def resolve_config(raw: dict | None) -> dict | None:
    """meta_info['muon'] -> config dict with defaults filled, or None when Muon is off."""
    if not raw or not raw.get('enable'):
        return None
    cfg = {k: (raw.get(k) if raw.get(k) is not None else v) for k, v in DEFAULTS.items()}
    targets = cfg['targets']
    cfg['targets'] = tuple(t.strip() for t in (targets.split(',') if isinstance(targets, str) else targets) if t.strip())
    cfg['momentum'] = float(cfg['momentum'])
    cfg['nesterov'] = bool(cfg['nesterov'])
    cfg['ns_steps'] = int(cfg['ns_steps'])
    cfg['step'] = int(cfg['step'])
    if not 0.0 <= cfg['momentum'] < 1.0:
        raise ValueError(f"muon.momentum must be in [0, 1), got {cfg['momentum']}")
    if cfg['ns_steps'] < 1:
        raise ValueError(f"muon.ns_steps must be >= 1, got {cfg['ns_steps']}")
    if cfg['scale_mode'] not in ('moonlight', 'none'):
        raise ValueError(f"muon.scale_mode must be 'moonlight' or 'none', got {cfg['scale_mode']!r}")
    if not cfg['targets']:
        raise ValueError('muon.targets is empty')
    return cfg


def update_scale(shape, scale_mode: str) -> float:
    """0.2 * sqrt(max(d, k)) matches the update RMS of AdamW (Moonlight eq. 4); 1.0 for 'none'."""
    return 0.2 * math.sqrt(max(int(shape[0]), int(shape[1]))) if scale_mode == 'moonlight' else 1.0


@torch.no_grad()
def newton_schulz5(U: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalisation (Keller Jordan's coefficients), fp32.
    The caller disables TF32 (logra.exact_fp32_matmul)."""
    assert U.ndim == 2
    a, b, c = NS_COEFFS
    X = U.to(torch.float32)
    X = X / (X.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        del A, B
    return X.T if transposed else X


class MuonState:
    """Persistent Muon state: fp32 momentum buffers keyed by clean parameter name.
    Identical on every data-parallel rank (computed redundantly from identical reduced grads)."""

    def __init__(self):
        self.momentum: dict[str, torch.Tensor] = {}
        self.num_updates = 0

    def buffer(self, name: str, like: torch.Tensor) -> torch.Tensor:
        """fp32 momentum buffer on like.device; zero-initialised, or a restored CPU buffer moved on first use."""
        M = self.momentum.get(name)
        if M is None or M.shape != like.shape:
            M = torch.zeros(like.shape, dtype=torch.float32, device=like.device)
            self.momentum[name] = M
        elif M.device != like.device or M.dtype != torch.float32:
            M = M.to(device=like.device, dtype=torch.float32)
            self.momentum[name] = M
        return M

    @torch.no_grad()
    def direction(self, cfg: dict, name: str, G: torch.Tensor) -> torch.Tensor:
        """Updates the momentum buffer in place with fp32 G and returns U (nesterov: G + mu M; else M itself)."""
        M = self.buffer(name, G)
        mu = cfg['momentum']
        M.mul_(mu).add_(G)
        return G.add(M, alpha=mu) if cfg['nesterov'] else M

    # ---------------------------------------------------------------- checkpoint
    def state_dict(self) -> dict:
        return {
            'version': 1,
            'momentum': {k: v.detach().to('cpu', copy=True) for k, v in self.momentum.items()},
            'num_updates': self.num_updates,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.momentum = {k: v.to(torch.float32) for k, v in sd.get('momentum', {}).items()}  # moved to device on use
        self.num_updates = int(sd.get('num_updates', 0))
