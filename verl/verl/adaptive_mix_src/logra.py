"""LoGRA: low-rank gradient sketches + predicted-KL step control (arXiv 2610.06647).

Math and persistent state only; the FSDP plumbing (full-param summon, probe forwards) lives in
mix_actor.MIXDataParallelPPOActor. Follows the reference implementation
(skzhang1/labs-molt@logra, molt/trainer/logra/{projection,optimizer,kl_probe}.py) where the paper is vague.

Per selected weight W in R^{d x k} (d = out rows, k = in columns) and update u:
  A_u in R^{r x k}, iid entries +-1/sqrt(r) (Rademacher) or N(0, 1/r), seeded by (seed, round, name)
  S_u = G_u A_u^T                                            (fp32 sketch of the reduced, clipped grad)
  U_u = S_u (sgd)  |  RowAdam(S_u)                             (see `LoGRAState.direction`)
  D_u = -eta U_u A_u                                           (proposal, rank <= r)
  W  <- W + alpha D_u,  alpha = min(alpha_max, sqrt(delta_eff / q(D)))
"""

from __future__ import annotations

import contextlib
import hashlib
import math

import torch

_MASK62 = (1 << 62) - 1

# Defaults; every key may be missing or None in meta_info['logra'] (None == missing).
DEFAULTS = {
    'enable': False,
    'rank': 256,  # paper main runs / reference CLI default
    'refresh': True,  # paper Tab. 4: projection refreshed every update
    'dist': 'rademacher',
    'optimizer': 'rowadam',  # paper main runs
    'lr': None,  # None => reference 'target' mode (eta set from target_norm); a number => literal D = -lr U A
    'target_norm': 6e-5,  # reference --actor.logra.target_norm (LograConfig.max_update_ratio, update_mode='target')
    'beta1': 0.9,  # unused: RowAdam keeps no first moment (paper App. B)
    'beta2': 0.95,  # paper Tab. 4 / reference math.sh adam betas (0.9, 0.95)
    'eps': 1e-8,  # relative: eps_t = eps * mean_i sqrt(vhat_i) (paper Tab. 4, reference eps_relative=True)
    'mismatch_subtraction': True,
    'mismatch_ema': 0.3,  # paper App. D.1 / reference kl_mismatch_ema
    'mismatch_floor': 0.5,  # paper eq. 17 / reference kl_mismatch_floor
    'kl_budget_start': 2e-4,
    'kl_budget_end': 2e-5,
    'kl_budget_steps': None,  # H of the cosine schedule; None => total_steps
    'alpha_max': 3.0,  # reference --actor.logra.alpha_max / LograConfig.kl_alpha_max
    'probe_rows': 2,  # reference --actor.logra.probe_sequences (per rank)
    'probe_tokens': 256,  # response positions per probe row
    'targets': 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
    'seed': 0,
    'step': 1,
    'total_steps': None,
}


def resolve_config(raw: dict | None) -> dict | None:
    """meta_info['logra'] -> config dict with defaults filled, or None when LoGRA is off."""
    if not raw or not raw.get('enable'):
        return None
    cfg = {k: (raw.get(k) if raw.get(k) is not None else v) for k, v in DEFAULTS.items()}
    targets = cfg['targets']
    cfg['targets'] = tuple(t.strip() for t in (targets.split(',') if isinstance(targets, str) else targets) if t.strip())
    cfg['rank'] = int(cfg['rank'])
    cfg['probe_rows'] = int(cfg['probe_rows'])
    cfg['probe_tokens'] = int(cfg['probe_tokens'])
    cfg['step'] = int(cfg['step'])
    if cfg['optimizer'] not in ('rowadam', 'sgd'):
        raise ValueError(f"logra.optimizer must be 'rowadam' or 'sgd', got {cfg['optimizer']!r}")
    if cfg['dist'] not in ('rademacher', 'gaussian'):
        raise ValueError(f"logra.dist must be 'rademacher' or 'gaussian', got {cfg['dist']!r}")
    if cfg['rank'] < 1:
        raise ValueError(f"logra.rank must be >= 1, got {cfg['rank']}")
    if cfg['lr'] is not None and not float(cfg['lr']) > 0:
        raise ValueError(f"logra.lr must be > 0 or null, got {cfg['lr']}")
    if not float(cfg['alpha_max']) > 0:
        raise ValueError(f"logra.alpha_max must be > 0, got {cfg['alpha_max']}")
    return cfg


def clean_name(name: str) -> str:
    return name.replace('_fsdp_wrapped_module.', '').replace('_checkpoint_wrapped_module.', '')


def is_target(name: str, param: torch.Tensor, targets) -> bool:
    """2-D projection weights of decoder layers whose (clean) name contains one of `targets`."""
    return param.ndim == 2 and '.layers.' in name and name.endswith('.weight') and any(t in name for t in targets)


def layer_seed(base_seed: int, round_idx: int, update_idx: int, name: str) -> int:
    """Deterministic 62-bit seed from (seed, round, update-within-step, parameter name); identical on all ranks."""
    key = f'{int(base_seed)}|{int(round_idx)}|{int(update_idx)}|{name}'.encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little') & _MASK62


def make_projection(r: int, k: int, seed: int, kind: str, device) -> torch.Tensor:
    """A in R^{r x k} with E[A_ij^2] = 1/r (so E[A^T A] = I_k), drawn on CPU for cross-device determinism."""
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed))
    if kind == 'rademacher':
        A = torch.randint(0, 2, (r, k), generator=g, dtype=torch.uint8).to(torch.float32).mul_(2.0).sub_(1.0)
    else:
        A = torch.randn(r, k, generator=g, dtype=torch.float32)
    return A.mul_(1.0 / math.sqrt(r)).to(device, non_blocking=True)


@contextlib.contextmanager
def exact_fp32_matmul():
    """Plain fp32 GEMMs (no TF32) for sketches, updates and probe logits."""
    prev_mm, prev_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def kl_budget(cfg: dict) -> float:
    """Paper eq. 16 (= reference kl_scale): cosine from kl_budget_start to kl_budget_end over H updates,
    held at kl_budget_end afterwards. u = cfg['step'] (1-based training step), H = kl_budget_steps or total_steps."""
    hi, lo = float(cfg['kl_budget_start']), float(cfg['kl_budget_end'])
    H = cfg['kl_budget_steps'] or cfg['total_steps']
    if not H:
        return hi
    frac = min(max(cfg['step'] - 1, 0) / max(int(H) - 1, 1), 1.0)
    return lo + (hi - lo) * (1.0 + math.cos(math.pi * frac)) / 2.0


def step_scale(q: float, budget: float, alpha_max: float) -> tuple[float, str]:
    """Paper eq. 7: alpha = min(alpha_max, sqrt(budget / q)). Returns (alpha, status).
    q non-finite -> alpha 0 (the trial step is reverted); q <= 0 (update invisible to the bf16 probe) -> alpha_max."""
    if not math.isfinite(q):
        return 0.0, 'q_nonfinite'
    if q <= 0.0:
        return float(alpha_max), 'q_zero'
    return float(min(alpha_max, math.sqrt(budget / q))), 'ok'


@torch.no_grad()
def probe_kl_sum(h0: torch.Tensor, h1: torch.Tensor, w_head: torch.Tensor, temperature: float, chunk: int = 128):
    """sum over probe positions of KL(softmax(h0 W^T / T) || softmax(h1 W^T / T)); logits in fp32, softmax in fp64."""
    total = torch.zeros((), dtype=torch.float64, device=w_head.device)
    with exact_fp32_matmul():
        for c in range(0, h0.shape[0], chunk):
            lp0 = (h0[c:c + chunk].to(w_head.device, torch.float32) @ w_head.T).double().div_(temperature).log_softmax(-1)
            lp1 = (h1[c:c + chunk].to(w_head.device, torch.float32) @ w_head.T).double().div_(temperature).log_softmax(-1)
            total += (lp0.exp() * (lp0 - lp1)).sum()
            del lp0, lp1
    return total


def stride_select(n: int, k: int, device=None) -> torch.Tensor:
    """Up to k indices spread evenly over range(n) (reference kl_probe._stride_select)."""
    if k <= 0 or n <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    step = max(1, n // k)
    return torch.arange(0, n, step, device=device)[:k]


class LoGRAState:
    """Persistent LoGRA state: RowAdam row second moments + counters, mismatch EMA, fixed projections.
    Identical on every data-parallel rank (computed redundantly from identical reduced grads)."""

    def __init__(self):
        self.v_row: dict[str, torch.Tensor] = {}  # name -> [d, 1] fp32 EMA of row mean-square of S
        self.row_t: dict[str, int] = {}  # name -> RowAdam bias-correction counter (persists across refresh)
        self.mismatch_ema: float | None = None
        self.num_updates = 0
        self._fixed_A: dict[str, torch.Tensor] = {}  # refresh=False cache (regenerated from the seed, not saved)

    # ---------------------------------------------------------------- projection
    def projection(self, cfg: dict, name: str, k: int, update_idx: int, device) -> torch.Tensor:
        if not cfg['refresh']:
            A = self._fixed_A.get(name)
            if A is None or A.shape != (cfg['rank'], k) or A.device != torch.device(device):
                A = make_projection(cfg['rank'], k, layer_seed(cfg['seed'], 0, 0, name), cfg['dist'], device)
                self._fixed_A[name] = A
            return A
        return make_projection(cfg['rank'], k, layer_seed(cfg['seed'], cfg['step'], update_idx, name), cfg['dist'], device)

    # ---------------------------------------------------------------- sketch optimizer
    @torch.no_grad()
    def direction(self, cfg: dict, name: str, S: torch.Tensor):
        """U from the sketch S [d, r]. Returns (U, pending) where `pending` is the RowAdam state update to
        pass to `commit` once the step is actually applied.

        sgd:     U = S
        rowadam: v_t = beta2 v_{t-1} + (1 - beta2) mean_j S_ij^2          (per row i, v_0 = 0)
                 vhat_t = v_t / (1 - beta2^t),   eps_t = max(eps * mean_i sqrt(vhat_t,i), tiny)
                 U~_i = S_i / (sqrt(vhat_t,i) + eps_t),   U = U~ * ||S||_F / ||U~||_F
        """
        if cfg['optimizer'] == 'sgd':
            return S, None
        b2 = float(cfg['beta2'])
        row = S.square().mean(dim=1, keepdim=True)
        v_prev = self.v_row.get(name)
        v = row.mul(1.0 - b2) if v_prev is None else v_prev.to(S.device).mul(b2).add_(row, alpha=1.0 - b2)
        t = self.row_t.get(name, 0) + 1
        vr = (v / (1.0 - b2 ** t)).sqrt()
        tiny = torch.finfo(vr.dtype).tiny
        eps_r = (float(cfg['eps']) * vr.mean()).clamp_min(tiny)
        U = S / (vr + eps_r)
        U.mul_(S.norm() / U.norm().clamp_min(tiny))
        return U, (v, t)

    def commit(self, name: str, pending) -> None:
        if pending is not None:
            self.v_row[name], self.row_t[name] = pending

    # ---------------------------------------------------------------- mismatch subtraction
    def observe_mismatch(self, delta_sum: float, n_tokens: float, rate: float) -> float:
        """kappa_u = EMA_rate(max(mean(rollout_logp - trainer_logp), 0)); no valid tokens => no update.
        Returns the raw (clamped) measurement, NaN if invalid."""
        if not n_tokens > 0:
            return float('nan')
        raw = max(delta_sum / n_tokens, 0.0)
        self.mismatch_ema = raw if self.mismatch_ema is None else (1.0 - rate) * self.mismatch_ema + rate * raw
        return raw

    def effective_budget(self, cfg: dict, scheduled: float) -> float:
        """Paper eq. 17: delta_eff = max(delta - kappa, floor * delta); no measurement yet => delta."""
        if not cfg['mismatch_subtraction'] or self.mismatch_ema is None:
            return scheduled
        return max(scheduled - self.mismatch_ema, float(cfg['mismatch_floor']) * scheduled)

    # ---------------------------------------------------------------- checkpoint
    def state_dict(self) -> dict:
        return {
            'version': 1,
            'v_row': {k: v.detach().to('cpu', copy=True) for k, v in self.v_row.items()},
            'row_t': dict(self.row_t),
            'mismatch_ema': self.mismatch_ema,
            'num_updates': self.num_updates,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.v_row = {k: v.to(torch.float32) for k, v in sd.get('v_row', {}).items()}  # moved to device on use
        self.row_t = {k: int(v) for k, v in sd.get('row_t', {}).items()}
        self.mismatch_ema = sd.get('mismatch_ema')
        self.num_updates = int(sd.get('num_updates', 0))
        self._fixed_A = {}
