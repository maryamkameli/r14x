"""Variant IV CPU checks (run from the kit's verl/ dir): truncated-IS gradient, lambda schedule, AdamW reset."""
import math, sys, torch
from omegaconf import OmegaConf
sys.path.insert(0, '.')
from verl.adaptive_mix_src.mix_core_alg import compute_token_on_off_policy_loss as F
torch.manual_seed(0)
B, L, cbar = 4, 12, 1.0
cfg = OmegaConf.create(dict(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2, off_policy_normalize=False, off_policy_max_clip=-1,
    off_policy_min_clip=-1, all_max_clip=-1, off_policy_reshape='p_div_p_0.1', loss_remove_token_mean=True, loss_remove_clip=True, is_cbar=cbar))
lp = (torch.randn(B, L).clamp(max=0) * 2).requires_grad_(True)
mu = lp.detach() + torch.randn(B, L)               # some tokens pi/mu > cbar (truncated), some < cbar
eos = torch.ones(B, L); adv = torch.tensor([.5, .5, -.5, .5])[:, None] * eos
pre = torch.zeros(B, L, dtype=torch.bool); pre[:2] = True; mm = pre.clone(); mm[1] = False   # row 0: IS, row 1: shaping
out = F(lp.detach(), lp, adv, eos, pre, config=cfg, teacher_logp=mu, teacher_mu_mask=mm)
out['pg_loss'].backward()
rho = torch.minimum(torch.full_like(mu, cbar), (lp.detach() - mu).exp())
expect = -adv[0] * rho[0] / L                        # d/dlogpi of -A*min(cbar, pi/mu)*logpi, sum/L normalizer
assert torch.allclose(lp.grad[0], expect, atol=1e-6), (lp.grad[0], expect)
trunc = (lp.detach() - mu)[0] > math.log(cbar)
assert trunc.any() and (lp.grad[0][trunc] != 0).all(), 'truncated tokens must keep a nonzero gradient'
p = lp.detach()[1].exp(); f = p / (p + 0.1)
assert torch.allclose(lp.grad[1], -adv[1] * f * (0.1 / (p + 0.1)) / L, atol=1e-6), 'rows without mu keep p/(p+0.1)'
print('IS ok: grad = -A min(cbar, pi/mu) dlogpi on mu tokens,', int(trunc.sum()), 'truncated tokens keep gradient; shaping rows unchanged')
# lambda schedule (as in the trainer)
lam = lambda t, T, l0=1.0: l0 * 0.5 * (1 + math.cos(math.pi * (t - 1) / T)) if t <= T else 0.0
T = 60
assert lam(1, T) == 1.0 and 0 < lam(T, T) < 1e-2 and lam(T + 1, T) == 0.0
print(f'lambda ok: l(1)=1, l(T_a)={lam(T, T):.2e}, l(T_a+1)=0')
# AdamW reset == fresh optimizer's first step
w = torch.nn.Parameter(torch.randn(5)); o = torch.optim.AdamW([w], lr=1e-2)
for _ in range(3):
    o.zero_grad(); (w ** 2).sum().backward(); o.step()
o.state.clear(); w2 = torch.nn.Parameter(w.detach().clone()); o2 = torch.optim.AdamW([w2], lr=1e-2)
for opt, x in ((o, w), (o2, w2)):
    opt.zero_grad(); (x ** 3).sum().backward(); opt.step()
assert torch.allclose(w, w2), 'reset must equal a fresh optimizer'
print('AdamW reset ok')
