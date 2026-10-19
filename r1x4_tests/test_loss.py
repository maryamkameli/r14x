"""CPU checks of the patched token loss (run from the kit's verl/ dir): p-drop, CHES keep mask, row-mean fix."""
import sys, torch
from omegaconf import OmegaConf
sys.path.insert(0, '.')
from verl.adaptive_mix_src.mix_core_alg import compute_token_on_off_policy_loss as F

torch.manual_seed(0)
B, L = 8, 16
cfg = lambda **kw: OmegaConf.create(dict(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2, off_policy_normalize=False,
    off_policy_max_clip=-1, off_policy_min_clip=-1, all_max_clip=-1, off_policy_reshape='p_div_p_0.1', loss_remove_token_mean=True,
    loss_remove_clip=True, **kw))
lp = torch.randn(B, L).clamp(max=0) * 3
olp = lp.detach().clone()
lens = torch.randint(4, L + 1, (B,))
eos = (torch.arange(L)[None] < lens[:, None]).float()
adv = torch.tensor([.5, .5, -.5, -.5, .5, -.5, .5, -.5])[:, None] * eos
pre = torch.zeros(B, L, dtype=torch.bool); pre[:4] = eos[:4].bool()          # rows 0-3 teacher
lp.requires_grad_(True)

def grad(**kw):
    lp.grad = None
    out = F(olp, lp, adv, eos, pre, config=cfg(**{k: v for k, v in kw.items() if k != 'keep'}), token_keep=kw.get('keep'))
    out['pg_loss'].backward()
    return lp.grad.clone(), out

g0, _ = grad()
g1, o1 = grad(teacher_token_min_prob=0.01)
low = pre & eos.bool() & (olp.exp() < 0.01)
assert torch.equal((g1 == 0) & (g0 != 0), low), 'p-drop must zero exactly teacher tokens with p_old < 0.01'
print('p-drop ok: dropped', int(low.sum()), 'of', int((pre & eos.bool()).sum()), 'teacher tokens; frac', float(o1['teacher_lowp_drop_frac']))
keep = torch.ones(B, L); keep[2, :2] = 0; keep[5, 1] = 0
g2, o2 = grad(keep=keep)
assert torch.equal((g2 == 0) & (g0 != 0), (keep == 0) & eos.bool()), 'CHES keep mask'
print('ches mask ok, frac', float(o2['ches_drop_frac']))
# row-mean fix: per-row weight must be 1/(L*mini) whatever the micro-batch split (actor multiplies by rows/mini)
def total(split, fix):
    lp.grad = None
    for idx in split:
        idx = torch.tensor(idx)
        out = F(olp[idx], lp[idx], adv[idx], eos[idx], pre[idx], config=cfg(loss_row_mean_fix=fix))
        (out['pg_loss'] * len(idx) / B).backward()           # = loss_scale_factor under use_dynamic_bsz
    return lp.grad.clone()
ref = total([[i] for i in range(B)], False)                   # micro=1: the original R1x4 setting
assert torch.allclose(total([[0, 1, 2], [3, 4], [5, 6, 7]], True), ref, atol=1e-7), 'fix must equal micro=1'
assert not torch.allclose(total([[0, 1, 2], [3, 4], [5, 6, 7]], False), ref), 'unfixed dynamic split should differ'
print('row-mean fix ok: dynamic split == micro=1; unfixed dynamic split differs (the AMPO weighting bug)')
