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
# generating-teacher p-drop: negative teacher tokens with p_teacher < 0.01 lose their gradient too; others unchanged
tgl = torch.randn(B, L).clamp(max=0) * 3
tgl = torch.where(pre, tgl, torch.zeros_like(tgl))           # 0 (= p 1) outside teacher rows, as the trainer stores it
def grad2(**kw):
    lp.grad = None
    out = F(olp, lp, adv, eos, pre, config=cfg(**kw), teacher_gen_logp=tgl)
    out['pg_loss'].backward()
    return lp.grad.clone(), out
neg_t = pre & eos.bool() & (adv.sum(-1, keepdim=True) < 0)
g4, o4 = grad2(teacher_token_min_prob=0.01, teacher_token_min_prob_rows='neg', teacher_token_min_teacher_prob=0.01)
g1n, _ = grad2(teacher_token_min_prob=0.01, teacher_token_min_prob_rows='neg')
low_t = neg_t & (tgl.exp() < 0.01)
assert torch.equal((g4 == 0) & (g1n != 0), low_t), 'teacher p-drop must zero exactly negative teacher tokens with p_teacher < 0.01'
assert torch.equal(g4[~low_t], g1n[~low_t]), 'all other tokens unchanged'
assert abs(float(o4['teacher_gen_lowp_drop_frac']) - float(low_t.sum() / neg_t.sum())) < 1e-6
print('teacher p-drop ok: dropped', int(low_t.sum()), 'of', int(neg_t.sum()), 'negative teacher tokens')
# outside-support drop: teacher tokens stored at -1e4 (= -inf, outside the generator's sampling support) lose their
# gradient on positive AND negative teacher rows; student rows and in-support teacher tokens are unchanged
tgo = tgl.clone()
tgo[(torch.rand(B, L) < 0.2) & pre] = -1e4
out_t = pre & eos.bool() & (tgo < -1e3)
def grad3(**kw):
    lp.grad = None
    out = F(olp, lp, adv, eos, pre, config=cfg(**kw), teacher_gen_logp=tgo)
    out['pg_loss'].backward()
    return lp.grad.clone(), out
g5, o5 = grad3(teacher_drop_outside_support=True)
g5b, _ = grad3()
assert (out_t & (adv.sum(-1, keepdim=True) > 0)).any() and (out_t & (adv.sum(-1, keepdim=True) < 0)).any(), 'test needs both kinds'
assert torch.equal((g5 == 0) & (g5b != 0), out_t), 'support drop must zero exactly the out-of-support teacher tokens, pos and neg rows'
assert torch.equal(g5[~out_t], g5b[~out_t]), 'all other tokens unchanged'
assert abs(float(o5['teacher_out_support_drop_frac']) - float(out_t.sum() / (pre & eos.bool()).sum())) < 1e-6
print('support drop ok: dropped', int(out_t.sum()), 'of', int((pre & eos.bool()).sum()), 'teacher tokens (pos + neg rows)')
# 1 - p weight: every token of a negative row (teacher AND student) x (1 - sg(p_old)); positive rows unchanged
g5, o5 = grad2(teacher_token_min_prob=0.01, teacher_token_min_prob_rows='neg', teacher_token_min_teacher_prob=0.01,
               neg_one_minus_p=True)
neg_all = eos.bool() & (adv.sum(-1, keepdim=True) < 0)
assert torch.allclose(g5[neg_all], (g4 * (1 - olp.exp()))[neg_all], atol=1e-6), 'negative tokens: grad x (1 - p_old)'
assert torch.equal(g5[~neg_all], g4[~neg_all]), 'positive rows unchanged'
assert (neg_all & ~pre).any() and (g5[neg_all & ~pre] != g4[neg_all & ~pre]).any(), 'student negative rows are weighted too'
print('1-p weight ok: mean weight', float(o5['neg_weight_mean']), 'on', int(neg_all.sum()), 'negative tokens (teacher + student)')
