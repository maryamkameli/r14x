"""Run R1x4Trainer.fit() for 4 steps on CPU with mocked workers (variant IV: T_a=2). Checks the batches sent to update_actor."""
import sys, math, types, torch, numpy as np
sys.path.insert(0, '.')
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl import DataProto
import verl.r1x4.trainer as T
from verl.r1x4.trainer import R1x4Trainer, R1x4Data, Curriculum, _CurriculumState

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
B, R, P = 4, 192, 1024
cfg = OmegaConf.create({'data': {'train_batch_size': B, 'max_prompt_length': P, 'max_response_length': R},
    'actor_rollout_ref': {'rollout': {'temperature': 1.0}}, 'trainer': {'project_name': 'x', 'experiment_name': 'x', 'logger': ['console'],
    'test_freq': 0, 'save_freq': 0, 'val_before_train': False, 'resume_mode': 'disable'},
    'r1x4': {'seed': 47, 'rank': {'enable': False}, 'ches': {'enable': False}, 'student_eos_ids': [151645],
             'anneal': {'enable': True, 'frac': 0.5, 'lambda0': 1.0, 'gate': 'always', 'len_ref': 0, 'reset_adam': True, 'diag_every': 1},
             'isamp': {'enable': True, 'cbar': 1.0, 'mu_pool': None}}})
APPEND = len(sys.argv) > 1 and sys.argv[1] == 'append'   # recipe layout: 4 students + K=2 positive rows, student baseline, + CHES + ranking
if APPEND:
    cfg.r1x4.layout, cfg.r1x4.k_teacher, cfg.r1x4.baseline = 'append', 2, 'student'
    cfg.r1x4.anneal.neg_weight, cfg.r1x4.anneal.len_ref = 0.0, 100.0
    cfg.r1x4.ches = {'enable': True, 'drop_frac': 0.5, 'score': 'cos', 'rows': 'all'}
    cfg.r1x4.rank = {'enable': True, 'pool': 16, 'top': 8, 'max_neg': 4, 'score': 'sum', 'chunk': 64}
tr = R1x4Trainer.__new__(R1x4Trainer)
tr.config, tr.tokenizer, tr.rc, tr.rng_seed, tr.total_training_steps, tr.val_reward_fn = cfg, tok, cfg.r1x4, 47, 4, None
tr.data = R1x4Data('/scratch.global/radke149/r1x4_runs/smoke/data/smoke.parquet', tok, P, R, tok.convert_tokens_to_ids('<|im_end|>'))
tr.cur = Curriculum(len(tr.data), 1000, 100, 2); tr.train_dataloader = _CurriculumState(tr.cur)
tr.mu = {}
for i, r in enumerate(tr.data.rows[:12]):               # fake mu-traces for half the prompts: R1 ids with random log-probs
    if i % 2 == 0:
        ids = tr.data.teacher_ids(i, r['correct_text'])
        tr.mu[r['luffy_index']] = [(ids, -torch.rand(len(ids)))]
correct = set()
for i, r in enumerate(tr.data.rows[:20]):
    correct.add(tuple(tr.data.teacher_ids(i, r['correct_text']).tolist()))
rng = np.random.default_rng(0)
class WG:
    world_size = 4
    sent = []
    def generate_sequences(self, g):
        n = len(g); resp = torch.full((n, R), tok.pad_token_id)
        am = torch.zeros(n, R, dtype=torch.long)
        for k in range(n):
            L = int(rng.integers(5, R)); resp[k, :L] = torch.randint(100, 1000, (L,)); resp[k, L - 1] = 151645; am[k, :L] = 1
        a = torch.cat([g.batch['attention_mask'], am], -1)
        return DataProto.from_dict(tensors={'prompts': g.batch['input_ids'], 'responses': resp, 'input_ids': torch.cat([g.batch['input_ids'], resp], -1),
                                            'attention_mask': a, 'position_ids': torch.clip(a.cumsum(-1) - 1, min=0)},
                                   non_tensors={'raw_prompt_ids': g.non_tensor_batch['raw_prompt_ids']}, meta_info={'timing': {}})
    def compute_log_prob(self, b):
        n = len(b); return DataProto.from_dict(tensors={'old_log_probs': -torch.rand(n, R) * 2, 'entropys': torch.rand(n, R)})
    def compute_hidden_stats(self, b):
        n, H = len(b), 8
        if b.meta_info['ches_mode'] == 'sum':
            return DataProto.from_dict(tensors={'ches_sum': torch.randn(n, H)})
        return DataProto.from_dict(tensors={'ches_score': torch.randn(n, R)})
    def update_actor(self, b):
        self.sent.append(b); return DataProto(meta_info={'metrics': {'actor/grad_norm': [0.1]}})
tr.actor_rollout_wg = WG()
class Rew:
    def __call__(self, d, return_dict=False):
        Pn = d.batch['prompts'].shape[-1]; lens = d.batch['attention_mask'][:, Pn:].sum(-1)
        rt = torch.zeros(len(d), R)
        for k in range(len(d)):
            ids = tuple(d.batch['responses'][k, :lens[k]].tolist())
            s = 1.0 if ids in correct else (float(rng.random() < 0.4) if max(ids[:-1]) < 1000 else 0.0)  # students: random token ids < 1000
            rt[k, lens[k] - 1] = s
        return {'reward_tensor': rt}
tr.reward_fn = Rew()
tr._load_checkpoint = lambda: 0
import verl.utils.tracking as trk
trk.Tracking = lambda **kw: types.SimpleNamespace(log=lambda data, step: logs.append((step, data)))
logs = []
tr.fit()
for t, b in enumerate(WG.sent, 1):
    pm, adv, m = b.batch['prefix_mask'], b.batch['advantages'], b.batch['response_mask'].float()
    row_adv = (adv * m).sum(-1) / m.sum(-1).clamp(min=1)
    lam = 0.5 * (1 + math.cos(math.pi * (t - 1) / 2)) if t <= 2 else 0
    md = dict(logs[t - 1][1])
    print(f"step {t}: rows {len(b)}, teacher rows {int(pm.any(-1).sum())}, lambda {md.get('teacher/lambda')}, reset {b.meta_info.get('reset_optim')}, "
          f"diag {b.meta_info.get('teacher_diag')}, mu rows {int(b.batch['teacher_mu_mask'].any(-1).sum()) if 'teacher_mu_mask' in b.batch.keys() else '-'}, "
          f"student |adv| {row_adv[~pm.any(-1)].abs().unique().tolist()}, teacher |adv| {row_adv[pm.any(-1)].abs().unique().tolist()}, "
          f"dropped {md.get('train/groups_dropped_label_mismatch')}")
    if APPEND and t <= 2:
        K = 2
        assert len(b) == B * (4 + K) and pm.any(-1).sum() == B * K, (len(b), int(pm.any(-1).sum()))
        sc = b.batch['token_level_scores'].sum(-1).view(B, 4 + K); lens = b.batch['attention_mask'][:, -R:].sum(-1).float().view(B, 4 + K)
        ok = (b.batch['response_mask'].sum(-1) > 0).view(B, 4 + K)
        base = sc[:, K:].mean(-1, keepdim=True)
        exp_s = (sc[:, K:] - base); exp_t = lam * (sc[:, :K] - base) * (100.0 / lens[:, :K])
        ra = row_adv.view(B, 4 + K)
        assert torch.allclose(ra[:, K:][ok[:, K:]], exp_s[ok[:, K:]], atol=1e-5), 'student-only baseline'
        assert torch.allclose(ra[:, :K][ok[:, :K]], exp_t[ok[:, :K]], atol=1e-5), 'teacher rows = lambda (1-b) L_ref/len'
        assert 'ches_keep' in b.batch.keys()
        print(f'   append ok: student adv r-mean(students), teacher adv lambda*(1-b)*L_ref/len, CHES mask dropped {int((b.batch["ches_keep"] == 0).sum())} tokens')
    elif t <= 2:
        assert len(b) == B * 8 and md['train/groups_dropped_label_mismatch'] == 0 and pm.any(-1).sum() == B * 4 and abs(md['teacher/lambda'] - lam) < 1e-9
        assert torch.allclose(row_adv[pm.any(-1)].abs(), torch.full((B * 4,), 0.5 * lam)), 'teacher rows x lambda'
        ok = ~b.batch['teacher_mu_mask'].any(-1) | pm.any(-1)
        assert ok.all(), 'mu only on teacher rows'
        mm = b.batch['teacher_mu_mask']
        assert (b.batch['teacher_logp'][mm] < 0).all(), 'teacher_logp must be aligned with mu tokens'
    else:
        assert len(b) == B * 4 and not pm.any() and not b.batch['teacher_mu_mask'].any()
        g = b.batch['token_level_scores'].sum(-1).view(B, 4)
        assert torch.allclose(row_adv.view(B, 4), (g - g.mean(-1, keepdim=True)) * (m.sum(-1).view(B, 4) > 0)), 'GRPO-4 advantages'
    assert b.meta_info.get('reset_optim') == (t == 3)
print('variant IV fit loop OK: teacher phase steps 1-2 (lambda 1 -> 0.5), Adam reset at 3, GRPO-4 at 3-4')
