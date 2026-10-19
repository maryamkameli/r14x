"""R1x4 trainer on the AMPO/verl-0.5 stack.

Every step, per prompt: 4 live student rollouts + 4 stored teacher rows, filled to EXACTLY 4 correct / 4 wrong
(k correct students -> 4-k copies of LUFFY's R1 trace + k distinct stored wrong traces). The group mean is pinned
at 0.5, so every advantage is exactly +-0.5 (plain GRPO over the 8 rows, no std). Teacher rows are off-policy
(prefix_mask) and use AMPO's token-level off-policy loss with the p/(p+0.1) reshape, exactly like our LUFFY-tree
4s4t runs (loss_remove_token_mean, no clip, lr 1e-6, no KL, entropy 1e-3).

Knobs (r1x4.* in the Hydra config, set by scripts/run.py):
  teacher_token_min_prob p  teacher tokens with pre-update policy probability < p lose their gradient (in the loss)
  ches.enable               per-token CHES on every negative row: drop the drop_frac of its response tokens whose
                            final hidden state is most aligned with the group's positive (R1) trace, S+ = sum_t h+_t
                            (cos or inner product); the shared prefix with y+ is never dropped; >= 1 token is kept
  rank.enable               curriculum: take the next `pool` (1000) prompts in LUFFY visit order, teacher-force their
                            R1 trace and their wrong traces under the CURRENT policy, keep the `top` (100) whose traces
                            are most likely, train on them in DECREASING likelihood, then the next `pool`, and so on
"""
import json
import math
import os
import time
import uuid
from collections import Counter, defaultdict
from pprint import pprint

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.metric_utils import reduce_metrics
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.adaptive_mix_src.mix_dapo_trainer import MIXRayDAPOTrainer

STUDENTS, TEACHERS, GROUP, CORRECT = 4, 4, 8, 4


class Curriculum:
    """LUFFY visit order over the dataset rows (they are stored sorted by visit), optional likelihood ranking, retirement."""

    def __init__(self, n, rank_pool, rank_top, retire_after):
        self.n, self.pool, self.top, self.retire_after = n, rank_pool, rank_top, retire_after
        self.cursor = 0          # next dataset row in visit order (wraps = new epoch)
        self.epoch = 0
        self.queue = []          # rows selected and not yet trained, in training order
        self.wrong_order = {}    # row -> wrong-trace order chosen by the ranking (most likely first)
        self.retired, self.streak = set(), {}

    def take_window(self, size):
        out = []
        while len(out) < size and len(self.retired) < self.n:
            i = self.cursor
            self.cursor += 1
            if self.cursor >= self.n:
                self.cursor, self.epoch = 0, self.epoch + 1
            if i not in self.retired:
                out.append(i)
        return out

    def observe(self, rows, all_correct):
        for i, a in zip(rows, all_correct):
            self.streak[i] = self.streak.get(i, 0) + 1 if a else 0
            if self.retire_after > 0 and self.streak[i] >= self.retire_after:
                self.retired.add(i)

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, s):
        self.__dict__.update(s)


class R1x4Data:
    def __init__(self, path, tokenizer, max_prompt_length, max_response_length, teacher_eos):
        t = pq.read_table(path)
        self.rows = t.to_pylist()
        assert all(a['visit'] <= b['visit'] for a, b in zip(self.rows, self.rows[1:])), 'rows must be sorted by LUFFY visit'
        self.tok, self.P, self.R, self.eos = tokenizer, max_prompt_length, max_response_length, teacher_eos
        self._p, self._t = {}, {}

    def __len__(self):
        return len(self.rows)

    def prompt(self, i):
        if i not in self._p:
            text = self.tok.apply_chat_template(self.rows[i]['prompt'], add_generation_prompt=True, tokenize=False)
            ids, am = verl_F.tokenize_and_postprocess_data(prompt=text, tokenizer=self.tok, max_length=self.P,
                                                           pad_token_id=self.tok.pad_token_id, left_pad=True, truncation='error')
            self._p[i] = (text, ids[0], am[0], compute_position_id_with_mask(am)[0])
        return self._p[i]

    def teacher_ids(self, i, text):
        """Response token ids of a stored trace: tokens + EOS, cut at R (a prompt ending in '<think>\\n' owns that prefix)."""
        key = (i, hash(text))
        if key not in self._t:
            if self.prompt(i)[0].endswith('<think>\n') and text.startswith('<think>\n'):
                text = text[len('<think>\n'):]
            ids = self.tok(text, add_special_tokens=False)['input_ids'] + [self.eos]
            self._t[key] = torch.tensor(ids[:self.R], dtype=torch.long)
            if len(self._t) > 200000:
                self._t.clear()
        return self._t[key]


class _CurriculumState:  # lets RayPPOTrainer._save/_load_checkpoint store the curriculum as 'data.pt'
    def __init__(self, cur):
        self.cur = cur

    def state_dict(self):
        return self.cur.state_dict()

    def load_state_dict(self, s):
        self.cur.load_state_dict(s)

    def __len__(self):
        return 1


class R1x4Trainer(MIXRayDAPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, off_policy_reward_fn=None, **kwargs)
        self.rc = self.config.r1x4
        self.rng_seed = int(self.rc.get('seed', 47))

    # ------------------------------------------------------------------ data
    def _create_dataloader(self):
        c, d = self.config, self.config.data
        eos = self.config.r1x4.get('teacher_eos_token', None)
        eos_id = self.tokenizer.convert_tokens_to_ids(eos) if eos else self.tokenizer.eos_token_id
        self.data = R1x4Data(d.train_files, self.tokenizer, d.max_prompt_length, d.max_response_length, eos_id)
        rk = self.config.r1x4.rank
        self.cur = Curriculum(len(self.data), rk.pool, rk.top, self.config.r1x4.retire_after_all_correct)
        self.train_dataloader = _CurriculumState(self.cur)
        self.mu = {}  # variant IV: luffy_index -> [(token_ids, token_logps)] of self-sampled correct teacher traces
        isamp = self.config.r1x4.isamp
        if isamp.enable:
            import glob
            files = sorted(f for pat in str(isamp.mu_pool).split(',') for f in glob.glob(pat))
            assert files, f'r1x4.isamp.mu_pool={isamp.mu_pool} matches no file'
            for f in files:
                for r in pq.read_table(f).to_pylist():
                    tr = [(torch.tensor(x['token_ids'], dtype=torch.long), torch.tensor(x['token_logps'], dtype=torch.float32))
                          for x in (r['traces'] or [])]
                    if tr:
                        self.mu[r['luffy_index']] = tr
            have = sum(1 for r in self.data.rows if r['luffy_index'] in self.mu)
            print(f'[r1x4] mu pool: {len(files)} files, {len(self.mu)} prompts with a mu-trace ({have} in this dataset)')
        self.val_dataset = RLHFDataset(data_files=d.val_files, tokenizer=self.tokenizer, config=d)
        self.val_dataloader = StatefulDataLoader(dataset=self.val_dataset, batch_size=len(self.val_dataset), num_workers=0,
                                                 shuffle=False, drop_last=False, collate_fn=collate_fn)
        self.total_training_steps = c.trainer.total_training_steps
        assert self.total_training_steps, 'set trainer.total_training_steps'
        with open_dict(c):
            c.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
        print(f'[r1x4] train rows {len(self.data)} ({d.train_files}), teacher EOS {self.tokenizer.convert_ids_to_tokens(eos_id)!r}, '
              f'val {len(self.val_dataset)}, steps {self.total_training_steps}')

    # ------------------------------------------------------------- helpers
    def _rows_proto(self, prompt_rows, responses):
        """Teacher-forced rows: left-padded prompt [P] + right-padded response [R]."""
        R, pad = self.config.data.max_response_length, self.tokenizer.pad_token_id
        pr = torch.stack([self.data.prompt(i)[1] for i in prompt_rows])
        pm = torch.stack([self.data.prompt(i)[2] for i in prompt_rows])
        rs = torch.full((len(responses), R), pad, dtype=torch.long)
        rm = torch.zeros((len(responses), R), dtype=pm.dtype)
        for k, r in enumerate(responses):
            rs[k, :len(r)], rm[k, :len(r)] = r, 1
        am = torch.cat([pm, rm], -1)
        return DataProto.from_dict(tensors={'prompts': pr, 'responses': rs, 'input_ids': torch.cat([pr, rs], -1),
                                            'attention_mask': am, 'position_ids': compute_position_id_with_mask(am)})

    def _wg_call(self, fn, proto, chunk=None):
        """Call a DP worker method on any number of rows: pad to the DP world size, optionally in chunks."""
        ws = self.actor_rollout_wg.world_size
        chunk = chunk or len(proto)
        outs = []
        for s in range(0, len(proto), chunk):
            part = proto.slice(s, min(s + chunk, len(proto))) if len(proto) > chunk else proto
            part.meta_info.update(proto.meta_info)
            padded, pad = pad_dataproto_to_divisor(part, ws)
            outs.append(unpad_dataproto(fn(padded), pad))
        return DataProto.concat(outs) if len(outs) > 1 else outs[0]

    @staticmethod
    def _mean_logp(lp, mask):
        return ((lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)).tolist()

    # ------------------------------------------------------------- ranking
    def _refill(self, batch_size, metrics, timing):
        rk = self.rc.rank
        while len(self.cur.queue) < batch_size:
            window = self.cur.take_window(rk.pool if rk.enable else batch_size - len(self.cur.queue))
            if not window:
                break
            if not rk.enable:
                self.cur.queue += window
                continue
            t0 = time.time()
            rows, resp, owner = [], [], []
            for i in window:
                r = self.data.rows[i]
                rows.append(i), resp.append(self.data.teacher_ids(i, r['correct_text'])), owner.append((i, 'pos'))
                for j, w in enumerate(r['wrong_texts'][:rk.max_neg]):
                    rows.append(i), resp.append(self.data.teacher_ids(i, w)), owner.append((i, j))
            proto = self._rows_proto(rows, resp)
            lp = self._wg_call(self.actor_rollout_wg.compute_log_prob, proto, chunk=rk.chunk).batch['old_log_probs']
            ml = self._mean_logp(lp, proto.batch['attention_mask'][:, -lp.shape[-1]:].float())
            pos, neg = {}, defaultdict(list)
            for (i, j), v in zip(owner, ml):
                (pos.__setitem__(i, v) if j == 'pos' else neg[i].append((v, j)))
            score = {}
            for i in window:
                best = max(neg[i])[0]
                score[i] = {'sum': pos[i] + best, 'min': min(pos[i], best), 'pos': pos[i], 'neg': best}[rk.score]
                self.cur.wrong_order[i] = [j for _, j in sorted(neg[i], reverse=True)]
            chosen = sorted(window, key=lambda i: -score[i])[:rk.top]  # top-k, in decreasing likelihood
            self.cur.queue += chosen
            sc = np.array([score[i] for i in window])
            metrics.update({'rank/window': len(window), 'rank/score_mean_window': float(sc.mean()),
                            'rank/score_mean_chosen': float(np.mean([score[i] for i in chosen])),
                            'rank/pos_logp_chosen': float(np.mean([pos[i] for i in chosen])),
                            'rank/pos_logp_window': float(np.mean([pos[i] for i in window])),
                            'rank/cursor': self.cur.cursor, 'rank/epoch': self.cur.epoch})
            timing['rank'] = timing.get('rank', 0) + time.time() - t0
            print(f'[r1x4 rank] window {len(window)} -> top {len(chosen)}, score chosen {metrics["rank/score_mean_chosen"]:.3f} '
                  f'vs window {sc.mean():.3f}, {time.time() - t0:.0f}s', flush=True)

    def _teacher_lambda(self, t, T_a):
        """p-independent teacher weight at step t (starts at 1): hold H steps at lambda0, then cosine / linear to lambda_min
        over T_a steps (or constant). Without annealing: 1 (plain R1x4)."""
        an = self.rc.anneal
        if not an.enable:
            return 1.0
        l0, lmin, H = float(an.lambda0), float(an.get('lambda_min', 0.0)), int(an.get('hold_steps', 0))
        kind = an.get('schedule', 'cosine')
        s = t - 1 - H
        if kind == 'constant' or s < 0:
            return l0
        u = min(1.0, s / T_a)
        if kind == 'linear':
            return l0 + (lmin - l0) * u
        return lmin + (l0 - lmin) * 0.5 * (1.0 + math.cos(math.pi * u))

    def _next_rows(self, batch_size, metrics, timing):
        self._refill(batch_size, metrics, timing)
        rows, self.cur.queue = self.cur.queue[:batch_size], self.cur.queue[batch_size:]
        return rows

    # ----------------------------------------------------------- CHES mask
    def _ches_keep(self, batch, group_rows, adv_row, scores_row, metrics, is_teacher):
        """Per-token CHES keep mask [rows, R] (1 = keep); only negative rows can lose tokens."""
        ch = self.rc.ches
        R = batch.batch['responses'].shape[-1]
        resp, rmask = batch.batch['responses'], batch.batch['response_mask'].bool()
        keep = torch.ones_like(resp, dtype=torch.float32)
        refs, negs = [], []
        for g, rows in enumerate(group_rows):
            pos = [r for r in rows if scores_row[r] == 1 and rmask[r].any()]
            neg = [r for r in rows if adv_row[r] < 0 and rmask[r].any()
                   and (ch.rows == 'all' or bool(is_teacher[r]) == (ch.rows == 'teacher'))]
            if pos and neg:
                refs.append(pos[0])  # teacher rows come first: the R1 trace when one is in the group
                negs += [(r, pos[0]) for r in neg]
        if not negs:
            return keep
        sel = lambda idx: batch.select(batch_keys=['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids']).select_idxs(idx)
        p1 = sel(refs)
        p1.meta_info = {'ches_mode': 'sum'}
        ssum = self._wg_call(self.actor_rollout_wg.compute_hidden_stats, p1).batch['ches_sum']
        ref_of = {r: ssum[k] for k, r in enumerate(refs)}
        p2 = sel([r for r, _ in negs])
        p2.batch['ches_ref'] = torch.stack([ref_of[p] for _, p in negs])
        p2.meta_info = {'ches_mode': 'score', 'ches_score': ch.score}
        score = self._wg_call(self.actor_rollout_wg.compute_hidden_stats, p2).batch['ches_score']
        dropped = cand_n = 0
        for k, (r, p) in enumerate(negs):
            m = rmask[r]
            # shared response prefix with y+ cancels between the + and - terms: never drop it
            L = min(int(m.sum()), int(rmask[p].sum()))
            same = (resp[r, :L] == resp[p, :L]).long().cumprod(0)
            cand = m.clone()
            cand[:L] &= ~same.bool()
            nk = int(ch.drop_frac * int(cand.sum()))
            if nk > 0:
                s = score[k].masked_fill(~cand, -float('inf'))
                top = s.topk(nk).indices
                keep[r, top] = 0
            if not (keep[r].bool() & m).any():  # never drop a whole response: keep its lowest-score token
                keep[r, score[k].masked_fill(~m, float('inf')).argmin()] = 1
            dropped += int(((keep[r] == 0) & m).sum())
            cand_n += int(m.sum())
        metrics.update({'ches/neg_rows': len(negs), 'ches/drop_frac_of_neg_tokens': dropped / max(cand_n, 1)})
        return keep

    # --------------------------------------------------------------- fit
    def fit(self):
        from verl.utils.tracking import Tracking
        logger = Tracking(project_name=self.config.trainer.project_name, experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger, config=OmegaConf.to_container(self.config, resolve=True))
        self.global_steps = 0
        self._load_checkpoint()
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True) and self.global_steps == 0:
            vm = self._validate()
            vm['val/avg_score'] = float(np.mean([v for k, v in vm.items() if k.startswith('val/test_score/')]))
            pprint(vm)
            logger.log(data=vm, step=0)
            if self.config.trainer.get('val_only', False):
                return
        B, P = self.config.data.train_batch_size, self.config.data.max_prompt_length
        dump_dir = self.config.trainer.get('rollout_data_dir', None)
        self.global_steps += 1
        while self.global_steps <= self.total_training_steps:
            metrics, timing, t_step = {}, {}, time.time()
            rows = self._next_rows(B, metrics, timing)
            if len(rows) < B:
                print('[r1x4] dataset exhausted (every prompt retired)')
                break
            # ---- students
            gen = collate_fn([{'input_ids': self.data.prompt(i)[1], 'attention_mask': self.data.prompt(i)[2],
                               'position_ids': self.data.prompt(i)[3],
                               'raw_prompt_ids': self.data.prompt(i)[1][self.data.prompt(i)[2].bool()].tolist()} for i in rows])
            gen = DataProto.from_single_dict(gen)
            gen.meta_info = {'global_steps': self.global_steps, 'do_sample': True, 'validate': False}
            t0 = time.time()
            out = self.actor_rollout_wg.generate_sequences(gen.repeat(STUDENTS, interleave=True))
            out.meta_info.pop('timing', None)
            timing['gen'] = time.time() - t0
            out.non_tensor_batch['reward_model'] = np.array([self.data.rows[i]['reward_model'] for i in rows for _ in range(STUDENTS)], dtype=object)
            t0 = time.time()
            s_scores = self.reward_fn(out, return_dict=True)['reward_tensor'].sum(-1).round().long().view(B, STUDENTS)
            # ---- teacher weight lambda_t (variant IV): p-independent schedule on the step counter; lambda = 0 -> no teacher rows
            an, isamp = self.rc.anneal, self.rc.isamp
            layout, K_app = self.rc.get('layout', 'r1x4'), int(self.rc.get('k_teacher', 2))
            t = self.global_steps
            T_a = max(1, round(an.frac * self.total_training_steps)) if an.enable else None
            lam = self._teacher_lambda(t, T_a)
            teacher_on = lam > 0
            T = (TEACHERS if layout == 'r1x4' else K_app) if teacher_on else 0
            G = STUDENTS + T
            plan, t_mu, t_kind = [], [], []
            if teacher_on:
                t_rows, t_resp = [], []
                for p, i in enumerate(rows):
                    r, k = self.data.rows[i], int(s_scores[p].sum())
                    mu = self.mu.get(r['luffy_index'], []) if isamp.enable else []
                    rng = np.random.default_rng([self.rng_seed, self.global_steps, p])
                    if layout == 'r1x4':  # k correct students -> (4-k) correct teachers + k distinct wrong traces
                        order = self.cur.wrong_order.get(i) if self.rc.rank.enable else None
                        if order is None:
                            order = list(rng.permutation(len(r['wrong_texts'])))
                        n_pos, wr = CORRECT - k, [order[m % len(order)] for m in range(k)]  # cycled if fewer distinct
                        pos_pick = list(range(n_pos))                                     # distinct mu-traces, cycled
                    else:                 # append: K positive rows, uniform over the prompt's correct traces, never p-dependent
                        n_pos, wr = K_app, []
                        pos_pick = list(rng.choice(max(len(mu), 1), K_app, replace=len(mu) < K_app))
                    for c in pos_pick:
                        t_rows.append(i), t_kind.append(1)
                        if mu:  # self-sampled correct trace with its log mu
                            ids, lps = mu[c % len(mu)]
                            t_resp.append(ids[:self.data.R]), t_mu.append(lps[:self.data.R])
                        else:   # no mu-trace for this prompt: LUFFY's R1 trace with the p/(p+0.1) shaping
                            t_resp.append(self.data.teacher_ids(i, r['correct_text'])), t_mu.append(None)
                    for j in wr:
                        t_rows.append(i), t_resp.append(self.data.teacher_ids(i, r['wrong_texts'][j])), t_kind.append(0), t_mu.append(None)
                    plan.append(len(wr))
                tb = self._rows_proto(t_rows, t_resp)
                tb.non_tensor_batch['reward_model'] = np.array([self.data.rows[i]['reward_model'] for i in t_rows], dtype=object)
                t_scores = self.reward_fn(tb, return_dict=True)['reward_tensor'].sum(-1).round().long().view(B, T)
                kinds = torch.tensor(t_kind).view(B, T)
                bad = (t_scores != kinds).any(-1)  # a stored trace graded against its label: the group trains nothing
            else:
                t_scores, bad = torch.zeros(B, 0, dtype=torch.long), torch.zeros(B, dtype=torch.bool)
            timing['reward'] = time.time() - t0
            # ---- assemble groups [T teachers, 4 students], contiguous
            sb = out.select(batch_keys=['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids'])
            sb.non_tensor_batch = {}
            idx_s = torch.arange(B * STUDENTS).view(B, STUDENTS)
            if teacher_on:
                tb.non_tensor_batch = {}
                idx_t = torch.arange(B * T).view(B, T)
                both = DataProto.concat([tb, sb])
                order_rows = torch.cat([torch.cat([idx_t[p], B * T + idx_s[p]]) for p in range(B)]).tolist()
                batch = both.select_idxs(order_rows)
            else:
                batch = sb
            R = batch.batch['responses'].shape[-1]
            scores_row = torch.cat([torch.cat([t_scores[p], s_scores[p]]) for p in range(B)]).float()
            is_teacher = torch.tensor(([1] * T + [0] * STUDENTS) * B, dtype=torch.bool)
            rmask = batch.batch['attention_mask'][:, -R:].clone()
            group_ok = (~bad).repeat_interleave(G)
            rmask[~group_ok] = 0
            batch.batch['response_mask'] = rmask
            batch.batch['prefix_mask'] = rmask.bool() & is_teacher[:, None]
            lens = batch.batch['attention_mask'][:, -R:].sum(-1)
            tls = torch.zeros(len(batch), R)
            tls[torch.arange(len(batch)), (lens - 1).clamp(min=0)] = scores_row
            batch.batch['token_level_scores'] = batch.batch['token_level_rewards'] = tls
            # baseline: R1x4 = mean of all G rows (pinned 0.5 for 4C/4W); 'student' = mean of the 4 student rows (Dr.GRPO /
            # grpo_split). After the teacher phase both are the student mean, i.e. GRPO-4.
            sg = scores_row.view(B, G)
            base = sg[:, T:].mean(-1, keepdim=True) if self.rc.get('baseline', 'group') == 'student' else sg.mean(-1, keepdim=True)
            adv_row = (sg - base).view(-1) * group_ok.float()
            if teacher_on and an.enable:  # teacher rows: x lambda_t x gate x neg weight x optional length normalisation
                gate = (s_scores.sum(-1) == 0).float() if an.gate == 'all_fail' else torch.ones(B)
                tw = lam * gate.repeat_interleave(G)
                tw = torch.where(scores_row < 0.5, tw * float(an.get('neg_weight', 1.0)), tw)
                if an.len_ref > 0:
                    tw = tw * (an.len_ref / lens.clamp(min=1).float())
                adv_row = torch.where(is_teacher, adv_row * tw, adv_row)
            batch.batch['advantages'] = adv_row[:, None] * rmask.float()
            if isamp.enable:
                tl, tm = torch.zeros(len(batch), R), torch.zeros(len(batch), R, dtype=torch.bool)
                for q, lps in enumerate(t_mu):
                    if lps is not None:
                        row = (q // T) * G + q % T
                        tl[row, :len(lps)], tm[row, :len(lps)] = lps, True
                batch.batch['teacher_logp'], batch.batch['teacher_mu_mask'] = tl, tm & rmask.bool()
            batch.non_tensor_batch['uid'] = np.array([str(i) for i in rows for _ in range(G)], dtype=object)
            batch.meta_info['global_token_num'] = batch.batch['attention_mask'].sum(-1).tolist()
            # ---- pre-update policy log-probs (p filter uses them; on-policy ratio baseline)
            t0 = time.time()
            olp = self.actor_rollout_wg.compute_log_prob(batch)
            ent = olp.batch.pop('entropys')
            batch = batch.union(olp)
            timing['old_log_prob'] = time.time() - t0
            if self.rc.ches.enable:
                t0 = time.time()
                group_rows = [list(range(g * G, (g + 1) * G)) for g in range(B)]
                batch.batch['ches_keep'] = self._ches_keep(batch, group_rows, adv_row, scores_row, metrics, is_teacher)
                timing['ches'] = time.time() - t0
            if an.enable:
                switch = lam == 0 and self._teacher_lambda(t - 1, T_a) > 0  # the step the teacher term turns off
                batch.meta_info['reset_optim'] = bool(an.reset_adam and switch)
                batch.meta_info['teacher_diag'] = bool(teacher_on and an.diag_every and (t - 1) % an.diag_every == 0)
                if not teacher_on:
                    assert not batch.batch['prefix_mask'].any()
                tr = is_teacher & group_ok
                metrics.update({'teacher/lambda': lam, 'teacher/rows': int(tr.sum()), 'teacher/T_a': T_a,
                                'teacher/neg_rows': int((tr & (scores_row < 0.5)).sum()),
                                'teacher/gate_frac': float((s_scores.sum(-1) == 0).float().mean()) if teacher_on else 0.0,
                                'teacher/len_mean': float(lens[tr].float().mean()) if tr.any() else 0.0,
                                'teacher/adam_reset': int(batch.meta_info['reset_optim'])})
            if teacher_on:  # per-token gradient weight on teacher tokens (IS rho_bar or LUFFY's 0.1p/(p+0.1)^2) and its pull
                tok_t = batch.batch['prefix_mask'] & rmask.bool()
                lp = batch.batch['old_log_probs'].float()
                pp = lp.exp()
                W = 0.1 * pp / (pp + 0.1) ** 2
                if isamp.enable:
                    tm = batch.batch['teacher_mu_mask']
                    lr = (lp - batch.batch['teacher_logp']).clamp(min=float(isamp.get('logratio_min', -20.0)))
                    W = torch.where(tm, lr.clamp(max=math.log(isamp.cbar)).exp(), W)
                    metrics['is/rows_with_mu'] = int(tm.any(-1).sum())
                    metrics['is/rows_without_mu_correct'] = int(sum(1 for q, l in enumerate(t_mu) if l is None and t_kind[q] == 1))
                    if tm.any():
                        rb = lr.clamp(max=math.log(isamp.cbar)).exp() * tm
                        v = lr[tm]
                        qv = v[torch.randperm(v.numel())[:1_000_000]]
                        ess = (rb.sum(-1) ** 2 / (tm.sum(-1).clamp(min=1) * (rb ** 2).sum(-1).clamp(min=1e-12)))[tm.any(-1)]
                        metrics.update({'is/rho_mean': float(rb.sum() / tm.sum()), 'is/frac_trunc': float((v > math.log(isamp.cbar)).float().mean()),
                                        'is/token_ess': float(ess.mean()), 'is/log_ratio_mean': float(v.mean()),
                                        'is/log_ratio_p05': float(qv.quantile(0.05)), 'is/log_ratio_p50': float(qv.quantile(0.5)),
                                        'is/log_ratio_p95': float(qv.quantile(0.95))})
                A = batch.batch['advantages'].abs()
                st = rmask.bool() & ~batch.batch['prefix_mask']
                metrics.update({'teacher/token_weight_mean': float(W[tok_t].mean()) if tok_t.any() else 0.0,
                                'teacher/pull_ratio': float((A * W * tok_t).sum() / (A * st).sum().clamp(min=1e-8)),
                                'loss/teacher_token_frac': float(tok_t.sum() / rmask.sum().clamp(min=1))})
            acc_p = s_scores.float().mean(-1)
            metrics.update({'student/solve_none_frac': float((acc_p == 0).float().mean()), 'student/solve_all_frac': float((acc_p == 1).float().mean())})
            batch.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
            t0 = time.time()
            upd = self.actor_rollout_wg.update_actor(batch)
            timing['update_actor'] = time.time() - t0
            metrics.update(reduce_metrics(upd.meta_info['metrics']))
            # ---- bookkeeping
            self.cur.observe(rows, (s_scores.sum(-1) == STUDENTS).tolist())
            sl = lens.view(B, G)[:, T:].float()
            eos_set = self.config.r1x4.get('student_eos_ids') or [self.tokenizer.eos_token_id]
            sresp = out.batch['responses']
            term = torch.zeros(len(sresp), dtype=torch.bool)
            for e in eos_set:
                term |= (sresp == e).any(-1)
            m = rmask.float()
            metrics.update({
                'train/student_acc': s_scores.float().mean().item(),
                'train/prompts_any_correct': (s_scores.sum(-1) > 0).float().mean().item(),
                'train/prompts_all_correct': (s_scores.sum(-1) == STUDENTS).float().mean().item(),
                'train/wrong_teachers_per_group': float(np.mean(plan)) if plan else 0.0,
                'train/groups_dropped_label_mismatch': int(bad.sum()),
                'train/student_len_mean': sl.mean().item(),
                'train/student_clip_rate': (sl >= R).float().mean().item(),
                'train/student_termination_rate': term.float().mean().item(),
                'train/teacher_len_mean': lens.view(B, G)[:, :T].float().mean().item() if T else 0.0,
                'train/adv_abs_mean': (batch.batch['advantages'].abs() * m).sum().item() / max(m.sum().item(), 1),
                'actor/entropy_pre_update': ((ent * m).sum() / m.sum().clamp(min=1)).item(),
                'train/student_entropy_pre_update': ((ent * m * (~is_teacher)[:, None]).sum() / (m * (~is_teacher)[:, None]).sum().clamp(min=1)).item(),
                'train/retired': len(self.cur.retired), 'train/cursor': self.cur.cursor, 'train/epoch': self.cur.epoch,
                'train/first_visit_in_batch': int(min(self.data.rows[i]['visit'] for i in rows)),
            })
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                with open(os.path.join(dump_dir, f'{self.global_steps}.jsonl'), 'w') as f:
                    texts = self.tokenizer.batch_decode(sresp, skip_special_tokens=True)
                    for p, i in enumerate(rows):
                        for s in range(STUDENTS):
                            f.write(json.dumps({'step': self.global_steps, 'luffy_index': self.data.rows[i]['luffy_index'],
                                                'visit': self.data.rows[i]['visit'], 'student': s, 'score': int(s_scores[p, s]),
                                                'len': int(sl[p, s]), 'text': texts[p * STUDENTS + s]}) + '\n')
            last = self.global_steps >= self.total_training_steps
            tf, sf = self.config.trainer.test_freq, self.config.trainer.save_freq
            if self.val_reward_fn is not None and tf > 0 and (last or self.global_steps % tf == 0):
                t0 = time.time()
                vm = self._validate()
                vm['val/avg_score'] = float(np.mean([v for k, v in vm.items() if k.startswith('val/test_score/')]))
                metrics.update(vm)
                timing['testing'] = time.time() - t0
            if sf > 0 and (last or self.global_steps % sf == 0):
                t0 = time.time()
                self._save_checkpoint()
                timing['save'] = time.time() - t0
            timing['step'] = time.time() - t_step
            metrics.update({f'timing_s/{k}': v for k, v in timing.items()})
            logger.log(data=metrics, step=self.global_steps)
            self.global_steps += 1
        print('[r1x4] training finished', flush=True)
