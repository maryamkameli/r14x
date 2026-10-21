"""Rolling-pool prompt selection by teacher-forced likelihood (data.pool_select.*).

Each round: top the pool up to `pool` prompts with the next-in-line rows (AMPO's seed-47 visit order), teacher-force
EVERY teacher trace of every pool prompt under the CURRENT policy, score a prompt by the mean token log-likelihood of
its most likely trace, and move the `top` best prompts into the training queue (in decreasing score). The unselected
prompts stay in the pool and are re-scored next round together with the newly drawn ones. With
train_batch_size == top, every training step is one round: 1000 scored -> 100 trained -> 900 + 100 new scored -> ...
"""
import time

import numpy as np
import torch

from verl import DataProto
from .rl_dataset_with_target import collate_fn


class RollingPoolSelector:
    """Stands in for the train dataloader: iterating yields collated batches; state_dict/load_state_dict go to data.pt."""

    def __init__(self, trainer, dataset, pool, top, batch_size, chunk, shuffle, seed=47):
        assert 0 < top <= pool <= len(dataset), f'need 0 < top ({top}) <= pool ({pool}) <= dataset ({len(dataset)})'
        self.trainer, self.dataset = trainer, dataset
        self.pool_size, self.top, self.batch_size, self.chunk, self.shuffle = pool, top, batch_size, chunk, shuffle
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.epoch, self.perm_index = 0, 0
        self.perm = self._new_perm()
        self.pool = []    # dataset rows waiting to be scored, oldest first
        self.queue = []   # selected rows not yet trained, in training order
        self.age = {}     # row -> rounds it has spent in the pool unselected
        self.rounds = 0
        self._items = {}  # row -> tokenized dataset item (prompt + all teacher traces), kept while the row is pooled
        self._metrics = {}

    def _new_perm(self):
        n = len(self.dataset)
        return torch.randperm(n, generator=self.generator) if self.shuffle else torch.arange(n)

    def _next_row(self):
        if self.perm_index >= len(self.perm):
            self.perm, self.perm_index, self.epoch = self._new_perm(), 0, self.epoch + 1
        self.perm_index += 1
        return int(self.perm[self.perm_index - 1])

    def _item(self, i):
        if i not in self._items:
            self._items[i] = self.dataset[i]
        return self._items[i]

    def _score(self, rows):
        """Mean token log-likelihood of each row's most likely teacher trace under the current policy."""
        best = []
        for s in range(0, len(rows), self.chunk):
            part = rows[s:s + self.chunk]
            gen_batch = DataProto.from_single_dict(collate_fn([self._item(i) for i in part]))
            sources = gen_batch.non_tensor_batch['target_source_list']
            assert all(len(x) > 0 for x in sources), 'pool_select needs at least one teacher trace per prompt'
            ll = self.trainer._target_loglik(gen_batch, gen_batch.batch['tgt_input_ids'], sources, 1, len(part))
            best += [max(ll[k].values()) for k in range(len(part))]
        return best

    def _round(self):
        t0 = time.time()
        new = 0
        pooled = set(self.pool) | set(self.queue)
        while len(self.pool) < self.pool_size:
            i = self._next_row()
            if i not in pooled:  # after an epoch wrap a row can still be pooled or queued
                self.pool.append(i), pooled.add(i)
                new += 1
        score = self._score(self.pool)
        order = np.argsort(-np.asarray(score), kind='stable')
        chosen = [self.pool[k] for k in order[:self.top]]
        chosen_set = set(chosen)
        ages = [self.age.get(i, 0) for i in chosen]
        self.pool = [i for i in self.pool if i not in chosen_set]
        for i in self.pool:
            self.age[i] = self.age.get(i, 0) + 1
        for i in chosen:
            self.age.pop(i, None)
        self.queue += chosen
        self.rounds += 1
        sc = np.asarray(score)
        top_sc = sc[order[:self.top]]
        self._metrics = {
            'pool/rounds': self.rounds, 'pool/new_rows': new, 'pool/epoch': self.epoch,
            'pool/cursor': self.perm_index, 'pool/score_mean_pool': float(sc.mean()),
            'pool/score_mean_chosen': float(top_sc.mean()), 'pool/score_min_chosen': float(top_sc.min()),
            'pool/score_mean_rest': float(sc[order[self.top:]].mean()) if len(order) > self.top else float('nan'),
            'pool/chosen_age_mean': float(np.mean(ages)), 'pool/chosen_new_frac': float(np.mean([a == 0 for a in ages])),
            'pool/rest_age_max': max((self.age[i] for i in self.pool), default=0),
            'timing_s/pool_select': time.time() - t0,
        }
        print(f'[pool_select] round {self.rounds}: {len(sc)} scored ({new} new) -> top {len(chosen)}, '
              f'loglik chosen {top_sc.mean():.3f} (min {top_sc.min():.3f}) vs pool {sc.mean():.3f}, '
              f'{self._metrics["timing_s/pool_select"]:.0f}s', flush=True)

    def pop_metrics(self):
        m, self._metrics = self._metrics, {}
        return m

    def __iter__(self):
        while True:
            if len(self.queue) < self.batch_size:
                self._round()
            rows, self.queue = self.queue[:self.batch_size], self.queue[self.batch_size:]
            batch = collate_fn([self._items.pop(i) if i in self._items else self.dataset[i] for i in rows])
            yield batch

    def __len__(self):
        return -(-len(self.dataset) // self.batch_size)

    def state_dict(self):
        return {'perm': self.perm, 'perm_index': self.perm_index, 'epoch': self.epoch, 'generator_state': self.generator.get_state(),
                'pool': list(self.pool), 'queue': list(self.queue), 'age': dict(self.age), 'rounds': self.rounds}

    def load_state_dict(self, s):
        self.perm, self.perm_index, self.epoch = s['perm'], s['perm_index'], s['epoch']
        self.generator.set_state(s['generator_state'])
        self.pool, self.queue, self.age, self.rounds = list(s['pool']), list(s['queue']), dict(s['age']), s['rounds']
        self._items = {}
