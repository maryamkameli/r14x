"""Training/validation grader = reward v4 of our earlier runs: math_verify(parse('$gold$'), parse(whole response)).

math_verify can hang in C code where its signal timeout never fires (it froze validation once), so every text is
graded in a worker process and a worker that exceeds `timeout` seconds on one text is killed and replaced; that text
scores -1 (ungraded; the reward manager counts it as 0).
"""
import multiprocessing as mp
import os
import queue
import time

import torch


def grade_one(text, gold):
    from math_verify import parse, verify
    try:
        g = parse('$' + gold + '$', parsing_timeout=5)
        return int(bool(verify(g, parse(text, parsing_timeout=5), timeout_seconds=5)))
    except Exception:
        return 0


def _worker(tasks, results):
    while True:
        item = tasks.get()
        if item is None:
            return
        k, text, gold = item
        results.put(('start', k, os.getpid()))
        results.put(('done', k, grade_one(text, gold)))


def grade_texts(texts, golds, workers=8, timeout=30.0):
    """1 correct, 0 wrong, -1 ungraded (timeout). Order preserved."""
    n = len(texts)
    out = [-1] * n
    if n == 0:
        return out
    ctx = mp.get_context('fork')
    tasks, results = ctx.Queue(), ctx.Queue()
    for k in range(n):
        tasks.put((k, texts[k], golds[k]))
    procs = {}

    def spawn():
        p = ctx.Process(target=_worker, args=(tasks, results), daemon=True)
        p.start()
        procs[p.pid] = [p, None, 0.0]  # process, current task, start time

    for _ in range(min(workers, n)):
        spawn()
    finished = set()

    def finish(k, v):
        if k not in finished:  # a killed worker's late 'done' must not count twice
            finished.add(k)
            out[k] = v

    while len(finished) < n:
        try:
            msg = results.get(timeout=1.0)
            if msg[0] == 'start':
                if msg[2] in procs:
                    procs[msg[2]][1:] = [msg[1], time.time()]
            else:
                finish(msg[1], msg[2])
                for v in procs.values():
                    if v[1] == msg[1]:
                        v[1] = None
        except queue.Empty:
            pass
        now = time.time()
        for pid, (p, k, t0) in list(procs.items()):
            if k is not None and now - t0 > timeout:  # hung: kill, leave -1, replace the worker
                p.kill()
                p.join()
                del procs[pid]
                finish(k, -1)
                spawn()
            elif not p.is_alive() and k is not None:  # died mid-task
                del procs[pid]
                finish(k, -1)
                spawn()
    for _ in procs:
        tasks.put(None)
    for p, _, _ in procs.values():
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    return out


class R1x4RewardManager:
    """Drop-in for AMPO's RewardManager(data, return_dict) used by training and _validate."""

    def __init__(self, tokenizer, workers=None, timeout=30.0):
        self.tokenizer = tokenizer
        self.workers = workers or max(4, min(32, (len(os.sched_getaffinity(0)) or 8) - 2))
        self.timeout = timeout

    def decode(self, data):
        P = data.batch['prompts'].shape[-1]
        lens = data.batch['attention_mask'][:, P:].sum(-1).tolist()
        return [self.tokenizer.decode(r[:l], skip_special_tokens=True) for r, l in zip(data.batch['responses'], lens)], lens

    def __call__(self, data, return_dict=False):
        texts, lens = self.decode(data)
        golds = [x['ground_truth'] for x in data.non_tensor_batch['reward_model']]
        scores = grade_texts(texts, golds, self.workers, self.timeout)
        rt = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        for i, (s, l) in enumerate(zip(scores, lens)):
            rt[i, max(l, 1) - 1] = float(s == 1)
        res = {'reward_tensor': rt, 'reward_extra_info': {'ungraded': [int(s == -1) for s in scores]}}
        return res if return_dict else rt
