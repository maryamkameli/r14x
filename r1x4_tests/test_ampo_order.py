"""Prompt ORDER of the AMPO trainer (verl.adaptive_mix_src) vs. upstream AMPO (EnigmaYYYY/AMPO, exp_scripts/train_ampo.sh).

Upstream AMPO (and LUFFY, same class) visits prompts with ResumableRandomSampler: torch.Generator seeded 47,
epoch e = the e-th torch.randperm(n) drawn from that one generator, batches of train_batch_size, drop_last=True.
So the reference order of an UNINTERRUPTED run is fully determined by (n, batch size):

    g = torch.Generator(); g.manual_seed(47); perms = [torch.randperm(n, generator=g) for _ in range(epochs)]

These tests build the train dataloader exactly like MIXRayDAPOTrainer._create_dataloader (same sampler class,
StatefulDataLoader, drop_last, num_workers), drive it with the trainer's loop (`for epoch ...: for batch in loader`),
and check (1) an uninterrupted run, (2) a run resumed from a checkpoint (state_dict -> new process -> load_state_dict),
including resumes after the first epoch boundary. CPU only, no model / tokenizer needed.

    cd verl && python -m pytest -q ../r1x4_tests/test_ampo_order.py
"""
import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.adaptive_mix_src.rl_dataset_with_target import ResumableRandomSampler

N, B = 8491, 128          # SII-Enigma/OpenR1-Math-4LongCoTs-8.5k rows, train_ampo.sh train_prompt_bsz
STEPS_PER_EPOCH = N // B  # drop_last=True -> 66


class _Idx(torch.utils.data.Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        return i


def reference(steps, n=N, b=B):
    g = torch.Generator()
    g.manual_seed(47)
    out, perm, pos = [], None, None
    while len(out) < steps:
        perm, pos = torch.randperm(n, generator=g), 0
        while pos + b <= n and len(out) < steps:
            out.append(perm[pos:pos + b].tolist())
            pos += b
    return out


def make_loader(num_workers):
    ds = _Idx()
    return StatefulDataLoader(dataset=ds, batch_size=B, num_workers=num_workers, drop_last=True,
                              collate_fn=lambda x: list(x), sampler=ResumableRandomSampler(data_source=ds))


def trainer_loop(loader, steps, stop_after=None, total_epochs=30):
    """MIXRayDAPOTrainer.fit: `for epoch in range(total_epochs): for batch_dict in self.train_dataloader:`."""
    got = []
    for _ in range(total_epochs):
        for batch in loader:
            got.append(list(batch))
            if stop_after is not None and len(got) == stop_after:
                return got, loader.state_dict()   # _save_checkpoint at the end of this step -> data.pt
            if len(got) == steps:
                return got, None
    return got, None


@pytest.mark.parametrize('num_workers', [0, 8])
def test_uninterrupted_matches_ampo(num_workers):
    steps = 3 * STEPS_PER_EPOCH + 5
    got, _ = trainer_loop(make_loader(num_workers), steps)
    assert got == reference(steps)


@pytest.mark.parametrize('num_workers', [0, 8])
@pytest.mark.parametrize('stop', [10, STEPS_PER_EPOCH, STEPS_PER_EPOCH + 7, 2 * STEPS_PER_EPOCH + 30])
def test_resume_matches_uninterrupted(num_workers, stop):
    steps = 3 * STEPS_PER_EPOCH + 5
    first, state = trainer_loop(make_loader(num_workers), steps, stop_after=stop)
    fresh = make_loader(num_workers)          # new job: new sampler (seed 47 again), new loader
    fresh.load_state_dict(state)              # RayPPOTrainer._load_checkpoint
    rest, _ = trainer_loop(fresh, steps - stop)
    ref = reference(steps)
    assert first == ref[:stop]
    bad = next((k for k, (a, r) in enumerate(zip(rest, ref[stop:])) if a != r), None)
    assert bad is None, f'resumed after step {stop}: step {stop + bad + 1} differs from the uninterrupted AMPO order'
