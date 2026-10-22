"""run.py --dry_run overrides vs the worker's batch normalisation (3 and 4 GPUs) and vs the longest validation prompt."""
import os, re, subprocess, sys
import pyarrow.parquet as pq
from omegaconf import OmegaConf

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get('R1X4_ROOT', '/mnt/data/radke149/r1x4')


def overrides(*extra):
    out = subprocess.run([sys.executable, os.path.join(KIT, 'scripts', 'run.py'), '--model', 'qwen3-1.7b', '--data', 'extended',
                          '--p', '1', '--ches', 'no', '--ranking', 'no', '--is', 'no', '--anneal', 'yes', '--dry_run', *extra],
                         capture_output=True, text=True, check=True).stdout
    return dict(m.groups() for m in re.finditer(r"^\s*'?([\w.+]+)=(.*?)'?\s*\\?$", out, re.M))


# mix_fsdp_worker.py normalisation + its asserts, and mix_actor.update_policy's assert
for gpus, batch in [(3, None), (4, None), (3, 6)]:
    o = overrides('--gpus', str(gpus), *(['--batch', str(batch)] if batch else []))
    n, ngpu = int(o['actor_rollout_ref.rollout.n']), int(o['trainer.n_gpus_per_node'])
    assert ngpu == gpus
    mini = int(o['actor_rollout_ref.actor.ppo_mini_batch_size']) * n // ngpu
    micro = int(o['actor_rollout_ref.actor.ppo_micro_batch_size']) // ngpu
    assert mini > 0 and micro > 0 and mini % micro == 0, (gpus, batch, mini, micro)
    print(f'gpus {gpus} batch {batch}: mini/gpu {mini}, micro/gpu {micro} ok')

# every validation prompt must fit max_prompt_length (truncation=error, no filtering: all 2,019 problems are scored)
o = overrides('--gpus', '3')
P = int(o['data.max_prompt_length'])
assert o['data.truncation'] == 'error' and o['data.filter_overlong_prompts'] == 'False'
val = os.path.join(ROOT, 'data', 'valid.parquet')
if not os.path.exists(val):
    print(f'skip val-length check: missing {val}')
    sys.exit(0)
from transformers import AutoTokenizer
sys.path.insert(0, os.path.join(KIT, 'verl'))
from verl.utils.dataset.rl_dataset import RLHFDataset
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')
rows = pq.read_table(val).to_pylist()
lens = [len(tok.apply_chat_template(r['prompt'], add_generation_prompt=True)) for r in rows]
worst = max(range(len(lens)), key=lens.__getitem__)
print(f'valid.parquet: {len(rows)} rows, longest prompt row {worst} = {lens[worst]} tokens, P = {P}')
assert lens[worst] <= P
ds = RLHFDataset(data_files=val, tokenizer=tok, config=OmegaConf.create(
    {'prompt_key': 'prompt', 'max_prompt_length': P, 'truncation': 'error', 'filter_overlong_prompts': False, 'return_raw_chat': False}))
assert len(ds) == len(rows)
assert ds[worst]['input_ids'].shape[-1] == P
print('ok')
