"""Baseline training file = our run's 26,801 problems with our exact prompt column; teacher (AMPO target_lst / LUFFY
off-policy) = the original DeepSeek-R1 trace of the same problem in LUFFY's openr1.parquet (row luffy_index), repeated
N_COPIES times (default 4, like the official AMPO data's 4 teachers) so AMPO can still replace up to num_off_policy_targets rollouts."""
import os, sys, pyarrow as pa, pyarrow.parquet as pq
ours, luffy, out = sys.argv[1:4]
O = pq.read_table(ours, columns=['data_source', 'prompt', 'reward_model', 'luffy_index']).to_pylist()
L = pq.read_table(luffy, columns=['target', 'reward_model']).to_pylist()
C = int(os.environ.get('N_COPIES', 4))
rows = []
for k, o in enumerate(O):
    l = L[o['luffy_index']]
    assert l['reward_model']['ground_truth'] == o['reward_model']['ground_truth'], k
    assert len(l['target']) == 1
    rows.append(dict(data_source=o['data_source'], prompt=o['prompt'], ability='math', reward_model=o['reward_model'],
                     target_lst=[dict(content=l['target'][0]['content'], quality_score=1.0, role='assistant', source='DeepSeek-R1')] * C,
                     extra_info=dict(index=k, num_teacher_models=C, split='train', luffy_index=o['luffy_index'])))
pq.write_table(pa.Table.from_pylist(rows), out)
print(len(rows), 'rows ->', out)
