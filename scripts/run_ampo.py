#!/usr/bin/env python3
"""Launch AMPO (exp_scripts/train_ampo.sh of EnigmaYYYY/AMPO) on this kit's fork.

  python scripts/run_ampo.py --model qwen3-4b --selection loglik      # the fork: per-update selection over ALL teacher traces
  python scripts/run_ampo.py --model qwen3-4b --selection static      # original AMPO behaviour (for the control run)

--selection
  loglik  (fork) no static preselection: every teacher trace of the prompt is kept (max_available_targets=-1, no
          quality_score sort) and, right before each update, all of them are teacher-forced under the CURRENT policy;
          the most likely ones fill the injected slots (one batched forward per step)
  prob    every trace kept, ranked per update by AMPO's own score (p(gold | teacher reasoning) - p(gold | prompt))
  static  the original: traces pre-sorted by the dataset's quality_score and truncated to 4, then AMPO's 'prob' ranking
--pool N --top K  (fork) rolling prompt pool instead of AMPO's shuffled batches: teacher-force every trace of the next N
          prompts in line under the CURRENT policy, train on the K whose best trace is most likely (K prompts per step),
          keep the N-K unselected in the pool, top it up with the next K prompts and repeat (needs --selection loglik)
Everything else follows train_ampo.sh (128 prompts x 8 rollouts, adaptive k, seq-level off-policy loss, lr 1e-6, ...);
with --pool the batch is K prompts and the mini-batch K/2, keeping train_ampo.sh's two updates per step.
--fix_loss turns on loss_row_mean_fix: with use_dynamic_bsz=True and loss_remove_token_mean=True (both set by
train_ampo.sh) AMPO weights each row by the number of rows that share its micro-batch; the fix makes it 1/(L*mini).
"""
import argparse, os, shlex, subprocess, sys, time

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {'qwen3-0.6b': ('Qwen/Qwen3-0.6B', 1, None), 'qwen3-1.7b': ('Qwen/Qwen3-1.7B', 1, None), 'qwen3-4b': ('Qwen/Qwen3-4B', 1, None),
          'qwen2.5-7b-instruct': ('Qwen/Qwen2.5-7B-Instruct', 2, 32768), 'qwen2.5-1.5b-instruct': ('Qwen/Qwen2.5-1.5B-Instruct', 1, 32768)}
ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--model', default='qwen3-4b')
ap.add_argument('--selection', default='loglik', choices=['loglik', 'prob', 'static'])
ap.add_argument('--root', default=os.environ.get('R1X4_ROOT', '/mnt/data/radke149/r1x4'))
ap.add_argument('--train', default=None, help='default <root>/data/ampo/openr1_math_4longcots.parquet')
ap.add_argument('--val', default=None, help='default <root>/data/ampo/valid.parquet (AMPO prompt format)')
ap.add_argument('--steps', type=int, default=510)
ap.add_argument('--gpus', type=int, default=4)
ap.add_argument('--pool', type=int, default=0, help='rolling prompt pool size (0 = off: AMPO shuffled batches)')
ap.add_argument('--top', type=int, default=100, help='prompts selected from the pool per step (= train batch)')
ap.add_argument('--save_freq', type=int, default=50)
ap.add_argument('--fix_loss', action='store_true')
ap.add_argument('--offload', default='auto', choices=['auto', 'yes', 'no'])
ap.add_argument('--attn', default=os.environ.get('R1X4_ATTN_IMPL', 'flash_attention_2'), choices=['flash_attention_2', 'sdpa'])
ap.add_argument('--name', default=None)
ap.add_argument('--keep_ckpt', type=int, default=1)
ap.add_argument('--dry_run', action='store_true')
ap.add_argument('overrides', nargs='*')
a = ap.parse_args()
path, tp, max_pos = MODELS.get(a.model.lower(), (a.model, 2, None))
train = a.train or os.path.join(a.root, 'data', 'ampo', 'openr1_math_4longcots.parquet')
val = a.val or os.path.join(a.root, 'data', 'ampo', 'valid.parquet')
name = a.name or (f'AMPO-{"PERUPDATE-LOGLIK-ALLTRACES" if a.selection == "loglik" else "PERUPDATE-PROB-ALLTRACES" if a.selection == "prob" else "ORIGINAL"}-{a.model.upper()}'
                  + (f'-POOL{a.pool}TOP{a.top}' if a.pool else '') + ('-LOSSFIX' if a.fix_loss else ''))
offload = (a.offload == 'yes') or (a.offload == 'auto' and not any(s in a.model.lower() for s in ('0.6b', '1.7b', '1.5b')))
if a.pool and a.selection != 'loglik':
    sys.exit("--pool needs --selection loglik ('prob'/'static' precompute AMPO's scoreA over the shuffled dataloader)")
B = a.top if a.pool else 128
sp = 2 if a.attn == 'flash_attention_2' and a.gpus % 2 == 0 else 1  # Ulysses SP must divide the GPU count
sel = {'loglik': ('none', -1, 'loglik'), 'prob': ('none', -1, 'prob'), 'static': ('short', 4, 'prob')}[a.selection]
P, R = 1024, 8192
ov = [
    'algorithm.adv_estimator=grpo', f'data.train_files={train}', f'data.val_files={val}', 'data.prompt_key=prompt',
    'data.truncation=right', f'data.max_prompt_length={P}', f'data.max_response_length={R}', f'data.train_batch_size={B}',
    'data.val_batch_size=512', 'data.shuffle=True', 'data.num_off_policy_targets=2', f'data.max_available_targets={sel[1]}',
    f'data.pool_select.enable={bool(a.pool)}', f'data.pool_select.pool={a.pool or 1000}', f'data.pool_select.top={a.top}',
    f'data.length_strategy={sel[0]}', 'data.reward_impl_version=5', 'data.val_reward_impl_version=5',
    'actor_rollout_ref.rollout.n=8', 'actor_rollout_ref.rollout.pass_num=4', 'actor_rollout_ref.rollout.use_adaptive_k=True',
    'algorithm.use_kl_in_reward=False', 'algorithm.kl_ctrl.kl_coef=0.0', 'algorithm.norm_adv_by_std_in_grpo=False',
    'algorithm.filter_groups.enable=False', 'algorithm.filter_groups.metric=seq_reward', 'algorithm.filter_groups.max_num_gen_batches=10',
    'actor_rollout_ref.actor.use_kl_loss=False', 'actor_rollout_ref.actor.kl_loss_coef=0.0',
    'actor_rollout_ref.actor.clip_ratio_low=0.2', 'actor_rollout_ref.actor.clip_ratio_high=0.2', 'actor_rollout_ref.actor.clip_ratio_c=10.0',
    f'actor_rollout_ref.model.use_remove_padding={a.attn == "flash_attention_2"}',
    'actor_rollout_ref.actor.use_dynamic_bsz=True', 'actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True',
    'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True', f'actor_rollout_ref.actor.ppo_max_token_len_per_gpu={P + R}',
    f'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={P + R}', f'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={P + R}',
    'actor_rollout_ref.rollout.name=vllm', f'actor_rollout_ref.model.path={path}', 'actor_rollout_ref.model.enable_gradient_checkpointing=True',
    'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.actor.optim.lr_warmup_steps=0', 'actor_rollout_ref.actor.optim.weight_decay=0.1',
    f'actor_rollout_ref.actor.ppo_mini_batch_size={B // 2}',
    # use_dynamic_bsz splits by tokens; the micro size only feeds the divisibility asserts (yaml default 64 does not divide K/2*8)
    f'actor_rollout_ref.actor.ppo_micro_batch_size={a.gpus}',
    f'actor_rollout_ref.actor.fsdp_config.param_offload={offload}', f'actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}',
    'actor_rollout_ref.actor.entropy_coeff=0.001', 'actor_rollout_ref.actor.grad_clip=1.0', 'actor_rollout_ref.actor.loss_agg_mode=token-mean',
    f'actor_rollout_ref.actor.ulysses_sequence_parallel_size={sp}',
    'actor_rollout_ref.rollout.gpu_memory_utilization=0.45', f'actor_rollout_ref.rollout.tensor_model_parallel_size={tp}',
    'actor_rollout_ref.rollout.enable_chunked_prefill=True', f'actor_rollout_ref.rollout.max_num_batched_tokens={P + R}',
    'actor_rollout_ref.rollout.temperature=1.0', 'actor_rollout_ref.rollout.top_p=1.0', 'actor_rollout_ref.rollout.top_k=-1',
    'actor_rollout_ref.rollout.val_kwargs.temperature=0.6', 'actor_rollout_ref.rollout.val_kwargs.top_p=1.0',
    'actor_rollout_ref.rollout.val_kwargs.top_k=-1', 'actor_rollout_ref.rollout.val_kwargs.do_sample=True', 'actor_rollout_ref.rollout.val_kwargs.n=1',
    f'actor_rollout_ref.ref.fsdp_config.param_offload={offload}', 'actor_rollout_ref.ref.ulysses_sequence_parallel_size=1',
    'actor_rollout_ref.actor.fsdp_config.fsdp_size=-1', f'actor_rollout_ref.rollout.max_prefix_len={R}',
    'actor_rollout_ref.rollout.prefix_share_across_samples=False', 'actor_rollout_ref.rollout.prefix_strategy=random',
    'actor_rollout_ref.rollout.min_prefix_ratio=1.0', 'actor_rollout_ref.rollout.max_prefix_ratio=1.0',
    'actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0', 'actor_rollout_ref.ref.use_ref=False',
    'actor_rollout_ref.actor.use_sft_multitask_loss=False', 'actor_rollout_ref.actor.use_off_policy_loss=True',
    'actor_rollout_ref.actor.off_policy_normalize=False', 'actor_rollout_ref.actor.off_policy_reshape=p_div_p_0.1',
    'actor_rollout_ref.actor.off_policy_loss_impl=seq', 'actor_rollout_ref.actor.loss_remove_token_mean=True',
    'actor_rollout_ref.actor.loss_remove_clip=False', f'actor_rollout_ref.actor.loss_row_mean_fix={a.fix_loss}',
    'actor_rollout_ref.rollout.injection_strategy=adaptive', 'actor_rollout_ref.rollout.target_selection_strategy=best_k',
    'reward_model.reward_manager=naive', '+reward_model.reward_kwargs.overlong_buffer_cfg.enable=False',
    f'+reward_model.reward_kwargs.overlong_buffer_cfg.len=4096', '+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0',
    '+reward_model.reward_kwargs.overlong_buffer_cfg.log=False', f'+reward_model.reward_kwargs.max_resp_len={R}',
    'reward_model.reward_manager_shaping_function_name=threshold_0', 'reward_model.compute_score_name=mean_exp_log_softmax',
    'reward_model.repetition_penalty=True', f'reward_model.off_policy_reward_manager={sel[2]}', 'reward_model.val_reward_manager=naive',
    'reward_model.format_mode=R1_nothink', 'reward_model.format_coefficient=0.1',
    'trainer.logger=["console","tensorboard"]', 'trainer.project_name=AMPO', f'trainer.experiment_name={name}',
    f'trainer.n_gpus_per_node={a.gpus}', 'trainer.nnodes=1', 'trainer.val_before_train=False', 'trainer.test_freq=10',
    f'trainer.save_freq={a.save_freq}', 'trainer.total_epochs=30', f'trainer.total_training_steps={a.steps}', f'+trainer.max_actor_ckpt_to_keep={a.keep_ckpt}',
    f'trainer.default_local_dir={os.path.join(a.root, "ckpt", name)}', 'trainer.default_hdfs_dir=null', 'trainer.resume_mode=auto',
    'trainer.log_val_generations=10',
    f'hydra.run.dir={os.path.join(a.root, "hydra", name, time.strftime("%m%d_%H%M%S"))}',
] + ([f'+actor_rollout_ref.model.override_config.max_position_embeddings={max_pos}'] if max_pos else []) + a.overrides
os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
log = os.path.join(a.root, 'logs', f'{name}_{time.strftime("%m%d_%H%M")}.log')
cmd = [sys.executable, '-m', 'verl.adaptive_mix_src.main_mix_dapo'] + ov
print(f'# {name}\n# log: {log}\ncd {os.path.join(KIT, "verl")} && \\\n  ' + ' \\\n  '.join(shlex.quote(c) for c in cmd))
if a.dry_run:
    sys.exit(0)
for f in (train, val):
    if not os.path.exists(f):
        sys.exit(f'missing {f}: copy data/ampo/ to <root>/data/ampo/ (README section 3)')
env = dict(os.environ, R1X4_ATTN_IMPL=a.attn, HYDRA_FULL_ERROR='1', TENSORBOARD_DIR=os.path.join(a.root, 'tensorboard', name))
env.pop('ROCR_VISIBLE_DEVICES', None)  # set by Slurm; verl refuses to start when it is set with CUDA_VISIBLE_DEVICES
# the conda env has verl installed editable from ONE checkout; driver and Ray workers must import THIS checkout's verl
env['PYTHONPATH'] = os.pathsep.join(filter(None, [os.path.join(KIT, 'verl'), os.environ.get('PYTHONPATH')]))
with open(log, 'w') as f:
    rc = subprocess.call(cmd, cwd=os.path.join(KIT, 'verl'), env=env, stdout=f, stderr=subprocess.STDOUT)
print(f'exit {rc}; log {log}')
sys.exit(rc)
