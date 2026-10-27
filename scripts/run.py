#!/usr/bin/env python3
"""Launch one R1x4 run.

  python scripts/run.py --model qwen3-1.7b --data core --p 1 --ches yes --ranking yes      # variant I
  python scripts/run.py --model qwen3-1.7b --data core --p 1 --ches no  --ranking yes      # variant II
  python scripts/run.py --model qwen3-4b   --data core --p 1 --ches no  --ranking no       # variant III
  python scripts/run.py ... --dry_run      # print the command only

--data   luffy21k | core | extended | r1qwen | qwen3   (or a path to a parquet built by data/build_datasets.py)
--p      percent: teacher tokens whose pre-update policy probability is below p% lose their gradient (0 = off)
--ches   yes | no: per-token CHES on negative rows (--ches_frac of each negative response dropped, default 50%)
--ranking yes | no: top --rank_top (100) of the next --rank_pool (1000) prompts in LUFFY order by teacher-forced
         likelihood of their pos+neg traces, trained in decreasing likelihood; otherwise plain LUFFY visit order
Everything else is the R1x4 recipe (4 students + 4 teachers, exactly 4C/4W, lr 1e-6, no KL, entropy 1e-3, T=1.0,
8,192-token responses, 48 prompts/step for Qwen3-1.7B and 40 for Qwen3-4B (--batch), p/(p+0.1) off-policy reshape, no clip).
--hw     rtx6000 | a40 | a100_40 | a100_80: GPU type; adjusts only memory knobs (vLLM share, token budgets, optimizer
         offload, attention default), never the recipe or the experiment name (a checkpoint resumes on another GPU type
         with the same GPU count)
"""
import argparse, os, shlex, subprocess, sys, time

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {  # path, rollout TP, vLLM memory share, optimizer offload, token budget per GPU micro-batch, student EOS ids
    'qwen3-0.6b': dict(path='Qwen/Qwen3-0.6B', tp=1, mem=0.45, offload=False, tok=32768, eos=[151645, 151643], tag='QWEN3-0.6B', mu='qwen3-4b', batch=96),
    'qwen3-1.7b': dict(path='Qwen/Qwen3-1.7B', tp=1, mem=0.45, offload=False, tok=32768, eos=[151645, 151643], tag='QWEN3-1.7B', mu='qwen3-4b', batch=48),
    'qwen3-4b': dict(path='Qwen/Qwen3-4B', tp=1, mem=0.45, offload=False, tok=24576, eos=[151645, 151643], tag='QWEN3-4B', mu='qwen3-32b', batch=40),
    'qwen2.5-math-7b': dict(path='Elliott/Qwen2.5-Math-7B-16k-think', tp=1, mem=0.50, offload=True, tok=18432, eos=[151643],
                            tag='QWEN2.5-MATH-7B', mu='luffy-qwen-math-7b-zero', batch=32),
}
# --hw: per-GPU-type memory knobs layered over MODELS (rtx6000 = the MODELS values, tuned on the 96 GB RTX PRO 6000).
# mem = vLLM gpu_memory_utilization, tok = actor ppo_max_token_len_per_gpu (log-prob budget stays 2*tok), offload = FSDP
# optimizer offload. A40 is planned with 40 GB usable (of 48). tok must stay >= max_prompt+max_response = 9728 (dynamic
# batching packs at least one full sequence per micro-batch). Key None = models not in MODELS.
GB40 = {'qwen3-0.6b': dict(mem=0.5, tok=16384), 'qwen3-1.7b': dict(mem=0.5, tok=12288),
        'qwen3-4b': dict(mem=0.45, tok=12288, offload=True), 'qwen2.5-math-7b': dict(mem=0.5, tok=9728, offload=True),
        None: dict(tok=9728, offload=True)}
HW = {'rtx6000': {}, 'a40': GB40, 'a100_40': GB40,
      'a100_80': {'qwen3-4b': dict(tok=16384), 'qwen2.5-math-7b': dict(tok=12288)}}
yes = lambda s: {'yes': True, 'y': True, 'true': True, '1': True, 'no': False, 'n': False, 'false': False, '0': False}[s.lower()]

ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--data', required=True)
ap.add_argument('--p', type=float, required=True, help='percent, e.g. 1 for p < 1%%; 0 disables the filter')
ap.add_argument('--ches', type=yes, required=True)
ap.add_argument('--ranking', type=yes, required=True)
ap.add_argument('--model', default='qwen3-1.7b', help=f'{"|".join(MODELS)} or an HF path/local dir')
ap.add_argument('--steps', type=int, default=500)
ap.add_argument('--gpus', type=int, default=4)
ap.add_argument('--hw', default='rtx6000', choices=list(HW), help='GPU type: memory knobs only (see HW); not part of the name')
ap.add_argument('--root', default=os.environ.get('R1X4_ROOT', '/mnt/data/radke149/r1x4'), help='data/, ckpt/, logs/ live here')
ap.add_argument('--val', default=None, help='validation parquet (default <root>/data/valid.parquet)')
ap.add_argument('--test_freq', type=int, default=10)
ap.add_argument('--save_freq', type=int, default=25)
ap.add_argument('--keep_ckpt', type=int, default=1, help='checkpoints kept on disk (/mnt/data is small)')
ap.add_argument('--ckpt_contents', default='model,optimizer,extra', help="'model,extra' halves the 8B checkpoint (no optimizer)")
ap.add_argument('--ches_frac', type=float, default=0.5)
ap.add_argument('--ches_score', default='cos', choices=['cos', 'ip'])
ap.add_argument('--ches_rows', default='all', choices=['all', 'teacher', 'student'])
ap.add_argument('--p_rows', default='all', choices=['all', 'pos', 'neg'])
ap.add_argument('--p_teacher', type=float, default=0,
                help='percent: wrong-teacher tokens with p < this under the GENERATING teacher are dropped too (needs '
                     'wrong_teacher_logps in --data); 0 disables')
ap.add_argument('--neg_one_minus_p', type=yes, default=False,
                help='weight every token of a negative row (teacher or student) by 1 - sg(p_old)')
ap.add_argument('--is', dest='isamp', type=yes, default=False, help='variant IV: truncated per-token IS on self-sampled correct teacher traces')
ap.add_argument('--anneal', type=yes, default=False, help='variant IV: teacher weight cosine-annealed to 0 at T_a, then pure GRPO-4')
ap.add_argument('--anneal_frac', type=float, default=0.2)
ap.add_argument('--layout', default='r1x4', choices=['r1x4', 'append'], help='r1x4: 4 students + 4 teachers, exactly 4C/4W; append: 4 students + --k_teacher positive teacher rows')
ap.add_argument('--k_teacher', type=int, default=2, help='teacher rows per prompt in the append layout')
ap.add_argument('--baseline', default='group', choices=['group', 'student'], help='group: mean of all rows (R1x4, pinned 0.5); student: mean of the 4 students (Dr.GRPO)')
ap.add_argument('--schedule', default='cosine', choices=['cosine', 'linear', 'constant'])
ap.add_argument('--hold_steps', type=int, default=0)
ap.add_argument('--lambda_min', type=float, default=0.0)
ap.add_argument('--neg_weight', type=float, default=1.0, help='multiplier for wrong teacher rows (0 drops them)')
ap.add_argument('--entropy', type=float, default=0.001, help='entropy bonus coefficient')
ap.add_argument('--entropy_student_only', type=yes, default=False)
ap.add_argument('--is_logratio_min', type=float, default=-20.0)
ap.add_argument('--lambda0', type=float, default=1.0)
ap.add_argument('--gate', default='always', choices=['always', 'all_fail'])
ap.add_argument('--len_ref', type=float, default=0.0)
ap.add_argument('--no_reset_adam', action='store_true')
ap.add_argument('--diag_every', type=int, default=10)
ap.add_argument('--is_cbar', type=float, default=1.0)
ap.add_argument('--mu_pool', default=None, help='default <root>/data/mu/mu_<teacher>_<data>*.parquet')
ap.add_argument('--rank_pool', type=int, default=1000)
ap.add_argument('--rank_top', type=int, default=100)
ap.add_argument('--rank_score', default='sum', choices=['sum', 'min', 'pos', 'neg', 'gold'],
                help='gold: mean probability of the gold answer after the correct teacher trace (AMPO-style)')
ap.add_argument('--rank_rolling', type=yes, default=False, help='rolling pool: unpicked prompts stay; pool topped up to --rank_pool each round')
ap.add_argument('--rlplus', type=yes, default=False, help='RL-PLUS ratio pi/(0.75 pi_old + 0.25/V) on teacher tokens, clipped at --rlplus_cbar')
ap.add_argument('--rlplus_warmup', type=int, default=10, help='steps with ratio pi/pi_old (no IS) before RL-PLUS starts')
ap.add_argument('--rlplus_cbar', type=float, default=1.0)
ap.add_argument('--mc_skip', type=yes, default=False, help='skip prompts at the solved fixed point, checked with --mc_samples extra rollouts')
ap.add_argument('--mc_samples', type=int, default=4)
ap.add_argument('--mc_z', type=float, default=1.0)
ap.add_argument('--logra', type=yes, default=False, help='LoGRA low-rank sketched updates + predicted-KL step control instead of AdamW')
ap.add_argument('--logra_rank', type=int, default=256)
ap.add_argument('--logra_refresh', type=yes, default=True)
ap.add_argument('--logra_optimizer', default='rowadam', choices=['rowadam', 'sgd'])
ap.add_argument('--logra_lr', type=float, default=None, help='nominal step eta (default: implementation default)')
ap.add_argument('--logra_kl_start', type=float, default=2e-4)
ap.add_argument('--logra_kl_end', type=float, default=2e-5)
ap.add_argument('--logra_alpha_max', type=float, default=None)
ap.add_argument('--muon', type=yes, default=False,
                help='Muon (Newton-Schulz orthogonalised momentum, Moonlight RMS scaling, reuses --lr / weight decay) on the '
                     'attention/MLP projection matrices, AdamW on embeddings and norms; exclusive with --logra')
ap.add_argument('--muon_momentum', type=float, default=0.95)
ap.add_argument('--muon_nesterov', type=yes, default=True)
ap.add_argument('--muon_ns_steps', type=int, default=5)
ap.add_argument('--muon_scale', default='moonlight', choices=['moonlight', 'none'])
ap.add_argument('--batch', type=int, default=None, help='prompts per step (default: 96 for qwen3-0.6b, 48 for qwen3-1.7b, 40 for qwen3-4b)')
ap.add_argument('--lr', type=float, default=1e-6)
ap.add_argument('--name', default=None, help='experiment name (default: descriptive, from the knobs)')
ap.add_argument('--resume', default='auto', choices=['auto', 'disable'])
ap.add_argument('--no_val_before_train', action='store_true')
ap.add_argument('--dump_rollouts', action='store_true')
ap.add_argument('--attn', default=None, choices=['flash_attention_2', 'sdpa'],
                help='default: $R1X4_ATTN_IMPL or flash_attention_2 with --hw rtx6000, flash_attention_2 on A40/A100')
ap.add_argument('--logger', default='["console","tensorboard"]')
ap.add_argument('--dry_run', action='store_true')
ap.add_argument('overrides', nargs='*', help='extra Hydra overrides, e.g. actor_rollout_ref.rollout.gpu_memory_utilization=0.4')
a = ap.parse_args()

if a.attn is None:
    a.attn = os.environ.get('R1X4_ATTN_IMPL', 'flash_attention_2') if a.hw == 'rtx6000' else 'flash_attention_2'
m = MODELS.get(a.model.lower(), dict(path=a.model, tp=1, mem=0.5, offload=True, tok=18432, eos=None, tag=os.path.basename(a.model).upper(), batch=32))
m = {**m, **HW[a.hw].get(a.model.lower() if a.model.lower() in MODELS else None, {})}
data = a.data if a.data.endswith('.parquet') else os.path.join(a.root, 'data', f'{a.data}.parquet')
val = a.val or os.path.join(a.root, 'data', 'valid.parquet')
for f in (data, val):
    if not os.path.exists(f) and not a.dry_run:
        sys.exit(f'missing {f} (build it with data/build_datasets.py, see README)')
if a.ches and a.attn == 'sdpa':
    pass  # compute_hidden_stats runs unpadded single rows, so it works with either backend
dset = os.path.basename(data).replace('.parquet', '').upper()
name = a.name or '-'.join(['R1X4', dset, m['tag'], f'P{a.p:g}PCT' if a.p > 0 else 'NOP', 'TOKCHES' if a.ches else 'NOCHES',
                           (f'{"GOLD" if a.rank_score == "gold" else ""}{"ROLL" if a.rank_rolling else ""}RANK{a.rank_top}OF{a.rank_pool}'
                            if a.ranking else 'LUFFYORDER')]
                          + (['ISTRUNC'] if a.isamp else []) + ([f'ANNEAL{a.anneal_frac:g}'] if a.anneal else [])
                          + ([f'APPEND{a.k_teacher}POS'] if a.layout == 'append' else []) + (['STUDENTBASE'] if a.baseline == 'student' else [])
                          + ([a.schedule.upper()] if a.anneal and a.schedule != 'cosine' else []) + ([f'HOLD{a.hold_steps}'] if a.hold_steps else [])
                          + ([f'NEGW{a.neg_weight:g}'] if a.neg_weight != 1 else []) + ([f'PT{a.p_teacher:g}PCT'] if a.p_teacher > 0 else []) + (['NEG1MP'] if a.neg_one_minus_p else []) + ([f'ENT{a.entropy:g}'] if a.entropy != 0.001 else [])
                          + (['ENTSTUDENTONLY'] if a.entropy_student_only else []) + ([f'LENREF{a.len_ref:g}'] if a.len_ref > 0 else [])
                          + (['GATEALLFAIL'] if a.gate == 'all_fail' else []) + ([f'LAMBDA0{a.lambda0:g}'] if a.lambda0 != 1 else [])
                          + ([f'RLPLUS{a.rlplus_warmup}'] if a.rlplus else []) + ([f'MCSKIP{a.mc_samples}'] if a.mc_skip else [])
                          + ([f'LOGRA{a.logra_rank}'] if a.logra else []) + (['MUON'] if a.muon else []))
assert not (a.logra and a.muon), '--logra and --muon are mutually exclusive'
ckpt, logs = os.path.join(a.root, 'ckpt', name), os.path.join(a.root, 'logs')
os.makedirs(logs, exist_ok=True)
R, P, B = 8192, 1536, a.batch or m['batch']  # P: the longest valid.parquet prompt (olympiad_bench row 1076) is 1407 tokens; truncation=error
assert (B * 4) % a.gpus == 0, f'--batch {B}: 4*batch rollouts must split over {a.gpus} GPUs'
assert a.layout == 'r1x4' or (B * (4 + a.k_teacher)) % a.gpus == 0, f'--batch {B} x (4+{a.k_teacher}) rows must split over {a.gpus} GPUs'
assert a.layout == 'r1x4' or a.anneal, '--layout append is the IS+annealing recipe: use it with --anneal yes'
rmpad = a.attn == 'flash_attention_2'

ov = [
    f'data.train_files={data}', f'data.val_files={val}', 'data.prompt_key=prompt', f'data.max_prompt_length={P}',
    f'data.max_response_length={R}', f'data.train_batch_size={B}', 'data.truncation=error', 'data.filter_overlong_prompts=False',
    'data.return_raw_chat=False', 'data.dataloader_num_workers=0',
    # model / FSDP
    f'actor_rollout_ref.model.path={m["path"]}', f'actor_rollout_ref.model.use_remove_padding={rmpad}',
    'actor_rollout_ref.model.enable_gradient_checkpointing=True',
    'actor_rollout_ref.actor.strategy=fsdp', 'actor_rollout_ref.actor.fsdp_config.param_offload=False',
    f'actor_rollout_ref.actor.fsdp_config.optimizer_offload={m["offload"]}',
    'actor_rollout_ref.actor.ulysses_sequence_parallel_size=1',
    # R1x4 objective (= train_4c4w.sbatch on the LUFFY tree)
    f'actor_rollout_ref.actor.optim.lr={a.lr}', 'actor_rollout_ref.actor.optim.lr_warmup_steps=0',
    'actor_rollout_ref.actor.optim.weight_decay=0.01', 'actor_rollout_ref.actor.grad_clip=1.0',
    f'actor_rollout_ref.actor.ppo_mini_batch_size={B * 8 // 4}',  # x rollout.n(4) = all 8 rows of every group: ONE update per step
    # use_dynamic_bsz splits by tokens; the micro size only feeds divisibility asserts after //n_gpus, so 1 per GPU divides any mini-batch
    f'actor_rollout_ref.actor.ppo_micro_batch_size={a.gpus}',
    'actor_rollout_ref.actor.ppo_epochs=1', 'actor_rollout_ref.actor.use_dynamic_bsz=True',
    f'actor_rollout_ref.actor.ppo_max_token_len_per_gpu={m["tok"]}',
    'actor_rollout_ref.actor.use_kl_loss=False', 'actor_rollout_ref.actor.kl_loss_coef=0.0', 'algorithm.kl_ctrl.kl_coef=0.0',
    'algorithm.use_kl_in_reward=False', 'algorithm.adv_estimator=grpo', 'algorithm.norm_adv_by_std_in_grpo=False',
    f'actor_rollout_ref.actor.entropy_coeff={a.entropy}', f'actor_rollout_ref.actor.entropy_student_only={a.entropy_student_only}',
    f'actor_rollout_ref.actor.is_logratio_min={a.is_logratio_min}', f'r1x4.isamp.logratio_min={a.is_logratio_min}',
    'actor_rollout_ref.actor.use_off_policy_loss=True', 'actor_rollout_ref.actor.off_policy_loss_impl=token',
    'actor_rollout_ref.actor.off_policy_normalize=False', 'actor_rollout_ref.actor.off_policy_reshape=p_div_p_0.1',
    'actor_rollout_ref.actor.loss_remove_token_mean=True', 'actor_rollout_ref.actor.loss_remove_clip=True',
    'actor_rollout_ref.actor.loss_row_mean_fix=True',
    f'actor_rollout_ref.actor.teacher_token_min_prob={a.p / 100.0}', f'actor_rollout_ref.actor.teacher_token_min_prob_rows={a.p_rows}',
    f'actor_rollout_ref.actor.teacher_token_min_teacher_prob={a.p_teacher / 100.0}',
    f'actor_rollout_ref.actor.neg_one_minus_p={a.neg_one_minus_p}',
    f'actor_rollout_ref.actor.checkpoint.save_contents=[{a.ckpt_contents}]',
    'actor_rollout_ref.ref.use_ref=False',
    # rollout
    'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.n=4', 'actor_rollout_ref.rollout.temperature=1.0',
    'actor_rollout_ref.rollout.top_p=1.0', 'actor_rollout_ref.rollout.top_k=-1',
    f'actor_rollout_ref.rollout.tensor_model_parallel_size={m["tp"]}', f'actor_rollout_ref.rollout.gpu_memory_utilization={m["mem"]}',
    'actor_rollout_ref.rollout.enable_chunked_prefill=True', f'actor_rollout_ref.rollout.max_num_batched_tokens={P + R}',
    'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True', f'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={2 * m["tok"]}',
    'actor_rollout_ref.rollout.val_kwargs.temperature=0.6', 'actor_rollout_ref.rollout.val_kwargs.top_p=1.0',
    'actor_rollout_ref.rollout.val_kwargs.do_sample=True', 'actor_rollout_ref.rollout.val_kwargs.n=1',
    'actor_rollout_ref.rollout.n_prefix=0',
    'reward_model.reward_manager=naive', 'reward_model.enable=False',
    # r1x4
    f'r1x4.ches.enable={a.ches}', f'r1x4.ches.drop_frac={a.ches_frac}', f'r1x4.ches.score={a.ches_score}', f'r1x4.ches.rows={a.ches_rows}',
    f'r1x4.rank.enable={a.ranking}', f'r1x4.rank.pool={a.rank_pool}', f'r1x4.rank.top={a.rank_top}', f'r1x4.rank.score={a.rank_score}',
    f'r1x4.anneal.enable={a.anneal}', f'r1x4.anneal.frac={a.anneal_frac}', f'r1x4.anneal.lambda0={a.lambda0}',
    f'r1x4.anneal.gate={a.gate}', f'r1x4.anneal.len_ref={a.len_ref}', f'r1x4.anneal.reset_adam={not a.no_reset_adam}',
    f'r1x4.anneal.diag_every={a.diag_every}', f'r1x4.anneal.schedule={a.schedule}', f'r1x4.anneal.hold_steps={a.hold_steps}',
    f'r1x4.anneal.lambda_min={a.lambda_min}', f'r1x4.anneal.neg_weight={a.neg_weight}',
    f'r1x4.layout={a.layout}', f'r1x4.k_teacher={a.k_teacher}', f'r1x4.baseline={a.baseline}', f'r1x4.isamp.enable={a.isamp}',
    # the truncated-IS loss clips at actor.is_cbar: the RL-PLUS clip when --rlplus, else variant IV's c-bar
    f'actor_rollout_ref.actor.is_cbar={a.rlplus_cbar if a.rlplus else a.is_cbar}', f'r1x4.isamp.cbar={a.is_cbar}',
    f'r1x4.rank.rolling={a.rank_rolling}',
    f'r1x4.rlplus.enable={a.rlplus}', f'r1x4.rlplus.warmup_steps={a.rlplus_warmup}', f'r1x4.rlplus.cbar={a.rlplus_cbar}',
    f'r1x4.mc_skip.enable={a.mc_skip}', f'r1x4.mc_skip.samples={a.mc_samples}', f'r1x4.mc_skip.z={a.mc_z}',
    f'r1x4.logra.enable={a.logra}', f'r1x4.logra.rank={a.logra_rank}', f'r1x4.logra.refresh={a.logra_refresh}',
    f'r1x4.logra.optimizer={a.logra_optimizer}', f'r1x4.logra.kl_budget_start={a.logra_kl_start}', f'r1x4.logra.kl_budget_end={a.logra_kl_end}',
    f'r1x4.logra.lr={"null" if a.logra_lr is None else a.logra_lr}',
    f'r1x4.logra.alpha_max={"null" if a.logra_alpha_max is None else a.logra_alpha_max}',
    f'r1x4.muon.enable={a.muon}', f'r1x4.muon.momentum={a.muon_momentum}', f'r1x4.muon.nesterov={a.muon_nesterov}',
    f'r1x4.muon.ns_steps={a.muon_ns_steps}', f'r1x4.muon.scale_mode={a.muon_scale}',
] + ([  # LoGRA / Muon update full matrices under summon_full_params(with_grads=True)
    '++actor_rollout_ref.actor.fsdp_config.use_orig_params=True',
] if a.logra or a.muon else []) + ([  # LoGRA mismatch subtraction needs the students' vLLM log-probs
    '++actor_rollout_ref.rollout.calculate_log_probs=True',
] if a.logra else []) + [
    "r1x4.isamp.mu_pool='" + (a.mu_pool or os.path.join(a.root, 'data', 'mu', 'mu_' + m.get('mu', 'teacher') + '_' + os.path.basename(data).replace('.parquet', '') + '*.parquet')) + "'",
    'r1x4.retire_after_all_correct=2', f'r1x4.student_eos_ids={m["eos"]}'.replace(' ', '') if m['eos'] else 'r1x4.student_eos_ids=null',
    # trainer
    f'trainer.project_name=r1x4', f'trainer.experiment_name={name}', f'trainer.logger={a.logger}',
    f'trainer.n_gpus_per_node={a.gpus}', 'trainer.nnodes=1', f'trainer.total_training_steps={a.steps}', 'trainer.total_epochs=1000',
    f'trainer.test_freq={a.test_freq}', f'trainer.save_freq={a.save_freq}', f'trainer.default_local_dir={ckpt}',
    f'trainer.resume_mode={a.resume}', f'+trainer.max_actor_ckpt_to_keep={a.keep_ckpt}',
    f'trainer.val_before_train={not a.no_val_before_train}', 'trainer.balance_batch=False', 'trainer.critic_warmup=0',
    'critic.enable=False',
    f'hydra.run.dir={os.path.join(a.root, "hydra", name, time.strftime("%m%d_%H%M%S"))}',
] + ([f'+trainer.rollout_data_dir={os.path.join(a.root, "rollouts", name)}'] if a.dump_rollouts else []) + a.overrides

env = dict(os.environ, R1X4_ATTN_IMPL=a.attn, HYDRA_FULL_ERROR='1', VLLM_USE_V1=os.environ.get('VLLM_USE_V1', '1'),
           TENSORBOARD_DIR=os.path.join(a.root, 'tensorboard', name), PYTHONUNBUFFERED='1')
env.pop('ROCR_VISIBLE_DEVICES', None)  # set by Slurm; verl refuses to start when it is set with CUDA_VISIBLE_DEVICES
cmd = [sys.executable, '-m', 'verl.r1x4.main_r1x4'] + ov
log = os.path.join(logs, f'{name}_{time.strftime("%m%d_%H%M")}.log')
print(f'# {name}\n# log: {log}\ncd {os.path.join(KIT, "verl")} && \\\n  ' + ' \\\n  '.join(shlex.quote(c) for c in cmd))
if a.dry_run:
    sys.exit(0)
with open(log, 'w') as f:
    f.write(' '.join(shlex.quote(c) for c in cmd) + '\n')
    f.flush()
    rc = subprocess.call(cmd, cwd=os.path.join(KIT, 'verl'), env=env, stdout=f, stderr=subprocess.STDOUT)
print(f'exit {rc}; log {log}')
sys.exit(rc)
