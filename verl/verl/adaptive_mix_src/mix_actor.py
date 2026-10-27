# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import math
import time
import itertools
from typing import Iterable, Tuple

import torch
import numpy as np
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.seqlen_balancing import rearrange_micro_batches
import verl.utils.torch_functional as verl_F
from verl.workers.config import ActorConfig

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

__all__ = ['MIXDataParallelPPOActor']

from verl.workers.actor.dp_actor import DataParallelPPOActor
from .logra import (LoGRAState, clean_name, exact_fp32_matmul, is_target, kl_budget, probe_kl_sum, resolve_config as resolve_logra_config,
                    step_scale, stride_select)
from .muon import MuonState, newton_schulz5, resolve_config as resolve_muon_config, update_scale

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class MIXDataParallelPPOActor(DataParallelPPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config: Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        super().__init__(config, actor_module, actor_optimizer)
        self.use_adaptive_temperature = self.config.get('use_adaptive_temperature', False)
        self.adaptive_temperature_target_entropy = self.config.get('adaptive_temperature_target_entropy', 1.0)
        if self.use_adaptive_temperature:
            self.log_alpha = torch.tensor(np.log(self.config.entropy_coeff), dtype=torch.float)
            self.log_alpha.requires_grad = True
            from torch import optim
            self.alpha_optimizer = optim.AdamW([self.log_alpha],
                                          lr=self.config.alpha_lr,
                                          betas=(0.9, 0.999),
                                          weight_decay=1e-2)
        else:
            self.alpha_optimizer = None
        self.logra_state = None  # LoGRAState, created on the first LoGRA step or restored from a checkpoint
        self.muon_state = None  # MuonState, created on the first Muon step or restored from a checkpoint

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
       
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            'prefix_mask'
        ]
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        # if self.config.use_off_policy_loss and self.config.off_policy_loss_impl == 'seq':
        #     select_keys.append('on_logprobs_mean')
        #     select_keys.append('on_logprobs_std')
        if self.config.use_off_policy_loss and self.config.use_off_policy_probs:
            select_keys.append('target_probs')
        if 'ches_keep' in data.batch.keys():  # r1x4 per-token CHES keep mask
            select_keys.append('ches_keep')
        for k in ('teacher_logp', 'teacher_mu_mask', 'teacher_gen_logp'):  # r1x4: IS (variant IV / RL-PLUS); generating-teacher log-probs
            if k in data.batch.keys():
                select_keys.append(k)
        logra_cfg = resolve_logra_config(data.meta_info.get('logra'))  # None => existing AdamW path
        muon_cfg = resolve_muon_config(data.meta_info.get('muon'))  # None => existing AdamW / LoGRA path
        assert logra_cfg is None or muon_cfg is None, 'meta_info enables both LoGRA and Muon; they are mutually exclusive'
        if muon_cfg is not None:
            assert isinstance(self.actor_module, FSDP) and self.actor_module._use_orig_params, (
                'Muon needs FSDP1 with actor_rollout_ref.actor.fsdp_config.use_orig_params=True '
                '(FSDP.summon_full_params(..., with_grads=True))')
        if logra_cfg is not None:
            for k in ('rollout_log_probs', 'rollout_log_probs_mask'):  # LoGRA mismatch subtraction (paper eq. 17)
                if k in data.batch.keys():
                    select_keys.append(k)
        if data.meta_info.get('reset_optim'):  # variant IV switch-off: fresh AdamW m, v, step from the next step()
            self.actor_optimizer.state.clear()
            print('[r1x4] AdamW state cleared at the end of the teacher phase', flush=True)
        teacher_diag = data.meta_info.get('teacher_diag', False)
            
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        if logra_cfg is not None:
            assert isinstance(self.actor_module, FSDP) and self.actor_module._use_orig_params, (
                'LoGRA needs FSDP1 with actor_rollout_ref.actor.fsdp_config.use_orig_params=True '
                '(FSDP.summon_full_params(..., with_grads=True))')
            assert self.ulysses_sequence_parallel_size == 1, 'LoGRA probe forwards need ulysses_sequence_parallel_size=1'
            metrics_logra_mismatch = self._logra_observe_mismatch(data, logra_cfg)
            logra_update_idx = 0
        
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details.
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        if logra_cfg is not None:
            append_to_dict(metrics, metrics_logra_mismatch)
        for _ in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(mini_batches):
                # split batch into micro_batches
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                if teacher_diag and batch_idx == 0:
                    append_to_dict(metrics, self._grad_diag(micro_batches, temperature))
                if logra_cfg is not None:  # predicted-KL probe, reference pass on the current (pre-update) weights
                    logra_probe = self._logra_probe_before(mini_batch, logra_cfg)
                self.actor_optimizer.zero_grad()
                if self.alpha_optimizer is not None:
                    self.alpha_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())  # actor device is cpu when using offload
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    
                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                        
                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    
                    clip_ratio = self.config.clip_ratio

                    if self.config.use_sft_multitask_loss:
                        assert self.config.use_off_policy_loss is False, 'Either use off-policy loss or sft multitask loss. You cannot set both to be True.'
                        from .mix_core_alg import compute_sft_pure_loss
                        off_policy_mask = model_inputs['prefix_mask'].any(-1) # [No]
                        off_policy_logprob = log_prob[off_policy_mask]
                        off_policy_eos_mask = response_mask[off_policy_mask]
                        
                        sft_loss = compute_sft_pure_loss(log_prob=off_policy_logprob,
                                                        eos_mask=off_policy_eos_mask)
                        
                        on_policy_mask = ~off_policy_mask
                        on_policy_logprob = log_prob[on_policy_mask]
                        on_policy_old_logprob = old_log_prob[on_policy_mask]
                        
                        # assert self.config.algorithm.adv_estimator == 'grpo_split'
                        # The on-policy advantages should not be computed together with the off-policy rewards
                        on_policy_advantages = advantages[on_policy_mask]
                        on_policy_eos_mask = response_mask[on_policy_mask]
                        
                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=on_policy_old_logprob,
                            log_prob=on_policy_logprob,
                            advantages=on_policy_advantages,
                            response_mask=on_policy_eos_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )
                        
                        pg_loss = sft_loss * self.config.sft_loss_coef + pg_loss

                    elif self.config.use_off_policy_loss:
                        from .mix_core_alg import compute_token_on_off_policy_loss, compute_token_on_seq_off_policy_loss
                        
                        if self.config.off_policy_loss_impl == 'seq':
                            loss_fn = compute_token_on_seq_off_policy_loss
                        elif self.config.off_policy_loss_impl == 'token':
                            loss_fn = compute_token_on_off_policy_loss
                        else:
                            raise ValueError(f"Invalid off-policy loss impl: {self.config.off_policy_loss_impl} should be one of ['seq', 'token']")

                        ret_dict = loss_fn(old_log_prob=old_log_prob, 
                            log_prob=log_prob,
                            advantages=advantages,
                            eos_mask=response_mask,
                            prefix_mask=model_inputs['prefix_mask'],
                            target_probs=model_inputs['target_probs'] if 'target_probs' in data else None,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            **({'token_keep': model_inputs['ches_keep']} if 'ches_keep' in model_inputs else {}),
                            **({'teacher_logp': model_inputs['teacher_logp'], 'teacher_mu_mask': model_inputs['teacher_mu_mask']}
                               if 'teacher_mu_mask' in model_inputs else {}),
                            **({'teacher_gen_logp': model_inputs['teacher_gen_logp']} if 'teacher_gen_logp' in model_inputs else {}),
                        )
                        pg_loss = ret_dict['pg_loss']
                        off_pg_loss = ret_dict['off_pg_loss']
                        on_pg_loss = ret_dict['on_pg_loss']
                        off_pg_clipfrac = ret_dict['off_pg_clipfrac']
                        pg_clipfrac = ret_dict['on_pg_clipfrac']
                        pg_clipfrac_lower = ret_dict['on_pg_clipfrac_lower']
                        ppo_kl = ret_dict['ppo_kl']
                        
                        data = {
                            'actor/on_pg_loss': on_pg_loss.detach().item(),
                            'actor/off_pg_clipfrac': off_pg_clipfrac.detach().item(),
                        }
                        if 'off_pg_loss' in ret_dict:
                            data['actor/off_pg_loss'] = off_pg_loss.detach().item()
                        for k in range(self.config.num_off_policy_targets):
                            target_key = f'off_policy_target_{k}_loss'
                            if target_key in ret_dict:
                                data[f'actor/{target_key}'] = ret_dict[target_key].detach().item()

                        if 'off_policy_prob' in ret_dict:
                            data['actor/off_policy_prob'] = ret_dict['off_policy_prob'].detach().item()
                        for k in range(self.config.num_off_policy_targets):
                            target_key = f'off_policy_target_{k}_prob'
                            if target_key in ret_dict:
                                data[f'actor/{target_key}'] = ret_dict[target_key].detach().item()
                                
                        if 'on_policy_prob' in ret_dict:
                            data['actor/on_policy_prob'] = ret_dict['on_policy_prob'].detach().item()
                        for key in ('teacher_lowp_drop_frac', 'ches_drop_frac', 'teacher_gen_lowp_drop_frac', 'neg_weight_mean'):
                            if key in ret_dict:
                                data[f'actor/{key}'] = ret_dict[key].detach().item()
                        if 'off_ratio_mean' in ret_dict:
                            data['actor/off_ratio_mean'] = ret_dict['off_ratio_mean'].detach().item()
                        if 'off_ratio_max_clip_frac' in ret_dict:
                            data['actor/off_ratio_max_clip_frac'] = ret_dict['off_ratio_max_clip_frac'].detach().item()
                        if 'off_ratio_min_clip_frac' in ret_dict:
                            data['actor/off_ratio_min_clip_frac'] = ret_dict['off_ratio_min_clip_frac'].detach().item()
                    
                        append_to_dict(metrics, data)
   
                    else:
                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )
                        
                    # compute entropy loss from entropy
                    if entropy_coeff != 0:
                        ent_mask = response_mask * (~model_inputs['prefix_mask']) if self.config.get('entropy_student_only', False) else response_mask
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=ent_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        entropy_loss = torch.tensor(0.0)
                        policy_loss = pg_loss

                    # compute policy loss
                    if self.config.use_adaptive_temperature:
                        if self.config.use_adaptive_temperature_fixed is False:
                            target_entropy = self.config.adaptive_temperature_target_entropy
                            entropy_coeff = self.log_alpha.exp()
                            if self.config.adaptive_temperature_clip > 0:
                                entropy_coeff = torch.clamp(entropy_coeff, max=self.config.adaptive_temperature_clip)
                            alpha_loss = verl_F.masked_mean(entropy - target_entropy, response_mask).detach() * entropy_coeff
                            alpha_loss = alpha_loss / self.gradient_accumulation
                            alpha_loss.backward()
                            
                            policy_loss = pg_loss - entropy_loss * entropy_coeff.detach().item()
                            micro_batch_metrics['actor/alpha_loss'] = alpha_loss.detach().item()
                            micro_batch_metrics['actor/entropy_coeff'] = entropy_coeff.detach().item()
                            micro_batch_metrics['actor/log_alpha'] = self.log_alpha.detach().item()
                        else: # fixed strategy for entropy coeff
                            target_entropy = self.config.adaptive_temperature_target_entropy
                            # cur_entropy = verl_F.masked_mean(entropy, response_mask)
                            entropy_coeff = (target_entropy / entropy_loss).detach().item() * self.config.entropy_coeff
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                            micro_batch_metrics['actor/entropy_coeff'] = entropy_coeff
                    else:
                        policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs['ref_log_prob']
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics['actor/kl_loss'] = kl_loss.detach().item()
                        micro_batch_metrics['actor/kl_coef'] = self.config.kl_loss_coef
                        
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()
                    
                    micro_batch_metrics.update(
                        {
                            'actor/entropy_loss': entropy_loss.detach().item(),
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                if muon_cfg is not None:
                    grad_norm, muon_metrics = self._muon_optimizer_step(muon_cfg)
                    append_to_dict(metrics, muon_metrics)
                elif logra_cfg is None:
                    grad_norm = self._optimizer_step()
                else:
                    grad_norm, logra_metrics = self._logra_optimizer_step(logra_cfg, logra_probe, temperature, logra_update_idx)
                    logra_update_idx += 1
                    append_to_dict(metrics, logra_metrics)
                mini_batch_metrics = {'actor/grad_norm': grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad()
        return metrics
    
    def _grad_diag(self, micro_batches, temperature):
        """r1x4 variant IV: pg-loss gradient of the student rows (g_S) vs the teacher rows (g_T), entropy excluded.
        Both use the real loss with the other rows' advantages zeroed (the loss is linear in A). Logs ||g_T||/||g_S||, cos."""
        from .mix_core_alg import compute_token_on_off_policy_loss as loss_fn
        params = [p for p in self.actor_module.parameters() if p.requires_grad]
        saved = {}
        for part in ('S', 'T'):
            self.actor_optimizer.zero_grad()
            for mb in micro_batches:
                mb = mb.to(get_device_id())
                mi = {**mb.batch}
                row_t = mi['prefix_mask'].any(-1, keepdim=True)
                adv = mi['advantages'] * (row_t if part == 'T' else ~row_t)
                _, lp = self._forward_micro_batch(mi, temperature=temperature, calculate_entropy=False)
                ret = loss_fn(old_log_prob=mi['old_log_probs'], log_prob=lp, advantages=adv, eos_mask=mi['response_mask'],
                              prefix_mask=mi['prefix_mask'], config=self.config, token_keep=mi.get('ches_keep'),
                              teacher_logp=mi.get('teacher_logp'), teacher_mu_mask=mi.get('teacher_mu_mask'),
                              teacher_gen_logp=mi.get('teacher_gen_logp'))
                scale = mi['response_mask'].shape[0] / self.config.ppo_mini_batch_size if self.config.use_dynamic_bsz else 1 / self.gradient_accumulation
                (ret['pg_loss'] * scale).backward()
            saved[part] = [p.grad.detach().float().clone() if p.grad is not None else None for p in params]
        self.actor_optimizer.zero_grad()
        sums = torch.zeros(3, device=get_device_id())
        for gs, gt in zip(saved['S'], saved['T']):
            if gs is not None and gt is not None:
                sums += torch.stack([(gs * gs).sum(), (gt * gt).sum(), (gs * gt).sum()])
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums)
        ss, tt, st = sums.tolist()
        if not all(np.isfinite([ss, tt, st])) or ss <= 0 or tt <= 0:
            return {}
        return {'diag/grad_ratio': (tt / ss) ** 0.5, 'diag/cos': st / (ss * tt) ** 0.5}

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.step()
        return grad_norm

    # ------------------------------------------------------------------ Muon (verl/adaptive_mix_src/muon.py)
    def muon_state_dict(self):
        return None if self.muon_state is None else self.muon_state.state_dict()

    def load_muon_state_dict(self, sd):
        self.muon_state = MuonState()
        self.muon_state.load_state_dict(sd)

    def _muon_optimizer_step(self, cfg):
        """Muon step on the target projection matrices + the actor AdamW on every other parameter:
        clip -> (summoned full params/grads, every rank redundantly) M <- mu M + G, U = G + mu M (nesterov) | M,
        O = NS5(U), W <- W (1 - lr wd) - lr scale O -> target grads set to None on the sharded views ->
        AdamW.step() (skips grad-None params, incl. weight decay) -> zero grads.
        lr / wd = the AdamW's param_groups[0] (scheduled lr). Non-finite grad norm => no update at all."""
        assert self.config.grad_clip is not None
        t0 = time.time()
        module = self.actor_module
        grad_norm = module.clip_grad_norm_(max_norm=self.config.grad_clip)
        if not torch.isfinite(grad_norm):  # skip Muon and AdamW, keep both states
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}; Muon/AdamW step skipped")
            self.actor_optimizer.zero_grad()
            if self.alpha_optimizer is not None:
                self.alpha_optimizer.step()
            return grad_norm, {'muon/skipped': 1.0, 'muon/time_s': time.time() - t0}
        if self.muon_state is None:
            self.muon_state = MuonState()
        state = self.muon_state
        group = self.actor_optimizer.param_groups[0]
        lr = float(group['lr'])
        wd = float(group.get('weight_decay', 0.0) or 0.0)
        decay = 1.0 - lr * wd
        # ||lr scale O||^2, ||dW||^2, ||W||^2 (pre-update), ||M||^2 over targets
        sums = torch.zeros(4, dtype=torch.float64, device=get_device_id())
        numel, n_no_grad = 0, 0
        targets = []  # same nn.Parameter objects as the sharded views outside the summon (use_orig_params)
        with FSDP.summon_full_params(module, writeback=True, with_grads=True), exact_fp32_matmul(), torch.no_grad():
            for name, p in module.named_parameters():
                cname = clean_name(name)
                if not is_target(cname, p, cfg['targets']):
                    continue
                targets.append(p)
                if p.grad is None:
                    n_no_grad += 1
                    continue
                G = p.grad.to(torch.float32)
                U = state.direction(cfg, cname, G)
                M = state.momentum[cname]
                O = newton_schulz5(U, cfg['ns_steps'])
                del U, G
                b = lr * update_scale(p.shape, cfg['scale_mode'])
                W = p.data
                W32 = W if W.dtype == torch.float32 else W.to(torch.float32)
                wsq, wo, osq = W32.square().sum(), (W32 * O).sum(), O.square().sum()
                a = lr * wd  # dW = -(a W + b O)
                sums += torch.stack([osq.double() * (b * b),
                                     wsq.double() * (a * a) + wo.double() * (2.0 * a * b) + osq.double() * (b * b),
                                     wsq.double(), M.square().sum().double()])
                numel += O.numel()
                if W32 is W:
                    W.mul_(decay).add_(O, alpha=-b)
                else:
                    W.copy_(W32.mul_(decay).add_(O, alpha=-b))
                del O, W32, M
        for p in targets:  # outside the summon: sharded grad views; AdamW then skips the targets entirely
            p.grad = None
        n_mat = len(targets) - n_no_grad
        del targets
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.step()
        state.num_updates += 1
        usq, dsq, wsq, msq = sums.tolist()
        return grad_norm, {
            'muon/num_matrices': float(n_mat),
            'muon/num_no_grad': float(n_no_grad),
            'muon/lr': lr,
            'muon/update_rms': math.sqrt(max(usq, 0.0) / numel) if numel > 0 else 0.0,
            'muon/update_rel_norm': math.sqrt(max(dsq, 0.0)) / math.sqrt(wsq) if wsq > 0 else 0.0,
            'muon/momentum_norm': math.sqrt(max(msq, 0.0)),
            'muon/skipped': 0.0,
            'muon/time_s': time.time() - t0,
        }

    # ------------------------------------------------------------------ LoGRA (verl/adaptive_mix_src/logra.py)
    def logra_state_dict(self):
        return None if self.logra_state is None else self.logra_state.state_dict()

    def load_logra_state_dict(self, sd):
        self.logra_state = LoGRAState()
        self.logra_state.load_state_dict(sd)

    @staticmethod
    def _logra_all_reduce(vals, op=None):
        t = torch.tensor(vals, dtype=torch.float64, device=get_device_id())
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(t, op=op if op is not None else torch.distributed.ReduceOp.SUM)
        return t.tolist()

    @torch.no_grad()
    def _logra_observe_mismatch(self, data, cfg):
        """Rollout-vs-trainer mismatch kappa (paper App. D.1): mean of rollout_logp - old_log_prob over valid rollout
        tokens (rollout_log_probs_mask; fallback: student rows) of completed (non-truncated) responses, clamped at 0,
        EMA-smoothed. Collective on every rank; no valid token anywhere => no measurement, kappa unchanged."""
        if not cfg['mismatch_subtraction']:
            return {}
        b = data.batch
        dsum, cnt = 0.0, 0.0
        if 'rollout_log_probs' in b.keys():
            resp = b['response_mask'].bool()
            mask = resp & ~resp[:, -1:]  # last response slot used => truncated row
            if 'rollout_log_probs_mask' in b.keys():
                mask = mask & b['rollout_log_probs_mask'].bool()
            else:
                mask = mask & ~b['prefix_mask'].any(-1, keepdim=True)
            delta = torch.where(mask, b['rollout_log_probs'].double() - b['old_log_probs'].double(), 0.0)
            dsum, cnt = float(delta.sum()), float(mask.sum())
        dsum, cnt = self._logra_all_reduce([dsum, cnt])
        if self.logra_state is None:
            self.logra_state = LoGRAState()
        raw = self.logra_state.observe_mismatch(dsum, cnt, float(cfg['mismatch_ema']))
        ema = self.logra_state.mismatch_ema
        return {'logra/mismatch_raw': raw, 'logra/mismatch_tokens': cnt,
                'logra/mismatch_ema': float('nan') if ema is None else ema}

    @torch.no_grad()
    def _logra_probe_hidden(self, rows):
        """Final-norm hidden states (lm_head input) at the probe positions, one unpadded row per forward
        (same recipe as MIXActorRolloutRefWorker.compute_hidden_stats). Collective: same row count on every rank."""
        module = self.actor_module
        norm = next(m for n, m in module.named_modules() if clean_name(n).endswith('model.norm'))
        box = {}
        hook = norm.register_forward_hook(lambda m, i, o: box.__setitem__('h', o))
        was_training = module.training
        module.eval()
        out = []
        try:
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                for ids, hpos in rows:
                    ids = ids.to(get_device_id())[None]
                    pos = torch.arange(ids.shape[1], device=ids.device)[None]
                    module(input_ids=ids, position_ids=pos, use_cache=False, logits_to_keep=1, return_dict=True)
                    out.append(box.pop('h')[0, hpos.to(ids.device)].float())
        finally:
            hook.remove()
            module.train(was_training)
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1 and module._handle is not None:
            module._handle.reshard(True)  # FSDP1 root stays unsharded after a no-grad forward
        return torch.cat(out)

    @torch.no_grad()
    def _logra_probe_before(self, mini_batch, cfg):
        """Pick up to probe_rows rows of this rank's mini-batch (student rows first, then teacher rows; rows need a
        response token), up to probe_tokens evenly strided response positions each, and record their hidden states
        on the current weights. Returns None on every rank if any rank has no usable row."""
        b = mini_batch.batch
        R = b['responses'].shape[-1]
        resp_mask = b['response_mask'].bool()
        has = resp_mask.any(-1)
        student = ~b['prefix_mask'].any(-1)
        order = torch.cat([torch.nonzero(has & student).flatten(), torch.nonzero(has & ~student).flatten()])
        order = order[:max(cfg['probe_rows'], 0)]
        n = int(self._logra_all_reduce([float(order.numel())], op=torch.distributed.ReduceOp.MIN)[0])
        if n == 0:
            return None
        rows = []
        for i in order[:n].tolist():
            ids = b['input_ids'][i][b['attention_mask'][i].bool()]
            n_tok, n_resp = ids.numel(), int(b['attention_mask'][i, -R:].sum())
            j = torch.nonzero(resp_mask[i, :n_resp]).flatten()
            j = j[stride_select(j.numel(), cfg['probe_tokens'])]
            hpos = n_tok - n_resp - 1 + j  # hidden state at hpos predicts response token j
            hpos = hpos[hpos >= 0]
            rows.append((ids[:int(hpos.max()) + 1], hpos))  # later tokens cannot affect these positions (causal)
        return {'rows': rows, 'h0': self._logra_probe_hidden(rows)}

    def _logra_optimizer_step(self, cfg, probe, temperature, update_idx):
        """LoGRA step replacing AdamW.step() (paper Alg. 1 + sec. 2.3, finite-difference KL probe):
        clip -> S = G A^T, U = RowAdam/SGD(S), D = -eta U A on every target -> W += D (trial, alpha = 1) ->
        q = mean KL(pi_before || pi_after) on the probe positions -> alpha = min(alpha_max, sqrt(delta_eff / q)) ->
        W += (alpha - 1) D. Every rank computes on the full (summoned) matrices; writeback updates its own shard.
        Non-target parameters are never updated; grads are zeroed afterwards."""
        assert self.config.grad_clip is not None
        t0 = time.time()
        module = self.actor_module
        grad_norm = module.clip_grad_norm_(max_norm=self.config.grad_clip)
        if not torch.isfinite(grad_norm):  # skip the update, keep the LoGRA state
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}; LoGRA step skipped")
            self.actor_optimizer.zero_grad()
            if self.alpha_optimizer is not None:
                self.alpha_optimizer.step()
            return grad_norm, {'logra/skipped': 1.0, 'logra/alpha': 0.0, 'logra/time_s': time.time() - t0}
        if self.logra_state is None:
            self.logra_state = LoGRAState()
        state = self.logra_state
        dev = get_device_id()
        updates = []  # (clean name, U [d, r], A [r, k], pending RowAdam state)
        sums = torch.zeros(3, dtype=torch.float64, device=dev)  # ||S||^2, ||U A||^2, ||W||^2 over targets
        n_no_grad = 0
        w_head = None
        with FSDP.summon_full_params(module, writeback=True, with_grads=True), exact_fp32_matmul(), torch.no_grad():
            if probe is not None:  # tied embedding / lm_head (not a target, unchanged by the step), fp32 copy
                w_head = module.module.get_output_embeddings().weight.detach().to(torch.float32).clone()
            targets = []
            for name, p in module.named_parameters():
                cname = clean_name(name)
                if not is_target(cname, p, cfg['targets']):
                    continue
                if p.grad is None:
                    n_no_grad += 1
                    continue
                A = state.projection(cfg, cname, p.shape[1], update_idx, p.device)
                S = p.grad.to(torch.float32) @ A.T
                U, pending = state.direction(cfg, cname, S)
                sums += torch.stack([S.square().sum(), ((U @ (A @ A.T)) * U).sum(),
                                     p.detach().to(torch.float32).square().sum()]).double()
                del S
                updates.append((cname, U, A, pending))
                targets.append(p)
            sketch_sq, usq, wsq = sums.tolist()
            if cfg['lr'] is not None:  # paper eq. 4 literally: D = -lr U A
                eta = float(cfg['lr'])
            else:  # reference update_mode='target': ||D||_F = target_norm * ||W_targets||_F
                eta = float(cfg['target_norm']) * math.sqrt(wsq) / math.sqrt(usq) if usq > 0 else 0.0
            for p, (_, U, A, _) in zip(targets, updates):  # trial step, alpha = 1
                p.data.add_(U @ A, alpha=-eta)
            del targets
        self.actor_optimizer.zero_grad()

        q, n_pos = float('nan'), 0
        if probe is not None:
            h1 = self._logra_probe_hidden(probe['rows'])
            kl_sum = float(probe_kl_sum(probe['h0'], h1, w_head, float(temperature)))
            kl_sum, n_pos = self._logra_all_reduce([kl_sum, float(h1.shape[0])])
            q = kl_sum / n_pos if n_pos > 0 else float('nan')
            del h1
        del w_head
        scheduled = kl_budget(cfg)
        budget = state.effective_budget(cfg, scheduled)
        if probe is None or not n_pos > 0:
            alpha, status = 1.0, 'no_probe'  # reference step_scale: no prediction -> unscaled step
        else:
            alpha, status = step_scale(q, budget, float(cfg['alpha_max']))
        if status != 'ok' and torch.distributed.get_rank() == 0:
            print(f"[LoGRA] step {cfg['step']}: {status} (q={q:.3e}, probe tokens={int(n_pos)}) -> alpha={alpha:.3f}", flush=True)
        if alpha != 1.0:
            with FSDP.summon_full_params(module, writeback=True, with_grads=False), exact_fp32_matmul(), torch.no_grad():
                params = {clean_name(n): p for n, p in module.named_parameters()}
                for cname, U, A, _ in updates:
                    params[cname].data.add_(U @ A, alpha=-eta * (alpha - 1.0))
                del params
        if alpha > 0:
            for cname, _, _, pending in updates:
                state.commit(cname, pending)
            state.num_updates += 1
        n_mat = len(updates)
        del updates
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.step()
        return grad_norm, {
            'logra/alpha': alpha,
            'logra/q_pred_kl': q,
            'logra/kl_budget': scheduled,
            'logra/kl_budget_eff': budget,
            'logra/eta': eta,
            'logra/update_norm': abs(alpha) * eta * math.sqrt(max(usq, 0.0)),
            'logra/update_rel_norm': abs(alpha) * eta * math.sqrt(max(usq, 0.0)) / math.sqrt(wsq) if wsq > 0 else 0.0,
            'logra/sketch_norm': math.sqrt(max(sketch_sq, 0.0)),
            'logra/num_matrices': float(n_mat),
            'logra/num_no_grad': float(n_no_grad),
            'logra/probe_tokens': float(n_pos),
            'logra/q_zero': float(status == 'q_zero'),
            'logra/q_nonfinite': float(status == 'q_nonfinite'),
            'logra/skipped': 0.0,
            'logra/time_s': time.time() - t0,
        }
