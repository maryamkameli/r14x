"""Entry point: python -m verl.r1x4.main_r1x4 <hydra overrides>  (run from the kit's verl/ directory; scripts/run.py builds the call)."""
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="../adaptive_mix_src/config", config_name="mix_dapo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN",
                                           **{k: v for k, v in os.environ.items() if k.startswith(("R1X4_", "VLLM_", "HF_", "CUDA_"))}}},
                 num_cpus=config.ray_init.num_cpus)
    runner = TaskRunner.remote()  # keep the handle: a temporary actor is garbage-collected before run() starts
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        from verl.adaptive_mix_src.mix_fsdp_worker import MIXActorRolloutRefWorker
        from .reward import R1x4RewardManager
        from .trainer import R1x4Trainer

        print(f"TaskRunner {socket.gethostname()} pid {os.getpid()}")
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config))
        tokenizer = hf_tokenizer(copy_to_local(config.actor_rollout_ref.model.path))
        pool = "global_pool"
        mgr = ResourcePoolManager(resource_pool_spec={pool: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
                                  mapping={Role.ActorRollout: pool})
        trainer = R1x4Trainer(config=config, tokenizer=tokenizer,
                              role_worker_mapping={Role.ActorRollout: ray.remote(MIXActorRolloutRefWorker)},
                              resource_pool_manager=mgr, ray_worker_group_cls=RayWorkerGroup,
                              reward_fn=R1x4RewardManager(tokenizer), val_reward_fn=R1x4RewardManager(tokenizer))
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
