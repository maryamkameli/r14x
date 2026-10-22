# Running the kit on MSI (agate)

The RTX-box files (`setup_rtx.sh`, `rtx_env.sh`, `overnight_agent.sh`, `queue_overnight.txt`, `AGENT_TASK.md`,
`BEFORE_SLEEP.md`) are for cs-u-ding6000pro only. On MSI use these instead:

| file | what |
|---|---|
| `requirements.txt` | pinned MSI stack: python 3.10, vLLM 0.8.5, torch 2.6.0, transformers 4.51.3, hydra 1.3.2, math-verify 0.8.0 |
| `scripts/msi_env.sh` | `source` it: sets `R1X4_ROOT`, `HF_HOME`, caches under `/scratch.global/$USER`, activates the env |
| `scripts/setup_msi.sh` / `setup_msi.sbatch` | builds the conda env at `/scratch.global/$USER/envs/r1x4` (CPU job) |
| `scripts/gpu_check_msi.sbatch` | 1-GPU check: torch, flash-attn (falls back to the SDPA shim), vLLM on Qwen3-1.7B, `r1x4_tests` |
| `scripts/*_msi.sbatch` | smoke / dataset build jobs (now path-independent; submit from the kit root) |

All defaults can be overridden before sourcing `msi_env.sh` (`R1X4_ROOT`, `R1X4_ENV`, `HF_HOME`, `R1X4_CACHE`).

## 1. Environment (once)
```bash
cd /scratch.global/$USER/r14x
J=$(sbatch --parsable scripts/setup_msi.sbatch)                  # ~20-40 min, log: r1x4_setup_<id>.out
sbatch --dependency=afterok:$J scripts/gpu_check_msi.sbatch      # log: r1x4_gpucheck_<id>.out -> GPU_CHECK_DONE
```
Do not pip install on a login node: login nodes kill processes after 15 CPU-minutes.

## 2. Data (NOT in git)
`.gitignore` excludes `*.parquet`, `data/mu/`, `data/ampo/`. The run scripts expect:
```
$R1X4_ROOT/data/{core,extended,luffy21k,r1qwen,qwen3,valid}.parquet     # R1x4 (run.py)
$R1X4_ROOT/data/ampo/{openr1_math_4longcots,valid}.parquet              # AMPO fork (run_ampo.py)
$R1X4_ROOT/data/mu/mu_qwen3-4b_core*.parquet                            # variant IV only
$R1X4_ROOT/smoke/data/{smoke,valid}.parquet                             # smoke sbatch files
```
Either copy the built parquets from the collaborator, or rebuild them with `sbatch scripts/build_msi.sbatch`. The rebuild
needs `data/src/` = `openr1.parquet`, `visit_order_rows.txt`, `luffy21k_r1x4.parquet`, `usable_negatives.jsonl`,
`negatives_extended.jsonl`. `valid.parquet` (2,019 problems) is not produced by `build_datasets.py`. A smoke set is just the
first rows of core: `python -c "import pandas as p; p.read_parquet('core.parquet').head(64).to_parquet('smoke.parquet')"`.

### Data already downloaded on MSI (pinned to the HF commit, so the row order is fixed)
| file under `$R1X4_ROOT/data/` | source | rows |
|---|---|---|
| `src/openr1.parquet` | `Elliott/Openr1-Math-46k-8192` @ `bd025079` (LUFFY's train set, 1 R1 trace/prompt), sha256 `33c2c51a…57bf84` | 45,792 |
| `ampo/openr1_math_4longcots.parquet` | `SII-Enigma/OpenR1-Math-4LongCoTs-8.5k` @ `2328db27` (= AMPO `data/prepare_train.py`) | 8,491 |
| `ampo/valid.parquet`, `valid.parquet` | AMPO repo `data/valid.parquet` (AIME24, AMC, MATH-500, Minerva, OlympiadBench) | 2,019 |

Still missing for R1x4 (`run.py`): `data/src/visit_order_rows.txt`, `luffy21k_r1x4.parquet`, `usable_negatives.jsonl`,
`negatives_extended.jsonl` (the stored wrong traces), so `build_datasets.py` cannot run yet.

### Prompt order = upstream AMPO
AMPO (and LUFFY, same class) visits prompts with `ResumableRandomSampler`: one `torch.Generator` seeded 47, epoch *e*
= the *e*-th `torch.randperm(n)` from it, `train_batch_size` prompts per step, `drop_last=True`. This kit uses the same
sampler, dataset class, `filter_overlong_prompts=False`, `data.shuffle=True` and 128 prompts/step (`run_ampo.py`), so an
uninterrupted run sees the prompts in exactly the upstream order. Conditions:
- same parquet bytes (row order defines the permutation), same `data.train_batch_size` (128); `--pool` changes the order
  by design (likelihood selection), so do not use it for the reproduction run;
- **resume fix**: upstream's sampler has `get_state/set_state` but not `state_dict/load_state_dict`, so torchdata's
  `StatefulDataLoader` could not restore it and replayed epoch 0's permutation after a resume past epoch 1 (measured:
  resume at step 73 -> step 74 differed). Added `state_dict/load_state_dict` (sampling unchanged).
- test: `cd verl && python -m pytest -q ../r1x4_tests/test_ampo_order.py` (uninterrupted + resumed at steps 10, 66,
  73, 162, with 0 and 8 dataloader workers, against the seed-47 reference).

## 3. Smoke test, then real runs
```bash
sbatch scripts/smoke_msi.sbatch                 # 3 steps of variant I, Qwen3-1.7B, 1 A100 -> SMOKE_EXIT 0
# full run, 4 GPUs (a100-4 / msigpu), e.g. variant III on core:
python scripts/run.py --model qwen3-1.7b --data core --p 1 --ches no --ranking no --steps 300 --gpus 4 --dry_run
```
`msigpu` has a 24 h limit. Runs resume automatically from the last checkpoint under the same `--name`, so chain jobs with
`--dependency=afterany`.
