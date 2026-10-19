#!/usr/bin/env bash
# 00:05 Mon: final neggen build -> rebuild the 5 R1x4 sets from it (CPU job) -> zip the kit for the RTX box.
L=/scratch.global/radke149/r1x4_runs/logs
while [ "$(date +%Y%m%d%H%M)" -lt 202610190005 ]; do sleep 60; done
/scratch.global/radke149/envs/luffy/bin/python /scratch.global/radke149/neggen/work/build_dataset.py > /scratch.global/radke149/neggen/logs/build_final.log 2>&1
echo "$(date) neggen final build exit $?" >> $L/chain.log
j=$(sbatch --parsable /scratch.global/radke149/r1x4kit_1p7b_4b/scripts/build_msi.sbatch luffy21k,core,extended,r1qwen,qwen3)
echo "$(date) r1x4 build job $j" >> $L/chain.log
while squeue -h -j $j 2>/dev/null | grep -q .; do sleep 60; done
grep -q '"set": "qwen3"' $L/build_$j.out && bash /scratch.global/radke149/r1x4kit_1p7b_4b/scripts/make_zip.sh >> $L/chain.log 2>&1
echo "$(date) chain done" >> $L/chain.log
