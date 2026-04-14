#!/bin/bash

#SBATCH --job-name=PDARTS
#SBATCH --output=slurm/slurm-%x-%A_%a.out
# SBATCH --time=5-00:10:00
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p tau
#SBATCH --exclude=margpu018,margpu021
# SBATCH --array=[1-3]

STAGGER_SECONDS=10
SLEEP_TIME=$(( (SLURM_ARRAY_TASK_ID - 1) * STAGGER_SECONDS ))
echo "Array task ${SLURM_ARRAY_TASK_ID}: sleeping ${SLEEP_TIME}s before starting"
sleep "${SLEEP_TIME}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi

if [ "${SLURM_ARRAY_JOB_ID}" ] ; then
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    JOB_ID="${SLURM_JOB_ID}"
fi
echo "JOB ID    = ${JOB_ID}"
echo "NODE NAME = ${SLURMD_NODENAME}"

echo "DATASET   = ${NAS_DATASET}"

python train_search.py --dataset "${NAS_DATASET}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs 25 --eval_epochs 200 \
    --eval_batch_size 4 \
    --experiment_name Budget --init_genotype PDARTS_eager_wood_36
    # --dropout_rate 0.0 --dropout_rate 0.4 --dropout_rate 0.7 --eval_auxiliary \
