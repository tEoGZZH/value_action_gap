#!/usr/bin/zsh
#SBATCH -J value_action_task1
#SBATCH -o /home/zs473554/jobscripts/value_action_gap/log/llama3.1-8B-base-7_3_task1__%j.txt
#SBATCH -t 05:00:00
#SBATCH -c 8
#SBATCH -p c23g
#SBATCH --gres=gpu:1
#SBATCH -A rwth1809

set -euo pipefail

# ====== paths ======
PROJECT_DIR=/home/zs473554/projects/value_action_gap
VENV_DIR=/work/zs473554/envs/value_action_gap/.venv311

# ====== model config ======
# base
# MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
# finetune(local)
MODEL_ID='/hpcwork/zs473554/checkpoints/open-instruct/llama3.1-8B-base/mixing/seven_three/'

OUT_DIR=${PROJECT_DIR}/results/task1/llama3.1-8B
mkdir -p ${OUT_DIR}

# ====== env ======
cd ${PROJECT_DIR}/src
source ${VENV_DIR}/bin/activate

# ====== run ======
srun python -u -m tasks.task1.eval_llm_statement \
  --model ${MODEL_ID} \
  --out_csv ${OUT_DIR}/mixing-7_3_results.csv
