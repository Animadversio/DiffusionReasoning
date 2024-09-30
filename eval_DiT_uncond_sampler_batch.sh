#!/bin/bash
#SBATCH -t 12:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100      # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=80G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 1-3
#SBATCH -o DiT_uncond_%A_%a.out  
#SBATCH -e DiT_uncond_%A_%a.err  
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'098-RAVEN10_abstract-uncond-DiT_S_1-stream0_016M_heldout0_20240816-0050
097-RAVEN10_abstract-uncond-DiT_S_1-stream0_016M_20240816-0049
095-RAVEN10_abstract-uncond-DiT_B_1-stream16M_heldout0_20240711-0205
'
export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"
# load modules
module load python
conda deactivate
mamba activate torch2
which python
which python3
# run code
cd /n/home12/binxuwang/Github/DiffusionReasoning

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"

## training
python DiT_uncond_sampler_sweep_CLI.py --epoch 1000000 --reps 10 \
        --expname $param_name



