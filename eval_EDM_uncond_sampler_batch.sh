#!/bin/bash
#SBATCH -t 12:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100      # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=80G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 1-9
#SBATCH -o UNet_uncond_%A_%a.out  
#SBATCH -e UNet_uncond_%A_%a.err  
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'WideBlnrX3_new_stream0_16M_heldout0_RAVEN10_abstract_20240708-2054
BigBlnrX3_new_stream0_16M_RAVEN10_abstract_20240828-1332
BigBlnrX3_new_stream0_16M_heldout0_RAVEN10_abstract_20240708-2308
BigBlnrX3_new_stream0_016M_heldout0_RAVEN10_abstract_20240816-0104
BigBlnrX3_new_stream16M_heldout0_RAVEN10_abstract_20240708-2052
BigBlnrX3_new_stream1_6M_heldout0_RAVEN10_abstract_20240708-2052
WideBlnrX3_new_stream0_016M_heldout0_RAVEN10_abstract_20240816-0105
WideBlnrX3_new_stream16M_heldout0_RAVEN10_abstract_20240708-2054
WideBlnrX3_new_stream1_6M_heldout0_RAVEN10_abstract_20240708-2054
'
# BigBlnrX3_new_stream0_016M_RAVEN10_abstract_20240816-0104
# BigBlnrX3_new_stream16M_RAVEN10_abstract_20240705-0237
# BigBlnrX3_new_stream1_6M_RAVEN10_abstract_20240705-1903
# WideBlnrX3_new_stream0_016M_RAVEN10_abstract_20240816-0105
# WideBlnrX3_new_stream16M_RAVEN10_abstract_20240705-0023
# WideBlnrX3_new_stream1_6M_RAVEN10_abstract_20240705-1908

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

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"

## training
python EDM_uncond_sampler_sweep_CLI.py --epoch 999999 --reps 5 \
        --expname $param_name


