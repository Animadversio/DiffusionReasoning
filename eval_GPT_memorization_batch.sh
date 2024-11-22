#!/bin/bash
#SBATCH -t 1:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100      # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=160G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 13-15
#SBATCH -o /n/home12/binxuwang/Github/DiffusionReasoning/cluster_log/GPT_memorization_%A_%a.out  
#SBATCH -e /n/home12/binxuwang/Github/DiffusionReasoning/cluster_log/GPT_memorization_%A_%a.err  
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'GPT2_base_joint_lm_cmb_emb_RAVEN_uncond_heldout0_stream0_16M-20241120-200425
GPT2_base_joint_lm_joint_emb_RAVEN_uncond_heldout0_stream0_16M-20241120-200329
GPT2_base_joint_lm_sep_emb_RAVEN_uncond_heldout0_stream0_016M-20241121-002559
GPT2_base_joint_lm_sep_emb_RAVEN_uncond_heldout0_stream0_16M-20241120-200339
GPT2_base_joint_lm_sep_emb_RAVEN_uncond_heldout0_stream16M-20241121-002522
GPT2_base_joint_lm_sep_emb_RAVEN_uncond_heldout0_stream1_6M-20241121-002546
GPT2_small_joint_lm_cmb_emb_RAVEN_uncond_heldout0_stream0_16M-20241120-200229
GPT2_small_joint_lm_joint_emb_RAVEN_uncond_heldout0_stream0_16M-20241120-200253
GPT2_small_joint_lm_sep_emb_RAVEN_uncond_heldout0_stream0_016M-20241121-140338
GPT2_small_joint_lm_sep_emb_RAVEN_uncond_heldout0_stream0_16M-20241120-200321
GPT2_small_joint_lm_sep_emb_RAVEN_uncond_heldout0_stream16M-20241121-140316
GPT2_small_joint_lm_sep_emb_RAVEN_uncond_heldout0_stream1_6M-20241121-140306
GPT2_medium_joint_lm_cmb_emb_RAVEN_uncond_heldout0_stream0_16M-20241120-202833
GPT2_medium_joint_lm_joint_emb_RAVEN_uncond_heldout0_stream0_16M-20241120-202725
GPT2_medium_joint_lm_sep_emb_RAVEN_uncond_heldout0_stream0_16M-20241120-202729
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
cd /n/home12/binxuwang/Github/DiffusionReasoning/

# exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"

## training
python GPT_eval_memorization_script_CLI.py \
        --expname $param_name


