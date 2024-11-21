#!/bin/bash
#SBATCH -t 15:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100      # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=130G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 1-3
#SBATCH -o /n/home12/binxuwang/Github/DiffusionReasoning/cluster_log/GPT_RAVEN_joint_train_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home12/binxuwang/Github/DiffusionReasoning/cluster_log/GPT_RAVEN_joint_train_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--explabel GPT2_small_joint_lm_joint_emb_RAVEN_uncond_heldout0_stream0_16M  --heldout_ids 1 16 20 34 37 --embed_type joint --cmb_per_class 4000 --n_class 0 --n_embd 384 --n_layer 12 --n_head 6  --batch_size 64 --total_steps 1000000
--explabel GPT2_small_joint_lm_sep_emb_RAVEN_uncond_heldout0_stream0_16M  --heldout_ids 1 16 20 34 37 --embed_type sep    --cmb_per_class 4000 --n_class 0 --n_embd 384 --n_layer 12 --n_head 6  --batch_size 64 --total_steps 1000000
--explabel GPT2_small_joint_lm_cmb_emb_RAVEN_uncond_heldout0_stream0_16M  --heldout_ids 1 16 20 34 37 --embed_type cmb    --cmb_per_class 4000 --n_class 0 --n_embd 384 --n_layer 12 --n_head 6  --batch_size 64 --total_steps 1000000
--explabel GPT2_base_joint_lm_joint_emb_RAVEN_uncond_heldout0_stream0_16M --heldout_ids 1 16 20 34 37 --embed_type joint --cmb_per_class 4000 --n_class 0 --n_embd 768 --n_layer 12 --n_head 12 --batch_size 64  --total_steps 1000000
--explabel GPT2_base_joint_lm_sep_emb_RAVEN_uncond_heldout0_stream0_16M --heldout_ids 1 16 20 34 37 --embed_type sep   --cmb_per_class 4000 --n_class 0 --n_embd 768 --n_layer 12 --n_head 12 --batch_size 64  --total_steps 1000000
--explabel GPT2_base_joint_lm_cmb_emb_RAVEN_uncond_heldout0_stream0_16M --heldout_ids 1 16 20 34 37 --embed_type cmb   --cmb_per_class 4000 --n_class 0 --n_embd 768 --n_layer 12 --n_head 12 --batch_size 64  --total_steps 1000000
'
# --explabel GPT2_medium_RAVEN_uncond_heldout0_stream0_16M --heldout_ids 1 16 20 34 37  --cmb_per_class 4000  --n_class 0 --n_embd 768 --n_layer 24 --n_head 12 --batch_size 256 --total_steps 1000000
# --explabel GPT2_base_RAVEN_uncond_heldout0_stream0_16M   --heldout_ids 1 16 20 34 37  --cmb_per_class 4000  --n_class 0 --n_embd 768 --n_layer 12 --n_head 12 --batch_size 256 --total_steps 1000000
# --explabel GPT2_small_RAVEN_uncond_heldout0_stream0_16M  --heldout_ids 1 16 20 34 37  --cmb_per_class 4000  --n_class 0 --n_embd 384 --n_layer 12 --n_head 6  --batch_size 256 --total_steps 1000000
# 36 hrs for big models
# 24 hrs for wide models

export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

# load modules
module load python
conda deactivate
mamba activate torch2
which python
which python3
# run code
cd /n/home12/binxuwang/Github/DiffusionReasoning/GPT_models

## training
python -u GPT_raven_joint_train_cond_stream_CLI.py  --lr 1e-4 --num_warmup_steps 100 \
    --eval_temperature 1.0 --save_ckpt_every_step 25000 --eval_model_every_step 2500 \
    $param_name        

