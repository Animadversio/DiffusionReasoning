#!/bin/bash
#SBATCH -t 8:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100      # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=130G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 7-11
#SBATCH -o /n/home12/binxuwang/Github/DiffusionReasoning/cluster_log/BERT_RAVEN_train_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home12/binxuwang/Github/DiffusionReasoning/cluster_log/BERT_RAVEN_train_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--explabel BERT_S_sep_emb_pilot_all_stream0_16M --cmb_per_class 4000 --heldout_ids  --embed_type sep --n_embd 384  --n_layer 12  --n_head 6   --lr 1e-4 --batch_size 64  --num_warmup_steps 1000 --total_steps 100000 
--explabel BERT_B_sep_emb_pilot_all_stream0_16M --cmb_per_class 4000 --heldout_ids  --embed_type sep --n_embd 768  --n_layer 12  --n_head 12   --lr 1e-4 --batch_size 64  --num_warmup_steps 1000 --total_steps 100000 
--explabel BERT_M_sep_emb_pilot_all_stream0_16M --cmb_per_class 4000 --heldout_ids  --embed_type sep --n_embd 768  --n_layer 24  --n_head 12   --lr 1e-4 --batch_size 64  --num_warmup_steps 1000 --total_steps 100000 
--explabel BERT_S_sep_emb_pilot_heldout0_stream0_16M --cmb_per_class 4000 --heldout_ids 1 16 20 34 37  --embed_type sep --n_embd 384  --n_layer 12  --n_head 6   --lr 1e-4 --batch_size 64  --num_warmup_steps 1000 --total_steps 100000 
--explabel BERT_B_sep_emb_pilot_heldout0_stream0_16M --cmb_per_class 4000 --heldout_ids 1 16 20 34 37  --embed_type sep --n_embd 768  --n_layer 12  --n_head 12   --lr 1e-4 --batch_size 64  --num_warmup_steps 1000 --total_steps 100000 
--explabel BERT_M_sep_emb_pilot_heldout0_stream0_16M --cmb_per_class 4000 --heldout_ids 1 16 20 34 37  --embed_type sep --n_embd 768  --n_layer 24  --n_head 12   --lr 1e-4 --batch_size 64  --num_warmup_steps 1000 --total_steps 100000 
--explabel BERT_S_sep_emb_pilot_all_stream0_16M --cmb_per_class 4000 --heldout_ids  --embed_type sep --n_embd 384  --n_layer 12  --n_head 6   --lr 1e-4 --batch_size 64  --num_warmup_steps 1000 --total_steps 250000 
--explabel BERT_B_sep_emb_pilot_all_stream0_16M --cmb_per_class 4000 --heldout_ids  --embed_type sep --n_embd 768  --n_layer 12  --n_head 12   --lr 1e-4 --batch_size 64  --num_warmup_steps 1000 --total_steps 250000 
--explabel BERT_M_sep_emb_pilot_all_stream0_16M --cmb_per_class 4000 --heldout_ids  --embed_type sep --n_embd 768  --n_layer 24  --n_head 12   --lr 1e-4 --batch_size 64  --num_warmup_steps 1000 --total_steps 250000 
--explabel BERT_B_cmb_emb_pilot_all_stream0_16M --cmb_per_class 4000 --heldout_ids  --embed_type cmb --n_embd 768  --n_layer 12  --n_head 12   --lr 1e-4 --batch_size 64  --num_warmup_steps 1000 --total_steps 250000 
--explabel BERT_B_joint_emb_pilot_all_stream0_16M --cmb_per_class 4000 --heldout_ids  --embed_type joint --n_embd 768  --n_layer 12  --n_head 12   --lr 1e-4 --batch_size 64  --num_warmup_steps 1000 --total_steps 250000 
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
cd /n/home12/binxuwang/Github/DiffusionReasoning/BERT_models

## training
python -u BERT_raven_train_CLI.py \
    --n_class 40 \
    --eval_model_every_step 1000 \
    --save_ckpt_every_step 5000 \
    $param_name 
