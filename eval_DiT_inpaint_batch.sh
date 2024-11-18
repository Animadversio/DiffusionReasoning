#!/bin/bash
#SBATCH -t 12:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner      # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=100G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 13-24
#SBATCH -o /n/home12/binxuwang/Github/DiffusionReasoning/cluster_log/DiT_inpaint_eval_%A_%a.out  
#SBATCH -e /n/home12/binxuwang/Github/DiffusionReasoning/cluster_log/DiT_inpaint_eval_%A_%a.err  
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
"103-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout012345_20241113-0052
104-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout01_20241113-0052
104-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout012_20241113-0052
105-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout0_20241113-0052
106-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout0123_20241113-0052
106-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout01234_20241113-0052
107-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_attr0_20241113-0056
107-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_attr3_20241113-0056
107-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel0_20241113-0056
107-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel3_20241113-0056
108-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel5_20241113-0057
108-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel8_20241113-0057
114-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel012_20241116-0330
114-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel01234_20241116-0330
114-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel56_20241116-0330
115-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel014_20241116-0330
116-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel1234_20241116-0331
117-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel0123456_20241116-0331
117-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel023_20241116-0331
117-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel034_20241116-0331
117-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel123_20241116-0331
117-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel789_20241116-0331
118-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel234_20241116-0331
119-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout_rel56789_20241116-0331
"
export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

module load python
conda deactivate
mamba activate torch2
which python
which python3

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"

epochs=(1000000 20000 100000 200000 500000 700000 -1)
# Loop over each epoch value
for epoch in "${epochs[@]}"; do
    python DiT_inpaint_tds_sampler_CLI.py \
        --tds_batch_size 25  --tds_step 100 \
        --sample_id_num 50  --sample_id_offset 50 \
        --expname $param_name \
        --epoch "$epoch"
done

