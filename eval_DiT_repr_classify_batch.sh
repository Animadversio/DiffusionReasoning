#!/bin/bash
#SBATCH -t 5:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100      # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=160G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 1-12
#SBATCH -o /n/home12/binxuwang/Github/DiffusionReasoning/cluster_log/DiT_repr_classify_%A_%a.out  
#SBATCH -e /n/home12/binxuwang/Github/DiffusionReasoning/cluster_log/DiT_repr_classify_%A_%a.err  
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
"
export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

module load python
conda deactivate
mamba activate torch2
which python
which python3

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
figdir="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning/repr_rule_classify"
echo $param_name 
# expname

epochs=(1000000 20000 100000 200000 500000 700000 -1)
# Loop over each epoch value
cd /n/home12/binxuwang/Github/DiffusionReasoning
for epoch in "${epochs[@]}"; do
    python DiT_repr_classifier_new_probe_CLI.py \
        --expname "$param_name" \
        --epoch "$epoch" \
        --t_scalars 0 10 25 50 100 200 500 1000 \
        --dim_red_method avgtoken pca384 \
        --layers 0 2 5 8 11 \
        --figdir "$figdir"
done

