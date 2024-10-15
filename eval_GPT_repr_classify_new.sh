#!/bin/bash

cd /n/home12/binxuwang/Github/DiffusionReasoning
exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/GPT2_raven"
# figdir="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning/repr_rule_classify_SiT"
for expname in "${exproot}"/GPT2_medium_RAVEN_uncond_heldout0_stream0_016M-20240820-02473*; do
# GPT2_medium_RAVEN_uncond_heldout0_stream0_16M-20240820-024*; do
    echo $expname
# 
    train_steps=(-1 999999 24999 49999 99999 199999 499999 699999 899999)
    # Loop over each train_step value
    for train_step in "${train_steps[@]}"; do
        python GPT_repr_classifier_new_probe_CLI.py \
            --expname "$(basename "$expname")" \
            --train_step "$train_step" \
            --dim_red_method avgtoken lasttoken pca384 pca768 \
            --layers 0 5 11 17 23 
    done

    # python GPT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --train_step -1 \
    #     --dim_red_method avgtoken lasttoken pca384 pca768 \
    #     --layers 0 5 11 17 23 

    # python GPT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --train_step 999999 \
    #     --dim_red_method avgtoken lasttoken pca384 pca768 \
    #     --layers 0 5 11 17 23
done






exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/GPT2_raven"

for expname in "${exproot}"/013-SiT_S_1-stream0_016M_heldout0-Linear-velocity-None; do

    echo $expname
    
    train_steps=(-1 999999 24999 49999 99999 199999 499999 699999 899999)
    # Loop over each train_step value
    for train_step in "${train_steps[@]}"; do
        python GPT_repr_classifier_new_probe_CLI.py \
            --expname "$(basename "$expname")" \
            --train_step "$train_step" \
            --dim_red_method avgtoken lasttoken pca384 pca768 \
            --layers 0 2 5 8 11 
    done

done
