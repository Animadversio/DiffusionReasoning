#!/bin/bash

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
figdir="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning/uncond_sampler"
# 45,46,47,48,49,50,51,52,53,54,55,56
# Assuming the experiment names follow the format "XXX-RAVEN10_abstract*"
# for expname in "${exproot}"/0{90}-RAVEN10_abstract*; do
for expname in "${exproot}"/{090,094,085,096}-RAVEN10_abstract*; do
    echo $expname
    python DiT_uncond_sampler_sweep_CLI.py --expname "$(basename "$expname")" \
        --epoch 1000000 --reps 10
done




exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
figdir="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning/repr_rule_classify"

for expname in "${exproot}"/098-RAVEN10_abstract-uncond-DiT_S_1-stream0_016M_heldout0*; do
    encoding="--encoding digit"
    if [[ "$expname" == *"onehot"* ]]; then
        encoding="--encoding onehot"
    fi

    echo $expname
    python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 --use_ema \
        --t_scalars 0 1 10 25 50 100 250 500 1000 --dim_red_method avgtoken pca384 \
        --layers 0 2 5 8 11 --figdir "$figdir"

    epochs=(20000 100000 200000 500000 700000 900000)

    # Loop over each epoch value
    for epoch in "${epochs[@]}"; do
        python DiT_repr_classifier_new_probe_CLI.py \
            --expname "$(basename "$expname")" \
            --epoch "$epoch" --use_ema \
            --t_scalars 0 1 10 25 50 100 250 500 1000 \
            --dim_red_method avgtoken pca384 \
            --layers 0 2 5 8 11 \
            --figdir "$figdir"
    done

    python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
        --t_scalars 0 1 10 25 50 100 250 500 1000 --dim_red_method avgtoken pca384 \
        --layers 0 2 5 8 11 --figdir "$figdir"
done


