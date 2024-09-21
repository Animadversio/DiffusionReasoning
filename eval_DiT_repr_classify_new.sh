#!/bin/bash

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
figdir="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning/repr_rule_classify"
# 45,46,47,48,49,50,51,52,53,54,55,56
# Assuming the experiment names follow the format "XXX-RAVEN10_abstract*"
# for expname in "${exproot}"/0{90}-RAVEN10_abstract*; do
for expname in "${exproot}"/090-RAVEN10_abstract*; do
    encoding="--encoding digit"
    if [[ "$expname" == *"onehot"* ]]; then
        encoding="--encoding onehot"
    fi

    echo $expname
    python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 \
        --t_scalars 0 1 10 25 50 100 250 500 1000 --dim_red_method avgtoken pca384 \
        --layers 0 2 5 8 11 --figdir "$figdir"

    epochs=(20000 100000 200000 500000 700000 1000000)

    # Loop over each epoch value
    for epoch in "${epochs[@]}"; do
        python DiT_repr_classifier_new_probe_CLI.py \
            --expname "$(basename "$expname")" \
            --epoch "$epoch" \
            --t_scalars 0 1 10 25 50 100 250 500 1000 \
            --dim_red_method avgtoken pca384 \
            --layers 0 2 5 8 11 \
            --figdir "$figdir"
    done

    python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
        --t_scalars 0 1 10 25 50 100 250 500 1000 --dim_red_method avgtoken pca384 \
        --layers 0 2 5 8 11 --figdir "$figdir"
done


for expname in "${exproot}"/090-RAVEN10_abstract*; do
    encoding="--encoding digit"
    if [[ "$expname" == *"onehot"* ]]; then
        encoding="--encoding onehot"
    fi

    epochs=(1000000 20000 100000 200000 500000 700000)

    # Loop over each epoch value
    for epoch in "${epochs[@]}"; do
        python DiT_repr_classifier_new_probe_CLI.py \
            --expname "$(basename "$expname")" \
            --epoch "$epoch" \
            --t_scalars 0 1 10 25 50 100 250 500 1000 --use_ema \
            --dim_red_method avgtoken pca384 \
            --layers 0 2 5 8 11 \
            --figdir "$figdir"
    done

done




for expname in "${exproot}"/090-RAVEN10_abstract*; do
    encoding="--encoding digit"
    if [[ "$expname" == *"onehot"* ]]; then
        encoding="--encoding onehot"
    fi

    echo $expname
    python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 \
        --t_scalars 0 1 10 25 50 100 250 500 1000 --dim_red_method pca128 pca384 pca512 pca1024 avgtoken lasttoken \
        --layers 0 2 5 8 11 --figdir "$figdir"

    epochs=(20000 100000 200000 500000 700000 1000000)

    # Loop over each epoch value
    for epoch in "${epochs[@]}"; do
        python DiT_repr_classifier_new_probe_CLI.py \
            --expname "$(basename "$expname")" \
            --epoch "$epoch" \
            --t_scalars 0 1 10 25 50 100 250 500 1000 \
            --dim_red_method pca384 pca512 pca1024 avgtoken \
            --layers 0 2 5 8 11 \
            --figdir "$figdir"
    done

    python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
        --t_scalars 0 1 10 25 50 100 250 500 1000 --dim_red_method pca128 pca384 pca512 pca1024 avgtoken lasttoken \
        --layers 0 2 5 8 11 --figdir "$figdir"
done


    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
    #     --t_scalars 0 1 10 25 50 100 250 500 1000 --dim_red_method pca128 pca384 pca512 pca1024 avgtoken lasttoken \
    #     --layers 0 2 5 8 11 --figdir "$figdir"

    

    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 20000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 100000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 200000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 500000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_new_probe_CLI.py.py --expname "$(basename "$expname")" --epoch 1000000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule





for expname in "${exproot}"/0{71,7}-RAVEN10_abstract*; do
    encoding="--encoding digit"
    if [[ "$expname" == *"onehot"* ]]; then
        encoding="--encoding onehot"
    fi

    echo $expname
    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
    #     --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
    python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 \
        --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
    python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 20000 \
        --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 100000 \
    #     --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 200000 \
    #     --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 500000 \
    #     --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule

    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 \
        --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --noPCA --figdir Figures_newrule
    python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 20000 \
        --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 100000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 200000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 500000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_new_probe_CLI.py.py --expname "$(basename "$expname")" --epoch 1000000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
done