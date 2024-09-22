#!/bin/bash

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/SiT/results"
# figdir="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning/repr_rule_classify_SiT"
for expname in "${exproot}"/014-SiT_S_1-stream0_16M_heldout0*; do
    encoding="--encoding digit"
    if [[ "$expname" == *"onehot"* ]]; then
        encoding="--encoding onehot"
    fi

    echo $expname
    python SiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 --use_ema \
        --t_scalars 0.0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 1.0  --dim_red_method avgtoken pca384 \
        --layers 0 2 5 8 11 

    epochs=(20000 100000 200000 500000 700000 900000)

    # Loop over each epoch value
    for epoch in "${epochs[@]}"; do
        python SiT_repr_classifier_new_probe_CLI.py \
            --expname "$(basename "$expname")" \
            --epoch "$epoch" \
            --use_ema \
            --t_scalars 0.0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 1.0  \
            --dim_red_method avgtoken pca384 \
            --layers 0 2 5 8 11 
    done

    python SiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
        --t_scalars 0.0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 1.0  --dim_red_method avgtoken pca384 \
        --layers 0 2 5 8 11 
    # --figdir "$figdir"
done

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/SiT/results"

for expname in "${exproot}"/013-SiT_S_1-stream0_016M_heldout0-Linear-velocity-None; do
    encoding="--encoding digit"
    if [[ "$expname" == *"onehot"* ]]; then
        encoding="--encoding onehot"
    fi

    echo $expname
    python SiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 --use_ema \
        --t_scalars 0.0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 1.0  --dim_red_method avgtoken pca384 \
        --layers 0 2 5 8 11 

    epochs=(20000 100000 200000 500000 700000 900000)

    # Loop over each epoch value
    for epoch in "${epochs[@]}"; do
        python SiT_repr_classifier_new_probe_CLI.py \
            --expname "$(basename "$expname")" \
            --epoch "$epoch" \
            --use_ema \
            --t_scalars 0.0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 1.0  \
            --dim_red_method avgtoken pca384 \
            --layers 0 2 5 8 11 
    done

    python SiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
        --t_scalars 0.0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 1.0  --dim_red_method avgtoken pca384 \
        --layers 0 2 5 8 11 
    --figdir "$figdir"
done

# for expname in "${exproot}"/090-RAVEN10_abstract*; do
#     encoding="--encoding digit"
#     if [[ "$expname" == *"onehot"* ]]; then
#         encoding="--encoding onehot"
#     fi

#     epochs=(1000000 20000 100000 200000 500000 700000)

#     # Loop over each epoch value
#     for epoch in "${epochs[@]}"; do
#         python SiT_repr_classifier_new_probe_CLI.py \
#             --expname "$(basename "$expname")" \
#             --epoch "$epoch" \
#             --t_scalars 0.0 0.01 0.05 0.1 0.2 0.5 0.6 0.7 0.8 0.9 0.95 0.99 1.0 --use_ema \
#             --dim_red_method avgtoken pca384 \
#             --layers 0 2 5 8 11 \
#             --figdir "$figdir"
#     done

# done








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





# for expname in "${exproot}"/0{71,7}-RAVEN10_abstract*; do
#     encoding="--encoding digit"
#     if [[ "$expname" == *"onehot"* ]]; then
#         encoding="--encoding onehot"
#     fi

#     echo $expname
#     # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
#     #     --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
#     python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 \
#         --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
#     python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 20000 \
#         --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
#     # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 100000 \
#     #     --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
#     # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 200000 \
#     #     --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
#     # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 500000 \
#     #     --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule

#     # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
#     #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
#     python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 \
#         --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --noPCA --figdir Figures_newrule
#     python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 20000 \
#         --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --noPCA --figdir Figures_newrule
#     # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 100000 \
#     #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
#     # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 200000 \
#     #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
#     # python DiT_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" --epoch 500000 \
#     #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
#     # python DiT_repr_classifier_new_probe_CLI.py.py --expname "$(basename "$expname")" --epoch 1000000 \
#     #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
# done