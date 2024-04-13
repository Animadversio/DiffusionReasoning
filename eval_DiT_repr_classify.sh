#!/bin/bash

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
# 45,46,47,48,49,50,51,52,53,54,55,56
# Assuming the experiment names follow the format "XXX-RAVEN10_abstract*"
for expname in "${exproot}"/0{76,77,78}-RAVEN10_abstract*; do
    encoding="--encoding digit"
    if [[ "$expname" == *"onehot"* ]]; then
        encoding="--encoding onehot"
    fi

    echo $expname
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
        --t_scalars 0.1 --PC_dim 1024 --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 \
        --t_scalars 0.1 --PC_dim 1024 --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 20000 \
        --t_scalars 0.1 --PC_dim 1024 --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 100000 \
        --t_scalars 0.1 --PC_dim 1024 --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 200000 \
        --t_scalars 0.1 --PC_dim 1024 --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 500000 \
        --t_scalars 0.1 --PC_dim 1024 --figdir Figures_newrule

    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
        --t_scalars 0.1 --noPCA --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 \
        --t_scalars 0.1 --noPCA --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 20000 \
        --t_scalars 0.1 --noPCA --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 100000 \
        --t_scalars 0.1 --noPCA --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 200000 \
        --t_scalars 0.1 --noPCA --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 500000 \
        --t_scalars 0.1 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_probe_CLI.py.py --expname "$(basename "$expname")" --epoch 1000000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
done




for expname in "${exproot}"/0{71,7}-RAVEN10_abstract*; do
    encoding="--encoding digit"
    if [[ "$expname" == *"onehot"* ]]; then
        encoding="--encoding onehot"
    fi

    echo $expname
    # python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
    #     --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 \
        --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 20000 \
        --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
    # python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 100000 \
    #     --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
    # python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 200000 \
    #     --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule
    # python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 500000 \
    #     --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --PC_dim 1024 --figdir Figures_newrule

    # python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch -1 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 1000000 \
        --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --noPCA --figdir Figures_newrule
    python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 20000 \
        --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 100000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 200000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --epoch 500000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
    # python DiT_repr_classifier_probe_CLI.py.py --expname "$(basename "$expname")" --epoch 1000000 \
    #     --t_scalars 0.1 --noPCA --figdir Figures_newrule
done