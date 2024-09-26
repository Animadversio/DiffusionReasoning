#!/bin/bash

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"
# Assuming the experiment names follow the format "XXX-RAVEN10_abstract*"
for expname in "${exproot}"/WideBlnrX3_new_stream0_16M_heldout0_RAVEN10_abstract_20240708*; do
    encoding="--encoding digit"
    if [[ "$expname" == *"onehot"* ]]; then
        encoding="--encoding onehot"
    fi

    echo $expname
    python EDM_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" \
        --epoch 999999 --use_ema \
        --dim_red_method avgspace pca512 pca384
        # --t_scalars 0 1 10 25 50 100 250 500 1000 --layers 0 2 5 8 11 \
        

    epochs=(20000 100000 200000 500000 700000 900000)

    # Loop over each epoch value
    for epoch in "${epochs[@]}"; do
        python EDM_repr_classifier_new_probe_CLI.py \
            --expname "$(basename "$expname")" \
            --use_ema --epoch "$epoch" \
            --dim_red_method avgspace pca512 pca384
            # --t_scalars 0 1 10 25 50 100 250 500 1000 --layers 0 2 5 8 11\
    done

    python EDM_repr_classifier_new_probe_CLI.py --expname "$(basename "$expname")" \
        --use_ema --epoch -1 \
        --dim_red_method avgspace pca512 pca384
        # --t_scalars 0 1 10 25 50 100 250 500 1000  --layers 0 2 5 8 11 \
done








expname="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/WideBlnrX3_new_RAVEN10_abstract_20240315-1327"

python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 20000
python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 40000
python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 100000
python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 200000
python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 500000
python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 700000
python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch -1

# --expname "$(basename "$expname")" --epoch -1

expname="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/WideBlnrX3_new_RAVEN10_abstract_20240412-1347"
epochs=(20000 40000 100000 200000 400000 600000 800000 999999 -1)
# Loop through each epoch value
for epoch in "${epochs[@]}"; do
    python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch $epoch
done

expname="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/BaseBlnrX3_new_RAVEN10_abstract_20240313-1736"
epochs=(20000 40000 100000 200000 400000 600000 800000 999999 -1)
# Loop through each epoch value
for epoch in "${epochs[@]}"; do
    python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch $epoch
done

expname="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/WideBlnrX3_new_noattn_RAVEN10_abstract_20240412-1254"
epochs=(20000 40000 100000 200000 400000 600000 800000 999999 -1)
# Loop through each epoch value
for epoch in "${epochs[@]}"; do
    python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch $epoch
done

expname="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/BigBlnrX3_new_RAVEN10_abstract_20240412-0143"
epochs=(20000 40000 100000 200000 400000 600000 800000 999999 -1)
# Loop through each epoch value
for epoch in "${epochs[@]}"; do
    python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch $epoch
done


# python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 20000
# python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 40000
# python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 100000
# python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 200000
# python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 500000
# python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 700000
# python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 900000
# python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 999999
# python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch -1