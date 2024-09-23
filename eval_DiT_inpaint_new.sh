#!/bin/bash
# Test run with held out set. 
exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
# figdir="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning/repr_rule_classify"
# Assuming the experiment names follow the format "XXX-RAVEN10_abstract*"
# for expname in "${exproot}"/0{90}-RAVEN10_abstract*; do
for expname in "${exproot}"/090-RAVEN10_abstract*; do
    echo $expname
    python DiT_inpaint_tds_sampler_CLI.py \
        --tds_batch_size 25  --sample_id_num 50  --sample_id_offset 50 \
        --expname "$(basename "$expname")" \
        --epoch 1000000

    epochs=(20000 100000 200000 500000 700000 900000)
    # Loop over each epoch value
    for epoch in "${epochs[@]}"; do
        python DiT_inpaint_tds_sampler_CLI.py \
            --tds_batch_size 25  --sample_id_num 50  --sample_id_offset 50 \
            --expname "$(basename "$expname")" \
            --epoch "$epoch"
    done
done

# Test run with full training set  
exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
# Assuming the experiment names follow the format "XXX-RAVEN10_abstract*"
for expname in "${exproot}"/085-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M*; do
    echo $expname
    python DiT_inpaint_tds_sampler_CLI.py \
        --tds_batch_size 25  --sample_id_num 50  --sample_id_offset 50 \
        --expname "$(basename "$expname")" \
        --epoch 1000000

    epochs=(20000 100000 200000 500000 700000 900000)
    # Loop over each epoch value
    for epoch in "${epochs[@]}"; do
        python DiT_inpaint_tds_sampler_CLI.py \
            --tds_batch_size 25  --sample_id_num 50  --sample_id_offset 50 \
            --expname "$(basename "$expname")" \
            --epoch "$epoch"
    done
done

# Sweep runs with various experiments
exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
for expname in "${exproot}"/{085..089}-RAVEN10_abstract*; do
    echo "Processing experiment: ${expname}"
    python DiT_inpaint_tds_sampler_CLI.py \
        --tds_batch_size 25  --sample_id_num 50  --sample_id_offset 50 \
        --expname "$(basename "$expname")" \
        --epoch 1000000
done

for expname in "${exproot}"/{091..100}-RAVEN10_abstract*; do
    echo "Processing experiment: ${expname}"
    python DiT_inpaint_tds_sampler_CLI.py \
        --tds_batch_size 25  --sample_id_num 50  --sample_id_offset 50 \
        --expname "$(basename "$expname")" \
        --epoch 1000000
done

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
for expname in "${exproot}"/{101..102}-RAVEN10_abstract*; do
    echo "Processing experiment: ${expname}"
    python DiT_inpaint_tds_sampler_CLI.py \
        --tds_batch_size 25  --sample_id_num 50  --sample_id_offset 50 \
        --expname "$(basename "$expname")" \
        --epoch 1000000
done

for expname in "${exproot}"/{101..102}-RAVEN10_abstract*; do
    echo $expname
    epochs=(20000 100000 200000 500000 700000 900000)
    # Loop over each epoch value
    for epoch in "${epochs[@]}"; do
        python DiT_inpaint_tds_sampler_CLI.py \
            --tds_batch_size 25  --sample_id_num 50  --sample_id_offset 50 \
            --expname "$(basename "$expname")" \
            --epoch "$epoch"
    done
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

for expname in "${exproot}"/{085..089,091..100}-RAVEN10_abstract*; do
# for expnum in {085..100}; do
#     if [ "$expnum" -eq 90 ]; then
#         echo "Skipping experiment: ${expnum}-RAVEN10_abstract*"
#         continue
#     fi
    # expname="${exproot}/${expnum}-RAVEN10_abstract*"
    # for expname in "${exproot}"/{088,089,094,093,095,098,100}-RAVEN10_abstract*; do
    echo "Processing experiment: $expname"
done

