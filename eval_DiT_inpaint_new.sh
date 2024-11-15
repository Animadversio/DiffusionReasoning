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


exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
for expname in "${exproot}"/{098,097}-RAVEN10_abstract*; do
    echo $expname
    epochs=(20000 100000 200000 500000 700000 1000000)
    # Loop over each epoch value
    for epoch in "${epochs[@]}"; do
        python DiT_inpaint_tds_sampler_CLI.py \
            --tds_batch_size 25  --sample_id_num 50  --sample_id_offset 50 \
            --expname "$(basename "$expname")" \
            --epoch "$epoch"
    done
done


exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
for expname in "${exproot}"/090-RAVEN10_abstract*; do
    echo $expname
    bsizes=(1 2 4 8 16 32 64 128 256)
    # Loop over each epoch value
    for bsize in "${bsizes[@]}"; do
        python DiT_inpaint_tds_sampler_CLI.py \
            --tds_batch_size "$bsize"  --sample_id_num 50  --sample_id_offset 50 \
            --expname "$(basename "$expname")" \
            --epoch 1000000
    done
done

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
for expname in "${exproot}"/090-RAVEN10_abstract*; do
    echo $expname
    steps=(10 20 50 80 125 250 500 1000)
    # Loop over each epoch value
    for step in "${steps[@]}"; do
        python DiT_inpaint_tds_sampler_CLI.py \
            --tds_batch_size 25 --tds_step "$step" --sample_id_num 50  --sample_id_offset 50 \
            --expname "$(basename "$expname")" \
            --epoch 1000000
    done
done




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
    
