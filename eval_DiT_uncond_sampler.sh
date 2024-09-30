#!/bin/bash

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
# figdir="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning/uncond_sampler"
# Assuming the experiment names follow the format "XXX-RAVEN10_abstract*"
# for expname in "${exproot}"/0{90}-RAVEN10_abstract*; do
for expname in "${exproot}"/{090,094,085,096}-RAVEN10_abstract*; do
    echo $expname
    python DiT_uncond_sampler_sweep_CLI.py --expname "$(basename "$expname")" \
        --epoch 1000000 --reps 10
done

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
for expname in "${exproot}"/{086,087,088,089,091,092,093,095,097,098,101,102}-RAVEN10_abstract*; do
    echo $expname
    python DiT_uncond_sampler_sweep_CLI.py --expname "$(basename "$expname")" \
        --epoch 1000000 --reps 10
done


exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
for expname in "${exproot}"/{102,101,098,097,095,}-RAVEN10_abstract*; do
    echo $expname
    python DiT_uncond_sampler_sweep_CLI.py --expname "$(basename "$expname")" \
        --epoch 1000000 --reps 10
done






