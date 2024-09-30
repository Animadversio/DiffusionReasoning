#!/bin/bash

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"
# figdir="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning/uncond_sampler"
# Assuming the experiment names follow the format "XXX-RAVEN10_abstract*"
# for expname in "${exproot}"/0{90}-RAVEN10_abstract*; do
for expname in "${exproot}"/{BaseBlnrX3_new_stream0_16M_RAVEN10_abstract_20240921-2218,\
BaseBlnrX3_new_stream0_16M_heldout0_RAVEN10_abstract_20240921-2218,\
WideBlnrX3_new_stream0_16M_RAVEN10_abstract_20240705-1908,\
WideBlnrX3_new_stream0_16M_heldout0_RAVEN10_abstract_20240708-2054,\
BigBlnrX3_new_stream0_16M_RAVEN10_abstract_20240828-1332,\
BigBlnrX3_new_stream0_16M_heldout0_RAVEN10_abstract_20240708-2308}; do
    echo $expname
    python EDM_uncond_sampler_sweep_CLI.py --expname "$(basename "$expname")" \
        --epoch 999999 --reps 5
done


# exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
# for expname in "${exproot}"/{086,087,088,089,091,092,093,095,097,098,101,102}-RAVEN10_abstract*; do
#     echo $expname
#     python DiT_uncond_sampler_sweep_CLI.py --expname "$(basename "$expname")" \
#         --epoch 1000000 --reps 10
# done


# exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
# for expname in "${exproot}"/{102,101,098,097,095,}-RAVEN10_abstract*; do
#     echo $expname
#     python DiT_uncond_sampler_sweep_CLI.py --expname "$(basename "$expname")" \
#         --epoch 1000000 --reps 10
# done






