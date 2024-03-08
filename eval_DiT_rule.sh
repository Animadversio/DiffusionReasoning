#!/bin/bash

# bash
# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 001-RAVEN10_abstract-DiT_B_1 

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 002-RAVEN10_abstract-DiT_S_1 

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 003-RAVEN10_abstract_onehot-DiT_S_1 --encoding onehot 

# 000-RAVEN10_abstract-DiT_S_1  
# 001-RAVEN10_abstract-DiT_B_1  
# 002-RAVEN10_abstract-DiT_S_1  
# 003-RAVEN10_abstract_onehot-DiT_S_1  

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 006-RAVEN10_abstract-DiT_S_1  --update

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 007-RAVEN10_abstract_onehot-DiT_S_1  --encoding onehot  --update

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 008-RAVEN10_abstract-DiT_S_3  --update       

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 009-RAVEN10_abstract-DiT_S_1   --update                              

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 011-RAVEN10_abstract-DiT_B_1  --update

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 012-RAVEN10_abstract-DiT_B_3  --update

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 013-RAVEN10_abstract-DiT_S_3  --update

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 014-RAVEN10_abstract_onehot-DiT_S_3  --encoding onehot  --update

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 015-RAVEN10_abstract_onehot-DiT_B_3  --encoding onehot  --update

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 016-RAVEN10_abstract-DiT_S_3  --update       

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 017-RAVEN10_abstract_onehot-DiT_S_1  --encoding onehot  --update                              

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 018-RAVEN10_abstract-DiT_S_1  --update

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 019-RAVEN10_abstract_onehot-DiT_S_1  --encoding onehot  --update                              

# python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
#         --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
#         --expname 020-RAVEN10_abstract-DiT_S_1   --update

exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"

# Assuming the experiment names follow the format "XXX-RAVEN10_abstract*"
for expname in "${exproot}"/0{3,4}?-RAVEN10_abstract*; do
    encoding="--encoding digit"
    if [[ "$expname" == *"onehot"* ]]; then
        encoding="--encoding onehot"
    fi

    python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
        --exproot "$exproot" \
        --expname "$(basename "$expname")" $encoding --update
done

