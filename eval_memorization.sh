
exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
# 45,46,47,48,49,50,51,52,53,54,55,56
# Assuming the experiment names follow the format "XXX-RAVEN10_abstract*"
for expname in "${exproot}"/0{45..60}-RAVEN10_abstract*; do
    encoding="--encoding digit"
    if [[ "$expname" == *"onehot"* ]]; then
        encoding="--encoding onehot"
    fi

    python check_memorization_CLI.py --ep_step 1000 --ep_start 1000 --ep_stop 1400000 \
        --fmt %07d.pt --exproot "$exproot" --figdir Figures_newrule \
        --expname "$(basename "$expname")" $encoding 
done

# exproot="/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/mini_edm/exps"
exproot="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"
# Assuming the experiment names follow the format "XXX-RAVEN10_abstract*"
for expname in "${exproot}"/*X3_new_RAVEN10_abstract_*2024031* ; do
    encoding="--encoding digit"
    if [[ "$expname" == *"onehot"* ]]; then
        encoding="--encoding onehot"
    fi

    python check_memorization_CLI.py --ep_step 1000 --ep_stop 1200000 \
        --fmt tensor_%s.pt --exproot "$exproot" --figdir Figures_newrule \
        --expname "$(basename "$expname")" $encoding 
done

