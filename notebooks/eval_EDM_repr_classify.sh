
expname="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/WideBlnrX3_new_RAVEN10_abstract_20240315-1327"

python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 20000
python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 40000
python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 100000
python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 200000
python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 500000
python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch 700000
python EDM_repr_classifier_probe_CLI.py --expname "$(basename "$expname")" --t_scalars 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 --epoch -1

# --expname "$(basename "$expname")" --epoch -1