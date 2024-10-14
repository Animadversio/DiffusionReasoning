# %%

# %%
import os
import re
import json
import pickle as pkl
from os.path import join
from tqdm.auto import trange, tqdm
from easydict import EasyDict as edict
import numpy as np
import torch as th
import einops
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time 

import sys
sys.path.append('/n/home12/binxuwang/Github/DiffusionReasoning')
from stats_plot_utils import estimate_CI, shaded_error, saveallforms
from rule_new_utils import get_rule_annot
from GPT_models.GPT_RAVEN_model_lib import MultiIdxGPT2Model, completion_eval, seqtsr2imgtsr, preprocess_ids
from repr_probe_lib import train_pca_sgd_classifiers, fit_SGD_linear_classifier, train_dimred_sgd_classifiers, extract_features_GPT
from stats_plot_utils import visualize_cm
from sklearn.metrics import confusion_matrix
import circuit_toolkit
print(circuit_toolkit.__file__)
from circuit_toolkit.layer_hook_utils import print_specific_layer, get_module_name_shapes, featureFetcher_module

# %% Set up paths
GPTroot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/GPT2_raven"
figroot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning/Figure_repr_linear_probe"

def load_gpt2_raven_model(expdir, pth_name='gpt2_final.pth'):
    # load the model configuration
    config = json.load(open(join(expdir, 'config.json')))
    n_embd = config['n_embd']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_class = config['n_class']
    heldout_id = config['heldout_id']

    print("Model configuration:")
    print(f"Embedding dimension: {n_embd}")
    print(f"Number of layers: {n_layer}")
    print(f"Number of attention heads: {n_head}")
    print(f"Number of classes: {n_class}")
    print(f"Held-out IDs: {heldout_id}")
    # load the model
    gpt2_raven = MultiIdxGPT2Model(attribute_dims=(7,10,10), vocab_size=27, max_length=83, 
                                   n_embd=n_embd, n_class=n_class, n_layer=n_layer, n_head=n_head)
    if pth_name is not None:
        gpt2_raven.load_state_dict(th.load(join(expdir, 'ckpt', pth_name)))
    else:
        print("Random initialization")
    # count the number of parameters
    num_params = sum(p.numel() for p in gpt2_raven.parameters())
    print(f"Number of parameters: {num_params}")
    return gpt2_raven.to('cuda').eval()


import argparse
def validate_dim_red(value):
    pattern = re.compile(r'^(pca\d+|none|avgtoken|lasttoken)$')
    if not pattern.match(value):
        raise argparse.ArgumentTypeError(
            f"Invalid choice: '{value}'. Must match one of: pca, none, avgtoken, lasttoken."
        )
    return value

parser = argparse.ArgumentParser(description="Load and evaluate a GPT-2 RAVEN model.")
# Experiment configuration
parser.add_argument(
    "--expname",
    type=str,
    default="GPT2_medium_RAVEN_uncond_heldout0_stream0_16M-20240820-024019",
    help="The name of the GPT-2 experiment."
)
parser.add_argument(
    "--train_step",
    type=int,
    default=999999,
    help="The training step of the model checkpoint to load."
)
# Evaluation parameters
parser.add_argument(
    "--layers",
    type=int, default=[0, 5, 11, 17, 23], nargs="+", 
    help="The layers of the GPT-2 model to evaluate."
)
parser.add_argument(
    "--dim_red_method", 
    type=validate_dim_red, default=["avgtoken", "lasttoken"], nargs="+",
    help="The dimensionality reduction method(s) to apply to the features. Options: 'pca512', 'pca256', 'none', 'avgtoken', 'lasttoken'."
)
parser.add_argument(
    "--figdir",
    type=str, default="Figures_newrule",
    help="The directory to save the output figures."
)
# Training parameters
parser.add_argument(
    "--batch_size",
    type=int,
    default=1024,
    help="Batch size for data loading."
)

args = parser.parse_args()
expname = args.expname
train_step = args.train_step
layers = args.layers
dim_red_method = args.dim_red_method
figdir = args.figdir
batch_size = args.batch_size

# ### Load the training and test datasets
train_attrs = np.load("/n/home12/binxuwang/Github/DiffusionReasoning/attr_all.npy")
train_attrs = th.from_numpy(train_attrs).to(int)
attr_seq_tsr = einops.rearrange(train_attrs,  'cls (B R) p (h w) attr -> cls B (R p h w) attr', h=3,w=3,p=3,R=3)
attr_seq_tsr = preprocess_ids(attr_seq_tsr) # add 1 to the attribute indices
y_rule = th.arange(attr_seq_tsr.shape[0], dtype=th.long).unsqueeze(1).repeat(1, attr_seq_tsr.shape[1])
print(attr_seq_tsr.shape, y_rule.shape)
X_train, X_test = attr_seq_tsr[:, :3000], attr_seq_tsr[:, 3000:]
y_train, y_test = y_rule[:, :3000], y_rule[:, 3000:]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train = X_train.reshape(-1, 81, 3)
X_test = X_test.reshape(-1, 81, 3)
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
print(X_train.shape, y_train.shape)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# %% [markdown]
# ## Load example model
expdir = join(GPTroot, expname)
repr_expdir = join(expdir, 'repr_classifier')
figexpdir = join(figroot, expname)
os.makedirs(repr_expdir, exist_ok=True)
os.makedirs(figexpdir, exist_ok=True)
config = json.load(open(join(expdir, 'config.json')))
heldout_ids = config['heldout_id']
if train_step == -1:
    gpt2_raven = load_gpt2_raven_model(expdir, pth_name=None)
    ckpt_str = "ckptRNDINIT"
    print("Random initialization")
    pass 
else:
    gpt2_raven = load_gpt2_raven_model(expdir, pth_name=f'gpt2_step{train_step}.pth')
    ckpt_str = f"ckpt{train_step:07d}"
    print(f"Loaded {ckpt_str}: from {expdir}/ckpt/gpt2_step{train_step}.pth")
# %% [markdown]
# ### Representation recording
fetcher = featureFetcher_module()
for i in args.layers:
    if i >= len(gpt2_raven.gpt2.h):
        print(f"Layer {i} not found in the model. Skipping.")
        continue
    fetcher.record_module(gpt2_raven.gpt2.h[i], target_name=f"blocks.{i}", record_raw=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# extract features
t_beg = time.time()
feature_col = extract_features_GPT(gpt2_raven, fetcher, train_loader, cond=False) # 4mins
feature_col_test = extract_features_GPT(gpt2_raven, fetcher, test_loader, cond=False)
t_feat = time.time()
print(f"Feature extraction time: {t_feat - t_beg:.2f}")
# %% [markdown]
# ### Train representation classifier
for dimred_str in args.dim_red_method:
    # if noPCA optimizing in the full space requires a smaller learning rate
    learning_rate = 0.0005 if dimred_str == 'none' else 0.005
    model_PCA_col, PC_proj_col, results_col = train_dimred_sgd_classifiers(
            feature_col, feature_col_test, y_train, y_test, dimred_str=dimred_str,
            num_classes=40, batch_size=None,
            num_epochs=5000, print_every=500, eval_every=1000, learning_rate=learning_rate,
            device='cuda'
        )
    pkl.dump(results_col, open(join(repr_expdir,f"results_col_{dimred_str}_{ckpt_str}.pkl"), "wb"))
    pkl.dump(PC_proj_col, open(join(repr_expdir,f"PC_proj_col_{dimred_str}_{ckpt_str}.pkl"), "wb"))
    pkl.dump(model_PCA_col, open(join(repr_expdir,f"model_PCA_col_{dimred_str}_{ckpt_str}.pkl"), "wb"))

    test_acc_synopsis = {key: results_col[key].test_record.accuracy.max() for key in results_col.keys()}
    train_acc_synopsis = {key: results_col[key].train_record.accuracy.max() for key in results_col.keys()}
    syn_df = pd.DataFrame([test_acc_synopsis, train_acc_synopsis]).T
    syn_df.columns = ["Test Accuracy", "Train Accuracy"]
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=syn_df, markers=True)
    plt.title(f"Accuracy\n{expname}\n{ckpt_str} {dimred_str}")
    saveallforms([repr_expdir,],f"{dimred_str}_{ckpt_str}_accuracy.png")
    plt.show()
    for layer in results_col:
        pred_cls = results_col[layer]['pred_cls'].cpu()
        cm = confusion_matrix(y_test, pred_cls)
        fig1, fig2 = visualize_cm(cm, heldout_rules=heldout_ids, titlestr=f"{layer} Avg token repr step{train_step}\n{expname}")
        saveallforms(figexpdir, f"acc_per_rule_{layer}_{dimred_str}_{ckpt_str}", fig1)
        saveallforms(figexpdir, f"rule_confmat_{layer}_{dimred_str}_{ckpt_str}", fig2)
    plt.close('all')
del feature_col
del feature_col_test
del PC_proj_col


# feature_avgtoken_train = {k: v.mean(dim=1) for k, v in feature_col.items()}
# feature_avgtoken_test = {k: v.mean(dim=1) for k, v in feature_col_test.items()}
# feature_lasttoken_train = {k: v[:, -1] for k, v in feature_col.items()}
# feature_lasttoken_test = {k: v[:, -1] for k, v in feature_col_test.items()}
# print(feature_avgtoken_train['blocks.11'].shape)
# print(feature_avgtoken_test['blocks.11'].shape)
# _, _, results_col = train_pca_sgd_classifiers(feature_avgtoken_train, feature_avgtoken_test, y_train, y_test,
#                           batch_size=None, num_epochs=5000, noPCA=True, num_classes=40,
#                           learning_rate = 0.005, print_every=500, eval_every=500,)
# _, _, results_col_ltk = train_pca_sgd_classifiers(feature_lasttoken_train, feature_lasttoken_test, y_train, y_test,
#                           batch_size=None, num_epochs=5000, noPCA=True, num_classes=40,
#                           learning_rate=0.005, print_every=500, eval_every=500,)
# _, _, results_col_PCA = train_pca_sgd_classifiers(feature_col, feature_col_test, y_train, y_test,
#                           batch_size=None, num_epochs=5000, noPCA=False, PC_dim=768, num_classes=40,
#                           learning_rate=0.005, print_every=500, eval_every=500, )

# th.cuda.empty_cache()