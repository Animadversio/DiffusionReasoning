# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
sys.path.append('/n/home12/binxuwang/Github/DiffusionReasoning')

# %%
import os
import re
import json
import pickle as pkl
from os.path import join
from tqdm import trange, tqdm
from os.path import join
from easydict import EasyDict as edict
import numpy as np
import torch as th
import einops
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from os.path import join
from edm_utils import parse_train_logfile
from dataset_utils import onehot2attr_tsr
from stats_plot_utils import estimate_CI, shaded_error, saveallforms
from stats_plot_utils import shaded_error, add_rectangles
from rule_utils import get_rule_list, get_obj_list, get_rule_img, check_consistent
from rule_new_utils import get_rule_annot
from GPT_models.GPT_RAVEN_model_lib import MultiIdxGPT2Model, completion_eval, seqtsr2imgtsr, preprocess_ids

# %%
GPTroot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/GPT2_raven"

# %%
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
    gpt2_raven.load_state_dict(th.load(join(expdir, 'ckpt', pth_name)))
    # count the number of parameters
    num_params = sum(p.numel() for p in gpt2_raven.parameters())
    print(f"Number of parameters: {num_params}")
    return gpt2_raven.to('cuda').eval()

# %% [markdown]
# ## Load example model

# %%
expname = "GPT2_medium_RAVEN_uncond_heldout0_stream0_16M-20240820-024019"
expdir = join(GPTroot, expname)

# %%
!ls {GPTroot}/GPT2_medium_RAVEN_uncond_heldout0_stream0_16M-20240820-024019/ckpt

# %%
config = json.load(open(join(expdir, 'config.json')))
heldout_ids = config['heldout_id']

# %%
train_step = 999999

# %%
gpt2_raven = load_gpt2_raven_model(expdir, pth_name=f'gpt2_step{train_step}.pth')

# %% [markdown]
# ### Load the training and test datasets

# %%
from torch.utils.data import DataLoader, TensorDataset

# %%
train_attrs = np.load("/n/home12/binxuwang/Github/DiffusionReasoning/attr_all.npy")
train_attrs = th.from_numpy(train_attrs).to(int)

# %%
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
# ### Representation recording

# %%
import circuit_toolkit
print(circuit_toolkit.__file__)
from circuit_toolkit.layer_hook_utils import print_specific_layer, get_module_name_shapes, featureFetcher_module
from repr_probe_lib import train_pca_sgd_classifiers, fit_SGD_linear_classifier, train_dimred_sgd_classifiers, extract_features_GPT

# %%
gpt2_raven.gpt2.h[0]

from collections import defaultdict
# extract_features_GPT
def extract_features_GPT(
    model,
    fetcher,
    data_loader,
    device='cuda',
    cond = False,
    progress_bar=True
):
    """
    Extracts features from specified layers of the model for the given dataset.

    Args:
        model (torch.nn.Module): The neural network model.
        fetcher (FeatureFetcher): An instance of the featureFetcher_module.
        data_loader (DataLoader): DataLoader for the dataset.
        device (str, optional): Device to perform computations on. Defaults to 'cuda'.
        progress_bar (bool, optional): Whether to display a progress bar. Defaults to True.

    Returns:
        dict: A dictionary with layer keys and concatenated activation tensors.
    """
    feature_col = defaultdict(list)
    loader = tqdm(data_loader) if progress_bar else data_loader
    for X_batch, y_batch in loader:
        # Prepare model inputs
        if cond:
            model_kwargs = {'y': y_batch.to(device)}
        else:
            model_kwargs = {'y': th.zeros(X_batch.size(0), dtype=th.long, device=device)}
        # Forward pass with no gradient computation
        with th.no_grad():
            model.forward(X_batch.to(device), **model_kwargs)
        # Collect activations
        for key, activations in fetcher.activations.items():
            # for GPT2Block, the output is a tuple of length 2, the first is the hidden state, the second is sth. else like attentions.
            if isinstance(activations, list) or isinstance(activations, tuple):
                activations = activations[0].detach().cpu()
            feature_col[key].append(activations)

    # Concatenate all activations for each layer
    for key in feature_col:
        feature_col[key] = th.cat(feature_col[key], dim=0)
        print(f"{key}: {feature_col[key].shape}")
    return feature_col

# %%
# layers = [0, 5, 11, 17, 23]
layers = [11, 23]
fetcher = featureFetcher_module()
for i in layers:
    fetcher.record_module(gpt2_raven.gpt2.h[i], target_name=f"blocks.{i}", record_raw=True)

# %%
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
feature_col = extract_features_GPT(gpt2_raven, fetcher, train_loader, cond=False) # 4mins
feature_col_test = extract_features_GPT(gpt2_raven, fetcher, test_loader, cond=False)

# %% [markdown]
# ### Train representation classifier


# %%
feature_avgtoken_train = {k: v.mean(dim=1) for k, v in feature_col.items()}
feature_avgtoken_test = {k: v.mean(dim=1) for k, v in feature_col_test.items()}
feature_lasttoken_train = {k: v[:, -1] for k, v in feature_col.items()}
feature_lasttoken_test = {k: v[:, -1] for k, v in feature_col_test.items()}
print(feature_avgtoken_train['blocks.11'].shape)
print(feature_avgtoken_test['blocks.11'].shape)

_, _, results_col = train_pca_sgd_classifiers(feature_avgtoken_train, feature_avgtoken_test, y_train, y_test,
                          batch_size=None, num_epochs=5000, noPCA=True, num_classes=40,
                          learning_rate = 0.005, print_every=500, eval_every=500,)

# %%
_, _, results_col_ltk = train_pca_sgd_classifiers(feature_lasttoken_train, feature_lasttoken_test, y_train, y_test,
                          batch_size=None, num_epochs=5000, noPCA=True, num_classes=40,
                          learning_rate=0.005, print_every=500, eval_every=500,)

# %%
_, _, results_col_PCA = train_pca_sgd_classifiers(feature_col, feature_col_test, y_train, y_test,
                          batch_size=None, num_epochs=5000, noPCA=False, PC_dim=768, num_classes=40,
                          learning_rate=0.005, print_every=500, eval_every=500, )

# %%
th.cuda.empty_cache()

# %% [markdown]
# ### Error dissection

# %%
from stats_plot_utils import visualize_cm
from sklearn.metrics import confusion_matrix

# %%
for layername in results_col:
    pred_cls = results_col[layername]['pred_cls'].cpu()
    cm = confusion_matrix(y_test, pred_cls)
    visualize_cm(cm, heldout_rules=heldout_ids, titlestr=f"{layername} Avg token repr step{train_step}\n{expname}")