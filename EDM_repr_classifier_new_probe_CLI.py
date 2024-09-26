
import argparse
import re
import os
import json
from os.path import join
import pickle as pkl
import time
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
import einops
import seaborn as sns
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
plt.rcParams['figure.dpi'] = 72
plt.rcParams['figure.figsize'] = [6.0, 4.0]
plt.rcParams['figure.edgecolor'] = (1, 1, 1, 0)
plt.rcParams['figure.facecolor'] = (1, 1, 1, 0)
# vector graphics type
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


import sys
sys.path.append("/n/home12/binxuwang/Github/mini_edm")
sys.path.append("/n/home12/binxuwang/Github/DiffusionReasoning")
sys.path.append("/n/home12/binxuwang/Github/DiT")
from rule_new_utils import check_r3_r2_batch, infer_rule_from_sample_batch, compute_rule_statistics
from dataset_utils import train_data2attr_tsr,load_raw_data,load_PGM_abstract
from stats_plot_utils import saveallforms
from repr_probe_lib import extract_features_DiT, train_dimred_sgd_classifiers, train_dimred_sgd_classifiers, train_dimred_sgd_classifiers, train_dimred_sgd_classifiers
from edm_utils import edm_sampler_inpaint, create_edm, get_default_config, create_edm_new
from train_edm import create_model, edm_sampler, EDM
# from diffusion import create_diffusion
# from models import DiT

import circuit_toolkit
from circuit_toolkit.layer_hook_utils import print_specific_layer, get_module_name_shapes, featureFetcher_module
# config_mapping = {
#     "WideBlnrX3_new_RAVEN10_abstract_20240315-1327": 
#         # --layers_per_block 2 --model_channels 128 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear
#         dict(layers_per_block=2, model_channels=128, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
#     "WideBlnrX3_new_RAVEN10_abstract_20240412-1347":
#         dict(layers_per_block=2, model_channels=128, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
#     "BigBlnrX3_new_RAVEN10_abstract_20240412-0143":
#         # --layers_per_block 3 --model_channels 192 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear
#         dict(layers_per_block=3, model_channels=192, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
#     "WideBlnrX3_new_noattn_RAVEN10_abstract_20240412-1254":
#         # --layers_per_block 2 --model_channels 128 --channel_mult 1 2 4 --attn_resolutions 0   --train_batch_size 256 --spatial_matching bilinear
#         dict(layers_per_block=2, model_channels=128, channel_mult=[1, 2, 4], attn_resolutions=[0], spatial_matching="bilinear"),
#     "BaseBlnrX3_new_RAVEN10_abstract_20240313-1736": 
#         # --layers_per_block 1 --model_channels 64  --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear
#         dict(layers_per_block=1, model_channels=64, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
# }
config_mapping = {
    "BaseBlnrX3" : dict(layers_per_block=1, model_channels=64, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
    "WideBlnrX3" : dict(layers_per_block=2, model_channels=128, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
    "BigBlnrX3"  : dict(layers_per_block=3, model_channels=192, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
}

heldout_id_dict = {
    'train_inputs_new.pt'       : [1, 16, 20, 34, 37], 
    'train_inputs_new_split0.pt': [1, 16, 20, 34, 37], 
    'train_inputs_new_split1.pt': [8, 12, 24, 36, 39],
    'train_inputs_new_split2.pt': [5, 17, 21, 33, 38],
    'train_inputs_new_split3.pt': [3, 10, 29, 31, 37],
    'train_inputs_new_split4.pt': [0, 14, 27, 35, 38],
    'train_inputs_new_split5.pt': [4, 19, 26, 30, 39],
    'train_inputs_new_split6.pt': [9, 13, 25, 32, 37],
    'train_inputs_new_split7.pt': [2, 18, 23, 30, 38],
    'train_inputs_new_split8.pt': [7, 15, 22, 34, 39],
    'train_inputs_new_split9.pt': [6, 11, 28, 33, 37],
}


def extract_features_EDM(model_EDM, fetcher, data_loader, 
                         dataset_Xmean, dataset_Xstd, t_scalar,
                         device='cuda', progress_bar=True):
    feature_col = defaultdict(list)
    loader = tqdm(data_loader) if progress_bar else data_loader
    for X_batch, y_batch in loader:
        t_vec = th.ones(X_batch.shape[0], dtype=torch.float, device=device) * t_scalar
        X_batch_norm = (X_batch.cuda().float() - dataset_Xmean) / dataset_Xstd
        with th.no_grad():
            model_EDM.model_forward_wrapper(X_batch_norm, t_vec, use_ema=True, labels=None)
        feature_col["input"].append(X_batch.float())
        for key, activations in fetcher.activations.items():
            feature_col[key].append(activations)
    for key in feature_col.keys():
        feature_col[key] = th.cat(feature_col[key], dim=0)
        print(key, feature_col[key].shape)
    return feature_col


def validate_dim_red(value):
    pattern = re.compile(r'^(pca\d+|none|avgtoken|lasttoken|avgspace)$')
    if not pattern.match(value):
        raise argparse.ArgumentTypeError(
            f"Invalid choice: '{value}'. Must match one of: pca, none, avgtoken, lasttoken."
        )
    return value


device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Load and evaluate a DiT model.")
parser.add_argument("--expname", type=str, default="045-RAVEN10_abstract-uncond-DiT_S_1_20240311-1256", 
                    help="The name of the experiment.")
parser.add_argument("--epoch", type=int, default=999999, 
                    help="The epoch of the model to load.")
parser.add_argument("--use_ema", action="store_true", 
                    help="Whether to use the EMA model.")
parser.add_argument("--t_scalars", type=float, default=[0.002, 0.01, 0.035, 0.1, 0.2, 0.3, 0.6, 0.8, 1.0, 1.3, 2.5, 4.6, 13.7, 40.0, 80.0], 
                    nargs="+", help="The time value to evaluate the score vectors.")
parser.add_argument("--layers", type=str, default=["input", 'enc.9x9_conv', 'enc.3x3_down', 'enc.1x1_down', 'dec.1x1_in0', 'dec.1x1_in1', 'dec.3x3_up', 'dec.9x9_up', "dec.9x9_aux_norm", "dec.9x9_aux_conv", ],
                    nargs="+", help="The layers to evaluate.")
parser.add_argument("--dim_red_method", type=validate_dim_red, 
                    default=["avgspace", "pca512"], #choices=["pca", "none", "avgtoken", "lasttoken", "avgspace"],
                    nargs="+", help="The dimensionality reduction method.")
# parser.add_argument("--noPCA", action="store_true", 
#                     help="Whether to use PCA for the feature vectors.")
# parser.add_argument("--PC_dim", type=int, default=1024, 
#                     help="The dimension of the PCA projection.")
# parser.add_argument("--figdir", type=str, default="Figures_newrule",
#                     help="The directory to save the figures.")

# 80.000, 53.559, 34.992, 22.240, 13.699, 8.139, 4.637, 2.515, 1.287, 0.613, 0.267, 0.1, 0.035, 0.010, 0.002

# model_EDM
exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"
args = parser.parse_args()
expname = args.expname
epoch = args.epoch
use_ema = args.use_ema
layers = args.layers
t_scalars = args.t_scalars
# PC_dim = args.PC_dim
# dim_red = args.dim_red_method
# figdir = args.figdir
# layers = ["input", 'enc.9x9_conv', 'enc.3x3_down', 'enc.1x1_down', 'dec.1x1_in0', 'dec.1x1_in1', 'dec.3x3_up', 'dec.9x9_up', "dec.9x9_aux_norm", "dec.9x9_aux_conv", ]

DATASET = "RAVEN10_abstract"
# prepare the dataset, training and testing
train_attrs = np.load("/n/home12/binxuwang/Github/DiffusionReasoning/attr_all.npy")
train_attrs = th.from_numpy(train_attrs).to(int)
if DATASET == "RAVEN10_abstract":
    train_row_img = einops.rearrange(train_attrs, 'c s pnl (H W) att -> c s att H (pnl W)', H=3, W=3, att=3, pnl=3)
    train_sample_img = einops.rearrange(train_row_img, 'c (S R) att H W -> c S att (R H) W', R=3,att=3, H=3, W=9)
    dataset_Xmean = th.tensor([1.5, 2.5, 2.5]).view(1, 3, 1, 1).to("cuda")
    dataset_Xstd = th.tensor([2.5, 3.5, 3.5]).view(1, 3, 1, 1).to("cuda")
elif DATASET == "RAVEN10_abstract_onehot":
    dataset_Xmean = th.tensor([0.5, ]).view(1, 1, 1, 1)
    dataset_Xstd = th.tensor([0.5, ]).view(1, 1, 1, 1)
    raise NotImplementedError("RAVEN10_abstract_onehot not implemented.")
    # use the one-hot encoding
labels_tsr = th.arange(40).to(int).view(-1,1).repeat(1, train_sample_img.shape[1]) # len(train_sample_img)

X_train = train_sample_img[:, :3000]
y_train = labels_tsr[:, :3000]
X_test = train_sample_img[:, 3000:]
y_test = labels_tsr[:, 3000:]
X_train = X_train.reshape(-1, 3, 9, 9)
y_train = y_train.reshape(-1)
X_test = X_test.reshape(-1, 3, 9, 9)
y_test = y_test.reshape(-1)
print(X_train.shape, y_train.shape)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"
expdir = join(exproot, expname)
ckptdir = join(expdir, "checkpoints")
repr_expdir = join(expdir, "repr_classifier")
os.makedirs(repr_expdir, exist_ok=True)
model_scale = expname.split("_")[0]
config_ft = get_default_config(DATASET, **config_mapping[model_scale])
use_ema = True
if epoch == -1:
    edm, model_EDM = create_edm_new(None, config_ft, device) 
    ckpt_str = "ckptRNDINIT"
    print("Random initialization")
else:
    ckpt_path = join(ckptdir, f"ema_{epoch}.pth")
    edm, model_EDM = create_edm_new(ckpt_path, config_ft, device) 
    ckpt_str = f"ckpt{epoch:07d}EMA"
    print(f"Loaded {ckpt_str}: from {ckpt_path}, use_ema: {use_ema}")
    
    
fetcher = featureFetcher_module()
record_module_list = layers
for blockname in list(edm.ema.enc):
    if f"enc.{blockname}" in record_module_list:
        fetcher.record_module(edm.ema.enc[blockname], target_name=f"enc.{blockname}")
for blockname in list(edm.ema.dec):
    if f"dec.{blockname}" in record_module_list:
        fetcher.record_module(edm.ema.dec[blockname], target_name=f"dec.{blockname}")

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
for t_scalar in args.t_scalars: 
    print(f"Processing t_scalar: {t_scalar}")
    t_beg = time.time()
    t_str = str(t_scalar).replace('.', '_')
    feature_col = extract_features_EDM(edm, fetcher, train_loader, dataset_Xmean, dataset_Xstd, t_scalar)
    feature_col_test = extract_features_EDM(edm, fetcher, test_loader, dataset_Xmean, dataset_Xstd, t_scalar)
    # feature_avg_col = {k: v.mean(dim=(2,3)) for k, v in feature_col.items()}
    # feature_avg_col_test = {k: v.mean(dim=(2,3)) for k, v in feature_col_test.items()}
    t_feat = time.time()
    print(f"Feature extraction time: {t_feat - t_beg:.2f}")
    for dimred_str in args.dim_red_method:
        # if noPCA optimizing in the full space requires a smaller learning rate
        learning_rate = 0.0005 if dimred_str == 'none' else 0.005
        model_PCA_col, PC_proj_col, results_col = train_dimred_sgd_classifiers(
                feature_col, feature_col_test, y_train, y_test, dimred_str=dimred_str,
                num_classes=40, batch_size=None,
                num_epochs=5000, print_every=250, eval_every=1000, learning_rate=learning_rate,
                device='cuda'
            )
        pkl.dump(results_col, open(join(repr_expdir,f"results_col_t{t_str}_{dimred_str}_{ckpt_str}.pkl"), "wb"))
        pkl.dump(PC_proj_col, open(join(repr_expdir,f"PC_proj_col_t{t_str}_{dimred_str}_{ckpt_str}.pkl"), "wb"))
        pkl.dump(model_PCA_col, open(join(repr_expdir,f"model_PCA_col_t{t_str}_{dimred_str}_{ckpt_str}.pkl"), "wb"))

        test_acc_synopsis = {key: results_col[key].test_record.accuracy.max() for key in results_col.keys()}
        train_acc_synopsis = {key: results_col[key].train_record.accuracy.max() for key in results_col.keys()}
        syn_df = pd.DataFrame([test_acc_synopsis, train_acc_synopsis]).T
        syn_df.columns = ["Test Accuracy", "Train Accuracy"]
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=syn_df, markers=True)
        plt.title(f"t_scalar: {t_scalar} Accuracy\n{expname}\n{ckpt_str} {dimred_str}")
        saveallforms([repr_expdir,],f"t{t_str}_{dimred_str}_{ckpt_str}_accuracy.png")
        plt.show()
    del feature_col
    del feature_col_test
    del PC_proj_col

del fetcher






